"""
Canva Connect API client — OAuth 2.0 Authorization Code + PKCE, asset
upload, and design creation.

IMPORTANT — VERIFY BEFORE RELYING ON THIS IN PRODUCTION:
This sandbox has no live network access to api.canva.com, so none of the
URLs, request/response shapes, or field names below have been verified
against a real call. They are written to match Canva's public Connect
API documentation (https://www.canva.dev/docs/connect/) as accurately as
possible, but Canva's API has changed shape before and may again. Before
you rely on this beyond local testing:
  - Confirm the OAuth authorize/token URLs against your Canva integration's
    own settings page (Canva sometimes shows the exact URLs to use there).
  - Confirm the asset-upload job response shape (`job.status`,
    `job.asset.id`) and the create-design request body
    (`design_type`/`asset_id`) against the current API reference — these
    are the two endpoints most likely to have evolved.
  - Confirm which scopes your specific integration type actually needs;
    the Developer Portal will tell you if a request is missing one.
Run TEST 1-5 (see the final report) and read the actual error bodies
Canva returns if anything here doesn't match — they're normally specific
enough to pinpoint exactly which field/URL needs correcting.
"""

import base64
import hashlib
import json
import logging
import secrets
import time
from typing import Optional

import requests

from app.config import settings
from app.services import canva_token_store

logger = logging.getLogger(__name__)

# Per Canva's published OAuth docs. The authorize endpoint is under the
# regular canva.com domain (it's a browser-facing consent screen); the
# API endpoints (token exchange, asset upload, design creation) are
# under api.canva.com.
CANVA_AUTHORIZE_URL = "https://www.canva.com/api/oauth/authorize"
CANVA_TOKEN_URL = "https://api.canva.com/rest/v1/oauth/token"
CANVA_ASSET_UPLOAD_URL = "https://api.canva.com/rest/v1/asset-uploads"
CANVA_DESIGNS_URL = "https://api.canva.com/rest/v1/designs"

_ASSET_UPLOAD_POLL_INTERVAL_SECONDS = 1.5
_ASSET_UPLOAD_POLL_TIMEOUT_SECONDS = 60


class CanvaAPIError(RuntimeError):
    """Raised for any Canva API failure — caught in app/routes/canva.py
    and turned into a clear HTTP error for the frontend (section 9)."""


# --- PKCE helpers -----------------------------------------------------------

def generate_pkce_pair() -> tuple[str, str]:
    """Returns (code_verifier, code_challenge) per RFC 7636 (S256)."""
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(40)).rstrip(b"=").decode("ascii")
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge


def build_authorization_url(state: str, code_challenge: str) -> str:
    if not settings.CANVA_CLIENT_ID:
        raise CanvaAPIError("CANVA_CLIENT_ID is not configured.")
    params = {
        "response_type": "code",
        "client_id": settings.CANVA_CLIENT_ID,
        "redirect_uri": settings.CANVA_REDIRECT_URI,
        "scope": settings.CANVA_SCOPES,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    query = "&".join(f"{k}={requests.utils.quote(str(v), safe='')}" for k, v in params.items())
    return f"{CANVA_AUTHORIZE_URL}?{query}"


# --- Token exchange / refresh -----------------------------------------------

def exchange_code_for_tokens(code: str, code_verifier: str) -> dict:
    """Server-side only — CANVA_CLIENT_SECRET is used here and only
    here (plus refresh_tokens below); it never appears in any response
    this service returns to the frontend."""
    if not settings.CANVA_CLIENT_ID or not settings.CANVA_CLIENT_SECRET:
        raise CanvaAPIError("CANVA_CLIENT_ID/CANVA_CLIENT_SECRET are not configured.")
    resp = requests.post(
        CANVA_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": code_verifier,
            "redirect_uri": settings.CANVA_REDIRECT_URI,
            "client_id": settings.CANVA_CLIENT_ID,
            "client_secret": settings.CANVA_CLIENT_SECRET,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    if not resp.ok:
        logger.error("Canva token exchange failed: %s %s", resp.status_code, resp.text)
        raise CanvaAPIError(f"Canva token exchange failed ({resp.status_code}): {resp.text}")
    return resp.json()


def refresh_tokens(refresh_token: str) -> dict:
    resp = requests.post(
        CANVA_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": settings.CANVA_CLIENT_ID,
            "client_secret": settings.CANVA_CLIENT_SECRET,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    if not resp.ok:
        logger.error("Canva token refresh failed: %s %s", resp.status_code, resp.text)
        raise CanvaAPIError(f"Canva token refresh failed ({resp.status_code}): {resp.text}")
    return resp.json()


def get_valid_access_token(session_id: str) -> str:
    """Returns a currently-valid access token for this session,
    transparently refreshing (and persisting the refreshed tokens) if
    the stored one is at/near expiry. Raises CanvaAPIError if this
    session has never connected Canva, or if refresh itself fails
    (section 9: invalid/expired token) — callers should turn that into
    a 401/"reconnect Canva" response, not silently retry."""
    tokens = canva_token_store.get_tokens(session_id)
    if tokens is None:
        raise CanvaAPIError("Canva is not connected for this session.")

    # 60s safety margin so a token doesn't expire mid-request.
    if time.time() < tokens["expires_at"] - 60:
        return tokens["access_token"]

    logger.info("Canva access token near/at expiry for session, refreshing.")
    refreshed = refresh_tokens(tokens["refresh_token"])
    canva_token_store.save_tokens(
        session_id,
        access_token=refreshed["access_token"],
        # Canva may or may not rotate the refresh_token on every refresh
        # response — fall back to the existing one if it's omitted, per
        # standard OAuth2 refresh-token behavior.
        refresh_token=refreshed.get("refresh_token", tokens["refresh_token"]),
        expires_in=refreshed["expires_in"],
        scope=refreshed.get("scope", tokens["scope"]),
    )
    return refreshed["access_token"]


# --- Asset upload ------------------------------------------------------------

def upload_asset(access_token: str, image_bytes: bytes, file_name: str) -> str:
    """Uploads one image to Canva's async asset-upload endpoint, polls
    until it finishes, and returns the resulting Canva asset ID.

    Canva's asset upload is a two-step async job (not a plain synchronous
    upload) per its documented API: POST the bytes with a metadata
    header describing the asset, get back a job id + "in_progress"
    status, then poll the job until it's "success" (asset ready,
    contains the asset id) or "failed" (section 9: asset upload failure
    / asset upload still processing after the timeout below)."""
    # Canva's Asset-Upload-Metadata header must be the RAW JSON string
    # itself — only the filename inside it is base64-encoded (per
    # Canva's docs: {"name_base64": "<base64 name>"}). An earlier
    # version of this function incorrectly base64-encoded the whole
    # JSON blob on top of that, producing a value Canva's API rejects
    # outright with 400 "Invalid upload metadata header" — this is
    # exactly that bug, fixed.
    name_b64 = base64.b64encode(file_name.encode("utf-8")).decode("ascii")
    metadata_header = json.dumps({"name_base64": name_b64})

    resp = requests.post(
        CANVA_ASSET_UPLOAD_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/octet-stream",
            "Asset-Upload-Metadata": metadata_header,
        },
        data=image_bytes,
        timeout=30,
    )
    if not resp.ok:
        logger.error("Canva asset upload failed to start: %s %s", resp.status_code, resp.text)
        raise CanvaAPIError(f"Canva asset upload failed ({resp.status_code}): {resp.text}")

    job = resp.json().get("job", {})
    job_id = job.get("id")
    if not job_id:
        raise CanvaAPIError(f"Canva asset upload response missing job id: {resp.text}")

    deadline = time.monotonic() + _ASSET_UPLOAD_POLL_TIMEOUT_SECONDS
    while True:
        status_resp = requests.get(
            f"{CANVA_ASSET_UPLOAD_URL}/{job_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=15,
        )
        if not status_resp.ok:
            raise CanvaAPIError(f"Canva asset upload status check failed ({status_resp.status_code}): {status_resp.text}")
        status_job = status_resp.json().get("job", {})
        status = status_job.get("status")

        if status == "success":
            asset_id = (status_job.get("asset") or {}).get("id")
            if not asset_id:
                raise CanvaAPIError(f"Canva asset upload succeeded but response has no asset id: {status_resp.text}")
            return asset_id
        if status == "failed":
            error = status_job.get("error", {})
            raise CanvaAPIError(f"Canva asset upload failed: {error}")

        # status == "in_progress" (or unrecognized) — keep polling.
        if time.monotonic() > deadline:
            raise CanvaAPIError(
                f"Canva asset upload for '{file_name}' is still processing after "
                f"{_ASSET_UPLOAD_POLL_TIMEOUT_SECONDS}s — try again shortly."
            )
        time.sleep(_ASSET_UPLOAD_POLL_INTERVAL_SECONDS)


# --- Design creation ---------------------------------------------------------

def create_design(
    access_token: str,
    asset_id: Optional[str] = None,
    width: int = 1080,
    height: int = 1080,
    correlation_id: Optional[str] = None,
) -> dict:
    """Creates a new Canva design and returns {"design_id", "edit_url"}.

    FIXED (was a real bug, confirmed by Canva's own error response, not
    speculation): originally used design_type {"type": "preset", "name":
    "doc"} — Canva's "doc" preset is a multi-page DOCUMENT type (like
    Canva Docs), and its export API explicitly rejects PNG export for it
    ("png export not supported for this design type"). Since this
    integration's whole point is "place one image, get a flat image
    back," a document type was the wrong choice from the start. Switched
    to {"type": "custom", "width", "height"} — a plain graphic canvas,
    which is what PNG export is actually meant for. Defaults to a
    1080x1080 square; pass the real image's pixel dimensions via
    width/height when known (see app/routes/canva.py) so the canvas
    isn't a mismatched aspect ratio from the source image.

    KNOWN LIMITATION (report this to the user, don't silently paper over
    it): Canva's create-design API accepts at most ONE asset_id to
    pre-place on the new design's first page. If the caller selected
    multiple images (section 4), only the first is placed automatically
    — the rest are still uploaded to the user's Canva account (via
    upload_asset above) and will appear in their Canva "Uploads" panel
    for manual drag-in, but app/routes/canva.py must surface this as an
    explicit warning, not claim all images were placed on the canvas.

    UNVERIFIED — RETURN NAVIGATION CORRELATION: see
    parse_rti_token_from_query()'s docstring below for what WAS
    confirmed empirically about how correlation_state/design_id come
    back on return (a signed JWT, not a bare query param, as originally
    assumed here) — that part of this docstring is now verified, not
    speculative. Still sending correlation_id both in this request body
    and appended to edit_url as a hedge; only the JWT round-trip is
    actually confirmed to matter.
    """
    body: dict = {"design_type": {"type": "custom", "width": width, "height": height}}
    if asset_id:
        body["asset_id"] = asset_id
    if correlation_id:
        body["correlation_state"] = correlation_id

    resp = requests.post(
        CANVA_DESIGNS_URL,
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        json=body,
        timeout=20,
    )
    if not resp.ok:
        logger.error("Canva design creation failed: %s %s", resp.status_code, resp.text)
        raise CanvaAPIError(f"Canva design creation failed ({resp.status_code}): {resp.text}")

    data = resp.json().get("design", {})
    design_id = data.get("id")
    edit_url = (data.get("urls") or {}).get("edit_url")
    if not design_id or not edit_url:
        raise CanvaAPIError(f"Canva design creation response missing id/edit_url: {resp.text}")

    if correlation_id:
        separator = "&" if "?" in edit_url else "?"
        edit_url = f"{edit_url}{separator}correlation_state={correlation_id}"

    return {"design_id": design_id, "edit_url": edit_url}


# --- Design export -----------------------------------------------------------
# UNVERIFIED against live Canva docs (see module docstring at top of this
# file) — written to match Canva's documented async export-job pattern
# (same shape as upload_asset()'s job-create-then-poll flow above), but
# the exact endpoint path, request body, and response field names below
# should be confirmed against https://www.canva.dev/docs/connect/ if
# anything here 400s/404s. Run the real flow and paste back the actual
# error body — same iterative process that fixed upload_asset()'s
# metadata header earlier in this project.

# --- Return-to-integration JWT (the "rti" token) ----------------------------
# Discovered empirically (not from live docs — see module docstring) by
# logging the real query string Canva sends on return: it's a JWT
# ("type":"rti" in its payload) containing, among other claims, exactly
# the `correlation_state` this integration sent when opening the design,
# AND the `design_id` directly. This is delivered as SOME query
# parameter on the Return URL redirect — parse_rti_token_from_query()
# below finds it by shape (three dot-separated base64url segments)
# rather than assuming a specific parameter name, since that name was
# not confirmed from documentation.
#
# SECURITY NOTE: this decodes the JWT payload WITHOUT verifying its
# signature — acceptable for now because the design registry lookup
# (app/services/canva_design_registry.py) independently requires
# correlation_state to match a record this server itself created with a
# cryptographically random value never exposed except inside the
# edit_url handed to the legitimate session — so a forged token would
# still need to guess that value. Before production, verify this JWT's
# signature against Canva's published JWKS (check
# https://www.canva.dev/docs/connect/ for the endpoint) so a compromised
# or malicious client can't inject an arbitrary design_id/correlation_state.

def _decode_jwt_payload_unverified(token: str) -> Optional[dict]:
    import base64 as _b64
    import json as _json
    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload_b64 = parts[1]
    padded = payload_b64 + "=" * (-len(payload_b64) % 4)
    try:
        return _json.loads(_b64.urlsafe_b64decode(padded))
    except Exception:  # noqa: BLE001 - not a JWT, or malformed — caller treats as "not found"
        return None


def parse_rti_token_from_query(query_params: dict) -> Optional[dict]:
    """Scans all query params for one that decodes as a `type: "rti"`
    JWT payload, regardless of what Canva actually names that
    parameter. Returns the decoded payload dict (with `correlation_state`
    and `design_id`) or None if no such parameter is present."""
    for value in query_params.values():
        payload = _decode_jwt_payload_unverified(value)
        if payload and payload.get("type") == "rti":
            return payload
    return None


CANVA_EXPORTS_URL = "https://api.canva.com/rest/v1/exports"
_EXPORT_POLL_INTERVAL_SECONDS = 1.5
_EXPORT_POLL_TIMEOUT_SECONDS = 60


def export_design(access_token: str, design_id: str, export_format: str = "png") -> bytes:
    """Creates an export job for `design_id`, polls until it completes,
    downloads the resulting file, and returns its raw bytes. Raises
    CanvaAPIError on any failure (design not found, export failed, export
    still processing after the timeout, download failure — section 10)."""
    resp = requests.post(
        CANVA_EXPORTS_URL,
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        json={"design_id": design_id, "format": {"type": export_format}},
        timeout=20,
    )
    if resp.status_code == 404:
        raise CanvaAPIError(f"Canva design {design_id} not found (it may have been deleted, or doesn't belong to this account).")
    if not resp.ok:
        logger.error("Canva export job creation failed: %s %s", resp.status_code, resp.text)
        raise CanvaAPIError(f"Canva export job creation failed ({resp.status_code}): {resp.text}")

    job = resp.json().get("job", {})
    job_id = job.get("id")
    if not job_id:
        raise CanvaAPIError(f"Canva export response missing job id: {resp.text}")

    deadline = time.monotonic() + _EXPORT_POLL_TIMEOUT_SECONDS
    while True:
        status_resp = requests.get(
            f"{CANVA_EXPORTS_URL}/{job_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=15,
        )
        if not status_resp.ok:
            raise CanvaAPIError(f"Canva export status check failed ({status_resp.status_code}): {status_resp.text}")
        status_job = status_resp.json().get("job", {})
        status = status_job.get("status")

        if status == "success":
            urls = status_job.get("urls") or []
            if not urls:
                raise CanvaAPIError(f"Canva export succeeded but response has no download urls: {status_resp.text}")
            download_resp = requests.get(urls[0], timeout=30)
            if not download_resp.ok:
                raise CanvaAPIError(f"Failed to download exported file from Canva ({download_resp.status_code}).")
            return download_resp.content
        if status == "failed":
            error = status_job.get("error", {})
            raise CanvaAPIError(f"Canva export failed: {error}")

        if time.monotonic() > deadline:
            raise CanvaAPIError(
                f"Canva export for design {design_id} is still processing after "
                f"{_EXPORT_POLL_TIMEOUT_SECONDS}s — try again shortly."
            )
        time.sleep(_EXPORT_POLL_INTERVAL_SECONDS)