"""
Canva integration routes — OAuth connect/callback/status/disconnect, plus
"Create New in Canva" and "Edit in Canva" (uploading already-generated
S3 images as Canva assets). See app/services/canva_service.py's module
docstring for what's verified-against-docs vs. best-effort here.

Mounted under /api/canva (see app/main.py's include_router call) —
follows the exact same router pattern as app/routes/content.py.
"""

import logging
import secrets
import uuid

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from app.services import (
    canva_design_registry,
    canva_service,
    canva_token_store,
    image_ownership_service,
    image_storage_service,
)
from app.services.canva_service import CanvaAPIError
from app.services.session_service import get_or_create_session_id

logger = logging.getLogger(__name__)
router = APIRouter()


# --- 1. OAuth --------------------------------------------------------------

@router.get("/connect")
def connect(request: Request, response: Response):
    """Starts the Canva OAuth Authorization Code + PKCE flow. Returns a
    redirect straight to Canva's consent screen — the frontend's
    "Connect Canva" button should navigate the browser here directly
    (e.g. window.location.href = "/api/canva/connect"), not fetch() it,
    since the user must land on Canva's real page to log in/consent."""
    session_id = get_or_create_session_id(request, response)

    code_verifier, code_challenge = canva_service.generate_pkce_pair()
    state = secrets.token_urlsafe(24)
    canva_token_store.save_oauth_state(state, code_verifier, session_id)

    try:
        auth_url = canva_service.build_authorization_url(state, code_challenge)
    except CanvaAPIError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    redirect = RedirectResponse(url=auth_url, status_code=302)
    # Copy the session cookie onto the actual redirect response — the
    # `response` object FastAPI injected above is a separate object from
    # what's actually returned when a route returns a RedirectResponse,
    # so a Set-Cookie applied only to it would be silently dropped.
    for key, value in response.headers.items():
        if key.lower() == "set-cookie":
            redirect.headers.append(key, value)
    return redirect


@router.get("/callback")
def callback(request: Request, response: Response, code: str = None, state: str = None, error: str = None):
    """Canva redirects the user's browser here after they approve/deny
    access. This is a top-level browser navigation (not an API call the
    frontend fetches), so on both success and failure we redirect back
    to the app's own page with a query flag the frontend JS checks on
    load — see frontend/index.html's handling of ?canva=connected /
    ?canva=error."""
    if error:
        # section 9: Canva authorization cancelled
        logger.info("Canva OAuth cancelled/denied by user: %s", error)
        return RedirectResponse(url=f"/?canva=error&reason={error}", status_code=302)

    if not code or not state:
        return RedirectResponse(url="/?canva=error&reason=missing_code_or_state", status_code=302)

    state_entry = canva_token_store.pop_oauth_state(state)
    if state_entry is None:
        # section 9: OAuth callback failure — expired/replayed/forged state
        logger.warning("Canva OAuth callback with unknown/expired state.")
        return RedirectResponse(url="/?canva=error&reason=invalid_state", status_code=302)

    session_id = state_entry["session_id"]
    try:
        tokens = canva_service.exchange_code_for_tokens(code, state_entry["code_verifier"])
    except CanvaAPIError as exc:
        logger.error("Canva token exchange failed during callback: %s", exc)
        return RedirectResponse(url="/?canva=error&reason=token_exchange_failed", status_code=302)

    canva_token_store.save_tokens(
        session_id,
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        expires_in=tokens["expires_in"],
        scope=tokens.get("scope", ""),
    )

    redirect = RedirectResponse(url="/?canva=connected", status_code=302)
    # The session cookie was almost certainly already set during
    # /connect (the user had to have a session to start this flow) —
    # this re-sets it defensively in case the browser is completing the
    # callback in a context where the original cookie didn't persist
    # (e.g. some in-app browsers), so the connection isn't silently
    # orphaned under a session_id the frontend never sees.
    for key, value in response.headers.items():
        if key.lower() == "set-cookie":
            redirect.headers.append(key, value)
    return redirect


@router.get("/status")
def status(request: Request, response: Response) -> dict:
    session_id = get_or_create_session_id(request, response)
    return {"connected": canva_token_store.is_connected(session_id)}


@router.post("/disconnect")
def disconnect(request: Request, response: Response) -> dict:
    session_id = get_or_create_session_id(request, response)
    canva_token_store.delete_tokens(session_id)
    return {"connected": False}


# --- 3. Create New in Canva -------------------------------------------------

@router.post("/designs")
def create_new_design(request: Request, response: Response) -> dict:
    session_id = get_or_create_session_id(request, response)
    correlation_id = uuid.uuid4().hex
    try:
        access_token = canva_service.get_valid_access_token(session_id)
        result = canva_service.create_design(access_token, correlation_id=correlation_id)
    except CanvaAPIError as exc:
        # section 9: Canva API failure / invalid-expired token — 401 if
        # not connected at all, so the frontend knows to prompt
        # "Connect Canva" rather than just showing a generic error.
        status_code = 401 if "not connected" in str(exc).lower() else 502
        raise HTTPException(status_code=status_code, detail=str(exc))

    # section 2: record the chain (session -> design_id) even for a
    # design with no source images, so "return to integration" still
    # works for designs started from scratch, not just ones started
    # from a generated image.
    canva_design_registry.record_design(correlation_id, session_id, result["design_id"])
    return result


# --- 4/5/6. Edit generated images in Canva ----------------------------------

class EditInCanvaRequest(BaseModel):
    generation_id: str
    indices: list[int]


@router.post("/designs/from-images")
def create_design_from_images(request: Request, response: Response, body: EditInCanvaRequest) -> dict:
    session_id = get_or_create_session_id(request, response)

    if not body.indices:
        raise HTTPException(status_code=400, detail="No images selected.")

    # section 9: user tries to access another user's image / user
    # selects an invalid image — verified BEFORE any S3 read or Canva
    # call, per explicit requirement.
    for index in body.indices:
        if not image_ownership_service.verify_ownership(session_id, body.generation_id, index):
            raise HTTPException(
                status_code=403,
                detail=f"Image index {index} of generation {body.generation_id} was not "
                "found for this session, or does not belong to it.",
            )

    try:
        access_token = canva_service.get_valid_access_token(session_id)
    except CanvaAPIError as exc:
        raise HTTPException(status_code=401, detail=str(exc))

    asset_ids = []
    for index in body.indices:
        try:
            image_bytes = image_storage_service.get_image_bytes(body.generation_id, index)
        except RuntimeError as exc:
            # section 9: S3 image unavailable
            raise HTTPException(status_code=404, detail=f"Image {index} unavailable in S3: {exc}")
        try:
            asset_id = canva_service.upload_asset(access_token, image_bytes, f"{body.generation_id}-{index}.png")
        except CanvaAPIError as exc:
            # section 9: asset upload failure / still processing
            raise HTTPException(status_code=502, detail=f"Canva asset upload failed for image {index}: {exc}")
        asset_ids.append(asset_id)

    correlation_id = uuid.uuid4().hex
    try:
        result = canva_service.create_design(access_token, asset_id=asset_ids[0], correlation_id=correlation_id)
    except CanvaAPIError as exc:
        # section 9: Canva design creation failure
        raise HTTPException(status_code=502, detail=f"Canva design creation failed: {exc}")

    # section 2: this is what lets /return and /exports later map
    # "the user clicked Return to integration" back to exactly this
    # design AND this session, and verify ownership before exporting
    # anything (section 9).
    canva_design_registry.record_design(
        correlation_id, session_id, result["design_id"],
        source_generation_id=body.generation_id, source_indices=body.indices,
    )

    warnings = []
    if len(asset_ids) > 1:
        # See create_design()'s docstring — Canva's API only auto-places
        # one asset. Report this honestly rather than implying all
        # selected images landed on the canvas.
        warnings.append(
            f"{len(asset_ids)} images were uploaded to your Canva account, but only the "
            "first was placed on the design automatically — the rest are available in "
            "Canva's Uploads panel to drag in manually (a current limitation of Canva's "
            "create-design API, not this integration)."
        )

    return {**result, "asset_ids": asset_ids, "warnings": warnings}


# --- Return-to-integration + export (NEW: brings the edited result back) ---
#
# Do NOT confuse this with /callback above — that's the OAuth authorization
# redirect (has `code`/`state`, happens once per Canva connection).
# THIS is the separate "user finished editing and clicked Return to
# integration" redirect, configured as its own Return URL under a
# DIFFERENT section of the Canva Developer Portal ("Return navigation",
# not "Authentication") — see the final report for exactly what to set
# it to. Reusing /callback for both would make it impossible to tell an
# OAuth login apart from a finished edit.

@router.get("/return")
def return_from_canva(request: Request, response: Response):
    """Canva redirects the user's browser here (a real Return URL
    configured in the Developer Portal, NOT /callback) after they click
    "Return to integration" in the editor.

    UNVERIFIED (see canva_service.create_design()'s docstring for the
    full explanation): this assumes Canva echoes back a
    `correlation_state` query parameter matching what we sent when the
    design was opened. The log line below prints the COMPLETE real query
    string Canva sends — read it after testing once, and if there's no
    `correlation_state` param (or it's named something else), that's the
    thing to fix, not this route's logic.
    """
    session_id = get_or_create_session_id(request, response)
    query_params = dict(request.query_params)
    logger.info("Canva return-navigation redirect received. Full query params: %s", query_params)

    # See canva_service.parse_rti_token_from_query()'s docstring — Canva
    # sends a signed JWT (found empirically, not a bare `correlation_state`
    # param as originally assumed) containing both correlation_state and
    # design_id. This scans for it by shape rather than a specific param
    # name, since that name was never confirmed against documentation.
    rti_payload = canva_service.parse_rti_token_from_query(query_params)
    correlation_id = rti_payload.get("correlation_state") if rti_payload else None

    if not correlation_id:
        # section 10: user returns without making changes / Canva
        # didn't send back what we expected — send them home with a
        # flag the frontend can show a clear (not silently-broken)
        # message for, rather than a raw 400.
        return RedirectResponse(url="/?canva_edit_error=missing_correlation_state", status_code=302)

    record = canva_design_registry.verify_design_ownership(session_id, correlation_id)
    if record is None:
        # section 9/10: invalid design ID, or belongs to a different
        # session — never reveal which, just refuse.
        logger.warning("Canva return-navigation with unknown/foreign correlation_id.")
        return RedirectResponse(url="/?canva_edit_error=unknown_design", status_code=302)

    # Canva's token hands us design_id directly too — cross-check against
    # what we stored at creation time rather than blindly trusting either
    # source alone.
    token_design_id = rti_payload.get("design_id")
    if token_design_id and token_design_id != record["design_id"]:
        logger.warning(
            "design_id mismatch: token says %s, registry says %s for correlation_id=%s",
            token_design_id, record["design_id"], correlation_id,
        )

    redirect = RedirectResponse(url=f"/?canva_edit={correlation_id}", status_code=302)
    for key, value in response.headers.items():
        if key.lower() == "set-cookie":
            redirect.headers.append(key, value)
    return redirect


@router.post("/exports/{correlation_id}")
def export_edited_design(correlation_id: str, request: Request, response: Response) -> dict:
    """Called by the frontend after landing back on ?canva_edit=<id> (see
    return_from_canva above) — this is the step that actually does
    section 4-6's work: export the edited design from Canva, download
    it, upload to S3, and return the edited image info to the UI.
    Idempotent-ish: if this correlation_id was already exported, returns
    the cached result instead of re-exporting from Canva every time the
    frontend calls it (e.g. on a page refresh)."""
    session_id = get_or_create_session_id(request, response)

    record = canva_design_registry.verify_design_ownership(session_id, correlation_id)
    if record is None:
        # section 9: a user cannot submit another user's design ID and
        # retrieve their result / section 10: invalid design ID.
        raise HTTPException(status_code=404, detail="Unknown design, or it does not belong to this session.")

    if record.get("export"):
        # Already exported previously — return the cached result rather
        # than re-hitting Canva's export API for no reason.
        cached = record["export"]
        return {"success": True, "design_id": record["design_id"], **cached}

    try:
        access_token = canva_service.get_valid_access_token(session_id)
    except CanvaAPIError as exc:
        # section 10: expired access token / refresh token failure
        raise HTTPException(status_code=401, detail=str(exc))

    try:
        image_bytes = canva_service.export_design(access_token, record["design_id"])
    except CanvaAPIError as exc:
        # section 10: Canva export failure / export still processing /
        # design not found / Canva API error — export_design() already
        # distinguishes these in its error message.
        raise HTTPException(status_code=502, detail=f"Canva export failed: {exc}")

    try:
        s3_key, edited_image_url = image_storage_service.upload_edited_image(
            image_bytes, session_id, record["design_id"],
        )
    except RuntimeError as exc:
        # section 10: S3 upload failure
        raise HTTPException(status_code=502, detail=f"Failed to save edited image to S3: {exc}")

    canva_design_registry.record_export_result(correlation_id, s3_key, edited_image_url)

    # Matches the response shape requested in section 6, adapted onto
    # this project's existing error/response conventions (HTTPException
    # for failures, plain dict for success — same pattern as every other
    # route in this file).
    return {
        "success": True,
        "design_id": record["design_id"],
        "edited_image_url": edited_image_url,
        "s3_key": s3_key,
    }