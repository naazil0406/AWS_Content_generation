"""
Image Generation Service — provider-independent abstraction over the
Image Generation Agent.

Every provider exposes the same `generate_image(prompt, negative_prompt="",
on_progress=None) -> bytes` interface, so `get_image_gen_service()` can
hand back whichever one settings.IMAGE_PROVIDER selects, and
`generate_variations()`/`generate_variations_events()` don't need to know
which provider it's talking to. Adding a new provider means adding one
class with a `generate_image` method — no other code changes.

Three transports are included:
  - FreepikImageService: Freepik Mystic / Flux / Seedream endpoints (default).
  - PollinationsImageService: free, no API key required.
  - NovaCanvasService: AWS Bedrock Nova Canvas.

`on_progress`, if given, is called from the worker thread with
`(status: str, detail: dict)` at each meaningful step (submitted,
polling, downloading, completed) — used by generate_variations_events()
to stream live per-image progress over SSE. It's a no-op by default so
the plain generate_image() call used elsewhere doesn't need to change.
"""

import base64
import io
import json
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import gcd
from typing import Callable, List, Optional
from urllib.parse import quote

import requests

from app.config import settings
from app.schemas.content import IMAGE_RESOLUTION_PIXELS, IMAGE_RESOLUTIONS

logger = logging.getLogger(__name__)

# Signature: on_progress(status, detail) -> None
ProgressCallback = Optional[Callable[[str, dict], None]]


def _noop_progress(status: str, detail: dict) -> None:
    return None


class PollinationsImageService:
    """Transport over the free Pollinations AI image-generation API."""

    def __init__(self, model: str, base_url: str, width: int, height: int, timeout: int = 60):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.width = width
        self.height = height
        self.timeout = timeout

    def generate_image(self, prompt: str, negative_prompt: str = "", on_progress: ProgressCallback = None) -> bytes:
        on_progress = on_progress or _noop_progress
        full_prompt = f"{prompt}. Avoid: {negative_prompt}" if negative_prompt else prompt
        url = f"{self.base_url}/{quote(full_prompt)}"
        params = {"model": self.model, "width": self.width, "height": self.height, "nologo": "true"}

        logger.info("Calling Pollinations AI model '%s' (prompt=%d chars).", self.model, len(prompt))
        on_progress("submitted", {"provider": "pollinations"})
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Pollinations AI request failed for model '{self.model}': {exc}") from exc

        if not response.content:
            raise RuntimeError("Pollinations AI returned no image data.")
        on_progress("completed", {})
        return response.content


class FreepikImageService:
    """Transport over Freepik's text-to-image APIs (Mystic / Flux / Seedream)."""

    # Maps a "W:H" ratio string (exactly what the frontend sends via
    # GenerateContentRequest.image_aspect_ratio — see
    # app/schemas/content.py's IMAGE_ASPECT_RATIOS) directly to the
    # `aspect_ratio` enum value Freepik's API expects. This is the
    # primary lookup now; _ASPECT_RATIOS (width/height-tuple keyed,
    # below) is kept only for the width/height-based providers
    # (Pollinations/Nova Canvas) and as a fallback when no explicit
    # ratio string was given.
    #
    # CONFIDENCE NOTE: the first five (1:1, 4:3, 3:4, 16:9, 9:16) were
    # already in this file before this change and are what production
    # has actually been sending to Freepik. The sixth, 2:3, is this
    # file's best-effort mapping to Freepik's own documented Mystic
    # aspect_ratio enum — this sandbox has no live network access to
    # api.freepik.com to re-confirm it at the moment this code was
    # written. It's the ONE ratio in this reduced set worth watching
    # closely in testing: if it comes back with a 400/422 from Freepik
    # (see _submit_with_retry's response-body logging, which will show
    # Freepik's exact validation message), that specific string needs
    # correcting against Freepik's current API reference — the other
    # five are unaffected either way, since each ratio is looked up
    # independently. Do not add another ratio here without the same
    # verification caveat — never invent a plausible-looking enum value
    # with no basis at all. (3:2, 1:2, 2:1, 4:5, and 21:9 were removed
    # from the supported set entirely — not just hidden from the UI —
    # per product decision; if you need to bring one back, its old
    # best-effort mapping is in this file's git history.)
    _ASPECT_RATIO_STRINGS = {
        "1:1": "square_1_1",
        "4:3": "classic_4_3",
        "3:4": "traditional_3_4",
        "16:9": "widescreen_16_9",
        "9:16": "social_story_9_16",
        "2:3": "portrait_2_3",  # best-effort, see CONFIDENCE NOTE above
    }

    # Legacy width/height-tuple keyed lookup — still used by
    # _aspect_ratio_for() below as the fallback path when a caller
    # doesn't pass an explicit ratio string (e.g. IMAGE_WIDTH/
    # IMAGE_HEIGHT-only callers, or the Pollinations/Nova Canvas
    # providers elsewhere in this file, which take raw width/height and
    # never touch this class at all). Kept in sync with
    # _ASPECT_RATIO_STRINGS above; if you add a ratio to one, add it to
    # both.
    _ASPECT_RATIOS = {
        (1, 1): "square_1_1",
        (4, 3): "classic_4_3",
        (3, 4): "traditional_3_4",
        (16, 9): "widescreen_16_9",
        (9, 16): "social_story_9_16",
        (2, 3): "portrait_2_3",
    }
    _FLUX_ENDPOINTS = {
        "flux-dev": "https://api.freepik.com/v1/ai/text-to-image/flux-dev",
        "hyperflux": "https://api.freepik.com/v1/ai/text-to-image/hyperflux",
        "seedream-v4": "https://api.freepik.com/v1/ai/text-to-image/seedream-v4",
        # Seedream 4.5 — ByteDance's model, marketed specifically for
        # "superior typography and text rendering" vs earlier models.
        # Selected purely via FREEPIK_MODEL=seedream-v4-5 (env var) — no
        # other code path changes, same as every other Freepik model here.
        "seedream-v4-5": "https://api.freepik.com/v1/ai/text-to-image/seedream-v4-5",
        # Seedream V5 Lite — ByteDance's newest model on Freepik as of
        # this writing. Selected via FREEPIK_MODEL=seedream-v5-lite.
        # NOTE: the exact string must be "seedream-v5-lite" (not
        # "seedream-v5" or "seedream-v.5") to match this dict key and
        # Freepik's actual endpoint path — any other spelling falls
        # through to the `.get(model, self._MYSTIC_URL)` default below
        # and silently submits to Mystic instead, which is the slow
        # ~90-130s model this change is meant to avoid.
        "seedream-v5-lite": "https://api.freepik.com/v1/ai/text-to-image/seedream-v5-lite",
    }
    _MYSTIC_URL = "https://api.freepik.com/v1/ai/mystic"

    # Which resolution tier LABELS ("2K"/"4K" — see
    # app/schemas/content.py's IMAGE_RESOLUTIONS) each Freepik model
    # actually accepts as a `resolution` request parameter. Keyed by
    # `self.model` exactly as it appears in _FLUX_ENDPOINTS above, plus
    # "" for Mystic (FREEPIK_MODEL unset) — Freepik's Mystic engine is
    # what "Magnific" refers to in product/UI copy for this integration;
    # there is no separate Magnific endpoint or code path here.
    #
    # Only 2K and 4K are offered anywhere in this application — 1K is
    # deliberately excluded by product decision (see
    # app/schemas/content.py's IMAGE_RESOLUTIONS), not a technical
    # limitation of any of these models.
    # Mystic: both tiers (moderate confidence, matches this file's
    # pre-existing 2k/4k default).
    # seedream-v4-5: 2K and 4K — confirmed in production: sending NO
    # resolution parameter at all to this endpoint produced a live 400
    # Bad Request (this model apparently requires one, unlike the other
    # Flux/Seedream endpoints, which reject it if you DO send one) —
    # added here to fix that.
    # Every other Flux/Seedream model (flux-dev, hyperflux, seedream-v4,
    # seedream-v5-lite): not in this dict at all -> no resolution
    # parameter is ever sent, exactly as before this change (their
    # `.get(model, set())` below returns an empty set).
    _RESOLUTION_SUPPORT = {
        "": {"2K", "4K"},
        "seedream-v4-5": {"2K", "4K"},
    }

    def __init__(
        self,
        api_key: str,
        model: str,
        resolution: str,
        width: int,
        height: int,
        filter_nsfw: bool,
        poll_interval: float,
        poll_timeout: float,
        max_concurrent_submissions: int = 1,
        submit_max_retries: int = 2,
        submit_retry_backoff: float = 0.75,
        aspect_ratio_key: Optional[str] = None,
    ):
        if not api_key:
            raise RuntimeError(
                "FREEPIK_API_KEY is not set — required when IMAGE_PROVIDER='freepik'. "
                "If you've already added this key via `sam deploy` or the Lambda console "
                "and are still seeing this: Lambda versions freeze their environment "
                "variables at publish time, so `aws lambda get-function-configuration` "
                "(which reads $LATEST by default) can show the key while the specific "
                "version the 'prod' alias actually points to — the one really serving "
                "traffic — was published before the key existed and has none. Check the "
                "version this exact request ran on via this log line's own cold-start "
                "message (search CloudWatch for 'Cold start' — it logs the version and "
                "whether the key was set for THAT version), or re-run "
                "`aws lambda get-function-configuration --qualifier prod` to check prod's "
                "actual version rather than $LATEST. The fix is a fresh `sam deploy` with "
                "the key in --parameter-overrides, which publishes a new version with the "
                "key baked in and moves prod to it — editing $LATEST directly never "
                "moves prod."
            )
        self.api_key = api_key
        self.model = model
        self.resolution = resolution
        # aspect_ratio_key ("16:9" etc., exactly what the frontend/request
        # sends — see app/schemas/content.py's IMAGE_ASPECT_RATIOS) is the
        # preferred, precise path via _ASPECT_RATIO_STRINGS. width/height
        # is kept only as a fallback for callers that don't pass an
        # explicit ratio string, so nothing that already worked (e.g. a
        # request that omits image_aspect_ratio entirely) changes
        # behavior.
        if aspect_ratio_key and aspect_ratio_key in self._ASPECT_RATIO_STRINGS:
            self.aspect_ratio = self._ASPECT_RATIO_STRINGS[aspect_ratio_key]
        else:
            self.aspect_ratio = self._aspect_ratio_for(width, height)
        self.filter_nsfw = filter_nsfw
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.submit_max_retries = max(0, submit_max_retries)
        self.submit_retry_backoff = submit_retry_backoff
        # Shared across this instance's worker threads (generate_variations()
        # creates ONE FreepikImageService and calls .generate_image() on it
        # from all 3 threads for a single request) — never across separate
        # users' concurrent requests, since each request builds its own
        # instance via get_image_gen_service().
        self._submit_semaphore = threading.Semaphore(max(1, max_concurrent_submissions))
        self.submit_url = self._FLUX_ENDPOINTS.get(model, self._MYSTIC_URL)
        self.is_mystic = self.submit_url == self._MYSTIC_URL
        # True for Mystic ("" key) and any Flux/Seedream model listed in
        # _RESOLUTION_SUPPORT above (currently just seedream-v4-5) — see
        # that dict's comments for why. `self.resolution` is expected to
        # already be resolved to a value valid for THIS model by the
        # caller (see get_image_gen_service()'s resolve_freepik_resolution()
        # below) — this flag only controls whether _build_body includes
        # the "resolution" key in the request body at all.
        self.resolution_supported = model in self._RESOLUTION_SUPPORT

    def _aspect_ratio_for(self, width: int, height: int) -> str:
        divisor = gcd(width, height) or 1
        return self._ASPECT_RATIOS.get((width // divisor, height // divisor), "square_1_1")

    def _build_body(self, full_prompt: str, negative_prompt: str = "") -> dict:
        if self.is_mystic:
            body = {
                "prompt": full_prompt,
                "resolution": self.resolution,
                "aspect_ratio": self.aspect_ratio,
                "filter_nsfw": self.filter_nsfw,
            }
            # Only include "model" when FREEPIK_MODEL was actually set to
            # something. self.model defaults to "" (see
            # settings.FREEPIK_MODEL in config.py) when unset, and sending
            # that empty string to Freepik as the "model" field is invalid
            # — Freepik's Mystic endpoint rejects it with a 400 Bad Request
            # (this was a real bug: the code correctly fell through to the
            # generic Mystic endpoint on an empty/unrecognized model name,
            # but then still forwarded that same empty string in the
            # request body instead of omitting the field). Omitting it
            # lets Freepik apply its own default model instead.
            if self.model:
                body["model"] = self.model
        else:
            body = {"prompt": full_prompt, "aspect_ratio": self.aspect_ratio}
            # Some Flux/Seedream endpoints DO accept (or even require) a
            # resolution parameter despite not being Mystic — confirmed
            # live for seedream-v4-5, which returned a 400 Bad Request
            # when no resolution was sent at all. See _RESOLUTION_SUPPORT
            # above for exactly which models/tiers this applies to; every
            # other Flux/Seedream model still gets no resolution field at
            # all, unchanged from before.
            if self.resolution_supported and self.resolution:
                body["resolution"] = self.resolution
        if negative_prompt:
            # A genuinely separate structured field, not just the trailing
            # ". Avoid: ..." text baked into full_prompt below — that
            # trailing text is STILL included too (defense-in-depth, and
            # harmless if this field isn't actually honored by whichever
            # specific Freepik endpoint is configured), but relying on it
            # ALONE was a real gap: with scene-mode prompts often running
            # to many lines, a single trailing
            # sentence of style exclusions gets diluted at the tail of an
            # increasingly long, dense positive description — a real,
            # plausible explanation for cartoon/illustration-style output
            # slipping through despite the positive prompt explicitly
            # requesting photorealism. Passing it as its own field gives
            # the model a real chance to weight it independently of the
            # positive prompt's own length. NOTE: verify this exact field
            # name against Freepik's current API docs for whichever
            # specific model is configured — this sandbox has no live
            # network access to confirm it live.
            body["negative_prompt"] = negative_prompt
        return body

    def _model_label(self) -> str:
        return self.model or "(Mystic default)"

    def generate_image(self, prompt: str, negative_prompt: str = "", on_progress: ProgressCallback = None) -> bytes:
        on_progress = on_progress or _noop_progress
        full_prompt = f"{prompt}. Avoid: {negative_prompt}" if negative_prompt else prompt
        headers = {"Content-Type": "application/json", "x-freepik-api-key": self.api_key}
        body = self._build_body(full_prompt, negative_prompt)

        logger.info("Submitting to Freepik (model=%s, prompt=%d chars).", self._model_label(), len(prompt))
        on_progress("submitted", {"provider": "freepik", "model": self._model_label()})
        response = self._submit_with_retry(body, headers)

        task = (response.json() or {}).get("data", {})
        task_id = task.get("task_id")
        if not task_id:
            raise RuntimeError(f"Freepik ({self._model_label()}) did not return a task_id.")

        image_url = self._poll_until_complete(task_id, headers, on_progress)
        on_progress("downloading", {})
        try:
            image_response = requests.get(image_url, timeout=60)
            image_response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to download the generated image from Freepik: {exc}") from exc

        if not image_response.content:
            raise RuntimeError("Freepik image download returned no data.")
        on_progress("completed", {})
        return image_response.content

    def _submit_with_retry(self, body: dict, headers: dict) -> "requests.Response":
        """POSTs to submit_url, serialized against this instance's other
        in-flight images via _submit_semaphore and retried up to
        submit_max_retries times on any failure.

        Why: observed in production, 2 of 3 truly-simultaneous submissions
        for the same request failed at the same millisecond with
        DIFFERENT error codes (401 on one, 422 on another) for otherwise
        identical requests — the signature of a collision at Freepik's
        gateway (a real bad key/payload fails every attempt identically;
        this doesn't). The semaphore avoids re-creating that collision on
        retry; the retry recovers from it if it still happens anyway.
        Only the fast initial POST goes through this — polling and
        downloading (the slow ~90-130s part for Mystic) are untouched and
        still run fully in parallel across images.
        """
        last_exc: Optional[Exception] = None
        attempts = self.submit_max_retries + 1
        for attempt in range(1, attempts + 1):
            with self._submit_semaphore:
                try:
                    response = requests.post(self.submit_url, headers=headers, json=body, timeout=30)
                    response.raise_for_status()
                    return response
                except requests.RequestException as exc:
                    last_exc = exc
                    status = getattr(exc.response, "status_code", None)
                    # The status code alone (e.g. "400") doesn't say WHICH
                    # field/value Freepik rejected — for a 4xx, that detail
                    # is in the response body, and this is genuinely the
                    # only reliable way to find out (this sandbox has no
                    # live access to Freepik's API to pre-verify every
                    # enum value, e.g. aspect_ratio strings for less common
                    # ratios/models). Log up to 500 chars of it; truncated
                    # defensively in case Freepik ever returns something
                    # huge (e.g. an HTML error page instead of JSON).
                    body_text = ""
                    if exc.response is not None:
                        try:
                            body_text = exc.response.text[:500]
                        except Exception:  # noqa: BLE001 - best-effort only
                            body_text = "<could not read response body>"
                    logger.warning(
                        "Freepik submission attempt %d/%d failed for model '%s' (status=%s): %s | "
                        "response body: %s | request body sent: %s",
                        attempt, attempts, self._model_label(), status, exc, body_text, body,
                    )
            if attempt < attempts:
                time.sleep(self.submit_retry_backoff * attempt)  # linear backoff: 0.75s, 1.5s, ...

        raise RuntimeError(
            f"Freepik submission failed for model '{self._model_label()}' after {attempts} attempt(s): {last_exc}"
        ) from last_exc


    def _poll_until_complete(self, task_id: str, headers: dict, on_progress: ProgressCallback = None) -> str:
        on_progress = on_progress or _noop_progress
        status_url = f"{self.submit_url}/{task_id}"
        started = time.monotonic()
        deadline = started + self.poll_timeout
        attempt = 0

        while True:
            attempt += 1
            try:
                status_response = requests.get(status_url, headers=headers, timeout=30)
                status_response.raise_for_status()
            except requests.RequestException as exc:
                raise RuntimeError(f"Freepik status check failed: {exc}") from exc

            data = (status_response.json() or {}).get("data", {})
            status = data.get("status")
            elapsed = time.monotonic() - started
            on_progress("polling", {"attempt": attempt, "elapsed_seconds": round(elapsed, 1), "freepik_status": status})

            if status == "COMPLETED":
                generated = data.get("generated", [])
                if not generated:
                    raise RuntimeError(f"Freepik task '{task_id}' completed but returned no image.")
                return generated[0]
            if status == "FAILED":
                raise RuntimeError(f"Freepik task '{task_id}' failed: {data}")
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Freepik task '{task_id}' did not complete within {self.poll_timeout:.0f}s "
                    f"(last status: {status!r})."
                )
            time.sleep(self.poll_interval)


class NovaCanvasService:
    """Transport over AWS Bedrock's Nova Canvas image-generation model."""

    def __init__(
        self,
        model: str,
        region_name: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        width: int,
        height: int,
        quality: str,
        cfg_scale: float,
    ):
        import boto3

        self.model = model
        self.width = width
        self.height = height
        self.quality = quality
        self.cfg_scale = cfg_scale

        self.client = boto3.client("bedrock-runtime", region_name=region_name)

    def generate_image(self, prompt: str, negative_prompt: str = "", on_progress: ProgressCallback = None) -> bytes:
        on_progress = on_progress or _noop_progress
        from botocore.exceptions import BotoCoreError, ClientError

        text_to_image_params = {"text": prompt}
        if negative_prompt:
            text_to_image_params["negativeText"] = negative_prompt

        body = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": text_to_image_params,
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "quality": self.quality,
                "width": self.width,
                "height": self.height,
                "cfgScale": self.cfg_scale,
            },
        }

        logger.info("Calling Nova Canvas model '%s' (prompt=%d chars).", self.model, len(prompt))
        on_progress("submitted", {"provider": "aws", "model": self.model})
        try:
            response = self.client.invoke_model(
                modelId=self.model, body=json.dumps(body),
                contentType="application/json", accept="application/json",
            )
        except (ClientError, BotoCoreError) as exc:
            raise RuntimeError(f"Bedrock invoke_model failed for Nova Canvas model '{self.model}': {exc}") from exc

        response_body = json.loads(response["body"].read())
        if response_body.get("error"):
            raise RuntimeError(f"Nova Canvas returned an error: {response_body['error']}")

        images = response_body.get("images", [])
        if not images:
            raise RuntimeError("Nova Canvas returned no images.")
        on_progress("completed", {})
        return base64.b64decode(images[0])


# --- Ratio -> pixel dimensions, for providers that take raw width/height
# instead of an aspect-ratio enum string (Pollinations, Nova Canvas) -----
_RATIO_TO_WH = {
    "1:1": (1, 1), "16:9": (16, 9), "9:16": (9, 16), "4:3": (4, 3),
    "3:4": (3, 4), "2:3": (2, 3),
}


def dimensions_for_ratio(
    ratio_key: Optional[str],
    fallback_width: int,
    fallback_height: int,
    resolution_pixels: Optional[int] = None,
) -> tuple:
    """Convert a "W:H" ratio string (see app/schemas/content.py's
    IMAGE_ASPECT_RATIOS) into concrete (width, height) pixel dimensions
    for the providers that need raw pixels rather than an enum string
    (Pollinations, Nova Canvas). Freepik's aspect ratio never calls this
    for the ratio itself — it uses FreepikImageService's own
    _ASPECT_RATIO_STRINGS lookup directly, which is the precise,
    provider-native path.

    `resolution_pixels`, if given, is the exact LONG EDGE pixel target
    from app/schemas/content.py's IMAGE_RESOLUTION_PIXELS (2K=2048,
    4K=3840 — see GenerateContentRequest.image_resolution) —
    the longer of width/height is set to exactly this value, and the
    shorter side is derived proportionally from the ratio, both rounded
    to a multiple of 64. Falls back to 1024 (the pre-existing default,
    used only when no resolution is given at all)
    when omitted, so a request that only sets image_aspect_ratio and not
    image_resolution behaves exactly as before this parameter existed.

    Nova Canvas specifically requires width/height to each be a multiple
    of 64 within a supported range, AND a total pixel count under
    roughly 4 megapixels (moderate confidence from AWS Bedrock
    documentation, not independently re-verified live in this sandbox —
    same caveat as the Freepik aspect-ratio strings elsewhere in this
    file) — both are enforced below so an extreme combination (e.g. 4K +
    9:16) can't produce an out-of-range or over-budget request; the
    result is scaled down proportionally rather than sent as-is. This is
    the "handle gracefully instead of sending an invalid API request"
    behavior for the width/height-based providers specifically —
    Freepik's own graceful-degradation path is
    app.routes.content._resolution_support_for_current_config().

    Falls back to (fallback_width, fallback_height) unchanged when
    ratio_key is None or not recognized, so omitting image_aspect_ratio
    on a request behaves exactly as before this feature existed.
    """
    if not ratio_key or ratio_key not in _RATIO_TO_WH:
        return fallback_width, fallback_height
    rw, rh = _RATIO_TO_WH[ratio_key]
    long_edge = resolution_pixels or 1024

    if rw >= rh:
        width = long_edge
        height = long_edge * rh / rw
    else:
        height = long_edge
        width = long_edge * rw / rh

    width = max(64, round(width / 64) * 64)
    height = max(64, round(height / 64) * 64)

    # Nova Canvas's documented supported per-side range is roughly
    # 320-4096 (moderate confidence, not re-verified live) — clamp
    # defensively.
    width = min(max(width, 320), 4096)
    height = min(max(height, 320), 4096)

    # ...and its documented total-pixel-count ceiling is roughly 4
    # megapixels (same confidence caveat) — scale both sides down
    # proportionally, still rounded to a multiple of 64, if the 4K tier
    # combined with a wide/tall ratio would exceed it. This is what
    # keeps a "4K" + "9:16" request from becoming an invalid Nova Canvas
    # call instead of erroring outright.
    max_total_pixels = 4_194_304
    if width * height > max_total_pixels:
        scale = (max_total_pixels / (width * height)) ** 0.5
        width = max(64, round((width * scale) / 64) * 64)
        height = max(64, round((height * scale) / 64) * 64)

    return width, height


def freepik_model_resolution_support(model: str) -> set:
    """The set of resolution tier labels ("2K"/"4K") `model`
    accepts as a `resolution` parameter — see
    FreepikImageService._RESOLUTION_SUPPORT for the authoritative table
    and the evidence behind each entry. Empty set means this model has
    no resolution parameter at all; a resolution is never sent to it."""
    return FreepikImageService._RESOLUTION_SUPPORT.get(model, set())


def freepik_model_supports_resolution(model: str) -> bool:
    """True when `model` accepts a resolution parameter in ANY tier —
    see freepik_model_resolution_support() above for which ones
    specifically. False means no resolution parameter is ever sent to
    this model at all."""
    return bool(freepik_model_resolution_support(model))


def resolve_freepik_resolution(model: str, requested: Optional[str]) -> str:
    """The exact lowercase resolution string to send to Freepik for
    `model` ("" == Mystic), or "" if this model has no resolution
    parameter at all (see freepik_model_resolution_support() above) —
    FreepikImageService._build_body only includes the "resolution" key
    when this returns non-empty AND self.resolution_supported is True.

    - `requested` given and supported by this model: used directly
      (e.g. "2K" -> "2k").
    - `requested` omitted, or given but NOT one of this model's
      supported tiers: falls back to settings.FREEPIK_RESOLUTION for
      Mystic (its long-standing default), or the lowest supported tier
      for any other model that has one at all — some Flux/Seedream
      endpoints (confirmed live for seedream-v4-5) reject a request
      with NO resolution parameter, so a model that supports resolution
      at all always gets a valid value here, never an omission.
    - `model` has no supported tiers: always "" — never sent, exactly
      as before this parameter existed for that model.
    """
    supported = freepik_model_resolution_support(model)
    if not supported:
        return ""
    if requested and requested in supported:
        return requested.lower()
    if model == "":  # Mystic
        return settings.FREEPIK_RESOLUTION
    return sorted(supported)[0].lower()


def resolution_support_status(resolution: Optional[str] = None) -> tuple:
    """(applies: bool, note: str) describing whether the CURRENTLY
    configured provider/model (app.config.settings.IMAGE_PROVIDER /
    FREEPIK_MODEL) can actually use `resolution` (or, when omitted, ANY
    resolution at all) in some form:
      - aws / pollinations: always True — resolution becomes the
        long-edge pixel target in dimensions_for_ratio() above.
      - freepik + a model with no resolution support at all: False.
      - freepik + a model that supports SOME tiers but not the specific
        `resolution` asked about: False, with a note naming which tiers
        it does support (e.g. some models support only one tier).
      - freepik + a model that supports `resolution` (or `resolution`
        omitted and the model supports at least one tier): True.
    `note` is a human-readable explanation, empty when applies is True.
    Used by GET /image-settings (informational, called with
    resolution=None for a general "does this provider support
    resolution at all" check) and by
    app.routes.content._resolution_support_for_current_config() (called
    with the actual requested tier, to decide whether to drop it for one
    call and surface a response warning — see GenerateContentResponse.
    warnings) — this function itself never raises or blocks anything.
    """
    if settings.IMAGE_PROVIDER in ("aws", "pollinations"):
        return True, ""
    if settings.IMAGE_PROVIDER == "freepik":
        supported = freepik_model_resolution_support(settings.FREEPIK_MODEL)
        if not supported:
            return False, (
                f"The configured Freepik model ('{settings.FREEPIK_MODEL}') has no resolution "
                "parameter — resolution selections are ignored."
            )
        if resolution is None or resolution in supported:
            return True, ""
        return False, (
            f"The configured Freepik model ('{settings.FREEPIK_MODEL}') only supports "
            f"{'/'.join(sorted(supported))} resolution — '{resolution}' selections are ignored."
        )
    return False, f"Resolution is not supported for IMAGE_PROVIDER='{settings.IMAGE_PROVIDER}'."


def get_image_gen_service(
    poll_timeout_override: Optional[float] = None,
    aspect_ratio: Optional[str] = None,
    resolution: Optional[str] = None,
):
    """Factory for the configured image-rendering backend. All three
    expose the same `.generate_image(prompt, negative_prompt="",
    on_progress=None) -> bytes` interface. Provider chosen entirely by
    settings.IMAGE_PROVIDER — no code change needed to switch. Defaults
    to Freepik.

    `poll_timeout_override`, if given, replaces FREEPIK_POLL_TIMEOUT for
    this instance only (used by generate_variations_events() to keep a
    single request's image phase inside its allotted budget without
    changing the global default for the plain /api/generate path).

    `aspect_ratio` / `resolution`, if given, are the PER-REQUEST values
    from GenerateContentRequest.image_aspect_ratio /
    .image_resolution (see app/schemas/content.py) — they override
    settings.IMAGE_WIDTH/IMAGE_HEIGHT/FREEPIK_RESOLUTION for this one
    call only, never touching the global default other concurrent
    requests get. Both are optional and independently overridable; when
    either is omitted, that one setting's server default applies exactly
    as before this feature existed.

    `resolution` is expected to already have passed
    app.routes.content._resolution_support_for_current_config() — this
    function does NOT re-check whether the current provider/model can
    actually honor it; a caller that wants graceful degradation (never
    sending Freepik a parameter its configured model doesn't support)
    should pass None here when that check said "no" and surface its own
    warning instead. Given a value, this function applies it as far as
    the current provider CAN use it: as the long-edge pixel target for
    aws/pollinations (see dimensions_for_ratio()), or as Freepik
    Mystic's own resolution string (lowercased, e.g. "2K" -> "2k") only
    when settings.FREEPIK_MODEL is unset — passed for a Flux/Seedream
    model, it is silently not included in the request body at all (see
    FreepikImageService._build_body's `if self.is_mystic:` guard), never
    sent as an invalid parameter.
    """
    resolution_pixels = IMAGE_RESOLUTION_PIXELS.get(resolution) if resolution else None
    width, height = dimensions_for_ratio(
        aspect_ratio, settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT, resolution_pixels
    )
    if settings.IMAGE_PROVIDER == "aws":
        return NovaCanvasService(
            model=settings.BEDROCK_IMAGE_GEN_MODEL,
            region_name=settings.AWS_IMAGE_REGION,
            aws_access_key_id=settings.AWS_IMAGE_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_IMAGE_SECRET_ACCESS_KEY,
            width=width,
            height=height,
            quality=settings.IMAGE_QUALITY,
            cfg_scale=settings.IMAGE_CFG_SCALE,
        )
    if settings.IMAGE_PROVIDER == "pollinations":
        return PollinationsImageService(
            model=settings.POLLINATIONS_MODEL,
            base_url=settings.POLLINATIONS_BASE_URL,
            width=width,
            height=height,
        )
    # Freepik: resolve to whatever this specific model actually accepts
    # (Mystic supports all 3 tiers; some Flux/Seedream models support a
    # subset or none at all — see resolve_freepik_resolution() and
    # FreepikImageService._RESOLUTION_SUPPORT for the full per-model
    # picture). Returns "" for a model with no resolution parameter at
    # all, which FreepikImageService._build_body then never includes.
    resolved_resolution = resolve_freepik_resolution(settings.FREEPIK_MODEL, resolution)
    return FreepikImageService(
        api_key=settings.FREEPIK_API_KEY,
        model=settings.FREEPIK_MODEL,
        resolution=resolved_resolution,
        width=width,
        height=height,
        aspect_ratio_key=aspect_ratio,
        filter_nsfw=settings.FREEPIK_FILTER_NSFW,
        poll_interval=settings.FREEPIK_POLL_INTERVAL,
        poll_timeout=poll_timeout_override if poll_timeout_override is not None else settings.FREEPIK_POLL_TIMEOUT,
        max_concurrent_submissions=settings.FREEPIK_MAX_CONCURRENT_SUBMISSIONS,
        submit_max_retries=settings.FREEPIK_SUBMIT_MAX_RETRIES,

        submit_retry_backoff=settings.FREEPIK_SUBMIT_RETRY_BACKOFF_SECONDS,
    )


# --- Image Count support ----------------------------------------------
# The Image Prompt Agent (prompts/image_prompt_system.txt) always
# produces exactly 3 distinct, content-anchored prompts — that logic is
# NOT touched by the image_count feature below. This function only
# decides, post-generation, how many of / how many times to actually
# RENDER those 3 prompts for this one request.
def resolve_prompts_for_count(prompts: List[str], count: Optional[int]) -> List[str]:
    """Expand or truncate `prompts` (always the 3 Image Prompt Agent
    entries, in practice) to exactly `count` entries:
      - count is None or <= 0: returns `prompts` unchanged (the
        pre-existing default behavior of "render all 3", exactly as
        before this feature existed).
      - count < len(prompts): returns the first `count` prompts.
      - count == len(prompts): returns `prompts` unchanged.
      - count > len(prompts): cycles through `prompts` round-robin to
        fill up to `count` entries (e.g. count=5 over 3 prompts ->
        [p0, p1, p2, p0, p1]) — reusing a prompt is safe because each
        render is an independent, stochastic call to the image model
        (the same property the existing per-image regenerate button
        already relies on), so a repeated prompt still produces a
        different image, not a literal duplicate.

    Callers that need content_anchors kept aligned by index with the
    returned list should apply this same function (same count, same
    input length) to that list too — see app/routes/content.py.
    """
    if not prompts:
        return []
    if count is None or count <= 0:
        return list(prompts)
    if count <= len(prompts):
        return prompts[:count]
    return [prompts[i % len(prompts)] for i in range(count)]


def generate_variations(
    prompts: List[str],
    negative_prompt: str = "",
    aspect_ratio: Optional[str] = None,
    resolution: Optional[str] = None,
) -> List[bytes]:
    """Generate one image per prompt in `prompts`, in parallel. Unlike the
    old "N variations of ONE prompt" design (which relied purely on the
    image model's own randomness and tended to produce near-identical
    images), each entry in `prompts` is expected to already be a distinct,
    content-related prompt (see the Image Prompt Generation Agent's
    "image_prompts" array in prompts/image_prompt_system.txt) — this
    function's job is just to render each one, not to introduce variety
    itself. A failure on one image does not abort the others; if fewer
    than len(prompts) succeed, the caller decides whether that's
    acceptable.

    Each generate_image() call is a blocking network round-trip (and, for
    polling providers like Freepik, several round-trips) with no shared
    state between images, so they run concurrently on a thread pool
    instead of one after another. This is the single biggest latency win
    in the pipeline — three ~10-20s calls in series is 30-60s+; in
    parallel it's ~one call's worth of wall-clock time.

    `aspect_ratio` / `resolution`: per-request overrides (see
    GenerateContentRequest.image_aspect_ratio / .image_resolution) passed
    straight through to get_image_gen_service() — every image in THIS
    call shares the same one instance/settings, same as before this
    feature existed for the request as a whole.

    NOTE: this only raises if EVERY image fails. If some succeed and
    some don't (e.g. 1 of 3), it still logs an ERROR per failure below
    but returns the shorter list quietly — the caller (the plain
    /api/generate route) doesn't currently surface "you asked for 3, got
    1" anywhere the user can see. If you're hitting fewer images than
    requested, check these WARNING/ERROR log lines first — it's very
    likely 2 of the 3 threads are failing (rate limit, Freepik timeout,
    etc.), not a config issue. For live per-image status instead of
    silent partial failure, use generate_variations_events() /
    POST /api/generate/stream.
    """
    service = get_image_gen_service(aspect_ratio=aspect_ratio, resolution=resolution)
    count = len(prompts)
    results: List[Optional[bytes]] = [None] * count
    errors: List[str] = []

    def _one(i: int) -> None:
        results[i] = service.generate_image(prompts[i], negative_prompt=negative_prompt)

    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = {pool.submit(_one, i): i for i in range(count)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - collect and continue
                logger.error("Image %d/%d failed: %s", i + 1, count, exc)
                errors.append(str(exc))

    images = [img for img in results if img is not None]
    if not images:
        raise RuntimeError(f"All {count} images failed: {'; '.join(errors)}")
    if len(images) < count:
        logger.warning(
            "Only %d/%d images succeeded. Failures: %s",
            len(images), count, "; ".join(errors),
        )
    return images


def generate_variations_events(
    prompts: List[str],
    negative_prompt: str = "",
    deadline_seconds: Optional[float] = None,
    aspect_ratio: Optional[str] = None,
    resolution: Optional[str] = None,
):
    """Generator version of generate_variations() for streaming/SSE
    consumers. Runs the same len(prompts) parallel generate_image() calls
    (one per distinct prompt — see generate_variations() above), but
    yields progress as it happens instead of blocking until everything's
    done. Each yielded dict is one of:

        {"type": "progress", "index": i, "status": "submitted"|"polling"|
            "downloading", "detail": {...}}
        {"type": "image", "index": i, "bytes": <bytes>}
        {"type": "failed", "index": i, "error": "<message>"}

    `deadline_seconds`, if given, bounds the Freepik provider's internal
    poll_timeout for this call only (via get_image_gen_service's
    poll_timeout_override) — this is what keeps a single request's image
    phase inside a caller-defined time budget (e.g. so the whole request
    finishes under 60s) without touching the global FREEPIK_POLL_TIMEOUT
    default used elsewhere. Each image still runs on its own thread in
    parallel, same as generate_variations().

    `aspect_ratio` / `resolution`: same per-request overrides as
    generate_variations() above, passed straight through to
    get_image_gen_service().
    """
    service = get_image_gen_service(
        poll_timeout_override=deadline_seconds, aspect_ratio=aspect_ratio, resolution=resolution
    )
    count = len(prompts)
    events: "queue.Queue" = queue.Queue()
    done_count = 0

    def _make_progress_cb(i: int):
        def _cb(status: str, detail: dict) -> None:
            events.put({"type": "progress", "index": i, "status": status, "detail": detail})
        return _cb

    def _one(i: int) -> None:
        try:
            image_bytes = service.generate_image(
                prompts[i], negative_prompt=negative_prompt, on_progress=_make_progress_cb(i)
            )
            events.put({"type": "image", "index": i, "bytes": image_bytes})
        except Exception as exc:  # noqa: BLE001 - report and let others continue
            logger.error("Image %d/%d failed: %s", i + 1, count, exc)
            events.put({"type": "failed", "index": i, "error": str(exc)})

    pool = ThreadPoolExecutor(max_workers=count)
    for i in range(count):
        pool.submit(_one, i)
    pool.shutdown(wait=False)  # don't block; we drain via the queue below

    # Each worker puts exactly one terminal event ("image" or "failed"),
    # plus zero or more "progress" events along the way. Stop once we've
    # seen `count` terminal events.
    while done_count < count:
        item = events.get()
        yield item
        if item["type"] in ("image", "failed"):
            done_count += 1