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

    _ASPECT_RATIOS = {
        (1, 1): "square_1_1",
        (4, 3): "classic_4_3",
        (3, 4): "traditional_3_4",
        (16, 9): "widescreen_16_9",
        (9, 16): "social_story_9_16",
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
                    logger.warning(
                        "Freepik submission attempt %d/%d failed for model '%s' (status=%s): %s",
                        attempt, attempts, self._model_label(), status, exc,
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


def get_image_gen_service(poll_timeout_override: Optional[float] = None):
    """Factory for the configured image-rendering backend. All three
    expose the same `.generate_image(prompt, negative_prompt="",
    on_progress=None) -> bytes` interface. Provider chosen entirely by
    settings.IMAGE_PROVIDER — no code change needed to switch. Defaults
    to Freepik.

    `poll_timeout_override`, if given, replaces FREEPIK_POLL_TIMEOUT for
    this instance only (used by generate_variations_events() to keep a
    single request's image phase inside its allotted budget without
    changing the global default for the plain /api/generate path).
    """
    if settings.IMAGE_PROVIDER == "aws":
        return NovaCanvasService(
            model=settings.BEDROCK_IMAGE_GEN_MODEL,
            region_name=settings.AWS_IMAGE_REGION,
            aws_access_key_id=settings.AWS_IMAGE_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_IMAGE_SECRET_ACCESS_KEY,
            width=settings.IMAGE_WIDTH,
            height=settings.IMAGE_HEIGHT,
            quality=settings.IMAGE_QUALITY,
            cfg_scale=settings.IMAGE_CFG_SCALE,
        )
    if settings.IMAGE_PROVIDER == "pollinations":
        return PollinationsImageService(
            model=settings.POLLINATIONS_MODEL,
            base_url=settings.POLLINATIONS_BASE_URL,
            width=settings.IMAGE_WIDTH,
            height=settings.IMAGE_HEIGHT,
        )
    return FreepikImageService(
        api_key=settings.FREEPIK_API_KEY,
        model=settings.FREEPIK_MODEL,
        resolution=settings.FREEPIK_RESOLUTION,
        width=settings.IMAGE_WIDTH,
        height=settings.IMAGE_HEIGHT,
        filter_nsfw=settings.FREEPIK_FILTER_NSFW,
        poll_interval=settings.FREEPIK_POLL_INTERVAL,
        poll_timeout=poll_timeout_override if poll_timeout_override is not None else settings.FREEPIK_POLL_TIMEOUT,
        max_concurrent_submissions=settings.FREEPIK_MAX_CONCURRENT_SUBMISSIONS,
        submit_max_retries=settings.FREEPIK_SUBMIT_MAX_RETRIES,
        submit_retry_backoff=settings.FREEPIK_SUBMIT_RETRY_BACKOFF_SECONDS,
    )


def generate_variations(prompts: List[str], negative_prompt: str = "") -> List[bytes]:
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
    service = get_image_gen_service()
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
    """
    service = get_image_gen_service(poll_timeout_override=deadline_seconds)
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