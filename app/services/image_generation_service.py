"""
Image Generation Service — provider-independent abstraction over the
Image Generation Agent.

Every provider exposes the same `generate_image(prompt, negative_prompt="")
-> bytes` interface, so `get_image_gen_service()` can hand back whichever
one settings.IMAGE_PROVIDER selects, and `generate_variations()` doesn't
need to know which provider it's talking to. Adding a new provider means
adding one class with a `generate_image` method — no other code changes.

Three transports are included:
  - FreepikImageService: Freepik Mystic / Flux / Seedream endpoints (default).
  - PollinationsImageService: free, no API key required.
  - NovaCanvasService: AWS Bedrock Nova Canvas.
"""

import base64
import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import gcd
from typing import List, Optional
from urllib.parse import quote

import requests

from app.config import settings

logger = logging.getLogger(__name__)


class PollinationsImageService:
    """Transport over the free Pollinations AI image-generation API."""

    def __init__(self, model: str, base_url: str, width: int, height: int, timeout: int = 60):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.width = width
        self.height = height
        self.timeout = timeout

    def generate_image(self, prompt: str, negative_prompt: str = "") -> bytes:
        full_prompt = f"{prompt}. Avoid: {negative_prompt}" if negative_prompt else prompt
        url = f"{self.base_url}/{quote(full_prompt)}"
        params = {"model": self.model, "width": self.width, "height": self.height, "nologo": "true"}

        logger.info("Calling Pollinations AI model '%s' (prompt=%d chars).", self.model, len(prompt))
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Pollinations AI request failed for model '{self.model}': {exc}") from exc

        if not response.content:
            raise RuntimeError("Pollinations AI returned no image data.")
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
    ):
        if not api_key:
            raise RuntimeError("FREEPIK_API_KEY is not set — required when IMAGE_PROVIDER='freepik'.")
        self.api_key = api_key
        self.model = model
        self.resolution = resolution
        self.aspect_ratio = self._aspect_ratio_for(width, height)
        self.filter_nsfw = filter_nsfw
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.submit_url = self._FLUX_ENDPOINTS.get(model, self._MYSTIC_URL)
        self.is_mystic = self.submit_url == self._MYSTIC_URL

    def _aspect_ratio_for(self, width: int, height: int) -> str:
        divisor = gcd(width, height) or 1
        return self._ASPECT_RATIOS.get((width // divisor, height // divisor), "square_1_1")

    def _build_body(self, full_prompt: str) -> dict:
        if self.is_mystic:
            return {
                "prompt": full_prompt,
                "model": self.model,
                "resolution": self.resolution,
                "aspect_ratio": self.aspect_ratio,
                "filter_nsfw": self.filter_nsfw,
            }
        return {"prompt": full_prompt, "aspect_ratio": self.aspect_ratio}

    def generate_image(self, prompt: str, negative_prompt: str = "") -> bytes:
        full_prompt = f"{prompt}. Avoid: {negative_prompt}" if negative_prompt else prompt
        headers = {"Content-Type": "application/json", "x-freepik-api-key": self.api_key}
        body = self._build_body(full_prompt)

        logger.info("Submitting to Freepik (model=%s, prompt=%d chars).", self.model, len(prompt))
        try:
            response = requests.post(self.submit_url, headers=headers, json=body, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Freepik submission failed for model '{self.model}': {exc}") from exc

        task = (response.json() or {}).get("data", {})
        task_id = task.get("task_id")
        if not task_id:
            raise RuntimeError(f"Freepik ({self.model}) did not return a task_id.")

        image_url = self._poll_until_complete(task_id, headers)
        try:
            image_response = requests.get(image_url, timeout=60)
            image_response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to download the generated image from Freepik: {exc}") from exc

        if not image_response.content:
            raise RuntimeError("Freepik image download returned no data.")
        return image_response.content

    def _poll_until_complete(self, task_id: str, headers: dict) -> str:
        status_url = f"{self.submit_url}/{task_id}"
        deadline = time.monotonic() + self.poll_timeout

        while True:
            try:
                status_response = requests.get(status_url, headers=headers, timeout=30)
                status_response.raise_for_status()
            except requests.RequestException as exc:
                raise RuntimeError(f"Freepik status check failed: {exc}") from exc

            data = (status_response.json() or {}).get("data", {})
            status = data.get("status")

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

    def generate_image(self, prompt: str, negative_prompt: str = "") -> bytes:
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
        return base64.b64decode(images[0])


def get_image_gen_service():
    """Factory for the configured image-rendering backend. All three
    expose the same `.generate_image(prompt, negative_prompt="") -> bytes`
    interface. Provider chosen entirely by settings.IMAGE_PROVIDER — no
    code change needed to switch. Defaults to Freepik.
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
        poll_timeout=settings.FREEPIK_POLL_TIMEOUT,
    )


def generate_variations(prompt: str, negative_prompt: str = "", count: int = 3) -> List[bytes]:
    """Generate `count` image variations from ONE optimized prompt, in
    parallel. Each call goes to the same provider/model, which — being
    stochastic — produces a distinct image per call; this is what gives
    "exactly THREE image variations" from a single prompt. A failure on
    one variation does not abort the others; if fewer than `count`
    succeed, the caller decides whether that's acceptable.

    Each generate_image() call is a blocking network round-trip (and, for
    polling providers like Freepik, several round-trips) with no shared
    state between variations, so they run concurrently on a thread pool
    instead of one after another. This is the single biggest latency win
    in the pipeline — three ~10-20s calls in series is 30-60s+; in
    parallel it's ~one call's worth of wall-clock time.
    """
    service = get_image_gen_service()
    results: List[Optional[bytes]] = [None] * count
    errors: List[str] = []

    def _one(i: int) -> None:
        results[i] = service.generate_image(prompt, negative_prompt=negative_prompt)

    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = {pool.submit(_one, i): i for i in range(count)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - collect and continue
                logger.error("Image variation %d/%d failed: %s", i + 1, count, exc)
                errors.append(str(exc))

    images = [img for img in results if img is not None]
    if not images:
        raise RuntimeError(f"All {count} image variations failed: {'; '.join(errors)}")
    return images