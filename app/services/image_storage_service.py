"""
Image Storage Service — uploads generated images to S3 and hands back
presigned URLs instead of embedding raw image bytes in the API response.

Why this exists: API Gateway (HTTP API) caps integration response
payloads at 6 MB, and three base64-encoded PNGs plus content text can
exceed that easily (base64 adds ~33% overhead on top of already-large
PNGs). Returning short-lived S3 URLs keeps the JSON response to a few KB
regardless of image size, and lets the browser fetch images directly
from S3 rather than routing them back through Lambda/API Gateway.
"""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import settings

logger = logging.getLogger(__name__)

_s3_client = None


def _client():
    global _s3_client
    if _s3_client is None:
        # No explicit credentials — boto3's default chain handles this
        # correctly via the Lambda execution role.
        _s3_client = boto3.client("s3", region_name=settings.S3_REGION)
    return _s3_client


def image_object_key(generation_id: str, index: int) -> str:
    """Single source of truth for the S3 key an image lives at. Used by
    every path that reads or writes a generated image's S3 object: the
    synchronous upload helpers below, ImageWorkerFunction
    (app/worker.py), and the main Lambda's pre-signed-URL-before-upload
    path (app/routes/content.py's presigned_url_for() calls, used by the
    SQS image-generation flow — see that module for why a URL can be
    generated before the object exists)."""
    return f"generated/{generation_id}/{index}.png"


def presigned_url_for(generation_id: str, index: int) -> str:
    """A presigned GET URL for image_object_key(generation_id, index),
    without requiring the object to exist yet. S3 presigned URLs are a
    signed HTTP request, not a check against the bucket — generating one
    is a local HMAC computation with no network call, and it will 404
    with NoSuchKey until something actually PUTs the object at that key
    (it's the same presigned URL either way, generated before or after
    the object exists).

    Used by the SQS image-generation flow (see app/routes/content.py):
    the main Lambda generates this URL immediately after deciding this
    image's deterministic key, hands it back in the API response, and
    ONLY THEN queues the actual generation job — so the client gets a
    real URL synchronously even though the image itself is produced
    later, out of process, by ImageWorkerFunction (app/worker.py's
    put_generated_image() call, using this exact same key function).
    """
    if not settings.GENERATED_IMAGES_BUCKET:
        raise RuntimeError(
            "GENERATED_IMAGES_BUCKET is not configured. Set it to the S3 "
            "bucket name created by template.yaml for storing generated images."
        )
    return _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.GENERATED_IMAGES_BUCKET, "Key": image_object_key(generation_id, index)},
        ExpiresIn=settings.S3_PRESIGNED_URL_EXPIRY,
    )


def put_generated_image(image_bytes: bytes, generation_id: str, index: int) -> str:
    """Upload one image to its deterministic key (image_object_key
    above) and return that key (not a URL — the caller doesn't need one;
    see put_generated_image()'s docstring for why a URL was already
    generated and returned to the client before this function ever
    runs). This is the ONLY function ImageWorkerFunction (app/worker.py)
    calls to actually write image bytes to S3 — it reuses this exact
    helper, not a separate upload path, so there is a single source of
    truth for how a generated image reaches S3 regardless of whether the
    request that produced it was synchronous or came off the SQS queue.
    """
    if not settings.GENERATED_IMAGES_BUCKET:
        raise RuntimeError(
            "GENERATED_IMAGES_BUCKET is not configured. Set it to the S3 "
            "bucket name created by template.yaml for storing generated images."
        )
    key = image_object_key(generation_id, index)
    try:
        _client().put_object(
            Bucket=settings.GENERATED_IMAGES_BUCKET,
            Key=key,
            Body=image_bytes,
            ContentType="image/png",
        )
    except (ClientError, BotoCoreError) as exc:
        logger.error("Failed to upload image %d for %s to S3: %s", index, generation_id, exc)
        raise RuntimeError(f"Failed to upload generated image to S3: {exc}") from exc
    return key


def _put_and_presign(client, image_bytes: bytes, generation_id: str, index: int) -> str:
    # `client` param kept for signature compatibility with existing
    # callers below (upload_images_to_s3's ThreadPoolExecutor loop) —
    # not actually needed anymore since put_generated_image()/
    # presigned_url_for() each resolve the same memoized singleton via
    # _client() internally, so composing them here doesn't create any
    # extra client instances or duplicate the key-naming logic.
    put_generated_image(image_bytes, generation_id, index)
    return presigned_url_for(generation_id, index)


def upload_images_to_s3(images: List[bytes], generation_id: str = None) -> List[str]:
    """Upload each image to S3 under a unique key and return a presigned
    GET URL for each, valid for settings.S3_PRESIGNED_URL_EXPIRY seconds.

    Raises RuntimeError if the bucket isn't configured or an upload fails
    — callers should treat this the same as an image-generation failure.
    """
    if not settings.GENERATED_IMAGES_BUCKET:
        raise RuntimeError(
            "GENERATED_IMAGES_BUCKET is not configured. Set it to the S3 "
            "bucket name created by template.yaml for storing generated images."
        )

    generation_id = generation_id or uuid.uuid4().hex
    client = _client()
    results: List[Optional[str]] = [None] * len(images)

    def _one(i: int, image_bytes: bytes) -> None:
        results[i] = _put_and_presign(client, image_bytes, generation_id, i)

    # Uploads are independent per-image, so run them concurrently rather
    # than one after another — presigned-URL generation is a local HMAC
    # computation (no extra network round trip), so the only real
    # network cost per image is the put_object call itself. Any failure
    # surfaces as RuntimeError — put_generated_image() (called via
    # _put_and_presign above) already logs and wraps the underlying
    # ClientError/BotoCoreError, so future.result() re-raising it here
    # needs no extra handling.
    with ThreadPoolExecutor(max_workers=len(images)) as pool:
        futures = {pool.submit(_one, i, img): i for i, img in enumerate(images)}
        for future, i in futures.items():
            future.result()

    return results


def upload_single_image_to_s3(image_bytes: bytes, generation_id: str, index: int) -> str:
    """Upload ONE image and return its presigned URL immediately, instead
    of waiting for a whole batch (see upload_images_to_s3). Used by the
    streaming endpoint (POST /api/generate/stream) so each image's URL
    can be pushed to the client the moment that image finishes, rather
    than waiting for all three. `index` must be unique per generation_id
    (e.g. the image's position 0/1/2) — it's the S3 key suffix, so reusing
    an index for two different images silently overwrites the first.

    Any failure surfaces as RuntimeError — see put_generated_image()'s
    docstring, called internally via _put_and_presign.
    """
    return _put_and_presign(_client(), image_bytes, generation_id, index)