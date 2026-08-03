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


def _put_and_presign(client, image_bytes: bytes, generation_id: str, index: int) -> str:
    key = f"generated/{generation_id}/{index}.png"
    client.put_object(
        Bucket=settings.GENERATED_IMAGES_BUCKET,
        Key=key,
        Body=image_bytes,
        ContentType="image/png",
    )
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.GENERATED_IMAGES_BUCKET, "Key": key},
        ExpiresIn=settings.S3_PRESIGNED_URL_EXPIRY,
    )


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
    # network cost per image is the put_object call itself.
    with ThreadPoolExecutor(max_workers=len(images)) as pool:
        futures = {pool.submit(_one, i, img): i for i, img in enumerate(images)}
        for future, i in futures.items():
            try:
                future.result()
            except (ClientError, BotoCoreError) as exc:
                logger.error("Failed to upload image %d to S3: %s", i, exc)
                raise RuntimeError(f"Failed to upload generated image to S3: {exc}") from exc

    return results


def upload_single_image_to_s3(image_bytes: bytes, generation_id: str, index: int) -> str:
    """Upload ONE image and return its presigned URL immediately, instead
    of waiting for a whole batch (see upload_images_to_s3). Used by the
    streaming endpoint (POST /api/generate/stream) so each image's URL
    can be pushed to the client the moment that image finishes, rather
    than waiting for all three. `index` must be unique per generation_id
    (e.g. the image's position 0/1/2) — it's the S3 key suffix, so reusing
    an index for two different images silently overwrites the first.
    """
    if not settings.GENERATED_IMAGES_BUCKET:
        raise RuntimeError(
            "GENERATED_IMAGES_BUCKET is not configured. Set it to the S3 "
            "bucket name created by template.yaml for storing generated images."
        )
    try:
        return _put_and_presign(_client(), image_bytes, generation_id, index)
    except (ClientError, BotoCoreError) as exc:
        logger.error("Failed to upload image %d to S3: %s", index, exc)
        raise RuntimeError(f"Failed to upload generated image to S3: {exc}") from exc