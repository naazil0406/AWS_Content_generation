"""
S3 Service — uploads user-provided campaign documents to the S3 prefix
that feeds the Campaigns Bedrock Knowledge Base Data Source.

This module only writes to `settings.UPLOAD_PREFIX` inside
`settings.UPLOAD_BUCKET_NAME`. It knows nothing about ingestion or the
Knowledge Base itself — see app/services/ingestion_service.py for that
half of the flow. Keeping the two separate means an S3 failure and a
Bedrock failure are diagnosed independently, and neither service needs
to know how the other one works.
"""

import logging
import time
import uuid
from pathlib import PurePosixPath
from typing import BinaryIO, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import settings

logger = logging.getLogger(__name__)

_s3_client = None


class S3UploadError(RuntimeError):
    """Raised when a file cannot be uploaded to S3."""


def _client():
    global _s3_client
    if _s3_client is None:
        # No explicit credentials — boto3's default chain handles this via
        # the Lambda execution role, same as every other AWS client in
        # this app (see knowledge_base_service.py / image_storage_service.py).
        _s3_client = boto3.client("s3", region_name=settings.KNOWLEDGE_BASE_REGION)
    return _s3_client


def _normalized_prefix() -> str:
    prefix = settings.UPLOAD_PREFIX or "campaigns/"
    return prefix if prefix.endswith("/") else prefix + "/"


def build_object_key(filename: str) -> str:
    """Builds the S3 key for an uploaded file under UPLOAD_PREFIX.

    Prefixes a timestamp + short random suffix so two uploads of the
    same filename (including two files in the same batch) never
    silently overwrite each other.
    """
    safe_name = PurePosixPath(filename).name or "upload"  # strip any path components a client might send
    unique = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    return f"{_normalized_prefix()}{unique}_{safe_name}"


def upload_file(file_obj: BinaryIO, filename: str, content_type: Optional[str] = None) -> str:
    """Uploads one file to the campaigns S3 prefix and returns its S3 key
    (e.g. "campaigns/1733400000_a1b2c3d4_safety-brief.pdf").

    Raises S3UploadError if UPLOAD_BUCKET_NAME isn't configured or the
    upload fails (including credential errors).
    """
    if not settings.UPLOAD_BUCKET_NAME:
        raise S3UploadError(
            "UPLOAD_BUCKET_NAME is not configured. Set it to the S3 bucket "
            "that the Campaigns Data Source's S3 folder lives in."
        )

    key = build_object_key(filename)
    extra_args = {"ContentType": content_type} if content_type else None

    logger.info("Uploading campaign document to s3://%s/%s", settings.UPLOAD_BUCKET_NAME, key)
    try:
        _client().upload_fileobj(file_obj, settings.UPLOAD_BUCKET_NAME, key, ExtraArgs=extra_args)
    except (ClientError, BotoCoreError) as exc:
        logger.error("Failed to upload '%s' to S3: %s", filename, exc)
        raise S3UploadError(f"Failed to upload '{filename}' to S3: {exc}") from exc

    logger.info("Upload complete: s3://%s/%s", settings.UPLOAD_BUCKET_NAME, key)
    return key