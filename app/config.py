"""
Configuration for the AI Content Generation Service.

Loads environment variables (via python-dotenv locally; Lambda supplies
them natively) and exposes a single immutable `settings` object. This is
the ONLY place environment variables are read — every other module takes
its configuration through `settings`, never `os.getenv` directly.
"""

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logging. CloudWatch captures whatever the Lambda
    runtime's root logger emits, so this is all that's needed for
    production logging — no extra handlers."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@dataclass(frozen=True)
class Settings:
    # --- Knowledge Base (Amazon Bedrock Knowledge Base) ---
    # Bedrock Knowledge Bases are addressed by ID, not a URL. Retrieval
    # happens through the bedrock-agent-runtime SDK client, not raw HTTP.
    KNOWLEDGE_BASE_ID: str = os.getenv("KNOWLEDGE_BASE_ID", "")
    KNOWLEDGE_BASE_REGION: str = os.getenv(
        "KNOWLEDGE_BASE_REGION", os.getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "us-east-1"))
    )
    KNOWLEDGE_BASE_TOP_K: int = int(os.getenv("KNOWLEDGE_BASE_TOP_K", "10"))
    # Retry settings specific to KB retrieval. Kept separate from the
    # generic HTTP_MAX_RETRIES/HTTP_BACKOFF_SECONDS below because Aurora
    # Serverless auto-pause resumes reliably take 15-30s — far longer
    # than a short generic backoff — while other KB failures should still
    # fail comparatively fast.
    KNOWLEDGE_BASE_MAX_RETRIES: int = int(os.getenv("KNOWLEDGE_BASE_MAX_RETRIES", "5"))
    KNOWLEDGE_BASE_RETRY_BACKOFF_SECONDS: float = float(os.getenv("KNOWLEDGE_BASE_RETRY_BACKOFF_SECONDS", "1.5"))
    KNOWLEDGE_BASE_RESUME_WAIT_SECONDS: float = float(os.getenv("KNOWLEDGE_BASE_RESUME_WAIT_SECONDS", "20.0"))

    # --- LLM provider: AWS Bedrock (only supported provider) ---
    LLM_PROVIDER: str = "bedrock"
    BEDROCK_REGION: str = os.getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "us-east-1"))
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()

    # --- Content Generation Agent (structured learner-facing content) ---
    CONTENT_MODEL: str = os.getenv("CONTENT_MODEL", "amazon.nova-micro-v1:0")
    CONTENT_MAX_TOKENS: int = int(os.getenv("CONTENT_MAX_TOKENS", "400"))
    CONTENT_TEMPERATURE: float = float(os.getenv("CONTENT_TEMPERATURE", "0.7"))

    # --- Image Prompt Generation Agent (turns content into an image prompt) ---
    IMAGE_PROMPT_MODEL: str = os.getenv("IMAGE_PROMPT_MODEL", "amazon.nova-lite-v1:0")
    # Must fit: 3 image_prompts (target 120-180 words each per
    # prompts/image_prompt_system.txt's Elaboration rule) + negative_prompt
    # + alt_text + tags + summary, all as one JSON object. 512 was sized
    # for shorter single-sentence-ish prompts; it's too tight for the
    # elaborated/content-specific version and causes the model to
    # compress under pressure — typically by dropping the content-specific
    # facts/items in favor of shorter generic filler, which is what makes
    # generated images drift from the actual Generated Content. ~1800
    # gives headroom for 3 x ~180 words (~250 tokens each) plus metadata,
    # with margin.
    IMAGE_PROMPT_MAX_TOKENS: int = int(os.getenv("IMAGE_PROMPT_MAX_TOKENS", "1800"))
    IMAGE_PROMPT_TEMPERATURE: float = float(os.getenv("IMAGE_PROMPT_TEMPERATURE", "0.3"))

    # --- Image Generation Service ---
    # "freepik" (Freepik Mystic/Flux, default), "pollinations" (free, no
    # key), or "aws" (Bedrock Nova Canvas). Changing providers is a
    # config-only change — see app/services/image_generation_service.py.
    IMAGE_PROVIDER: str = os.getenv("IMAGE_PROVIDER", "freepik")
    IMAGE_VARIATIONS_COUNT: int = int(os.getenv("IMAGE_VARIATIONS_COUNT", "3"))
    IMAGE_WIDTH: int = int(os.getenv("IMAGE_WIDTH", "1024"))
    IMAGE_HEIGHT: int = int(os.getenv("IMAGE_HEIGHT", "1024"))

    # Separate AWS account/region for the image pipeline, if desired (only
    # used when IMAGE_PROVIDER=aws). Falls back to the main AWS_* creds/
    # region above when left blank, so one shared account still works out
    # of the box.
    AWS_IMAGE_ACCESS_KEY_ID: str = (os.getenv("AWS_IMAGE_ACCESS_KEY_ID", "").strip() or os.getenv("AWS_ACCESS_KEY_ID", "").strip())
    AWS_IMAGE_SECRET_ACCESS_KEY: str = (os.getenv("AWS_IMAGE_SECRET_ACCESS_KEY", "").strip() or os.getenv("AWS_SECRET_ACCESS_KEY", "").strip())
    AWS_IMAGE_REGION: str = os.getenv("AWS_IMAGE_REGION", "").strip() or os.getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "us-east-1"))
    BEDROCK_IMAGE_GEN_MODEL: str = os.getenv("BEDROCK_IMAGE_GEN_MODEL", "amazon.nova-canvas-v1:0")
    IMAGE_QUALITY: str = os.getenv("IMAGE_QUALITY", "standard")
    IMAGE_CFG_SCALE: float = float(os.getenv("IMAGE_CFG_SCALE", "8.0"))

    POLLINATIONS_MODEL: str = os.getenv("POLLINATIONS_MODEL", "flux")
    POLLINATIONS_BASE_URL: str = os.getenv("POLLINATIONS_BASE_URL", "https://image.pollinations.ai/prompt")

    FREEPIK_API_KEY: str = os.getenv("FREEPIK_API_KEY", "").strip()
    # Blank -> falls through to Mystic (slowest, ~90-130s/image at 2K).
    # Faster options: "hyperflux" (fastest), "flux-dev", "seedream-v4",
    # "seedream-v4-5", or "seedream-v5-lite" (newest, ~20-40s/image per
    # third-party benchmarks). Must match a key in _FLUX_ENDPOINTS above
    # exactly, or it silently falls back to Mystic.
    FREEPIK_MODEL: str = os.getenv("FREEPIK_MODEL", "")
    FREEPIK_RESOLUTION: str = os.getenv("FREEPIK_RESOLUTION", "2k")
    FREEPIK_FILTER_NSFW: bool = os.getenv("FREEPIK_FILTER_NSFW", "true").strip().lower() == "true"
    FREEPIK_POLL_INTERVAL: float = float(os.getenv("FREEPIK_POLL_INTERVAL", "3.0"))
    FREEPIK_POLL_TIMEOUT: float = float(os.getenv("FREEPIK_POLL_TIMEOUT", "120.0"))

    # --- Generated Image Storage (S3) ---
    # Images are uploaded here and served back to the client as presigned
    # URLs instead of being embedded as base64 in the API response — see
    # app/services/image_storage_service.py for why.
    GENERATED_IMAGES_BUCKET: str = os.getenv("GENERATED_IMAGES_BUCKET", "")
    S3_REGION: str = os.getenv("S3_REGION", os.getenv("AWS_REGION", "us-east-1"))
    S3_PRESIGNED_URL_EXPIRY: int = int(os.getenv("S3_PRESIGNED_URL_EXPIRY", "3600"))

    # --- Streaming endpoint (/api/generate/stream) timing budget ---
    # The whole request — KB retrieval + combined LLM call + N parallel
    # image renders + S3 uploads — is budgeted to finish comfortably
    # under a 60s target (matches typical API Gateway/browser tolerance
    # for a single request). TOTAL_DEADLINE_SECONDS is the ceiling for
    # the whole request; IMAGE_STREAM_MIN_DEADLINE_SECONDS is a floor so
    # a slow KB/LLM stage never leaves images with an unreasonably short
    # window. Whatever's left after KB+LLM is what image generation gets,
    # clamped to this floor.
    TOTAL_DEADLINE_SECONDS: float = float(os.getenv("TOTAL_DEADLINE_SECONDS", "55.0"))
    IMAGE_STREAM_MIN_DEADLINE_SECONDS: float = float(os.getenv("IMAGE_STREAM_MIN_DEADLINE_SECONDS", "15.0"))

    # --- Retries (external API calls: KB, LLM, image provider) ---
    HTTP_MAX_RETRIES: int = int(os.getenv("HTTP_MAX_RETRIES", "3"))
    HTTP_BACKOFF_SECONDS: float = float(os.getenv("HTTP_BACKOFF_SECONDS", "1.5"))

    # --- Campaign Document Upload (new Bedrock KB Data Source) ---
    # Uploaded files land in this S3 prefix, then trigger an ingestion job
    # scoped ONLY to UPLOAD_DATA_SOURCE_ID. The existing data source in
    # the same Knowledge Base (KNOWLEDGE_BASE_ID above) is never touched
    # by this — see app/services/ingestion_service.py. Once ingestion
    # completes, uploaded documents are automatically retrievable through
    # the existing KnowledgeBaseService.retrieve() with zero code changes
    # there, since Bedrock KB retrieval searches across all data sources
    # in a Knowledge Base.
    UPLOAD_BUCKET_NAME: str = os.getenv("UPLOAD_BUCKET_NAME", "")
    UPLOAD_PREFIX: str = os.getenv("UPLOAD_PREFIX", "campaigns/")
    UPLOAD_DATA_SOURCE_ID: str = os.getenv("UPLOAD_DATA_SOURCE_ID", "")

    # --- Logging ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
setup_logging(settings.LOG_LEVEL)