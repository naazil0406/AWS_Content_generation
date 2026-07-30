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
    # --- Knowledge Base (external managed dependency) ---
    # The KB already exists and owns its own chunking/embedding/vector
    # store. This service only calls it over HTTP and consumes whatever
    # context chunks it returns.
    KNOWLEDGE_BASE_URL: str = os.getenv("KNOWLEDGE_BASE_URL", "")
    KNOWLEDGE_BASE_API_KEY: str = os.getenv("KNOWLEDGE_BASE_API_KEY", "")
    KNOWLEDGE_BASE_TOP_K: int = int(os.getenv("KNOWLEDGE_BASE_TOP_K", "10"))
    KNOWLEDGE_BASE_TIMEOUT: int = int(os.getenv("KNOWLEDGE_BASE_TIMEOUT", "30"))

    # --- LLM provider switch: "openrouter" or "bedrock" ---
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "bedrock").strip().lower()
    BEDROCK_REGION: str = os.getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "us-east-1"))
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()

    # --- OpenRouter (optional alternative LLM transport) ---
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct")
    OPENROUTER_SITE_URL: str = os.getenv("OPENROUTER_SITE_URL", "")
    OPENROUTER_SITE_NAME: str = os.getenv("OPENROUTER_SITE_NAME", "")

    # --- Content Generation Agent (structured learner-facing content) ---
    CONTENT_MODEL: str = os.getenv("CONTENT_MODEL", "amazon.nova-micro-v1:0")
    CONTENT_MAX_TOKENS: int = int(os.getenv("CONTENT_MAX_TOKENS", "400"))
    CONTENT_TEMPERATURE: float = float(os.getenv("CONTENT_TEMPERATURE", "0.7"))

    # --- Image Prompt Generation Agent (turns content into an image prompt) ---
    IMAGE_PROMPT_MODEL: str = os.getenv("IMAGE_PROMPT_MODEL", "amazon.nova-lite-v1:0")
    IMAGE_PROMPT_MAX_TOKENS: int = int(os.getenv("IMAGE_PROMPT_MAX_TOKENS", "512"))
    IMAGE_PROMPT_TEMPERATURE: float = float(os.getenv("IMAGE_PROMPT_TEMPERATURE", "0.3"))

    # --- Image Generation Service ---
    # "huggingface" (FLUX.1-dev), "pollinations" (free, no key), "freepik"
    # (Freepik Mystic/Flux), or "aws" (Bedrock Nova Canvas). Changing
    # providers is a config-only change — see app/services/image_generation_service.py.
    IMAGE_PROVIDER: str = os.getenv("IMAGE_PROVIDER", "aws")
    IMAGE_VARIATIONS_COUNT: int = int(os.getenv("IMAGE_VARIATIONS_COUNT", "3"))
    IMAGE_WIDTH: int = int(os.getenv("IMAGE_WIDTH", "1024"))
    IMAGE_HEIGHT: int = int(os.getenv("IMAGE_HEIGHT", "1024"))

    # Separate AWS account/region for the image pipeline, if desired. Falls
    # back to the main AWS_* creds/region above when left blank, so one
    # shared account still works out of the box.
    AWS_IMAGE_ACCESS_KEY_ID: str = (os.getenv("AWS_IMAGE_ACCESS_KEY_ID", "").strip() or os.getenv("AWS_ACCESS_KEY_ID", "").strip())
    AWS_IMAGE_SECRET_ACCESS_KEY: str = (os.getenv("AWS_IMAGE_SECRET_ACCESS_KEY", "").strip() or os.getenv("AWS_SECRET_ACCESS_KEY", "").strip())
    AWS_IMAGE_REGION: str = os.getenv("AWS_IMAGE_REGION", "").strip() or os.getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "us-east-1"))
    BEDROCK_IMAGE_GEN_MODEL: str = os.getenv("BEDROCK_IMAGE_GEN_MODEL", "amazon.nova-canvas-v1:0")
    IMAGE_QUALITY: str = os.getenv("IMAGE_QUALITY", "standard")
    IMAGE_CFG_SCALE: float = float(os.getenv("IMAGE_CFG_SCALE", "8.0"))

    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    HF_FLUX_MODEL: str = os.getenv("HF_FLUX_MODEL", "black-forest-labs/FLUX.1-dev")
    HF_INFERENCE_PROVIDER: str = os.getenv("HF_INFERENCE_PROVIDER", "auto")
    FLUX_NUM_INFERENCE_STEPS: int = int(os.getenv("FLUX_NUM_INFERENCE_STEPS", "45"))
    FLUX_GUIDANCE_SCALE: float = float(os.getenv("FLUX_GUIDANCE_SCALE", "7.5"))

    POLLINATIONS_MODEL: str = os.getenv("POLLINATIONS_MODEL", "flux")
    POLLINATIONS_BASE_URL: str = os.getenv("POLLINATIONS_BASE_URL", "https://image.pollinations.ai/prompt")

    FREEPIK_API_KEY: str = os.getenv("FREEPIK_API_KEY", "").strip()
    FREEPIK_MODEL: str = os.getenv("FREEPIK_MODEL", "realism")
    FREEPIK_RESOLUTION: str = os.getenv("FREEPIK_RESOLUTION", "2k")
    FREEPIK_FILTER_NSFW: bool = os.getenv("FREEPIK_FILTER_NSFW", "true").strip().lower() == "true"
    FREEPIK_POLL_INTERVAL: float = float(os.getenv("FREEPIK_POLL_INTERVAL", "3.0"))
    FREEPIK_POLL_TIMEOUT: float = float(os.getenv("FREEPIK_POLL_TIMEOUT", "120.0"))

    # --- Retries (external API calls: KB, LLM, image provider) ---
    HTTP_MAX_RETRIES: int = int(os.getenv("HTTP_MAX_RETRIES", "3"))
    HTTP_BACKOFF_SECONDS: float = float(os.getenv("HTTP_BACKOFF_SECONDS", "1.5"))

    # --- Logging ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
setup_logging(settings.LOG_LEVEL)
