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
    # Passed into the boto3/botocore Config() for the bedrock-agent-runtime
    # client in knowledge_base_service.py — same pattern as
    # BEDROCK_CONNECT_TIMEOUT_SECONDS/BEDROCK_READ_TIMEOUT_SECONDS below.
    # connect_timeout is just the TCP handshake, NOT the Aurora
    # auto-pause resume wait — that's already handled separately by
    # KNOWLEDGE_BASE_RESUME_WAIT_SECONDS + the retry loop above, so this
    # can stay short.
    KNOWLEDGE_BASE_CONNECT_TIMEOUT_SECONDS: float = float(os.getenv("KNOWLEDGE_BASE_CONNECT_TIMEOUT_SECONDS", "10.0"))
    KNOWLEDGE_BASE_READ_TIMEOUT_SECONDS: float = float(os.getenv("KNOWLEDGE_BASE_READ_TIMEOUT_SECONDS", "30.0"))

    # --- LLM provider: AWS Bedrock (only supported provider) ---
    LLM_PROVIDER: str = "bedrock"
    BEDROCK_REGION: str = os.getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "us-east-1"))
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    # Passed into the boto3/botocore Config() for the Bedrock client in
    # llm_service.py's BedrockLLMService.__init__. connect_timeout caps
    # how long to wait to establish the TCP connection; read_timeout caps
    # how long to wait for Bedrock's response once the request is sent
    # (generation can legitimately take a while, hence the higher default);
    # max_attempts is botocore's own internal retry count for this client,
    # separate from this project's own HTTP_MAX_RETRIES/HTTP_BACKOFF_SECONDS
    # retry loop further down.
    BEDROCK_CONNECT_TIMEOUT_SECONDS: float = float(os.getenv("BEDROCK_CONNECT_TIMEOUT_SECONDS", "10.0"))
    BEDROCK_READ_TIMEOUT_SECONDS: float = float(os.getenv("BEDROCK_READ_TIMEOUT_SECONDS", "60.0"))
    BEDROCK_MAX_ATTEMPTS: int = int(os.getenv("BEDROCK_MAX_ATTEMPTS", "3"))

    # --- Content Generation Agent (structured learner-facing content) ---
    CONTENT_MODEL: str = os.getenv("CONTENT_MODEL", "amazon.nova-micro-v1:0")
    CONTENT_MAX_TOKENS: int = int(os.getenv("CONTENT_MAX_TOKENS", "400"))
    CONTENT_TEMPERATURE: float = float(os.getenv("CONTENT_TEMPERATURE", "0.7"))
    # Gates whether content_generation_engine.py uses llm_service.py's
    # get_combined_llm()/generate_combined_package() path (one LLM call
    # producing both the content AND the draft image prompts together)
    # instead of the separate Content Generation Agent + Image Prompt
    # Generation Agent calls described in this project's README. Defaults
    # to False to preserve that documented two-call (plus validator)
    # pipeline unless explicitly opted into — flip to true only after
    # confirming generate_combined_package()'s output still gets validated
    # the same way the separate path's does.
    USE_COMBINED_GENERATION_CALL: bool = os.getenv("USE_COMBINED_GENERATION_CALL", "false").strip().lower() == "true"

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
    # SAFE DEFAULT only, not a fixed setting: GenerateContentRequest.
    # image_count (see app/schemas/content.py) overrides this per-request
    # via resolve_prompts_for_count() in image_generation_service.py.
    # This value only applies when a request omits image_count.
    IMAGE_VARIATIONS_COUNT: int = int(os.getenv("IMAGE_VARIATIONS_COUNT", "3"))
    # SAFE DEFAULT only, not a fixed setting: GenerateContentRequest.
    # image_aspect_ratio overrides this per-request via
    # get_image_gen_service()'s dimensions_for_ratio() (Pollinations/Nova
    # Canvas) or FreepikImageService's aspect_ratio_key (Freepik). No
    # longer wired into template.yaml as a deploy-time parameter — see
    # that file's Parameters block for the full history/rationale. This
    # value only applies when a request omits image_aspect_ratio.
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
    # SAFE DEFAULT only, not a fixed setting: GenerateContentRequest.
    # image_resolution overrides this per-request (Mystic endpoint only —
    # see get_image_gen_service()/FreepikImageService._build_body). No
    # longer wired into template.yaml as a deploy-time parameter. This
    # value only applies when a request omits image_resolution.
    FREEPIK_RESOLUTION: str = os.getenv("FREEPIK_RESOLUTION", "2k")
    FREEPIK_FILTER_NSFW: bool = os.getenv("FREEPIK_FILTER_NSFW", "true").strip().lower() == "true"
    FREEPIK_POLL_INTERVAL: float = float(os.getenv("FREEPIK_POLL_INTERVAL", "3.0"))
    # 170s (was 120s): 120s sat INSIDE Mystic's documented ~90-130s
    # real-world latency range, not safely above it — meaning a
    # meaningful share of legitimate, successful renders were being cut
    # off by this timeout and reported as failures. 170s leaves real
    # margin over the 90-130s range while still fitting comfortably
    # under ContentGenerationFunction's Timeout (see template.yaml,
    # which now also exposes both of these as deploy-time Parameters —
    # FreepikPollTimeoutSeconds/FreepikPollIntervalSeconds — so this
    # default is a starting point, not the only tunable value).
    FREEPIK_POLL_TIMEOUT: float = float(os.getenv("FREEPIK_POLL_TIMEOUT", "170.0"))
    # Freepik's gateway appears to reject some fraction of truly
    # simultaneous submissions from the same API key/plan — observed in
    # production as 2 of 3 parallel image requests failing at the same
    # millisecond with DIFFERENT error codes (401 on one, 422 on
    # another) for otherwise-identical requests, which is the signature
    # of a collision at their gateway rather than a deterministic config
    # problem (a real bad key/payload would fail all three identically).
    # Only the initial POST to submit_url is serialized through this —
    # the slow part (polling + download, ~90-130s for Mystic) still runs
    # fully in parallel across the 3 images, so this costs at most a
    # couple of quick sequential POSTs' worth of latency, not 3x the
    # total time. Default 1 = fully serialized submissions; raise if
    # Freepik's actual plan limit turns out to allow more than one
    # in-flight submission.
    FREEPIK_MAX_CONCURRENT_SUBMISSIONS: int = int(os.getenv("FREEPIK_MAX_CONCURRENT_SUBMISSIONS", "1"))
    # Extra attempts (beyond the first) for the initial submission only,
    # if it fails for ANY reason — including the 401/422 collision above,
    # but also covers ordinary transient network blips. A genuinely bad
    # key/payload will still fail every attempt and surface the same
    # error, just after a short, bounded delay instead of immediately.
    FREEPIK_SUBMIT_MAX_RETRIES: int = int(os.getenv("FREEPIK_SUBMIT_MAX_RETRIES", "2"))
    FREEPIK_SUBMIT_RETRY_BACKOFF_SECONDS: float = float(os.getenv("FREEPIK_SUBMIT_RETRY_BACKOFF_SECONDS", "0.75"))
    # Hard-enforced floor for scene-mode image prompts (see
    # app/services/content_generation_engine.py's call to
    # expand_short_image_prompts() and prompts/image_prompt_system.txt's
    # LINE COUNT AND FORMAT section). 15-20 lines is the PREFERRED/normal
    # target when the content supports that much detail — this setting
    # is the minimum a prompt must reach before it's allowed through to
    # Freepik at all, not the target itself.
    IMAGE_PROMPT_MIN_LINES: int = int(os.getenv("IMAGE_PROMPT_MIN_LINES", "12"))

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

    # --- Logging ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # --- Canva integration (OAuth 2.0 + Authorization Code + PKCE) ---
    # Local dev default matches CANVA_REDIRECT_URI you register in the
    # Canva Developer Portal for this integration. NEVER hardcode
    # CANVA_CLIENT_SECRET anywhere but an environment variable / .env —
    # see app/services/canva_service.py for where it's used (only in
    # server-side token-exchange calls; it is never sent to the
    # frontend).
    CANVA_CLIENT_ID: str = os.getenv("CANVA_CLIENT_ID", "")
    CANVA_CLIENT_SECRET: str = os.getenv("CANVA_CLIENT_SECRET", "")
    CANVA_REDIRECT_URI: str = os.getenv("CANVA_REDIRECT_URI", "http://localhost:8000/api/canva/callback")
    # Space-separated, per Canva's OAuth scope format. asset:write/read
    # is needed for the asset-upload flow (section 5); design:content:write
    # and design:meta:read are needed to create a design and get back its
    # edit URL. Verify these exact scope names against your Canva
    # integration's configured scopes in the Developer Portal — Canva
    # only grants scopes that are BOTH requested here AND enabled for
    # your integration there.
    CANVA_SCOPES: str = os.getenv(
        "CANVA_SCOPES",
        "asset:read asset:write design:content:read design:content:write design:meta:read",
    )

    # --- Local development storage (Canva tokens + image ownership) ---
    # A single small JSON-file-backed key/value store used ONLY for
    # local development — see app/services/local_dev_store.py for the
    # interface and exactly what to swap in for production (DynamoDB).
    # File-backed (not just an in-memory dict) specifically so state
    # survives `uvicorn --reload` restarts during local testing — a
    # dropped Canva connection every time you edit a .py file would
    # make local OAuth testing painful for no reason.
    LOCAL_DEV_STORE_PATH: str = os.getenv("LOCAL_DEV_STORE_PATH", ".local_dev_store.json")


settings = Settings()
setup_logging(settings.LOG_LEVEL)