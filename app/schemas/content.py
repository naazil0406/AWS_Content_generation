"""Pydantic request/response models for the AI Content Generation Service."""

from typing import List, Optional

from pydantic import BaseModel, Field

# The exact set of content types the Content Generation Engine supports.
# See prompts/content_generation_system.txt for the full definition of
# each type. Kept here (not just documented) so the API layer can
# validate/normalize independent of the engine.
CONTENT_TYPES = [
    "Recall Card",
    "Infographic",
    "Scenario",
    "Spot the Mistake",
    "Question",
    "Safety Tip",
    "Fun Fact",
]

# --- Per-request image generation settings ---------------------------------
# These used to be fixed, deployment-wide values (IMAGE_WIDTH/IMAGE_HEIGHT/
# FREEPIK_RESOLUTION template.yaml parameters — see template.yaml and
# app/config.py). They're now per-request choices the frontend sends on
# GenerateContentRequest; config.py's settings only supply the SAFE DEFAULT
# used when a request omits one, never the only value everyone gets. See
# app/services/image_generation_service.py for where each is actually
# applied to the Freepik/Nova Canvas/Pollinations call.

# All 11 ratios requested for the UI. Every one of these has a direct
# width:height meaning; how each maps to the specific string Freepik's API
# expects is handled in image_generation_service.py's _ASPECT_RATIO_MAP —
# see the confidence-level comments there for which of these are verified
# against Freepik's own docs vs. best-effort (this sandbox has no live
# network access to api.freepik.com to confirm the less common ones).
IMAGE_ASPECT_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "2:3"]

# Exactly the two tiers requested for the UI, with an explicit,
# authoritative pixel mapping — this is the single source of truth for
# "2K -> 2048, 4K -> 3840" referenced everywhere else
# (image_generation_service.py's dimensions_for_ratio() for width/height
# providers, and FreepikImageService's own resolution string for the
# Mystic/Magnific integration). 1K is deliberately NOT offered — only
# 2K and 4K, per explicit product requirement.
# Always presented as both options regardless of which provider/model is
# currently configured — GET /image-settings' resolution_note and
# resolution_applies fields (see ImageSettingsResponse below) tell the
# frontend/caller whether the CURRENT config can actually honor it, but
# the control itself is never hidden: a request naming an unsupported-
# by-the-current-model resolution is still valid, just handled
# gracefully (see app/routes/content.py's
# _resolution_support_for_current_config() — the resolution is dropped
# for that one call with a warning surfaced in the response, never a
# hard error, and never an invalid parameter sent to Freepik).
IMAGE_RESOLUTIONS = ["2K", "4K"]
IMAGE_RESOLUTION_PIXELS = {"2K": 2048, "4K": 3840}
DEFAULT_IMAGE_RESOLUTION = "2K"

# How many images a single request may ask for — user-selectable, capped
# at 3 because the Image Prompt Agent always produces exactly 3 distinct,
# content-anchored prompts (unchanged — see
# prompts/image_prompt_system.txt): asking for fewer renders a subset of
# those 3; there is no "more than 3" option since there's nothing beyond
# the Agent's 3 distinct prompts to render without reusing one (see
# image_generation_service.resolve_prompts_for_count() — still used here
# as the count<3 truncation path, its count>3 cycling path is simply
# never reached now that IMAGE_COUNT_MAX is 3).
IMAGE_COUNT_MIN = 1
IMAGE_COUNT_MAX = 3
DEFAULT_IMAGE_COUNT = 3


class GenerateContentRequest(BaseModel):
    """Incoming request: a user prompt describing what to generate."""

    prompt: str = Field(..., min_length=1, description="The user's content request / topic.")
    content_type: str = Field(
        default="Recall Card",
        description=f"One of: {', '.join(CONTENT_TYPES)}",
    )
    industry: Optional[str] = Field(
        default=None,
        description="An explicit industry/professional context for this request (e.g. "
        "'Manufacturing', 'Healthcare', 'Construction'), if the caller has one. Preserved "
        "as-is and passed through to Content Generation and the Image Prompt Agent as the "
        "authoritative setting/professional context for the story and all three images. "
        "If omitted, the engine randomly selects exactly ONE industry from the list "
        "maintained in prompts/image_prompt_system.txt and uses that same randomly-chosen "
        "industry — never a hardcoded default — consistently across the content and all "
        "three images for this request. See the response's content.industry field for "
        "which industry (given or randomly chosen) was actually used.",
    )
    generate_images: bool = Field(
        default=True,
        description="True (the frontend's 'Generate + Image' action) runs the full pipeline: "
        "content generation, then the Image Prompt Agent, then Freepik + S3 for exactly three "
        "images. False (the frontend's 'Generate' action) stops after content generation — the "
        "Image Prompt Agent, Freepik, and S3 image upload are never invoked.",
    )
    image_aspect_ratio: Optional[str] = Field(
        default=None,
        description=f"One of: {', '.join(IMAGE_ASPECT_RATIOS)}. Per-request aspect ratio for "
        "every generated image in this request. Omit to use the server's safe default "
        "(app.config.settings.IMAGE_WIDTH/IMAGE_HEIGHT, historically 1:1). Ignored entirely "
        "when generate_images is False.",
    )
    image_count: Optional[int] = Field(
        default=None,
        ge=IMAGE_COUNT_MIN,
        le=IMAGE_COUNT_MAX,
        description=f"How many images to generate for this request, {IMAGE_COUNT_MIN}-"
        f"{IMAGE_COUNT_MAX}. Omit to use the server's safe default ({DEFAULT_IMAGE_COUNT}). "
        "The Image Prompt Agent still always produces exactly 3 distinct prompts regardless "
        "of this value — asking for fewer renders a subset of those 3 (see "
        "image_generation_service.resolve_prompts_for_count()). Ignored entirely when "
        "generate_images is False.",
    )
    image_resolution: Optional[str] = Field(
        default=None,
        description=f"One of: {', '.join(IMAGE_RESOLUTIONS)} — mapped to exact pixel targets "
        f"{IMAGE_RESOLUTION_PIXELS} (see app.schemas.content.IMAGE_RESOLUTION_PIXELS). Applied "
        "as Freepik Mystic's own resolution parameter when that's the configured model, or as "
        "the target pixel dimensions (combined with image_aspect_ratio) for providers that take "
        "raw width/height (aws, pollinations). Freepik's Flux/Seedream endpoints have no "
        "resolution or width/height parameter at all — see GET /image-settings' "
        "resolution_applies field for whether the currently configured provider/model can "
        "actually honor this. A resolution the current config can't honor is never an error: "
        "it's dropped for that call and a note is added to the response's warnings field (see "
        "GenerateContentResponse.warnings). Omit to use the server's safe default "
        f"({DEFAULT_IMAGE_RESOLUTION}).",
    )
    # Optional extra sources, all treated as fact sources (never wording)
    # by the Content Generation Agent — see the system prompt.
    monthly_topic_content: Optional[str] = None
    common_data: Optional[str] = None
    web_results: Optional[str] = None
    avoid_repeating: Optional[List[str]] = Field(
        default=None,
        description="Previously-generated content for this same (content_type, prompt) "
        "pair, if the caller keeps its own history. The model is told not to repeat "
        "or lightly reword any of these.",
    )


class ImageSettingsResponse(BaseModel):
    """GET /image-settings response — lets the frontend build its Image
    Size / Image Count / Pixel-Resolution controls from the server's
    actual, currently-configured provider/model instead of hardcoding
    assumptions that could drift out of sync with a deploy-time model
    change (see app.config.settings.IMAGE_PROVIDER / FREEPIK_MODEL)."""

    provider: str = Field(..., description="Currently configured IMAGE_PROVIDER.")
    model: str = Field(
        default="",
        description="Currently configured FREEPIK_MODEL ('' = Mystic default). Empty for "
        "non-Freepik providers.",
    )
    aspect_ratios: List[str] = Field(default_factory=lambda: list(IMAGE_ASPECT_RATIOS))
    resolutions: List[str] = Field(
        default_factory=lambda: list(IMAGE_RESOLUTIONS),
        description="Always all three tiers (1K/2K/4K) — the Resolution control is never "
        "hidden based on provider/model, unlike earlier drafts of this endpoint. Whether the "
        "CURRENT provider/model can actually honor a given selection is reported separately "
        "in resolution_applies/resolution_note below; an unsupported selection is still a "
        "valid request (see GenerateContentRequest.image_resolution) — it's just gracefully "
        "dropped for that call rather than sent to Freepik as an invalid parameter.",
    )
    resolution_pixels: dict = Field(
        default_factory=lambda: dict(IMAGE_RESOLUTION_PIXELS),
        description="The exact pixel mapping for each resolution tier (1K=1024, 2K=2048, "
        "4K=3840) — see app.schemas.content.IMAGE_RESOLUTION_PIXELS.",
    )
    resolution_applies: bool = Field(
        ...,
        description="True when the currently configured provider/model can actually use a "
        "selected resolution in some form (Freepik Mystic's native resolution parameter, or "
        "target pixel dimensions for aws/pollinations). False for Freepik's Flux/Seedream "
        "endpoints, which have no resolution or width/height parameter at all — see "
        "resolution_note for a human-readable explanation. Purely informational; the frontend "
        "may still let the user pick a resolution either way (a request naming one is never "
        "rejected), but may want to show resolution_note as a hint.",
    )
    resolution_note: str = Field(
        default="",
        description="Human-readable explanation of resolution_applies above, e.g. \"The "
        "configured model (hyperflux) has no resolution parameter — selections are ignored.\" "
        "Empty when resolution_applies is True.",
    )
    default_aspect_ratio: str = "1:1"
    default_resolution: str = DEFAULT_IMAGE_RESOLUTION
    default_count: int = DEFAULT_IMAGE_COUNT
    min_count: int = IMAGE_COUNT_MIN
    max_count: int = IMAGE_COUNT_MAX


class GeneratedContent(BaseModel):
    title: str = ""
    summary: str = ""
    content: str
    industry: str = Field(
        default="",
        description="The industry actually used for this request — either the caller's "
        "explicit GenerateContentRequest.industry, or, when that was omitted, the industry "
        "the engine randomly selected. The same value was used for this content and (when "
        "generate_images was True) all three images.",
    )
    hashtags: List[str] = []
    cta: str = ""
    image_prompt: str = Field(
        default="",
        description="Deprecated — kept for backward compatibility, equal to image_prompts[0]. "
        "Use image_prompts for the full set of (up to) three distinct prompts, one per "
        "generated image.",
    )
    image_prompts: List[str] = Field(
        default_factory=list,
        description="One distinct image prompt per generated image variation — see "
        "prompts/image_prompt_system.txt's OUTPUT FORMAT for how the three are meant to vary.",
    )
    content_anchors: List[str] = Field(
        default_factory=list,
        description="Traceability data, one entry per image_prompts entry at the same index: "
        "the exact phrase or clause from `content` that image represents. Produced by the "
        "Image Prompt Agent and checked/corrected by a second validation pass — see "
        "prompts/image_prompt_validator_system.txt — before the prompt is ever sent to the "
        "image renderer. Every image should be traceable back to a specific part of `content` "
        "via this field, not merely related to the industry or content type.",
    )


class GenerateContentResponse(BaseModel):
    content: GeneratedContent
    images: List[str] = Field(
        default_factory=list,
        description="Presigned S3 URLs for the generated image variations — up to "
        "request.image_count (1-3, default 3, see GenerateContentRequest.image_count) when "
        "generate_images was True, fewer if some renders failed, empty when "
        "generate_images=False (content-only, the 'Generate' action).",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal notices about this request, e.g. \"Selected resolution '4K' "
        "is not supported by the currently configured image model (hyperflux) and was "
        "ignored.\" Never blocks the request — see "
        "app.routes.content._resolution_support_for_current_config().",
    )


class RegenerateImageRequest(BaseModel):
    """Regenerate ONE image from an already-produced image prompt — used
    by the frontend's per-image recycle/regenerate button. Does not
    re-run content or image-prompt generation; the caller already has
    image_prompt/negative_prompt from a prior /generate or
    /generate/stream response and just wants a fresh render of it."""

    image_prompt: str = Field(..., min_length=1)
    negative_prompt: Optional[str] = None
    image_aspect_ratio: Optional[str] = Field(
        default=None,
        description=f"One of: {', '.join(IMAGE_ASPECT_RATIOS)}. Should normally match "
        "whatever the original GenerateContentRequest used, so the regenerated image stays "
        "visually consistent with the rest of the set. Omit to use the server's safe default.",
    )
    image_resolution: Optional[str] = Field(
        default=None,
        description=f"One of: {', '.join(IMAGE_RESOLUTIONS)}. Same rationale as "
        "image_aspect_ratio above — see GenerateContentRequest.image_resolution for the full "
        "pixel-mapping and graceful-degradation behavior.",
    )


class RegenerateImageResponse(BaseModel):
    url: str = Field(..., description="Presigned S3 URL for the newly regenerated image.")


class ErrorResponse(BaseModel):
    detail: str