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
        description="Presigned S3 URLs for the generated image variations — exactly three "
        "when the request had generate_images=True, empty when generate_images=False "
        "(content-only, the 'Generate' action).",
    )


class RegenerateImageRequest(BaseModel):
    """Regenerate ONE image from an already-produced image prompt — used
    by the frontend's per-image recycle/regenerate button. Does not
    re-run content or image-prompt generation; the caller already has
    image_prompt/negative_prompt from a prior /generate or
    /generate/stream response and just wants a fresh render of it."""

    image_prompt: str = Field(..., min_length=1)
    negative_prompt: Optional[str] = None


class RegenerateImageResponse(BaseModel):
    url: str = Field(..., description="Presigned S3 URL for the newly regenerated image.")


class ErrorResponse(BaseModel):
    detail: str