"""Pydantic request/response models for the AI Content Generation Service."""

from typing import List, Optional

from pydantic import BaseModel, Field

# The exact set of content types the Content Generation Engine supports.
# See prompts/content_generation_system.txt for the full definition of
# each type. Kept here (not just documented) so the API layer can
# validate/normalize independent of the engine.
CONTENT_TYPES = [
    "Recall Card",
    "AI Image",
    "Infographic",
    "Flashcard",
    "Scenario",
    "Spot the Mistake Challenge",
    "Daily Quiz",
    "Fun Fact",
    "Reflection Question",
    "Safety / Best Practice Tip",
    "Daily Tip",
]


class GenerateContentRequest(BaseModel):
    """Incoming request: a user prompt describing what to generate."""

    prompt: str = Field(..., min_length=1, description="The user's content request / topic.")
    content_type: str = Field(
        default="Recall Card",
        description=f"One of: {', '.join(CONTENT_TYPES)}",
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
    hashtags: List[str] = []
    cta: str = ""
    image_prompt: str


class GenerateContentResponse(BaseModel):
    content: GeneratedContent
    images: List[str] = Field(
        ..., description="Exactly three presigned S3 URLs for the generated image variations."
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