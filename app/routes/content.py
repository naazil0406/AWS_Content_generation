"""API routes for AI Content Generation."""

import logging

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.schemas.content import GenerateContentRequest, GenerateContentResponse
from app.services.content_generation_engine import ContentGenerationEngine, ContentGenerationError
from app.services.image_generation_service import generate_variations
from app.services.knowledge_base_service import KnowledgeBaseError, KnowledgeBaseService
from app.services.llm_service import get_content_llm, get_image_prompt_llm
from app.services.response_builder import build_response

logger = logging.getLogger(__name__)
router = APIRouter()


def _build_engine() -> ContentGenerationEngine:
    """Construct the engine with fresh provider instances per request.
    Providers are cheap to construct (HTTP/boto3 clients only), so no
    caching is needed in a Lambda's short-lived execution context.
    """
    return ContentGenerationEngine(
        knowledge_base=KnowledgeBaseService(),
        content_llm=get_content_llm(),
        image_prompt_llm=get_image_prompt_llm(),
    )


@router.post("/generate", response_model=GenerateContentResponse)
def generate(request: GenerateContentRequest) -> GenerateContentResponse:
    """Generate structured AI content plus three image variations for
    the given prompt and content type.
    """
    engine = _build_engine()

    try:
        result = engine.generate(
            prompt=request.prompt,
            content_type=request.content_type,
            monthly_topic_content=request.monthly_topic_content,
            common_data=request.common_data,
            web_results=request.web_results,
            avoid_repeating=request.avoid_repeating,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except KnowledgeBaseError as exc:
        raise HTTPException(status_code=502, detail=f"Knowledge Base error: {exc}")
    except ContentGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception:
        logger.exception("Unexpected error during content generation.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

    try:
        images = generate_variations(
            prompt=result["image_prompt"],
            negative_prompt=result["negative_prompt"],
            count=settings.IMAGE_VARIATIONS_COUNT,
        )
    except Exception as exc:
        logger.error("Image generation failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Image generation failed: {exc}")

    return build_response(result, images)
