"""API routes for AI Content Generation."""

import json
import logging
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from app.schemas.content import (
    CONTENT_TYPES,
    GenerateContentRequest,
    GenerateContentResponse,
    RegenerateImageRequest,
    RegenerateImageResponse,
)
from app.services.content_generation_engine import ContentGenerationEngine, ContentGenerationError
from app.services.image_generation_service import generate_variations, generate_variations_events
from app.services.image_storage_service import upload_images_to_s3
from app.services.knowledge_base_service import KnowledgeBaseError, KnowledgeBaseService
from app.services.llm_service import get_combined_llm
from app.services.response_builder import build_response

logger = logging.getLogger(__name__)
router = APIRouter()


def _build_engine() -> ContentGenerationEngine:
    """Construct the engine with separate Content LLM and Image Prompt LLM instances."""
    from app.services.llm_service import get_content_llm, get_image_prompt_llm

    return ContentGenerationEngine(
        knowledge_base=KnowledgeBaseService(),
        content_llm=get_content_llm(),
        image_prompt_llm=get_image_prompt_llm(),
    )


@router.get("/content-types")
def list_content_types() -> dict:
    """Content types the frontend's dropdown populates itself from.
    Sourced from app/schemas/content.py's CONTENT_TYPES, which mirrors
    the table in prompts/image_prompt_system.txt.
    """
    return {"content_types": CONTENT_TYPES}


@router.post("/generate", response_model=GenerateContentResponse)
def generate(request: GenerateContentRequest) -> GenerateContentResponse:
    """Generate content for the given prompt/content type/industry, and
    — only when request.generate_images is True (the "Generate + Image"
    action) — also generate three grounded image variations for it.

    When request.generate_images is False (the plain "Generate" action),
    this returns content only: the Image Prompt Agent, Freepik, and S3
    image upload are never invoked.
    """
    engine = _build_engine()

    try:
        result = engine.generate(
            prompt=request.prompt,
            content_type=request.content_type,
            industry=request.industry,
            generate_images=request.generate_images,
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

    if not request.generate_images:
        # "Generate" action: content-only response, no images.
        return build_response(result, [])

    try:
        images = generate_variations(
            prompts=result["image_prompts"],
            negative_prompt=result["negative_prompt"],
        )
    except Exception as exc:
        logger.error("Image generation failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Image generation failed: {exc}")

    try:
        # Upload to S3 and return short presigned URLs instead of embedding
        # base64 image bytes in the JSON response — see
        # app/services/image_storage_service.py for why.
        image_urls = upload_images_to_s3(images)
    except Exception as exc:
        logger.error("Image upload to S3 failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Image upload failed: {exc}")

    return build_response(result, image_urls)


@router.post("/regenerate-image", response_model=RegenerateImageResponse)
def regenerate_image(request: RegenerateImageRequest) -> RegenerateImageResponse:
    """Regenerate ONE new image from an already-produced image_prompt/
    negative_prompt — backs the frontend's per-image recycle button.

    Deliberately skips content generation and image-prompt generation
    entirely (the caller already has both from a prior /generate or
    /generate/stream response) and just renders one fresh variation of
    the same prompt, since the image models are stochastic and a second
    call on the same prompt produces a different image.
    """
    try:
        images = generate_variations(
            prompts=[request.image_prompt],
            negative_prompt=request.negative_prompt or "",
        )
    except Exception as exc:
        logger.error("Image regeneration failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Image regeneration failed: {exc}")

    try:
        import uuid as _uuid

        from app.services.image_storage_service import upload_single_image_to_s3

        url = upload_single_image_to_s3(images[0], _uuid.uuid4().hex, 0)
    except Exception as exc:
        logger.error("Regenerated image upload to S3 failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Image upload failed: {exc}")

    return RegenerateImageResponse(url=url)


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event. Each event is `event: <name>` plus
    a single `data: <json>` line, terminated by a blank line — SSE's
    required framing. json.dumps with default=str so any stray
    non-JSON-serializable value (shouldn't happen, but cheap insurance)
    doesn't crash the whole stream mid-generation.
    """
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/generate/stream")
def generate_stream(request: GenerateContentRequest):
    """Same pipeline as POST /generate (KB retrieve -> combined Nova Lite
    call -> N parallel image renders -> S3 upload), but streamed as
    Server-Sent Events so a frontend can show live per-image progress
    instead of one long blocking spinner. Event sequence:

        stage            {"stage": "retrieving_context"}
        stage            {"stage": "generating_content"}
        content_ready    {content_text, summary, tags, alt_text, image_prompts, negative_prompt}
        image_progress   {"index": i, "status": "submitted"|"polling"|"downloading", "detail": {...}}
        image_ready      {"index": i, "url": "..."}
        image_failed     {"index": i, "error": "..."}
        warning          {"message": "..."}   (e.g. fewer images than requested)
        done             {content: {...}, images: [url, ...]}   -- same shape as POST /generate
        error             {"message": "..."}   -- terminal, stage failed

    Timing budget: the whole request targets TOTAL_DEADLINE_SECONDS
    (default 115s, under the 120s Content + 3 Images ceiling). Whatever's
    left after KB retrieval + the combined LLM call is what image
    generation gets (floor: IMAGE_STREAM_MIN_DEADLINE_SECONDS), passed as
    a per-request poll_timeout override to the image provider — see
    generate_variations_events(). A slow/cold KB (e.g. Aurora resuming)
    eats into this budget same as anything else; there's no way to
    special-case that and still guarantee <120s total.

    Note for local testing: EventSource (the browser's native SSE
    client) only supports GET requests with no body, so this is a POST
    endpoint meant to be consumed via fetch() + a manual
    ReadableStream/TextDecoder read loop instead — see frontend/index.html
    for a working example.
    """
    start = time.monotonic()

    def event_stream():
        try:
            yield _sse("stage", {"stage": "retrieving_context"})
            engine = _build_engine()

            yield _sse("stage", {"stage": "generating_content"})
            try:
                result = engine.generate(
                    prompt=request.prompt,
                    content_type=request.content_type,
                    industry=request.industry,
                    generate_images=request.generate_images,
                    monthly_topic_content=request.monthly_topic_content,
                    common_data=request.common_data,
                    web_results=request.web_results,
                    avoid_repeating=request.avoid_repeating,
                )
            except (ValueError, KnowledgeBaseError, ContentGenerationError) as exc:
                yield _sse("error", {"message": str(exc)})
                return

            yield _sse("content_ready", {
                "content_text": result["content_text"],
                "summary": result.get("summary", ""),
                "tags": result.get("tags", []),
                "alt_text": result.get("alt_text", ""),
                "image_prompts": result["image_prompts"],
                "content_anchors": result.get("content_anchors", []),
                "negative_prompt": result.get("negative_prompt", ""),
                "industry": result.get("selected_industry", ""),
            })

            if not request.generate_images:
                # "Generate" action: content-only — no Image Prompt Agent,
                # no Freepik, no S3. Finish the stream right here.
                response = build_response(result, [])
                yield _sse("done", response.model_dump())
                return

            # Whatever time is left of the total budget goes to image
            # generation; S3 uploads are fast (~1-2s total, parallel) so
            # they aren't given their own separate budget slice.
            elapsed = time.monotonic() - start
            remaining = settings.TOTAL_DEADLINE_SECONDS - elapsed
            image_deadline = max(remaining, settings.IMAGE_STREAM_MIN_DEADLINE_SECONDS)

            generation_id = None
            image_urls: dict = {}
            count = len(result["image_prompts"])
            try:
                for event in generate_variations_events(
                    prompts=result["image_prompts"],
                    negative_prompt=result.get("negative_prompt", ""),
                    deadline_seconds=image_deadline,
                ):
                    if event["type"] == "progress":
                        yield _sse("image_progress", {
                            "index": event["index"], "status": event["status"], "detail": event["detail"],
                        })
                    elif event["type"] == "image":
                        if generation_id is None:
                            import uuid as _uuid
                            generation_id = _uuid.uuid4().hex
                        from app.services.image_storage_service import upload_single_image_to_s3
                        try:
                            url = upload_single_image_to_s3(event["bytes"], generation_id, event["index"])
                            image_urls[event["index"]] = url
                            yield _sse("image_ready", {"index": event["index"], "url": url})
                        except Exception as exc:  # noqa: BLE001
                            yield _sse("image_failed", {"index": event["index"], "error": f"Upload failed: {exc}"})
                    elif event["type"] == "failed":
                        yield _sse("image_failed", {"index": event["index"], "error": event["error"]})
            except Exception as exc:  # noqa: BLE001 - image phase failed outright
                yield _sse("error", {"message": f"Image generation failed: {exc}"})
                return

            if not image_urls:
                yield _sse("error", {"message": "All image variations failed."})
                return
            if len(image_urls) < count:
                yield _sse("warning", {
                    "message": f"Only {len(image_urls)}/{count} image variation(s) succeeded.",
                })

            ordered_urls = [image_urls[i] for i in sorted(image_urls)]
            response = build_response(result, ordered_urls)
            yield _sse("done", response.model_dump())

        except Exception as exc:  # noqa: BLE001 - last-resort so the stream always closes cleanly
            logger.exception("Unexpected error during streamed content generation.")
            yield _sse("error", {"message": f"Unexpected error: {exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )