"""API routes for AI Content Generation."""

import json
import logging
import time
import uuid

import boto3
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from app.schemas.content import (
    CONTENT_TYPES,
    IMAGE_ASPECT_RATIOS,
    IMAGE_COUNT_MAX,
    IMAGE_COUNT_MIN,
    IMAGE_RESOLUTIONS,
    IMAGE_RESOLUTION_PIXELS,
    DEFAULT_IMAGE_COUNT,
    DEFAULT_IMAGE_RESOLUTION,
    GenerateContentRequest,
    GenerateContentResponse,
    ImageSettingsResponse,
    RegenerateImageRequest,
    RegenerateImageResponse,
)
from app.services.content_generation_engine import ContentGenerationEngine, ContentGenerationError
from app.services.image_generation_service import (
    resolution_support_status,
    generate_variations,
    generate_variations_events,
    resolve_prompts_for_count,
)
from app.services.image_storage_service import (
    presigned_url_for,
    upload_images_to_s3,
    upload_single_image_to_s3,
)
from app.services.knowledge_base_service import KnowledgeBaseError, KnowledgeBaseService
from app.services.llm_service import get_combined_llm
from app.services.response_builder import build_response

logger = logging.getLogger(__name__)
router = APIRouter()

# A 1-image request always renders synchronously (see below) and never
# touches this queue at all — only 2- or 3-image requests go async via
# SQS, since that's the only case where there's genuine parallel work to
# hand off instead of doing inline. This keeps the common/fast case
# (1 image) exactly as fast as a direct Freepik call always was.
_SQS_IMAGE_COUNT_THRESHOLD = 2

_sqs_client = None


def _sqs():
    global _sqs_client
    if _sqs_client is None:
        # No explicit credentials — boto3's default chain handles this via
        # the Lambda execution role, same pattern as every other boto3
        # client in this codebase.
        _sqs_client = boto3.client("sqs", region_name=settings.SQS_REGION)
    return _sqs_client


def _enqueue_image_jobs(
    request_id: str,
    prompts: list,
    negative_prompt: str,
    aspect_ratio,
    resolution,
) -> None:
    """Send ONE SQS message per entry in `prompts` to ImageGenerationQueue
    — only called for 2- or 3-image requests (see
    _SQS_IMAGE_COUNT_THRESHOLD above). Each message already contains the
    fully-generated image_prompt — app/image_worker.py never regenerates
    content or image prompts, it only renders what's already here.

    Each message is self-contained: request_id + image_index + everything
    needed to render and store that ONE image, so the worker never needs
    to look anything up elsewhere (no DynamoDB, no shared job-tracking
    state). Concurrent requests from different users never share a
    request_id (each call mints its own via uuid.uuid4() — see callers),
    so their messages/S3 keys never collide.
    """
    if not settings.IMAGE_QUEUE_URL:
        raise RuntimeError(
            "IMAGE_QUEUE_URL is not configured. Set it to the SQS queue URL "
            "created by template.yaml's ImageGenerationQueue resource."
        )
    client = _sqs()
    for index, prompt in enumerate(prompts):
        message = {
            "request_id": request_id,
            "image_index": index,
            "image_prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
        }
        client.send_message(QueueUrl=settings.IMAGE_QUEUE_URL, MessageBody=json.dumps(message))


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


@router.get("/image-settings", response_model=ImageSettingsResponse)
def image_settings() -> ImageSettingsResponse:
    """Lets the frontend build its Image Size / Image Count / Pixel-
    Resolution controls from the server's actual, currently-configured
    provider/model (app.config.settings.IMAGE_PROVIDER / FREEPIK_MODEL)
    instead of hardcoding assumptions that could silently drift out of
    sync with a deploy-time model change.

    The Resolution control (aspect_ratios) is ALWAYS all three tiers
    regardless of provider/model — resolution_applies/resolution_note
    below are purely informational; a request naming an unsupported-by-
    the-current-model resolution is still valid (see
    _resolution_support_for_current_config() below), never rejected.
    """
    is_freepik = settings.IMAGE_PROVIDER == "freepik"
    applies, note = resolution_support_status()
    return ImageSettingsResponse(
        provider=settings.IMAGE_PROVIDER,
        model=settings.FREEPIK_MODEL if is_freepik else "",
        aspect_ratios=list(IMAGE_ASPECT_RATIOS),
        resolutions=list(IMAGE_RESOLUTIONS),
        resolution_pixels=dict(IMAGE_RESOLUTION_PIXELS),
        resolution_applies=applies,
        resolution_note=note,
        default_aspect_ratio="1:1",
        default_resolution=DEFAULT_IMAGE_RESOLUTION,
        default_count=DEFAULT_IMAGE_COUNT,
        min_count=IMAGE_COUNT_MIN,
        max_count=IMAGE_COUNT_MAX,
    )


def _resolution_support_for_current_config(resolution: str = None) -> tuple:
    """Thin wrapper around image_generation_service.resolution_support_status()
    — kept as a separate name in this module since GenerateContentResponse's
    docstring and this file's other comments reference it directly.
    `resolution`, if given, checks that SPECIFIC tier (e.g. some Freepik
    models support only a subset — see FreepikImageService._RESOLUTION_SUPPORT);
    omitted, it's a general "does this provider/model support resolution
    at all" check (used by GET /image-settings)."""
    return resolution_support_status(resolution)


def _validate_image_settings(request: GenerateContentRequest):
    """Backend validation for the three per-request image settings (see
    app/schemas/content.py's GenerateContentRequest.image_aspect_ratio /
    .image_count / .image_resolution). image_count's range (1-3) is
    already enforced by Pydantic's Field(ge=, le=) and would 422 before
    this function ever runs; the explicit re-check here is cheap
    defensive consistency, not the primary enforcement for that one
    field.

    This only validates that image_resolution is one of the 2 known
    tier labels ("2K"/"4K") — it deliberately does NOT check
    whether the currently configured provider/model can actually honor
    it; that's a separate, non-blocking concern (see
    _resolution_support_for_current_config() and _apply_resolution()
    below) so a request naming a resolution the current model can't use
    is still a perfectly valid request, just gracefully degraded rather
    than rejected.

    Returns (aspect_ratio, resolution, count) — each None when the
    request omitted it, meaning "use the server's safe default" (see
    get_image_gen_service() / resolve_prompts_for_count()).
    """
    aspect_ratio = request.image_aspect_ratio
    if aspect_ratio is not None and aspect_ratio not in IMAGE_ASPECT_RATIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image_aspect_ratio '{aspect_ratio}'. Must be one of: "
            f"{', '.join(IMAGE_ASPECT_RATIOS)}.",
        )
    resolution = request.image_resolution
    if resolution is not None and resolution not in IMAGE_RESOLUTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image_resolution '{resolution}'. Must be one of: "
            f"{', '.join(IMAGE_RESOLUTIONS)}.",
        )
    count = request.image_count
    if count is not None and not (IMAGE_COUNT_MIN <= count <= IMAGE_COUNT_MAX):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image_count {count}. Must be between {IMAGE_COUNT_MIN} and {IMAGE_COUNT_MAX}.",
        )
    return aspect_ratio, resolution, count


def _apply_resolution(resolution):
    """Given a validated (but not yet support-checked) resolution tier
    label, decide what to actually pass on to image generation and
    whether a warning is needed. Returns (resolution_for_call, warning):
      - resolution is None (omitted): (None, None) — nothing to do,
        server default applies as before this feature existed.
      - the current provider/model CAN honor it: (resolution, None) —
        passed straight through to generate_variations()/_events().
      - the current provider/model CANNOT honor it (Freepik running a
        Flux/Seedream model — see resolution_support_status()):
        (None, "<human-readable warning>") — dropped for this call
        rather than sent as an invalid Freepik parameter; the caller
        appends the warning to the response's `warnings` field (see
        GenerateContentResponse.warnings) instead of raising, so the
        rest of the request still completes normally.
    """
    if not resolution:
        return None, None
    applies, note = _resolution_support_for_current_config(resolution)
    if applies:
        return resolution, None
    warning = (
        f"Selected resolution '{resolution}' is not supported by the currently configured "
        f"image model and was ignored."
    )
    if note:
        warning += f" {note}"
    return None, warning


@router.post("/generate", response_model=GenerateContentResponse)
def generate(request: GenerateContentRequest) -> GenerateContentResponse:
    """Generate content for the given prompt/content type/industry, and
    — only when request.generate_images is True (the "Generate + Image"
    action) — also generate the requested number of grounded image
    variations for it.

    When request.generate_images is False (the plain "Generate" action),
    this returns content only: the Image Prompt Agent, Freepik, and S3
    image upload are never invoked.
    """
    engine = _build_engine()
    aspect_ratio, resolution, count = _validate_image_settings(request)

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
            # Image Prompt Agent now drafts exactly this many prompts
            # from the start (see ContentGenerationEngine.generate()'s
            # image_count param) rather than always drafting 3 and
            # discarding the unused ones downstream. `count` is None
            # when the request omitted image_count — DEFAULT_IMAGE_COUNT
            # (3) applies in that case, same default as before.
            image_count=count or DEFAULT_IMAGE_COUNT,
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
        # "Generate" action: content-only response. No image prompts, no
        # SQS messages, no Freepik, no S3 — exactly as before.
        return build_response(result, [])

    # Graceful degradation for a resolution the currently configured
    # provider/model can't actually honor (see _apply_resolution() /
    # resolution_support_status()) — never an error, just dropped for
    # this call with a note surfaced in the response's warnings field.
    resolution_for_call, resolution_warning = _apply_resolution(resolution)
    warnings = [resolution_warning] if resolution_warning else []

    # Existing Image Prompt Agent logic — completely unchanged. It still
    # always produces exactly 3 distinct prompts (see
    # prompts/image_prompt_system.txt); image_count only decides how
    # many of those 3 to actually render for this one request (1-3).
    # content_anchors is resolved with the identical (count,
    # source-length) inputs so it stays index-aligned with
    # render_prompts / the final images list.
    render_prompts = resolve_prompts_for_count(result["image_prompts"], count)
    render_anchors = resolve_prompts_for_count(result.get("content_anchors") or [], count)
    result = {**result, "image_prompts": render_prompts, "content_anchors": render_anchors}

    if len(render_prompts) < _SQS_IMAGE_COUNT_THRESHOLD:
        # Exactly 1 image: render it inline, synchronously — no SQS at
        # all. There's no parallel work to hand off for a single image,
        # so a queue round-trip would only add latency here, not reduce
        # it. This is the same direct-Freepik call this endpoint has
        # always used.
        try:
            images = generate_variations(
                prompts=render_prompts,
                negative_prompt=result["negative_prompt"],
                aspect_ratio=aspect_ratio,
                resolution=resolution_for_call,
            )
        except Exception as exc:
            logger.error("Image generation failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"Image generation failed: {exc}")

        try:
            # Upload to S3 and return short presigned URLs instead of
            # embedding base64 image bytes in the JSON response — see
            # app/services/image_storage_service.py for why.
            image_urls = upload_images_to_s3(images)
        except Exception as exc:
            logger.error("Image upload to S3 failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"Image upload failed: {exc}")

        response = build_response(result, image_urls)
        response.warnings = warnings
        return response

    # 2 or 3 images: hand each off to SQS instead of rendering inline —
    # this is the actual parallelism win, since 2-3 independent Freepik
    # calls no longer have to complete one after another (or even in
    # parallel threads within THIS invocation's timeout budget) before
    # this request can return. Each job already carries its own fully-
    # generated image_prompt (app/image_worker.py never regenerates
    # content or prompts, only renders). Presigned URLs are minted BEFORE
    # the objects exist (see image_storage_service.presigned_url_for()'s
    # docstring for why this is an ordinary, safe S3 operation) so the
    # response is still immediately useful, even though the images
    # themselves aren't ready yet.
    request_id = uuid.uuid4().hex
    try:
        image_urls = [presigned_url_for(request_id, i) for i in range(len(render_prompts))]
    except Exception as exc:
        logger.error("Failed to prepare presigned image URLs: %s", exc)
        raise HTTPException(status_code=502, detail=f"Failed to prepare image URLs: {exc}")

    try:
        _enqueue_image_jobs(
            request_id=request_id,
            prompts=render_prompts,
            negative_prompt=result.get("negative_prompt", ""),
            aspect_ratio=aspect_ratio,
            resolution=resolution_for_call,
        )
    except Exception as exc:
        logger.error("Failed to queue image generation jobs: %s", exc)
        raise HTTPException(status_code=502, detail=f"Failed to queue image generation: {exc}")

    warnings.append(
        f"{len(render_prompts)} image(s) are processing in the background (request_id={request_id}). "
        "The URLs above will resolve as each one finishes — this can take a couple of minutes; "
        "retry loading an image if it doesn't appear immediately."
    )
    response = build_response(result, image_urls)
    response.warnings = warnings
    return response


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
    aspect_ratio = request.image_aspect_ratio
    if aspect_ratio is not None and aspect_ratio not in IMAGE_ASPECT_RATIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image_aspect_ratio '{aspect_ratio}'. Must be one of: "
            f"{', '.join(IMAGE_ASPECT_RATIOS)}.",
        )
    resolution = request.image_resolution
    if resolution is not None and resolution not in IMAGE_RESOLUTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image_resolution '{resolution}'. Must be one of: "
            f"{', '.join(IMAGE_RESOLUTIONS)}.",
        )
    # Same graceful degradation as /generate — a resolution the current
    # provider/model can't honor is dropped for this call rather than
    # sent to Freepik as an invalid parameter. RegenerateImageResponse
    # has no warnings field (it's a single-image, best-effort re-render
    # of an already-produced prompt), so this is logged rather than
    # surfaced in the response body.
    resolution_for_call, resolution_warning = _apply_resolution(resolution)
    if resolution_warning:
        logger.info("regenerate-image: %s", resolution_warning)
    try:
        images = generate_variations(
            prompts=[request.image_prompt],
            negative_prompt=request.negative_prompt or "",
            aspect_ratio=aspect_ratio,
            resolution=resolution_for_call,
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
        warning          {"message": "..."}   (e.g. fewer images than requested, or an unsupported resolution was dropped)
        done             {content: {...}, images: [url, ...], warnings: [...]}   -- same shape as POST /generate
        error            {"message": "..."}   -- terminal, stage failed

    Timing budget: the whole request targets TOTAL_DEADLINE_SECONDS
    (default 115s, under the Lambda's own Timeout). Whatever's left
    after KB retrieval + the combined LLM call is what image generation
    gets (floor: IMAGE_STREAM_MIN_DEADLINE_SECONDS), passed as a
    per-request poll_timeout override to the image provider — see
    generate_variations_events(). A slow/cold KB (e.g. Aurora resuming)
    eats into this budget same as anything else; there's no way to
    special-case that and still guarantee the request finishes in time.

    Note for local testing: EventSource (the browser's native SSE
    client) only supports GET requests with no body, so this is a POST
    endpoint meant to be consumed via fetch() + a manual
    ReadableStream/TextDecoder read loop instead — see frontend/index.html
    for a working example.
    """
    start = time.monotonic()
    aspect_ratio, resolution, requested_count = _validate_image_settings(request)

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
                    # See the /generate route's identical comment above —
                    # requested_count is None when the request omitted
                    # image_count; DEFAULT_IMAGE_COUNT (3) applies then.
                    image_count=requested_count or DEFAULT_IMAGE_COUNT,
                )
            except (ValueError, KnowledgeBaseError, ContentGenerationError) as exc:
                yield _sse("error", {"message": str(exc)})
                return

            yield _sse("content_ready", {
                "content_text": result["content_text"],
                "summary": result.get("summary", ""),
                "tags": result.get("tags", []),
                "alt_text": result.get("alt_text", ""),
                # Always the Image Prompt Agent's full set of 3 distinct
                # prompts here, regardless of image_count — this event is
                # informational/traceability, separate from how many are
                # actually rendered below (see render_prompts).
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

            # Graceful degradation for a resolution the currently
            # configured provider/model can't actually honor — see
            # _apply_resolution()/resolution_support_status(). Never a
            # stream-ending error; surfaced as a "warning" SSE event and
            # in the final "done" payload's warnings field.
            resolution_for_call, resolution_warning = _apply_resolution(resolution)
            stream_warnings = []
            if resolution_warning:
                stream_warnings.append(resolution_warning)
                yield _sse("warning", {"message": resolution_warning})

            # See _validate_image_settings()/resolve_prompts_for_count()
            # in app/routes/content.py — image_count only decides how
            # many of the Agent's 3 prompts to RENDER (1-3), never how
            # many it generates (the content_ready event above always
            # shows the full 3 regardless).
            render_prompts = resolve_prompts_for_count(result["image_prompts"], requested_count)
            render_anchors = resolve_prompts_for_count(result.get("content_anchors") or [], requested_count)
            result = {**result, "image_prompts": render_prompts, "content_anchors": render_anchors}

            if len(render_prompts) < _SQS_IMAGE_COUNT_THRESHOLD:
                # Exactly 1 image: render it inline with real-time
                # progress events, same as always — no SQS, no queueing
                # delay, since there's no parallel work to hand off for
                # a single image.
                generation_id = None
                image_urls: dict = {}
                count = len(render_prompts)
                try:
                    for event in generate_variations_events(
                        prompts=render_prompts,
                        negative_prompt=result.get("negative_prompt", ""),
                        deadline_seconds=image_deadline,
                        aspect_ratio=aspect_ratio,
                        resolution=resolution_for_call,
                    ):
                        if event["type"] == "progress":
                            yield _sse("image_progress", {
                                "index": event["index"], "status": event["status"], "detail": event["detail"],
                            })
                        elif event["type"] == "image":
                            if generation_id is None:
                                generation_id = uuid.uuid4().hex
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
                response.warnings = stream_warnings
                yield _sse("done", response.model_dump())
                return

            # 2 or 3 images: enqueue via SQS and return immediately —
            # this Lambda never waits for Freepik across multiple images
            # in series or even in parallel-within-this-invocation; each
            # job is picked up by a SEPARATE invocation of this same
            # function (see app/main.py's handler / app/image_worker.py),
            # running concurrently with the others (see template.yaml's
            # ScalingConfig.MaximumConcurrency on the SQS event source).
            request_id = uuid.uuid4().hex
            try:
                image_urls_list = [presigned_url_for(request_id, i) for i in range(len(render_prompts))]
            except Exception as exc:  # noqa: BLE001
                yield _sse("error", {"message": f"Failed to prepare image URLs: {exc}"})
                return

            try:
                _enqueue_image_jobs(
                    request_id=request_id,
                    prompts=render_prompts,
                    negative_prompt=result.get("negative_prompt", ""),
                    aspect_ratio=aspect_ratio,
                    resolution=resolution_for_call,
                )
            except Exception as exc:  # noqa: BLE001
                yield _sse("error", {"message": f"Failed to queue image generation: {exc}"})
                return

            # "image_processing" (not "image_queued"/"image_ready") is
            # deliberate wording: the frontend must represent this as
            # PROCESSING, never as a failure, no matter how long the
            # actual render takes — there is no real completion signal
            # available on this stream once these jobs are handed off
            # (no job-tracking/status system was added), so the frontend
            # polls the presigned URL itself and simply keeps showing
            # "processing" until it resolves.
            for i, url in enumerate(image_urls_list):
                yield _sse("image_processing", {"index": i, "url": url})

            stream_warnings.append(
                f"{len(render_prompts)} image(s) are processing in the background (request_id={request_id})."
            )
            response = build_response(result, image_urls_list)
            response.warnings = stream_warnings
            yield _sse("done", response.model_dump())

        except Exception as exc:  # noqa: BLE001 - last-resort so the stream always closes cleanly
            logger.exception("Unexpected error during streamed content generation.")
            yield _sse("error", {"message": f"Unexpected error: {exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )