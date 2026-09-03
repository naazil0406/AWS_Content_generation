"""API routes for AI Content Generation."""

import json
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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
    CheckTopicResponse,
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
    upload_images_to_s3,
    upload_single_image_to_s3,
)
from app.services.knowledge_base_service import KnowledgeBaseError, KnowledgeBaseService
from app.services.llm_service import get_combined_llm
from app.services.response_builder import build_response
from app.services import validation_service
from app.services.prompt_builder import load_available_industries
from app.services.validation_service import ValidationError

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


def _kb_sample_fn():
    """Passed as validation_service.run_input_validation()'s kb_sample_fn
    — a broad, generic query (not the user's own, possibly-unrelated one)
    used ONLY to surface a few real example topics when input fails the
    meaningful-content-request check (PART 3). "safety topics" matches
    this app's actual designed domain (every content type/prompt in this
    project is workplace-safety-training-focused) — adjust this one
    query if your Knowledge Base covers a different domain."""
    return KnowledgeBaseService().retrieve("safety topics", top_k=8)


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


class CheckTopicRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


@router.post("/check-topic", response_model=CheckTopicResponse)
def check_topic(request: CheckTopicRequest) -> CheckTopicResponse:
    """PART 2-5 of the KB-relevance spec: runs input validation +
    understanding, retrieves Knowledge Base context for the (corrected)
    prompt, and classifies the match as exact/similar/none — WITHOUT
    generating any content or images. The frontend calls this FIRST,
    before /generate or /generate/stream, and only proceeds to actually
    generate for "exact" matches directly, or "similar" matches after
    explicit user approval (resending with approved_alternative_topic
    set) — "none" matches stop here with a warning, no generation call
    is ever made.
    """
    validation_llm = get_combined_llm()
    corrected_prompt, spelling_warning, intent = validation_service.run_input_validation(
        validation_llm, request.prompt, kb_sample_fn=_kb_sample_fn,
    )  # raises ValidationError (400) for empty/special-chars/meaningless/moderation — same as /generate

    # Query the Knowledge Base with kb_search_query — a version of the
    # prompt that strips ONLY the conversational/instructional wrapper
    # ("give me an image related to...") while preserving every
    # substantive concept (subject, action, context, industry, who's
    # involved). Deliberately NOT the same as intent["topic"] (which can
    # be a short label like "rushing") — reducing the KB query down to
    # just the topic word risks losing real semantic information (e.g.
    # "a woman rushing in healthcare industry" needs "woman"/"healthcare
    # industry" preserved too, not just "rushing"). Falls back to
    # intent["topic"], then corrected_prompt, if the LLM didn't populate
    # kb_search_query for some reason.
    kb_query = (intent or {}).get("kb_search_query") or (intent or {}).get("topic") or corrected_prompt

    try:
        kb = KnowledgeBaseService()
        chunks = kb.retrieve(kb_query)
    except KnowledgeBaseError as exc:
        raise HTTPException(status_code=502, detail=f"Knowledge Base error: {exc}")

    # [KB] Diagnostic visibility into exactly what was searched and what
    # came back — needed because "why did a seemingly-relevant prompt
    # get classified as no-match" is otherwise unanswerable without
    # seeing the actual query text and chunk scores, not just guessing.
    top_scores = sorted(
        (c.get("score") for c in chunks if isinstance(c.get("score"), (int, float))), reverse=True,
    )[:5]
    logger.info(
        "[KB] check-topic: original_prompt=%r intent_topic=%r kb_query=%r chunks_returned=%d top_scores=%s",
        request.prompt, (intent or {}).get("topic"), kb_query, len(chunks), top_scores,
    )

    match = validation_service.classify_kb_match(validation_llm, request.prompt, chunks)
    logger.info("[KB] check-topic: classification=%s", match)

    if match["match_type"] == "none":
        base_message = validation_service.get_rule(
            "KNOWLEDGE_BASE_RELEVANCE", "no_match_warning",
            "We couldn't find relevant information for your prompt in the knowledge base. Please try another prompt.",
        )
        # PART 3: don't leave the user at a dead end — sample a BROAD
        # slice of the Knowledge Base (a generic query, not the user's
        # specific/unrelated one) and extract a few real topic names
        # they could actually ask about instead. Best-effort: if this
        # sample or extraction fails for any reason, the base message
        # alone is still a complete, correct response — this only adds
        # to it, never blocks on it.
        message = base_message
        try:
            sample_chunks = kb.retrieve("safety topics", top_k=8)
            suggested_topics = validation_service.suggest_kb_topics(validation_llm, sample_chunks)
            if suggested_topics:
                message += " Topics you can ask about include: " + ", ".join(suggested_topics) + "."
        except Exception as exc:  # noqa: BLE001 - suggestions are a nice-to-have, never fatal
            logger.warning("Failed to sample Knowledge Base for topic suggestions: %s", exc)
    elif match["match_type"] == "similar":
        template = validation_service.get_rule(
            "KNOWLEDGE_BASE_RELEVANCE", "similar_match_prompt_template",
            "We couldn't find an exact match for your prompt. We found related information about {topic}. "
            "Would you like us to generate content based on this related topic?",
        )
        message = template.format(topic=match["matched_topic"] or "a related topic")
    else:
        message = ""  # exact match — frontend proceeds directly, no message needed

    return CheckTopicResponse(
        match_type=match["match_type"],
        matched_topic=match.get("matched_topic"),
        message=message,
        corrected_prompt=corrected_prompt,
        spelling_warning=spelling_warning,
    )


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
    aspect_ratio, resolution, count = _validate_image_settings(request)

    # Validation pipeline (basic -> special-char -> meaningful -> spelling
    # -> moderation -> understanding) — runs BEFORE any content/image
    # generation call, and BEFORE the engine (Knowledge Base client, LLM
    # clients) is even built, per the explicit "don't call expensive
    # generation APIs for clearly invalid input" performance requirement.
    # Raises ValidationError, caught by main.py's exception handler and
    # turned into {success: false, warning, error_type}.
    validation_llm = get_combined_llm()
    # [PERF] available_industries is passed straight into the SAME
    # validation call below (see analyze_prompt()'s docstring) instead
    # of a separate suggest_consistent_industry() round trip afterward —
    # cuts one full sequential LLM call off the happy path, which
    # matters because this entire chain runs before Freepik/image
    # rendering even starts, inside API Gateway's fixed ~29s ceiling.
    corrected_prompt, spelling_warning, intent = validation_service.run_input_validation(
        validation_llm, request.prompt, kb_sample_fn=_kb_sample_fn,
        available_industries=list(load_available_industries()),
    )
    warnings: List[str] = [spelling_warning] if spelling_warning else []
    if request.approved_alternative_topic:
        # PART 5: exact required message format for a user-approved
        # alternative topic (see POST /api/check-topic — this field is
        # only ever set after that explicit user approval, never
        # inferred here).
        banner_template = validation_service.get_rule(
            "KNOWLEDGE_BASE_RELEVANCE", "alternative_topic_banner_template",
            "Showing results related to {topic} since we couldn't find an exact match for your prompt.",
        )
        warnings.append(banner_template.format(topic=request.approved_alternative_topic))

    # General (not example-specific) fix for topic/industry mismatch —
    # see app/services/validation_service.analyze_prompt()'s docstring:
    # rather than relying solely on prompt_builder.py's hand-curated
    # per-industry keyword lists (which can never anticipate every
    # possible topic), the SAME validation LLM call above already asked
    # whether THIS topic clearly implies one specific industry, for ANY
    # topic. Only overrides an explicit request.industry if the caller
    # didn't already provide one; None (from intent) means no override —
    # existing random/keyword-hint selection in
    # ContentGenerationEngine.generate() runs unchanged — for genuinely
    # generic topics like "workplace safety".
    effective_industry = request.industry or (intent or {}).get("suggested_industry")

    engine = _build_engine()

    _request_started = time.monotonic()
    try:
        result = engine.generate(
            # corrected_prompt (not request.prompt) drives actual
            # generation — request.prompt (the true original) remains
            # available below for the content-relevance check, which
            # must always judge against what the user actually typed,
            # not the spelling-corrected version (PART 9: original
            # prompt is always the source of truth).
            prompt=corrected_prompt,
            content_type=request.content_type,
            industry=effective_industry,
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

        # Content relevance check (PART 10) — one bounded retry with a
        # stricter regeneration instruction if the first attempt drifted
        # off-topic. Judged against request.prompt (the TRUE original),
        # never corrected_prompt, per PART 9.
        if not validation_service.check_content_relevance(validation_llm, request.prompt, result["content_text"]):
            warnings.append(
                "⚠️ The generated content does not sufficiently match your request. "
                "Regenerating based on your original prompt."
            )
            retry_prompt = corrected_prompt + validation_service.STRICT_REGENERATION_SUFFIX
            result = engine.generate(
                prompt=retry_prompt,
                content_type=request.content_type,
                industry=effective_industry,
                generate_images=request.generate_images,
                monthly_topic_content=request.monthly_topic_content,
                common_data=request.common_data,
                web_results=request.web_results,
                avoid_repeating=request.avoid_repeating,
                image_count=count or DEFAULT_IMAGE_COUNT,
            )
            # PART 16: "if the retry still fails, return a clear warning
            # instead of an unrelated result" — the previous version of
            # this code used the retry's output unconditionally, with no
            # second check. Retry limit is exhausted after this one
            # attempt either way (no loop) — the difference is now the
            # user is honestly told the result may still not fully match,
            # rather than the mismatch being silently presented as if it
            # were resolved.
            if not validation_service.check_content_relevance(validation_llm, request.prompt, result["content_text"]):
                warnings.append(
                    "⚠️ The regenerated content still may not fully match your request. "
                    "Consider rephrasing your prompt with a more specific topic."
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

    logger.info(
        "[PERF] Content + image-prompt generation total (POST /generate): %.1fs",
        time.monotonic() - _request_started,
    )

    if not request.generate_images:
        # "Generate" action: content-only response. No image prompts, no
        # SQS messages, no Freepik, no S3 — exactly as before.
        response = build_response(result, [])
        response.warnings = warnings
        return response

    # Graceful degradation for a resolution the currently configured
    # provider/model can't actually honor (see _apply_resolution() /
    # resolution_support_status()) — never an error, just dropped for
    # this call with a note surfaced in the response's warnings field.
    resolution_for_call, resolution_warning = _apply_resolution(resolution)
    if resolution_warning:
        warnings.append(resolution_warning)

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

    # [PERF]/diagnostic: the actual text about to be sent to Freepik —
    # needed to verify explicit persona/demographic/detail requirements
    # from the user's prompt (e.g. "a woman") actually survived into the
    # image prompt, since that's invisible otherwise until the rendered
    # image itself.
    logger.info("[IMG] initial image_prompts: %s", render_prompts)

    # Image PROMPT relevance check (PART 9/13 of the spec) — checked
    # against the actual TEXT about to be sent to Freepik, BEFORE
    # spending time/money rendering it. This is not a check of rendered
    # pixels (no vision-model call exists here — see
    # validation_service.py's module docstring), but it directly catches
    # the documented failure mode where content stays on-topic while the
    # image prompt drifts (e.g. into a generic office/industry scene)
    # — one bounded retry, same pattern as the content-relevance check.
    if not validation_service.check_image_prompt_relevance(validation_llm, request.prompt, result["content_text"], render_prompts):
        warnings.append(
            "⚠️ The generated image does not sufficiently match your request. Regenerating based on your original prompt."
        )
        retry_prompt = corrected_prompt + validation_service.STRICT_REGENERATION_SUFFIX
        result = engine.generate(
            prompt=retry_prompt,
            content_type=request.content_type,
            industry=effective_industry,
            generate_images=request.generate_images,
            monthly_topic_content=request.monthly_topic_content,
            common_data=request.common_data,
            web_results=request.web_results,
            avoid_repeating=request.avoid_repeating,
            image_count=count or DEFAULT_IMAGE_COUNT,
        )
        render_prompts = resolve_prompts_for_count(result["image_prompts"], count)
        render_anchors = resolve_prompts_for_count(result.get("content_anchors") or [], count)
        result = {**result, "image_prompts": render_prompts, "content_anchors": render_anchors}
        if not validation_service.check_image_prompt_relevance(validation_llm, request.prompt, result["content_text"], render_prompts):
            warnings.append(
                "⚠️ The regenerated image may still not fully match your request. "
                "Consider rephrasing your prompt with a more specific topic."
            )

    # All 1-3 images render inline, synchronously, in this same Lambda
    # invocation — no SQS, no second invocation, no queue drift to
    # manage. generate_variations() already runs each image on its own
    # thread (ThreadPoolExecutor), so 2-3 images still render fully in
    # parallel — image 2 never waits on image 1 — the only difference
    # from the old SQS path is that THIS request blocks until all of
    # them finish, instead of returning immediately and rendering off
    # to the side. That trade means this endpoint's total time is
    # roughly one image's worth of latency (all images run in parallel
    # threads), not N images' worth — but the whole request must still
    # fit inside API Gateway HTTP API's hard ~29-30s integration
    # timeout. With Freepik Mystic (~90-130s/image) this WILL exceed
    # that and return a 504 for 2-3 image requests; with a fast model
    # (hyperflux, seedream-v5-lite, ~20-40s) it fits. If you need
    # Mystic-level quality with multiple images inside a hard 30s
    # window, use POST /generate/stream instead (below) — same
    # blocking-per-request constraint applies there too, since API
    # Gateway HTTP API does not support true response streaming for
    # Lambda proxy integrations; SSE only helps if the Lambda itself
    # finishes in time.
    _image_gen_started = time.monotonic()
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
    logger.info(
        "[PERF] Freepik generation (%d image(s), parallel threads): %.1fs",
        len(render_prompts), time.monotonic() - _image_gen_started,
    )

    # Minted here (not inside upload_images_to_s3, which used to
    # generate its own if omitted) so callers get a stable identifier
    # for this specific generation, tying together images[i] with index
    # i under this generation_id.
    generation_id = uuid.uuid4().hex

    _s3_upload_started = time.monotonic()
    try:
        # Upload to S3 and return short presigned URLs instead of
        # embedding base64 image bytes in the JSON response — see
        # app/services/image_storage_service.py for why.
        image_urls = upload_images_to_s3(images, generation_id=generation_id)
    except Exception as exc:
        logger.error("Image upload to S3 failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Image upload failed: {exc}")
    logger.info("[PERF] S3 upload (%d image(s)): %.1fs", len(images), time.monotonic() - _s3_upload_started)

    response = build_response(result, image_urls)
    response.warnings = warnings
    response.generation_id = generation_id
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

    # Validation pipeline — same as POST /generate (see that route's
    # identical block for full rationale). Runs BEFORE the SSE stream
    # even starts: on rejection, ValidationError propagates out of this
    # synchronous section and is caught by main.py's exception handler,
    # returning a normal {success: false, warning, error_type} JSON
    # response instead of ever opening an event stream — there is
    # nothing to stream progress for on rejected input.
    validation_llm = get_combined_llm()
    # [PERF] See POST /generate's identical block — available_industries
    # is folded into this SAME validation call instead of a separate
    # suggest_consistent_industry() round trip, cutting one sequential
    # LLM call off the happy path.
    corrected_prompt, spelling_warning, intent = validation_service.run_input_validation(
        validation_llm, request.prompt, kb_sample_fn=_kb_sample_fn,
        available_industries=list(load_available_industries()),
    )
    stream_warnings: List[str] = [spelling_warning] if spelling_warning else []
    if request.approved_alternative_topic:
        banner_template = validation_service.get_rule(
            "KNOWLEDGE_BASE_RELEVANCE", "alternative_topic_banner_template",
            "Showing results related to {topic} since we couldn't find an exact match for your prompt.",
        )
        stream_warnings.append(banner_template.format(topic=request.approved_alternative_topic))

    # See POST /generate's identical block for full rationale.
    effective_industry = request.industry or (intent or {}).get("suggested_industry")

    def event_stream():
        try:
            yield _sse("stage", {"stage": "retrieving_context"})
            engine = _build_engine()

            yield _sse("stage", {"stage": "generating_content"})
            try:
                result = engine.generate(
                    # corrected_prompt drives generation; request.prompt
                    # (true original) is what the relevance check below
                    # judges against — see PART 9.
                    prompt=corrected_prompt,
                    content_type=request.content_type,
                    industry=effective_industry,
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

                # Content relevance check (PART 10) — one bounded retry,
                # same as POST /generate.
                if not validation_service.check_content_relevance(validation_llm, request.prompt, result["content_text"]):
                    relevance_warning = (
                        "⚠️ The generated content does not sufficiently match your request. "
                        "Regenerating based on your original prompt."
                    )
                    stream_warnings.append(relevance_warning)
                    yield _sse("warning", {"message": relevance_warning})
                    result = engine.generate(
                        prompt=corrected_prompt + validation_service.STRICT_REGENERATION_SUFFIX,
                        content_type=request.content_type,
                        industry=effective_industry,
                        generate_images=request.generate_images,
                        monthly_topic_content=request.monthly_topic_content,
                        common_data=request.common_data,
                        web_results=request.web_results,
                        avoid_repeating=request.avoid_repeating,
                        image_count=requested_count or DEFAULT_IMAGE_COUNT,
                    )
                    # PART 16: check the retry too — see the identical
                    # fix/comment in POST /generate above for why this
                    # was missing before.
                    if not validation_service.check_content_relevance(validation_llm, request.prompt, result["content_text"]):
                        still_mismatched_warning = (
                            "⚠️ The regenerated content still may not fully match your request. "
                            "Consider rephrasing your prompt with a more specific topic."
                        )
                        stream_warnings.append(still_mismatched_warning)
                        yield _sse("warning", {"message": still_mismatched_warning})
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
                response.warnings = stream_warnings
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
            logger.info("[IMG] initial image_prompts: %s", render_prompts)
            result = {**result, "image_prompts": render_prompts, "content_anchors": render_anchors}

            # Image PROMPT relevance check — same as POST /generate, see
            # that route's identical block for full rationale.
            if not validation_service.check_image_prompt_relevance(validation_llm, request.prompt, result["content_text"], render_prompts):
                image_relevance_warning = (
                    "⚠️ The generated image does not sufficiently match your request. "
                    "Regenerating based on your original prompt."
                )
                stream_warnings.append(image_relevance_warning)
                yield _sse("warning", {"message": image_relevance_warning})
                result = engine.generate(
                    prompt=corrected_prompt + validation_service.STRICT_REGENERATION_SUFFIX,
                    content_type=request.content_type,
                    industry=effective_industry,
                    generate_images=request.generate_images,
                    monthly_topic_content=request.monthly_topic_content,
                    common_data=request.common_data,
                    web_results=request.web_results,
                    avoid_repeating=request.avoid_repeating,
                    image_count=requested_count or DEFAULT_IMAGE_COUNT,
                )
                render_prompts = resolve_prompts_for_count(result["image_prompts"], requested_count)
                render_anchors = resolve_prompts_for_count(result.get("content_anchors") or [], requested_count)
                result = {**result, "image_prompts": render_prompts, "content_anchors": render_anchors}
                if not validation_service.check_image_prompt_relevance(validation_llm, request.prompt, result["content_text"], render_prompts):
                    still_mismatched_image_warning = (
                        "⚠️ The regenerated image may still not fully match your request. "
                        "Consider rephrasing your prompt with a more specific topic."
                    )
                    stream_warnings.append(still_mismatched_image_warning)
                    yield _sse("warning", {"message": still_mismatched_image_warning})

            # All 1-3 images render inline via generate_variations_events(),
            # which already runs each image on its own thread and yields
            # image_ready/image_failed the instant each one finishes — no
            # SQS, no second invocation. This whole generator function
            # still has to finish within API Gateway HTTP API's hard
            # ~29-30s integration timeout (it does not support true
            # response streaming for Lambda proxy integrations, so SSE
            # here only improves what the CLIENT sees once the Lambda
            # itself returns — it does not extend how long the Lambda is
            # allowed to run past that ceiling). With Freepik Mystic
            # (~90-130s/image) multi-image requests will not fit; use a
            # fast model (hyperflux, seedream-v5-lite) if you need 2-3
            # images through this endpoint.
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
            response.generation_id = generation_id
            yield _sse("done", response.model_dump())
            return

        except Exception as exc:  # noqa: BLE001 - last-resort so the stream always closes cleanly
            logger.exception("Unexpected error during streamed content generation.")
            yield _sse("error", {"message": f"Unexpected error: {exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )