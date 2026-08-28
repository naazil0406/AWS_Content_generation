"""
Content Generation Engine — the core business logic of this service.

Responsibilities (per the application spec):
  1. Understand user intent (content_type + prompt/topic).
  2. Analyze retrieved Knowledge Base context.
  3. Generate an optimized LLM prompt (see prompt_builder.py) and call
     the Content Generation Agent.
  4. Generate an optimized image prompt from that content, via the Image
     Prompt Generation Agent.
  5. Validate output at each stage.
  6. Return structured content ready for the Response Builder.

This module is intentionally free of FastAPI, AWS, and I/O-provider
details — it only depends on the KnowledgeBaseService and BaseLLMService
interfaces, which are injected in, so it stays easy to test and reuse.
"""

import logging
from typing import List, Optional

from app.config import settings
from app.schemas.content import CONTENT_TYPES
from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.llm_service import BaseLLMService
from app.services.prompt_builder import (
    build_combined_system_prompt,
    detect_explicit_industry,
    image_mode_for_content_type,
    infographic_prompt_has_literal_text,
    infographic_prompt_is_text_dense,
    load_content_generation_system_prompt,
    load_image_prompt_system_prompt,
    load_image_prompt_validator_system_prompt,
    negative_prompt_for_mode,
    pick_daily_tip_focus,
    pick_industry,
)

logger = logging.getLogger(__name__)

# Daily Tip's word-count contract, per prompts/content_generation_system.txt's
# "DAILY TIP" section ("100-180 words. Not a range to pad toward — say
# the one useful thing and stop."). If the model misses the range, retry
# up to this many times before falling back to the closest attempt.
_DAILY_TIP_MIN_WORDS = 100
_DAILY_TIP_MAX_WORDS = 180
# Was 3 — each retry is a full sequential LLM round trip, which is too
# costly against a <10s total budget. One attempt only; if the model
# misses the word-count range, we accept the closest attempt rather than
# pay for a retry. See _generate_daily_tip() below.
_DAILY_TIP_MAX_ATTEMPTS = 1


class ContentGenerationError(RuntimeError):
    """Raised when the engine cannot produce valid content or an image prompt."""


class ContentGenerationEngine:
    """Orchestrates Knowledge Base retrieval, content generation, and
    image-prompt generation into one structured result.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBaseService,
        content_llm: Optional[BaseLLMService] = None,
        image_prompt_llm: Optional[BaseLLMService] = None,
        llm: Optional[BaseLLMService] = None,
        combined_llm: Optional[BaseLLMService] = None,
    ):
        """`content_llm` generates the educational text; `image_prompt_llm`
        turns that text into a focused visual prompt for image models.
        `combined_llm` is only ever used when settings.USE_COMBINED_GENERATION_CALL
        is true (see generate() below) — constructed lazily so nothing
        changes for the default configuration where it's never touched.
        """
        from app.services.llm_service import get_content_llm, get_image_prompt_llm

        self.knowledge_base = knowledge_base
        self.content_llm = content_llm or llm or get_content_llm()
        self.image_prompt_llm = image_prompt_llm or llm or get_image_prompt_llm()
        self._combined_llm = combined_llm or llm

    def generate(
        self,
        prompt: str,
        content_type: str,
        industry: Optional[str] = None,
        generate_images: bool = True,
        monthly_topic_content: Optional[str] = None,
        common_data: Optional[str] = None,
        web_results: Optional[str] = None,
        avoid_repeating: Optional[List[str]] = None,
        previous_industry: Optional[str] = None,
        image_count: int = 3,
    ) -> dict:
        """Run the engine pipeline for one request.

        image_count: exactly how many image_prompts the Image Prompt
          Agent should produce (1, 2, or 3 — see
          GenerateContentRequest.image_count in app/schemas/content.py).
          Ignored entirely when generate_images is False. Passed straight
          through to whichever image-prompt-generation path runs
          (combined or separate — see below); the Agent drafts exactly
          this many prompts from the start rather than always drafting 3
          and having the unused ones discarded downstream by
          app/routes/content.py's resolve_prompts_for_count(), which
          remains in place afterward purely as a defensive normalizer in
          case the model doesn't comply with the requested count.

        industry: an explicit industry to use for this request, if the
          caller has one (e.g. "Manufacturing"). If omitted, the engine
          randomly selects exactly ONE industry from the single-source
          list in prompts/image_prompt_system.txt (see
          prompt_builder.pick_industry()) — never hardcoded to one
          industry, and never re-picked per call within this request.
          Either way, the resolved value (see `selected_industry` in the
          returned dict) is passed IDENTICALLY to the Content Generation
          Agent and — when generate_images is True — the Image Prompt
          Agent, so the generated story and all three images share one
          consistent professional context. Neither agent ever picks its
          own industry independently.

        generate_images: True runs the full pipeline (content, then the
          Image Prompt Agent for exactly three grounded prompts). False
          (the "Generate" action — content only) stops right after
          content generation and never calls the Image Prompt Agent at
          all — not even to discard its output — per the "Generate must
          be content-only" requirement. The randomly-selected industry
          still applies to the content in this case; only the image half
          of the pipeline is skipped.

        avoid_repeating: previously-generated content for this same
          (content_type, prompt) pair, e.g. supplied by a caller that
          keeps its own generation history. The model is told not to
          repeat or lightly reword any of these. Optional — a one-off
          generation doesn't need it.

        previous_industry: the industry used for this same caller's
          immediately preceding piece of content, if the caller tracks
          it. Used only to steer the random pick away from repeating the
          same industry twice in a row when no explicit `industry` is
          given — never consulted when `industry` above is given.
          Optional — a one-off generation doesn't need it.

        Returns a dict with keys: content_text, image_prompts,
        content_anchors, visual_concepts, negative_prompt, alt_text,
        tags, summary, context_chunks, daily_tip_focus,
        selected_industry. `selected_industry` is always populated — the
        single industry (explicit or randomly chosen) used throughout
        this request. `content_anchors`/`visual_concepts` are the
        traceability data (which phrase/clause of content_text each
        image_prompts entry represents) produced by the drafting call and
        checked/corrected by the validation pass below — see step 4.
        When generate_images is False, image_prompts and friends are
        empty/blank — the Image Prompt Agent was never called.
        """
        topic = (prompt or "").strip()
        if not topic:
            raise ValueError("prompt must not be empty.")
        if content_type not in CONTENT_TYPES:
            raise ValueError(
                f"Unsupported content_type '{content_type}'. Supported types: {', '.join(CONTENT_TYPES)}"
            )
        industry = (industry or "").strip() or None
        if image_count not in (1, 2, 3):
            image_count = 3

        # 1 & 2: retrieve and analyze Knowledge Base context.
        try:
            context_chunks: List[dict] = self.knowledge_base.retrieve(topic)
        except Exception as exc:
            logger.error("Knowledge Base retrieval failed: %s", exc)
            raise ContentGenerationError("Failed to retrieve context from the Knowledge Base.") from exc

        # 3. Sequential Execution: the Image Prompt Generation Agent's whole
        # job is to turn the ALREADY-GENERATED content into an image prompt
        # (see prompts/image_prompt_system.txt — "content is the image's
        # single source of truth"). That means it must run AFTER content
        # generation completes and must receive content_text, not the raw
        # topic. Running these two calls in parallel (as a prior version of
        # this method did, passing `topic` into the image-prompt call) broke
        # that contract: the image prompt ended up generated blind from a
        # one-line topic instead of from what the content actually says,
        # producing images that don't match the generated content.
        #
        # settings.USE_COMBINED_GENERATION_CALL (default False) offers a
        # middle ground instead of reintroducing that parallelism: when
        # true, content generation AND image-prompt drafting happen in ONE
        # LLM call (generate_combined_package(), see llm_service.py) rather
        # than two sequential ones — content_text is still produced first,
        # within that same call, and the drafted image_prompts are still
        # required to be derived from it, just without a second network
        # round trip. The dedicated validator step below (step 4) still
        # runs identically afterward either way, so combined mode saves
        # one of three model calls, not the rigor a validator pass adds.
        # Daily Tip never uses combined mode — _generate_daily_tip()'s
        # word-count retry contract doesn't fit the combined JSON format.
        mode = image_mode_for_content_type(content_type)
        content_sys_prompt = load_content_generation_system_prompt()
        img_sys_prompt = load_image_prompt_system_prompt()
        # Content Type = "Infographic" now uses the SAME two prompt files
        # as every other Content Type — image_prompt_system.txt's own
        # "MODE: infographic" section (which requires literal, exact
        # in-image text) and image_prompt_validator_system.txt's checks
        # J/M enforce this, no separate file involved. `is_infographic`
        # is kept here only to drive the code-level literal-text safety
        # net below (see infographic_prompt_has_literal_text()) — a
        # defensive re-check that doesn't rely solely on the LLM having
        # followed those instructions.
        is_infographic = content_type == "Infographic"

        daily_tip_focus = pick_daily_tip_focus() if content_type == "Daily Tip" else None
        # Resolve exactly ONE industry for this entire request, once,
        # right here — before either agent runs. Both agents below
        # receive this same `selected_industry` value; neither is ever
        # allowed to pick its own.
        #
        # Priority, matching "explicit always wins, random is a last
        # resort and should still be relevant":
        #   1. `industry` — an explicit value the caller passed directly
        #      (kept for backward compatibility with any caller that
        #      still sets this; the current UI does not, since its
        #      Industry dropdown was removed and it always sends None).
        #   2. detect_explicit_industry(topic) — the user named a
        #      specific industry (or a recognized alias, e.g. "hospital"
        #      for Healthcare) directly in their free-text topic. This
        #      is the fix: previously nothing ever checked the topic
        #      text itself, so an explicit mention there was silently
        #      ignored and a random industry was picked anyway on every
        #      single request.
        #   3. pick_industry(topic, avoid=...) — no explicit mention
        #      anywhere; pick randomly, but biased toward industries
        #      whose contextual hints (objects/roles/hazards) actually
        #      appear in the topic, so the random pick is topic-relevant
        #      rather than arbitrary — see prompt_builder.py.
        selected_industry = (
            industry
            or detect_explicit_industry(topic)
            or pick_industry(topic=topic, avoid=previous_industry)
        )

        use_combined = (
            settings.USE_COMBINED_GENERATION_CALL
            and generate_images
            and content_type != "Daily Tip"
        )
        image_package: Optional[dict] = None  # populated by the combined call below, if used

        try:
            if content_type == "Daily Tip":
                content_text = self._generate_daily_tip(
                    system_prompt=content_sys_prompt,
                    topic=topic,
                    context_chunks=context_chunks,
                    common_data=common_data,
                    web_results=web_results,
                    avoid_repeating=avoid_repeating,
                    focus=daily_tip_focus,
                )
            elif use_combined:
                from app.services.llm_service import get_combined_llm
                from app.services.prompt_builder import build_combined_system_prompt

                combined_llm = self._combined_llm or get_combined_llm()
                combined_result = combined_llm.generate_combined_package(
                    system_prompt=build_combined_system_prompt(),
                    content_type=content_type,
                    topic=topic,
                    context_chunks=context_chunks,
                    mode=mode,
                    industry=selected_industry,
                    monthly_topic_content=monthly_topic_content,
                    common_data=common_data,
                    web_results=web_results,
                    daily_tip_focus=daily_tip_focus,
                    avoid_repeating=avoid_repeating,
                    image_count=image_count,
                )
                content_text = combined_result["content_text"]
                # Same shape generate_image_prompt_package() would have
                # returned — the validator step below treats this
                # identically regardless of which path produced it.
                image_package = {
                    "image_prompts": combined_result["image_prompts"],
                    "prompt_entries": combined_result["prompt_entries"],
                    "negative_prompt": combined_result["negative_prompt"],
                    "alt_text": combined_result["alt_text"],
                    "tags": combined_result["tags"],
                    "summary": combined_result["summary"],
                }
            else:
                content_text = self.content_llm.generate_content(
                    system_prompt=content_sys_prompt,
                    content_type=content_type,
                    topic=topic,
                    context_chunks=context_chunks,
                    industry=selected_industry,
                    monthly_topic_content=monthly_topic_content,
                    common_data=common_data,
                    web_results=web_results,
                    daily_tip_focus=daily_tip_focus,
                    avoid_repeating=avoid_repeating,
                )
        except Exception as exc:
            logger.error("Content generation failed: %s", exc)
            raise ContentGenerationError("Failed to generate content.") from exc

        if not content_text.strip():
            raise ContentGenerationError("Generated content was empty.")

        if not generate_images:
            # "Generate" action: content-only. The Image Prompt Agent must
            # never be called here — not Freepik, not S3, not even a
            # discarded call — per the content-only requirement. The
            # selected industry still applied to the content above.
            return {
                "content_text": content_text,
                "image_prompts": [],
                "content_anchors": [],
                "visual_concepts": [],
                "negative_prompt": "",
                "alt_text": "",
                "tags": [],
                "summary": "",
                "context_chunks": context_chunks,
                "daily_tip_focus": daily_tip_focus,
                "selected_industry": selected_industry,
            }

        if image_package is None:
            # Combined mode wasn't used (or wasn't eligible) — run the
            # separate Image Prompt Agent call, exactly as before.
            try:
                image_package = self.image_prompt_llm.generate_image_prompt_package(
                    system_prompt=img_sys_prompt,
                    content_type=content_type,
                    topic=topic,
                    generated_content=content_text,
                    context_chunks=context_chunks,
                    mode=mode,
                    industry=selected_industry,
                    image_count=image_count,
                )
            except Exception as exc:
                logger.error("Image prompt generation failed: %s", exc)
                raise ContentGenerationError("Failed to generate image prompt from content.") from exc

        if not image_package.get("image_prompts"):
            raise ContentGenerationError("Generated image prompt list was empty.")

        # 4. Second-pass validation — BEFORE anything reaches Freepik.
        # The drafting call above can still produce a prompt that's
        # merely industry-flavored (a farm glove, a boot, a wrench)
        # rather than genuinely grounded in content_text; this is exactly
        # the failure mode prompts/image_prompt_validator_system.txt
        # exists to catch and fix. Reuses the same image_prompt_llm
        # instance — no new model or external service. Best-effort: if
        # the validator call itself fails or returns something
        # unusable, log it and fall back to the (still individually
        # instructed-to-be-grounded) drafts rather than failing the
        # whole request over a QA-step hiccup.
        prompt_entries = image_package.get("prompt_entries") or [
            {"content_anchor": "", "visual_concept": "", "prompt": p} for p in image_package["image_prompts"]
        ]
        try:
            validator_sys_prompt = load_image_prompt_validator_system_prompt()
            validated_package = self.image_prompt_llm.validate_image_prompt_package(
                system_prompt=validator_sys_prompt,
                content_type=content_type,
                topic=topic,
                mode=mode,
                generated_content=content_text,
                prompt_entries=prompt_entries,
                negative_prompt=image_package.get("negative_prompt", ""),
                alt_text=image_package.get("alt_text", ""),
                tags=image_package.get("tags", []),
                summary=image_package.get("summary", ""),
                industry=selected_industry,
            )
            if validated_package.get("image_prompts"):
                for entry in validated_package.get("prompt_entries", []):
                    if not entry.get("passed", True):
                        logger.info(
                            "Image prompt validator corrected an entry (anchor=%r): %s",
                            entry.get("content_anchor", ""), entry.get("reason", ""),
                        )
                image_package = validated_package
        except Exception as exc:  # noqa: BLE001 - validation is best-effort, never fatal
            logger.warning("Image prompt validation step failed, using unvalidated drafts: %s", exc)

        prompt_entries = image_package.get("prompt_entries") or prompt_entries

        # Code-level safety net for Infographic requests only: the LLM-
        # side instructions (infographic_image_prompt_system.txt /
        # infographic_image_prompt_validator_system.txt) are the primary
        # enforcement of "literal in-image text is required" and "text
        # density stays low enough to render legibly," but the
        # application should not blindly trust an LLM followed them.
        # If ANY of the three final prompts still has no quoted literal
        # text, OR has MORE quoted strings than the TEXT DENSITY LIMIT
        # allows (a busy layout is exactly what garbles legibility, per
        # the production issue this check was added for), do exactly one
        # full regeneration (draft again, validate again) before falling
        # back — never send a textless or text-overloaded prompt to
        # Freepik for an Infographic request if a retry can fix it.
        if is_infographic and generate_images:
            def _bad_infographic_prompts(prompts):
                return [
                    p for p in prompts
                    if not infographic_prompt_has_literal_text(p)
                    or infographic_prompt_is_text_dense(p)
                ]

            bad_prompts = _bad_infographic_prompts(image_package.get("image_prompts", []))
            if bad_prompts:
                logger.warning(
                    "Infographic image prompt(s) missing literal text or over the "
                    "text-density limit after drafting/validation (%d of %d) — retrying once.",
                    len(bad_prompts), len(image_package.get("image_prompts", [])),
                )
                try:
                    retry_package = self.image_prompt_llm.generate_image_prompt_package(
                        system_prompt=img_sys_prompt,
                        content_type=content_type,
                        topic=topic,
                        generated_content=content_text,
                        context_chunks=context_chunks,
                        mode=mode,
                        industry=selected_industry,
                        retry_hint=True,
                        image_count=image_count,
                    )
                    retry_entries = retry_package.get("prompt_entries") or [
                        {"content_anchor": "", "visual_concept": "", "prompt": p}
                        for p in retry_package.get("image_prompts", [])
                    ]
                    retry_validated = self.image_prompt_llm.validate_image_prompt_package(
                        system_prompt=validator_sys_prompt,
                        content_type=content_type,
                        topic=topic,
                        mode=mode,
                        generated_content=content_text,
                        prompt_entries=retry_entries,
                        negative_prompt=retry_package.get("negative_prompt", ""),
                        alt_text=retry_package.get("alt_text", ""),
                        tags=retry_package.get("tags", []),
                        summary=retry_package.get("summary", ""),
                        industry=selected_industry,
                    )
                    retry_prompts = retry_validated.get("image_prompts") or retry_package.get("image_prompts", [])
                    still_bad = _bad_infographic_prompts(retry_prompts)
                    if retry_prompts and len(still_bad) < len(bad_prompts):
                        # Strictly better (or fully fixed) — use the retry.
                        image_package = retry_validated if retry_validated.get("image_prompts") else retry_package
                        prompt_entries = image_package.get("prompt_entries") or prompt_entries
                        if still_bad:
                            logger.warning(
                                "Infographic retry improved but did not fully fix text "
                                "presence/density (%d of %d still bad) — proceeding with retry result.",
                                len(still_bad), len(retry_prompts),
                            )
                    else:
                        logger.error(
                            "Infographic retry did not improve text presence/density — "
                            "proceeding with the original (best-effort) drafts."
                        )
                except Exception as exc:  # noqa: BLE001 - retry is best-effort, never fatal
                    logger.warning("Infographic text-quality retry failed, using prior drafts: %s", exc)

        # Fall back to mode-specific fixed negative prompt if model returned an empty string.
        negative_prompt = image_package.get("negative_prompt", "").strip() or negative_prompt_for_mode(mode)

        return {
            "content_text": content_text,
            "image_prompts": image_package["image_prompts"],
            "content_anchors": [e.get("content_anchor", "") for e in prompt_entries],
            "visual_concepts": [e.get("visual_concept", "") for e in prompt_entries],
            "negative_prompt": negative_prompt,
            "alt_text": image_package.get("alt_text", ""),
            "tags": image_package.get("tags", []),
            "summary": image_package.get("summary", ""),
            "context_chunks": context_chunks,
            "daily_tip_focus": daily_tip_focus,
            "selected_industry": selected_industry,
        }

    def _generate_daily_tip(
        self,
        system_prompt: str,
        topic: str,
        context_chunks: List[dict],
        common_data: Optional[str],
        web_results: Optional[str],
        avoid_repeating: Optional[List[str]],
        focus,
    ) -> str:
        """Generate a Daily Tip content_text."""
        for attempt in range(1, _DAILY_TIP_MAX_ATTEMPTS + 1):
            content_text = self.content_llm.generate_content(
                system_prompt=system_prompt,
                content_type="Daily Tip",
                topic=topic,
                context_chunks=context_chunks,
                common_data=common_data,
                web_results=web_results,
                avoid_repeating=avoid_repeating,
                daily_tip_focus=focus,
            )
            word_count = len(content_text.split())
            if _DAILY_TIP_MIN_WORDS <= word_count <= _DAILY_TIP_MAX_WORDS:
                return content_text
            logger.warning(
                "Daily Tip attempt %d/%d was %d words (need %d-%d).",
                attempt, _DAILY_TIP_MAX_ATTEMPTS, word_count,
                _DAILY_TIP_MIN_WORDS, _DAILY_TIP_MAX_WORDS,
            )
        return content_text