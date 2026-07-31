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

from app.schemas.content import CONTENT_TYPES
from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.llm_service import BaseLLMService
from app.services.prompt_builder import (
    build_combined_system_prompt,
    image_mode_for_content_type,
    negative_prompt_for_mode,
    pick_daily_tip_focus,
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
        llm: BaseLLMService,
    ):
        """`llm` handles the merged content+image-prompt call (see
        BaseLLMService.generate_combined_package / llm_service.get_combined_llm).
        Content generation and image-prompt generation used to be two
        sequential network calls; they're now one, since the second
        agent's only real input beyond static instructions was the
        first agent's output.
        """
        self.knowledge_base = knowledge_base
        self.llm = llm

    def generate(
        self,
        prompt: str,
        content_type: str,
        monthly_topic_content: Optional[str] = None,
        common_data: Optional[str] = None,
        web_results: Optional[str] = None,
        avoid_repeating: Optional[List[str]] = None,
    ) -> dict:
        """Run the full engine pipeline for one request.

        avoid_repeating: previously-generated content for this same
          (content_type, prompt) pair, e.g. supplied by a caller that
          keeps its own generation history. The model is told not to
          repeat or lightly reword any of these. Optional — a one-off
          generation doesn't need it.

        Returns a dict with keys: content_text, image_prompt,
        negative_prompt, alt_text, tags, summary, context_chunks,
        daily_tip_focus.
        """
        topic = (prompt or "").strip()
        if not topic:
            raise ValueError("prompt must not be empty.")
        if content_type not in CONTENT_TYPES:
            raise ValueError(
                f"Unsupported content_type '{content_type}'. Supported types: {', '.join(CONTENT_TYPES)}"
            )

        # 1 & 2: retrieve and analyze Knowledge Base context.
        try:
            context_chunks: List[dict] = self.knowledge_base.retrieve(topic)
        except Exception as exc:
            logger.error("Knowledge Base retrieval failed: %s", exc)
            raise ContentGenerationError("Failed to retrieve context from the Knowledge Base.") from exc

        # 3 & 4: generate structured, learner-facing content AND the
        # image prompt derived from it, in one combined LLM call (see
        # prompt_builder.build_combined_system_prompt/build_combined_user_prompt
        # and BaseLLMService.generate_combined_package). The image
        # prompt is still always derived from the generated content
        # (never the raw topic alone) — that ordering happens within the
        # model's single response, not across two network calls.
        mode = image_mode_for_content_type(content_type)
        system_prompt = build_combined_system_prompt()

        # Daily Tip has its own contract: every request is anchored to
        # one randomly-chosen State/Error from the framework (see
        # prompts/content_generation_system.txt's "DAILY TIP" section).
        # Its 100-180 word range used to be enforced with retries; that's
        # now a single best-effort attempt (see _DAILY_TIP_MAX_ATTEMPTS)
        # to keep total latency bounded.
        daily_tip_focus = pick_daily_tip_focus() if content_type == "Daily Tip" else None

        try:
            if content_type == "Daily Tip":
                package = self._generate_daily_tip(
                    system_prompt=system_prompt,
                    topic=topic,
                    context_chunks=context_chunks,
                    mode=mode,
                    common_data=common_data,
                    web_results=web_results,
                    avoid_repeating=avoid_repeating,
                    focus=daily_tip_focus,
                )
            else:
                package = self.llm.generate_combined_package(
                    system_prompt=system_prompt,
                    content_type=content_type,
                    topic=topic,
                    context_chunks=context_chunks,
                    mode=mode,
                    monthly_topic_content=monthly_topic_content,
                    common_data=common_data,
                    web_results=web_results,
                    avoid_repeating=avoid_repeating,
                )
        except Exception as exc:
            logger.error("Combined content+image generation failed: %s", exc)
            raise ContentGenerationError("Failed to generate content and image prompt.") from exc

        content_text = package["content_text"]
        image_package = package

        # 5: validate.
        if not content_text.strip():
            raise ContentGenerationError("Generated content was empty.")
        if not image_package.get("image_prompt", "").strip():
            raise ContentGenerationError("Generated image prompt was empty.")

        # The LLM's negative_prompt field is sometimes empty even when it
        # returns a valid image_prompt; fall back to the mode-specific
        # fixed negative prompt rather than sending the image renderer no
        # negative prompt at all.
        negative_prompt = image_package.get("negative_prompt", "").strip() or negative_prompt_for_mode(mode)

        return {
            "content_text": content_text,
            "image_prompt": image_package["image_prompt"],
            "negative_prompt": negative_prompt,
            "alt_text": image_package.get("alt_text", ""),
            "tags": image_package.get("tags", []),
            "summary": image_package.get("summary", ""),
            "context_chunks": context_chunks,
            "daily_tip_focus": daily_tip_focus,
        }

    def _generate_daily_tip(
        self,
        system_prompt: str,
        topic: str,
        context_chunks: List[dict],
        mode: str,
        common_data: Optional[str],
        web_results: Optional[str],
        avoid_repeating: Optional[List[str]],
        focus,
    ) -> dict:
        """Generate a Daily Tip (content_text + image package together,
        via the combined call), up to _DAILY_TIP_MAX_ATTEMPTS times if
        content_text falls outside the 100-180 word range. Returns the
        closest attempt's full package if every attempt misses.
        _DAILY_TIP_MAX_ATTEMPTS is 1 by default — see its definition —
        so in practice this is a single best-effort call, not a retry
        loop; the loop is kept so raising the constant back up later
        (e.g. for a background/async job that isn't latency-constrained)
        is a one-line change.
        """
        last_package: Optional[dict] = None
        for attempt in range(1, _DAILY_TIP_MAX_ATTEMPTS + 1):
            last_package = self.llm.generate_combined_package(
                system_prompt=system_prompt,
                content_type="Daily Tip",
                topic=topic,
                context_chunks=context_chunks,
                mode=mode,
                common_data=common_data,
                web_results=web_results,
                avoid_repeating=avoid_repeating,
                daily_tip_focus=focus,
            )
            word_count = len(last_package["content_text"].split())
            if _DAILY_TIP_MIN_WORDS <= word_count <= _DAILY_TIP_MAX_WORDS:
                return last_package
            logger.warning(
                "Daily Tip attempt %d/%d was %d words (need %d-%d).",
                attempt, _DAILY_TIP_MAX_ATTEMPTS, word_count,
                _DAILY_TIP_MIN_WORDS, _DAILY_TIP_MAX_WORDS,
            )
        return last_package