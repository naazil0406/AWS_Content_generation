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
    load_content_generation_system_prompt,
    load_image_prompt_system_prompt,
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
        content_llm: Optional[BaseLLMService] = None,
        image_prompt_llm: Optional[BaseLLMService] = None,
        llm: Optional[BaseLLMService] = None,
    ):
        """`content_llm` generates the educational text; `image_prompt_llm`
        turns that text into a focused visual prompt for image models.
        """
        from app.services.llm_service import get_content_llm, get_image_prompt_llm

        self.knowledge_base = knowledge_base
        self.content_llm = content_llm or llm or get_content_llm()
        self.image_prompt_llm = image_prompt_llm or llm or get_image_prompt_llm()

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

        # 3. Parallel Execution: Run Content Generation LLM and Image Prompt LLM concurrently!
        # Both LLM calls fire simultaneously, cutting total LLM wait time from ~5s down to ~2.5s.
        from concurrent.futures import ThreadPoolExecutor

        mode = image_mode_for_content_type(content_type)
        content_sys_prompt = load_content_generation_system_prompt()
        img_sys_prompt = load_image_prompt_system_prompt()

        daily_tip_focus = pick_daily_tip_focus() if content_type == "Daily Tip" else None

        def _run_content():
            if content_type == "Daily Tip":
                return self._generate_daily_tip(
                    system_prompt=content_sys_prompt,
                    topic=topic,
                    context_chunks=context_chunks,
                    common_data=common_data,
                    web_results=web_results,
                    avoid_repeating=avoid_repeating,
                    focus=daily_tip_focus,
                )
            return self.content_llm.generate_content(
                system_prompt=content_sys_prompt,
                content_type=content_type,
                topic=topic,
                context_chunks=context_chunks,
                monthly_topic_content=monthly_topic_content,
                common_data=common_data,
                web_results=web_results,
                daily_tip_focus=daily_tip_focus,
                avoid_repeating=avoid_repeating,
            )

        def _run_image_prompt(content_hint: str):
            return self.image_prompt_llm.generate_image_prompt_package(
                system_prompt=img_sys_prompt,
                content_type=content_type,
                topic=topic,
                generated_content=content_hint,
                context_chunks=context_chunks,
                mode=mode,
            )

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                f_content = executor.submit(_run_content)
                f_image = executor.submit(_run_image_prompt, topic)

                content_text = f_content.result()
                image_package = f_image.result()
        except Exception as exc:
            logger.error("Parallel LLM generation failed: %s", exc)
            raise ContentGenerationError("Failed to generate content or image prompt in parallel.") from exc

        if not content_text.strip():
            raise ContentGenerationError("Generated content was empty.")

        if not image_package.get("image_prompt", "").strip():
            raise ContentGenerationError("Generated image prompt was empty.")

        # Fall back to mode-specific fixed negative prompt if model returned an empty string.
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