"""
LLM Service — abstraction over the Content Generation Agent and the
Image Prompt Generation Agent, backed by AWS Bedrock.

BaseLLMService defines the two operations the rest of the application
needs. The concrete provider only has to implement `_call_llm(system, user)
-> str` — everything else (prompt assembly, output cleanup, validation)
is shared here.

One transport is included: BedrockLLMService, over AWS Bedrock's
Converse API. It works across every Bedrock-hosted model family (Nova,
Claude, Llama, Mistral, ...) without a code change — only the model ID
(settings.CONTENT_MODEL / settings.IMAGE_PROMPT_MODEL) changes.
"""

import json
import logging
import re
from typing import List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:\w+)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_prompt_entries(data: dict, keep_validation_fields: bool = False) -> List[dict]:
    """Parse the `image_prompts` field of a drafting- or validator-agent
    JSON response into a list of dicts, each with at minimum
    "content_anchor", "visual_concept", and "prompt" — the traceability
    structure required by prompts/image_prompt_system.txt's OUTPUT FORMAT
    (see also image_prompt_validator_system.txt).

    Backward-compatible with two older shapes a less-cooperative model
    (or a not-yet-updated prompt) might still return:
      - a plain list of strings (the pre-content_anchor schema) — wrapped
        with empty content_anchor/visual_concept so callers can rely on
        the keys always being present.
      - the even older singular "image_prompt" string field.
    Entries missing a non-empty "prompt" are dropped rather than crashing
    the whole request on one bad entry.

    keep_validation_fields: when True (used for the validator agent's
    response), also carries through "passed"/"reason" if present.
    """
    raw_entries = data.get("image_prompts")
    if not isinstance(raw_entries, list):
        single = (data.get("image_prompt") or "").strip()
        raw_entries = [single] if single else []

    parsed: List[dict] = []
    for item in raw_entries:
        if isinstance(item, str):
            prompt_text = item.strip()
            if not prompt_text:
                continue
            entry = {"content_anchor": "", "visual_concept": "", "prompt": prompt_text}
        elif isinstance(item, dict):
            prompt_text = (item.get("prompt") or "").strip()
            if not prompt_text:
                continue
            entry = {
                "content_anchor": (item.get("content_anchor") or "").strip(),
                "visual_concept": (item.get("visual_concept") or "").strip(),
                "prompt": prompt_text,
            }
            if keep_validation_fields:
                entry["passed"] = bool(item.get("passed", True))
                entry["reason"] = (item.get("reason") or "").strip()
        else:
            continue
        parsed.append(entry)
    return parsed


class BaseLLMService:
    """Base class for any LLM transport.

    A subclass only needs to implement _call_llm(system_prompt,
    user_prompt) -> str.
    """

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError

    def generate_content(
        self,
        system_prompt: str,
        content_type: str,
        topic: str,
        context_chunks: List[dict],
        industry: Optional[str] = None,
        monthly_topic_content: Optional[str] = None,
        common_data: Optional[str] = None,
        web_results: Optional[str] = None,
        daily_tip_focus: Optional[tuple] = None,
        avoid_repeating: Optional[List[str]] = None,
    ) -> str:
        """Content Generation Agent: turn (Content Type + Topic +
        retrieved Knowledge Base context) into one short, original piece
        of content, per prompts/content_generation_system.txt.

        industry / daily_tip_focus / avoid_repeating: see
        prompt_builder.build_content_generation_prompt() for what each
        does — passed straight through here.
        """
        from app.services.prompt_builder import build_content_generation_prompt

        user_prompt = build_content_generation_prompt(
            content_type=content_type,
            topic=topic,
            context_chunks=context_chunks,
            industry=industry,
            monthly_topic_content=monthly_topic_content,
            common_data=common_data,
            web_results=web_results,
            daily_tip_focus=daily_tip_focus,
            avoid_repeating=avoid_repeating,
        )
        raw = self._call_llm(system_prompt, user_prompt).strip()
        content_text = _strip_code_fences(raw)
        content_text = re.sub(
            rf"^{re.escape(content_type)}\s*:\s*", "", content_text, flags=re.IGNORECASE
        )
        content_text = content_text.strip().strip('"').strip()

        if not content_text:
            raise RuntimeError(
                f"The model returned no content for '{content_type}' on topic '{topic}'."
            )
        return content_text

    def generate_combined_package(
        self,
        system_prompt: str,
        content_type: str,
        topic: str,
        context_chunks: List[dict],
        mode: str,
        monthly_topic_content: Optional[str] = None,
        common_data: Optional[str] = None,
        web_results: Optional[str] = None,
        daily_tip_focus: Optional[tuple] = None,
        avoid_repeating: Optional[List[str]] = None,
    ) -> dict:
        """Merged Content Generation Agent + Image Prompt Generation Agent
        call: one LLM round trip instead of two, since the second agent's
        only input beyond static instructions is the first agent's output
        — there's no reason that has to be two separate network calls.
        Returns a dict with keys: content_text, image_prompt,
        negative_prompt, alt_text, tags, summary.
        """
        from app.services.prompt_builder import build_combined_user_prompt

        user_prompt = build_combined_user_prompt(
            content_type=content_type,
            topic=topic,
            context_chunks=context_chunks,
            mode=mode,
            monthly_topic_content=monthly_topic_content,
            common_data=common_data,
            web_results=web_results,
            daily_tip_focus=daily_tip_focus,
            avoid_repeating=avoid_repeating,
        )
        raw = _strip_code_fences(self._call_llm(system_prompt, user_prompt).strip())

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Combined content+image generation did not return valid JSON: {exc}\n"
                f"Raw response: {raw[:500]}"
            ) from exc

        content_text = (data.get("content_text") or "").strip().strip('"').strip()
        content_text = re.sub(
            rf"^{re.escape(content_type)}\s*:\s*", "", content_text, flags=re.IGNORECASE
        )
        if not content_text:
            raise RuntimeError(
                f"The model returned no content_text for '{content_type}' on topic '{topic}'."
            )

        image_prompt = (data.get("image_prompt") or "").strip()
        if not image_prompt:
            raise RuntimeError("Combined generation returned an empty image_prompt.")

        return {
            "content_text": content_text,
            "image_prompt": image_prompt,
            "negative_prompt": (data.get("negative_prompt") or "").strip(),
            "alt_text": (data.get("alt_text") or "").strip(),
            "tags": data.get("tags") or [],
            "summary": (data.get("summary") or "").strip(),
        }

    def generate_image_prompt_package(
        self,
        system_prompt: str,
        content_type: str,
        topic: str,
        generated_content: str,
        context_chunks: List[dict],
        mode: str,
        industry: Optional[str] = None,
    ) -> dict:
        """Image Prompt Generation Agent: turn the already-generated
        content into THREE distinct, content-anchored image prompts (see
        OUTPUT FORMAT in prompts/image_prompt_system.txt) + a shared
        negative_prompt + alt_text + tags + summary. Returns a dict with:
          image_prompts: List[str] — exactly 3 (or however many the model
            returned; validated non-empty below) prompt strings, one
            image rendered per entry — this is what Freepik consumes.
          prompt_entries: List[dict] — the same 3, each as
            {"content_anchor", "visual_concept", "prompt"} — the
            traceability data (which piece of the Generated Content each
            image represents) fed into the validation pass below.
          negative_prompt, alt_text, tags, summary — same as before,
            shared across all three images.

        industry: the industry resolved ONCE for this request by
          ContentGenerationEngine.generate() — see
          build_image_prompt_request(). Always authoritative when
          present; identical to what generate_content() above received
          for this same request.
        """
        from app.services.prompt_builder import build_image_prompt_request

        user_prompt = build_image_prompt_request(
            content_type=content_type,
            topic=topic,
            generated_content=generated_content,
            context_chunks=context_chunks,
            mode=mode,
            industry=industry,
        )
        raw = _strip_code_fences(self._call_llm(system_prompt, user_prompt).strip())

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Image prompt generation did not return valid JSON: {exc}\n"
                f"Raw response: {raw[:500]}"
            ) from exc

        prompt_entries = _parse_prompt_entries(data)
        if not prompt_entries:
            raise RuntimeError("Image prompt generation returned no non-empty image_prompts.")

        return {
            "image_prompts": [e["prompt"] for e in prompt_entries],
            "prompt_entries": prompt_entries,
            "negative_prompt": (data.get("negative_prompt") or "").strip(),
            "alt_text": (data.get("alt_text") or "").strip(),
            "tags": data.get("tags") or [],
            "summary": (data.get("summary") or "").strip(),
        }

    def validate_image_prompt_package(
        self,
        system_prompt: str,
        content_type: str,
        mode: str,
        generated_content: str,
        prompt_entries: List[dict],
        negative_prompt: str,
        alt_text: str,
        tags: List[str],
        summary: str,
        industry: Optional[str] = None,
    ) -> dict:
        """Image Prompt Validator: the second-pass QA step (see
        prompts/image_prompt_validator_system.txt) that checks each of
        the drafting agent's three prompt_entries against the Generated
        Content — content grounding, semantic match, industry
        consistency, specificity, the Generic Image Test, the Content
        Type Test, and distinctness — and rewrites any that fail, BEFORE
        anything is sent to Freepik. Returns the same shape as
        generate_image_prompt_package() above (image_prompts,
        prompt_entries, negative_prompt, alt_text, tags, summary), plus
        prompt_entries carrying "passed"/"reason" for logging.

        This call reuses whichever LLM instance it's invoked on (the
        same image_prompt_llm the drafting call used) — no separate
        model or external service is introduced.
        """
        from app.services.prompt_builder import build_image_prompt_validation_request

        user_prompt = build_image_prompt_validation_request(
            content_type=content_type,
            mode=mode,
            generated_content=generated_content,
            draft_entries=prompt_entries,
            negative_prompt=negative_prompt,
            alt_text=alt_text,
            tags=tags,
            summary=summary,
            industry=industry,
        )
        raw = _strip_code_fences(self._call_llm(system_prompt, user_prompt).strip())

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Image prompt validation did not return valid JSON: {exc}\n"
                f"Raw response: {raw[:500]}"
            ) from exc

        validated_entries = _parse_prompt_entries(data, keep_validation_fields=True)
        if not validated_entries:
            raise RuntimeError("Image prompt validation returned no non-empty image_prompts.")

        return {
            "image_prompts": [e["prompt"] for e in validated_entries],
            "prompt_entries": validated_entries,
            "negative_prompt": (data.get("negative_prompt") or "").strip() or negative_prompt,
            "alt_text": (data.get("alt_text") or "").strip() or alt_text,
            "tags": data.get("tags") or tags,
            "summary": (data.get("summary") or "").strip() or summary,
        }


class BedrockLLMService(BaseLLMService):
    """LLM transport over AWS Bedrock's unified Converse API. Works
    across every Bedrock-hosted model family (Nova, Claude, Llama,
    Mistral, ...) without a code change — only the model ID changes.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int,
        temperature: float,
        region_name: str,
        aws_access_key_id: str = "",
        aws_secret_access_key: str = "",
    ):
        import boto3

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.client = boto3.client("bedrock-runtime", region_name=region_name)
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        from botocore.exceptions import BotoCoreError, ClientError

        logger.info(
            "Calling Bedrock model '%s' (system=%d chars, user=%d chars).",
            self.model, len(system_prompt), len(user_prompt),
        )
        try:
            response = self.client.converse(
                modelId=self.model,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                inferenceConfig={
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
        except (ClientError, BotoCoreError) as exc:
            raise RuntimeError(f"Bedrock inference failed for model '{self.model}': {exc}") from exc

        content_blocks = response.get("output", {}).get("message", {}).get("content", [])
        text = "".join(block.get("text", "") for block in content_blocks).strip()
        if not text:
            raise RuntimeError("Bedrock returned an empty response.")
        return text


def get_content_llm() -> BaseLLMService:
    """Factory for the Content Generation Agent's LLM (AWS Bedrock)."""
    return BedrockLLMService(
        model=settings.CONTENT_MODEL,
        max_tokens=settings.CONTENT_MAX_TOKENS,
        temperature=settings.CONTENT_TEMPERATURE,
        region_name=settings.BEDROCK_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )


def get_combined_llm() -> BaseLLMService:
    """Factory for the merged content+image-prompt call. Uses
    IMAGE_PROMPT_MODEL (Nova Lite by default) rather than CONTENT_MODEL
    (Nova Micro), since the merged task is strictly harder than either
    original task alone — it has to do both jobs correctly in one pass —
    and Nova Lite is the more capable of the two. max_tokens is the sum
    of both original budgets since the response now contains both
    payloads.
    """
    return BedrockLLMService(
        model=settings.IMAGE_PROMPT_MODEL,
        max_tokens=settings.CONTENT_MAX_TOKENS + settings.IMAGE_PROMPT_MAX_TOKENS,
        temperature=settings.IMAGE_PROMPT_TEMPERATURE,
        region_name=settings.BEDROCK_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )


def get_image_prompt_llm() -> BaseLLMService:
    """Factory for the Image Prompt Generation Agent's LLM (AWS Bedrock)."""
    return BedrockLLMService(
        model=settings.IMAGE_PROMPT_MODEL,
        max_tokens=settings.IMAGE_PROMPT_MAX_TOKENS,
        temperature=settings.IMAGE_PROMPT_TEMPERATURE,
        region_name=settings.BEDROCK_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )