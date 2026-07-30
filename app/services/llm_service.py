"""
LLM Service — provider-independent abstraction over the Content
Generation Agent and the Image Prompt Generation Agent.

BaseLLMService defines the two operations the rest of the application
needs. A concrete provider only has to implement `_call_llm(system, user)
-> str` — everything else (prompt assembly, output cleanup, validation)
is shared here, so switching providers is a configuration change
(settings.LLM_PROVIDER), never a business-logic change.

Two transports are included: OpenRouterLLMService (chat-completions API)
and BedrockLLMService (AWS Bedrock Converse API). Adding a third provider
means adding one more class with a `_call_llm` method — nothing else in
the application changes.
"""

import json
import logging
import re
from typing import List, Optional

import requests

from app.config import settings

logger = logging.getLogger(__name__)

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:\w+)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class BaseLLMService:
    """Base class for any LLM transport (OpenRouter, Bedrock, ...).

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
        monthly_topic_content: Optional[str] = None,
        common_data: Optional[str] = None,
        web_results: Optional[str] = None,
        daily_tip_focus: Optional[tuple] = None,
        avoid_repeating: Optional[List[str]] = None,
    ) -> str:
        """Content Generation Agent: turn (Content Type + Topic +
        retrieved Knowledge Base context) into one short, original piece
        of content, per prompts/content_generation_system.txt.

        daily_tip_focus / avoid_repeating: see
        prompt_builder.build_content_generation_prompt() for what each
        does — passed straight through here.
        """
        from app.services.prompt_builder import build_content_generation_prompt

        user_prompt = build_content_generation_prompt(
            content_type=content_type,
            topic=topic,
            context_chunks=context_chunks,
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

    def generate_image_prompt_package(
        self,
        system_prompt: str,
        content_type: str,
        topic: str,
        generated_content: str,
        context_chunks: List[dict],
        mode: str,
    ) -> dict:
        """Image Prompt Generation Agent: turn the already-generated
        content into an image prompt + negative_prompt + alt_text + tags
        + summary, per prompts/image_prompt_system.txt. Returns a dict
        with those five keys.
        """
        from app.services.prompt_builder import build_image_prompt_request

        user_prompt = build_image_prompt_request(
            content_type=content_type,
            topic=topic,
            generated_content=generated_content,
            context_chunks=context_chunks,
            mode=mode,
        )
        raw = _strip_code_fences(self._call_llm(system_prompt, user_prompt).strip())

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Image prompt generation did not return valid JSON: {exc}\n"
                f"Raw response: {raw[:500]}"
            ) from exc

        image_prompt = (data.get("image_prompt") or "").strip()
        if not image_prompt:
            raise RuntimeError("Image prompt generation returned an empty image_prompt.")

        return {
            "image_prompt": image_prompt,
            "negative_prompt": (data.get("negative_prompt") or "").strip(),
            "alt_text": (data.get("alt_text") or "").strip(),
            "tags": data.get("tags") or [],
            "summary": (data.get("summary") or "").strip(),
        }


class OpenRouterLLMService(BaseLLMService):
    """LLM transport over OpenRouter's chat-completions API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int,
        temperature: float,
        base_url: str = OPENROUTER_CHAT_COMPLETIONS_URL,
        site_url: str = "",
        site_name: str = "",
        timeout: int = 90,
    ):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required when LLM_PROVIDER='openrouter'.")

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = base_url
        self.timeout = timeout

        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if site_url:
            self.headers["HTTP-Referer"] = site_url
        if site_name:
            self.headers["X-Title"] = site_name

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        logger.info(
            "Calling OpenRouter model '%s' (system=%d chars, user=%d chars).",
            self.model, len(system_prompt), len(user_prompt),
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        response = requests.post(
            self.base_url, headers=self.headers, json=payload, timeout=self.timeout
        )
        if not response.ok:
            raise RuntimeError(f"{response.status_code} error from OpenRouter: {response.text[:500]}")
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("OpenRouter returned no choices.")
        content = choices[0].get("message", {}).get("content", "").strip()
        if not content:
            raise RuntimeError("OpenRouter returned an empty response.")
        return content


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
        import boto3  # local import: keeps boto3 optional if only OpenRouter is used

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        client_kwargs = {"region_name": region_name}
        if aws_access_key_id and aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key

        self.client = boto3.client("bedrock-runtime", **client_kwargs)

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
    """Factory for the Content Generation Agent's LLM. Provider chosen
    entirely by settings.LLM_PROVIDER — no code change needed to switch.
    """
    if settings.LLM_PROVIDER == "openrouter":
        return OpenRouterLLMService(
            api_key=settings.OPENROUTER_API_KEY,
            model=settings.OPENROUTER_MODEL,
            max_tokens=settings.CONTENT_MAX_TOKENS,
            temperature=settings.CONTENT_TEMPERATURE,
            site_url=settings.OPENROUTER_SITE_URL,
            site_name=settings.OPENROUTER_SITE_NAME,
        )
    return BedrockLLMService(
        model=settings.CONTENT_MODEL,
        max_tokens=settings.CONTENT_MAX_TOKENS,
        temperature=settings.CONTENT_TEMPERATURE,
        region_name=settings.BEDROCK_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )


def get_image_prompt_llm() -> BaseLLMService:
    """Factory for the Image Prompt Generation Agent's LLM. Same
    provider switch as get_content_llm(), independently configurable."""
    if settings.LLM_PROVIDER == "openrouter":
        return OpenRouterLLMService(
            api_key=settings.OPENROUTER_API_KEY,
            model=settings.OPENROUTER_MODEL,
            max_tokens=settings.IMAGE_PROMPT_MAX_TOKENS,
            temperature=settings.IMAGE_PROMPT_TEMPERATURE,
            site_url=settings.OPENROUTER_SITE_URL,
            site_name=settings.OPENROUTER_SITE_NAME,
        )
    return BedrockLLMService(
        model=settings.IMAGE_PROMPT_MODEL,
        max_tokens=settings.IMAGE_PROMPT_MAX_TOKENS,
        temperature=settings.IMAGE_PROMPT_TEMPERATURE,
        region_name=settings.BEDROCK_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )
