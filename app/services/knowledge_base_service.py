"""
Knowledge Base Service — a thin client over the existing, externally
managed Knowledge Base.

This service does NOT chunk, embed, or index anything. It does not talk
to a vector database. All of that already exists elsewhere and is
someone else's operational responsibility. This module's entire job is:
take a query string, call the Knowledge Base's retrieval endpoint, and
hand back whatever context chunks it returns.

If your Knowledge Base exposes a different contract (a different path,
a different response shape, or an SDK instead of raw HTTP), change only
this file — nothing else in the application needs to know how retrieval
actually happens.
"""

import logging
import time
from typing import List, Optional

import requests

from app.config import settings

logger = logging.getLogger(__name__)


class KnowledgeBaseError(RuntimeError):
    """Raised when the Knowledge Base cannot be reached or returns an
    unusable response."""


class KnowledgeBaseService:
    """Retrieves relevant context chunks from the external Knowledge Base.

    Expects the Knowledge Base to expose a JSON retrieval endpoint that
    accepts {"query": str, "top_k": int} and returns
    {"chunks": [{"text": str, "source": str, ...}, ...]}. Adjust
    `_build_request` / `_parse_response` below if your Knowledge Base's
    contract differs — the rest of the app only depends on `retrieve()`.
    """

    def __init__(
        self,
        base_url: str = settings.KNOWLEDGE_BASE_URL,
        api_key: str = settings.KNOWLEDGE_BASE_API_KEY,
        top_k: int = settings.KNOWLEDGE_BASE_TOP_K,
        timeout: int = settings.KNOWLEDGE_BASE_TIMEOUT,
        max_retries: int = settings.HTTP_MAX_RETRIES,
        backoff_seconds: float = settings.HTTP_BACKOFF_SECONDS,
    ):
        if not base_url:
            raise KnowledgeBaseError(
                "KNOWLEDGE_BASE_URL is not configured. Set it to the "
                "existing Knowledge Base's retrieval endpoint."
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.top_k = top_k
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        """Retrieve relevant context chunks for `query` from the
        Knowledge Base. Returns a list of {"text": str, "source": str}
        dicts (empty list if the Knowledge Base finds nothing relevant).

        Retries transient failures (timeouts, 5xx) up to
        settings.HTTP_MAX_RETRIES times with exponential backoff before
        raising KnowledgeBaseError.
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("query must not be empty.")

        payload = {"query": query, "top_k": top_k or self.top_k}
        url = f"{self.base_url}/retrieve"

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    url, headers=self._headers(), json=payload, timeout=self.timeout
                )
                if response.status_code >= 500:
                    raise KnowledgeBaseError(
                        f"Knowledge Base returned {response.status_code}: {response.text[:300]}"
                    )
                response.raise_for_status()
                return self._parse_response(response.json())
            except (requests.RequestException, KnowledgeBaseError, ValueError) as exc:
                last_exc = exc
                logger.warning(
                    "Knowledge Base retrieval attempt %d/%d failed: %s",
                    attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * attempt)

        raise KnowledgeBaseError(
            f"Knowledge Base retrieval failed after {self.max_retries} attempts: {last_exc}"
        ) from last_exc

    @staticmethod
    def _parse_response(data: dict) -> List[dict]:
        chunks = data.get("chunks", data.get("results", []))
        if not isinstance(chunks, list):
            raise KnowledgeBaseError(
                "Knowledge Base response did not contain a 'chunks' list."
            )
        parsed = []
        for chunk in chunks:
            if isinstance(chunk, str):
                parsed.append({"text": chunk, "source": ""})
            elif isinstance(chunk, dict):
                parsed.append(
                    {
                        "text": chunk.get("text") or chunk.get("content") or "",
                        "source": chunk.get("source") or chunk.get("filename") or "",
                    }
                )
        return [c for c in parsed if c["text"].strip()]
