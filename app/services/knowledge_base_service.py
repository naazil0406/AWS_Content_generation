"""
Knowledge Base Service — a thin client over an Amazon Bedrock Knowledge
Base.

This service does NOT chunk, embed, or index anything. It does not manage
a vector database directly. All of that already exists inside the Bedrock
Knowledge Base and is someone else's operational responsibility. This
module's entire job is: take a query string, call the Knowledge Base's
`retrieve` API via bedrock-agent-runtime, and hand back whatever context
chunks it returns.

Bedrock Knowledge Bases are addressed by KNOWLEDGE_BASE_ID (found in the
Bedrock console), not a URL — there is no HTTP endpoint to call directly.

If you ever swap to a different, externally hosted Knowledge Base that
exposes its own REST contract instead, change only this file — nothing
else in the application needs to know how retrieval actually happens.
"""

import logging
import time
from typing import List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import settings

logger = logging.getLogger(__name__)


class KnowledgeBaseError(RuntimeError):
    """Raised when the Knowledge Base cannot be reached or returns an
    unusable response."""


class KnowledgeBaseService:
    """Retrieves relevant context chunks from an Amazon Bedrock
    Knowledge Base.

    Calls `bedrock-agent-runtime`'s `retrieve` API with
    `knowledgeBaseId=<KNOWLEDGE_BASE_ID>` and a `retrievalQuery`, and
    returns a list of {"text": str, "source": str} dicts. Adjust
    `_parse_response` below if you need additional fields (e.g. score,
    metadata) — the rest of the app only depends on `retrieve()`.
    """

    def __init__(
        self,
        kb_id: str = settings.KNOWLEDGE_BASE_ID,
        region: str = settings.KNOWLEDGE_BASE_REGION,
        top_k: int = settings.KNOWLEDGE_BASE_TOP_K,
        max_retries: int = settings.HTTP_MAX_RETRIES,
        backoff_seconds: float = settings.HTTP_BACKOFF_SECONDS,
    ):
        if not kb_id:
            raise KnowledgeBaseError(
                "KNOWLEDGE_BASE_ID is not configured. Set it to your "
                "Bedrock Knowledge Base's ID (Bedrock console -> "
                "Knowledge Bases -> your KB -> Knowledge base overview)."
            )
        self.kb_id = kb_id
        self.region = region
        self.top_k = top_k
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._client = None  # lazy init so import-time never touches AWS

    @property
    def client(self):
        if self._client is None:
            client_kwargs = {"region_name": self.region}
            if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
                client_kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
                client_kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
            self._client = boto3.client("bedrock-agent-runtime", **client_kwargs)
        return self._client

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        """Retrieve relevant context chunks for `query` from the
        Knowledge Base. Returns a list of {"text": str, "source": str}
        dicts (empty list if the Knowledge Base finds nothing relevant).

        Retries transient failures up to settings.HTTP_MAX_RETRIES times
        with exponential backoff before raising KnowledgeBaseError.
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("query must not be empty.")

        number_of_results = top_k or self.top_k

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.retrieve(
                    knowledgeBaseId=self.kb_id,
                    retrievalQuery={"text": query},
                    retrievalConfiguration={
                        "vectorSearchConfiguration": {
                            "numberOfResults": number_of_results
                        }
                    },
                )
                return self._parse_response(response)
            except (ClientError, BotoCoreError) as exc:
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
    def _parse_response(response: dict) -> List[dict]:
        results = response.get("retrievalResults", [])
        if not isinstance(results, list):
            raise KnowledgeBaseError(
                "Knowledge Base response did not contain a 'retrievalResults' list."
            )
        parsed = []
        for result in results:
            content = result.get("content") or {}
            text = content.get("text", "")

            location = result.get("location") or {}
            source = (
                (location.get("s3Location") or {}).get("uri")
                or (location.get("webLocation") or {}).get("url")
                or (location.get("confluenceLocation") or {}).get("url")
                or (location.get("salesforceLocation") or {}).get("url")
                or (location.get("sharePointLocation") or {}).get("url")
                or ""
            )

            if text and text.strip():
                parsed.append({"text": text, "source": source})
        return parsed