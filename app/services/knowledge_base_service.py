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

# When the underlying Aurora Serverless vector store has auto-paused, its
# error message contains this phrase. Detected specifically so we can wait
# long enough for it to resume (typically 15-30s) instead of giving up
# after only a few seconds of generic backoff.
_AURORA_RESUME_MARKERS = ("auto-paused", "is resuming")


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
        max_retries: int = settings.KNOWLEDGE_BASE_MAX_RETRIES,
        backoff_seconds: float = settings.KNOWLEDGE_BASE_RETRY_BACKOFF_SECONDS,
        resume_wait_seconds: float = settings.KNOWLEDGE_BASE_RESUME_WAIT_SECONDS,
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
        self.resume_wait_seconds = resume_wait_seconds
        self._client = None  # lazy init so import-time never touches AWS

    @property
    def client(self):
        if self._client is None:
            # No explicit credentials passed — boto3's default credential
            # chain handles this correctly in every environment: the
            # Lambda execution role (including its session token) when
            # deployed, or ~/.aws/credentials / exported AWS_* env vars
            # locally.
            #
            # max_attempts=1 deliberately disables botocore's OWN
            # automatic retries for this client: retrieve() below already
            # implements its own Aurora-aware manual retry/backoff loop,
            # and letting botocore ALSO retry underneath it would
            # multiply attempts (and worst-case latency) rather than
            # bound it. connect/read timeouts keep a single attempt from
            # hanging indefinitely.
            from botocore.config import Config

            client_config = Config(
                connect_timeout=settings.KNOWLEDGE_BASE_CONNECT_TIMEOUT_SECONDS,
                read_timeout=settings.KNOWLEDGE_BASE_READ_TIMEOUT_SECONDS,
                retries={"max_attempts": 1},
            )
            self._client = boto3.client("bedrock-agent-runtime", region_name=self.region, config=client_config)
        return self._client

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        """Retrieve relevant context chunks for `query` from the
        Knowledge Base. Returns a list of {"text": str, "source": str}
        dicts (empty list if the Knowledge Base finds nothing relevant).

        Retries transient failures up to `max_retries` times. Most
        failures use short exponential backoff, but an Aurora
        auto-pause/resume error gets a much longer, fixed wait per
        attempt (`resume_wait_seconds`), since that specific condition
        reliably takes 15-30s to clear and a short backoff just burns
        through all retries while Aurora is still waking up.
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
                is_resuming = self._is_aurora_resuming(exc)
                logger.warning(
                    "Knowledge Base retrieval attempt %d/%d failed%s: %s",
                    attempt, self.max_retries,
                    " (Aurora resuming from auto-pause)" if is_resuming else "",
                    exc,
                )
                if attempt < self.max_retries:
                    wait = self.resume_wait_seconds if is_resuming else self.backoff_seconds * attempt
                    time.sleep(wait)

        raise KnowledgeBaseError(
            f"Knowledge Base retrieval failed after {self.max_retries} attempts: {last_exc}"
        ) from last_exc

    @staticmethod
    def _is_aurora_resuming(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(marker in message for marker in _AURORA_RESUME_MARKERS)

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
                parsed.append({"text": text, "source": source, "score": result.get("score")})
        return parsed