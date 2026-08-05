"""
Ingestion Service — starts a Bedrock Knowledge Base ingestion job scoped
to the Campaigns Data Source ONLY.

Every call in this module is hard-scoped to
`settings.UPLOAD_DATA_SOURCE_ID`, so ingesting newly uploaded campaign
documents can never re-ingest, pause, or otherwise disturb whatever the
existing data source is already doing in the same Knowledge Base.

Uses `bedrock-agent` (the control-plane client for managing Knowledge
Bases/Data Sources/ingestion jobs) — distinct from
`bedrock-agent-runtime`, which app/services/knowledge_base_service.py
uses for retrieval at generation time.
"""

import logging

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import settings

logger = logging.getLogger(__name__)

_bedrock_agent_client = None


class IngestionError(RuntimeError):
    """Raised when a Bedrock Knowledge Base ingestion job cannot be started."""


def _client():
    global _bedrock_agent_client
    if _bedrock_agent_client is None:
        _bedrock_agent_client = boto3.client("bedrock-agent", region_name=settings.KNOWLEDGE_BASE_REGION)
    return _bedrock_agent_client


def start_campaign_ingestion() -> dict:
    """Starts an ingestion job for the Campaigns Data Source only.

    Returns {"ingestion_job_id": str, "status": str}.

    Raises IngestionError if KNOWLEDGE_BASE_ID / UPLOAD_DATA_SOURCE_ID
    aren't configured, or the Bedrock API call itself fails.
    """
    if not settings.KNOWLEDGE_BASE_ID:
        raise IngestionError("KNOWLEDGE_BASE_ID is not configured.")
    if not settings.UPLOAD_DATA_SOURCE_ID:
        raise IngestionError(
            "UPLOAD_DATA_SOURCE_ID is not configured. Set it to the Data "
            "Source ID of the Campaigns Data Source (created inside the "
            "existing Knowledge Base) — not the existing data source's ID."
        )

    try:
        response = _client().start_ingestion_job(
            knowledgeBaseId=settings.KNOWLEDGE_BASE_ID,
            dataSourceId=settings.UPLOAD_DATA_SOURCE_ID,
        )
    except (ClientError, BotoCoreError) as exc:
        logger.error("Failed to start ingestion job for Campaigns Data Source: %s", exc)
        raise IngestionError(f"Failed to start ingestion job: {exc}") from exc

    job = response.get("ingestionJob", {}) or {}
    job_id = job.get("ingestionJobId", "")
    status = job.get("status", "UNKNOWN")
    logger.info("Started Campaigns Data Source ingestion job %s (status=%s)", job_id, status)
    return {"ingestion_job_id": job_id, "status": status}