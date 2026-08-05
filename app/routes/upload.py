"""API route for Campaign Document Upload -> Bedrock Knowledge Base ingestion.

This feature is fully additive. It uploads documents to a dedicated S3
prefix (app/services/s3_service.py) and starts an ingestion job scoped
to the Campaigns Data Source only (app/services/ingestion_service.py).
It does NOT touch knowledge_base_service.py, content_generation_engine.py,
prompt_builder.py, or llm_service.py — the existing /api/generate
pipeline and its retrieval logic are completely unchanged. Once an
ingestion job completes, uploaded documents become retrievable through
the existing KnowledgeBaseService.retrieve() automatically, since
Bedrock KB retrieval searches across every data source in a Knowledge
Base by default.
"""

import logging
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.upload import UploadResponse
from app.services.ingestion_service import IngestionError, start_campaign_ingestion
from app.services.s3_service import S3UploadError, upload_file

logger = logging.getLogger(__name__)
router = APIRouter()

_ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}


def _validate_extension(filename: str) -> None:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type for '{filename}'. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
        )


@router.post("/upload", response_model=UploadResponse)
async def upload_campaign_documents(files: List[UploadFile] = File(...)) -> UploadResponse:
    """Uploads one or more campaign documents to S3, then starts a single
    ingestion job (covering everything just uploaded) for the Campaigns
    Data Source.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided.")

    for f in files:
        _validate_extension(f.filename or "")

    uploaded_keys: List[str] = []
    for f in files:
        try:
            key = upload_file(f.file, f.filename, content_type=f.content_type)
        except S3UploadError as exc:
            logger.error("Upload failed for '%s': %s", f.filename, exc)
            raise HTTPException(status_code=502, detail=str(exc))
        uploaded_keys.append(key)

    try:
        ingestion = start_campaign_ingestion()
    except IngestionError as exc:
        logger.error("Ingestion job failed to start: %s", exc)
        # Files are already safely in S3 at this point, so surface a
        # partial success rather than a 5xx -- the caller needs to know
        # the upload itself worked even though ingestion didn't start.
        return UploadResponse(
            status="partial_success",
            message=f"Documents uploaded, but ingestion failed to start: {exc}",
            uploaded_files=uploaded_keys,
            ingestion_started=False,
        )

    return UploadResponse(
        status="success",
        message="Documents uploaded successfully.",
        uploaded_files=uploaded_keys,
        ingestion_started=True,
        ingestion_job_id=ingestion["ingestion_job_id"],
    )