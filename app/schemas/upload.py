"""Pydantic response model for the Campaign Document Upload feature."""

from typing import List

from pydantic import BaseModel


class UploadResponse(BaseModel):
    status: str  # "success" | "partial_success"
    message: str
    uploaded_files: List[str]
    ingestion_started: bool
    ingestion_job_id: str = ""