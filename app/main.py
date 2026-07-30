"""
AI Content Generation Service — FastAPI application entry point.

Local development:
    uvicorn app.main:app --reload --port 8000

AWS Lambda: `handler` below is the Lambda entry point (Mangum wraps the
FastAPI app in the ASGI-to-Lambda adapter), configured in template.yaml
as `app.main.handler`.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

from app.routes import content

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Content Generation Service",
    description="Retrieves Knowledge Base context, generates structured content, "
    "and generates three image variations for a given prompt.",
    version="1.0.0",
)

# CORS is permissive by default for ease of integration; tighten
# allow_origins to your known frontend origin(s) before production use.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(content.router, prefix="/api")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


# AWS Lambda entry point (API Gateway HTTP API -> Lambda -> Mangum -> FastAPI).
handler = Mangum(app)
