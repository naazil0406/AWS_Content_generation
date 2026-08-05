"""
AI Content Generation Service — FastAPI application entry point.

Local development:
    uvicorn app.main:app --reload --port 8000

AWS Lambda: `handler` below is the Lambda entry point (Mangum wraps the
FastAPI app in the ASGI-to-Lambda adapter), configured in template.yaml
as `app.main.handler`.

The frontend (frontend/index.html) is served from this same app at "/"
so it shares one origin with the /api/* routes below — the page's fetch
calls use relative paths (e.g. "/api/generate"), which only works when
frontend and API are served from the same domain. Serving it here avoids
a separate S3/CloudFront setup and any CORS configuration.
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from mangum import Mangum

from app.routes import content, upload
from app.services.knowledge_base_service import KnowledgeBaseError, KnowledgeBaseService

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Content Generation Service",
    description="Retrieves Knowledge Base context, generates structured content, "
    "and generates three image variations for a given prompt.",
    version="1.0.0",
)

# CORS is permissive by default for ease of integration; tighten
# allow_origins to your known frontend origin(s) before production use.
# (Not load-bearing for the bundled frontend below, since it's served
# same-origin — this only matters if something else calls the API
# cross-origin.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(content.router, prefix="/api")
app.include_router(upload.router, prefix="/api")

# Loaded once per Lambda cold start, reused across warm invocations.
_FRONTEND_PATH = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
_FRONTEND_HTML = _FRONTEND_PATH.read_text(encoding="utf-8") if _FRONTEND_PATH.exists() else None


@app.get("/", response_class=HTMLResponse)
def serve_frontend() -> str:
    if _FRONTEND_HTML is None:
        # Frontend not bundled — API still works standalone (e.g. local
        # dev without the frontend/ folder, or if you deploy API-only).
        return HTMLResponse(
            "<h1>AI Content Generation Service</h1>"
            "<p>API is running. frontend/index.html was not found in this deployment.</p>",
            status_code=200,
        )
    return _FRONTEND_HTML


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


# --- Aurora keep-alive ---
#
# Aurora Serverless auto-pauses after sitting idle past its idle timeout,
# and waking it back up (15-30s) is too slow to fit inside API Gateway's
# hard 29-30s integration timeout (see knowledge_base_service.py). Rather
# than touching the Aurora/RDS configuration, the bundled frontend
# (frontend/index.html) pings this endpoint every ~4 minutes for as long
# as someone has the page open, so Aurora never goes idle long enough to
# pause during active use. It's a plain, cheap `retrieve()` call whose
# result is discarded — only its side effect (keeping Aurora active)
# matters. Multiple users' tabs pinging concurrently is harmless: it's a
# stateless read, so redundant simultaneous pings just no-op past the
# first one to land.
_KEEPALIVE_QUERY = "keepalive-ping"


@app.get("/api/warmup")
def warmup() -> dict:
    try:
        KnowledgeBaseService().retrieve(_KEEPALIVE_QUERY, top_k=1)
        return {"status": "ok"}
    except KnowledgeBaseError as exc:
        # Best-effort ping, not user-facing — don't raise a 5xx over
        # this. Log it so CloudWatch shows if pings start failing (e.g.
        # Aurora was already paused when the ping fired, or the
        # ping itself is what's waking it up right now).
        logger.warning("Aurora keep-alive ping failed: %s", exc)
        return {"status": "error", "detail": str(exc)}


# AWS Lambda entry point (API Gateway HTTP API -> Lambda -> Mangum -> FastAPI).
handler = Mangum(app)