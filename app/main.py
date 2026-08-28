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

# Deploy marker (no functional effect) — bump this comment to force SAM
# to detect a code change and publish a new Lambda version.
# build: 2026-08-20-01

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from mangum import Mangum

from app.routes import content
from app.services.knowledge_base_service import KnowledgeBaseError, KnowledgeBaseService
from app.config import settings

logger = logging.getLogger(__name__)

# Lambda sets AWS_LAMBDA_FUNCTION_VERSION automatically (no template/env
# config needed) — logged once per cold start purely so CloudWatch Logs
# show which immutable Lambda version handled a given batch of requests,
# which is what makes it possible to tell "Version 4's logs" apart from
# "Version 3's logs" during a canary rollout or after a rollback. Note:
# this is the resolved version number only — Lambda does not expose
# which alias (e.g. "prod") was used to invoke it, since alias
# resolution happens before your code ever runs. Application logic never
# reads or branches on this value — see template.yaml's
# AutoPublishAlias/DeploymentPreference for the actual versioning/
# rollback mechanism.
#
# IMAGE_PROVIDER and whether FREEPIK_API_KEY is set (never the key's
# value — presence only) are logged alongside it deliberately: since
# each Lambda VERSION freezes its own environment variables at publish
# time, this is what lets you look at one version's CloudWatch logs and
# immediately see whether THAT specific version — not $LATEST, not
# whatever `aws lambda get-function-configuration` shows without a
# --qualifier — actually has the key baked in, without four separate
# AWS CLI round trips to figure out a version/alias mismatch.
logger.info(
    "Cold start — Lambda function: %s, version: %s, image_provider: %s, freepik_key_set: %s",
    os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "n/a (not running in Lambda)"),
    os.environ.get("AWS_LAMBDA_FUNCTION_VERSION", "n/a"),
    settings.IMAGE_PROVIDER,
    bool(settings.FREEPIK_API_KEY),
)

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

_FRONTEND_PATH = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@app.get("/", response_class=HTMLResponse)
def serve_frontend() -> str:
    # Read fresh on every request rather than caching the file's content
    # in a module-level variable at import time. The old cached-once
    # approach was fine for Lambda (a fresh deploy always publishes a
    # new version, which re-imports this module from scratch at its
    # next cold start) but was a real footgun for local development:
    # `uvicorn --reload` only watches .py files by default, so editing
    # frontend/index.html and even fully restarting deployment tooling
    # could still serve stale HTML indefinitely from whatever this
    # module-level variable was set to at the process's last import —
    # no amount of browser hard-refreshing fixes that, since the SERVER
    # itself was returning old content, not something cached client-side.
    # A single small HTML file read per page load is negligible cost
    # (this is not a hot API path — it's the one-time initial page
    # load, not something called per generation request) and it removes
    # this entire class of "why isn't my frontend change showing up"
    # confusion for good, in both local dev and Lambda.
    if not _FRONTEND_PATH.exists():
        # Frontend not bundled — API still works standalone (e.g. local
        # dev without the frontend/ folder, or if you deploy API-only).
        return HTMLResponse(
            "<h1>AI Content Generation Service</h1>"
            "<p>API is running. frontend/index.html was not found in this deployment.</p>",
            status_code=200,
        )
    return _FRONTEND_PATH.read_text(encoding="utf-8")


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