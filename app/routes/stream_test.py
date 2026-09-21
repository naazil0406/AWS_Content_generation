"""
Artificial streaming test endpoint — NOT part of the real application
logic. Exists solely to answer one narrow question in isolation:

    "Does data actually arrive at the client progressively, across a
    connection that survives past ~29-30 seconds, on whatever transport
    this Lambda is deployed behind?"

No Freepik call, no Bedrock call, no S3 upload — nothing here can fail
because of a real upstream provider, a missing API key, or a real cold
Aurora resume. If this endpoint doesn't stream correctly, the problem is
the transport (API Gateway / Lambda invoke mode / Mangum vs. a streaming-
capable runtime) — not your generation logic. If it DOES stream
correctly but your real /api/generate/stream still doesn't behave the
same way once deployed, that tells you the problem is something specific
to the real pipeline (e.g. an exception before the first yield, or a
buffering proxy/library on the client side) rather than a fundamental
transport limitation.

Included in app.main via a router, same pattern as app/routes/content.py.
"""

import asyncio
import json
import time

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

router = APIRouter()


def _sse(event: str, data: dict) -> str:
    # Identical framing to app/routes/content.py's real _sse() — this
    # test proves something about your ACTUAL wire format, not a generic
    # SSE example.
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


# (elapsed_seconds_to_wait_before_this_event, event_name, payload_extra)
# Cumulative: each entry's delay is measured from the PREVIOUS event, not
# from t=0 — see the loop below. Checkpoints match the timeline requested:
# immediately, then 5s, 10s, 20s, 30s, 40s, 50s, 60s from start.
_SCHEDULE = [
    (0, "stream_started", {"message": "stream started"}),
    (5, "progress", {"step": 1, "message": "progress 1"}),
    (5, "progress", {"step": 2, "message": "progress 2"}),
    (10, "progress", {"step": 3, "message": "progress 3"}),
    (10, "still_running", {"message": "still running"}),
    (10, "progress", {"step": 4, "message": "progress 4"}),
    (10, "progress", {"step": 5, "message": "progress 5"}),
    (10, "completed", {"message": "completed"}),
]
# Cumulative elapsed times this produces: 0, 5, 10, 20, 30, 40, 50, 60s.


@router.get("/stream-test")
async def stream_test(request: Request, fail_at: float = Query(default=None)):
    """GET /api/stream-test

    Normal run:   GET /api/stream-test
    Error test:   GET /api/stream-test?fail_at=35
                  (raises a simulated error partway through, after some
                  real progress events already reached the client, then
                  emits a clean SSE "error" event and closes -- proves
                  errors mid-stream don't hang the connection or leak a
                  raw traceback to the client)

    Client-disconnect handling: checks request.is_disconnected() between
    every event and stops cleanly (logs it) if the client has gone away,
    instead of continuing to "run" server-side for the full 60s with
    nobody listening.
    """
    async def event_stream():
        t0 = time.monotonic()
        try:
            for delay, event_name, payload in _SCHEDULE:
                # Check for disconnect on a tight ~1s cadence instead of
                # only once per (up to 10s-apart) scheduled event -- this
                # is what actually makes disconnect detection responsive
                # rather than a documentation-only claim.
                remaining = delay
                while remaining > 0:
                    step = min(1.0, remaining)
                    await asyncio.sleep(step)
                    remaining -= step
                    if await request.is_disconnected():
                        elapsed = time.monotonic() - t0
                        print(f"[stream-test] client disconnected at {elapsed:.1f}s — stopping early.", flush=True)
                        return

                elapsed = time.monotonic() - t0

                if fail_at is not None and elapsed >= fail_at:
                    yield _sse("error", {
                        "message": f"Simulated failure at {elapsed:.1f}s (requested via ?fail_at={fail_at}).",
                        "elapsed_seconds": round(elapsed, 1),
                    })
                    return

                yield _sse(event_name, {**payload, "elapsed_seconds": round(elapsed, 1)})
        except Exception as exc:  # noqa: BLE001 - last-resort so the stream always closes cleanly
            yield _sse("error", {"message": f"Unexpected error: {exc}", "elapsed_seconds": round(time.monotonic() - t0, 1)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )