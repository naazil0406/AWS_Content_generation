"""
Tracks the chain: application session -> generated image(s) -> Canva
design_id -> (later) export result. This is what makes the
"return to integration" flow able to say "which design was this, and
does it belong to the session that's asking?" (section 2/9).

Keyed by `correlation_id` — a value WE mint, embed on the way into Canva
(see canva_service.create_design()'s correlation_state parameter and the
docstring on that function for exactly what's verified-vs-assumed about
how Canva round-trips it), and expect Canva to hand back on the return
redirect. See app/routes/canva.py's /return handler for where that
round-trip is actually consumed — and where to look first if it turns
out Canva does NOT preserve this value the way this module assumes.

Built on app/services/local_dev_store.py — same DynamoDB swap-in note as
every other Canva storage module in this project applies here too.
"""

import time
from typing import Optional, TypedDict

from app.services import local_dev_store

_DESIGN_KEY_PREFIX = "canva_design:"
_RECORD_TTL_SECONDS = 60 * 60 * 6  # 6 hours — plenty for someone to finish editing in Canva and return


class DesignRecord(TypedDict):
    session_id: str
    design_id: str
    source_generation_id: Optional[str]  # None for "Create New in Canva" (no source images)
    source_indices: list
    created_at: float
    export: Optional[dict]  # populated once /exports succeeds — see canva.py


def record_design(
    correlation_id: str,
    session_id: str,
    design_id: str,
    source_generation_id: Optional[str] = None,
    source_indices: Optional[list] = None,
) -> None:
    """Call immediately after create_design() returns, for BOTH "Create
    New in Canva" and "Edit in Canva" (source_generation_id/indices are
    only set for the latter — see app/routes/canva.py)."""
    record: DesignRecord = {
        "session_id": session_id,
        "design_id": design_id,
        "source_generation_id": source_generation_id,
        "source_indices": source_indices or [],
        "created_at": time.time(),
        "export": None,
    }
    local_dev_store.put_item(_DESIGN_KEY_PREFIX + correlation_id, record, ttl_seconds=_RECORD_TTL_SECONDS)


def get_design(correlation_id: str) -> Optional[DesignRecord]:
    return local_dev_store.get_item(_DESIGN_KEY_PREFIX + correlation_id)


def verify_design_ownership(session_id: str, correlation_id: str) -> Optional[DesignRecord]:
    """Returns the record only if it exists AND belongs to session_id —
    the single check that satisfies section 9's "a user cannot submit
    another user's Canva design ID and retrieve their result." Returns
    None on any mismatch/absence; callers should turn that into 403/404,
    never leak WHICH check failed (see app/routes/canva.py)."""
    record = get_design(correlation_id)
    if record is None or record["session_id"] != session_id:
        return None
    return record


def record_export_result(correlation_id: str, s3_key: str, edited_image_url: str) -> None:
    """Called once the export -> download -> S3 upload pipeline
    succeeds (app/routes/canva.py's /exports/{correlation_id}) — lets a
    second call for the same correlation_id (e.g. the frontend re-polling
    after a page refresh) return the already-computed result instead of
    re-exporting from Canva every time."""
    record = get_design(correlation_id)
    if record is None:
        return
    record["export"] = {"s3_key": s3_key, "edited_image_url": edited_image_url}
    local_dev_store.put_item(_DESIGN_KEY_PREFIX + correlation_id, record, ttl_seconds=_RECORD_TTL_SECONDS)