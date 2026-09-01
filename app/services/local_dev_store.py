"""
Local-development key/value store — Canva OAuth tokens and image
ownership records for LOCAL TESTING ONLY.

+-----------------------------------------------------------------------+
| PRODUCTION TODO                                                       |
|                                                                        |
| This is a single JSON file on local disk with a threading.Lock — fine |
| for one developer running `uvicorn --reload` on their laptop, and     |
| completely wrong for Lambda: concurrent invocations don't share a     |
| filesystem, `/tmp` is wiped between cold starts, and there's no       |
| locking across invocations at all. Before deploying the Canva         |
| integration to AWS, replace this module's implementation with a       |
| DynamoDB-backed one that exposes the EXACT SAME three functions       |
| (get_item, put_item, delete_item) with the same signatures. Every     |
| other Canva module (canva_token_store.py, image_ownership_service.py) |
| only imports and calls these three functions — none of them know or  |
| care that the backing store is a JSON file, so swapping the           |
| implementation here is the ONLY change needed; no calling code should |
| need to change.                                                       |
|                                                                        |
| Suggested DynamoDB shape: a single table (e.g. "AppLocalStore") with  |
| a partition key "pk" (String) = the same `key` string used here, an   |
| attribute "value" (String, JSON-encoded same as now), and a TTL       |
| attribute "expires_at" (Number, epoch seconds) so Canva OAuth state   |
| entries (short-lived, see canva_token_store.py) and old image-        |
| ownership records expire automatically instead of growing forever.    |
| get_item/put_item/delete_item below already accept an optional        |
| `ttl_seconds` for exactly this reason.                                |
+-----------------------------------------------------------------------+
"""

import json
import logging
import os
import threading
import time
from typing import Any, Optional

from app.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def _read_all() -> dict:
    if not os.path.exists(settings.LOCAL_DEV_STORE_PATH):
        return {}
    try:
        with open(settings.LOCAL_DEV_STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Local dev store file unreadable/corrupt (%s) — starting fresh.", exc)
        return {}


def _write_all(data: dict) -> None:
    tmp_path = settings.LOCAL_DEV_STORE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp_path, settings.LOCAL_DEV_STORE_PATH)  # atomic on POSIX and Windows


def put_item(key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
    """Store `value` (must be JSON-serializable) under `key`. If
    ttl_seconds is given, get_item() will treat the item as absent once
    it expires (this file-backed store does not proactively evict —
    see the DynamoDB TTL note above for how production should)."""
    with _lock:
        data = _read_all()
        data[key] = {
            "value": value,
            "expires_at": (time.time() + ttl_seconds) if ttl_seconds else None,
        }
        _write_all(data)


def get_item(key: str) -> Optional[Any]:
    """Return the stored value for `key`, or None if absent/expired."""
    with _lock:
        data = _read_all()
        entry = data.get(key)
        if entry is None:
            return None
        if entry.get("expires_at") is not None and time.time() > entry["expires_at"]:
            del data[key]
            _write_all(data)
            return None
        return entry["value"]


def delete_item(key: str) -> None:
    with _lock:
        data = _read_all()
        if key in data:
            del data[key]
            _write_all(data)