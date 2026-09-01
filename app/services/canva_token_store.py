"""
Canva OAuth token storage — keyed by app session_id (see
session_service.py), one Canva connection per session/user (section 7:
different application users can authorize different Canva accounts;
nothing here is global).

Built on top of app/services/local_dev_store.py's generic get/put/delete
— see that module's docstring for the DynamoDB swap-in plan. This module
itself does not need to change when that swap happens; only
local_dev_store.py's internals do, since this module only calls its
generic interface.

Nothing in this module ever logs or returns a raw access_token/
refresh_token to a caller outside the Canva service layer — see
app/routes/canva.py's status endpoint, which reports connected: true/false
only, never the token itself.
"""

import time
from typing import Optional, TypedDict

from app.services import local_dev_store

_TOKENS_KEY_PREFIX = "canva_tokens:"
_OAUTH_STATE_KEY_PREFIX = "canva_oauth_state:"
_OAUTH_STATE_TTL_SECONDS = 600  # 10 minutes — plenty for a user to complete the Canva consent screen


class CanvaTokens(TypedDict):
    access_token: str
    refresh_token: str
    expires_at: float  # epoch seconds
    scope: str


def save_tokens(session_id: str, access_token: str, refresh_token: str, expires_in: int, scope: str) -> None:
    tokens: CanvaTokens = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        # expires_in is seconds-from-now per OAuth2 spec — stored as an
        # absolute timestamp so callers never need to know when it was
        # issued, just compare against time.time().
        "expires_at": time.time() + expires_in,
        "scope": scope,
    }
    local_dev_store.put_item(_TOKENS_KEY_PREFIX + session_id, tokens)


def get_tokens(session_id: str) -> Optional[CanvaTokens]:
    return local_dev_store.get_item(_TOKENS_KEY_PREFIX + session_id)


def delete_tokens(session_id: str) -> None:
    local_dev_store.delete_item(_TOKENS_KEY_PREFIX + session_id)


def is_connected(session_id: str) -> bool:
    return get_tokens(session_id) is not None


# --- Short-lived OAuth state (CSRF protection + PKCE code_verifier) --------
# Canva's authorization redirect only gives the callback a `state` and a
# `code` — the PKCE code_verifier used to build the original
# code_challenge must be remembered server-side to complete the token
# exchange (see canva_service.py's exchange_code_for_tokens()). Keyed by
# the random `state` value itself (not session_id) because the
# session cookie may not reliably round-trip through Canva's redirect in
# every browser/setting — state is the one value guaranteed to come back
# unchanged in the callback query string.

def save_oauth_state(state: str, code_verifier: str, session_id: str) -> None:
    local_dev_store.put_item(
        _OAUTH_STATE_KEY_PREFIX + state,
        {"code_verifier": code_verifier, "session_id": session_id},
        ttl_seconds=_OAUTH_STATE_TTL_SECONDS,
    )


def pop_oauth_state(state: str) -> Optional[dict]:
    """Look up and immediately delete — a `state` value must only ever
    be usable once (replay protection)."""
    entry = local_dev_store.get_item(_OAUTH_STATE_KEY_PREFIX + state)
    if entry is not None:
        local_dev_store.delete_item(_OAUTH_STATE_KEY_PREFIX + state)
    return entry