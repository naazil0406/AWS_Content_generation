"""
Lightweight anonymous session identity.

This application has NO existing authentication/user/login system — it's
a stateless content-generation API. Canva's requirements (section 7:
different users must authorize different Canva accounts; section 4/9:
verify a selected image belongs to "the current application user") both
need SOME notion of "who is asking," so this module adds the smallest
thing that satisfies that: an anonymous, random, httponly session cookie.
It identifies "this browser" — nothing more. It is NOT a login system and
carries no username/password/identity claim.

+-----------------------------------------------------------------------+
| IF YOU ADD REAL AUTHENTICATION LATER                                  |
| Replace get_or_create_session_id()'s cookie-based ID with your real   |
| authenticated user ID (e.g. from a JWT/session claim) at the call     |
| site in app/routes/canva.py and app/routes/content.py — every         |
| downstream function here (canva_token_store, image_ownership_service) |
| just takes a `session_id: str` and has no opinion on where it came    |
| from, so swapping the source of that string is the only change       |
| needed.                                                                |
+-----------------------------------------------------------------------+
"""

import uuid

from fastapi import Request, Response

SESSION_COOKIE_NAME = "app_session_id"
_SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


def get_or_create_session_id(request: Request, response: Response) -> str:
    """Read the session cookie off `request` if present; otherwise mint
    a new random one and set it on `response`. Call this in every route
    that needs to know "which browser/user" is calling (Canva connect
    status, image ownership checks, etc.) — both `request` and
    `response` must be real FastAPI-injected objects for the Set-Cookie
    to actually reach the client.

    secure=False below is correct for CANVA_REDIRECT_URI's
    http://localhost:8000 default — set secure=True once this is served
    over HTTPS in production (plain http:// cookies with secure=True
    are silently dropped by browsers, so this must change together with
    your deployment's scheme, not independently).
    """
    existing = request.cookies.get(SESSION_COOKIE_NAME)
    if existing:
        return existing

    new_session_id = uuid.uuid4().hex
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=new_session_id,
        max_age=_SESSION_COOKIE_MAX_AGE,
        httponly=True,   # never readable from frontend JS — not that it's secret, just no reason to expose it
        samesite="lax",
        secure=False,    # see docstring — flip to True when served over HTTPS
    )
    return new_session_id