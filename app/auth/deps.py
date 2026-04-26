import uuid

from fastapi import Header, HTTPException, Request

from app.auth.jwt import JWTError, decode_token


def _trace_id(request: Request | None) -> str:
    if request is None:
        return str(uuid.uuid4())
    return getattr(request.state, "trace_id", None) or str(uuid.uuid4())


async def require_user(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error": "UNAUTHORIZED", "message": "Missing bearer token",
                    "traceId": _trace_id(request)},
        )
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = decode_token(token)
    except JWTError as e:
        raise HTTPException(
            status_code=401,
            detail={"error": "UNAUTHORIZED", "message": str(e),
                    "traceId": _trace_id(request)},
        )
    return payload


def enforce_tenancy(user: dict, requested_user_id: str, request: Request | None = None) -> None:
    if user["sub"] != requested_user_id:
        raise HTTPException(
            status_code=403,
            detail={"error": "FORBIDDEN", "message": "Cross-tenant access denied.",
                    "traceId": _trace_id(request)},
        )
