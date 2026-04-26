import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

log = logging.getLogger("request")


class TracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace_id = request.headers.get("x-trace-id") or str(uuid.uuid4())
        request.state.trace_id = trace_id
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = int((time.perf_counter() - start) * 1000)
        user_id = None
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            try:
                from app.auth.jwt import decode_token
                user_id = decode_token(auth.split(" ", 1)[1]).get("sub")
            except Exception:  # noqa: BLE001
                user_id = None
        rec = logging.LogRecord("request", logging.INFO, __file__, 0, "request", None, None)
        rec.extras = {
            "traceId": trace_id, "userId": user_id, "latency": latency_ms,
            "statusCode": response.status_code, "path": request.url.path,
            "method": request.method,
        }
        log.handle(rec)
        response.headers["x-trace-id"] = trace_id
        return response
