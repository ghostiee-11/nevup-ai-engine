from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.audit.router import router as audit_router
from app.coaching.router import router as coaching_router
from app.memory.router import router as memory_router
from app.observability.logging import configure_logging
from app.observability.middleware import TracingMiddleware
from app.profiling.router import router as profile_router

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="NevUp AI Engine", version="0.1.0", lifespan=lifespan)
# TracingMiddleware is added first → ends up innermost (runs after CORS).
app.add_middleware(TracingMiddleware)
# CORS added last → outermost → handles preflight before any other middleware.
# allow_credentials=False is required when allow_origins=["*"] (per CORS spec).
# This service uses bearer JWTs in the Authorization header, not cookies, so no creds needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Trace-Id"],
    expose_headers=["X-Trace-Id"],
    max_age=600,
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "ERROR", "message": str(exc.detail)},
    )


@app.get("/")
async def root():
    return {
        "service": "NevUp AI Engine",
        "version": "0.1.0",
        "ui": "https://nevup-ui.vercel.app",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "GET /profile/{userId}",
            "PUT /memory/{userId}/sessions/{sessionId}",
            "GET /memory/{userId}/context?relevant_to=...",
            "GET /memory/{userId}/sessions/{sessionId}",
            "POST /session/events?user_id=...",
            "POST /audit",
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "queue_lag": 0, "db": "ok"}


app.include_router(memory_router)
app.include_router(profile_router)
app.include_router(coaching_router)
app.include_router(audit_router)
