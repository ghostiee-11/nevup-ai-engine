from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="NevUp AI Engine", version="0.1.0", lifespan=lifespan)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "ERROR", "message": str(exc.detail)},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "queue_lag": 0, "db": "ok"}
