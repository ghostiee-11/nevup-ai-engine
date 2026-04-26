from contextlib import asynccontextmanager
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="NevUp AI Engine", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "queue_lag": 0, "db": "ok"}
