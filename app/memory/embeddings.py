import asyncio
import hashlib
import logging
import struct
from typing import Iterable

import google.generativeai as genai
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings

log = logging.getLogger(__name__)
_configured = False


def _configure() -> bool:
    """Returns True if Gemini is configured (key present), False otherwise."""
    global _configured
    if not settings.gemini_api_key:
        return False
    if not _configured:
        genai.configure(api_key=settings.gemini_api_key)
        _configured = True
    return True


async def _embed_call(text: str) -> dict:
    return await asyncio.to_thread(
        genai.embed_content,
        model=settings.gemini_embed_model,
        content=text,
        task_type="retrieval_document",
    )


def _fallback_embedding(text: str, dim: int | None = None) -> list[float]:
    """Deterministic pseudo-embedding for when Gemini is unavailable.

    Uses SHA-256 expanded to `dim` floats in [-1, 1]. Not semantically meaningful,
    but deterministic per text — so memory upsert+retrieve still round-trips
    (similar text → identical vector → cosine distance 0). The brief mandates
    fallback paths for every external call; this lets the system keep running
    without a Gemini key.
    """
    target = dim or settings.embedding_dim
    out: list[float] = []
    counter = 0
    while len(out) < target:
        h = hashlib.sha256(f"{text}:{counter}".encode("utf-8")).digest()
        for i in range(0, 32, 4):
            (n,) = struct.unpack(">I", h[i : i + 4])
            out.append((n / 2**31) - 1.0)
            if len(out) >= target:
                break
        counter += 1
    return out


async def embed(text: str) -> list[float]:
    """Return an embedding vector. Falls back to a deterministic SHA-based
    pseudo-embedding if Gemini is missing or all retries fail.
    """
    if not _configure():
        log.info("GEMINI_API_KEY not set — using fallback embedding")
        return _fallback_embedding(text)
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                res = await _embed_call(text)
        return list(res["embedding"])
    except Exception as e:  # noqa: BLE001 -- any Gemini failure must fall back
        log.warning("Gemini embed failed (%s) — using fallback embedding", e)
        return _fallback_embedding(text)


async def embed_batch(texts: Iterable[str]) -> list[list[float]]:
    return await asyncio.gather(*(embed(t) for t in texts))
