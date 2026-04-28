import asyncio
import hashlib
import logging
import struct
from typing import Iterable

import google.generativeai as genai
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings
from app.observability.metrics import embedding_fallback_total

log = logging.getLogger(__name__)

# Three-tier fallback chain:
#   1. Gemini API  (real semantic, 768d native)
#   2. fastembed   (real semantic, 384d → zero-padded to 768d)
#   3. SHA-256     (deterministic pseudo-embedding, 768d, last resort)
# In-process counter mirrors the labelled metric so tests can assert on it
# without spinning up the metrics module's snapshot.
EMBED_PATH_COUNTER: dict[str, int] = {"gemini": 0, "local": 0, "sha": 0}


def _bump(tier: str) -> None:
    EMBED_PATH_COUNTER[tier] += 1
    embedding_fallback_total.inc(tier=tier)

_gemini_configured = False
_local_model = None  # lazy-loaded fastembed.TextEmbedding singleton
_LOCAL_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_LOCAL_DIM = 384  # padded to settings.embedding_dim for column compatibility


def _configure_gemini() -> bool:
    """Returns True if Gemini is configured (key present), False otherwise."""
    global _gemini_configured
    if not settings.gemini_api_key:
        return False
    if not _gemini_configured:
        genai.configure(api_key=settings.gemini_api_key)
        _gemini_configured = True
    return True


def _get_local_model():
    """Lazy-load the fastembed singleton. First call downloads the model (~80MB);
    subsequent calls reuse the in-memory instance.
    """
    global _local_model
    if _local_model is None:
        from fastembed import TextEmbedding  # imported here to keep cold-start light
        log.info("loading local embedding model %s (first call only)", _LOCAL_MODEL_NAME)
        _local_model = TextEmbedding(model_name=_LOCAL_MODEL_NAME)
    return _local_model


async def _embed_gemini(text: str) -> list[float]:
    res = await asyncio.to_thread(
        genai.embed_content,
        model=settings.gemini_embed_model,
        content=text,
        task_type="retrieval_document",
    )
    return list(res["embedding"])


async def _embed_local(text: str) -> list[float]:
    """Run fastembed on a worker thread; pad 384d → 768d so the existing pgvector
    column accepts the result. Cosine similarity is preserved under zero-padding
    when *all* vectors are produced the same way.
    """
    def _run() -> list[float]:
        model = _get_local_model()
        vec = next(model.embed([text]))
        out = list(map(float, vec))
        if len(out) < settings.embedding_dim:
            out.extend([0.0] * (settings.embedding_dim - len(out)))
        return out[: settings.embedding_dim]
    return await asyncio.to_thread(_run)


def _fallback_embedding(text: str, dim: int | None = None) -> list[float]:
    """Deterministic pseudo-embedding for when both Gemini AND fastembed fail.

    Uses SHA-256 expanded to `dim` floats in [-1, 1]. Not semantic — same text
    in always yields same vector out — so memory upsert+retrieve still
    round-trips. Real semantics return as soon as Gemini OR fastembed works.
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
    """Return an embedding vector. Tries Gemini → fastembed → SHA in that order;
    each tier increments its counter in EMBED_PATH_COUNTER for /metrics.
    """
    # Tier 1: Gemini
    if _configure_gemini():
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
                retry=retry_if_exception_type(Exception),
                reraise=True,
            ):
                with attempt:
                    vec = await _embed_gemini(text)
            _bump("gemini")
            return vec
        except Exception as e:  # noqa: BLE001 — fall through to next tier
            log.warning("Gemini embed failed (%s) — trying local model", e)

    # Tier 2: fastembed local
    try:
        vec = await _embed_local(text)
        _bump("local")
        return vec
    except Exception as e:  # noqa: BLE001 — fall through to last-resort
        log.warning("local embed failed (%s) — using SHA fallback", e)

    # Tier 3: SHA pseudo-embedding (never raises)
    _bump("sha")
    return _fallback_embedding(text)


async def embed_batch(texts: Iterable[str]) -> list[list[float]]:
    return await asyncio.gather(*(embed(t) for t in texts))
