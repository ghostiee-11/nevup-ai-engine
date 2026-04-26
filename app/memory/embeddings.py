import asyncio
import logging
from typing import Iterable

import google.generativeai as genai
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings

log = logging.getLogger(__name__)
_configured = False


def _configure():
    global _configured
    if not _configured and settings.gemini_api_key:
        genai.configure(api_key=settings.gemini_api_key)
        _configured = True


async def _embed_call(text: str) -> dict:
    _configure()
    return await asyncio.to_thread(
        genai.embed_content,
        model=settings.gemini_embed_model,
        content=text,
        task_type="retrieval_document",
    )


async def embed(text: str) -> list[float]:
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    ):
        with attempt:
            res = await _embed_call(text)
    return list(res["embedding"])


async def embed_batch(texts: Iterable[str]) -> list[list[float]]:
    return await asyncio.gather(*(embed(t) for t in texts))
