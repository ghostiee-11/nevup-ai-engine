import logging
from typing import AsyncIterator

from groq import AsyncGroq

from app.config import settings

log = logging.getLogger(__name__)
_client: AsyncGroq | None = None


def _client_lazy() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=settings.groq_api_key)
    return _client


async def stream_groq(system: str, user: str) -> AsyncIterator[str]:
    if not settings.groq_api_key:
        for tok in ["[stub coaching — no GROQ_API_KEY] ", system[:40], "..."]:
            yield tok
        return
    stream = await _client_lazy().chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        stream=True,
        max_tokens=400,
        temperature=0.4,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta
