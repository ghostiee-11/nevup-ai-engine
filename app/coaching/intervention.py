from typing import AsyncIterator

from app.coaching.groq_client import stream_groq
from app.memory.service import get_context

SYSTEM = (
    "You are NevUp, a trading-psychology coach. The user just closed a trade.\n"
    "A deterministic detector has flagged a behavioral signal. Your job is to acknowledge\n"
    "the signal, reference at most ONE prior session by sessionId from the provided context\n"
    "(do not invent IDs), and propose a single concrete next action.\n"
    "Constraints: <= 120 words. Plain prose. Never claim a session existed unless it is\n"
    "in the supplied context. If the context is empty, do not cite any sessionId."
)


async def stream_coaching(user_id: str, signal: dict, current_trade: dict) -> AsyncIterator[str]:
    relevant = f"signal {signal.get('type')} for asset {current_trade.get('asset')}"
    try:
        ctx = await get_context(user_id, relevant, limit=3)
        cite_block = "\n".join(
            f"- sessionId={s.session_id} tags={s.tags} summary={s.summary[:160]}"
            for s in ctx.sessions
        ) or "(no prior memory)"
    except Exception:  # noqa: BLE001 — never break the stream on memory issues
        cite_block = "(no prior memory)"
    user_prompt = (
        f"Detected signal: {signal}\n"
        f"Current trade: {current_trade}\n"
        f"Prior memory:\n{cite_block}\n"
        "Coach the user."
    )
    async for tok in stream_groq(SYSTEM, user_prompt):
        yield tok
