from typing import AsyncIterator

from app.coaching.groq_client import stream_groq
from app.memory.service import get_context

SYSTEM_SINGLE = (
    "You are NevUp, a trading-psychology coach. The user just closed a trade.\n"
    "A deterministic detector has flagged a behavioral signal. Your job is to acknowledge\n"
    "the signal, reference at most ONE prior session by sessionId from the provided context\n"
    "(do not invent IDs), and propose a single concrete next action.\n"
    "Constraints: <= 120 words. Plain prose. Never claim a session existed unless it is\n"
    "in the supplied context. If the context is empty, do not cite any sessionId."
)

SYSTEM_MULTI = (
    "You are NevUp, a trading-psychology coach. The user just closed a trade.\n"
    "A deterministic detector has flagged MULTIPLE behavioral signals. Address them in the\n"
    "order given (highest priority first). For each signal, briefly acknowledge it and\n"
    "propose ONE concrete next action. Reference at most ONE prior session by sessionId\n"
    "across the whole response (do not invent IDs).\n"
    "Constraints: <= 200 words total. Plain prose. Number each signal (1., 2., ...).\n"
    "Never claim a session existed unless it is in the supplied context."
)


async def stream_coaching(
    user_id: str, signals: list[dict] | dict, current_trade: dict,
) -> AsyncIterator[str]:
    """Stream coaching tokens for one or more detected signals.

    Accepts a list (multi-label) or a single dict (back-compat with old callers).
    Selects the appropriate system prompt based on signal count and assembles
    the user prompt around the detected signal(s) and retrieved memory context.
    """
    if isinstance(signals, dict):
        signals = [signals]
    primary = signals[0] if signals else {"type": "post_trade_review"}
    multi = len(signals) > 1

    relevant = (
        f"signals: {', '.join(s.get('type', 'unknown') for s in signals)} · "
        f"asset {current_trade.get('asset')}"
    )
    try:
        ctx = await get_context(user_id, relevant, limit=3)
        cite_block = "\n".join(
            f"- sessionId={s.session_id} tags={s.tags} summary={s.summary[:160]}"
            for s in ctx.sessions
        ) or "(no prior memory)"
    except Exception:  # noqa: BLE001 — never break the stream on memory issues
        cite_block = "(no prior memory)"

    if multi:
        signals_block = "\n".join(
            f"  {i + 1}. {s}" for i, s in enumerate(signals[:3])
        )
        user_prompt = (
            f"Detected signals (priority order):\n{signals_block}\n"
            f"Current trade: {current_trade}\n"
            f"Prior memory:\n{cite_block}\n"
            "Coach the user — address each signal in order."
        )
        system = SYSTEM_MULTI
    else:
        user_prompt = (
            f"Detected signal: {primary}\n"
            f"Current trade: {current_trade}\n"
            f"Prior memory:\n{cite_block}\n"
            "Coach the user."
        )
        system = SYSTEM_SINGLE

    async for tok in stream_groq(system, user_prompt):
        yield tok
