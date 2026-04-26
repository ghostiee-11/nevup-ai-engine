"""LLM-assisted profile narrative. Takes the rule-scored output (with real citations)
and produces a structured profile JSON. The LLM never invents IDs - it can only
paraphrase the evidence already produced by rules.py.
"""
import asyncio
import json
import logging

import google.generativeai as genai

from app.config import settings

log = logging.getLogger(__name__)


PROFILE_SCHEMA_PROMPT = """You are a trading-psychology profiler.

INPUT: trader stats, top scored pathologies (rule-derived) with their evidence (real tradeIds and sessionIds).

OUTPUT: strict JSON matching this schema:
{
  "userId": "<echo>",
  "primaryPathology": "<one of the pathology labels OR 'none'>",
  "confidence": <float 0..1>,
  "weaknesses": [
    {"pattern": "<short label>", "failureMode": "<short>", "peakWindow": "<e.g. '09:30-10:30 UTC'|null>",
     "citations": [{"sessionId": "<from input>", "tradeId": "<from input or null>"}]}
  ],
  "narrative": "<<= 500 chars, references concrete cited evidence only>"
}

RULES:
- Citations MUST be drawn from the evidence array provided. Never invent IDs.
- If primary score < 0.3, primaryPathology = "none".
- Output JSON only - no markdown, no commentary.
"""


_configured = False


def _configure():
    global _configured
    if not _configured and settings.gemini_api_key:
        genai.configure(api_key=settings.gemini_api_key)
        _configured = True


def _rules_only_profile(user_id: str, scored: list[dict]) -> dict:
    top = scored[0]
    return {
        "userId": user_id,
        "primaryPathology": top["pathology"] if top["score"] >= 0.3 else "none",
        "confidence": top["score"],
        "weaknesses": [
            {
                "pattern": top["pathology"],
                "failureMode": "rule-detected",
                "peakWindow": None,
                "citations": [
                    {"sessionId": e.get("session_id"), "tradeId": e.get("trade_id")}
                    for e in top["evidence"]
                ],
            }
        ],
        "narrative": f"Rule-based profile: dominant pattern {top['pathology']} score={top['score']}.",
    }


async def narrate_profile(user_id: str, scored: list[dict], stats: dict) -> dict:
    _configure()
    if not settings.gemini_api_key:
        return _rules_only_profile(user_id, scored)
    payload = {"userId": user_id, "stats": stats, "scored": scored[:3]}
    try:
        model = genai.GenerativeModel(
            settings.gemini_profile_model,
            system_instruction=PROFILE_SCHEMA_PROMPT,
            generation_config={"response_mime_type": "application/json"},
        )
        response = await asyncio.to_thread(model.generate_content, json.dumps(payload, default=str))
        parsed = json.loads(response.text)
    except Exception as e:  # noqa: BLE001 -- LLM path must never break the rules-only fallback
        log.warning("LLM profile generation failed (%s) - falling back to rules-only", e)
        return _rules_only_profile(user_id, scored)
    parsed["userId"] = user_id  # never trust the LLM with this
    return parsed
