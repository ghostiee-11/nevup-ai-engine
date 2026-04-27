"""Verifies the synthetic generators produce traders whose feature signatures
fire the corresponding rule in app/profiling/rules.py.

These tests serve a dual purpose:
1. Prove generators are correct (each labelled trader scores ≥ 0.3 on its target).
2. Anchor the generator's contract — if rules.py changes its definition, these
   break first, surfacing the contract drift.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from app.profiling.rules import score_pathologies
from scripts.generate_synthetic_traders import PATHOLOGIES, generate


def _trades_for_user(payload: dict, user_id: str) -> list[dict]:
    """Flatten all trades for a user and convert to the rules-layer dict shape."""
    trader = next(t for t in payload["traders"] if t["userId"] == user_id)
    flat: list[dict] = []
    for sess in trader["sessions"]:
        for t in sess["trades"]:
            flat.append({
                "trade_id": t["tradeId"],
                "session_id": t["sessionId"],
                "user_id": t["userId"],
                "asset": t["asset"],
                "asset_class": t["assetClass"],
                "direction": t["direction"],
                "entry_price": t["entryPrice"],
                "exit_price": t.get("exitPrice"),
                "quantity": t["quantity"],
                "entry_at": datetime.fromisoformat(t["entryAt"].replace("Z", "+00:00")),
                "exit_at": datetime.fromisoformat(t["exitAt"].replace("Z", "+00:00"))
                if t.get("exitAt") else None,
                "status": t["status"],
                "outcome": t.get("outcome"),
                "pnl": t.get("pnl"),
                "plan_adherence": t.get("planAdherence"),
                "emotional_state": t.get("emotionalState"),
                "entry_rationale": t.get("entryRationale"),
                "revenge_flag": bool(t.get("revengeFlag", False)),
            })
    return flat


@pytest.fixture(scope="module")
def payload() -> dict:
    """Generate a small but representative dataset once per module."""
    return generate(n_per_class=3, seed=42)


def test_schema_top_level_keys(payload: dict) -> None:
    assert {"meta", "groundTruthLabels", "splits", "traders"} <= payload.keys()
    assert payload["meta"]["traderCount"] == len(payload["traders"])
    assert payload["meta"]["totalSessions"] == sum(len(t["sessions"]) for t in payload["traders"])
    assert payload["meta"]["totalTrades"] == sum(t["stats"]["totalTrades"] for t in payload["traders"])


def test_split_is_stratified_and_disjoint(payload: dict) -> None:
    train = set(payload["splits"]["train"])
    test = set(payload["splits"]["test"])
    assert train & test == set(), "train/test sets must be disjoint"
    all_ids = {t["userId"] for t in payload["traders"]}
    assert train | test == all_ids, "split must cover all traders"


def test_determinism_same_seed_same_payload() -> None:
    a = generate(n_per_class=2, seed=7)
    b = generate(n_per_class=2, seed=7)
    # `meta.generated` is wall-clock and intentionally differs; everything else identical.
    a["meta"].pop("generated")
    b["meta"].pop("generated")
    assert a == b


@pytest.mark.parametrize("pathology", list(PATHOLOGIES))
def test_each_generator_fires_its_target_rule(payload: dict, pathology: str) -> None:
    """Every labelled trader must score ≥ 0.3 on its labelled pathology."""
    target_traders = [
        t for t in payload["traders"]
        if t["groundTruthPathologies"] == [pathology]
    ]
    assert target_traders, f"no traders generated for {pathology}"
    failures: list[tuple[str, float]] = []
    for t in target_traders:
        trades = _trades_for_user(payload, t["userId"])
        scored = score_pathologies(trades)
        match = next(s for s in scored if s["pathology"] == pathology)
        if match["score"] < 0.3:
            failures.append((t["userId"][:8], match["score"]))
    assert not failures, (
        f"{pathology}: {len(failures)}/{len(target_traders)} traders scored below 0.3 — "
        f"{failures[:3]}"
    )


def test_controls_score_low_across_all_pathologies(payload: dict) -> None:
    """Control traders should not strongly trip ANY pathology."""
    controls = [t for t in payload["traders"] if not t["groundTruthPathologies"]]
    assert controls, "expected control traders"
    failures: list[tuple[str, str, float]] = []
    for t in controls:
        trades = _trades_for_user(payload, t["userId"])
        scored = score_pathologies(trades)
        top = scored[0]
        if top["score"] >= 0.5:
            failures.append((t["userId"][:8], top["pathology"], top["score"]))
    # Some controls may exhibit minor patterns. Tolerate up to 1 in 3 with score ≥ 0.5.
    max_allowed = max(1, len(controls) // 3)
    assert len(failures) <= max_allowed, (
        f"too many controls ({len(failures)}/{len(controls)}) trip a pathology ≥0.5: {failures}"
    )
