"""Eval harness assertions on both the seed fixture and the synthetic held-out split."""
from pathlib import Path

import pytest
from sklearn.metrics import f1_score

from scripts.eval_harness import PATHOLOGIES, run


@pytest.mark.integration
async def test_eval_recovers_seed_labels(seeded_db):
    """Seed eval is the original 10-trader fixture. The bar is high — these are the
    examples we tuned against. We require >= 9/10 to catch any rule regression."""
    out = await run(dataset="seed")
    correct = sum(1 for t, p in zip(out["y_true"], out["y_pred"]) if t == p)
    assert correct >= 9, f"seed eval regressed: only {correct}/10 correct — {out['details']}"


def test_eval_holdout_macro_f1_above_threshold():
    """Held-out 30 traders from the synthetic split. This is the *honest* metric.
    Threshold of 0.65 mirrors the planning doc's lower-bound for credible generalization
    within the synthetic distribution. Skipped if the synthetic dataset hasn't been
    generated yet — regenerate with: python -m scripts.generate_synthetic_traders
    """
    if not Path("data/synthetic_dataset.json").exists():
        pytest.skip("data/synthetic_dataset.json missing — run scripts.generate_synthetic_traders")
    import asyncio
    out = asyncio.run(run(dataset="synthetic_test"))
    f1 = f1_score(out["y_true"], out["y_pred"], labels=PATHOLOGIES, average="macro", zero_division=0)
    assert f1 >= 0.65, f"held-out macro-F1 {f1:.4f} below 0.65 — methodological regression"
