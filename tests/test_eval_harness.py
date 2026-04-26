import pytest

from scripts.eval_harness import run


@pytest.mark.integration
async def test_eval_correctly_predicts_majority(seeded_db):
    out = await run()
    correct = sum(1 for t, p in zip(out["y_true"], out["y_pred"]) if t == p)
    assert correct >= 7  # at least 7 of 10 traders correctly classified
