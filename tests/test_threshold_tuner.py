"""Asserts the tuner is deterministic and reports honest deltas."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.tune_thresholds import TUNING_GRID, tune


@pytest.fixture(scope="module")
def dataset_path() -> Path:
    p = Path("data/synthetic_dataset.json")
    if not p.exists():
        pytest.skip("synthetic dataset missing — run scripts.generate_synthetic_traders")
    return p


def test_tuner_is_deterministic(dataset_path: Path) -> None:
    """Same input + same starting thresholds must produce the same proposal."""
    a = tune(dataset_path)
    b = tune(dataset_path)
    assert a["tuned_thresholds"] == b["tuned_thresholds"]
    assert a["final"]["train_macro_f1"] == b["final"]["train_macro_f1"]
    assert a["final"]["test_macro_f1"] == b["final"]["test_macro_f1"]


def test_tuner_does_not_regress_held_out(dataset_path: Path) -> None:
    """Tuned thresholds may not lower held-out macro-F1 vs baseline."""
    result = tune(dataset_path)
    assert result["final"]["test_macro_f1"] >= result["baseline"]["test_macro_f1"] - 1e-6


def test_tuner_does_not_dirty_global_state(dataset_path: Path) -> None:
    """The tuner restores `app.profiling.thresholds.THRESHOLDS` after running."""
    from app.profiling.thresholds import THRESHOLDS
    snapshot = {p: dict(v) for p, v in THRESHOLDS.items()}
    tune(dataset_path)
    after = {p: dict(v) for p, v in THRESHOLDS.items()}
    assert snapshot == after, "tuner left THRESHOLDS modified — global-state leak"


def test_grid_covers_all_pathologies() -> None:
    """Every scored pathology must have at least one tunable parameter in the grid."""
    expected = {
        "revenge_trading", "overtrading", "fomo_entries", "plan_non_adherence",
        "premature_exit", "loss_running", "session_tilt", "time_of_day_bias",
        "position_sizing_inconsistency",
    }
    assert set(TUNING_GRID.keys()) == expected
