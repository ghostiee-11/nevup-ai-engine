"""Per-pathology threshold tuner using coordinate descent on macro-F1.

For each pathology, sweep its single most-impactful gate parameter over a small
discrete grid; pick the value that maximizes macro-F1 *on the training split*.
After all pathologies tuned, report tuned thresholds + train and held-out
macro-F1 vs baseline.

Usage:
    python -m scripts.tune_thresholds \\
        --dataset data/synthetic_dataset.json --seed 42 \\
        --out eval/tuned_thresholds.json

The tuner does NOT mutate `app/profiling/thresholds.py` — it writes a JSON
proposal that a human applies after reading `eval/audit_phase_2.md`.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score

from app.profiling import thresholds as thresholds_mod
from app.profiling.rules import score_pathologies
from scripts.feature_extractor import trader_trades

# Single most-impactful gate per pathology + grid.
TUNING_GRID: dict[str, dict[str, list[float]]] = {
    "revenge_trading": {"denom_factor": [0.10, 0.15, 0.20, 0.25, 0.30]},
    "overtrading": {"events_multiplier": [0.30, 0.35, 0.40, 0.45, 0.50]},
    "fomo_entries": {"greedy_dominance_min": [0.50, 0.55, 0.60, 0.65, 0.70]},
    "plan_non_adherence": {"ratio_subtract": [0.05, 0.10, 0.15, 0.20]},
    "premature_exit": {"win_rate_min": [0.60, 0.65, 0.70, 0.75, 0.80]},
    "loss_running": {"ratio_subtract": [0.30, 0.35, 0.40, 0.45, 0.50]},
    "session_tilt": {"anxious_ratio_min": [0.40, 0.50, 0.60]},
    "time_of_day_bias": {"loss_rate_min": [0.55, 0.60, 0.65, 0.70, 0.75]},
    "position_sizing_inconsistency": {"max_cv_min": [0.65, 0.75, 0.85, 0.95]},
}


def _predict(trader_trade_lists: list[tuple[str, list[dict]]], min_score: float = 0.3) -> list[str]:
    """Return top-1 prediction per trader, defaulting to 'none' if top score below threshold."""
    preds = []
    for _, trades in trader_trade_lists:
        scored = score_pathologies(trades) if trades else [{"pathology": "none", "score": 0.0}]
        top = scored[0]
        preds.append(top["pathology"] if top["score"] >= min_score else "none")
    return preds


def _macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    return f1_score(y_true, y_pred, labels=_LABELS, average="macro", zero_division=0)


_LABELS = list(TUNING_GRID.keys()) + ["none"]


def _label_of(trader: dict) -> str:
    g = trader.get("groundTruthPathologies") or []
    return g[0] if g else "none"


def tune(dataset_path: Path) -> dict:
    payload = json.loads(dataset_path.read_text())
    train_ids = set(payload["splits"]["train"])
    test_ids = set(payload["splits"]["test"])

    train_traders = [t for t in payload["traders"] if t["userId"] in train_ids]
    test_traders = [t for t in payload["traders"] if t["userId"] in test_ids]

    train_data = [(_label_of(t), trader_trades(t)) for t in train_traders]
    test_data = [(_label_of(t), trader_trades(t)) for t in test_traders]
    train_y = [lbl for lbl, _ in train_data]
    test_y = [lbl for lbl, _ in test_data]

    # Snapshot the original thresholds so we can restore them later.
    original = deepcopy(thresholds_mod.THRESHOLDS)
    baseline_train_f1 = _macro_f1(train_y, _predict(train_data))
    baseline_test_f1 = _macro_f1(test_y, _predict(test_data))

    # Coordinate-descent: tune each pathology while keeping others at current-best.
    tuned: dict[str, dict[str, float]] = {p: dict(original[p]) for p in original}
    deltas: list[dict] = []

    for pathology, params in TUNING_GRID.items():
        for param, candidates in params.items():
            best_value = original[pathology][param]
            best_f1 = baseline_train_f1
            for v in candidates:
                thresholds_mod.THRESHOLDS[pathology][param] = v
                cand_f1 = _macro_f1(train_y, _predict(train_data))
                if cand_f1 > best_f1 + 1e-6:  # require strict improvement
                    best_f1 = cand_f1
                    best_value = v
            # Lock in the best, advance the baseline F1 for subsequent pathologies.
            thresholds_mod.THRESHOLDS[pathology][param] = best_value
            tuned[pathology][param] = best_value
            deltas.append({
                "pathology": pathology,
                "param": param,
                "from": original[pathology][param],
                "to": best_value,
                "train_macro_f1_after": round(best_f1, 4),
            })
            baseline_train_f1 = best_f1  # accept the improvement (or no-op if tied)

    final_train_f1 = _macro_f1(train_y, _predict(train_data))
    final_train_pred = _predict(train_data)
    final_test_pred = _predict(test_data)
    final_test_f1 = _macro_f1(test_y, final_test_pred)

    # Restore originals before returning so the in-process state isn't dirty.
    for p in original:
        thresholds_mod.THRESHOLDS[p] = dict(original[p])

    return {
        "baseline": {
            "train_macro_f1": round(_macro_f1(train_y, _predict(train_data)), 4),  # original
            "test_macro_f1": round(baseline_test_f1, 4),
        },
        "tuned_thresholds": tuned,
        "deltas": deltas,
        "final": {
            "train_macro_f1": round(final_train_f1, 4),
            "test_macro_f1": round(final_test_f1, 4),
            "test_classification_report": classification_report(
                test_y, final_test_pred, labels=_LABELS, zero_division=0, output_dict=True,
            ),
        },
        "splits": {
            "train_n": len(train_traders),
            "test_n": len(test_traders),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("eval/tuned_thresholds.json"))
    p.add_argument("--seed", type=int, default=42, help="reserved for compatibility")
    args = p.parse_args()
    np.random.seed(args.seed)

    result = tune(args.dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(
        f"baseline train F1: {result['baseline']['train_macro_f1']}  "
        f"baseline test F1:  {result['baseline']['test_macro_f1']}\n"
        f"tuned    train F1: {result['final']['train_macro_f1']}  "
        f"tuned    test F1:  {result['final']['test_macro_f1']}\n"
        f"wrote {args.out}"
    )


if __name__ == "__main__":
    main()
