"""Stratified k-fold cross-validation over the synthetic dataset, with bootstrap
confidence intervals on macro-F1.

This is a methodology check — does our pipeline give consistent F1 across
re-shuffles? The answer informs whether the held-out F1 in `eval_harness.py
--dataset synthetic_test` is a stable measure or a lucky split.

Usage:
    python -m scripts.cv_eval --dataset data/synthetic_dataset.json \\
        --folds 5 --seed 42 --bootstrap 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from app.profiling.rules import score_pathologies
from scripts.feature_extractor import trader_trades

LABELS = [
    "revenge_trading", "overtrading", "fomo_entries", "plan_non_adherence",
    "premature_exit", "loss_running", "session_tilt", "time_of_day_bias",
    "position_sizing_inconsistency", "none",
]


def _label_of(trader: dict) -> str:
    g = trader.get("groundTruthPathologies") or []
    return g[0] if g else "none"


def _predict(trader: dict, min_score: float = 0.3) -> str:
    trades = trader_trades(trader)
    if not trades:
        return "none"
    scored = score_pathologies(trades)
    top = scored[0]
    return top["pathology"] if top["score"] >= min_score else "none"


def _bootstrap_ci(
    y_true: list[str], y_pred: list[str], n: int, rng: np.random.Generator,
) -> tuple[float, float, float]:
    arr_y = np.array(y_true)
    arr_p = np.array(y_pred)
    f1s = []
    for _ in range(n):
        idx = rng.integers(0, len(arr_y), size=len(arr_y))
        f = f1_score(arr_y[idx], arr_p[idx], labels=LABELS, average="macro", zero_division=0)
        f1s.append(f)
    return (
        float(np.percentile(f1s, 2.5)),
        float(np.median(f1s)),
        float(np.percentile(f1s, 97.5)),
    )


def cross_validate(dataset_path: Path, folds: int, seed: int, bootstrap: int) -> dict:
    payload = json.loads(dataset_path.read_text())
    traders = payload["traders"]
    y = [_label_of(t) for t in traders]

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_results = []
    all_y_true: list[str] = []
    all_y_pred: list[str] = []
    for fold_i, (_, test_idx) in enumerate(skf.split(traders, y)):
        fold_y_true = [_label_of(traders[i]) for i in test_idx]
        fold_y_pred = [_predict(traders[i]) for i in test_idx]
        fold_f1 = f1_score(fold_y_true, fold_y_pred, labels=LABELS, average="macro", zero_division=0)
        fold_results.append({"fold": fold_i, "n": len(test_idx), "macro_f1": round(fold_f1, 4)})
        all_y_true.extend(fold_y_true)
        all_y_pred.extend(fold_y_pred)

    rng = np.random.default_rng(seed)
    lo, mid, hi = _bootstrap_ci(all_y_true, all_y_pred, n=bootstrap, rng=rng)

    f1_values = [f["macro_f1"] for f in fold_results]
    return {
        "dataset": str(dataset_path),
        "folds": folds,
        "seed": seed,
        "bootstrap_n": bootstrap,
        "fold_results": fold_results,
        "macro_f1": {
            "fold_mean": round(float(np.mean(f1_values)), 4),
            "fold_std": round(float(np.std(f1_values)), 4),
            "bootstrap_p2_5": round(lo, 4),
            "bootstrap_median": round(mid, 4),
            "bootstrap_p97_5": round(hi, 4),
        },
        "n_traders": len(traders),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("data/synthetic_dataset.json"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--out", type=Path, default=Path("eval/cv_report.json"))
    args = p.parse_args()

    result = cross_validate(args.dataset, args.folds, args.seed, args.bootstrap)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    f1 = result["macro_f1"]
    print(
        f"{result['folds']}-fold CV on {result['n_traders']} traders\n"
        f"  fold mean ± std: {f1['fold_mean']:.4f} ± {f1['fold_std']:.4f}\n"
        f"  bootstrap 95% CI: [{f1['bootstrap_p2_5']:.4f}, {f1['bootstrap_p97_5']:.4f}] "
        f"(median {f1['bootstrap_median']:.4f})\n"
        f"wrote {args.out}"
    )


if __name__ == "__main__":
    main()
