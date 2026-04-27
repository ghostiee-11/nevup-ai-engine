"""Run the rule-based profiler against a labelled dataset and emit a
sklearn classification report.

Datasets:
- `seed`: the original 10-trader fixture loaded from Postgres (requires DB).
- `synthetic_test`: held-out 30 traders from data/synthetic_dataset.json.
- `synthetic_full`: all 100 synthetic traders.
- a path to any JSON file with the seed-schema shape (also accepts a `splits` field).

Reviewers can run `python -m scripts.eval_harness --dataset seed` (DB-backed)
or `python -m scripts.eval_harness --dataset data/synthetic_dataset.json --split test`
without any API keys.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import MultiLabelBinarizer
from sqlalchemy import select

from app.db import SessionLocal
from app.models import Trade, Trader
from app.profiling.rules import score_pathologies
from scripts.feature_extractor import trade_to_dict, trader_trades

log = logging.getLogger("eval")

PATHOLOGIES = [
    "revenge_trading", "overtrading", "fomo_entries", "plan_non_adherence",
    "premature_exit", "loss_running", "session_tilt", "time_of_day_bias",
    "position_sizing_inconsistency", "none",
]


def _trade_row_to_dict(t: Trade) -> dict:
    return {
        "trade_id": t.trade_id, "user_id": t.user_id, "session_id": t.session_id,
        "asset": t.asset, "asset_class": t.asset_class, "direction": t.direction,
        "entry_price": t.entry_price, "exit_price": t.exit_price, "quantity": t.quantity,
        "entry_at": t.entry_at, "exit_at": t.exit_at, "status": t.status,
        "outcome": t.outcome, "pnl": t.pnl, "plan_adherence": t.plan_adherence,
        "emotional_state": t.emotional_state, "entry_rationale": t.entry_rationale,
        "revenge_flag": t.revenge_flag,
    }


async def _run_seed() -> tuple[list[str], list[str], list[dict]]:
    async with SessionLocal() as db:
        traders = (await db.execute(select(Trader))).scalars().all()
        per_user: dict[str, list[Trade]] = {}
        rows = (await db.execute(select(Trade).order_by(Trade.entry_at))).scalars().all()
        for r in rows:
            per_user.setdefault(r.user_id, []).append(r)

    y_true: list[str] = []
    y_pred: list[str] = []
    details: list[dict] = []
    for trader in traders:
        truth = trader.ground_truth_pathologies[0] if trader.ground_truth_pathologies else "none"
        trades = [_trade_row_to_dict(t) for t in per_user.get(trader.user_id, [])]
        scored = score_pathologies(trades) if trades else [{"pathology": "none", "score": 0.0}]
        top = scored[0]
        pred = top["pathology"] if top["score"] >= 0.3 else "none"
        y_true.append(truth)
        y_pred.append(pred)
        details.append({
            "userId": trader.user_id, "name": trader.name,
            "truth": truth, "pred": pred, "topScore": top["score"],
        })
    return y_true, y_pred, details


def _run_json(path: Path, split: str | None = None) -> tuple[list[str], list[str], list[dict]]:
    payload = json.loads(path.read_text())
    traders = payload["traders"]
    if split and "splits" in payload:
        ids = set(payload["splits"][split])
        traders = [t for t in traders if t["userId"] in ids]

    y_true: list[str] = []
    y_pred: list[str] = []
    details: list[dict] = []
    for trader in traders:
        truth = trader["groundTruthPathologies"][0] if trader["groundTruthPathologies"] else "none"
        trades = trader_trades(trader)
        scored = score_pathologies(trades) if trades else [{"pathology": "none", "score": 0.0}]
        top = scored[0]
        pred = top["pathology"] if top["score"] >= 0.3 else "none"
        y_true.append(truth)
        y_pred.append(pred)
        details.append({
            "userId": trader["userId"], "name": trader.get("name"),
            "truth": truth, "pred": pred, "topScore": top["score"],
        })
    return y_true, y_pred, details


async def run(dataset: str = "seed", split: str | None = None) -> dict:
    if dataset == "seed":
        y_true, y_pred, details = await _run_seed()
        out_name = "report.json"
    elif dataset == "synthetic_test":
        y_true, y_pred, details = _run_json(Path("data/synthetic_dataset.json"), split="test")
        out_name = "holdout_report.json"
    elif dataset == "synthetic_full":
        y_true, y_pred, details = _run_json(Path("data/synthetic_dataset.json"))
        out_name = "synthetic_report.json"
    else:
        # treat as a path
        y_true, y_pred, details = _run_json(Path(dataset), split=split)
        out_name = "custom_report.json"

    report = classification_report(y_true, y_pred, labels=PATHOLOGIES, zero_division=0, output_dict=True)
    out = {
        "dataset": dataset,
        "split": split,
        "report": report,
        "details": details,
        "y_true": y_true,
        "y_pred": y_pred,
    }
    Path("eval").mkdir(exist_ok=True)
    Path(f"eval/{out_name}").write_text(json.dumps(out, indent=2))
    return out


def run_multi_label(path: Path, score_threshold: float = 0.3) -> dict:
    """Multi-label evaluation: predict the set of pathologies with score >= threshold,
    compare against the trader's full ground-truth set. Reports subset accuracy,
    Hamming loss, and per-class precision/recall/F1.
    """
    payload = json.loads(path.read_text())
    traders = payload["traders"]
    label_classes = [
        "revenge_trading", "overtrading", "fomo_entries", "plan_non_adherence",
        "premature_exit", "loss_running", "session_tilt", "time_of_day_bias",
        "position_sizing_inconsistency",
    ]
    mlb = MultiLabelBinarizer(classes=label_classes)

    y_true_sets: list[list[str]] = []
    y_pred_sets: list[list[str]] = []
    details: list[dict] = []
    for trader in traders:
        truth = list(trader.get("groundTruthPathologies") or [])
        trades = trader_trades(trader)
        scored = score_pathologies(trades) if trades else []
        pred = [s["pathology"] for s in scored if s["score"] >= score_threshold]
        y_true_sets.append(truth)
        y_pred_sets.append(pred)
        details.append({
            "userId": trader["userId"],
            "name": trader.get("name"),
            "truth": truth,
            "pred": pred,
            "subset_match": set(truth) == set(pred),
        })

    y_true_bin = mlb.fit_transform(y_true_sets)
    y_pred_bin = mlb.transform(y_pred_sets)

    subset_acc = sum(1 for d in details if d["subset_match"]) / len(details)
    h_loss = hamming_loss(y_true_bin, y_pred_bin)
    macro_f1 = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    p, r, f, s = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, labels=list(range(len(label_classes))), zero_division=0,
    )
    per_class = {
        cls: {"precision": round(p[i], 4), "recall": round(r[i], 4), "f1": round(f[i], 4), "support": int(s[i])}
        for i, cls in enumerate(label_classes)
    }

    out = {
        "dataset": str(path),
        "score_threshold": score_threshold,
        "n_traders": len(traders),
        "subset_accuracy": round(subset_acc, 4),
        "hamming_loss": round(h_loss, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "per_class": per_class,
        "details": details,
    }
    Path("eval").mkdir(exist_ok=True)
    Path("eval/multi_label_report.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="seed",
                   help="'seed', 'synthetic_test', 'synthetic_full', or a JSON path")
    p.add_argument("--split", default=None, help="train|test (when --dataset is a JSON path)")
    p.add_argument("--multi-label", action="store_true",
                   help="multi-label mode: predict ALL pathologies above threshold per trader")
    p.add_argument("--score-threshold", type=float, default=0.3)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.multi_label:
        out = run_multi_label(Path(args.dataset), score_threshold=args.score_threshold)
        print(json.dumps({k: v for k, v in out.items() if k != "details"}, indent=2))
    else:
        out = asyncio.run(run(dataset=args.dataset, split=args.split))
        print(json.dumps(out["report"], indent=2))


if __name__ == "__main__":
    main()
