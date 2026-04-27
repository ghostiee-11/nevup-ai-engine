"""Centralized thresholds for the rule-based pathology scorers.

Why a separate module? The scorers in `app/profiling/rules.py` historically had
literal numbers inline, hand-tuned against the 10-trader seed dataset. That is
fixture-overfit. Phase 2's tuner (`scripts/tune_thresholds.py`) sweeps these
values on the synthetic *training* split and reports proposed updates that are
applied here only after holdout F1 is verified.

Each entry under `THRESHOLDS[pathology]` documents one tunable parameter with
a one-line `# why` comment so a future engineer can tell what each number
controls without reading the rule body.

Default values below match the pre-Phase-2 production behavior (zero drift).
"""
from __future__ import annotations

from copy import deepcopy

THRESHOLDS: dict[str, dict[str, float]] = {
    "revenge_trading": {
        # Score is len(cites) divided by max(1, len(trades) * denom_factor); a smaller
        # denom means even a few flagged trades push the score to 1.0.
        "denom_factor": 0.2,
    },
    "overtrading": {
        # Trades per 30-min window above this trip a violation event.
        "max_in_30min": 10,
        # Score formula: events * events_multiplier + max_excess * excess_multiplier.
        "events_multiplier": 0.4,
        "excess_multiplier": 0.05,
    },
    "fomo_entries": {
        # Fraction of trades in greedy state required to even consider this label.
        "greedy_dominance_min": 0.6,
        # Multiplier applied to (greedy_low_adh / total) to scale into [0,1].
        "score_multiplier": 1.2,
    },
    "plan_non_adherence": {
        # Loss-rate gate prevents disciplined-but-low-adherence noise from firing.
        "loss_rate_min": 0.15,
        # Score: max(0, ratio - subtract) * multiplier, where ratio = non_greedy_low / rated.
        "ratio_subtract": 0.1,
        "score_multiplier": 2.0,
    },
    "premature_exit": {
        # Win rate must be high (overall good outcomes coming from cutting winners early).
        "win_rate_min": 0.7,
        # A "quick" win is one that exited within this many minutes of entry.
        "quick_minutes_max": 10,
        # Of all wins, this fraction must be quick to qualify.
        "quick_ratio_to_wins_min": 0.5,
        # Score: win_rate * quick_ratio_overall * multiplier.
        "score_multiplier": 1.2,
    },
    "loss_running": {
        # Avoid noise on traders with too few losses to characterize.
        "min_losses": 5,
        # A "very long" loss is held at least this many hours.
        "very_long_hours": 2,
        # Score: max(0, ratio - subtract) * multiplier, where ratio = very_long_non_greedy / total_losses.
        "ratio_subtract": 0.4,
        "score_multiplier": 2.5,
    },
    "session_tilt": {
        # Overall loss rate gate — tilt is only meaningful when losing dominates.
        "min_loss_rate": 0.5,
        # Per-session minimum trade count to count toward tilt.
        "min_session_trades": 3,
        # Per-session loss-following ratio to call the session "tilted".
        "loss_following_min": 0.5,
        # Of the loss-following trades in a session, fraction that must be anxious/fearful.
        "anxious_ratio_min": 0.5,
    },
    "time_of_day_bias": {
        # Need enough trades total to bucket reliably by hour.
        "min_total_trades": 10,
        # Within an hour, need at least this many trades to consider it bad.
        "min_trades_per_hour": 3,
        # Loss rate at-or-above which an hour is flagged bad.
        "loss_rate_min": 0.7,
        # If bad-hour trades dominate >this fraction of all trades, this is just a bad
        # trader, not a time-of-day pattern — reject.
        "max_bad_ratio": 0.65,
        # Score: min(1, len(bad_hours)/3) * (bad/total) * multiplier.
        "score_multiplier": 1.5,
    },
    "position_sizing_inconsistency": {
        # Per-asset-class minimum sample to compute coefficient of variation.
        "min_trades_per_class": 3,
        # Coefficient of variation cutoff — below this, sizing is consistent.
        "max_cv_min": 0.85,
        # Score: max_cv - score_subtract.
        "score_subtract": 0.5,
    },
}


def get_thresholds() -> dict[str, dict[str, float]]:
    """Return a deep copy so callers can't accidentally mutate the canonical dict."""
    return deepcopy(THRESHOLDS)
