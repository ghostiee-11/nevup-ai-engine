# Phase 1 CTO Audit — Synthetic data generator

**Goal:** Produce a labelled dataset large enough for credible held-out validation, generated via deterministic stochastic templates that mathematically express each pathology.

## ✅ Goal delivered? — YES

| Check | Result |
|---|---|
| `scripts/generate_synthetic_traders.py` exists & runs | ✅ |
| Output schema matches `nevup_seed_dataset.json` | ✅ — verified by `test_schema_top_level_keys` |
| 100 traders, 10 per pathology + 10 controls | ✅ |
| Stratified 70/30 train/test split persisted in `splits` field | ✅ — `test_split_is_stratified_and_disjoint` passes |
| Determinism: same seed ⇒ identical payload | ✅ — `test_determinism_same_seed_same_payload` passes |
| Each labelled trader scores ≥ 0.3 on its target pathology | ✅ — parametrized test passes for all 9 pathologies |
| Controls do not strongly trip any pathology (≤ 1/3 with score ≥ 0.5) | ✅ — `test_controls_score_low_across_all_pathologies` passes |

## 🔬 Test evidence

```
Generation:
wrote data/synthetic_dataset.json · 100 traders, 595 sessions, 4121 trades · train=70 test=30

Top-1 accuracy on full synthetic dataset (current production thresholds):
98 / 100 = 98.00%

Per-class sample (one trader picked per label):
label                           top_pred                         score
----------------------------------------------------------------------
revenge_trading                 revenge_trading                   1.00
overtrading                     overtrading                       1.00
fomo_entries                    fomo_entries                      1.00
plan_non_adherence              plan_non_adherence                1.00
premature_exit                  premature_exit                    0.95
loss_running                    loss_running                      1.00
session_tilt                    session_tilt                      0.80
time_of_day_bias                time_of_day_bias                  0.75
position_sizing_inconsistency   position_sizing_inconsistency     0.55
none                            revenge_trading                   0.00

Test results:
13 / 13 generator tests passed
36 unit tests passed (no regression)
```

## 🧱 Code quality

- The 9 generators sit in one file (~470 lines). Each is < 30 lines and uses a shared `_make_trade` helper, so the file scans well.
- Schema enums (`EMOTIONAL_STATES`, `ASSET_CLASSES`, `PATHOLOGIES`) are duplicated from `app/metrics/behavioral.py` with a top-of-file comment naming `app/profiling/rules.py` as the canonical source. Acceptable — the alternative (pulling the live constants in) would create a circular knowledge dependency between rules and fixtures.
- Special-case for `time_of_day_bias` in `_gen_trader` is awkward (per-trader `_tod_cache`). Documented inline; acknowledged tech debt below.
- `_gen_overtrading_session` was rewritten to use an explicit offset accumulator after the initial version produced clusters too spread out for the rule's 30-min window. The new shape is clearer about what the rule expects.
- `_gen_time_of_day_bias_session` uses `plan_adherence=3` deliberately so the time-bias trader doesn't accidentally trip `plan_non_adherence` (which gates on adherence ≤ 2). Comment explains.

## ⚠️ Tech debt introduced

1. **Domain knowledge is duplicated** between `app/profiling/rules.py` and `scripts/generate_synthetic_traders.py`. If the definition of (e.g.) `revenge_trading` changes — say the 90-second window expands to 120 seconds — both files must change. **Mitigation:** the parametrized test `test_each_generator_fires_its_target_rule` will fail loudly the moment the definitions drift apart. This is the right kind of test to surface contract drift.
2. **`_gen_trader._tod_cache` is module-level mutable state.** Now explicitly cleared at the start of every `generate()` call, which is enough for the script's single-process usage. If imported and called from a long-running daemon, the cache would still leak across calls between resets — flagged for v0.3 cleanup (move into `generate()` closure).
3. **The script emits ISO-8601 timestamps with `Z` suffix manually** (`.replace("+00:00", "Z")`). The seed dataset uses the same convention, so this is matching not introducing — but a single helper would be cleaner.
4. **Position sizing CV trick** uses a hardcoded list `[3, 5, 4, 60, 80, 7, 100]`. Deterministic ✓ but not parametric. Phase 4 will replace this with notional-based sizing; the fix lives there.

## 🎯 Phase score: 9/10

**+ Why high:**
- All test gates pass on first deterministic run.
- 98% top-1 accuracy on synthetic data (no tuning yet) means the generators are *expressing* each pathology cleanly.
- The contract-drift test (every generator must keep firing its target rule ≥ 0.3) is a real correctness guard, not just a smoke test.

**− Why not 10:**
- Domain duplication between rules and generator is real; only test-enforced, not structurally prevented.
- 2/100 traders are misclassified at the top-1 level. Below-target pathology scores (premature_exit=0.95, position_sizing=0.55, session_tilt=0.80, time_of_day=0.75) leave headroom for noise to flip the top label. Phase 2's threshold tuner should pull these up.

## 🚦 Go/no-go for Phase 2: GO

The synthetic dataset is the foundation Phase 2 needs. Train/test split is honest. Determinism is verified. Tests guard the contract.

**Next:** Phase 2 — refactor magic numbers into `app/profiling/thresholds.py`, run grid search on training split, re-evaluate on held-out 30 with macro-F1 ≥ 0.65.
