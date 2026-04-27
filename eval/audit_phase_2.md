# Phase 2 CTO Audit — Held-out validation + threshold optimization

**Goal:** Replace seed-tuned magic numbers with thresholds optimized on the synthetic *training* split and validated on the held-out *test* split. Honest cross-validation on top.

## ✅ Goal delivered? — YES

| Check | Result |
|---|---|
| `app/profiling/thresholds.py` exists with one entry per scorer | ✅ — 9 scorer blocks, every parameter named with a `# why` comment |
| `app/profiling/rules.py` reads from `THRESHOLDS` (no inline literals) | ✅ — refactor confirmed by 36 unit tests passing pre-and-post |
| `scripts/feature_extractor.py` produces a feature dict per trader | ✅ |
| `scripts/tune_thresholds.py` runs deterministically and writes JSON | ✅ — `eval/tuned_thresholds.json` produced |
| `scripts/cv_eval.py` runs 5-fold stratified CV with bootstrap CIs | ✅ — `eval/cv_report.json` produced |
| `scripts/eval_harness.py` accepts `--dataset` (seed, synthetic_test, synthetic_full, path) | ✅ |
| `tests/test_eval_harness.py` enforces seed ≥9/10 AND held-out F1 ≥ 0.65 | ✅ |
| `tests/test_threshold_tuner.py` enforces determinism, no global-state leak, no held-out regression | ✅ — 4/4 pass |

## 🔬 Test evidence

```
Refactor proof:
36 unit tests passing pre- and post-refactor (zero behavior drift)

Threshold tuner output (eval/tuned_thresholds.json):
  baseline train macro-F1: 0.9708
  baseline test  macro-F1: 1.0000   (held-out 30 traders)
  tuned    train macro-F1: 1.0000
  tuned    test  macro-F1: 1.0000

  parameter changes:
    plan_non_adherence.ratio_subtract: 0.10 -> 0.15

5-fold stratified CV on full synthetic (n=100, eval/cv_report.json):
  fold mean ± std:    0.9787 ± 0.0261
  bootstrap 95% CI:   [0.9437, 1.0000]   (median 0.9800, n=1000)

Held-out per-class report (eval/holdout_report.json):
  every class: precision=1.0, recall=1.0, F1=1.0, support=3
  accuracy=1.00, macro avg F1=1.00

Test results:
  60 tests pass (up from 36)
    + 4 tuner tests  + 1 held-out F1 test  + 13 generator tests
    + 6 integration tests (seed eval, memory, profile, audit, coaching, etc.)
  1 skipped (opt-in persistence)
```

## 🧱 Code quality

- **`thresholds.py` is parametric and documented.** Every entry has a `# why` comment so a reader can see what each number controls without re-reading the rule body. The dict is opaque to type checkers in the value type (`float`), but consumers fetch named keys not positions, so this is fine.
- **The refactor introduced no behavior drift.** This is the strongest claim we can make — same tests pass with same results, only the source of truth changed.
- **The tuner uses coordinate descent**, which is greedy. It does not explore parameter joint interactions. This is documented in the tuner's docstring; for a 9-pathology × 4-5-value-each grid, a full joint sweep would be 10⁵+ evaluations, infeasible.
- **`tune()` saves and restores `THRESHOLDS`** — the tuner-state-leak test (`test_tuner_does_not_dirty_global_state`) catches future drift here.
- **The `--dataset` flag of eval_harness adds 3 valid string options + an arbitrary path**, with output filenames keyed off the dataset choice. Reviewers can run any flavor without code edits.

## ⚠️ Tech debt introduced

1. **Held-out F1 = 1.0 is high but partially expected.** The synthetic generators are shaped by the rule definitions; they emit traders whose feature signatures the rules detect. So the held-out test is measuring "do the rules generalize within the rule-shaped distribution" — not "do they generalize to real-world traders." This is the honest ceiling for synthetic data and is documented in the test's docstring.
2. **CV variance is small (std 0.026)** because the dataset is clean. If we added noise traders (mixed signals, edge-case adherence patterns), the spread would widen. v0.3 work: noise-injection as part of the generator.
3. **Tuner only adjusts one parameter per pathology.** The grid (`TUNING_GRID`) lists exactly one knob per scorer to keep evaluation cheap. Multiple-parameter tuning would catch more nuanced trade-offs but add combinatorial cost.
4. **Tuner does not auto-apply to `thresholds.py`.** It writes a proposal to `eval/tuned_thresholds.json` and a human reads the audit + diff. This is a deliberate guardrail against accidental drift, but it adds a manual step at deploy time.
5. **The single tuned change** (`plan_non_adherence.ratio_subtract: 0.10 → 0.15`) was not applied to the live `thresholds.py` because held-out F1 is already 1.0 with the baseline. Applying it would help train F1 only — and with only 5 marginal cases on training, the prior 0.10 is just as defensible. **Decision: leave baseline thresholds in place; document tuner output in this audit so the value is captured.**

## 🎯 Phase score: 8.5/10

**+ Why high:**
- Refactor is clean, behavior-preserving, parametric.
- Held-out validation exists with proper train/test stratification.
- CV with bootstrap CIs gives an honest spread, not a single number.
- Tuner is deterministic and verified-no-global-state-leak.
- Test bar raised: seed ≥9/10 (was ≥7) AND held-out ≥0.65.

**− Why not 10:**
- Held-out F1 = 1.0 sits at the ceiling because synthetic data is rule-shaped. The methodology is sound but the fixture is generous. A more adversarial synthetic generator (with mixed signals, weak signatures near gates) would produce a lower, more informative number. The plan addresses some of this in Phase 3 (multi-label dual-pathology traders).
- The tuner finds only 1 productive change because the rules are already well-tuned for the clean distribution. Real value of the tuner emerges when the distribution shifts (e.g., a new dataset with different greedy ratios) — at that point, re-running it is a one-command rollback to optimal.

## 🚦 Go/no-go for Phase 3: GO

The methodology infrastructure is in place: held-out split, CV, tuner, parametric thresholds. Phase 3's multi-label work will exercise this infra against harder traders (two simultaneous pathologies), which is where we expect to see Hamming loss > 0 and meaningful per-class F1 spread.

**Next:** Phase 3 — multi-label `detect_signals`, multi-pathology profile output, multi-signal coaching prompt, and a separate multi-label test set with N=30 dual-pathology traders.
