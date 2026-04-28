# Methodology

How we evaluate the rule-based pathology profiler. The headline is: **methodology before metrics**. A 10/10 number on a fixture-overfit dataset is meaningless; this document explains the evidence we generate to keep ourselves honest.

## Datasets

| Dataset | N traders | Source | Used for |
|---|---|---|---|
| `seed` | 10 | Brief fixture (Postgres-loaded) | Sanity check; `correct ≥ 9` enforced in tests |
| `synthetic_full` | 100 (10 per pathology + 10 controls) | [`scripts/generate_synthetic_traders.py`](../scripts/generate_synthetic_traders.py) seed=42 | Train/test methodology demonstration |
| `synthetic_test` | 30 (3 per class) | Stratified 30% holdout from `synthetic_full` | **Held-out** macro-F1 (the honest metric) |
| `multi_label_test` | 30 dual-pathology | [`scripts/generate_multi_label_traders.py`](../scripts/generate_multi_label_traders.py) seed=7 | Multi-label Hamming + per-class F1 |

### How synthetic data is generated

Each pathology has a dedicated session generator that emits trades with the *signature* the rule actually detects. The generator does NOT borrow the rule — it expresses the same domain definition independently. A parametrized contract test (`tests/test_synthetic_generator.py::test_each_generator_fires_its_target_rule`) asserts every labelled trader scores ≥ 0.3 on its target pathology — surfacing definition drift between rules and generators when it happens.

Default sizing is target-notional ($1000-2000 USD per trade); only the position-sizing-inconsistency generator opts out and varies notional deliberately. This means most synthetic traders look "disciplined in dollars," matching what real risk-aware traders look like.

## Train/test split

```
synthetic_full (n=100)
├── train (n=70, stratified 7 per class) ← threshold tuner uses ONLY this
└── test  (n=30, stratified 3 per class) ← eval harness reports macro-F1 here
```

Split is stored inside the JSON payload (`splits.train`, `splits.test`) so the same userIds are always in the same fold across re-runs.

## Threshold tuner

Coordinate descent. For each pathology, sweep its single most-impactful gate parameter ([`scripts/tune_thresholds.py::TUNING_GRID`](../scripts/tune_thresholds.py)) over a small discrete grid; pick the value that maximizes macro-F1 on the *training split*. After all 9 pathologies tuned, evaluate on held-out test.

The tuner does not mutate `app/profiling/thresholds.py` — it writes a JSON proposal to `eval/tuned_thresholds.json` for human review. Determinism, no global-state leak, and no held-out regression are unit-tested in `tests/test_threshold_tuner.py`.

## Metrics

### Single-label (`scripts/eval_harness.py`)

- **Accuracy** = correct / total. Used in seed eval (10 traders, 1 dimension).
- **Macro-F1** = unweighted mean of per-class F1. Used on synthetic_test (more classes, balanced).

### Multi-label (`scripts/eval_harness.py --multi-label`)

- **Subset accuracy** — exact-set match. Harsh: even one missed/extra label fails the trader.
- **Hamming loss** — per-(trader, label) error rate. Lower is better.
- **Macro-F1 / Micro-F1** — unweighted vs weighted-by-frequency.
- **Per-class precision/recall/F1** with support.

### Cross-validation (`scripts/cv_eval.py`)

5-fold stratified CV on `synthetic_full`, plus 1000-iteration bootstrap for 95% CI on macro-F1. This is what produces the "0.9787 ± 0.0261, CI [0.94, 1.00]" in the README.

## Honest results (v0.2.0)

| Surface | Number | Comment |
|---|---|---|
| Seed top-1 | 10/10 | Original brief fixture; intentionally tuned for |
| Synthetic_test macro-F1 | 1.00 | At ceiling because generators are rule-shaped |
| 5-fold CV macro-F1 | 0.9787 ± 0.0261 | Real spread; 95% bootstrap CI [0.9437, 1.0000] |
| Multi-label Hamming | 0.237 | Below 0.25 floor (good) |
| Multi-label micro-F1 | 0.59 | Above 0.55 floor; some single-rule gates fail in dual-label settings (especially `fomo_entries` and `time_of_day_bias`) |
| Multi-label subset accuracy | 0.0 | Architectural limit of hand-written gating; v0.3 work to replace with learned classifier |
| Load test SSE first-byte p95 | 512ms | **Misses 400ms target by 28%**. Three v0.3 fixes identified |
| Load test correctness | 100% (1988 reqs) | All requests succeeded with `event: done` terminator |

## Reproducing locally

```bash
# 1. Boot DB
docker compose up -d db

# 2. Generate synthetic + multi-label datasets
python -m scripts.generate_synthetic_traders --n-per-class 10 --seed 42 --out data/synthetic_dataset.json
python -m scripts.generate_multi_label_traders --n 30 --seed 7 --out data/multi_label_test.json

# 3. Run all evals
python -m scripts.eval_harness --dataset seed
python -m scripts.eval_harness --dataset synthetic_test
python -m scripts.cv_eval --folds 5 --seed 42 --bootstrap 1000
python -m scripts.eval_harness --multi-label --dataset data/multi_label_test.json

# 4. Load test
loadtest/k6_run.sh
```

Reports land in `eval/`. None of these require a Gemini or Groq key (the rule pipeline is pure-Python).

## What we did NOT measure (and what would matter)

- **Real-world held-out**. Synthetic data has rule-shaped distributions. The right next step is a labelled real-world dataset; we do not have one.
- **Multi-task fine-tuned coaching prompt quality**. The `SYSTEM_MULTI` prompt is engineered, not measured. A golden-set of expected coaching responses would let us regression-test prompt quality.
- **Inter-rater reliability of pathology labels**. Pathologies in real trading are subjective; a single ground-truth label assumes consensus. v0.3.
- **Performance numbers against live Render free tier** (only against local docker). Render's 0.1 CPU / 512MB free tier saturates at ~3 concurrent SSE clients; we do not include those numbers because they reflect infrastructure, not application.
