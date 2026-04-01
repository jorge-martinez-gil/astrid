---
title: Time Series Dataset Analyzer
---

# Time Series Dataset Analyzer

Temporal dataset analyzer focused on stability and drift.

---

## Method behavior (granular)

| Method | What it does | How it behaves (step-by-step) | Output |
|--------|-------------|-------------------------------|--------|
| `guess_ts_columns` | Suggests roles | • Scores columns for time, entity, label<br>• Uses datetime parsing and cardinality | Dict |
| `series_time_profile` | Analyzes time axis | • Parsing success<br>• Duplicate timestamps<br>• Cadence stats<br>• Gap detection | Dict |
| `time_slice_labels` | Creates slices | • Converts timestamps to periods (month/week/day) | Series |
| `drift_ks_first_last` | Measures drift | • Compares first vs last slice<br>• KS statistics per feature | Dict |
| `assess_quality` | Evaluates quality | • Missingness, duplicates<br>• Label stats<br>• Time axis health | Dict |
| `assess_reliability` | Evaluates stability | • Slice-based missingness<br>• KS drift<br>• Schema consistency | Dict |
| `assess_robustness` | Detects anomalies | • MAD anomaly scoring<br>• Cadence irregularity<br>• Gap stress | Dict |
| `assess_fairness` | Evaluates disparities | • Group distribution<br>• Missingness gaps<br>• Outcome disparity | Dict |
| `assess_security` | Detects risks | • PII scan<br>• Hash + size | Dict |
| `assess_all` | Runs pipeline | • Executes all checks | Dict |
| `verdict_panel` | Final decision | • Flags drift, parsing issues, PII | Verdict |
| `build_recommendations` | Suggests actions | • Cleaning, resampling, retraining | List |

---

[← Back to home](./index.html)