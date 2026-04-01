---
title: Tabular Dataset Analyzer
---

# Tabular Dataset Analyzer

Structured dataset analyzer with checks for quality, reliability, robustness, fairness, and security.

---

## Method behavior (granular)

### Dataset Analyzer — Method Behavior

| Method | What it does | How it behaves (step-by-step) | Output |
|--------|-------------|-------------------------------|--------|
| `guess_columns` | Suggests column roles | • Computes uniqueness ratios and datetime parsing success<br>• Scores column names using regex patterns<br>• Prefers low-cardinality labels and high-uniqueness IDs<br>• Selects top candidates | Dict |
| `detect_task_type` | Infers ML task | • Uses label dtype and number of unique values<br>• Classifies as regression, binary, or multi-class | String |
| `assess_quality` | Evaluates data quality | • Missingness, duplicates, ID duplication<br>• Outliers (IQR)<br>• Label stats<br>• Annotator agreement<br>• Split leakage via hashing | Dict |
| `assess_reliability` | Evaluates stability | • Uses time or split slicing<br>• Missingness per slice<br>• KS drift first vs last<br>• Schema consistency | Dict |
| `assess_robustness` | Detects fragile patterns | • Rare category-label dominance<br>• MAD anomaly scoring<br>• Logistic regression AUC (if binary) | Dict |
| `assess_fairness` | Evaluates disparities | • Group representation<br>• Missingness gaps<br>• Outcome disparity (binary labels) | Dict |
| `assess_security` | Detects risks | • PII pattern scan<br>• Dataset hash<br>• File size | Dict |
| `assess_all` | Runs pipeline | • Executes all checks<br>• Aggregates results | Dict |
| `verdict_panel` | Final decision | • Flags leakage, drift, PII<br>• Returns OK or Needs review | Verdict |
| `build_recommendations` | Suggests actions | • Maps issues to fixes (cleaning, re-splitting, masking) | List |

---

[← Back to home](./index.html)