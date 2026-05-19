# ASTRID Experiments

This directory contains reproducible experiment runners for turning ASTRID from
an interactive audit tool into a publishable evaluation framework.

## Tabular Fault Injection

`fault_injection_tabular.py` creates a deterministic synthetic industrial
dataset, injects one controlled reliability fault at a time, runs the headless
ASTRID tabular analyzer, and exports summary tables for paper figures.

Run from the repository root:

```bash
python experiments/fault_injection_tabular.py
```

Useful options:

```bash
python experiments/fault_injection_tabular.py \
  --rows 8000 \
  --severities 0,0.01,0.05,0.10,0.20,0.40 \
  --save-reports
```

Generated outputs are written to:

- `experiments/results/tabular_fault_injection/tabular_fault_injection_summary.csv`
- `experiments/results/tabular_fault_injection/tabular_fault_injection_results.json`
- `experiments/results/tabular_fault_injection/reports/*.json` when `--save-reports` is set

The generated `results/` directory is ignored by git so repeated runs do not
pollute the repository. Archive selected outputs separately when preparing a
paper artifact.

## Current Faults

The initial runner covers six controlled tabular failure modes:

- `missingness`: random missing numeric cells.
- `duplicates`: exact duplicate row insertion.
- `split_leakage`: duplicated samples moved across train/test splits.
- `drift`: feature shifts in the final temporal slice.
- `pii`: synthetic email-like strings in text fields.
- `fairness`: group-dependent positive-rate disparity.

For each run, the summary records the ASTRID score, grade, policy-gate status,
primary metric value, detection flag, recommendation coverage flag, and runtime.

## Paper Use

This supports the first empirical research question in
`docs/paper_blueprint.md`: whether ASTRID detects controlled reliability
failures across severity levels. The next experiment layers should add:

1. Time-series and image fault injection.
2. Baseline tool comparisons per fault category.
3. Downstream model degradation correlation.
4. Score sensitivity and threshold calibration.
