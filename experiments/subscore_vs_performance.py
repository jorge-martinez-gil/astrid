"""Block B (correctly scoped): per-dimension sub-score vs downstream AUC.
Scope matches the paper's headline protocol: datasets with DESTROYABLE signal
(cylinder_bands, robot), so boundary cases (APS near-separable, SECOM near-chance)
do not invert the pooled sign. Compares Composite H vs Quality vs Robustness vs Q+Rob.
"""
import numpy as np, pandas as pd
from scipy import stats

R = "/sessions/epic-trusting-cray/mnt/astrid/experiments/results/reliability_vs_performance/"
runs = pd.read_csv(R + "runs_with_lr.csv")
SIGNAL = ["cylinder_bands", "robot"]
DETECT = ["missingness", "outliers", "duplicates", "combined"]

def zwithin(df, col):
    return df.groupby("dataset")[col].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))

def pooled(df, x, y):
    d = df.dropna(subset=[x, y]).copy()
    if d[x].nunique() < 2: return (len(d), np.nan, np.nan, np.nan, np.nan)
    zx, zy = zwithin(d, x), zwithin(d, y)
    pr, pp = stats.pearsonr(zx, zy); sr, sp = stats.spearmanr(zx, zy)
    return len(d), pr, pp, sr, sp

for axis_name, faults in [("combined axis only", ["combined"]), ("all detectable faults", DETECT)]:
    base = runs[(runs.dataset.isin(SIGNAL)) & (runs.fault.isin(faults))].copy()
    base["qrob"] = base["q"] + base["rob"]
    print(f"\n===== Block B — {axis_name} | datasets={SIGNAL} | n={len(base)} =====")
    out = []
    for model, col in [("GBM (robust)", "test_auc"), ("LR (sensitive)", "lr_auc")]:
        for name, c in [("Composite H", "astrid_score"), ("Quality", "q"),
                        ("Robustness", "rob"), ("Quality+Robustness", "qrob")]:
            n, pr, pp, sr, sp = pooled(base, c, col)
            out.append([model, name, n, pr, pp, sr, sp])
    df = pd.DataFrame(out, columns=["model", "predictor", "n", "r", "p", "rho", "rp"])
    print(df.to_string(index=False, formatters={"r": "{:+.3f}".format, "p": "{:.1e}".format,
          "rho": "{:+.3f}".format, "rp": "{:.1e}".format}))

# Per-dataset combined-axis sanity (should match report: cyl LR strong +, robot LR moderate +)
print("\n===== Per-dataset combined-axis composite vs LR (sanity vs report) =====")
for ds in SIGNAL:
    d = runs[(runs.dataset == ds) & (runs.fault == "combined")]
    if len(d) > 2:
        r, p = stats.pearsonr(d.astrid_score, d.lr_auc)
        rho, rp = stats.spearmanr(d.astrid_score, d.lr_auc)
        print(f"  {ds:15s} n={len(d):2d}  LR Pearson r={r:+.3f} (p={p:.1e})  Spearman={rho:+.3f}")
