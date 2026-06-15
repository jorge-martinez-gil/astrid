"""Analyse ASTRID reliability-score vs. downstream-performance experiment.

Reads runs_with_lr.csv (ASTRID composite score + GBM-AUC + LR-AUC for every
corruption run) and produces correlation tables and figures used in the report.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results" / "reliability_vs_performance"
FIG = RES / "figures"; FIG.mkdir(parents=True, exist_ok=True)
DETECTABLE = ["missingness", "outliers", "duplicates", "combined"]
PRETTY = {"cylinder_bands": "Cylinder Bands", "secom": "SECOM", "aps": "APS"}
COL = {"cylinder_bands": "#1f77b4", "secom": "#d62728", "aps": "#2ca02c"}


def corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return dict(n=len(x), pr=np.nan, pp=np.nan, sr=np.nan, sp=np.nan)
    pr, pp = stats.pearsonr(x, y); sr, sp = stats.spearmanr(x, y)
    return dict(n=len(x), pr=pr, pp=pp, sr=sr, sp=sp)


def zpool(d, ycol):
    d = d.copy()
    d["sz"] = d.groupby("dataset").astrid_score.transform(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1))
    d["az"] = d.groupby("dataset")[ycol].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1))
    return corr(d.sz, d.az)


def main():
    m = pd.read_csv(RES / "runs_with_lr.csv")
    det = m[m.fault.isin(DETECTABLE)]
    rows = []
    for model, yc in [("GBM", "test_auc"), ("LR", "lr_auc")]:
        for ds, g in det.groupby("dataset"):
            c = corr(g.astrid_score, g[yc]); c.update(scope=f"{ds} / detectable", model=model); rows.append(c)
        c = zpool(det, yc); c.update(scope="POOLED z / detectable", model=model); rows.append(c)
        c = zpool(m, yc); c.update(scope="POOLED z / all faults", model=model); rows.append(c)
        for ds, g in m[m.fault == "combined"].groupby("dataset"):
            c = corr(g.astrid_score, g[yc]); c.update(scope=f"{ds} / combined", model=model); rows.append(c)
    tab = pd.DataFrame(rows)[["scope", "model", "n", "pr", "pp", "sr", "sp"]]
    tab.columns = ["scope", "model", "n", "pearson_r", "pearson_p", "spearman_rho", "spearman_p"]
    tab.to_csv(RES / "correlation_summary.csv", index=False)
    (RES / "correlation_summary.json").write_text(json.dumps(tab.round(4).to_dict("records"), indent=2))

    # Fig 1: score vs GBM-AUC, detectable faults (shows weak/flat relationship)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for ds, g in det.groupby("dataset"):
        ax.scatter(g.astrid_score, g.test_auc, s=24, alpha=.65, color=COL[ds], label=PRETTY[ds])
        if g.astrid_score.std() > 0:
            b, a = np.polyfit(g.astrid_score, g.test_auc, 1)
            xs = np.linspace(g.astrid_score.min(), g.astrid_score.max(), 40)
            ax.plot(xs, a + b * xs, color=COL[ds], lw=2)
    ax.set_xlabel("ASTRID composite reliability score (0-100)")
    ax.set_ylabel("Downstream GBM test ROC-AUC")
    ax.set_title("Composite score vs. gradient-boosting AUC (detectable faults)")
    ax.legend(frameon=False); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(FIG / "fig1_score_vs_gbm.png", dpi=140); plt.close(fig)

    # Fig 2: severity trajectories for combined fault (both axes)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, ds in zip(axes, ["cylinder_bands", "secom", "aps"]):
        g = m[(m.dataset == ds) & (m.fault == "combined")].groupby("severity").mean(numeric_only=True).reset_index()
        ax2 = ax.twinx()
        ax.plot(g.severity, g.astrid_score, "o-", color="#1f77b4")
        ax2.plot(g.severity, g.test_auc, "s--", color="#d62728")
        ax2.plot(g.severity, g.lr_auc, "^:", color="#7f7f7f")
        ax.set_title(PRETTY[ds]); ax.set_xlabel("Combined corruption severity")
        ax.set_ylabel("ASTRID score", color="#1f77b4"); ax2.set_ylabel("AUC", color="#d62728")
        ax.grid(alpha=.2)
    fig.suptitle("Combined corruption: score falls, but GBM-AUC (red) barely moves; LR-AUC (grey) is model-dependent")
    fig.tight_layout(); fig.savefig(FIG / "fig2_severity_combined.png", dpi=140); plt.close(fig)

    # Fig 3: label-noise contrast (the key explanatory figure)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, ds in zip(axes, ["cylinder_bands", "secom", "aps"]):
        g = m[(m.dataset == ds) & (m.fault == "label_noise")].groupby("severity").mean(numeric_only=True).reset_index()
        ax2 = ax.twinx()
        ax.plot(g.severity, g.astrid_score, "o-", color="#1f77b4")
        ax2.plot(g.severity, g.test_auc, "s--", color="#d62728")
        ax.set_ylim(0, 100); ax2.set_ylim(0.4, 1.0)
        ax.set_title(PRETTY[ds]); ax.set_xlabel("Label-noise rate")
        ax.set_ylabel("ASTRID score", color="#1f77b4"); ax2.set_ylabel("GBM AUC", color="#d62728")
        ax.grid(alpha=.2)
    fig.suptitle("Label noise collapses downstream AUC (red) while the reliability score (blue) stays flat")
    fig.tight_layout(); fig.savefig(FIG / "fig3_label_noise_contrast.png", dpi=140); plt.close(fig)

    # Fig 4: per-fault Pearson (score vs GBM-AUC) heat-ish bar
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    faults = ["missingness", "outliers", "duplicates", "combined", "label_noise"]
    width = 0.25
    for i, ds in enumerate(["cylinder_bands", "secom", "aps"]):
        vals = []
        for f in faults:
            g = m[(m.dataset == ds) & (m.fault == f)]
            vals.append(corr(g.astrid_score, g.test_auc)["pr"])
        ax.bar(np.arange(len(faults)) + i * width, vals, width, color=COL[ds], label=PRETTY[ds])
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(np.arange(len(faults)) + width); ax.set_xticklabels(faults, rotation=15)
    ax.set_ylabel("Pearson r (score vs GBM-AUC)")
    ax.set_title("Per-fault correlation is small and inconsistent in sign")
    ax.legend(frameon=False); ax.grid(alpha=.2, axis="y"); fig.tight_layout()
    fig.savefig(FIG / "fig4_per_fault_corr.png", dpi=140); plt.close(fig)

    print(tab.to_string(index=False))
    print("\nfigures ->", FIG)


if __name__ == "__main__":
    main()
