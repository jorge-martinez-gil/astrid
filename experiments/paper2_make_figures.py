"""Generate the Paper-2 high-quality figure pack (E6 law, E8 gate, E9 ablation,
upgraded teaser + severity trajectories). Saves PDF (for LaTeX) + PNG (preview)."""
import warnings; warnings.filterwarnings("ignore")
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper2" / "figs_new"; OUT.mkdir(parents=True, exist_ok=True)
RES = ROOT / "experiments" / "results"

# ---- house style (Paper 2 = warm "analysis" palette) ----
plt.rcParams.update({
    "font.size": 9, "font.family": "serif", "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 140,
})
WARM = ["#B2182B", "#D6604D", "#F4A582", "#E08214", "#998EC3", "#542788", "#4D4D4D"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("saved", name)


def zpool(g, col):
    return g.groupby("dataset")[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))


stats_out = {}

# ============ E6: sensitivity law with bootstrap CIs ============
ms = pd.read_csv(RES / "reliability_vs_performance" / "model_spectrum.csv")
seeds = sorted(ms.seed.unique())


def law_point(df, model):
    g = df[df.model == model].copy()
    g["sc_z"] = zpool(g, "astrid_score"); g["au_z"] = zpool(g, "auc")
    r = np.corrcoef(g["sc_z"], g["au_z"])[0, 1]
    base = g[g.severity == 0].groupby("dataset")["auc"].mean()
    hi = g[g.severity >= 0.4].groupby("dataset")["auc"].mean()
    sens = float((base - hi).mean())
    return sens, r

models = sorted(ms.model.unique())
pts = {m: law_point(ms, m) for m in models}
# bootstrap over seeds
B = 600; boot = {m: [] for m in models}; rho_boot = []
rng = np.random.default_rng(0)
for _ in range(B):
    samp = rng.choice(seeds, len(seeds), replace=True)
    d = pd.concat([ms[ms.seed == s] for s in samp], ignore_index=True)
    xs, ys = [], []
    for m in models:
        s, r = law_point(d, m); boot[m].append((s, r)); xs.append(s); ys.append(r)
    rho_boot.append(stats.spearmanr(xs, ys)[0])
rho = stats.spearmanr([pts[m][0] for m in models], [pts[m][1] for m in models])[0]
rho_lo, rho_hi = np.percentile(rho_boot, [2.5, 97.5])
stats_out["sensitivity_law"] = {"rho": rho, "rho_ci": [rho_lo, rho_hi], "n_learners": len(models),
                                 "points": {m: {"sensitivity": pts[m][0], "r": pts[m][1]} for m in models}}

fig, ax = plt.subplots(figsize=(5.4, 4.0))
for i, m in enumerate(sorted(models, key=lambda k: pts[k][0])):
    sb = np.array(boot[m]); s, r = pts[m]
    sl, sh = np.percentile(sb[:, 0], [2.5, 97.5]); rl, rh = np.percentile(sb[:, 1], [2.5, 97.5])
    c = WARM[i % len(WARM)]
    ax.errorbar(s, r, xerr=[[s - sl], [sh - s]], yerr=[[r - rl], [rh - r]],
                fmt="o", ms=8, color=c, ecolor=c, elinewidth=1.2, capsize=3, zorder=3)
    ax.annotate(m, (s, r), xytext=(6, -2), textcoords="offset points", fontsize=8, color=c)
# fit line
xs = np.array([pts[m][0] for m in models]); ys = np.array([pts[m][1] for m in models])
b1, b0 = np.polyfit(xs, ys, 1); xx = np.linspace(xs.min(), xs.max(), 50)
ax.plot(xx, b0 + b1 * xx, "--", color="#333333", lw=1.2, zorder=1)
ax.set_xlabel("Model defect-sensitivity  (mean held-out AUC drop under corruption)")
ax.set_ylabel("Score–AUC correlation  (within-dataset $z$-pooled)")
ax.set_title("The sensitivity law: the score predicts a model\nexactly to the degree the model is sensitive to the defects")
ax.text(0.04, 0.93, f"Spearman $\\rho$ = {rho:.2f}\n95% CI [{rho_lo:.2f}, {rho_hi:.2f}]",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="#FDEBE3", ec="#B2182B", alpha=0.9))
save(fig, "fig_p2_sensitivity_law")

# ============ E8: gate-as-harm-classifier ============
ga = pd.read_csv(RES / "paper2_downstream" / "gate_ablation.csv")
clean = ga[ga.severity == 0].groupby("dataset")["auc_lr"].mean()
ga["harm"] = ga.apply(lambda r: (clean[r.dataset] - r.auc_lr) >= 0.05, axis=1)
ga["fail"] = ga.gate_status.str.upper().str.contains("FAIL")
y = ga.harm.values.astype(int); sc = -ga.astrid_score.values
fpr, tpr, _ = roc_curve(y, sc); auroc = roc_auc_score(y, sc)
prec, rec, _ = precision_recall_curve(y, sc)
# gate operating point
tp = ((ga.fail) & (ga.harm)).sum(); fp = ((ga.fail) & (~ga.harm)).sum()
fn = ((~ga.fail) & (ga.harm)).sum()
g_rec = tp / (tp + fn + 1e-9); g_fpr = fp / (fp + ((~ga.fail) & (~ga.harm)).sum() + 1e-9)
g_prec = tp / (tp + fp + 1e-9)
stats_out["gate_harm"] = {"auroc_score": float(auroc), "harm_rate": float(ga.harm.mean()),
                          "gate_precision": float(g_prec), "gate_recall": float(g_rec),
                          "gate_fpr": float(g_fpr), "n": int(len(ga))}
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
ax = axes[0]
ax.plot(fpr, tpr, color="#B2182B", lw=2, label=f"score threshold (AUROC={auroc:.2f})")
ax.plot([0, 1], [0, 1], ":", color="gray", lw=1)
ax.scatter([g_fpr], [g_rec], color="#542788", s=70, zorder=5, label="policy gate (FAIL)")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
ax.set_title("Gate / score as a detector of\nharmful datasets (AUC drop $\\geq$ 0.05)")
ax.legend(loc="lower right")
# cost curve: expected cost vs threshold for a range of cost ratios
ax2 = axes[1]
order = np.argsort(-sc); ys = y[order]
thr = np.arange(1, len(ys) + 1)
flagged_harm = np.cumsum(ys); flagged_tot = thr
miss = ys.sum() - flagged_harm  # harmful not yet flagged
false_block = flagged_tot - flagged_harm
for ratio, c in [(5, "#B2182B"), (2, "#E08214"), (1, "#998EC3")]:
    cost = ratio * miss + false_block
    ax2.plot(thr / len(ys), cost / len(ys), color=c, lw=1.8, label=f"cost(miss)/cost(block)={ratio}")
ax2.set_xlabel("Fraction of datasets blocked (score-ranked)")
ax2.set_ylabel("Expected cost per dataset")
ax2.set_title("Operating cost of a score-ranked\ndataset gate")
ax2.legend()
save(fig, "fig_p2_gate_harm")

# ============ E9: dimension-ablation heatmap ============
learners = {"auc_gbm": "Robust GBM", "auc_lr": "Sensitive LR"}
dims = ["astrid_score", "pen_quality", "pen_robustness", "pen_reliability", "pen_security", "pen_fairness"]
dim_lbl = ["Composite H", "Quality", "Robustness", "Reliability", "Security", "Fairness"]
comb = ga[ga.fault == "combined"].copy()
M = np.zeros((len(dims), len(learners)))
for j, (lc, ln) in enumerate(learners.items()):
    g = comb.copy(); g["au_z"] = zpool(g, lc)
    for i, d in enumerate(dims):
        # higher score = better; penalties are "badness" so flip sign so + = predictive
        v = g[d]
        sign = 1  # score_components are positive contributions (higher = better)
        gz = (v - v.mean()) / (v.std() + 1e-9)
        M[i, j] = np.corrcoef(gz, g["au_z"])[0, 1] if g[d].std() > 0 else 0
fig, ax = plt.subplots(figsize=(4.6, 4.4))
im = ax.imshow(M, cmap="RdYlBu", vmin=-0.7, vmax=0.7, aspect="auto")
ax.set_xticks(range(len(learners))); ax.set_xticklabels(learners.values())
ax.set_yticks(range(len(dims))); ax.set_yticklabels(dim_lbl)
for i in range(len(dims)):
    for j in range(len(learners)):
        ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=8,
                color="black")
ax.set_title("Which signal predicts downstream AUC?\n(combined-fault axis, $z$-pooled correlation)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="correlation with AUC")
save(fig, "fig_p2_dimension_heatmap")
stats_out["dimension_ablation"] = {dim_lbl[i]: {ln: float(M[i, j]) for j, ln in enumerate(learners.values())}
                                   for i in range(len(dims))}

# ============ F1 upgrade: teaser robust vs sensitive ============
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7), sharey=False)
for ax, (lc, ln, col) in zip(axes, [("auc_gbm", "Robust GBM", "#4D4D4D"), ("auc_lr", "Sensitive LR", "#B2182B")]):
    for ds, mk in [("cylinder_bands", "o"), ("mechanical", "s")]:
        d = ga[(ga.dataset == ds) & (ga.fault == "combined")]
        sizes = 18 + d.severity * 120
        ax.scatter(d.astrid_score, d[lc], s=sizes, alpha=0.6, marker=mk,
                   color=col, edgecolor="white", linewidth=0.4, label=ds.replace("_", " "))
    ax.set_xlabel("ASTRID health score $H(D)$"); ax.set_ylabel("held-out ROC-AUC")
    ax.set_title(ln); ax.legend(fontsize=7)
fig.suptitle("Whether the score predicts the model depends on the model\n(marker size $\\propto$ corruption severity)", y=1.05)
save(fig, "fig_p2_teaser")

# ============ F4: severity trajectories ============
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
for ax, ds in zip(axes, ["cylinder_bands", "mechanical"]):
    d = ga[(ga.dataset == ds) & (ga.fault == "combined")].groupby("severity").mean(numeric_only=True)
    ax.plot(d.index, d.astrid_score / 100, "-o", color="#542788", label="ASTRID score /100")
    ax.plot(d.index, d.auc_gbm, "-s", color="#4D4D4D", label="robust GBM AUC")
    ax.plot(d.index, d.auc_lr, "-^", color="#B2182B", label="sensitive LR AUC")
    ax.set_xlabel("corruption severity $\\alpha$"); ax.set_title(ds.replace("_", " "))
    ax.set_ylim(0.3, 1.02); ax.legend(fontsize=7)
fig.suptitle("Combined-corruption trajectories: score falls; only the sensitive model's AUC follows", y=1.04)
save(fig, "fig_p2_trajectories")

json.dump(stats_out, open(OUT / "paper2_stats.json", "w"), indent=2)
print("STATS:", json.dumps(stats_out, indent=2)[:1500])
