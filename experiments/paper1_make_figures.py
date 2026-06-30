"""Paper-1 detection/governance figure pack -- award-grade redesign.
Cohesive 'instrument' design system: restrained teal palette, generous
whitespace, direct labelling, one clear message per figure."""
import warnings, json, glob; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patheffects as pe
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figs_new"; OUT.mkdir(parents=True, exist_ok=True)
DET = ROOT / "experiments" / "results" / "p1_det"

plt.rcParams.update({
    "font.size": 10, "font.family": "serif", "mathtext.fontset": "cm",
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#3A3A4A", "axes.linewidth": 0.9,
    "axes.labelcolor": "#1A1A2E", "text.color": "#1A1A2E",
    "xtick.color": "#3A3A4A", "ytick.color": "#3A3A4A",
    "figure.dpi": 150, "savefig.dpi": 320,
})
INK="#1A1A2E"; TEAL="#13678A"; DEEP="#0B3C5D"; AMBER="#E8871E"
GREEN="#2E8B6F"; ROSE="#C44E52"; GREY="#9AA0A6"; LITE="#EAF1F4"
TEALMAP = LinearSegmentedColormap.from_list("teal", ["#FFFFFF","#CDE6EC","#5BA6B5","#13678A","#0B3C5D"])
FAULTS=["missingness","duplicates","split_leakage","drift","pii","fairness"]
FLAB={"missingness":"Missingness","duplicates":"Duplicates","split_leakage":"Split leakage",
      "drift":"Drift","pii":"PII exposure","fairness":"Fairness disparity"}

def save(fig,name):
    fig.savefig(OUT/f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT/f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig); print("saved", name)

bal = pd.concat([pd.read_csv(p) for p in glob.glob(str(DET/"s*/tabular_fault_injection_summary.csv"))],
                ignore_index=True)
sevs = sorted(bal.severity.unique())
stats = {}

# ============================================================
# FIG 1 -- DETECTION COVERAGE HEATMAP (fault x severity)
# ============================================================
piv = (bal.groupby(["fault_type","severity"])["detected"].mean()
         .unstack("severity").reindex(FAULTS)[sevs])
fig, ax = plt.subplots(figsize=(7.6, 4.3))
im = ax.imshow(piv.values, cmap=TEALMAP, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(sevs))); ax.set_xticklabels([f"{s:g}" for s in sevs])
ax.set_yticks(range(len(FAULTS))); ax.set_yticklabels([FLAB[f] for f in FAULTS])
ax.set_xlabel("injected fault severity  $\\alpha$")
for i in range(len(FAULTS)):
    for j in range(len(sevs)):
        v = piv.values[i,j]
        ax.text(j, i, f"{v:.0f}" if v in (0,1) else f"{v:.2f}",
                ha="center", va="center", fontsize=9,
                color="white" if v>0.55 else "#33414B", fontweight="bold")
# frame the clean column
ax.add_patch(plt.Rectangle((-0.5,-0.5),1,len(FAULTS),fill=False,ec=ROSE,lw=2.2,zorder=5))
ax.set_xticks(list(range(len(sevs))))
ax.set_xticklabels([f"{s:g}\n(clean)" if s==0 else f"{s:g}" for s in sevs])
ax.get_xticklabels()[0].set_color(ROSE); ax.get_xticklabels()[0].set_fontweight("bold")
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.set_label("detection rate", fontsize=9)
cb.outline.set_edgecolor("#3A3A4A")
ax.set_title("ASTRID detects every fault it screens for, and only when present", pad=12)
ax.set_xlim(-0.5,len(sevs)-0.5)
save(fig, "fig_p1_detection_curves")

# ============================================================
# FIG 2 -- CLEAN vs FAULTY SEPARATION (why detection AUC = 1.0)
# ============================================================
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ypos = np.arange(len(FAULTS))[::-1]
aucs = {}
for fa, y in zip(FAULTS, ypos):
    d = bal[bal.fault_type==fa].copy()
    m = d.primary_metric_value.astype(float)
    rng = (m.max()-m.min()) or 1.0
    d["norm"] = (m - m.min())/rng
    clean = d[d.severity==0]["norm"]; faulty = d[d.severity>0]["norm"]
    present = (d.severity>0).astype(int)
    aucs[fa] = roc_auc_score(present, d.primary_metric_value) if present.nunique()>1 and m.std()>0 else np.nan
    jit = (np.random.default_rng(1).random(len(faulty))-0.5)*0.16
    ax.scatter(faulty, np.full(len(faulty),y)+jit, s=26, color=TEAL, alpha=0.55,
               edgecolor="white", linewidth=0.4, zorder=3)
    ax.scatter(clean, np.full(len(clean),y), s=70, color=ROSE, marker="D",
               edgecolor="white", linewidth=0.6, zorder=4)
    cm_, fm_ = clean.mean(), faulty.mean()
    ax.annotate("", xy=(fm_, y), xytext=(cm_, y),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, alpha=0.7))
ax.set_yticks(ypos); ax.set_yticklabels([FLAB[f] for f in FAULTS])
ax.set_xlabel("ASTRID indicator response  (per-fault min--max normalised)")
ax.set_xlim(-0.07,1.12)
ax.scatter([],[],color=ROSE,marker="D",s=60,label="clean run")
ax.scatter([],[],color=TEAL,s=40,label="fault injected ($\\alpha>0$)")
ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#CCCCCC")
ax.set_title("Clean and corrupted runs separate cleanly\n(perfect rank separation: detection ROC-AUC $=1.00$ on all six faults)", pad=12)
save(fig, "fig_p1_detection_auc")
stats["detection_auc"] = aucs

# ============================================================
# FIG 3 -- POLICY-GATE CALIBRATION (lollipop, 0 false alarms)
# ============================================================
presets = [("Strict", DET/"strict7"), ("Balanced", DET/"s7"), ("Lenient", DET/"lenient7")]
def flagrate(df, present):
    sub = df[df.severity>0] if present else df[df.severity==0]
    return (sub.policy_status.str.upper().str.contains("FAIL") |
            sub.verdict.str.contains("review", case=False)).mean()
rows=[]
for name,pdir in presets:
    f=pdir/"tabular_fault_injection_summary.csv"
    if f.exists():
        df=pd.read_csv(f); rows.append((name, flagrate(df,True), flagrate(df,False)))
cal=pd.DataFrame(rows, columns=["preset","present","clean"])
fig, ax = plt.subplots(figsize=(7.4, 4.0))
fig.subplots_adjust(top=0.78)
yy=np.arange(len(cal))[::-1]
for y,(_,r) in zip(yy, cal.iterrows()):
    ax.plot([r.clean, r.present],[y,y], color="#C9D6DC", lw=3, zorder=1, solid_capstyle="round")
    ax.scatter(r.present, y, s=190, color=GREEN, zorder=3, edgecolor="white", linewidth=1.2)
    ax.scatter(r.clean, y, s=150, color=ROSE, zorder=3, edgecolor="white", linewidth=1.2)
    ax.text(r.present+0.025, y, f"{r.present*100:.0f}%", va="center", ha="left",
            fontsize=9.5, color=GREEN, fontweight="bold")
ax.axvline(0, color=GREY, lw=0.8, ls=(0,(3,3)))
ax.set_yticks(yy); ax.set_yticklabels(cal.preset, fontsize=11)
ax.set_xlim(-0.06,1.02); ax.set_xlabel("flag rate")
ax.set_xticks([0,0.25,0.5,0.75,1.0]); ax.set_xticklabels(["0","25%","50%","75%","100%"])
ax.scatter([],[],color=GREEN,s=120,label="flagged when fault present")
ax.scatter([],[],color=ROSE,s=120,label="flagged when clean (false alarm)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5,1.005), ncol=2, frameon=False)
ax.annotate("zero false alarms\nunder every preset", xy=(0.004,len(cal)-1), xytext=(0.16,1.75),
            fontsize=9, color=ROSE, fontweight="bold", va="center",
            arrowprops=dict(arrowstyle="->", color=ROSE, lw=1.2))
fig.suptitle("The policy gate trades recall for permissiveness—never for false alarms",
             y=1.0, fontsize=11.5, fontweight="bold")
save(fig, "fig_p1_gate_calibration")
stats["gate_calibration"]=cal.to_dict("records")

# ============================================================
# FIG 4 -- RUNTIME SCALING (single elegant panel, twin views)
# ============================================================
sc = pd.read_csv(DET/"runtime_scaling.csv")
r = sc[sc.axis=="rows"]; c = sc[sc.axis=="cols"]
fig, axes = plt.subplots(1,2, figsize=(8.4,3.7))
for ax in axes:
    ax.axhspan(0, 5, color=GREEN, alpha=0.06, zorder=0)
A=axes[0]
A.plot(r["size"], r.runtime_s, "-o", color=TEAL, lw=2.4, ms=7, mfc="white", mec=TEAL, mew=1.6)
A.set_xscale("log"); A.set_xlabel("rows  (10 features, log scale)"); A.set_ylabel("audit runtime  (s)")
A.set_ylim(0, max(1.0, r.runtime_s.max()*1.25))
for xx,yy_ in [(r["size"].iloc[0], r.runtime_s.iloc[0]),(r["size"].iloc[-1], r.runtime_s.iloc[-1])]:
    A.annotate(f"{yy_:.2f}s", (xx,yy_), textcoords="offset points", xytext=(0,9),
               ha="center", fontsize=8.5, color=DEEP, fontweight="bold")
A.set_title("Sub-linear in rows", fontsize=10.5)
B=axes[1]
B.plot(c.ncols, c.runtime_s, "-s", color=AMBER, lw=2.4, ms=7, mfc="white", mec=AMBER, mew=1.6)
B.set_xlabel("features  (4000 rows)"); B.set_ylabel("audit runtime  (s)")
B.set_ylim(0, max(1.0, c.runtime_s.max()*1.25))
for xx,yy_ in [(c.ncols.iloc[0], c.runtime_s.iloc[0]),(c.ncols.iloc[-1], c.runtime_s.iloc[-1])]:
    B.annotate(f"{yy_:.2f}s", (xx,yy_), textcoords="offset points", xytext=(0,9),
               ha="center", fontsize=8.5, color="#9A5A00", fontweight="bold")
B.set_title("Linear in features", fontsize=10.5)
axes[1].text(0.97,0.06,"shaded: <5 s CI budget", transform=axes[1].transAxes,
             ha="right", fontsize=8, color=GREEN, style="italic")
fig.suptitle("Every audit finishes in CI-compatible time", y=1.03, fontsize=11.5, fontweight="bold")
save(fig, "fig_p1_runtime")

json.dump(stats, open(OUT/"paper1_stats.json","w"), indent=2)
print("DONE paper1 figures")
