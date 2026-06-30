"""Paper-2 audit-correctness figure pack. Cool 'instrument' palette, vector PDF.
Distinct from Paper 1 (which uses downstream scatter/analysis figures)."""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import pearsonr

R = Path("experiments/results/p2_audit"); OUT = Path("paper2"); OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size":9,"font.family":"serif","mathtext.fontset":"cm",
 "axes.titlesize":10,"axes.titleweight":"bold","axes.labelsize":9,"legend.fontsize":8,
 "xtick.labelsize":8,"ytick.labelsize":8,"axes.spines.top":False,"axes.spines.right":False,
 "axes.edgecolor":"#33414B","axes.linewidth":0.8,"figure.dpi":150,"savefig.dpi":320})
INK="#16222B"; STEEL="#13678A"; DEEP="#0B3C5D"; TEAL="#1B998B"; AMBER="#E08A1E"
ROSE="#B4436C"; VIOLET="#5D5FA6"; GREY="#9AA7AE"
CMAP=LinearSegmentedColormap.from_list("steel",["#FFFFFF","#CFE3EA","#6FB0C0","#13678A","#0B3C5D"])
FA=["missingness","duplicates","split_leakage","drift","pii","fairness"]
FL={"missingness":"Missingness","duplicates":"Duplicates","split_leakage":"Split leakage",
    "drift":"Drift","pii":"PII exposure","fairness":"Fairness disparity"}
FC={"missingness":STEEL,"duplicates":TEAL,"split_leakage":DEEP,"drift":AMBER,"pii":ROSE,"fairness":VIOLET}
def save(fig,n): fig.savefig(OUT/f"{n}.pdf",bbox_inches="tight"); fig.savefig(OUT/f"{n}.png",bbox_inches="tight",dpi=300); plt.close(fig); print("saved",n)

bal=pd.read_csv(R/"detection_grid_balanced.csv")
strict=pd.read_csv(R/"detection_grid_strict.csv"); len_=pd.read_csv(R/"detection_grid_lenient.csv")
ca=pd.read_csv(R/"carriers.csv"); ra=pd.read_csv(R/"reference_agreement.csv")
attr=pd.read_csv(R/"ablation_attribution.csv",index_col=0)
rs=pd.read_csv(Path("experiments/results/p1_det/runtime_scaling.csv"))

# 1) COVERAGE HEATMAP -------------------------------------------------------
sevs=sorted(bal.severity.unique())
piv=bal.groupby(["fault","severity"])["detected"].mean().unstack().reindex(FA)[sevs]
fig,ax=plt.subplots(figsize=(5.4,3.0))
im=ax.imshow(piv.values,cmap=CMAP,vmin=0,vmax=1,aspect="auto")
ax.set_xticks(range(len(sevs))); ax.set_xticklabels([f"{s:g}" for s in sevs])
ax.set_yticks(range(len(FA))); ax.set_yticklabels([FL[f] for f in FA])
ax.set_xlabel(r"injected fault severity  $\alpha$")
for i in range(len(FA)):
    for j in range(len(sevs)):
        v=piv.values[i,j]
        ax.text(j,i,f"{v:.0f}" if v in(0,1) else f"{v:.2f}",ha="center",va="center",
                fontsize=8,color="white" if v>0.55 else "#22323B",fontweight="bold")
ax.axvline(0.5,color="#22323B",lw=1.0,ls=(0,(3,2)))
ax.text(0,-0.85,"clean",fontsize=7.5,color="#22323B",ha="center")
cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.03); cb.set_label("detection rate",fontsize=8)
save(fig,"fig_coverage")

# 2) DETECTION ROC (real-data carriers) ------------------------------------
fig,ax=plt.subplots(figsize=(3.5,3.3))
for f in FA:
    s=ca[ca.fault==f]; y=(s.severity>0).astype(int).to_numpy(); v=s.value.to_numpy()
    if len(set(y))<2: continue
    fpr,tpr,_=roc_curve(y,v); a=roc_auc_score(y,v)
    ax.plot(fpr,tpr,color=FC[f],lw=1.8,label=f"{FL[f]} ({a:.2f})")
ax.plot([0,1],[0,1],color=GREY,lw=0.9,ls=":")
ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)
ax.legend(title="fault (AUROC)",loc="lower right",fontsize=7,title_fontsize=7.5,frameon=False)
save(fig,"fig_roc")

# 3) GATE CALIBRATION ------------------------------------------------------
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.0,3.0))
presets=["Strict","Balanced","Lenient"]; data={"Strict":strict,"Balanced":bal,"Lenient":len_}
fb=[data[p][data[p].severity==0].policy_fail.mean() for p in presets]
tb=[data[p][data[p].severity>0].policy_fail.mean() for p in presets]
x=np.arange(len(presets)); w=0.36
a1.bar(x-w/2,fb,w,color=ROSE,label="false-block (clean)")
a1.bar(x+w/2,tb,w,color=STEEL,label="true-block (fault present)")
a1.set_xticks(x); a1.set_xticklabels(presets); a1.set_ylabel("gate block rate"); a1.set_ylim(0,1)
a1.legend(frameon=False,fontsize=7.5); a1.set_title("(a) Gate operating point by preset")
for p,c in zip(presets,[DEEP,STEEL,TEAL]):
    g=data[p].groupby("severity").policy_fail.mean()
    a2.plot(g.index,g.values,marker="o",ms=4,color=c,lw=1.7,label=p)
a2.set_xlabel(r"injected severity $\alpha$"); a2.set_ylabel("true-block rate"); a2.set_ylim(-0.02,1.02)
a2.legend(frameon=False,fontsize=7.5); a2.set_title("(b) True-block vs severity")
save(fig,"fig_gate")

# 4) REFERENCE-TOOL AGREEMENT ----------------------------------------------
fig,axs=plt.subplots(1,3,figsize=(7.0,2.55))
spec=[("Fairlearn","fairness","positive-rate disparity",STEEL),
      ("AlibiDetect","drift","max KS distance",AMBER),
      ("Presidio","pii","PII row rate",ROSE)]
for ax,(tool,f,lab,c) in zip(axs,spec):
    s=ra[ra.tool==tool]
    ax.scatter(s.ref_value,s.astrid_value,s=26,color=c,edgecolor="white",lw=0.5,zorder=3)
    lo=min(s.ref_value.min(),s.astrid_value.min()); hi=max(s.ref_value.max(),s.astrid_value.max())
    pad=(hi-lo)*0.08+1e-6; ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],color=GREY,ls=":",lw=0.9)
    try: r=pearsonr(s.ref_value,s.astrid_value)[0]
    except Exception: r=float("nan")
    ag=(s.astrid_flag==s.ref_flag).mean()
    ax.set_title(f"{tool}\n$r$={r:.2f}, agree={ag:.0%}",fontsize=8.5)
    ax.set_xlabel(f"{tool}"); 
axs[0].set_ylabel("ASTRID")
fig.subplots_adjust(top=0.74)
fig.suptitle("ASTRID vs specialist tool on injected faults",y=1.13,fontsize=10,fontweight="bold")
save(fig,"fig_agreement")

# 5) RUNTIME SCALING -------------------------------------------------------
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.0,2.8))
rr=rs[rs.axis=="rows"]; a1.plot(rr["size"],rr.runtime_s,marker="o",ms=4,color=STEEL,lw=1.8)
a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlabel("rows (log)"); a1.set_ylabel("runtime (s, log)")
a1.set_title("(a) Runtime vs rows"); a1.grid(True,which="both",alpha=0.18)
rc=rs[rs.axis=="cols"]; a2.plot(rc["ncols"],rc.runtime_s,marker="s",ms=4,color=AMBER,lw=1.8)
a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlabel("columns (log)"); a2.set_ylabel("runtime (s, log)")
a2.set_title("(b) Runtime vs columns"); a2.grid(True,which="both",alpha=0.18)
save(fig,"fig_scaling")

# 6) DIMENSION ATTRIBUTION --------------------------------------------------
DIMS=["quality","security","reliability","robustness","fairness"]
A=attr.reindex(FA)[DIMS]
fig,ax=plt.subplots(figsize=(5.2,3.0))
im=ax.imshow(A.values,cmap=CMAP,aspect="auto",vmin=0,vmax=float(A.values.max()))
ax.set_xticks(range(len(DIMS))); ax.set_xticklabels([d.capitalize() for d in DIMS],rotation=20,ha="right")
ax.set_yticks(range(len(FA))); ax.set_yticklabels([FL[f] for f in FA])
for i in range(len(FA)):
    for j in range(len(DIMS)):
        v=A.values[i,j]
        if v>0.05: ax.text(j,i,f"{v:.0f}",ha="center",va="center",fontsize=8,
                           color="white" if v>A.values.max()*0.55 else "#22323B",fontweight="bold")
cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.03); cb.set_label("score points lost",fontsize=8)
ax.set_title(r"Dimension attribution of detection ($\alpha=0.4$)")
save(fig,"fig_ablation")
print("ALL FIGURES DONE")
