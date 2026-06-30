"""Fixes for Paper-2 figures: (1) sensitivity-law label collisions,
(2) a clean replacement for the per-fault correlation figure."""
import warnings, json; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"paper2"/"figs_new"; OUT.mkdir(parents=True, exist_ok=True)
RES = ROOT/"experiments"/"results"
plt.rcParams.update({
    "font.size":10,"font.family":"serif","mathtext.fontset":"cm",
    "axes.titlesize":11.5,"axes.titleweight":"bold","axes.labelsize":10,
    "legend.fontsize":9,"xtick.labelsize":9,"ytick.labelsize":9,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.edgecolor":"#3A3A4A","axes.linewidth":0.9,
    "axes.labelcolor":"#241A1A","text.color":"#241A1A",
    "figure.dpi":150,"savefig.dpi":320})
def save(fig,name):
    fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight")
    fig.savefig(OUT/f"{name}.png",bbox_inches="tight",dpi=300)
    plt.close(fig); print("saved",name)

WARM=["#B2182B","#D6604D","#E08214","#D9A441","#7C9A6B","#5E8C8B","#4D4D4D"]

# ================= FIG 6 : sensitivity law (no label overlap) =================
ms = pd.read_csv(RES/"reliability_vs_performance"/"model_spectrum.csv")
seeds=sorted(ms.seed.unique())
def zp(g,c): return g.groupby("dataset")[c].transform(lambda x:(x-x.mean())/(x.std()+1e-9))
def law_point(df,m):
    g=df[df.model==m].copy(); g["sz"]=zp(g,"astrid_score"); g["az"]=zp(g,"auc")
    r=np.corrcoef(g["sz"],g["az"])[0,1]
    base=g[g.severity==0].groupby("dataset")["auc"].mean()
    hi=g[g.severity>=0.4].groupby("dataset")["auc"].mean()
    return float((base-hi).mean()), r
models=sorted(ms.model.unique())
pts={m:law_point(ms,m) for m in models}
B=600; boot={m:[] for m in models}; rho_b=[]; rng=np.random.default_rng(0)
for _ in range(B):
    s=rng.choice(seeds,len(seeds),True); d=pd.concat([ms[ms.seed==x] for x in s],ignore_index=True)
    xs=[];ys=[]
    for m in models:
        a,b=law_point(d,m); boot[m].append((a,b)); xs.append(a); ys.append(b)
    rho_b.append(stats.spearmanr(xs,ys)[0])
rho=stats.spearmanr([pts[m][0] for m in models],[pts[m][1] for m in models])[0]
lo,hi=np.percentile(rho_b,[2.5,97.5])

# hand-tuned label placement to avoid collisions (all to the right, y-nudged)
LBL={"HistGBM":(10,-13,"left"),"DecisionTree":(10,10,"left"),"RandomForest":(12,0,"left"),
     "kNN":(12,1,"left"),"LogReg":(12,1,"left"),"LinearSVM":(10,-13,"left"),"GaussianNB":(10,9,"left")}
order=sorted(models,key=lambda k:pts[k][0])
cmap={m:WARM[i%len(WARM)] for i,m in enumerate(order)}
fig,ax=plt.subplots(figsize=(6.4,4.6))
xs=np.array([pts[m][0] for m in models]); ys=np.array([pts[m][1] for m in models])
b1,b0=np.polyfit(xs,ys,1); xx=np.linspace(xs.min()-0.005,xs.max()+0.01,50)
ax.plot(xx,b0+b1*xx,"--",color="#555555",lw=1.3,zorder=1)
for m in order:
    sb=np.array(boot[m]); s,r=pts[m]; c=cmap[m]
    sl,sh=np.percentile(sb[:,0],[2.5,97.5]); rl,rh=np.percentile(sb[:,1],[2.5,97.5])
    ax.errorbar(s,r,xerr=[[s-sl],[sh-s]],yerr=[[r-rl],[rh-r]],fmt="o",ms=9,color=c,
                ecolor=c,elinewidth=1.3,capsize=3,zorder=3,mec="white",mew=0.8)
    dx,dy,ha=LBL[m]
    ax.annotate(m,(s,r),xytext=(dx,dy),textcoords="offset points",fontsize=8.6,
                color=c,fontweight="bold",ha=ha,va="center",
                arrowprops=dict(arrowstyle="-",color=c,lw=0.6,alpha=0.5,
                                shrinkA=0,shrinkB=4))
ax.set_xlim(xs.min()-0.02, xs.max()+0.085)
ax.set_xlabel("model defect-sensitivity\n(mean held-out AUC drop under corruption)")
ax.set_ylabel("score–AUC correlation\n(within-dataset $z$-pooled)")
ax.set_title("The sensitivity law", pad=8)
ax.text(0.035,0.955,f"Spearman $\\rho = {rho:.2f}$\n95% CI [{lo:.2f}, {hi:.2f}]",
        transform=ax.transAxes,va="top",fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4",fc="#FBEDE8",ec="#B2182B",alpha=0.95))
ax.text(0.5,1.005,"the score predicts a model exactly to the degree the model is sensitive to the measured defects",
        transform=ax.transAxes,ha="center",va="bottom",fontsize=8.6,style="italic",color="#555555")
save(fig,"fig_p2_sensitivity_law")

# ================= FIG 3 : per-fault correlation (clean forest plot) =================
ga=pd.read_csv(RES/"paper2_downstream"/"gate_ablation.csv")
faults=["missingness","outliers","duplicates","combined","label_noise"]
flab={"missingness":"Missingness","outliers":"Outliers","duplicates":"Duplicates",
      "combined":"Combined","label_noise":"Label noise"}
dsets=[d for d in ["cylinder_bands","mechanical","secom","aps"] if d in ga.dataset.unique()]
dcol={"cylinder_bands":"#B2182B","mechanical":"#E08214","secom":"#5E8C8B","aps":"#7C5Ca0"}
dlab={"cylinder_bands":"Cylinder Bands","mechanical":"Mechanical","secom":"SECOM","aps":"APS"}
def corr(sub):
    if len(sub)<4 or sub.astrid_score.std()==0 or sub.auc_gbm.std()==0: return np.nan
    return np.corrcoef(sub.astrid_score, sub.auc_gbm)[0,1]
fig,ax=plt.subplots(figsize=(6.8,4.3))
yb=np.arange(len(faults))[::-1]
ax.axvspan(-0.2,0.2,color="#BBBBBB",alpha=0.13,zorder=0)
ax.axvline(0,color="#555555",lw=1.0,zorder=1)
for fa,y in zip(faults,yb):
    for k,ds in enumerate(dsets):
        r=corr(ga[(ga.fault==fa)&(ga.dataset==ds)])
        if not np.isnan(r):
            off=(k-(len(dsets)-1)/2)*0.16
            ax.scatter(r,y+off,s=70,color=dcol.get(ds,"#888"),edgecolor="white",
                       linewidth=0.7,zorder=3)
ax.set_yticks(yb); ax.set_yticklabels([flab[f] for f in faults])
ax.set_xlim(-1.05,1.05); ax.set_xlabel("score–AUC correlation for the robust learner (per dataset)")
ax.set_title("Naïve per-fault correlation is weak and inconsistent", pad=20)
ax.text(0.5,1.005,"no fault yields a consistent, correctly-signed signal for a robust model — the grey band marks $|r|<0.2$",
        transform=ax.transAxes,ha="center",va="bottom",fontsize=8.4,style="italic",color="#555555")
for ds in dsets:
    ax.scatter([],[],color=dcol.get(ds,"#888"),s=60,label=dlab.get(ds,ds))
ax.legend(loc="lower left",frameon=True,framealpha=0.95,edgecolor="#CCCCCC",ncol=2,fontsize=8)
save(fig,"fig_p2_perfault")
print("DONE p2 fixes; rho=%.3f CI[%.2f,%.2f]"%(rho,lo,hi))
