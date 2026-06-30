import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, itertools
rng=np.random.default_rng(12345)
def pear(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    return np.corrcoef(a,b)[0,1] if (a.std()>0 and b.std()>0 and len(a)>2) else np.nan
def rankdata(x):
    x=np.asarray(x,float); _,inv,c=np.unique(x,return_inverse=True,return_counts=True)
    o=np.argsort(x,kind="mergesort"); r=np.empty(len(x)); r[o]=np.arange(1,len(x)+1)
    s=np.zeros(len(c)); np.add.at(s,inv,r); return (s/c)[inv]
def spear(a,b): return pear(rankdata(a),rankdata(b))
def fisher_mean(rs):
    rs=np.clip(np.array([r for r in rs if np.isfinite(r)]),-0.999,0.999)
    return np.tanh(np.mean(np.arctanh(rs)))
def boot_ci_within(x,y,B=10000):
    x=np.asarray(x,float);y=np.asarray(y,float);n=len(x);out=[]
    for _ in range(B):
        idx=rng.integers(0,n,n); out.append(pear(x[idx],y[idx]))
    out=np.array([o for o in out if np.isfinite(o)]); return np.nanpercentile(out,[2.5,97.5])
def perm_p_within(x,y,B=20000):
    x=np.asarray(x,float);y=np.asarray(y,float);obs=pear(x,y);c=0
    for _ in range(B):
        if abs(pear(x,rng.permutation(y)))>=abs(obs)-1e-12: c+=1
    return (c+1)/(B+1)

# ============ TABULAR: per-dataset combined-corruption r, dataset-as-unit ============
print("="*70); print("TABULAR  (combined corruption; dataset = unit of analysis)"); print("="*70)
rl=pd.read_csv("results/reliability_vs_performance/runs_with_lr.csv")
comb=rl[rl.fault=="combined"].copy()
ms=pd.read_csv("results/reliability_vs_performance/model_spectrum.csv")  # mechanical+cyl, combined sweep
# mechanical from model_spectrum: pivot GBM(HistGBM) & LR(LogReg)
def from_spectrum(dsname):
    g=ms[(ms.dataset==dsname)]
    gb=g[g.model=="HistGBM"][["severity","seed","astrid_score","auc"]].rename(columns={"auc":"auc_gbm","astrid_score":"H"})
    lr=g[g.model=="LogReg"][["severity","seed","auc"]].rename(columns={"auc":"auc_lr"})
    m=gb.merge(lr,on=["severity","seed"]); m["dataset"]=dsname; return m
tab_rows={}
for ds in ["cylinder_bands","secom","aps","robot"]:
    d=comb[comb.dataset==ds]
    tab_rows[ds]=pd.DataFrame({"dataset":ds,"H":d.astrid_score.values,"auc_gbm":d.test_auc.values,"auc_lr":d.lr_auc.values})
tab_rows["mechanical"]=from_spectrum("mechanical")[["dataset","H","auc_gbm","auc_lr"]]
# also rebuild cylinder from spectrum to cross-check (use runs_with_lr version as primary)
print(f"{'dataset':16s} {'n':>3} {'cleanLR':>7} {'r_LR':>7} {'CI_LR':>16} {'permp_LR':>9} | {'r_GBM':>7}")
per_r_lr={}; per_r_gbm={}
for ds,t in tab_rows.items():
    rlr=pear(t.H,t.auc_lr); rgb=pear(t.H,t.auc_gbm)
    ci=boot_ci_within(t.H.values,t.auc_lr.values); pp=perm_p_within(t.H.values,t.auc_lr.values)
    cleanlr=t[t.H==t.H.max()].auc_lr.mean()
    per_r_lr[ds]=rlr; per_r_gbm[ds]=rgb
    print(f"{ds:16s} {len(t):>3} {cleanlr:7.3f} {rlr:+7.3f} [{ci[0]:+.2f},{ci[1]:+.2f}]  {pp:9.4g} | {rgb:+7.3f}")
# degradable-signal subset (paper's two): cylinder_bands, mechanical
deg=["cylinder_bands","mechanical"]
rs_lr=[per_r_lr[d] for d in deg]; rs_gb=[per_r_gbm[d] for d in deg]
print(f"\nDataset-as-unit (degradable-signal datasets {deg}):")
print(f"  LR : per-dataset r = {[round(x,3) for x in rs_lr]}  Fisher-mean = {fisher_mean(rs_lr):+.3f}")
print(f"  GBM: per-dataset r = {[round(x,3) for x in rs_gb]}  Fisher-mean = {fisher_mean(rs_gb):+.3f}")
# cluster bootstrap over ALL datasets (resample datasets, then runs within)
allds=list(tab_rows.keys())
def cluster_boot(metric, B=10000):
    out=[]
    for _ in range(B):
        chosen=rng.choice(allds,size=len(allds),replace=True)
        rr=[]
        for ds in chosen:
            t=tab_rows[ds]; rr.append(pear(t.H, t[metric]))
        out.append(fisher_mean(rr))
    out=np.array([o for o in out if np.isfinite(o)]); return np.nanpercentile(out,[2.5,97.5])
print(f"  Cluster-bootstrap over ALL {len(allds)} datasets (Fisher-mean r):")
print(f"    LR  95% CI = {np.round(cluster_boot('auc_lr'),3)}  (point Fisher-mean over all = {fisher_mean([per_r_lr[d] for d in allds]):+.3f})")
print(f"    GBM 95% CI = {np.round(cluster_boot('auc_gbm'),3)}  (point = {fisher_mean([per_r_gbm[d] for d in allds]):+.3f})")

# ============ TIME-SERIES (HRSS) ============
print("\n"+"="*70); print("TIME-SERIES  (HRSS, combined corruption, real TS analyzer)"); print("="*70)
ts=pd.read_csv("/tmp/ts_results.csv")
rlr=pear(ts.H,ts.auc_lr); rgb=pear(ts.H,ts.auc_gbm)
print(f"n={len(ts)} runs | clean LR AUC={ts[ts.severity==0].auc_lr.mean():.3f} clean GBM AUC={ts[ts.severity==0].auc_gbm.mean():.3f}")
print(f"  LR : r={rlr:+.3f}  95%CI={np.round(boot_ci_within(ts.H.values,ts.auc_lr.values),3)}  permp={perm_p_within(ts.H.values,ts.auc_lr.values):.4g}")
print(f"  GBM: r={rgb:+.3f}  95%CI={np.round(boot_ci_within(ts.H.values,ts.auc_gbm.values),3)}  permp={perm_p_within(ts.H.values,ts.auc_gbm.values):.4g}")

# ============ SENSITIVITY LAW (learner = unit) ============
print("\n"+"="*70); print("SENSITIVITY LAW  (learner = unit; 7 learners)"); print("="*70)
def zpool(sub):
    zs=[];za=[]
    for ds,g in sub.groupby("dataset"):
        s=g.astrid_score.values.astype(float);a=g.auc.values.astype(float)
        zs+=list((s-s.mean())/s.std()) if s.std()>0 else list(s*0)
        za+=list((a-a.mean())/a.std()) if a.std()>0 else list(a*0)
    return np.array(zs),np.array(za)
r_by={}; sens={}
for m in sorted(ms.model.unique()):
    sub=ms[ms.model==m]; zs,za=zpool(sub); r_by[m]=pear(zs,za)
    drops=[sub[(sub.dataset==ds)&(sub.severity==0)].auc.mean()-sub[(sub.dataset==ds)&(sub.severity==sub.severity.max())].auc.mean() for ds in sub.dataset.unique()]
    sens[m]=np.mean(drops)
mods=sorted(ms.model.unique()); sv=np.array([sens[m] for m in mods]); rv=np.array([r_by[m] for m in mods])
obs=spear(sv,rv)
# exact permutation (7!)
perms=np.array([spear(sv,rv[list(p)]) for p in itertools.permutations(range(len(mods)))])
p_exact=np.mean(np.abs(perms)>=abs(obs)-1e-12)
# bootstrap CI on rho (resample learners)
bs=[]
for _ in range(20000):
    idx=rng.integers(0,len(mods),len(mods))
    if len(set(idx))>2: bs.append(spear(sv[idx],rv[idx]))
bs=np.array([b for b in bs if np.isfinite(b)]); ci=np.nanpercentile(bs,[2.5,97.5])
for m in mods: print(f"  {m:13s} sens={sens[m]:+.3f}  r={r_by[m]:+.3f}")
print(f"\n  Spearman rho={obs:+.4f}  EXACT-perm p={p_exact:.4g}  bootstrap95%CI=[{ci[0]:.2f},{ci[1]:.2f}]")
