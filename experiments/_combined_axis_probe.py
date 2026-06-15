import sys,warnings,numpy as np,pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
import experiments.reliability_vs_performance as E
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats
runs=pd.read_csv('experiments/results/reliability_vs_performance/reliability_vs_performance_runs.csv')
def mk():
    return {
      "lr_standard": make_pipeline(SimpleImputer(strategy="mean"),StandardScaler(),LogisticRegression(max_iter=400,C=1.0)),
      "lr_minmax":   make_pipeline(SimpleImputer(strategy="mean"),MinMaxScaler(),LogisticRegression(max_iter=400,C=1.0)),
      "lr_noscale":  make_pipeline(SimpleImputer(strategy="mean"),LogisticRegression(max_iter=400,C=1.0)),
    }
ds=sys.argv[1]
frame,feat,lab=E.load_dataset(ds,E.DATASETS[ds])
tr,te=train_test_split(frame,test_size=0.3,random_state=0,stratify=frame[lab]); tr=tr.reset_index(drop=True); te=te.reset_index(drop=True)
teX=te[feat]; tey=te[lab].astype(int).values
print("==",ds)
for name in mk():
    sev_means={}; sc_all=[]; au_all=[]
    for sev in [0.0,0.1,0.2,0.3,0.4,0.5]:
        aus=[]; scs=[]
        for seed in [7,11,19]:
            rng=np.random.default_rng(seed)
            c=tr.copy() if sev==0 else E.inject_combined(tr,feat,lab,sev,rng)
            mdl=mk()[name]; mdl.fit(c[feat],c[lab].astype(int)); a=roc_auc_score(tey,mdl.predict_proba(teX)[:,1])
            row=runs[(runs.dataset==ds)&(runs.fault=='combined')&(np.isclose(runs.severity,sev))&(runs.seed==seed)]
            s=float(row.astrid_score.iloc[0])
            aus.append(a); scs.append(s); au_all.append(a); sc_all.append(s)
        sev_means[sev]=(np.mean(scs),np.mean(aus))
    r,p=stats.pearsonr(sc_all,au_all); s,sp=stats.spearmanr(sc_all,au_all)
    traj=" ".join(f"{sv}:{m[0]:.0f}/{m[1]:.3f}" for sv,m in sev_means.items())
    print(f"  {name:12s} Pearson={r:+.3f}(p={p:.2g}) Spearman={s:+.3f}  traj(sev:score/auc) {traj}")
