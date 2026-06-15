"""Clean contrast: robust (GBM) vs non-robust (standardized LR) on the combined
corruption axis, for the two datasets with genuine destroyable signal."""
import sys,warnings,numpy as np,pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0,".")
import experiments.reliability_vs_performance as E
from astrid_core import analyze_tabular_dataframe
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
RES=E.ROOT/"experiments"/"results"/"reliability_vs_performance"
out_csv=RES/"fragile_vs_robust_clean.csv"
SEVS=[0.0,0.1,0.2,0.3,0.4,0.5,0.6]; SEEDS=[7,11,19,23,31,37]
def brittle(): return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(max_iter=1000))
def score(fr): return analyze_tabular_dataframe(fr,dataset_name="d",preset="Balanced (recommended)",mode="Quick Scan")["score"]
recs=[]
if out_csv.exists() and out_csv.stat().st_size>0:
    recs=pd.read_csv(out_csv).to_dict("records")
done={(r["dataset"],r["severity"],r["seed"]) for r in recs}
import time; t0=time.time()
for ds in ["cylinder_bands","robot"]:
    frame,feat,lab=E.load_dataset(ds,E.DATASETS[ds])
    tr,te=train_test_split(frame,test_size=0.3,random_state=0,stratify=frame[lab]); tr=tr.reset_index(drop=True); te=te.reset_index(drop=True)
    teX=te[feat]; tey=te[lab].astype(int).values
    for sev in SEVS:
        for seed in SEEDS:
            if (ds,sev,seed) in done: continue
            if time.time()-t0>40: break
            rng=np.random.default_rng(seed)
            c=tr.copy() if sev==0 else E.inject_combined(tr,feat,lab,sev,rng)
            s=score(c[feat+[lab]])
            br=brittle().fit(c[feat],c[lab].astype(int)); ba=roc_auc_score(tey,br.predict_proba(teX)[:,1])
            gb=HistGradientBoostingClassifier(random_state=seed).fit(c[feat],c[lab].astype(int)); ga=roc_auc_score(tey,gb.predict_proba(teX)[:,1])
            recs.append(dict(dataset=ds,severity=sev,seed=seed,astrid_score=s,brittle_auc=round(float(ba),5),gbm_auc=round(float(ga),5)))
            pd.DataFrame(recs).to_csv(out_csv,index=False)
n=len(recs); full=2*len(SEVS)*len(SEEDS)
print(f"{n}/{full} done")
