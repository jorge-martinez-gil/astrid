import sys,warnings,numpy as np,pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
import experiments.reliability_vs_performance as E
from astrid_core import analyze_tabular_dataframe
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats
def fragile(): return make_pipeline(SimpleImputer(strategy="mean"),StandardScaler(),LogisticRegression(max_iter=500))
def score(fr): return analyze_tabular_dataframe(fr,dataset_name="d",preset="Balanced (recommended)",mode="Quick Scan")["score"]
df=pd.read_csv("Datasets/HRSS/astrid_dataset.csv")
df=df.sample(8000,random_state=0).reset_index(drop=True)
lab="label"; drop={"Timestamp","label","source_file","condition","configuration","Labels"}
df=df[df[lab].notna()].copy(); df[lab]=df[lab].astype(int)
feat=[c for c in df.select_dtypes('number').columns if c not in drop]
fr=df[feat+[lab]].reset_index(drop=True); fr[feat]=fr[feat].astype(float)
tr,te=train_test_split(fr,test_size=0.3,random_state=0,stratify=fr[lab]); tr=tr.reset_index(drop=True); te=te.reset_index(drop=True)
teX=te[feat]; tey=te[lab].astype(int).values
sc=[];au=[];traj=[]
for sev in [0.0,0.1,0.2,0.3,0.4,0.5,0.6]:
    ss=[];aa=[]
    for seed in [7,11,19]:
        rng=np.random.default_rng(seed)
        c=tr.copy() if sev==0 else E.inject_combined(tr,feat,lab,sev,rng)
        s=score(c[feat+[lab]]); m=fragile().fit(c[feat],c[lab].astype(int)); a=roc_auc_score(tey,m.predict_proba(teX)[:,1])
        ss.append(s);aa.append(a);sc.append(s);au.append(a)
    traj.append(f"{sev}:{np.mean(ss):.0f}/{np.mean(aa):.3f}")
r,p=stats.pearsonr(sc,au); rho,_=stats.spearmanr(sc,au)
print(f"hrss rows={len(fr)} feat={len(feat)} pos={fr[lab].mean():.2f} Pearson={r:+.3f}(p={p:.2g}) Spearman={rho:+.3f}")
print("traj(sev:score/auc)"," ".join(traj))
