import warnings; warnings.filterwarnings('ignore')
import sys, types, importlib.util, numpy as np, pandas as pd
# ---- import REAL time-series analyzer (streamlit-stubbed) ----
st=types.ModuleType("streamlit")
st.__getattr__=lambda n: (lambda *a,**k: None)
st.session_state={}
sys.modules["streamlit"]=st
for m in ["plotly","plotly.express","plotly.graph_objects","altair","matplotlib","matplotlib.pyplot"]:
    sys.modules.setdefault(m, types.ModuleType(m))
spec=importlib.util.spec_from_file_location("ts_page","pages/02_Time_Series.py")
tsmod=importlib.util.module_from_spec(spec); sys.modules["ts_page"]=tsmod
try: spec.loader.exec_module(tsmod)
except Exception: pass
from utils import compute_health_score, DEFAULT_WEIGHTS
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

OUTLIER_MAGNITUDE=15.0
def inj_missing(tr,f,sev,rng):
    o=tr.copy(); a=o[f].to_numpy(float); a[rng.random(a.shape)<sev]=np.nan; o[f]=a; return o
def inj_outliers(tr,f,sev,rng):
    o=tr.copy(); a=o[f].to_numpy(float); sd=np.nanstd(a,0); sd[sd==0]=1.0
    n=int(round(sev*len(o)))
    if n>0:
        rows=rng.choice(len(o),n,replace=False); sgn=rng.choice([-1.,1.],size=(n,a.shape[1]))
        a[rows]=a[rows]+sgn*OUTLIER_MAGNITUDE*sd
    o[f]=a; return o
def inj_dup(tr,f,sev,rng):
    n=int(round(sev*len(tr)))
    if n==0: return tr.copy()
    idx=rng.choice(len(tr),n,replace=True); return pd.concat([tr,tr.iloc[idx]],ignore_index=True)
def inj_combined(tr,f,sev,rng):
    o=inj_missing(tr,f,sev,rng); o=inj_outliers(o,f,sev*0.5,rng); o=inj_dup(o,f,sev,rng); return o

# ---- HRSS ----
df=pd.read_csv("Datasets/HRSS/astrid_dataset.csv")
df=df[df["label"].notna()].copy(); df["label"]=df["label"].astype(int)
# stratified subsample for tractable scoring
df,_=train_test_split(df, train_size=6000, random_state=0, stratify=df["label"]); df=df.reset_index(drop=True)
TIME="Timestamp"; LABEL="label"
drop={TIME,LABEL,"Labels"}
feats=[c for c in df.select_dtypes(include="number").columns if c not in drop]
print("HRSS rows",len(df),"features",len(feats),"pos_rate",round(df[LABEL].mean(),3))

# build TS analyzer config
Th=tsmod.Thresholds; AC=tsmod.AssessConfig
import inspect
th=Th(drift_ks_threshold=0.30, pii_hit_rate_threshold=0.05, label_noise_rate_warn=0.05,
      time_parse_ok_min=0.9, dup_timestamp_rate_max=0.5, cadence_irregularity_max=1.0)
def ts_health(frame):
    cfg=AC(label_col=LABEL, split_col=None, time_col=TIME, entity_cols=[], group_cols=[],
           annotator_label_cols=[], id_cols=[], time_slice_mode="first_vs_last", random_state=0,
           max_categories_for_stats=50, thresholds=th, mode="Quick Scan",
           pii_max_rows=2000, pii_max_text_cols=10, rare_max_cat_cols=10, drift_max_num_cols=30)
    rep=tsmod.assess_all(frame, cfg, frame.to_csv(index=False).encode())
    score,grade,comp=compute_health_score(rep, th.drift_ks_threshold, weights=DEFAULT_WEIGHTS)
    return score, comp

# clean holdout
tr0,te=train_test_split(df, test_size=0.30, random_state=0, stratify=df[LABEL])
teX=te[feats]; tey=te[LABEL].astype(int).values

SEV=[0.0,0.1,0.2,0.3,0.4,0.5]; SEEDS=[7,11,19]
rows=[]
for seed in SEEDS:
    for sev in SEV:
        rng=np.random.default_rng(seed)
        cor=inj_combined(tr0, feats, sev, rng)
        # keep time col present for the analyzer
        H,comp=ts_health(cor[[TIME]+feats+[LABEL]] if TIME in cor else cor)
        # robust GBM
        gb=HistGradientBoostingClassifier(random_state=seed).fit(cor[feats], cor[LABEL].astype(int))
        auc_gb=roc_auc_score(tey, gb.predict_proba(teX)[:,1])
        # sensitive LR
        lr=make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
        lr.fit(cor[feats], cor[LABEL].astype(int))
        auc_lr=roc_auc_score(tey, lr.predict_proba(teX)[:,1])
        rows.append(dict(modality="time_series",dataset="HRSS",severity=sev,seed=seed,H=H,auc_gbm=auc_gb,auc_lr=auc_lr))
        print(f"sev={sev} seed={seed} H={H} AUC_gbm={auc_gb:.3f} AUC_lr={auc_lr:.3f}")
res=pd.DataFrame(rows); res.to_csv("/tmp/ts_results.csv",index=False)
def pear(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    return np.corrcoef(a,b)[0,1] if a.std()>0 and b.std()>0 else float('nan')
print("\nHRSS within-dataset Pearson r:  H~AUC_lr =", round(pear(res.H,res.auc_lr),3), " H~AUC_gbm =", round(pear(res.H,res.auc_gbm),3))
print("clean LR AUC:", round(res[res.severity==0].auc_lr.mean(),3), " clean GBM AUC:", round(res[res.severity==0].auc_gbm.mean(),3))
