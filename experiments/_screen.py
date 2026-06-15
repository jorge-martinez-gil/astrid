import sys,warnings,numpy as np,pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
CAND={
 "cylinder_bands":("Datasets/cilinder bands/astrid_dataset.csv","label",["band_type","timestamp"]),
 "genesis":("Datasets/Genesis/astrid_dataset.csv","label",["Timestamp"]),
 "mechanical":("Datasets/mechanical analysis/astrid_dataset.csv","label",[]),
 "robot":("Datasets/robot/astrid_dataset.csv","failure_binary",["class_label"]),
 "hrss":("Datasets/HRSS/astrid_dataset.csv","label",["Timestamp"]),
}
def fragile():
    return make_pipeline(SimpleImputer(strategy="mean"),StandardScaler(),LogisticRegression(max_iter=400))
for name,(path,lab,drop) in CAND.items():
    try:
        df=pd.read_csv(path)
        if len(df)>12000: df=df.sample(12000,random_state=0)
        if lab not in df.columns: print(name,"NO LABEL",list(df.columns)[:6]); continue
        df=df[df[lab].notna()].copy(); 
        try: df[lab]=df[lab].astype(int)
        except: pass
        vc=df[lab].value_counts().to_dict()
        feats=[c for c in df.select_dtypes('number').columns if c not in set(drop)|{lab}]
        if df[lab].nunique()!=2: print(f"{name}: not binary, label vals {list(vc)[:5]}"); continue
        y=df[lab].astype(int).values; X=df[feats].astype(float)
        if min(np.bincount(y))<10: print(f"{name}: tiny class {vc}"); continue
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
        m=fragile().fit(Xtr,ytr); a=roc_auc_score(yte,m.predict_proba(Xte)[:,1])
        print(f"{name:14s} rows={len(df):6d} nfeat={len(feats):4d} pos={y.mean():.3f} fragileAUC={a:.3f}")
    except Exception as e:
        print(name,"ERR",repr(e)[:120])
