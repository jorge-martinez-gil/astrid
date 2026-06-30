import warnings; warnings.filterwarnings("ignore")
import sys, os, time, csv; sys.path.insert(0,"/tmp")
import numpy as np, pandas as pd
from img_lib import image_health
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
S=20; OUT="/tmp/img_results2.csv"
def gen(n,rng):
    X=np.empty((n,S,S)); y=rng.integers(0,2,n)
    for i in range(n):
        img=rng.normal(120,55,(S,S)); amp=15.0
        if y[i]==1: img[5:10,4:16]+=amp
        else: img[4:16,5:10]+=amp
        X[i]=np.clip(img,0,255)
    return X.astype("uint8"),y
rng0=np.random.default_rng(0); X,y=gen(500,rng0)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
Xtef=Xte.reshape(len(Xte),-1)
def corrupt(rng,sev):
    arrs=[a.copy() for a in Xtr]; labs=list(ytr)
    n=int(sev*len(arrs))
    for j in rng.choice(len(arrs),n,replace=False): arrs[j]=rng.uniform(0,255,(S,S)).astype("uint8")
    nd=int(sev*len(Xtr))
    for j in rng.choice(len(Xtr),nd,replace=True): arrs.append(Xtr[j].copy()); labs.append(ytr[j])
    return arrs,labs
done=set()
if os.path.exists(OUT):
    for _,r in pd.read_csv(OUT).iterrows(): done.add((round(r.severity,3),int(r.seed)))
newf=not os.path.exists(OUT); f=open(OUT,"a",newline=""); w=csv.writer(f)
if newf: w.writerow(["modality","dataset","severity","seed","H","auc_lr","auc_gbm"]); f.flush()
t0=time.time(); stop=False
for seed in [7,11,19]:
    for sev in [0.0,0.1,0.2,0.3,0.4,0.5]:
        if (round(sev,3),seed) in done: continue
        if time.time()-t0>32: stop=True; break
        rng=np.random.default_rng(seed); arrs,labs=corrupt(rng,sev)
        H,sc,rep=image_health(arrs,labs)
        Xc=np.array([a.reshape(-1) for a in arrs]); yc=np.array(labs)
        lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=400)).fit(Xc,yc)
        gb=HistGradientBoostingClassifier(random_state=seed,max_iter=100).fit(Xc,yc)
        alr=roc_auc_score(yte,lr.predict_proba(Xtef)[:,1]); agb=roc_auc_score(yte,gb.predict_proba(Xtef)[:,1])
        w.writerow(["image","synthetic_lowSNR",sev,seed,H,round(alr,4),round(agb,4)]); f.flush()
        print(f"sev={sev} seed={seed} H={H} auc_lr={alr:.3f} auc_gbm={agb:.3f}")
    if stop: break
f.close(); print("done so far:", len(done)+ (0))
