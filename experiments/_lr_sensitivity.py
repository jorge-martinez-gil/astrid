import sys,warnings,numpy as np,pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
import experiments.reliability_vs_performance as E
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
RES=E.ROOT/'experiments'/'results'/'reliability_vs_performance'
target_ds=sys.argv[1]
def lr_auc(tr,teX,tey,feat,lab):
    p=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                    LogisticRegression(max_iter=400,C=0.5))
    p.fit(tr[feat],tr[lab].astype(int)); return roc_auc_score(tey,p.predict_proba(teX)[:,1])
frame,feat,lab=E.load_dataset(target_ds,E.DATASETS[target_ds])
tr,te=train_test_split(frame,test_size=0.3,random_state=0,stratify=frame[lab]); tr=tr.reset_index(drop=True); te=te.reset_index(drop=True)
teX=te[feat]; tey=te[lab].astype(int).values
rows=[]
for f in ['missingness','outliers','duplicates','combined','label_noise']:
    for sev in [0.0,0.1,0.2,0.3,0.4,0.5]:
        for seed in [7,11,19]:
            rng=np.random.default_rng(seed)
            c=tr.copy() if sev==0 else E.FAULTS[f](tr,feat,lab,sev,rng)
            rows.append(dict(dataset=target_ds,fault=f,severity=sev,seed=seed,lr_auc=round(float(lr_auc(c,teX,tey,feat,lab)),5)))
out=RES/('lr_'+target_ds+'.csv'); pd.DataFrame(rows).to_csv(out,index=False)
print("saved",len(rows),"->",out.name)
