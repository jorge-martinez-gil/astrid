import sys,warnings,numpy as np,pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
import experiments.reliability_vs_performance as E
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from scipy import stats

def models():
    return {
        "knn":      make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        "gnb":      make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), GaussianNB()),
        "lr_unscaled": make_pipeline(SimpleImputer(strategy="mean"), LogisticRegression(max_iter=300)),
        "lr_minmax": make_pipeline(SimpleImputer(strategy="mean"), MinMaxScaler(), LogisticRegression(max_iter=300)),
    }

runs=pd.read_csv('experiments/results/reliability_vs_performance/reliability_vs_performance_runs.csv')
faults=['missingness','outliers','duplicates','combined']
for ds in [sys.argv[1]]:
    frame,feat,lab=E.load_dataset(ds,E.DATASETS[ds])
    tr,te=train_test_split(frame,test_size=0.3,random_state=0,stratify=frame[lab]); tr=tr.reset_index(drop=True); te=te.reset_index(drop=True)
    teX=te[feat]; tey=te[lab].astype(int).values
    rowsbymodel={k:[] for k in models()}
    scores=[]
    for f in faults:
        for sev in [0.0,0.1,0.2,0.3,0.4,0.5]:
            for seed in [7,11,19]:
                rng=np.random.default_rng(seed)
                c=tr.copy() if sev==0 else E.FAULTS[f](tr,feat,lab,sev,rng)
                row=runs[(runs.dataset==ds)&(runs.fault==f)&(np.isclose(runs.severity,sev))&(runs.seed==seed)]
                sc=row.astrid_score.iloc[0]; scores.append(sc)
                for name,mdl in models().items():
                    try:
                        mdl.fit(c[feat],c[lab].astype(int)); a=roc_auc_score(tey,mdl.predict_proba(teX)[:,1])
                    except Exception:
                        a=np.nan
                    rowsbymodel[name].append(a)
    scores=np.array(scores,float)
    print("==",ds)
    for name,aucs in rowsbymodel.items():
        a=np.array(aucs,float); ok=np.isfinite(a)&np.isfinite(scores)
        r,p=stats.pearsonr(scores[ok],a[ok]); s,sp=stats.spearmanr(scores[ok],a[ok])
        print(f"   {name:12s} baseAUC~{np.nanmean(a[:3]):.3f}  Pearson={r:+.3f}(p={p:.2g})  Spearman={s:+.3f}")
