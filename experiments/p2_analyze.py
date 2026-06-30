"""Consolidate all Paper-2 audit-correctness numbers into one memo + JSON."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr, spearmanr
R = Path("experiments/results/p2_audit")
FA = ["missingness","duplicates","split_leakage","drift","pii","fairness"]
M = {}

# ---- detection grid ----
grids = {p: pd.read_csv(R/f"detection_grid_{p}.csv") for p in ["strict","balanced","lenient"]}
bal = grids["balanced"]
cov = bal.groupby(["fault","severity"])["detected"].mean().unstack().reindex(FA)
M["coverage_balanced"] = cov.round(3).to_dict()
# min detectable severity (first severity with detection rate==1)
mds = {}
for f in FA:
    row = cov.loc[f]
    hit = [s for s in row.index if row[s] >= 0.999 and s > 0]
    mds[f] = float(min(hit)) if hit else None
M["min_detectable_severity_balanced"] = mds
# synthetic per-fault detection AUROC (pos=alpha>0, neg=alpha=0), score=value
syn_auc = {}
for f in FA:
    s = bal[bal.fault==f]
    y = (s.severity>0).astype(int).to_numpy(); v = s.value.to_numpy()
    syn_auc[f] = float(roc_auc_score(y,v)) if len(set(y))>1 else None
M["synthetic_detection_auroc"] = syn_auc

# ---- gate calibration per preset ----
gate = {}
for p,g in grids.items():
    fb = g[g.severity==0]["policy_fail"].mean()      # false-block (clean)
    tb = g[g.severity>0]["policy_fail"].mean()        # true-block (fault present)
    gate[p] = {"false_block": round(float(fb),3), "true_block": round(float(tb),3)}
M["gate_calibration"] = gate
# true-block by severity (balanced)
M["true_block_by_severity_balanced"] = bal.groupby("severity")["policy_fail"].mean().round(3).to_dict()

# ---- runtime ----
M["runtime_grid_s"] = {"mean": round(float(bal.runtime_s.mean()),3),
                        "p95": round(float(bal.runtime_s.quantile(.95)),3),
                        "max": round(float(bal.runtime_s.max()),3)}
rs = pd.read_csv(R.parent/"p1_det"/"runtime_scaling.csv")
M["runtime_scaling_rows"] = rs[rs.axis=="rows"][["size","runtime_s"]].round(3).values.tolist()
M["runtime_scaling_cols"] = rs[rs.axis=="cols"][["ncols","runtime_s"]].round(3).values.tolist()

# ---- reference agreement ----
ra = pd.read_csv(R/"reference_agreement.csv")
ref = {}
for tool in ra.tool.unique():
    s = ra[ra.tool==tool]
    agree = float((s.astrid_flag==s.ref_flag).mean())
    try: pr = float(pearsonr(s.astrid_value, s.ref_value)[0])
    except Exception: pr = None
    # cohen kappa
    a,b = s.astrid_flag.to_numpy(), s.ref_flag.to_numpy()
    po = (a==b).mean(); pe = ((a.mean()*b.mean())+((1-a.mean())*(1-b.mean())))
    kappa = float((po-pe)/(1-pe)) if pe<1 else 1.0
    ref[tool] = {"decision_agreement": round(agree,3), "value_r": None if pr is None else round(pr,3),
                 "cohen_kappa": round(kappa,3), "n": int(len(s))}
M["reference_agreement"] = ref
# alibi per-severity to explain the 0.6
al = ra[ra.tool=="AlibiDetect"].groupby("severity")[["astrid_flag","ref_flag"]].mean()
M["alibi_flags_by_severity"] = al.round(2).to_dict()

# ---- carriers ----
ca = pd.read_csv(R/"carriers.csv")
M["carrier_false_alarm_rate"] = round(float(ca[ca.severity==0].detected.mean()),3)
M["carrier_n_datasets"] = int(ca.dataset.nunique())
car_auc = {}; car_ap = {}
for f in FA:
    s = ca[ca.fault==f]
    y = (s.severity>0).astype(int).to_numpy(); v = s.value.to_numpy()
    if len(set(y))>1:
        car_auc[f]=float(roc_auc_score(y,v)); car_ap[f]=float(average_precision_score(y,v))
M["carrier_detection_auroc"]={k:round(v,3) for k,v in car_auc.items()}
M["carrier_detection_ap"]={k:round(v,3) for k,v in car_ap.items()}
M["carrier_detection_rate_by_fault_sevpos"]=ca[ca.severity>0].groupby("fault").detected.mean().round(3).to_dict()
M["carrier_macro_auroc"]=round(float(np.mean(list(car_auc.values()))),3)

# ---- ablation ----
attr = pd.read_csv(R/"ablation_attribution.csv", index_col=0)
M["attribution_argmax"]={f:str(attr.loc[f].idxmax()) for f in attr.index}
loo = pd.read_csv(R/"ablation_leaveoneout.csv")
M["ablation_summary"]={"note":"fraction of composite drop retained when aligned dim removed (low=attributable)"}

(R/"paper2_numbers.json").write_text(json.dumps(M, indent=2, default=str))
print(json.dumps(M, indent=2, default=str))
