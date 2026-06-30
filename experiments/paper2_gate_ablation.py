"""Paper 2 — lean collection for the gate-as-harm-classifier (E8) and the
dimension-ablation-for-prediction (E9) experiments.

For each (dataset, fault, severity, seed) we record:
  * ASTRID composite score, grade, the five dimension penalties, and the
    policy-gate PASS/FAIL verdict (on the corrupted training matrix);
  * held-out ROC-AUC for a ROBUST learner (HistGB) and a SENSITIVE learner
    (median-impute -> standardize -> logistic regression).

Only two learners are trained here (fast); the multi-learner sensitivity law
(E6) is computed separately from the existing model_spectrum.csv.

Resumable + time-budgeted:
    python experiments/paper2_gate_ablation.py --time-budget 40   # repeat
"""
from __future__ import annotations
import argparse, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astrid_core import analyze_tabular_dataframe  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

DATASETS = {
    "cylinder_bands": dict(path="Datasets/cilinder bands/astrid_dataset.csv",
                           label="label", drop_cols=["band_type", "timestamp"],
                           max_rows=None, max_features=None),
    "mechanical": dict(path="Datasets/mechanical analysis/astrid_dataset.csv",
                       label="label", drop_cols=[], max_rows=2500, max_features=120),
    "secom": dict(path="Datasets/secom/astrid_dataset.csv",
                  label="label", drop_cols=["class_label", "timestamp"],
                  max_rows=None, max_features=80),
    "aps": dict(path="Datasets/aps/astrid_dataset.csv",
                label="label", drop_cols=["class"], max_rows=2500, max_features=80),
}
FAULT_LIST = ["missingness", "outliers", "duplicates", "combined", "label_noise"]
SEVERITIES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
SEEDS = [7, 11, 19, 23, 31]
DIMENSIONS = ["quality", "security", "reliability", "robustness", "fairness"]
OUTLIER_MAG = 15.0


def load(cfg):
    df = pd.read_csv(ROOT / cfg["path"])
    label = cfg["label"]
    df = df[df[label].notna()].copy()
    df[label] = df[label].astype(int)
    if df[label].nunique() > 2:  # binarize multiclass: majority vs rest
        majority = df[label].value_counts().idxmax()
        df[label] = (df[label] == majority).astype(int)
    if cfg.get("max_rows") and len(df) > cfg["max_rows"]:
        df, _ = train_test_split(df, train_size=cfg["max_rows"], random_state=0,
                                 stratify=df[label])
        df = df.reset_index(drop=True)
    drop = set(cfg.get("drop_cols", [])) | {label}
    feats = [c for c in df.select_dtypes(include="number").columns if c not in drop]
    k = cfg.get("max_features")
    if k and len(feats) > k:
        feats = list(df[feats].var().sort_values(ascending=False).index[:k])
    frame = df[feats + [label]].reset_index(drop=True)
    frame[feats] = frame[feats].astype(float)
    return frame, feats, label


def inj_missing(tr, f, lb, s, rng):
    out = tr.copy(); a = out[f].to_numpy(float); a[rng.random(a.shape) < s] = np.nan
    out[f] = a; return out
def inj_outliers(tr, f, lb, s, rng):
    out = tr.copy(); a = out[f].to_numpy(float); sd = np.nanstd(a, 0); sd[sd == 0] = 1
    n = int(round(s * len(out)))
    if n:
        r = rng.choice(len(out), n, False)
        a[r] += rng.choice([-1., 1.], (n, a.shape[1])) * OUTLIER_MAG * sd
    out[f] = a; return out
def inj_dup(tr, f, lb, s, rng):
    n = int(round(s * len(tr)))
    return tr.copy() if n == 0 else pd.concat([tr, tr.iloc[rng.choice(len(tr), n, True)]], ignore_index=True)
def inj_label(tr, f, lb, s, rng):
    out = tr.copy(); flip = rng.random(len(out)) < s; out.loc[flip, lb] = 1 - out.loc[flip, lb]; return out
def inj_combined(tr, f, lb, s, rng):
    return inj_dup(inj_outliers(inj_missing(tr, f, lb, s, rng), f, lb, s * 0.5, rng), f, lb, s, rng)
FAULTS = {"missingness": inj_missing, "outliers": inj_outliers, "duplicates": inj_dup,
          "label_noise": inj_label, "combined": inj_combined}


def astrid_full(frame):
    r = analyze_tabular_dataframe(frame, dataset_name="run",
                                  preset="Balanced (recommended)", mode="Quick Scan")
    comps = r.get("score_components", {}) or {}
    pol = r.get("policy_result") or {}
    status = pol.get("status") if isinstance(pol, dict) else None
    status = status or r.get("verdict_status") or r.get("verdict") or ""
    return r["score"], r["grade"], comps, str(status)


def auc_gbm(Xtr, ytr, Xte, yte, seed):
    m = HistGradientBoostingClassifier(random_state=seed).fit(Xtr, ytr)
    return float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
def auc_lr(Xtr, ytr, Xte, yte, seed):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
    return float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-budget", type=float, default=0.0)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = ap.parse_args()
    out_dir = ROOT / "experiments" / "results" / "paper2_downstream"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "gate_ablation.csv"

    done = set(); header = True
    if out_csv.exists() and out_csv.stat().st_size > 0:
        prev = pd.read_csv(out_csv); header = False
        done = {(r.dataset, r.fault, round(float(r.severity), 4), int(r.seed)) for r in prev.itertuples()}

    cache = {}
    def get(name):
        if name not in cache:
            fr, ft, lb = load(DATASETS[name])
            tr, te = train_test_split(fr, test_size=0.30, random_state=0, stratify=fr[lb])
            cache[name] = (ft, lb, tr.reset_index(drop=True), te.reset_index(drop=True))
        return cache[name]

    plan = [(d, f, round(s, 4), sd) for d in args.datasets for f in FAULT_LIST
            for s in SEVERITIES for sd in SEEDS]
    todo = [k for k in plan if k not in done]
    print(f"plan={len(plan)} done={len(done)} todo={len(todo)}", flush=True)

    t0 = time.time(); n = 0
    for (dn, fault, sev, seed) in todo:
        if args.time_budget and time.time() - t0 > args.time_budget:
            break
        ft, lb, tr, te = get(dn)
        Xte, yte = te[ft], te[lb].astype(int).values
        rng = np.random.default_rng(seed)
        corr = tr.copy() if sev == 0.0 else FAULTS[fault](tr, ft, lb, sev, rng)
        score, grade, comps, status = astrid_full(corr[ft + [lb]])
        Xtr, ytr = corr[ft], corr[lb].astype(int).values
        row = dict(dataset=dn, fault=fault, severity=sev, seed=seed,
                   astrid_score=score, grade=grade, gate_status=status,
                   auc_gbm=auc_gbm(Xtr, ytr, Xte, yte, seed),
                   auc_lr=auc_lr(Xtr, ytr, Xte, yte, seed))
        for d in DIMENSIONS:
            row[f"pen_{d}"] = comps.get(d, np.nan)
        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=header, index=False)
        header = False; n += 1
    rem = len(todo) - n
    rem = len(todo) - n
    print(f"wrote {n} in {time.time()-t0:.1f}s | remaining {rem}", flush=True)
    if rem == 0:
        print("GATE/ABLATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
