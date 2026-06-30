"""Paper 2 (E1+): real-data detection CARRIERS.
Use real industrial datasets as clean carriers; inject each fault with
ground-truth severity and test whether ASTRID detects it, using the SAME
detection thresholds as the synthetic study. Gives (a) false-alarm rate on real
clean data and (b) per-fault detection ROC/PR pooled over datasets. No model trained.
Resumable per dataset -> results/p2_audit/carriers.csv
"""
from __future__ import annotations
import argparse, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from astrid_core import (TabularAssessConfig, TABULAR_PRESETS,
                         analyze_tabular_dataframe, dataframe_to_bytes)
from experiments.fault_injection_tabular import extract_primary_metric

DSETS = {
    "Cylinder Bands": "Datasets/cilinder bands/astrid_dataset.csv",
    "Mechanical": "Datasets/mechanical analysis/astrid_dataset.csv",
    "SECOM": "Datasets/secom/astrid_dataset.csv",
    "APS": "Datasets/aps/astrid_dataset.csv",
    "Genesis": "Datasets/Genesis/astrid_dataset.csv",
    "Bolts": "Datasets/bolts/astrid_dataset.csv",
}
FAULTS = ["missingness", "duplicates", "split_leakage", "drift", "pii", "fairness"]
SEV = [0.05, 0.20, 0.40]
SEEDS = [7, 11]
MAXR, MAXF = 1500, 40

def load_carrier(path):
    df = pd.read_csv(ROOT/path)
    # label
    lab = "label" if "label" in df.columns else ("target" if "target" in df.columns else df.columns[-1])
    df = df[df[lab].notna()].copy()
    try:
        df[lab] = df[lab].astype(int)
    except Exception:
        df[lab] = (df[lab].astype("category").cat.codes > 0).astype(int)
    reserved = {"target","sample_id","split","site","timestamp","operator_note"}
    num = [c for c in df.select_dtypes("number").columns if c != lab and c not in reserved]
    if len(num) > MAXF:
        num = list(df[num].var().sort_values(ascending=False).index[:MAXF])
    if len(df) > MAXR:
        df = df.sample(MAXR, random_state=0)
    df = df.reset_index(drop=True)
    out = df[num].astype(float).copy()
    out["target"] = df[lab].astype(int).to_numpy()
    n = len(out)
    rng = np.random.default_rng(0)
    out["sample_id"] = [f"r{i:06d}" for i in range(n)]
    out["split"] = rng.choice(["train", "test"], size=n, p=[0.7, 0.3])
    out["site"] = rng.choice(["a", "b", "c"], size=n)
    out["timestamp"] = pd.date_range("2025-01-01", periods=n, freq="h").astype(str)
    out["operator_note"] = "normal"
    return out, num

def inject(df, num, fault, sev, seed):
    rng = np.random.default_rng(seed)
    out = df.copy()
    if sev <= 0 or fault is None:
        return out
    n = len(out)
    if fault == "missingness":
        arr = out[num].to_numpy(float)
        arr[rng.random(arr.shape) < sev] = np.nan
        out[num] = arr
    elif fault == "duplicates":
        k = max(1, int(n*sev)); out = pd.concat([out, out.sample(k, random_state=seed)], ignore_index=True)
    elif fault == "split_leakage":
        k = max(1, int(n*sev)); lk = out.sample(k, random_state=seed).copy()
        lk["split"] = np.where(lk["split"].eq("test"), "train", "test")
        out = pd.concat([out, lk], ignore_index=True)
    elif fault == "drift":
        ts = pd.to_datetime(out["timestamp"]); mask = ts >= ts.quantile(0.75)
        for c in num:
            s = out[c].std() or 1.0
            out.loc[mask, c] = out.loc[mask, c] + sev*3.0*s
    elif fault == "pii":
        k = max(1, int(n*sev)); idx = rng.choice(out.index.to_numpy(), size=k, replace=False)
        for i, ri in enumerate(idx):
            out.at[ri, "operator_note"] = f"contact op_{i:05d}@example.com"
    elif fault == "fairness":
        adv = out["site"].eq("c").to_numpy(); t = out["target"].to_numpy(copy=True)
        t[adv] = rng.binomial(1, min(0.95, 0.5+sev), size=int(adv.sum()))
        t[~adv] = rng.binomial(1, max(0.05, 0.5-sev), size=int((~adv).sum()))
        out["target"] = t
    return out

def audit(df, seed):
    cfg = TabularAssessConfig(label_col="target", split_col="split", time_col="timestamp",
        group_cols=["site"], id_cols=["sample_id"], mode="Quick Scan",
        thresholds=TABULAR_PRESETS["Balanced (recommended)"], random_state=seed)
    return analyze_tabular_dataframe(df, config=cfg, dataset_bytes=dataframe_to_bytes(df),
        dataset_name="c.csv", preset="Balanced (recommended)", mode="Quick Scan",
        use_auto_columns=False)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=list(DSETS))
    args = ap.parse_args()
    out = ROOT/"experiments"/"results"/"p2_audit"; out.mkdir(parents=True, exist_ok=True)
    path = out/"carriers.csv"
    done = set()
    if path.exists():
        done = set(pd.read_csv(path)["dataset"].unique().tolist())
    for name in args.datasets:
        if name in done:
            print("skip", name); continue
        base, num = load_carrier(DSETS[name])
        rows = []
        for seed in SEEDS:
            # clean negative (one per fault's metric): audit clean once, eval each metric
            rc = audit(base, seed)
            for f in FAULTS:
                prim = extract_primary_metric(rc, f)
                rows.append(dict(dataset=name, seed=seed, fault=f, severity=0.0,
                    value=float(prim["value"]), detected=bool(prim["detected"]), score=rc["score"]))
            for f in FAULTS:
                for s in SEV:
                    d = inject(base, num, f, s, seed*100+int(s*100))
                    r = audit(d, seed); prim = extract_primary_metric(r, f)
                    rows.append(dict(dataset=name, seed=seed, fault=f, severity=s,
                        value=float(prim["value"]), detected=bool(prim["detected"]), score=r["score"]))
        df = pd.DataFrame(rows)
        df.to_csv(path, mode="a", header=not path.exists(), index=False)
        print("appended", name, df.shape, flush=True)
    print("done", path)

if __name__ == "__main__":
    raise SystemExit(main())
