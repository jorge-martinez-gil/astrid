"""Reliability-score vs. downstream-performance correlation experiment for ASTRID.

For each real industrial dataset we hold out a clean stratified test set, then
inject controlled reliability faults into the training partition at increasing
severities. At every step we (a) score the corrupted training set with the
headless ASTRID tabular analyzer and (b) train a HistGradientBoosting classifier
on that same corrupted training set and measure ROC-AUC on the clean test set.

This yields paired (reliability_score, test_AUC) observations whose correlation
quantifies how well the ASTRID composite reliability score tracks downstream
model performance. The runner is resumable and time-budgeted so it can be driven
in short chunks under a hard per-call wall-clock limit:

    python experiments/reliability_vs_performance.py --fresh --time-budget 40
    python experiments/reliability_vs_performance.py --time-budget 40   # resume
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astrid_core import analyze_tabular_dataframe  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

# drop_cols encode the raw label (leakage) or are identifiers/time columns.
DATASETS = {
    "cylinder_bands": {
        "path": "Datasets/cilinder bands/astrid_dataset.csv",
        "label": "label",
        "drop_cols": ["band_type", "timestamp"],
        "max_rows": None,
    },
    "secom": {
        "path": "Datasets/secom/astrid_dataset.csv",
        "label": "label",
        "drop_cols": ["class_label", "timestamp"],
        "max_rows": None,
        "max_features": 120,
    },
    "aps": {
        "path": "Datasets/aps/astrid_dataset.csv",
        "label": "label",
        "drop_cols": ["class"],
        "max_rows": 8000,
        "max_features": 120,
    },
    "robot": {
        "path": "Datasets/robot/astrid_dataset.csv",
        "label": "failure_binary",
        "drop_cols": ["class_label", "source_file", "learning_problem", "task_description"],
        "max_rows": None,
    },
}

DEFAULT_FAULTS = ["missingness", "outliers", "duplicates", "combined", "label_noise"]
DEFAULT_SEVERITIES = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
DEFAULT_SEEDS = [7, 11, 19]
OUTLIER_MAGNITUDE = 15.0


def load_dataset(name: str, cfg: Dict) -> Tuple[pd.DataFrame, List[str], str]:
    df = pd.read_csv(ROOT / cfg["path"])
    label = cfg["label"]
    df = df[df[label].notna()].copy()
    df[label] = df[label].astype(int)
    if cfg.get("max_rows") and len(df) > cfg["max_rows"]:
        df, _ = train_test_split(
            df, train_size=cfg["max_rows"], random_state=0, stratify=df[label]
        )
        df = df.reset_index(drop=True)
    drop = set(cfg.get("drop_cols", [])) | {label}
    features = [c for c in df.select_dtypes(include="number").columns if c not in drop]
    k = cfg.get("max_features")
    if k and len(features) > k:
        variances = df[features].var().sort_values(ascending=False)
        features = list(variances.index[:k])
    frame = df[features + [label]].reset_index(drop=True)
    frame[features] = frame[features].astype(float)
    return frame, features, label


def inject_missingness(train, features, label, sev, rng):
    out = train.copy()
    arr = out[features].to_numpy(dtype=float)
    arr[rng.random(arr.shape) < sev] = np.nan
    out[features] = arr
    return out


def inject_outliers(train, features, label, sev, rng):
    out = train.copy()
    arr = out[features].to_numpy(dtype=float)
    std = np.nanstd(arr, axis=0)
    std[std == 0] = 1.0
    n = int(round(sev * len(out)))
    if n > 0:
        rows = rng.choice(len(out), size=n, replace=False)
        signs = rng.choice([-1.0, 1.0], size=(n, arr.shape[1]))
        arr[rows] = arr[rows] + signs * OUTLIER_MAGNITUDE * std
    out[features] = arr
    return out


def inject_duplicates(train, features, label, sev, rng):
    n = int(round(sev * len(train)))
    if n == 0:
        return train.copy()
    idx = rng.choice(len(train), size=n, replace=True)
    return pd.concat([train, train.iloc[idx]], ignore_index=True)


def inject_label_noise(train, features, label, sev, rng):
    out = train.copy()
    flip = rng.random(len(out)) < sev
    out.loc[flip, label] = 1 - out.loc[flip, label]
    return out


def inject_combined(train, features, label, sev, rng):
    out = inject_missingness(train, features, label, sev, rng)
    out = inject_outliers(out, features, label, sev * 0.5, rng)
    out = inject_duplicates(out, features, label, sev, rng)
    return out


FAULTS = {
    "missingness": inject_missingness,
    "outliers": inject_outliers,
    "duplicates": inject_duplicates,
    "label_noise": inject_label_noise,
    "combined": inject_combined,
}


def astrid_score(frame: pd.DataFrame):
    res = analyze_tabular_dataframe(
        frame, dataset_name="run", preset="Balanced (recommended)", mode="Quick Scan"
    )
    return res["score"], res["grade"], res.get("score_components", {})


def downstream_auc(train, test_X, test_y, features, label, seed) -> float:
    model = HistGradientBoostingClassifier(random_state=seed)
    model.fit(train[features], train[label].astype(int))
    proba = model.predict_proba(test_X)[:, 1]
    return roc_auc_score(test_y, proba)


def run(args) -> None:
    out_dir = ROOT / "experiments" / "results" / "reliability_vs_performance"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = out_dir / "reliability_vs_performance_runs.csv"
    runs_json = out_dir / "reliability_vs_performance_runs.json"
    done_marker = out_dir / "_DONE"

    if args.fresh:
        if runs_csv.exists():
            runs_csv.write_text("")
        if done_marker.exists():
            done_marker.write_text("")

    records: List[Dict] = []
    done_keys = set()
    if not args.fresh and runs_csv.exists() and runs_csv.stat().st_size > 0:
        prev = pd.read_csv(runs_csv)
        records = prev.to_dict("records")
        done_keys = {
            (r["dataset"], r["fault"], round(float(r["severity"]), 4), int(r["seed"]))
            for r in records
        }

    plan = [
        (d, f, round(float(s), 4), seed)
        for d in args.datasets
        for f in args.faults
        for s in args.severities
        for seed in args.seeds
    ]
    todo = [k for k in plan if k not in done_keys]
    if not todo:
        runs_json.write_text(json.dumps(records, indent=2))
        done_marker.write_text(str(len(records)) + " runs complete\n")
        print("ALL DONE: " + str(len(records)) + " runs already complete")
        return

    t0 = time.time()
    cache: Dict[str, Dict] = {}

    def get_ds(dname):
        if dname not in cache:
            cfg = DATASETS[dname]
            frame, features, label = load_dataset(dname, cfg)
            train_base, test = train_test_split(
                frame, test_size=0.30, random_state=0, stratify=frame[label]
            )
            cache[dname] = dict(
                features=features,
                label=label,
                train_base=train_base.reset_index(drop=True),
                test=test.reset_index(drop=True),
                pos_rate=float(frame[label].mean()),
            )
        return cache[dname]

    n_new = 0
    for (dname, fault, sev, seed) in todo:
        if args.time_budget and (time.time() - t0) > args.time_budget:
            break
        ds = get_ds(dname)
        features, label = ds["features"], ds["label"]
        train_base, test = ds["train_base"], ds["test"]
        test_X = test[features]
        test_y = test[label].astype(int).values
        rng = np.random.default_rng(seed)
        if sev == 0.0:
            corrupted = train_base.copy()
        else:
            corrupted = FAULTS[fault](train_base, features, label, sev, rng)
        score, grade, comps = astrid_score(corrupted[features + [label]])
        auc = downstream_auc(corrupted, test_X, test_y, features, label, seed)
        records.append({
            "dataset": dname,
            "n_features": len(features),
            "base_pos_rate": round(ds["pos_rate"], 5),
            "fault": fault,
            "severity": sev,
            "seed": seed,
            "astrid_score": score,
            "grade": grade,
            "test_auc": round(float(auc), 5),
            "q": comps.get("quality"),
            "sec": comps.get("security"),
            "rel": comps.get("reliability"),
            "rob": comps.get("robustness"),
            "fair": comps.get("fairness"),
        })
        n_new += 1
        pd.DataFrame(records).to_csv(runs_csv, index=False)
        print("[" + str(len(records)) + "/" + str(len(plan)) + "] " + dname
              + " " + fault + " sev=" + str(sev) + " seed=" + str(seed)
              + " -> score=" + str(score) + " auc=" + str(round(auc, 3))
              + " (" + str(round(time.time() - t0, 1)) + "s)", flush=True)

    remaining = len(plan) - len(records)
    if remaining <= 0:
        runs_json.write_text(json.dumps(records, indent=2))
        done_marker.write_text(str(len(records)) + " runs complete\n")
        print("ALL DONE: " + str(len(records)) + " runs complete")
    else:
        print("CHUNK DONE: +" + str(n_new) + " new, " + str(remaining) + " remaining")


def parse_args():
    p = argparse.ArgumentParser(description="ASTRID reliability vs. performance sweep")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--faults", nargs="+", default=DEFAULT_FAULTS)
    p.add_argument("--severities", type=float, nargs="+", default=DEFAULT_SEVERITIES)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--time-budget", type=float, default=0.0)
    p.add_argument("--fresh", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
