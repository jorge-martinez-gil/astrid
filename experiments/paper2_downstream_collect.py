"""Paper 2 (downstream validity) data collection.

One resumable pass that, for each (dataset, fault, severity, seed):
  * injects the fault into the training partition (clean 30% test held out),
  * scores the corrupted training matrix with ASTRID (composite + dimension
    sub-scores + policy-gate verdict),
  * trains an EXPANDED learner panel and records held-out ROC-AUC for each.

The resulting tidy CSV feeds three Paper-2 analyses:
  E6  expanded sensitivity law + bootstrap CIs   (per-learner score-AUC corr)
  E8  gate-as-harm-classifier                    (gate PASS/FAIL vs AUC drop)
  E9  dimension-ablation for prediction          (sub-score vs AUC)

Resumable + time-budgeted so it can be driven in <45s chunks:
    python experiments/paper2_downstream_collect.py --time-budget 40   # repeat
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the validated injectors / loader / dataset registry from the original runner.
from experiments.reliability_vs_performance import (  # noqa: E402
    DATASETS as BASE_DATASETS,
    FAULTS,
    load_dataset,
)
from astrid_core import analyze_tabular_dataframe  # noqa: E402

from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

# Mechanical Analysis carries destroyable signal (clean AUC well above chance); add it.
DATASETS = dict(BASE_DATASETS)
DATASETS["mechanical"] = {
    "path": "Datasets/mechanical analysis/astrid_dataset.csv",
    "label": "label",
    "drop_cols": [],
    "max_rows": 2500,
    "max_features": 120,
}
# Cap the large boundary datasets for tractable multi-seed scoring (paper subsamples too).
DATASETS["aps"] = {**DATASETS["aps"], "max_rows": 2500, "max_features": 80}
DATASETS["secom"] = {**DATASETS["secom"], "max_features": 80}

DEFAULT_DATASETS = ["cylinder_bands", "mechanical", "secom", "aps"]
DEFAULT_FAULTS = ["missingness", "outliers", "duplicates", "combined", "label_noise"]
DEFAULT_SEVERITIES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
DEFAULT_SEEDS = [7, 11, 19, 23, 31]

DIMENSIONS = ["quality", "security", "reliability", "robustness", "fairness"]


def _linear():
    # deliberately defect-sensitive linear pipeline
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )


def make_learners(seed: int):
    """Fresh learner panel (imputation built in for the scale-sensitive ones)."""
    imp = lambda est: make_pipeline(SimpleImputer(strategy="median"), est)
    scl = lambda est: make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), est
    )
    return {
        "HistGBM": HistGradientBoostingClassifier(random_state=seed),
        "RandomForest": imp(RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)),
        "ExtraTrees": imp(ExtraTreesClassifier(n_estimators=100, random_state=seed, n_jobs=-1)),
        "DecisionTree": imp(DecisionTreeClassifier(random_state=seed)),
        "AdaBoost": imp(AdaBoostClassifier(random_state=seed)),
        "GaussianNB": imp(GaussianNB()),
        "kNN": scl(KNeighborsClassifier(n_neighbors=15)),
        "LogReg": scl(LogisticRegression(max_iter=1000)),
        "LinearSVM": scl(LinearSVC(C=1.0, max_iter=2000)),
        "MLP": scl(MLPClassifier(hidden_layer_sizes=(48,), max_iter=200, random_state=seed)),
    }


def auc_of(model, Xtr, ytr, Xte, yte) -> float:
    model.fit(Xtr, ytr)
    if hasattr(model, "predict_proba"):
        s = model.predict_proba(Xte)[:, 1]
    else:  # LinearSVM (no proba) -> decision function
        s = model.decision_function(Xte)
    try:
        return float(roc_auc_score(yte, s))
    except Exception:
        return float("nan")


def astrid_full(frame: pd.DataFrame):
    r = analyze_tabular_dataframe(
        frame, dataset_name="run", preset="Balanced (recommended)", mode="Quick Scan"
    )
    comps = r.get("score_components", {}) or {}
    status = r.get("verdict_status") or r.get("verdict") or ""
    pol = r.get("policy_result") or {}
    if isinstance(pol, dict):
        status = pol.get("status", status)
    return r["score"], r["grade"], comps, str(status)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--faults", nargs="+", default=DEFAULT_FAULTS)
    ap.add_argument("--severities", type=float, nargs="+", default=DEFAULT_SEVERITIES)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--time-budget", type=float, default=0.0)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "experiments" / "results" / "paper2_downstream"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "downstream_panel.csv"

    done = set()
    if out_csv.exists() and not args.fresh and out_csv.stat().st_size > 0:
        prev = pd.read_csv(out_csv)
        done = {
            (r["dataset"], r["fault"], round(float(r["severity"]), 4), int(r["seed"]))
            for _, r in prev.iterrows()
        }
    write_header = not out_csv.exists() or args.fresh or out_csv.stat().st_size == 0
    if args.fresh and out_csv.exists():
        out_csv.unlink()
        write_header = True

    # cache: dataset -> split (clean test never corrupted)
    cache = {}

    def get_ds(name):
        if name not in cache:
            frame, feats, label = load_dataset(name, DATASETS[name])
            tr, te = train_test_split(
                frame, test_size=0.30, random_state=0, stratify=frame[label]
            )
            cache[name] = dict(
                feats=feats, label=label,
                train=tr.reset_index(drop=True), test=te.reset_index(drop=True),
            )
        return cache[name]

    plan = [
        (d, f, round(float(s), 4), seed)
        for d in args.datasets for f in args.faults
        for s in args.severities for seed in args.seeds
    ]
    todo = [k for k in plan if k not in done]
    print(f"plan={len(plan)} done={len(done)} todo={len(todo)}")

    t0 = time.time()
    n = 0
    for (dname, fault, sev, seed) in todo:
        if args.time_budget and (time.time() - t0) > args.time_budget:
            break
        ds = get_ds(dname)
        feats, label = ds["feats"], ds["label"]
        train, test = ds["train"], ds["test"]
        Xte, yte = test[feats], test[label].astype(int).values
        rng = np.random.default_rng(seed)
        corrupted = train.copy() if sev == 0.0 else FAULTS[fault](train, feats, label, sev, rng)
        score, grade, comps, status = astrid_full(corrupted[feats + [label]])
        Xtr, ytr = corrupted[feats], corrupted[label].astype(int).values
        row = {
            "dataset": dname, "fault": fault, "severity": sev, "seed": seed,
            "astrid_score": score, "grade": grade, "gate_status": status,
        }
        for dim in DIMENSIONS:
            row[f"pen_{dim}"] = comps.get(dim, np.nan)
        for lname, model in make_learners(seed).items():
            row[f"auc_{lname}"] = auc_of(model, Xtr, ytr, Xte, yte)
        pd.DataFrame([row]).to_csv(
            out_csv, mode="a", header=write_header, index=False
        )
        write_header = False
        n += 1
    elapsed = time.time() - t0
    remaining = len(todo) - n
    print(f"wrote {n} rows in {elapsed:.1f}s | remaining {remaining}")
    if remaining == 0:
        print("COLLECTION COMPLETE")


if __name__ == "__main__":
    main()
