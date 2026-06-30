"""Paper 2 (audit-correctness): synthetic fault-DETECTION grid.
Identical injectors/detection definitions to fault_injection_tabular; sweeps all
three gate presets across seeds. Writes/append per preset, resumable per seed.
results/p2_audit/detection_grid_<preset>.csv  (no model trained)
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from astrid_core import (TabularAssessConfig, TABULAR_PRESETS,
                         analyze_tabular_dataframe, dataframe_to_bytes)
from experiments.fault_injection_tabular import (
    DEFAULT_FAULTS, make_clean_dataset, inject_fault,
    extract_primary_metric, recommendation_hit)
SEVERITIES = [0.0, 0.01, 0.05, 0.10, 0.20, 0.40]
SEEDS = [7, 11, 19]
DIMS = ["quality", "security", "reliability", "robustness", "fairness"]
MAXC = {"quality": 35.0, "security": 25.0, "reliability": 20.0,
        "robustness": 10.0, "fairness": 10.0}

def run_seed(preset, seed):
    rows = []
    clean = make_clean_dataset(n_rows=2500, seed=seed)
    for fi, fault in enumerate(DEFAULT_FAULTS):
        for si, sev in enumerate(SEVERITIES):
            corrupted = inject_fault(clean, fault, float(sev), seed + fi*10000 + si)
            cfg = TabularAssessConfig(label_col="target", split_col="split",
                time_col="timestamp", group_cols=["site"], id_cols=["sample_id"],
                mode="Quick Scan", thresholds=TABULAR_PRESETS[preset], random_state=seed)
            t0 = time.perf_counter()
            r = analyze_tabular_dataframe(corrupted, config=cfg,
                dataset_bytes=dataframe_to_bytes(corrupted), dataset_name="syn.csv",
                preset=preset, mode="Quick Scan", use_auto_columns=False)
            dt = time.perf_counter() - t0
            prim = extract_primary_metric(r, fault); comp = r.get("score_components", {})
            row = {"preset": preset.split()[0], "seed": seed, "fault": fault,
                   "severity": float(sev), "n_rows": len(corrupted), "score": r["score"],
                   "grade": r["grade"], "policy_fail": r["policy_result"]["status"] == "FAIL",
                   "metric": prim["metric"], "value": float(prim["value"]),
                   "detected": bool(prim["detected"]),
                   "rec_hit": bool(recommendation_hit(r, fault)), "runtime_s": dt}
            for d in DIMS:
                row["c_"+d] = float(comp.get(d, 0.0)); row["drop_"+d] = MAXC[d]-float(comp.get(d, 0.0))
            rows.append(row)
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True,
                    choices=["Strict", "Balanced (recommended)", "Lenient"])
    args = ap.parse_args()
    out_dir = ROOT/"experiments"/"results"/"p2_audit"; out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir/("detection_grid_"+args.preset.split()[0].lower()+".csv")
    done = set()
    if path.exists():
        done = set(pd.read_csv(path)["seed"].unique().tolist())
    for seed in SEEDS:
        if seed in done:
            print("skip seed", seed); continue
        df = run_seed(args.preset, seed)
        hdr = not path.exists()
        df.to_csv(path, mode="a", header=hdr, index=False)
        print("appended preset", args.preset.split()[0], "seed", seed, df.shape, flush=True)
    print("done", path)

if __name__ == "__main__":
    raise SystemExit(main())
