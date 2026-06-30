"""Paper 2 (E3): reference-tool AGREEMENT.
On identical injected faults, compare ASTRID's flag/value against an independent
specialist tool on its own turf:
  fairness -> Fairlearn demographic_parity_difference
  drift    -> Alibi Detect KSDrift
  pii      -> Microsoft Presidio EmailRecognizer
Writes results/p2_audit/reference_agreement.csv  (no model trained)
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from astrid_core import (TabularAssessConfig, TABULAR_PRESETS,
                         analyze_tabular_dataframe, dataframe_to_bytes)
from experiments.fault_injection_tabular import (
    make_clean_dataset, inject_fault, _max_drift_ks, _max_positive_rate_disparity)

from fairlearn.metrics import demographic_parity_difference
from alibi_detect.cd import KSDrift
from presidio_analyzer.predefined_recognizers import EmailRecognizer

SEED_LIST = [7, 11, 19]
SEV = [0.0, 0.05, 0.10, 0.20, 0.40]
NUM = ["sensor_temp", "sensor_pressure", "vibration_rms", "humidity"]
EMAIL = EmailRecognizer()

def astrid_report(df, seed):
    cfg = TabularAssessConfig(label_col="target", split_col="split", time_col="timestamp",
        group_cols=["site"], id_cols=["sample_id"], mode="Quick Scan",
        thresholds=TABULAR_PRESETS["Balanced (recommended)"], random_state=seed)
    return analyze_tabular_dataframe(df, config=cfg, dataset_bytes=dataframe_to_bytes(df),
        dataset_name="x.csv", preset="Balanced (recommended)", mode="Quick Scan",
        use_auto_columns=False)

def ref_fairness(df):
    y = df["target"].to_numpy()
    return float(demographic_parity_difference(y, y, sensitive_features=df["site"].to_numpy()))

def ref_drift(df):
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    ref = df.loc[ts < ts.median(), NUM].to_numpy(float)
    cut = ts.quantile(0.75)
    test = df.loc[ts >= cut, NUM].to_numpy(float)
    ref = np.nan_to_num(ref); test = np.nan_to_num(test)
    det = KSDrift(ref, p_val=0.05)
    pred = det.predict(test)
    dist = pred["data"]["distance"]
    return int(pred["data"]["is_drift"]), float(np.max(dist))

def ref_pii(df):
    col = df["operator_note"].astype(str)
    hits = sum(1 for v in col if EMAIL.analyze(v, entities=["EMAIL_ADDRESS"], nlp_artifacts=None))
    return int(hits > 0), float(hits) / max(1, len(col))   # fraction of rows flagged

def main():
    rows = []
    for seed in SEED_LIST:
        clean = make_clean_dataset(2500, seed=seed)
        for sev in SEV:
            # fairness
            d = inject_fault(clean, "fairness", sev, seed+1)
            r = astrid_report(d, seed)
            a_val = _max_positive_rate_disparity(r); a_flag = int(a_val > 0.20)
            r_val = ref_fairness(d); r_flag = int(r_val > 0.20)
            rows.append(dict(tool="Fairlearn", fault="fairness", severity=sev, seed=seed,
                astrid_value=a_val, ref_value=r_val, astrid_flag=a_flag, ref_flag=r_flag))
            # drift
            d = inject_fault(clean, "drift", sev, seed+2)
            r = astrid_report(d, seed)
            a_val = _max_drift_ks(r)
            a_flag = int(a_val > r["config"]["thresholds"]["drift_ks_threshold"])
            r_flag, r_val = ref_drift(d)
            rows.append(dict(tool="AlibiDetect", fault="drift", severity=sev, seed=seed,
                astrid_value=a_val, ref_value=r_val, astrid_flag=a_flag, ref_flag=r_flag))
            # pii
            d = inject_fault(clean, "pii", sev, seed+3)
            r = astrid_report(d, seed)
            cwh = r["report"]["security"]["confidentiality_pii_heuristics"]["columns_with_hits"]
            a_val = float(cwh.get("operator_note", {}).get("email", 0.0))
            a_flag = int(len(cwh) > 0)
            r_flag, r_val = ref_pii(d)
            rows.append(dict(tool="Presidio", fault="pii", severity=sev, seed=seed,
                astrid_value=a_val, ref_value=r_val, astrid_flag=a_flag, ref_flag=r_flag))
    out = ROOT/"experiments"/"results"/"p2_audit"; out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(out/"reference_agreement.csv", index=False)
    print("wrote", out/"reference_agreement.csv", df.shape)
    for tool in ["Fairlearn", "AlibiDetect", "Presidio"]:
        s = df[df.tool == tool]
        agree = (s.astrid_flag == s.ref_flag).mean()
        from scipy.stats import pearsonr
        try: pr = pearsonr(s.astrid_value, s.ref_value)[0]
        except Exception: pr = float("nan")
        print(f"{tool}: decision-agreement={agree:.2f}  value-r={pr:.3f}  n={len(s)}")

if __name__ == "__main__":
    raise SystemExit(main())
