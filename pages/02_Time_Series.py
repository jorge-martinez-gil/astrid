"""Time Series Dataset Safety Analyzer — upgraded version."""
from __future__ import annotations

import json
import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import (
    SHARED_CSS, to_json_safe, sha256_bytes, clip_text,
    badge, kpi, health_ring_html, progress_bar_html, check_status_card,
    compute_health_score, get_dimension_status, render_transparency_tab,
    build_html_report, PII_PATTERNS, infer_column_types,
    numeric_cols, categorical_cols, approx_iqr_outlier_rate, ks_statistic,
)

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

st.set_page_config(page_title="Time Series Analyzer", page_icon="📈", layout="wide")
st.markdown(SHARED_CSS, unsafe_allow_html=True)


@dataclass(frozen=True)
class Thresholds:
    drift_ks_threshold: float
    pii_hit_rate_threshold: float
    time_parse_ok_min: float
    dup_timestamp_rate_max: float
    cadence_irregularity_max: float

PRESETS: Dict[str, Thresholds] = {
    "Balanced (recommended)": Thresholds(0.30, 0.01, 0.98, 0.01, 2.0),
    "Strict":                 Thresholds(0.20, 0.005, 0.995, 0.005, 1.5),
    "Lenient":                Thresholds(0.40, 0.02, 0.95, 0.02, 3.0),
}

@dataclass
class AssessConfig:
    label_col: Optional[str]
    split_col: Optional[str]
    time_col: Optional[str]
    entity_cols: List[str]
    group_cols: List[str]
    annotator_label_cols: List[str]
    id_cols: List[str]
    time_slice_mode: str = "month"
    random_state: int = 7
    max_categories_for_stats: int = 50
    thresholds: Thresholds = field(default_factory=lambda: PRESETS["Balanced (recommended)"])
    mode: str = "Quick Scan"
    pii_max_rows: int = 2000
    pii_max_text_cols: int = 10
    rare_max_cat_cols: int = 50
    drift_max_num_cols: int = 50


def _name_score(name: str, patterns: List[str]) -> float:
    n = name.lower().strip()
    return sum(1.0 for p in patterns if re.search(p, n))


def to_datetime_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s.dtype): return s
    try: return pd.to_datetime(s, errors="coerce", utc=False)
    except: return s


def time_slice_labels(t: pd.Series, mode: str) -> Optional[pd.Series]:
    if t.isna().all(): return None
    mapping = {"month": "M", "week": "W", "day": "D", "quarter": "Q"}
    freq = mapping.get(mode)
    if freq: return t.dt.to_period(freq).astype("string")
    return None


def guess_ts_columns(df: pd.DataFrame) -> Dict[str, Any]:
    cols = df.columns.tolist()
    nrows = max(1, len(df))
    nunique = {c: int(df[c].nunique(dropna=True)) for c in cols}
    uniq_ratio = {c: float(nunique[c] / nrows) for c in cols}
    dt_success: Dict[str, float] = {}
    for c in cols[:300]:
        if pd.api.types.is_datetime64_any_dtype(df[c].dtype): dt_success[c] = 1.0
        elif pd.api.types.is_numeric_dtype(df[c].dtype): dt_success[c] = 0.0
        else:
            p = to_datetime_series(df[c])
            dt_success[c] = float(p.notna().mean()) if pd.api.types.is_datetime64_any_dtype(p.dtype) else 0.0

    time_pats   = [r"\btime\b", r"\bdate\b", r"\btimestamp\b", r"\bcreated\b", r"\bupdated\b"]
    entity_pats = [r"\bentity\b", r"\bseries\b", r"\bdevice\b", r"\bsensor\b", r"\buser\b", r"\bsite\b"]
    label_pats  = [r"\blabel\b", r"\btarget\b", r"\boutcome\b", r"\bclass\b"]
    grp_pats    = [r"\bgender\b", r"\bsex\b", r"\bage\b", r"\bregion\b", r"\bcountry\b"]
    id_pats     = [r"\bid\b", r"\buuid\b", r"\bguid\b", r"\bserial\b"]

    def rank_time(c):
        return _name_score(c, time_pats) * 3 + dt_success.get(c, 0) * 3 - (2.0 if dt_success.get(c, 0) < 0.3 else 0)
    def rank_entity(c):
        s = _name_score(c, entity_pats) * 2.5
        if pd.api.types.is_numeric_dtype(df[c].dtype): s -= 2
        if dt_success.get(c, 0) > 0.8: s -= 3
        if 2 <= nunique[c] <= min(2000, int(0.2 * nrows) + 2): s += 2
        return s
    def rank_label(c):
        s = _name_score(c, label_pats) * 3
        if dt_success.get(c, 0) > 0.8: s -= 3
        if uniq_ratio[c] > 0.9: s -= 2
        if 2 <= nunique[c] <= 50: s += 2
        return s
    def rank_grp(c):
        s = _name_score(c, grp_pats) * 2
        if 2 <= nunique[c] <= 50: s += 2
        if dt_success.get(c, 0) > 0.8: s -= 2
        return s
    def rank_id(c):
        s = _name_score(c, id_pats) * 2.5
        if uniq_ratio[c] > 0.95: s += 3
        if dt_success.get(c, 0) > 0.8: s -= 3
        return s

    sorted_time   = sorted(cols, key=rank_time,   reverse=True)
    sorted_entity = sorted(cols, key=rank_entity, reverse=True)
    sorted_label  = sorted(cols, key=rank_label,  reverse=True)
    sorted_grp    = sorted(cols, key=rank_grp,    reverse=True)
    sorted_id     = sorted(cols, key=rank_id,     reverse=True)

    time_col  = sorted_time[0]  if sorted_time  and rank_time(sorted_time[0])   >= 2 else None
    label_col = sorted_label[0] if sorted_label and rank_label(sorted_label[0]) >= 2 else None

    entity_cols, id_cols, grp_cols = [], [], []
    for c in sorted_entity:
        if c in {time_col, label_col}: continue
        if rank_entity(c) >= 2.5: entity_cols.append(c)
        if len(entity_cols) >= 2: break
    for c in sorted_id:
        if c in {time_col, label_col} or c in set(entity_cols): continue
        if rank_id(c) >= 3: id_cols.append(c)
        if len(id_cols) >= 3: break
    for c in sorted_grp:
        if c in {time_col, label_col} or c in set(entity_cols) or c in set(id_cols): continue
        if rank_grp(c) >= 2.5: grp_cols.append(c)
        if len(grp_cols) >= 3: break

    notes = ([f"Guessed time: {time_col}"] if time_col else []) + \
            ([f"Guessed entities: {', '.join(entity_cols)}"] if entity_cols else []) + \
            ([f"Guessed label: {label_col}"] if label_col else []) + \
            ([f"Guessed groups: {', '.join(grp_cols)}"] if grp_cols else [])

    return {"time_col": time_col, "entity_cols": entity_cols, "label_col": label_col,
            "id_cols": id_cols, "group_cols": grp_cols, "notes": notes}


def series_time_profile(df, time_col, entity_cols):
    t = to_datetime_series(df[time_col])
    ok = t.notna()
    out: Dict[str, Any] = {"time_parse": {"time_col": time_col, "parse_ok_rate": float(ok.mean()), "num_invalid": int((~ok).sum())}}
    if ok.sum() == 0: return {**out, "note": "Parsing failed for all rows."}

    dft = df.loc[ok].copy()
    dft["_t"] = t.loc[ok].astype("datetime64[ns]")
    out["time_range"] = {"min": str(dft["_t"].min()), "max": str(dft["_t"].max()),
                         "span_days": float((dft["_t"].max() - dft["_t"].min()) / pd.Timedelta(days=1))}

    sort_cols = (entity_cols or []) + ["_t"]
    dft = dft.sort_values(sort_cols, kind="mergesort")

    dup_keys = (entity_cols + [time_col]) if entity_cols else [time_col]
    out["duplicate_timestamps"] = {"keys": dup_keys, "duplicate_rate": float(dft.duplicated(subset=(entity_cols + ["_t"]) if entity_cols else ["_t"]).mean())}

    dt = dft["_t"].diff().dropna()
    dt_sec = dt.dt.total_seconds().to_numpy()
    dt_sec = dt_sec[dt_sec > 0]
    if len(dt_sec) >= 50:
        med = float(np.median(dt_sec))
        p10, p90 = float(np.percentile(dt_sec, 10)), float(np.percentile(dt_sec, 90))
        out["cadence_global"] = {"median_step_seconds": med, "p10_seconds": p10, "p90_seconds": p90,
                                  "irregularity_ratio": (p90-p10)/max(1e-9, med)}
    else:
        out["cadence_global"] = {"note": "Not enough consecutive timestamps."}

    gap = dt.dt.total_seconds()
    med_gap = gap.median()
    out["gaps_global"] = {
        "median_gap_seconds": float(med_gap) if pd.notna(med_gap) else None,
        "p95_gap_seconds":    float(gap.quantile(0.95)) if len(gap) else None,
        "largest_gap_seconds":float(gap.max()) if len(gap) else None,
        "large_gap_rate":     float((gap > 10 * med_gap).mean()) if (med_gap and med_gap > 0 and len(gap)) else None,
    }

    if entity_cols:
        counts = dft.groupby(entity_cols, dropna=False).size().sort_values(ascending=False)
        out["entities"] = {"num_entities": int(counts.shape[0]),
                           "top_entities_counts": {str(k): int(v) for k, v in counts.head(10).items()}}
    return out


def drift_ks_first_last(df, time_col, numeric, slice_mode):
    t = to_datetime_series(df[time_col])
    ok = t.notna()
    if ok.sum() == 0: return {"note": "Time parsing produced no valid timestamps."}
    dft = df.loc[ok].copy()
    dft["_t"] = t.loc[ok].astype("datetime64[ns]")
    slices = time_slice_labels(dft["_t"], slice_mode)
    if slices is None: return {"note": "Slice labeling failed."}
    uniq = pd.Series(slices).dropna().unique().tolist()
    if len(uniq) < 2: return {"slice_mode": slice_mode, "note": "Need ≥ 2 slices for drift."}
    uniq_sorted = sorted(map(str, uniq))
    s1, s2 = uniq_sorted[0], uniq_sorted[-1]
    s_ser = pd.Series(slices).astype("string")
    g1, g2 = dft[s_ser == s1], dft[s_ser == s2]
    drift = {c: d for c in numeric
             if (d := ks_statistic(pd.to_numeric(g1[c], errors="coerce").to_numpy(),
                                   pd.to_numeric(g2[c], errors="coerce").to_numpy())) is not None}
    return {"slice_mode": slice_mode, "first_slice": s1, "last_slice": s2,
            "first_slice_rows": int(len(g1)), "last_slice_rows": int(len(g2)),
            "top_10_ks": dict(sorted(drift.items(), key=lambda kv: kv[1], reverse=True)[:10]),
            "num_numeric_evaluated": int(len(drift))}


def assess_quality(df, cfg):
    out: Dict[str, Any] = {}
    miss = df.isna().mean().sort_values(ascending=False)
    out["missingness"] = {"overall_missing_rate": float(df.isna().mean().mean()),
                           "top_10_columns_missing_rate": miss.head(10).to_dict()}
    out["duplicates"] = {"exact_duplicate_row_rate": float(df.duplicated().mean())}

    num = numeric_cols(df, exclude=[c for c in [cfg.label_col, cfg.split_col, cfg.time_col] if c])
    outlier_rates = {c: r for c in num if (r := approx_iqr_outlier_rate(df[c])) is not None}
    out["outliers_iqr"] = {"columns_evaluated": len(outlier_rates),
                            "top_10_outlier_rate": dict(sorted(outlier_rates.items(), key=lambda kv: kv[1], reverse=True)[:10])}

    if cfg.label_col and cfg.label_col in df.columns:
        y = df[cfg.label_col]
        vc = y.value_counts(dropna=True)
        out["label_stats"] = {"label_missing_rate": float(y.isna().mean()), "label_cardinality": int(y.nunique(dropna=True)),
                               "top_classes_share": (vc / vc.sum()).head(10).to_dict() if 2 <= len(vc) <= cfg.max_categories_for_stats else None}

    if cfg.split_col and cfg.split_col in df.columns:
        split = df[cfg.split_col].astype("string")
        row_hash = pd.util.hash_pandas_object(df[[c for c in df.columns if c != cfg.split_col]], index=False).astype("uint64")
        distinct = pd.DataFrame({"split": split, "row_hash": row_hash}).groupby("row_hash")["split"].nunique()
        out["split_leakage"] = {"row_hash_cross_split_rate": float((distinct > 1).mean()) if len(distinct) else None}

    if cfg.time_col and cfg.time_col in df.columns:
        out["time_axis_health"] = series_time_profile(df, cfg.time_col, [c for c in cfg.entity_cols if c in df.columns])
    else:
        out["time_axis_health"] = {"note": "Select a time column to compute time-axis health."}
    return out


def assess_reliability(df, cfg):
    out: Dict[str, Any] = {}
    if not (cfg.time_col and cfg.time_col in df.columns):
        return {"note": "Select a time column for time slicing, stability, and drift."}

    t = to_datetime_series(df[cfg.time_col])
    ok = t.notna()
    if ok.sum() == 0: return {"note": "Time column parsing produced no valid timestamps."}
    dft = df.loc[ok].copy(); dft["_t"] = t.loc[ok].astype("datetime64[ns]")
    slices = time_slice_labels(dft["_t"], cfg.time_slice_mode)
    if slices is None: return {"note": "Slice labeling failed."}

    out["slice_mode"] = cfg.time_slice_mode
    out["missing_rate_by_time_slice"] = {str(sv): float(g.isna().mean().mean()) for sv, g in dft.groupby(slices, dropna=False)}

    exclude = [c for c in [cfg.label_col, cfg.split_col, cfg.time_col] if c]
    num_all = numeric_cols(df, exclude=exclude)
    num = num_all
    if cfg.drift_max_num_cols and len(num_all) > cfg.drift_max_num_cols:
        variances = [(c, float(pd.to_numeric(df[c], errors="coerce").var(skipna=True))) for c in num_all]
        num = [c for c, _ in sorted(variances, key=lambda t: t[1], reverse=True)[:cfg.drift_max_num_cols]]

    out["numeric_drift_ks_first_last"] = drift_ks_first_last(df, cfg.time_col, num, cfg.time_slice_mode)
    out["schema_consistency"] = {"num_rows": int(len(df)), "num_cols": int(df.shape[1]),
                                  "dtypes": infer_column_types(df),
                                  "constant_columns": [c for c in df.columns if df[c].nunique(dropna=False) <= 1]}
    return out


def assess_robustness(df, cfg):
    out: Dict[str, Any] = {"sklearn_available": SKLEARN_OK}
    exclude = [c for c in [cfg.label_col, cfg.split_col, cfg.time_col] if c]

    num = numeric_cols(df, exclude=exclude)
    if num:
        X = df[num].apply(pd.to_numeric, errors="coerce")
        med = X.median(axis=0, skipna=True)
        mad = (X - med).abs().median(axis=0, skipna=True).replace(0, np.nan)
        row_score = (X - med).abs().divide(mad).mean(axis=1, skipna=True)
        out["row_anomaly_score_mad"] = {"mean": float(row_score.mean(skipna=True)), "p95": float(row_score.quantile(0.95)),
                                         "p99": float(row_score.quantile(0.99)), "max": float(row_score.max(skipna=True)),
                                         "top_20_row_indices": row_score.sort_values(ascending=False).head(20).index.tolist()}

    if cfg.time_col and cfg.time_col in df.columns:
        prof = series_time_profile(df, cfg.time_col, [c for c in cfg.entity_cols if c in df.columns])
        out["time_series_stressors"] = {"cadence": prof.get("cadence_global", {}), "gaps": prof.get("gaps_global", {})}
    return out


def assess_fairness(df, cfg):
    if not cfg.group_cols: return {"note": "Select group columns to compute fairness checks."}
    out: Dict[str, Any] = {}
    label_ok = bool(cfg.label_col and cfg.label_col in df.columns)
    per: Dict[str, Any] = {}
    for gcol in [c for c in cfg.group_cols if c in df.columns]:
        counts = df[gcol].value_counts(dropna=False)
        shares = counts / max(1, counts.sum())
        stats: Dict[str, Any] = {"num_groups": int(len(counts)),
                                  "min_group_share": float(shares.min()) if len(shares) else None,
                                  "max_group_share": float(shares.max()) if len(shares) else None,
                                  "representation_share_top10": shares.sort_values(ascending=False).head(10).to_dict()}
        miss_disp = {}
        for c in df.columns:
            mr = df.groupby(gcol)[c].apply(lambda s: s.isna().mean())
            if mr.shape[0] >= 2: miss_disp[c] = float(mr.max() - mr.min())
        stats["missingness_disparity_top10"] = dict(sorted(miss_disp.items(), key=lambda kv: kv[1], reverse=True)[:10])
        if label_ok:
            y = df[cfg.label_col]
            if y.dropna().nunique() == 2:
                tmp = df.copy(); tmp["_y"], _ = pd.factorize(tmp[cfg.label_col])
                pos = tmp[tmp[cfg.label_col].notna()].groupby(gcol)["_y"].mean()
                if len(pos) >= 2:
                    stats["positive_rate_by_group"] = pos.to_dict()
                    stats["positive_rate_disparity"] = float(pos.max() - pos.min())
        per[gcol] = stats
    out["group_checks"] = per
    return out


def assess_security(df, cfg, dataset_bytes):
    out: Dict[str, Any] = {}
    text_cols = categorical_cols(df, exclude=[c for c in [cfg.label_col, cfg.split_col, cfg.time_col] if c])[:cfg.pii_max_text_cols]
    hits = {}
    for c in text_cols:
        s = df[c].dropna().astype("string")
        if s.empty: continue
        n = min(len(s), cfg.pii_max_rows)
        sample = s.sample(n=n, random_state=cfg.random_state) if len(s) > n else s
        col_hits = {nm: float(sample.str.contains(pat, regex=True).mean())
                    for nm, pat in PII_PATTERNS.items()
                    if float(sample.str.contains(pat, regex=True).mean()) >= cfg.thresholds.pii_hit_rate_threshold}
        if col_hits: hits[c] = col_hits
    out["confidentiality_pii_heuristics"] = {"note": "Heuristic. Validate with domain and legal review.",
                                              "threshold_hit_rate": float(cfg.thresholds.pii_hit_rate_threshold),
                                              "rows_sampled_per_col_max": int(cfg.pii_max_rows),
                                              "text_cols_scanned_max": int(cfg.pii_max_text_cols),
                                              "columns_with_hits": hits}
    out["integrity"] = {"sha256": sha256_bytes(dataset_bytes)}
    out["availability_asset_checks"] = {"byte_size": int(len(dataset_bytes))}
    if cfg.entity_cols:
        out["identifiers"] = {"entity_cols": cfg.entity_cols, "note": "Confirm these identifiers are allowed in the dataset."}
    return out


def assess_all(df, cfg, dataset_bytes):
    if cfg.mode == "Full Scan":
        cfg.pii_max_rows = max(cfg.pii_max_rows, 5000)
        cfg.pii_max_text_cols = max(cfg.pii_max_text_cols, 25)
        cfg.rare_max_cat_cols = max(cfg.rare_max_cat_cols, 100)
        cfg.drift_max_num_cols = max(cfg.drift_max_num_cols, 100)
    return {"quality":     assess_quality(df, cfg),
            "reliability": assess_reliability(df, cfg),
            "robustness":  assess_robustness(df, cfg),
            "fairness":    assess_fairness(df, cfg),
            "security":    assess_security(df, cfg, dataset_bytes),
            "notes": {"sklearn_available": SKLEARN_OK, "mode": cfg.mode}}


def verdict_panel(report, cfg):
    reasons = []
    q = report.get("quality", {}); r = report.get("reliability", {}); s = report.get("security", {})
    th = cfg.thresholds
    th_data = q.get("time_axis_health", {})
    parse_ok = th_data.get("time_parse", {}).get("parse_ok_rate", None)
    dup_ts   = th_data.get("duplicate_timestamps", {}).get("duplicate_rate", None)
    irr      = th_data.get("cadence_global", {}).get("irregularity_ratio", None)
    drift    = r.get("numeric_drift_ks_first_last", {}).get("top_10_ks", {})
    pii      = s.get("confidentiality_pii_heuristics", {}).get("columns_with_hits", {})
    leak     = q.get("split_leakage", {}).get("row_hash_cross_split_rate", None)
    if parse_ok is not None and float(parse_ok) < th.time_parse_ok_min:
        reasons.append("Time parsing success rate is below threshold.")
    if dup_ts is not None and float(dup_ts) > th.dup_timestamp_rate_max:
        reasons.append("Duplicate timestamps exceed the configured maximum.")
    if isinstance(irr, (int, float)) and float(irr) > th.cadence_irregularity_max:
        reasons.append("Sampling cadence is irregular.")
    if any(v is not None and float(v) > th.drift_ks_threshold for v in drift.values()):
        reasons.append("Potential drift — at least one KS statistic exceeds the threshold.")
    if pii: reasons.append("PII-like patterns detected — review confidentiality and legal basis.")
    if leak is not None and float(leak) > 0: reasons.append("Potential split leakage.")
    if reasons: return "Needs review", "warn", reasons
    return "Looks OK (evidence-based)", "ok", ["No major red flags under current checks."]


def build_recommendations(report, cfg):
    recs = []
    q = report.get("quality", {}); r = report.get("reliability", {}); s = report.get("security", {})
    th = cfg.thresholds
    miss = q.get("missingness", {}).get("overall_missing_rate", None)
    dup  = q.get("duplicates", {}).get("exact_duplicate_row_rate", None)
    th_data = q.get("time_axis_health", {})
    if miss is not None and float(miss) > 0.05:
        recs.append("Missingness > 5% — inspect top missing columns and decide: drop, impute, or recollect.")
    if dup is not None and float(dup) > 0.01:
        recs.append("Duplicate rows > 1% — deduplicate and verify duplicates don't cross splits.")
    parse_ok = th_data.get("time_parse", {}).get("parse_ok_rate", None)
    if parse_ok is not None and float(parse_ok) < th.time_parse_ok_min:
        recs.append("Time parsing is weak — standardize timestamp format and timezone, then rerun.")
    dup_ts = th_data.get("duplicate_timestamps", {}).get("duplicate_rate", None)
    if dup_ts is not None and float(dup_ts) > th.dup_timestamp_rate_max:
        recs.append("Duplicate timestamps detected — confirm aggregation level and deduplicate per entity+timestamp.")
    irr = th_data.get("cadence_global", {}).get("irregularity_ratio", None)
    if isinstance(irr, (int, float)) and float(irr) > th.cadence_irregularity_max:
        recs.append("Cadence irregular — consider resampling, interpolation, and explicit missing indicators.")
    drift = r.get("numeric_drift_ks_first_last", {}).get("top_10_ks", {})
    if drift and any(float(v) > th.drift_ks_threshold for v in drift.values() if v is not None):
        recs.append("Drift above threshold — compare first vs last slice and consider retraining or recalibration.")
    pii = s.get("confidentiality_pii_heuristics", {}).get("columns_with_hits", {})
    if pii: recs.append("PII-like patterns found — mask or remove flagged columns.")
    if not recs: recs.append("No major red flags. Keep the report and rerun after dataset updates.")
    return recs


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="dsa-card" style="padding:24px 28px 18px 28px; margin-bottom:16px;">
  <div style="display:flex; align-items:center; gap:10px;">
    <span style="font-size:1.8rem;">📈</span>
    <div>
      <h2 style="margin:0;">Time Series Dataset Analyzer</h2>
      <div class="muted">Upload a time-indexed CSV or Parquet, configure columns, run the scan.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📁 Upload")
    uploaded = st.file_uploader("CSV or Parquet", type=["csv", "parquet"])

if uploaded is None:
    st.info("⬆️  Upload a dataset from the sidebar to begin.")
    st.stop()

file_bytes = uploaded.getvalue()
file_name  = uploaded.name or "dataset"

try:
    if file_name.lower().endswith(".csv"):
        df = pd.read_csv(BytesIO(file_bytes)); detected = "CSV"
    else:
        df = pd.read_parquet(BytesIO(file_bytes)); detected = "Parquet"
except Exception as e:
    st.error(f"Failed to load file: {e}"); st.stop()

cols = df.columns.tolist()
if "ts_guesses" not in st.session_state:
    st.session_state["ts_guesses"] = guess_ts_columns(df)

with st.sidebar:
    st.header("⚙️ Run mode")
    mode = st.radio("Mode", ["Quick Scan", "Full Scan"], index=0)
    st.header("🎛 Thresholds")
    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
    th = PRESETS[preset_name]
    with st.expander("Show thresholds"):
        st.write({"drift_ks": th.drift_ks_threshold, "pii_hit_rate": th.pii_hit_rate_threshold,
                  "time_parse_ok_min": th.time_parse_ok_min, "dup_ts_max": th.dup_timestamp_rate_max,
                  "cadence_irr_max": th.cadence_irregularity_max})

    st.header("🗂 Columns")
    use_auto = st.toggle("Use suggested columns", value=True)
    guesses  = st.session_state["ts_guesses"]
    if use_auto and guesses.get("notes"):
        st.caption("Suggestions")
        for n in guesses["notes"]: st.write("• " + n)

    col_filter = st.text_input("Filter columns", value="")
    filtered = [c for c in cols if col_filter.lower() in c.lower()] if col_filter else cols

    def pick_one(label, auto_val):
        opts = ["(none)"] + filtered
        idx = opts.index(auto_val) if (use_auto and auto_val in filtered) else 0
        return (lambda x: None if x == "(none)" else x)(st.selectbox(label, opts, index=idx))

    time_col  = pick_one("Time column",  guesses.get("time_col"))
    label_col = pick_one("Label column", guesses.get("label_col"))
    split_col = pick_one("Split column", None)

    entity_cols = st.multiselect("Entity columns", filtered,
                                 default=[c for c in (guesses.get("entity_cols",[]) if use_auto else []) if c in filtered])
    id_cols     = st.multiselect("ID columns", filtered,
                                 default=[c for c in (guesses.get("id_cols",[]) if use_auto else []) if c in filtered])
    group_cols  = st.multiselect("Group columns (fairness)", filtered,
                                 default=[c for c in (guesses.get("group_cols",[]) if use_auto else []) if c in filtered])

    time_slice_mode = st.selectbox("Time slice mode", ["month","week","day","quarter"], index=0)

    with st.expander("Advanced"):
        annotator_cols = st.multiselect("Annotator label columns", filtered, default=[])
        random_state   = st.number_input("Random seed", min_value=0, max_value=10000, value=7, step=1)

    st.divider()
    run = st.button("🔬 Run analysis", type="primary", use_container_width=True)

if not run:
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1: kpi("Rows",    f"{df.shape[0]:,}", f"Format: {detected}")
    with c2: kpi("Columns", f"{df.shape[1]:,}", f"Numeric: {len(numeric_cols(df))} · Cat: {len(categorical_cols(df))}")
    with c3: kpi("SHA-256", sha256_bytes(file_bytes)[:14]+"…", "File fingerprint")
    with c4: kpi("Time column", time_col or "(not selected)", "For temporal analysis")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

cfg = AssessConfig(label_col=label_col, split_col=split_col, time_col=time_col,
                   entity_cols=entity_cols, group_cols=group_cols,
                   annotator_label_cols=annotator_cols, id_cols=id_cols,
                   time_slice_mode=time_slice_mode, random_state=int(random_state),
                   thresholds=th, mode=mode)

with st.spinner("Running checks…"):
    report = assess_all(df, cfg, file_bytes)

safe_report = to_json_safe(report)
verdict, vkind, reasons = verdict_panel(report, cfg)
recs = build_recommendations(report, cfg)
score, grade, score_components = compute_health_score(report, th.drift_ks_threshold)
dim_status = get_dimension_status(report, th.drift_ks_threshold)

cfg_dict = {"mode": mode, "preset": preset_name, "drift_ks_threshold": th.drift_ks_threshold,
            "pii_hit_rate_threshold": th.pii_hit_rate_threshold, "label_col": label_col,
            "split_col": split_col, "time_col": time_col, "id_cols": id_cols,
            "group_cols": group_cols, "random_state": int(random_state), "column_roles": {}}

pii_cols = report["security"]["confidentiality_pii_heuristics"]["columns_with_hits"]
miss_rate = report["quality"]["missingness"]["overall_missing_rate"]
dup_rate  = report["quality"]["duplicates"]["exact_duplicate_row_rate"]

st.markdown(f"""
<div class="verdict-card">
  <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;">
    <div>
      <div class="verdict-title">Verdict</div>
      <div class="verdict-text">{verdict}</div>
      <div class="muted" style="margin-top:4px;">
        Mode: <span class="code-pill">{mode}</span>
        &nbsp; Preset: <span class="code-pill">{preset_name}</span>
        &nbsp; Slice: <span class="code-pill">{time_slice_mode}</span>
      </div>
    </div>
    <div style="display:flex; gap:8px; flex-wrap:wrap;">
      {badge(verdict, vkind)}
      {badge(f"Score {score}/100", 'ok' if score>=80 else ('warn' if score>=60 else 'bad'))}
      {badge(f"Grade {grade}", 'ok' if grade in ('A','B') else ('warn' if grade=='C' else 'bad'))}
    </div>
  </div>
  <hr>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
    <div><div style="font-weight:700; font-size:0.85rem; margin-bottom:6px;">Findings</div>
         <ul style="margin:0; padding-left:1.1em; font-size:0.83rem; line-height:1.7;">{''.join(f'<li>{clip_text(r,200)}</li>' for r in reasons)}</ul></div>
    <div><div style="font-weight:700; font-size:0.85rem; margin-bottom:6px;">Recommended actions</div>
         <ul style="margin:0; padding-left:1.1em; font-size:0.83rem; line-height:1.7;">{''.join(f'<li>{clip_text(r,200)}</li>' for r in recs)}</ul></div>
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4, gap="large")
with c1: kpi("Rows",       f"{df.shape[0]:,}", f"Format: {detected}")
with c2: kpi("Columns",    f"{df.shape[1]:,}", "")
with c3: kpi("Missingness", f"{miss_rate:.2%}", "⚠ Above 5%" if miss_rate > 0.05 else "Within limits",
             color="#f59e0b" if miss_rate > 0.05 else "#22c55e")
with c4: kpi("PII flags",   str(len(pii_cols)), f"{len(pii_cols)} col(s) flagged" if pii_cols else "None detected",
             color="#ef4444" if pii_cols else "#22c55e")
st.markdown("<br>", unsafe_allow_html=True)

tab_ov, tab_q, tab_r, tab_rb, tab_f, tab_t, tab_sec, tab_exp = st.tabs(
    ["Overview","Quality","Reliability","Robustness","Fairness","Transparency","Security","Export"])

with tab_ov:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Health Score & Dimension Status")
    col_ring, col_dims = st.columns([1, 3], gap="large")
    with col_ring:
        st.markdown(health_ring_html(score, grade), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;opacity:0.5;margin-bottom:8px;">Score breakdown</div>', unsafe_allow_html=True)
        for dim, pts in score_components.items():
            weight = {"quality": 35, "security": 25, "reliability": 20, "robustness": 10, "fairness": 10}.get(dim, 10)
            st.markdown(progress_bar_html(dim.title(), pts / weight if weight else 0, reverse=False, fmt=".0%"), unsafe_allow_html=True)
    with col_dims:
        icons = {"Quality": "🔍", "Reliability": "📉", "Robustness": "🛡", "Fairness": "⚖️", "Security": "🔒"}
        for i, (dim, (status, detail)) in enumerate(dim_status.items()):
            if i % 3 == 0:
                if i > 0: st.markdown("</div>", unsafe_allow_html=True)
                st.markdown('<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">', unsafe_allow_html=True)
            st.markdown(check_status_card(dim, icons.get(dim, "•"), status, detail), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Time axis quick stats
    th_data = report["quality"].get("time_axis_health", {})
    if th_data and "note" not in th_data:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**Time axis health**")
        tm1, tm2, tm3, tm4 = st.columns(4)
        with tm1: st.metric("Parse OK rate", f"{th_data.get('time_parse',{}).get('parse_ok_rate', 0):.4f}")
        with tm2:
            dr = th_data.get("time_range", {})
            st.metric("Span (days)", f"{dr.get('span_days', 0):.0f}" if dr else "n/a")
        with tm3: st.metric("Dup timestamps", f"{th_data.get('duplicate_timestamps',{}).get('duplicate_rate', 0):.4f}")
        with tm4:
            irr = th_data.get("cadence_global", {}).get("irregularity_ratio", None)
            st.metric("Cadence irregularity", f"{float(irr):.3f}" if isinstance(irr, (int,float)) else "n/a")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_q:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Quality signals")
    q = report["quality"]
    miss_top = q["missingness"]["top_10_columns_missing_rate"]
    if miss_top:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**Missingness — top 10 columns**")
            html_bars = "".join(progress_bar_html(col, v, max_val=1.0, reverse=True) for col, v in sorted(miss_top.items(), key=lambda x: x[1], reverse=True))
            st.markdown(html_bars, unsafe_allow_html=True)
        with col2:
            st.markdown("**Outlier rate — top 10 columns**")
            out_top = q["outliers_iqr"]["top_10_outlier_rate"]
            if out_top:
                html_bars2 = "".join(progress_bar_html(col, v, max_val=1.0, reverse=True) for col, v in sorted(out_top.items(), key=lambda x: x[1], reverse=True))
                st.markdown(html_bars2, unsafe_allow_html=True)
            else: st.info("No numeric columns to evaluate.")

    # Time axis health details
    th_data = q.get("time_axis_health", {})
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Time-axis health**")
    if "note" in th_data:
        st.info(th_data["note"])
    else:
        cadence = th_data.get("cadence_global", {})
        gaps    = th_data.get("gaps_global", {})
        entities= th_data.get("entities", {})
        if "note" not in cadence:
            ca1, ca2, ca3 = st.columns(3)
            with ca1: st.metric("Median step (s)", f"{cadence.get('median_step_seconds',0):.1f}")
            with ca2: st.metric("P10 step (s)",    f"{cadence.get('p10_seconds',0):.1f}")
            with ca3: st.metric("P90 step (s)",    f"{cadence.get('p90_seconds',0):.1f}")
            irr = cadence.get("irregularity_ratio", None)
            if irr is not None:
                irr_status = "bad" if float(irr) > th.cadence_irregularity_max else "ok"
                st.markdown(f"Irregularity ratio: {badge(f'{float(irr):.3f}', irr_status)}", unsafe_allow_html=True)
        if gaps:
            st.metric("Largest gap (s)", f"{gaps.get('largest_gap_seconds', 0):.0f}")
        if entities:
            st.metric("Entities", str(entities.get("num_entities", 0)))
            with st.expander("Top entities by observation count"):
                ec = entities.get("top_entities_counts", {})
                ec_df = pd.DataFrame(list(ec.items()), columns=["Entity", "Count"]).sort_values("Count", ascending=False)
                st.dataframe(ec_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_r:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Reliability — Stability & Drift")
    r = report["reliability"]
    if "note" in r:
        st.warning(r["note"])
    else:
        if "missing_rate_by_time_slice" in r:
            srs = pd.Series(r["missing_rate_by_time_slice"]).sort_index()
            st.markdown("**Missingness by time slice**")
            st.line_chart(srs)
        if "numeric_drift_ks_first_last" in r:
            ks_info = r["numeric_drift_ks_first_last"]
            ks = ks_info.get("top_10_ks", {})
            if "note" in ks_info:
                st.info(ks_info["note"])
            elif ks:
                st.markdown(f"**Numeric drift (KS): `{ks_info.get('first_slice')}` vs `{ks_info.get('last_slice')}`**")
                html_bars = "".join(progress_bar_html(col, v, max_val=max(v*1.5, th.drift_ks_threshold*1.5), reverse=True)
                                    for col, v in sorted(ks.items(), key=lambda x: x[1], reverse=True) if v is not None)
                st.markdown(html_bars, unsafe_allow_html=True)
                ks_df = pd.DataFrame([(c, v, v > th.drift_ks_threshold) for c, v in ks.items() if v is not None],
                                     columns=["Column", "KS Statistic", "Above threshold"]).sort_values("KS Statistic", ascending=False)
                st.dataframe(ks_df, use_container_width=True, hide_index=True)
                st.bar_chart(ks_df.set_index("Column")["KS Statistic"])
    with st.expander("Schema consistency"):
        sc = r.get("schema_consistency", {})
        if sc.get("constant_columns"):
            st.warning(f"Constant columns: {', '.join(sc['constant_columns'])}")
        else:
            st.success("✓ No constant columns.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_rb:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Robustness")
    rb = report["robustness"]
    ram = rb.get("row_anomaly_score_mad", {})
    if ram:
        st.markdown("**Row anomaly score (MAD-based)**")
        r1, r2, r3, r4 = st.columns(4)
        with r1: st.metric("Mean", f"{ram.get('mean',0):.3f}")
        with r2: st.metric("P95",  f"{ram.get('p95',0):.3f}")
        with r3: st.metric("P99",  f"{ram.get('p99',0):.3f}")
        with r4: st.metric("Max",  f"{ram.get('max',0):.3f}")
    stressors = rb.get("time_series_stressors", {})
    if stressors:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**Time-series stressors**")
        with st.expander("Cadence & gaps detail"):
            st.json(to_json_safe(stressors))
    st.markdown('</div>', unsafe_allow_html=True)

with tab_f:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Fairness proxies")
    f = report["fairness"]
    if "note" in f:
        st.warning(f["note"])
    else:
        for gcol, stats_ in f["group_checks"].items():
            st.markdown(f"#### Group column: `{gcol}`")
            rep = stats_.get("representation_share_top10", {})
            if rep:
                rep_df = pd.DataFrame(list(rep.items()), columns=["Group","Share"]).sort_values("Share", ascending=False)
                st.bar_chart(rep_df.set_index("Group")["Share"])
            if "positive_rate_by_group" in stats_:
                pr = pd.Series(stats_["positive_rate_by_group"]).sort_index()
                st.markdown("**Positive rate by group**"); st.bar_chart(pr)
                disp = stats_.get("positive_rate_disparity", 0)
                st.markdown(f"Disparity: {badge(f'{disp:.4f}', 'bad' if disp>0.2 else 'warn' if disp>0.1 else 'ok')}", unsafe_allow_html=True)
            st.markdown("---")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_t:
    render_transparency_tab(df, file_name, file_bytes, cfg_dict, report)

with tab_sec:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Security — CIA prerequisites")
    s = report["security"]
    sha = s["integrity"]["sha256"]
    st.markdown("**File integrity**")
    st.markdown(f'<div class="mono" style="font-size:0.8rem;word-break:break-all;padding:10px;border:1px solid var(--c-border);border-radius:8px;">{sha}</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    pii_cols_found = s.get("confidentiality_pii_heuristics", {}).get("columns_with_hits", {})
    if pii_cols_found:
        rows = [{"Column": col, "Pattern": k, "Hit rate": v} for col, hits in pii_cols_found.items() for k, v in hits.items()]
        st.dataframe(pd.DataFrame(rows).sort_values("Hit rate", ascending=False), use_container_width=True, hide_index=True,
                     column_config={"Hit rate": st.column_config.ProgressColumn(min_value=0, max_value=1)})
        st.error(f"⚠️ {len(pii_cols_found)} column(s) with PII-like patterns.")
    else:
        st.success("✓ No PII-like patterns flagged.")
    if "identifiers" in s:
        st.info(f"Entity columns: {', '.join(s['identifiers']['entity_cols'])} — confirm these are allowed in reports.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_exp:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Export")
    col_e1, col_e2, col_e3 = st.columns(3, gap="large")
    with col_e1:
        st.markdown("**JSON report**")
        st.download_button("⬇ Download JSON", data=json.dumps(safe_report, indent=2).encode(),
                           file_name="ts_dataset_report.json", mime="application/json", use_container_width=True)
    with col_e2:
        st.markdown("**Markdown summary**")
        md = f"# Time Series Dataset Report\n\n- Mode: {mode}\n- Preset: {preset_name}\n- Score: {score}/100 Grade {grade}\n- Verdict: {verdict}\n\n## Findings\n" + "\n".join(f"- {r}" for r in reasons)
        st.download_button("⬇ Download Markdown", data=md.encode(), file_name="ts_report.md", mime="text/markdown", use_container_width=True)
    with col_e3:
        st.markdown("**HTML report**")
        html_content = build_html_report(df=df, report=report, cfg_dict=cfg_dict, file_name=file_name,
                                          file_bytes=file_bytes, verdict=verdict, reasons=reasons,
                                          recs=recs, score=score, grade=grade)
        st.download_button("⬇ Download HTML", data=html_content.encode(), file_name="ts_report.html", mime="text/html", use_container_width=True)
    with st.expander("Raw JSON (preview)"):
        st.json(safe_report)
    st.markdown('</div>', unsafe_allow_html=True)
