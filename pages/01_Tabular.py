"""Tabular Dataset Safety Analyzer — upgraded version."""
from __future__ import annotations

import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import (
    SHARED_CSS, to_json_safe, sha256_bytes, clip_text,
    badge, kpi, health_ring_html, progress_bar_html, check_status_card,
    compute_health_score, get_dimension_status, render_transparency_tab,
    build_html_report, PII_PATTERNS, infer_column_types,
    numeric_cols, categorical_cols, approx_iqr_outlier_rate, ks_statistic,
    to_datetime_if_possible
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

st.set_page_config(page_title="Tabular Analyzer", page_icon="📊", layout="wide")
st.markdown(SHARED_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Thresholds:
    drift_ks_threshold: float
    pii_hit_rate_threshold: float

PRESETS: Dict[str, Thresholds] = {
    "Balanced (recommended)": Thresholds(drift_ks_threshold=0.30, pii_hit_rate_threshold=0.01),
    "Strict":                 Thresholds(drift_ks_threshold=0.20, pii_hit_rate_threshold=0.005),
    "Lenient":                Thresholds(drift_ks_threshold=0.40, pii_hit_rate_threshold=0.02),
}

# ──────────────────────────────────────────────────────────────────────────────
# Metric documentation registry
# ──────────────────────────────────────────────────────────────────────────────

THRESHOLD_DOCS = {
    "drift_ks_threshold": {
        "label": "Drift KS threshold",
        "description": "Reference cutoff for flagging strong numeric distribution shift between slices.",
        "balanced": 0.30,
        "strict": 0.20,
        "lenient": 0.40,
        "interpretation": "Higher values mean larger first-vs-last slice change.",
    },
    "pii_hit_rate_threshold": {
        "label": "PII hit-rate threshold",
        "description": "Minimum regex hit-rate needed before a text column is flagged by heuristic confidentiality checks.",
        "balanced": 0.01,
        "strict": 0.005,
        "lenient": 0.02,
        "interpretation": "Lower values make the scan more sensitive.",
    },
}

METRIC_DOCS = {
    "missingness.overall_missing_rate": {
        "label": "Overall missingness",
        "dimension": "Quality",
        "what_it_is": "Average missing-value rate across all cells.",
        "why_it_matters": "High missingness weakens validity, can bias training, and may break downstream steps.",
        "how_computed": "Mean of df.isna() over the full table.",
        "interpretation": "Lower is better.",
        "related_thresholds": [],
    },
    "missingness.top_10_columns_missing_rate": {
        "label": "Column missingness profile",
        "dimension": "Quality",
        "what_it_is": "Worst columns ranked by missing-value rate.",
        "why_it_matters": "Pinpoints which fields need repair or exclusion.",
        "how_computed": "Column-wise df.isna().mean(), top 10 only.",
        "interpretation": "Lower is better.",
        "related_thresholds": [],
    },
    "duplicates.exact_duplicate_row_rate": {
        "label": "Exact duplicate row rate",
        "dimension": "Quality",
        "what_it_is": "Fraction of rows duplicated exactly.",
        "why_it_matters": "Duplicates can inflate support and hide leakage.",
        "how_computed": "df.duplicated().mean().",
        "interpretation": "Lower is better.",
        "related_thresholds": [],
    },
    "duplicates.duplicate_id_rate": {
        "label": "Duplicate ID rate",
        "dimension": "Quality",
        "what_it_is": "Duplicate rate when restricted to selected ID columns.",
        "why_it_matters": "Repeated identifiers often signal merge errors or broken primary keys.",
        "how_computed": "df.duplicated(subset=id_cols).mean().",
        "interpretation": "Lower is better.",
        "related_thresholds": [],
    },
    "outliers_iqr.top_10_outlier_rate": {
        "label": "IQR outlier rate",
        "dimension": "Quality",
        "what_it_is": "Share of values outside the 1.5 IQR rule for each numeric column.",
        "why_it_matters": "Finds suspicious tails and potential data-entry or sensor issues.",
        "how_computed": "Column-wise approximate IQR outlier rate, top 10 columns shown.",
        "interpretation": "Lower is usually better, but true rare events can be valid.",
        "related_thresholds": [],
    },
    "label_stats.label_missing_rate": {
        "label": "Label missing rate",
        "dimension": "Quality",
        "what_it_is": "Fraction of rows without a label in the selected target column.",
        "why_it_matters": "Missing labels reduce usable training signal.",
        "how_computed": "y.isna().mean().",
        "interpretation": "Lower is better.",
        "related_thresholds": [],
    },
    "label_stats.label_cardinality": {
        "label": "Label cardinality",
        "dimension": "Quality",
        "what_it_is": "Number of distinct labels.",
        "why_it_matters": "Affects class balance, difficulty, and model design.",
        "how_computed": "y.nunique(dropna=True).",
        "interpretation": "Descriptive, not inherently good or bad.",
        "related_thresholds": [],
    },
    "label_agreement.exact_agreement_rate": {
        "label": "Exact annotator agreement",
        "dimension": "Quality",
        "what_it_is": "Share of rows where selected annotator columns agree exactly.",
        "why_it_matters": "Low agreement may indicate ambiguous labeling or annotation drift.",
        "how_computed": "Row-wise unique non-null labels <= 1.",
        "interpretation": "Higher is better.",
        "related_thresholds": [],
    },
    "split_leakage.row_hash_cross_split_rate": {
        "label": "Cross-split leakage rate",
        "dimension": "Quality",
        "what_it_is": "Fraction of unique row hashes that appear across multiple splits.",
        "why_it_matters": "Leakage inflates offline evaluation and weakens trust in performance estimates.",
        "how_computed": "Hash rows excluding split column, then check if one hash appears in more than one split.",
        "interpretation": "Lower is better.",
        "related_thresholds": [],
    },
    "reliability.missing_rate_by_slice": {
        "label": "Missingness by slice",
        "dimension": "Reliability",
        "what_it_is": "Average missingness per split or time slice.",
        "why_it_matters": "Shows whether data quality changes over time or across partitions.",
        "how_computed": "Mean missing-value rate inside each slice.",
        "interpretation": "Stable profiles are preferred.",
        "related_thresholds": [],
    },
    "reliability.numeric_drift_ks_first_last": {
        "label": "Numeric drift KS",
        "dimension": "Reliability",
        "what_it_is": "Kolmogorov-Smirnov statistic comparing first and last slice for numeric columns.",
        "why_it_matters": "Flags distribution shift that may break old models or stale assumptions.",
        "how_computed": "First-vs-last slice KS statistic per numeric column.",
        "interpretation": "Lower is better.",
        "related_thresholds": ["drift_ks_threshold"],
    },
    "reliability.schema_consistency": {
        "label": "Schema consistency",
        "dimension": "Reliability",
        "what_it_is": "Schema snapshot with inferred dtypes and constant columns.",
        "why_it_matters": "Unexpected schema changes often precede pipeline failures.",
        "how_computed": "Counts rows, columns, inferred types, and no-variance columns.",
        "interpretation": "Few constant or malformed columns is better.",
        "related_thresholds": [],
    },
    "robustness.rare_category_label_concentration": {
        "label": "Rare-category label concentration",
        "dimension": "Robustness",
        "what_it_is": "Rare categorical values that map almost entirely to one label.",
        "why_it_matters": "Can reveal hidden shortcuts, leakage, or brittle spurious cues.",
        "how_computed": "Inspect rare category values and label concentration within each.",
        "interpretation": "Fewer findings are better.",
        "related_thresholds": [],
    },
    "robustness.row_anomaly_score_mad": {
        "label": "MAD anomaly score",
        "dimension": "Robustness",
        "what_it_is": "Row-level anomaly score from median absolute deviation across numeric features.",
        "why_it_matters": "Helps find records far from the bulk of the dataset.",
        "how_computed": "Average scaled absolute distance from per-column medians.",
        "interpretation": "Lower is safer; very high tails deserve review.",
        "related_thresholds": [],
    },
    "robustness.label_predictability_auc": {
        "label": "Label predictability AUC",
        "dimension": "Robustness",
        "what_it_is": "How well numeric features alone predict a binary label.",
        "why_it_matters": "Very high AUC can signal leakage or trivially easy shortcuts.",
        "how_computed": "Logistic regression with a train/test split and ROC AUC on held-out data.",
        "interpretation": "Mid-range values are often healthier than near-perfect values.",
        "related_thresholds": [],
    },
    "fairness.representation_share_top10": {
        "label": "Group representation",
        "dimension": "Fairness",
        "what_it_is": "Observed share of each group.",
        "why_it_matters": "Severe imbalance can produce uneven quality across groups.",
        "how_computed": "Value-count share by selected group column.",
        "interpretation": "More balanced representation is usually safer.",
        "related_thresholds": [],
    },
    "fairness.positive_rate_by_group": {
        "label": "Positive rate by group",
        "dimension": "Fairness",
        "what_it_is": "Observed positive-label rate in each group for binary tasks.",
        "why_it_matters": "Large gaps can indicate label imbalance or structural disparities.",
        "how_computed": "Mean of factorized binary label within each group.",
        "interpretation": "Smaller disparities are preferred.",
        "related_thresholds": [],
    },
    "fairness.positive_rate_disparity": {
        "label": "Positive rate disparity",
        "dimension": "Fairness",
        "what_it_is": "Difference between the maximum and minimum group positive rates.",
        "why_it_matters": "Summarizes the spread in observed label prevalence.",
        "how_computed": "max(group positive rate) - min(group positive rate).",
        "interpretation": "Lower is better.",
        "related_thresholds": [],
    },
    "fairness.missingness_disparity_top10": {
        "label": "Missingness disparity",
        "dimension": "Fairness",
        "what_it_is": "Largest between-group missingness gaps per column.",
        "why_it_matters": "Some groups may be systematically less complete than others.",
        "how_computed": "Per-column max missing rate minus min missing rate across groups.",
        "interpretation": "Lower is better.",
        "related_thresholds": [],
    },
    "security.confidentiality_pii_heuristics": {
        "label": "PII heuristic scan",
        "dimension": "Security",
        "what_it_is": "Regex-based scan for email, phone-like, IP, and card-like patterns.",
        "why_it_matters": "Flags confidentiality risk before sharing or deploying the dataset.",
        "how_computed": "Sample text-like columns and compute regex hit-rates.",
        "interpretation": "Fewer flagged columns are better. Heuristic only.",
        "related_thresholds": ["pii_hit_rate_threshold"],
    },
    "security.integrity.sha256": {
        "label": "File SHA-256",
        "dimension": "Security",
        "what_it_is": "Cryptographic hash of the uploaded file.",
        "why_it_matters": "Supports traceability and integrity checks.",
        "how_computed": "sha256_bytes(dataset_bytes).",
        "interpretation": "Descriptive fingerprint.",
        "related_thresholds": [],
    },
    "security.availability_asset_checks.byte_size": {
        "label": "File byte size",
        "dimension": "Security",
        "what_it_is": "Observed upload size.",
        "why_it_matters": "Useful for reproducibility and basic asset inventory.",
        "how_computed": "len(dataset_bytes).",
        "interpretation": "Descriptive inventory field.",
        "related_thresholds": [],
    },
}

def _threshold_table_rows():
    return [
        {
            "Threshold": spec["label"],
            "Key": key,
            "Balanced": spec["balanced"],
            "Strict": spec["strict"],
            "Lenient": spec["lenient"],
            "Meaning": spec["description"],
        }
        for key, spec in THRESHOLD_DOCS.items()
    ]

def _metric_doc_rows():
    rows = []
    for key, spec in METRIC_DOCS.items():
        rows.append({
            "Metric key": key,
            "Label": spec["label"],
            "Dimension": spec["dimension"],
            "What it is": spec["what_it_is"],
            "Why it matters": spec["why_it_matters"],
            "How computed": spec["how_computed"],
            "Interpretation": spec["interpretation"],
            "Thresholds": ", ".join(spec.get("related_thresholds", [])),
        })
    return rows

def render_metric_help_block(metric_keys: List[str], title: str = "What these checks mean") -> None:
    rows = []
    for key in metric_keys:
        spec = METRIC_DOCS.get(key)
        if spec:
            rows.append({
                "Metric": spec["label"],
                "Purpose": spec["why_it_matters"],
                "Method": spec["how_computed"],
                "Interpretation": spec["interpretation"],
                "Thresholds": ", ".join(spec.get("related_thresholds", [])) or "None",
            })
    if rows:
        with st.expander(title):
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def render_research_grade_transparency_tab(df, file_name, file_bytes, cfg_dict, report, preset_name, mode):
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Transparency and methodology")
    st.caption("This view exposes the active configuration, metric registry, and threshold semantics used by the analyzer.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Run configuration**")
        cfg_rows = [{"Parameter": k, "Value": json.dumps(to_json_safe(v), ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)} for k, v in cfg_dict.items()]
        cfg_rows += [
            {"Parameter": "preset_name", "Value": str(preset_name)},
            {"Parameter": "mode", "Value": str(mode)},
            {"Parameter": "file_name", "Value": str(file_name)},
            {"Parameter": "sha256", "Value": str(sha256_bytes(file_bytes))},
        ]
        st.dataframe(pd.DataFrame(cfg_rows), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Threshold reference table**")
        st.dataframe(pd.DataFrame(_threshold_table_rows()), use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Metric registry**")
    st.dataframe(pd.DataFrame(_metric_doc_rows()), use_container_width=True, hide_index=True)

    registry_payload = {
        "analyzer": "tabular",
        "threshold_docs": THRESHOLD_DOCS,
        "metric_docs": METRIC_DOCS,
    }
    st.download_button(
        "⬇ Download metric registry JSON",
        data=json.dumps(to_json_safe(registry_payload), indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="tabular_metric_registry.json",
        mime="application/json",
        use_container_width=False,
    )
    st.markdown('</div>', unsafe_allow_html=True)

@dataclass
class AssessConfig:
    label_col: Optional[str]
    split_col: Optional[str]
    time_col: Optional[str]
    group_cols: List[str]
    annotator_label_cols: List[str]
    id_cols: List[str]
    random_state: int = 7
    max_categories_for_stats: int = 50
    mode: str = "Quick Scan"
    thresholds: Thresholds = field(default_factory=lambda: PRESETS["Balanced (recommended)"])
    pii_max_rows: int = 2000
    pii_max_text_cols: int = 10
    rare_max_cat_cols: int = 50
    drift_max_num_cols: int = 50

# ──────────────────────────────────────────────────────────────────────────────
# Auto-guess
# ──────────────────────────────────────────────────────────────────────────────

import re

def _name_score(name: str, patterns: List[str]) -> float:
    n = name.lower().strip()
    return sum(1.0 for p in patterns if re.search(p, n))

def guess_columns(df: pd.DataFrame) -> Dict[str, Any]:
    cols = df.columns.tolist()
    nrows = max(1, len(df))
    nunique = {c: int(df[c].nunique(dropna=True)) for c in cols}
    uniq_ratio = {c: float(nunique[c] / nrows) for c in cols}

    dt_success: Dict[str, float] = {}
    for c in cols[:250]:
        if pd.api.types.is_datetime64_any_dtype(df[c].dtype):
            dt_success[c] = 1.0
        elif pd.api.types.is_numeric_dtype(df[c].dtype):
            dt_success[c] = 0.0
        else:
            parsed = to_datetime_if_possible(df[c])
            dt_success[c] = float(parsed.notna().mean()) if pd.api.types.is_datetime64_any_dtype(parsed.dtype) else 0.0

    label_pats = [r"\blabel\b", r"\btarget\b", r"\boutcome\b", r"\bclass\b", r"\bgt\b", r"\by\b"]
    split_pats = [r"\bsplit\b", r"\bfold\b", r"\bset\b", r"\bpartition\b"]
    time_pats  = [r"\btime\b", r"\bdate\b", r"\btimestamp\b", r"\bcreated\b", r"\bupdated\b"]
    id_pats    = [r"\bid\b", r"\buuid\b", r"\bguid\b", r"\buser[_\s-]?id\b", r"\bserial\b"]
    grp_pats   = [r"\bgender\b", r"\bsex\b", r"\bage\b", r"\bregion\b", r"\bcountry\b", r"\bgroup\b"]

    def rank_label(c):
        s = _name_score(c, label_pats) * 3
        if 2 <= nunique[c] <= min(50, int(0.01 * nrows) + 2): s += 2
        if uniq_ratio[c] < 0.2: s += 1
        if uniq_ratio[c] > 0.9: s -= 2
        if dt_success.get(c, 0) > 0.8: s -= 2
        return s

    def rank_split(c):
        s = _name_score(c, split_pats) * 3
        if nunique[c] <= 20: s += 2
        try:
            vals = " ".join(df[c].dropna().astype("string").str.lower().value_counts().head(12).index)
            if any(x in vals for x in ["train","test","val","valid","dev"]): s += 2
        except: pass
        if uniq_ratio[c] > 0.5: s -= 2
        return s

    def rank_time(c):
        s = _name_score(c, time_pats) * 3 + dt_success.get(c, 0) * 2
        if dt_success.get(c, 0) < 0.3: s -= 1.5
        return s

    def rank_id(c):
        s = _name_score(c, id_pats) * 3
        if uniq_ratio[c] > 0.95: s += 2.5
        if dt_success.get(c, 0) > 0.8: s -= 2
        return s

    def rank_grp(c):
        s = _name_score(c, grp_pats) * 2
        if 2 <= nunique[c] <= 50: s += 2
        if uniq_ratio[c] > 0.5: s -= 1
        if dt_success.get(c, 0) > 0.8: s -= 2
        return s

    sorted_label = sorted(cols, key=rank_label, reverse=True)
    sorted_split = sorted(cols, key=rank_split, reverse=True)
    sorted_time  = sorted(cols, key=rank_time, reverse=True)
    sorted_id    = sorted(cols, key=rank_id, reverse=True)
    sorted_grp   = sorted(cols, key=rank_grp, reverse=True)

    label = sorted_label[0] if sorted_label and rank_label(sorted_label[0]) >= 2 else None
    split = sorted_split[0] if sorted_split and rank_split(sorted_split[0]) >= 2 else None
    time  = sorted_time[0]  if sorted_time  and rank_time(sorted_time[0]) >= 2  else None

    ids, groups = [], []
    for c in sorted_id:
        if c in {label, split, time}: continue
        if rank_id(c) >= 3.5: ids.append(c)
        if len(ids) >= 3: break
    for c in sorted_grp:
        if c in {label, split, time} or c in set(ids): continue
        if rank_grp(c) >= 2.5: groups.append(c)
        if len(groups) >= 3: break

    notes = (
        ([f"Guessed label: {label}"] if label else []) +
        ([f"Guessed split: {split}"] if split else []) +
        ([f"Guessed time: {time}"] if time else []) +
        ([f"Guessed IDs: {', '.join(ids)}"] if ids else []) +
        ([f"Guessed groups: {', '.join(groups)}"] if groups else [])
    )
    return {"label": label, "split": split, "time": time, "ids": ids, "groups": groups, "notes": notes}

def detect_task_type(df: pd.DataFrame, label_col: Optional[str]) -> str:
    if not label_col or label_col not in df.columns: return "No label selected"
    y = df[label_col]
    if pd.api.types.is_numeric_dtype(y.dtype):
        nu = int(y.nunique(dropna=True))
        if nu > 20 and nu > 0.05 * max(1, len(y)): return "Regression"
    nu = int(y.nunique(dropna=True))
    if nu == 2: return "Binary classification"
    if 2 < nu <= 50: return "Multi-class classification"
    return f"Label selected (card={nu})"

# ──────────────────────────────────────────────────────────────────────────────
# Assessment functions
# ──────────────────────────────────────────────────────────────────────────────

def assess_quality(df, cfg):
    out = {}
    miss = df.isna().mean().sort_values(ascending=False)
    out["missingness"] = {
        "overall_missing_rate": float(df.isna().mean().mean()),
        "top_10_columns_missing_rate": miss.head(10).to_dict(),
    }
    out["duplicates"] = {"exact_duplicate_row_rate": float(df.duplicated().mean())}
    if cfg.id_cols:
        id_cols_ = [c for c in cfg.id_cols if c in df.columns]
        if id_cols_:
            out["duplicates"]["duplicate_id_rate"] = float(df.duplicated(subset=id_cols_).mean())

    num = numeric_cols(df, exclude=[c for c in [cfg.label_col, cfg.split_col, cfg.time_col] if c])
    outlier_rates = {}
    for c in num:
        r = approx_iqr_outlier_rate(df[c])
        if r is not None: outlier_rates[c] = r
    out["outliers_iqr"] = {
        "columns_evaluated": len(outlier_rates),
        "top_10_outlier_rate": dict(sorted(outlier_rates.items(), key=lambda kv: kv[1], reverse=True)[:10]),
    }

    if cfg.label_col and cfg.label_col in df.columns:
        y = df[cfg.label_col]
        vc = y.value_counts(dropna=True)
        out["label_stats"] = {
            "label_missing_rate": float(y.isna().mean()),
            "label_cardinality": int(y.nunique(dropna=True)),
            "top_classes_share": (vc / vc.sum()).head(10).to_dict() if 2 <= len(vc) <= cfg.max_categories_for_stats else None,
        }

    if cfg.annotator_label_cols:
        acols = [c for c in cfg.annotator_label_cols if c in df.columns]
        if len(acols) >= 2:
            lab = df[acols]
            valid = ~lab.isna().all(axis=1)
            agree = (lab.nunique(axis=1, dropna=True) <= 1) & valid
            out["label_agreement"] = {
                "annotator_cols": acols,
                "rows_with_any_label": int(valid.sum()),
                "exact_agreement_rate": float(agree[valid].mean()) if valid.sum() else None,
                "disagreement_rate": float((~agree & valid).mean()) if valid.sum() else None,
            }

    if cfg.split_col and cfg.split_col in df.columns:
        split = df[cfg.split_col].astype("string")
        row_hash = pd.util.hash_pandas_object(df[[c for c in df.columns if c != cfg.split_col]], index=False).astype("uint64")
        tmp = pd.DataFrame({"split": split, "row_hash": row_hash})
        distinct = tmp.groupby("row_hash")["split"].nunique()
        out["split_leakage"] = {
            "row_hash_cross_split_rate": float((distinct > 1).mean()) if len(distinct) else None,
            "num_unique_rows_hashed": int(distinct.shape[0]),
        }

    return out


def assess_reliability(df, cfg):
    out = {}
    exclude = [c for c in [cfg.label_col, cfg.split_col, cfg.time_col] if c]
    num_all = numeric_cols(df, exclude=exclude)
    slices, slice_type = None, None

    if cfg.time_col and cfg.time_col in df.columns:
        slice_type = "time"
        t = to_datetime_if_possible(df[cfg.time_col])
        if pd.api.types.is_datetime64_any_dtype(t.dtype):
            slices = t.dt.to_period("M").astype("string")
        else:
            v = pd.to_numeric(df[cfg.time_col], errors="coerce")
            try: slices = pd.qcut(v, q=4, duplicates="drop").astype("string")
            except: slices = None
    elif cfg.split_col and cfg.split_col in df.columns:
        slice_type = "split"
        slices = df[cfg.split_col].astype("string")

    out["slice_type"] = slice_type
    if slices is None:
        out["note"] = "Select a time or split column to compute stability and drift."
        out["schema_consistency"] = {
            "num_rows": int(len(df)), "num_cols": int(df.shape[1]),
            "dtypes": infer_column_types(df),
            "constant_columns": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
        }
        return out

    miss_by_slice = {str(sv): float(g.isna().mean().mean()) for sv, g in df.groupby(slices, dropna=False)}
    out["missing_rate_by_slice"] = miss_by_slice

    uniq = pd.Series(slices).dropna().unique().tolist()
    num = num_all
    if cfg.drift_max_num_cols and len(num_all) > cfg.drift_max_num_cols:
        variances = [(c, float(pd.to_numeric(df[c], errors="coerce").var(skipna=True))) for c in num_all]
        num = [c for c, _ in sorted(variances, key=lambda t: t[1], reverse=True)[:cfg.drift_max_num_cols]]

    drift = {}
    if len(uniq) >= 2 and num:
        uniq_sorted = sorted(map(str, uniq))
        s1, s2 = uniq_sorted[0], uniq_sorted[-1]
        s_ser = pd.Series(slices).astype("string")
        g1, g2 = df[s_ser == s1], df[s_ser == s2]
        for c in num:
            d = ks_statistic(pd.to_numeric(g1[c], errors="coerce").to_numpy(),
                             pd.to_numeric(g2[c], errors="coerce").to_numpy())
            if d is not None: drift[c] = d
        out["numeric_drift_ks_first_last"] = {
            "first_slice": s1, "last_slice": s2,
            "first_slice_rows": int(len(g1)), "last_slice_rows": int(len(g2)),
            "num_cols_evaluated": int(len(num)),
            "top_10_ks": dict(sorted(drift.items(), key=lambda kv: kv[1], reverse=True)[:10]),
        }

    out["schema_consistency"] = {
        "num_rows": int(len(df)), "num_cols": int(df.shape[1]),
        "dtypes": infer_column_types(df),
        "constant_columns": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
    }
    return out


def assess_robustness(df, cfg):
    out = {"sklearn_available": SKLEARN_OK}
    exclude = [c for c in [cfg.label_col, cfg.split_col, cfg.time_col] if c]

    if cfg.label_col and cfg.label_col in df.columns:
        y = df[cfg.label_col]
        cat_all = categorical_cols(df, exclude=exclude)
        cat = cat_all[:cfg.rare_max_cat_cols]
        suspicious = []
        for c in cat:
            vc = df[c].value_counts(dropna=True)
            rare = vc[vc <= max(5, int(0.001 * len(df)))].index.tolist()
            for v in rare[:200]:
                mask = df[c] == v
                if mask.sum() < 5: continue
                dist = y[mask].value_counts(normalize=True, dropna=True)
                if len(dist) >= 1 and float(dist.iloc[0]) >= 0.95:
                    suspicious.append({"column": c, "value": str(v), "count": int(mask.sum()),
                                       "top_label": str(dist.index[0]), "top_label_share": float(dist.iloc[0])})
        out["rare_category_label_concentration"] = {
            "num_findings": len(suspicious),
            "top_findings": sorted(suspicious, key=lambda d: (-d["top_label_share"], -d["count"]))[:20],
            "columns_scanned": int(len(cat)),
        }

    num = numeric_cols(df, exclude=exclude)
    if num:
        X = df[num].apply(pd.to_numeric, errors="coerce")
        med = X.median(axis=0, skipna=True)
        mad = (X - med).abs().median(axis=0, skipna=True).replace(0, np.nan)
        z = (X - med).abs().divide(mad)
        row_score = z.mean(axis=1, skipna=True)
        out["row_anomaly_score_mad"] = {
            "mean": float(row_score.mean(skipna=True)),
            "p95": float(row_score.quantile(0.95)),
            "p99": float(row_score.quantile(0.99)),
            "max": float(row_score.max(skipna=True)),
            "top_20_row_indices": row_score.sort_values(ascending=False).head(20).index.tolist(),
        }

    if SKLEARN_OK and cfg.label_col and cfg.label_col in df.columns and num:
        y = df[cfg.label_col]
        if y.dropna().nunique() == 2:
            y_enc, _ = pd.factorize(y)
            mask = y.notna()
            X = df.loc[mask, num].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            y_enc = y_enc[mask.to_numpy()]
            if len(X) >= 200:
                Xtr, Xte, ytr, yte = train_test_split(X, y_enc, test_size=0.25,
                                                       random_state=cfg.random_state, stratify=y_enc)
                try:
                    clf = LogisticRegression(max_iter=250, n_jobs=1)
                    clf.fit(Xtr, ytr)
                    out["label_predictability_auc"] = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
                except Exception as e:
                    out["label_predictability_auc"] = {"error": str(e)}
            else:
                out["label_predictability_auc"] = {"note": "Need ≥ 200 labeled rows."}
        else:
            out["label_predictability_auc"] = {"note": "Binary labels only."}
    else:
        out["label_predictability_auc"] = {"note": "Install scikit-learn + select label + numeric features."}

    return out


def assess_fairness(df, cfg):
    if not cfg.group_cols:
        return {"note": "Select group columns to compute fairness checks."}
    out = {}
    label_ok = bool(cfg.label_col and cfg.label_col in df.columns)
    per = {}
    for gcol in [c for c in cfg.group_cols if c in df.columns]:
        counts = df[gcol].value_counts(dropna=False)
        shares = counts / max(1, counts.sum())
        stats: Dict[str, Any] = {
            "num_groups": int(len(counts)),
            "min_group_share": float(shares.min()) if len(shares) else None,
            "max_group_share": float(shares.max()) if len(shares) else None,
            "representation_share_top10": shares.sort_values(ascending=False).head(10).to_dict(),
        }
        miss_disp = {}
        for c in df.columns:
            mr = df.groupby(gcol)[c].apply(lambda s: s.isna().mean())
            if mr.shape[0] >= 2: miss_disp[c] = float(mr.max() - mr.min())
        stats["missingness_disparity_top10"] = dict(sorted(miss_disp.items(), key=lambda kv: kv[1], reverse=True)[:10])
        if label_ok:
            stats["label_missingness_by_group"] = df.groupby(gcol)[cfg.label_col].apply(lambda s: s.isna().mean()).to_dict()
            y = df[cfg.label_col]
            if y.dropna().nunique() == 2:
                tmp = df.copy()
                tmp["_y"], _ = pd.factorize(tmp[cfg.label_col])
                pos = tmp[tmp[cfg.label_col].notna()].groupby(gcol)["_y"].mean()
                if len(pos) >= 2:
                    stats["positive_rate_by_group"] = pos.to_dict()
                    stats["positive_rate_disparity"] = float(pos.max() - pos.min())
        per[gcol] = stats
    out["group_checks"] = per
    return out


def assess_security(df, cfg, dataset_bytes):
    out = {}
    text_cols = categorical_cols(df, exclude=[c for c in [cfg.label_col, cfg.split_col, cfg.time_col] if c])
    text_cols = text_cols[:cfg.pii_max_text_cols]
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
    out["confidentiality_pii_heuristics"] = {
        "note": "Heuristic scan. Validate with domain and legal review.",
        "threshold_hit_rate": float(cfg.thresholds.pii_hit_rate_threshold),
        "rows_sampled_per_col_max": int(cfg.pii_max_rows),
        "text_cols_scanned_max": int(cfg.pii_max_text_cols),
        "columns_with_hits": hits,
    }
    out["integrity"] = {"sha256": sha256_bytes(dataset_bytes)}
    out["availability_asset_checks"] = {"byte_size": int(len(dataset_bytes))}
    return out


def assess_all(df, cfg, dataset_bytes):
    if cfg.mode == "Full Scan":
        cfg.pii_max_rows = max(cfg.pii_max_rows, 5000)
        cfg.pii_max_text_cols = max(cfg.pii_max_text_cols, 25)
        cfg.rare_max_cat_cols = max(cfg.rare_max_cat_cols, 100)
        cfg.drift_max_num_cols = max(cfg.drift_max_num_cols, 100)
    return {
        "quality":     assess_quality(df, cfg),
        "reliability": assess_reliability(df, cfg),
        "robustness":  assess_robustness(df, cfg),
        "fairness":    assess_fairness(df, cfg),
        "security":    assess_security(df, cfg, dataset_bytes),
        "notes": {
            "sklearn_available": SKLEARN_OK, "mode": cfg.mode,
            "thresholds": {"drift_ks_threshold": cfg.thresholds.drift_ks_threshold,
                           "pii_hit_rate_threshold": cfg.thresholds.pii_hit_rate_threshold},
        },
    }

# ──────────────────────────────────────────────────────────────────────────────
# Verdict & recommendations
# ──────────────────────────────────────────────────────────────────────────────

def verdict_panel(report, cfg):
    reasons = []
    pii = report["security"]["confidentiality_pii_heuristics"]["columns_with_hits"]
    leak = report["quality"].get("split_leakage", {}).get("row_hash_cross_split_rate", None)
    drift = report["reliability"].get("numeric_drift_ks_first_last", {}).get("top_10_ks", {})
    if pii: reasons.append("PII-like patterns detected — review confidentiality and legal basis.")
    if leak is not None and float(leak) > 0: reasons.append("Potential split leakage — identical rows appear across splits.")
    if any(v is not None and float(v) > cfg.thresholds.drift_ks_threshold for v in drift.values()):
        reasons.append("Potential drift — at least one KS statistic exceeds the threshold.")
    if reasons: return "Needs review", "warn", reasons
    return "Looks OK (evidence-based)", "ok", ["No major red flags under current checks."]


def build_recommendations(report, cfg):
    recs = []
    q = report.get("quality", {})
    miss = q.get("missingness", {}).get("overall_missing_rate", None)
    dup  = q.get("duplicates", {}).get("exact_duplicate_row_rate", None)
    leak = q.get("split_leakage", {}).get("row_hash_cross_split_rate", None)
    drift = report.get("reliability", {}).get("numeric_drift_ks_first_last", {}).get("top_10_ks", {})
    pii   = report.get("security", {}).get("confidentiality_pii_heuristics", {}).get("columns_with_hits", {})

    if miss is not None and float(miss) > 0.05:
        recs.append("Missingness > 5% — inspect top missing columns and decide: drop, impute, or recollect.")
    if dup is not None and float(dup) > 0.01:
        recs.append("Duplicate rows > 1% — deduplicate and verify duplicates don't cross splits.")
    if leak is not None and float(leak) > 0:
        recs.append("Split leakage detected — re-split at entity level using ID columns.")
    if drift and any(float(v) > cfg.thresholds.drift_ks_threshold for v in drift.values() if v is not None):
        recs.append("Drift above threshold — compare distributions and consider retraining or recalibration.")
    if pii:
        recs.append("PII-like patterns found — mask or remove flagged columns and confirm legal basis.")
    if not recs:
        recs.append("No major red flags. Keep the report as evidence and rerun after data updates.")
    return recs

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="dsa-card" style="padding:24px 28px 18px 28px; margin-bottom:16px;">
  <div style="display:flex; align-items:center; gap:10px;">
    <span style="font-size:1.8rem;">📊</span>
    <div>
      <h2 style="margin:0;">Tabular Dataset Analyzer</h2>
      <div class="muted">Upload a CSV or Parquet, configure columns, run the scan.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

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
if "tab_guesses" not in st.session_state:
    st.session_state["tab_guesses"] = guess_columns(df)

with st.sidebar:
    st.header("⚙️ Run mode")
    mode = st.radio("Mode", ["Quick Scan", "Full Scan"], index=0)

    st.header("🎛 Thresholds")
    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
    th = PRESETS[preset_name]
    with st.expander("Show thresholds"):
        st.write({"drift_ks": th.drift_ks_threshold, "pii_hit_rate": th.pii_hit_rate_threshold})

    st.header("🗂 Columns")
    use_auto = st.toggle("Use suggested columns", value=True)
    guesses = st.session_state["tab_guesses"]

    if use_auto and guesses.get("notes"):
        st.caption("Suggestions")
        for n in guesses["notes"]: st.write("• " + n)

    col_filter = st.text_input("Filter columns", value="")
    filtered = [c for c in cols if col_filter.lower() in c.lower()] if col_filter else cols

    def pick_one(label, auto_val):
        opts = ["(none)"] + filtered
        idx = opts.index(auto_val) if (use_auto and auto_val in filtered) else 0
        chosen = st.selectbox(label, opts, index=idx)
        return None if chosen == "(none)" else chosen

    label_col = pick_one("Label column", guesses.get("label"))
    split_col  = pick_one("Split column", guesses.get("split"))
    time_col   = pick_one("Time column", guesses.get("time"))

    id_cols    = st.multiselect("ID columns", filtered,
                                default=[c for c in (guesses.get("ids",[]) if use_auto else []) if c in filtered])
    group_cols = st.multiselect("Group columns (fairness)", filtered,
                                default=[c for c in (guesses.get("groups",[]) if use_auto else []) if c in filtered])

    with st.expander("Advanced"):
        annotator_cols = st.multiselect("Annotator label columns", filtered, default=[])
        random_state   = st.number_input("Random seed", min_value=0, max_value=10000, value=7, step=1)

    st.divider()
    run = st.button("🔬 Run analysis", type="primary", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Preview (before run)
# ──────────────────────────────────────────────────────────────────────────────

if not run:
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1: kpi("Rows",     f"{df.shape[0]:,}", f"Format: {detected}")
    with c2: kpi("Columns",  f"{df.shape[1]:,}", f"Numeric: {len(numeric_cols(df))} · Cat: {len(categorical_cols(df))}")
    with c3: kpi("SHA-256",  sha256_bytes(file_bytes)[:14]+"…", "File fingerprint")
    with c4: kpi("scikit-learn", "✓ Available" if SKLEARN_OK else "✗ Not installed", "AUC check")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader(f"Preview — {detect_task_type(df, label_col)}")
    st.dataframe(df.head(50), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

cfg = AssessConfig(
    label_col=label_col, split_col=split_col, time_col=time_col,
    group_cols=group_cols, annotator_label_cols=annotator_cols,
    id_cols=id_cols, random_state=int(random_state),
    mode=mode, thresholds=th,
)

with st.spinner("Running checks…"):
    report = assess_all(df, cfg, file_bytes)

safe_report = to_json_safe(report)
verdict, vkind, reasons = verdict_panel(report, cfg)
recs = build_recommendations(report, cfg)
score, grade, score_components = compute_health_score(report, th.drift_ks_threshold)
dim_status = get_dimension_status(report, th.drift_ks_threshold)

cfg_dict = {
    "mode": mode, "preset": preset_name,
    "drift_ks_threshold": th.drift_ks_threshold,
    "pii_hit_rate_threshold": th.pii_hit_rate_threshold,
    "label_col": label_col, "split_col": split_col, "time_col": time_col,
    "id_cols": id_cols, "group_cols": group_cols,
    "random_state": int(random_state),
    "column_roles": {c: "label" for c in ([label_col] if label_col else [])} |
                    {c: "split" for c in ([split_col] if split_col else [])} |
                    {c: "time" for c in ([time_col] if time_col else [])} |
                    {c: "id" for c in id_cols} | {c: "group" for c in group_cols},
}

# ──────────────────────────────────────────────────────────────────────────────
# Verdict banner
# ──────────────────────────────────────────────────────────────────────────────

pii_cols = report["security"]["confidentiality_pii_heuristics"]["columns_with_hits"]

st.markdown(f"""
<div class="verdict-card">
  <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;">
    <div>
      <div class="verdict-title">Verdict</div>
      <div class="verdict-text">{verdict}</div>
      <div class="muted" style="margin-top:4px;">
        Mode: <span class="code-pill">{mode}</span>
        &nbsp; Preset: <span class="code-pill">{preset_name}</span>
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
    <div>
      <div style="font-weight:700; font-size:0.85rem; margin-bottom:6px;">Findings</div>
      <ul style="margin:0; padding-left:1.1em; font-size:0.83rem; line-height:1.7;">
        {''.join(f'<li>{clip_text(r, 200)}</li>' for r in reasons)}
      </ul>
    </div>
    <div>
      <div style="font-weight:700; font-size:0.85rem; margin-bottom:6px;">Recommended actions</div>
      <ul style="margin:0; padding-left:1.1em; font-size:0.83rem; line-height:1.7;">
        {''.join(f'<li>{clip_text(r, 200)}</li>' for r in recs)}
      </ul>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# KPI row
# ──────────────────────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5, gap="large")
miss_rate = report["quality"]["missingness"]["overall_missing_rate"]
dup_rate  = report["quality"]["duplicates"]["exact_duplicate_row_rate"]

with c1: kpi("Rows",        f"{df.shape[0]:,}",           f"Format: {detected}")
with c2: kpi("Columns",     f"{df.shape[1]:,}",           f"Task: {detect_task_type(df, cfg.label_col)}")
with c3: kpi("Missingness",  f"{miss_rate:.2%}",
             "Within limits" if miss_rate <= 0.05 else "⚠ Above 5%",
             color="#22c55e" if miss_rate <= 0.05 else "#f59e0b")
with c4: kpi("Duplicates",   f"{dup_rate:.2%}",
             "Within limits" if dup_rate <= 0.01 else "⚠ Above 1%",
             color="#22c55e" if dup_rate <= 0.01 else "#f59e0b")
with c5: kpi("PII flags",    str(len(pii_cols)),
             "No flags" if not pii_cols else f"{len(pii_cols)} col(s) flagged",
             color="#22c55e" if not pii_cols else "#ef4444")

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────

tab_ov, tab_q, tab_r, tab_rb, tab_f, tab_t, tab_sec, tab_exp = st.tabs(
    ["Overview", "Quality", "Reliability", "Robustness", "Fairness", "Transparency", "Security", "Export"]
)

# ── Overview ──────────────────────────────────────────────────────────────────
with tab_ov:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Health Score & Dimension Status")

    col_ring, col_dims = st.columns([1, 3], gap="large")
    with col_ring:
        st.markdown(health_ring_html(score, grade), unsafe_allow_html=True)

        # Component breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;opacity:0.5;margin-bottom:8px;">Score breakdown</div>', unsafe_allow_html=True)
        for dim, pts in score_components.items():
            weight = {"quality": 35, "security": 25, "reliability": 20, "robustness": 10, "fairness": 10}.get(dim, 10)
            pct = pts / weight if weight else 0
            st.markdown(progress_bar_html(dim.title(), pct, max_val=1.0, reverse=False, fmt=".0%"), unsafe_allow_html=True)

    with col_dims:
        icons = {"Quality": "🔍", "Reliability": "📉", "Robustness": "🛡", "Fairness": "⚖️", "Security": "🔒"}
        for i, (dim, (status, detail)) in enumerate(dim_status.items()):
            if i % 3 == 0:
                if i > 0: st.markdown("</div>", unsafe_allow_html=True)
                st.markdown('<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">', unsafe_allow_html=True)
            st.markdown(check_status_card(dim, icons.get(dim, "•"), status, detail), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="dsa-card" style="margin-top:14px;">', unsafe_allow_html=True)
    st.subheader("Quick metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Overall missingness",   f"{miss_rate:.4f}")
    with m2: st.metric("Duplicate row rate",    f"{dup_rate:.4f}")
    with m3:
        leak = report["quality"].get("split_leakage", {}).get("row_hash_cross_split_rate", None)
        st.metric("Split leakage",  "n/a" if leak is None else f"{float(leak):.4f}")
    with m4: st.metric("PII flagged columns", str(len(pii_cols)))
    st.caption("💡 Configure split, time, and group columns to unlock drift and fairness checks.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Quality ───────────────────────────────────────────────────────────────────
with tab_q:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Quality signals")
    q = report["quality"]
    render_metric_help_block([
        "missingness.overall_missing_rate", "missingness.top_10_columns_missing_rate",
        "duplicates.exact_duplicate_row_rate", "duplicates.duplicate_id_rate",
        "outliers_iqr.top_10_outlier_rate", "label_stats.label_missing_rate",
        "label_stats.label_cardinality", "label_agreement.exact_agreement_rate",
        "split_leakage.row_hash_cross_split_rate"
    ])

    # Missingness bar chart
    miss_top = q["missingness"]["top_10_columns_missing_rate"]
    if miss_top:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**Missingness — top 10 columns**")
            html_bars = "".join(progress_bar_html(col, v, max_val=1.0, reverse=True)
                               for col, v in sorted(miss_top.items(), key=lambda x: x[1], reverse=True))
            st.markdown(html_bars, unsafe_allow_html=True)

        with col2:
            st.markdown("**Outlier rate (IQR rule) — top 10 columns**")
            out_top = q["outliers_iqr"]["top_10_outlier_rate"]
            if out_top:
                html_bars2 = "".join(progress_bar_html(col, v, max_val=1.0, reverse=True)
                                    for col, v in sorted(out_top.items(), key=lambda x: x[1], reverse=True))
                st.markdown(html_bars2, unsafe_allow_html=True)
            else:
                st.info("No numeric columns to evaluate.")
    else:
        st.success("✓ No missingness in top columns.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Duplicates
    dup_info = q["duplicates"]
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Exact duplicate rows", f"{dup_info['exact_duplicate_row_rate']:.4f}",
                  delta="OK" if dup_info["exact_duplicate_row_rate"] <= 0.01 else "⚠ Above 1%",
                  delta_color="normal" if dup_info["exact_duplicate_row_rate"] <= 0.01 else "inverse")
    with c2:
        dup_id = dup_info.get("duplicate_id_rate")
        st.metric("Duplicate ID rate", "n/a" if dup_id is None else f"{dup_id:.4f}")

    # Label stats
    if "label_stats" in q:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**Label distribution**")
        ls = q["label_stats"]
        lc1, lc2, lc3 = st.columns(3)
        with lc1: st.metric("Missing rate", f"{ls['label_missing_rate']:.4f}")
        with lc2: st.metric("Cardinality",  str(ls["label_cardinality"]))
        with lc3: st.metric("Task type",    detect_task_type(df, cfg.label_col))

        if ls.get("top_classes_share"):
            cls_df = pd.DataFrame(list(ls["top_classes_share"].items()), columns=["class", "share"])
            st.bar_chart(cls_df.set_index("class")["share"])

    # Split leakage
    if "split_leakage" in q:
        st.markdown("<hr>", unsafe_allow_html=True)
        sl = q["split_leakage"]
        leak_rate = sl.get("row_hash_cross_split_rate")
        if leak_rate is not None and float(leak_rate) > 0:
            st.error(f"⚠️ Split leakage: {float(leak_rate):.4f} of unique rows appear in multiple splits.")
        else:
            st.success(f"✓ No split leakage detected. ({sl.get('num_unique_rows_hashed', 0):,} unique row hashes checked)")

    # Annotator agreement
    if "label_agreement" in q:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**Annotator agreement**")
        la = q["label_agreement"]
        a1, a2 = st.columns(2)
        with a1: st.metric("Agreement rate",    "n/a" if la["exact_agreement_rate"] is None else f"{la['exact_agreement_rate']:.4f}")
        with a2: st.metric("Disagreement rate", "n/a" if la["disagreement_rate"] is None else f"{la['disagreement_rate']:.4f}")

    # Column distribution explorer
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("🔎 Column Distribution Explorer"):
        sel_col = st.selectbox("Select column", df.columns.tolist(), key="dist_col")
        if sel_col:
            col_data = df[sel_col]
            st.caption(f"dtype: {col_data.dtype}  |  null: {col_data.isna().mean():.2%}  |  unique: {col_data.nunique():,}")
            if pd.api.types.is_numeric_dtype(col_data.dtype):
                vals = pd.to_numeric(col_data, errors="coerce").dropna()
                if len(vals) > 0:
                    desc = vals.describe()
                    dc1, dc2, dc3, dc4 = st.columns(4)
                    with dc1: st.metric("Mean", f"{desc['mean']:.4g}")
                    with dc2: st.metric("Std",  f"{desc['std']:.4g}")
                    with dc3: st.metric("Min",  f"{desc['min']:.4g}")
                    with dc4: st.metric("Max",  f"{desc['max']:.4g}")
                    binned = pd.cut(vals, bins=min(30, int(len(vals)**0.5)+5))
                    hist = binned.value_counts().sort_index()
                    hist.index = hist.index.astype(str)
                    st.bar_chart(hist.rename("count"))
            else:
                vc = col_data.value_counts(dropna=False).head(20)
                st.bar_chart(vc)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Reliability ───────────────────────────────────────────────────────────────
with tab_r:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Reliability - Stability and drift")
    r = report["reliability"]
    render_metric_help_block([
        "reliability.missing_rate_by_slice", "reliability.numeric_drift_ks_first_last",
        "reliability.schema_consistency"
    ])

    if "note" in r:
        st.warning(r["note"])
    else:
        st.markdown(f'Slice type: {badge(r.get("slice_type","unknown"), "info")}', unsafe_allow_html=True)

        if "missing_rate_by_slice" in r:
            srs = pd.Series(r["missing_rate_by_slice"]).sort_index()
            st.markdown("**Missingness by slice**")
            st.line_chart(srs)

        if "numeric_drift_ks_first_last" in r:
            ks_info = r["numeric_drift_ks_first_last"]
            ks = ks_info.get("top_10_ks", {})
            st.markdown(f"**Numeric drift (KS): `{ks_info.get('first_slice')}` vs `{ks_info.get('last_slice')}`**")

            if ks:
                # Color-coded bar chart
                html_bars = "".join(
                    progress_bar_html(col, v, max_val=max(v*1.5, th.drift_ks_threshold*1.5), reverse=True)
                    for col, v in sorted(ks.items(), key=lambda x: x[1], reverse=True)
                    if v is not None
                )
                st.markdown(html_bars, unsafe_allow_html=True)

                ks_df = pd.DataFrame([(c, v) for c, v in ks.items() if v is not None],
                                     columns=["Column", "KS Statistic"]).sort_values("KS Statistic", ascending=False)
                ks_df["Above threshold"] = ks_df["KS Statistic"] > th.drift_ks_threshold
                st.dataframe(ks_df, use_container_width=True, hide_index=True)

                if not ks_df.empty:
                    st.bar_chart(ks_df.set_index("Column")["KS Statistic"])

    with st.expander("Schema consistency"):
        sc = r.get("schema_consistency", {})
        const_cols = sc.get("constant_columns", [])
        if const_cols:
            st.warning(f"Constant columns (no variance): {', '.join(const_cols)}")
        else:
            st.success("✓ No constant columns found.")
        st.write(f"Rows: {sc.get('num_rows',0):,} · Columns: {sc.get('num_cols',0):,}")
        dtypes_df = pd.DataFrame(list(sc.get("dtypes", {}).items()), columns=["Column", "Type"])
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Robustness ────────────────────────────────────────────────────────────────
with tab_rb:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Robustness")
    rb = report["robustness"]
    render_metric_help_block([
        "robustness.rare_category_label_concentration", "robustness.row_anomaly_score_mad",
        "robustness.label_predictability_auc"
    ])

    if not SKLEARN_OK:
        st.warning("scikit-learn not installed. AUC check is disabled.")

    # Rare category label concentration
    rcl = rb.get("rare_category_label_concentration", {})
    if rcl:
        st.markdown(f"**Rare-category label concentration** — {rcl.get('num_findings', 0)} findings in {rcl.get('columns_scanned', 0)} columns scanned")
        if rcl.get("top_findings"):
            findings_df = pd.DataFrame(rcl["top_findings"])
            st.dataframe(findings_df, use_container_width=True, hide_index=True,
                         column_config={"top_label_share": st.column_config.ProgressColumn("Top label share", min_value=0, max_value=1)})
        else:
            st.success("✓ No suspicious rare-category concentration found.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Row anomaly score
    ram = rb.get("row_anomaly_score_mad", {})
    if ram:
        st.markdown("**Row anomaly score (MAD-based)**")
        r1, r2, r3, r4 = st.columns(4)
        with r1: st.metric("Mean score", f"{ram.get('mean', 0):.3f}")
        with r2: st.metric("P95",        f"{ram.get('p95', 0):.3f}")
        with r3: st.metric("P99",        f"{ram.get('p99', 0):.3f}",
                            delta="⚠ High" if float(ram.get('p99', 0)) > 10 else "OK",
                            delta_color="normal" if float(ram.get('p99', 0)) <= 10 else "inverse")
        with r4: st.metric("Max",        f"{ram.get('max', 0):.3f}")
        if ram.get("top_20_row_indices"):
            with st.expander("Top 20 anomalous row indices"):
                st.write(ram["top_20_row_indices"])

    st.markdown("<hr>", unsafe_allow_html=True)

    # AUC
    auc = rb.get("label_predictability_auc", {})
    st.markdown("**Label predictability AUC** *(binary labels only)*")
    if isinstance(auc, float):
        status = "bad" if auc > 0.85 else ("warn" if auc > 0.75 else "ok")
        st.markdown(badge(f"AUC = {auc:.4f}", status), unsafe_allow_html=True)
        if auc > 0.85:
            st.warning("High AUC indicates numeric features predict the label trivially — check for leakage or trivially-easy features.")
        else:
            st.success("✓ AUC is within reasonable range.")
    elif isinstance(auc, dict):
        if "note" in auc: st.info(auc["note"])
        elif "error" in auc: st.error(auc["error"])

    st.markdown('</div>', unsafe_allow_html=True)

# ── Fairness ──────────────────────────────────────────────────────────────────
with tab_f:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Fairness proxies")
    f = report["fairness"]
    render_metric_help_block([
        "fairness.representation_share_top10", "fairness.positive_rate_by_group",
        "fairness.positive_rate_disparity", "fairness.missingness_disparity_top10"
    ])

    if "note" in f:
        st.warning(f["note"])
    else:
        for gcol, stats_ in f["group_checks"].items():
            st.markdown(f"#### Group column: `{gcol}`")
            fa1, fa2, fa3 = st.columns(3)
            with fa1: st.metric("Groups", str(stats_["num_groups"]))
            with fa2: st.metric("Min share", f"{stats_.get('min_group_share', 0):.3f}")
            with fa3: st.metric("Max share", f"{stats_.get('max_group_share', 0):.3f}")

            rep = stats_.get("representation_share_top10", {})
            if rep:
                rep_df = pd.DataFrame(list(rep.items()), columns=["Group", "Share"]).sort_values("Share", ascending=False)
                st.markdown("**Representation (top 10)**")
                st.bar_chart(rep_df.set_index("Group")["Share"])

            if "positive_rate_by_group" in stats_:
                pr = pd.Series(stats_["positive_rate_by_group"]).sort_index()
                st.markdown("**Positive rate by group**")
                st.bar_chart(pr)
                disp = stats_.get("positive_rate_disparity", 0)
                dstatus = "bad" if disp > 0.2 else ("warn" if disp > 0.1 else "ok")
                st.markdown(f"Disparity (max − min): {badge(f'{disp:.4f}', dstatus)}", unsafe_allow_html=True)

            miss_disp = stats_.get("missingness_disparity_top10", {})
            if miss_disp:
                md_df = pd.DataFrame(list(miss_disp.items()), columns=["Column", "Max−Min missing"]).sort_values("Max−Min missing", ascending=False)
                with st.expander("Missingness disparity by column"):
                    st.dataframe(md_df, use_container_width=True, hide_index=True)

            st.markdown("---")

    st.markdown('</div>', unsafe_allow_html=True)

# ── Transparency ──────────────────────────────────────────────────────────────
with tab_t:
    render_research_grade_transparency_tab(df, file_name, file_bytes, cfg_dict, report, preset_name, mode)

# ── Security ──────────────────────────────────────────────────────────────────
with tab_sec:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Security - CIA prerequisites")
    s = report["security"]
    render_metric_help_block([
        "security.confidentiality_pii_heuristics", "security.integrity.sha256",
        "security.availability_asset_checks.byte_size"
    ])

    # Integrity
    sha = s["integrity"]["sha256"]
    st.markdown("**File integrity**")
    sc1, sc2 = st.columns([2, 1])
    with sc1:
        st.markdown(f'<div class="mono" style="font-size:0.8rem;word-break:break-all;padding:10px;border:1px solid var(--c-border);border-radius:8px;">{sha}</div>', unsafe_allow_html=True)
    with sc2:
        st.metric("File size", f"{s['availability_asset_checks']['byte_size'] / 1024:.1f} KB")

    st.markdown("<hr>", unsafe_allow_html=True)

    # PII
    pii_h = s.get("confidentiality_pii_heuristics", {})
    pii_cols_found = pii_h.get("columns_with_hits", {})
    st.markdown("**PII heuristic scan**")
    st.caption(pii_h.get("note", ""))

    if pii_cols_found:
        rows = [{"Column": col, "Pattern": k, "Hit rate": v}
                for col, hits in pii_cols_found.items() for k, v in hits.items()]
        pii_df = pd.DataFrame(rows).sort_values("Hit rate", ascending=False)
        st.dataframe(pii_df, use_container_width=True, hide_index=True,
                     column_config={"Hit rate": st.column_config.ProgressColumn(min_value=0, max_value=1)})
        st.error(f"⚠️ {len(pii_cols_found)} column(s) with PII-like patterns detected. Validate with legal review before deployment.")
    else:
        st.success("✓ No PII-like patterns flagged in the heuristic scan.")

    st.markdown('</div>', unsafe_allow_html=True)

# ── Export ────────────────────────────────────────────────────────────────────
with tab_exp:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Export")
    st.caption("All exports include report data, thresholds, and mode. JSON values are pandas/numpy-safe.")

    col_e1, col_e2, col_e3 = st.columns(3, gap="large")

    with col_e1:
        st.markdown("**JSON report**")
        json_bytes = json.dumps(safe_report, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button("⬇ Download JSON", data=json_bytes,
                           file_name="dataset_report.json", mime="application/json",
                           use_container_width=True)

    with col_e2:
        st.markdown("**Markdown summary**")
        md_lines = [
            f"# Dataset Safety Report — {file_name}", "",
            f"- **Mode:** {mode}", f"- **Preset:** {preset_name}",
            f"- **Health score:** {score}/100 (Grade {grade})",
            f"- **Verdict:** {verdict}", "",
            "## Findings", *[f"- {r}" for r in reasons], "",
            "## Recommended actions", *[f"- {r}" for r in recs], "",
            "## Key metrics",
            f"- Rows: {df.shape[0]:,}", f"- Columns: {df.shape[1]:,}",
            f"- Missingness: {miss_rate:.4f}", f"- Duplicate rows: {dup_rate:.4f}",
            f"- PII flagged columns: {len(pii_cols)}", f"- SHA-256: {sha256_bytes(file_bytes)}", "",
            "---", "*Heuristic report. Validate with domain and legal review.*"
        ]
        md_bytes = "\n".join(md_lines).encode("utf-8")
        st.download_button("⬇ Download Markdown", data=md_bytes,
                           file_name="dataset_report.md", mime="text/markdown",
                           use_container_width=True)

    with col_e3:
        st.markdown("**HTML report**")
        html_content = build_html_report(
            df=df, report=report, cfg_dict=cfg_dict, file_name=file_name,
            file_bytes=file_bytes, verdict=verdict, reasons=reasons,
            recs=recs, score=score, grade=grade,
        )
        st.download_button("⬇ Download HTML", data=html_content.encode("utf-8"),
                           file_name="dataset_report.html", mime="text/html",
                           use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    registry_payload = {"analyzer": "tabular", "threshold_docs": THRESHOLD_DOCS, "metric_docs": METRIC_DOCS}
    st.download_button("⬇ Download metric registry JSON", data=json.dumps(to_json_safe(registry_payload), indent=2, ensure_ascii=False).encode("utf-8"), file_name="tabular_metric_registry.json", mime="application/json", use_container_width=True)
    with st.expander("Raw JSON (preview)"):
        st.json(safe_report)
    st.markdown('</div>', unsafe_allow_html=True)
