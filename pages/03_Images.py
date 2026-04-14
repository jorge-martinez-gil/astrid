# 04_Images_experimental.py
# Streamlit app: COMPREHENSIVE image-dataset trustworthiness analyzer
# Implements the full metric catalog for Quality, Reliability, Robustness,
# Fairness, Transparency, and Security — aligned with EU AI Act,
# ISO/IEC standards, NIST AI RMF, and ENISA guidance.
#
# Install:
#   pip install streamlit pandas numpy pillow imagehash scikit-learn scipy pyyaml
#   pip install pyarrow   # optional, for parquet metadata
#
# Run:
#   streamlit run 04_Images_experimental.py
#
# Notes:
# - Heuristic + statistical checks. Validate any flags with domain + legal review.
# - No face detection, no deep model embeddings (kept dependency-light).
# - Annotation files (JSON/XML/TXT) inside the ZIP are parsed when present.

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import zipfile
import base64
from collections import Counter, defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageStat, ImageOps

# Optional perceptual hashing
try:
    import imagehash
    IMAGEHASH_OK = True
except Exception:
    IMAGEHASH_OK = False

# Optional sklearn for outlier detection & mutual information
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import mutual_info_score, cohen_kappa_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import entropy as sp_entropy
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# =========================
# Styling
# =========================

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from utils import SHARED_CSS, build_html_report, sha256_bytes as _sha256_util, badge as _badge_util, kpi as _kpi_util, health_ring_html, progress_bar_html, compute_health_score, to_json_safe as _to_json_safe_util
    HAS_UTILS = True
except Exception:
    HAS_UTILS = False

st.set_page_config(page_title="Image Dataset Trustworthiness Analyzer (Experimental)", page_icon="🔬", layout="wide")

CUSTOM_CSS = """
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 3rem; max-width: 1300px; }
h1, h2, h3 { letter-spacing: -0.02em; }
.small-muted { color: rgba(128,128,128,0.95); font-size: 0.92rem; }
.kpi-card {
  border: 1px solid rgba(120,120,120,0.20);
  border-radius: 14px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.03);
}
.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 0.85rem;
  border: 1px solid rgba(120,120,120,0.25);
}
.badge-ok { background: rgba(0, 200, 0, 0.12); }
.badge-warn { background: rgba(255, 165, 0, 0.12); }
.badge-bad { background: rgba(255, 0, 0, 0.10); }
.section-card {
  border: 1px solid rgba(120,120,120,0.20);
  border-radius: 16px;
  padding: 18px 18px 8px 18px;
  background: rgba(255,255,255,0.02);
}
.dsa-card {
  border: 1px solid rgba(120,120,120,0.20);
  border-radius: 16px;
  padding: 18px 18px 8px 18px;
  background: rgba(255,255,255,0.02);
  margin-bottom: 12px;
}
.verdict-card {
  border: 1px solid rgba(120,120,120,0.20);
  border-radius: 16px;
  padding: 18px 20px;
  background: rgba(255,255,255,0.02);
  margin-bottom: 12px;
}
.config-row { display:flex; gap:12px; padding:3px 0; font-size:0.9rem; }
.config-key { font-weight:600; min-width:200px; }
.config-val { opacity:0.85; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:0.85rem; }
.muted { color: rgba(128,128,128,0.85); font-size:0.85rem; }
.transparency-header { font-weight:700; font-size:1.05rem; margin-bottom:8px; }
hr { border: none; height: 1px; background: rgba(120,120,120,0.22); margin: 1.2rem 0; }
.code-pill {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.85rem;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid rgba(120,120,120,0.25);
}
.metric-table { width:100%; border-collapse:collapse; font-size:0.88rem; }
.metric-table th, .metric-table td { padding:6px 10px; border-bottom:1px solid rgba(120,120,120,0.15); text-align:left; }
.metric-table th { font-weight:700; background:rgba(120,120,120,0.05); }
</style>
"""
if HAS_UTILS:
    st.markdown(SHARED_CSS, unsafe_allow_html=True)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================
# JSON safety
# =========================

def to_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return to_json_safe(obj.tolist())
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if obj is pd.NA:
        return None
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


# =========================
# Helpers
# =========================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
ANN_EXTS = {".json", ".xml", ".txt", ".csv"}

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def clip_text(s: str, n: int = 90) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"

def to_datetime_if_possible(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return s
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return s

def categorical_cols(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    exclude_set = set(exclude or [])
    return [c for c in df.columns
            if c not in exclude_set
            and not pd.api.types.is_datetime64_any_dtype(df[c].dtype)
            and not pd.api.types.is_numeric_dtype(df[c].dtype)]

def numeric_cols(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    exclude_set = set(exclude or [])
    return [c for c in df.columns
            if c not in exclude_set and pd.api.types.is_numeric_dtype(df[c].dtype)]

def ks_statistic(x1: np.ndarray, x2: np.ndarray) -> Optional[float]:
    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]
    if len(x1) < 30 or len(x2) < 30:
        return None
    x1 = np.sort(x1)
    x2 = np.sort(x2)
    all_vals = np.sort(np.unique(np.concatenate([x1, x2])))
    cdf1 = np.searchsorted(x1, all_vals, side="right") / len(x1)
    cdf2 = np.searchsorted(x2, all_vals, side="right") / len(x2)
    return float(np.max(np.abs(cdf1 - cdf2)))

def badge(label: str, kind: str) -> str:
    cls = {"ok": "badge badge-ok", "warn": "badge badge-warn", "bad": "badge badge-bad"}.get(kind, "badge")
    return f'<span class="{cls}">{label}</span>'

def kpi(title: str, value: str, hint: str = "") -> None:
    st.markdown(
        f"""<div class="kpi-card">
          <div style="font-size:0.92rem; opacity:0.9;">{title}</div>
          <div style="font-size:1.5rem; font-weight:700; margin-top:4px;">{value}</div>
          <div class="muted" style="margin-top:6px;">{hint}</div>
        </div>""",
        unsafe_allow_html=True,
    )

def normalized_entropy(counts: Dict[Any, int]) -> Tuple[float, float, float]:
    """Return (entropy, normalized_entropy, effective_number)."""
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        return 0.0, 1.0, float(len(counts))
    probs = [c / total for c in counts.values()]
    h = -sum(p * math.log(p) for p in probs if p > 0)
    h_norm = h / math.log(len(counts))
    n_eff = math.exp(h)
    return h, h_norm, n_eff

def jsd_distributions(p_dict: Dict[str, float], q_dict: Dict[str, float]) -> Optional[float]:
    """Jensen-Shannon divergence between two distributions given as dicts."""
    if not SCIPY_OK:
        return None
    keys = sorted(set(p_dict) | set(q_dict))
    if len(keys) == 0:
        return None
    p = np.array([p_dict.get(k, 0.0) for k in keys], dtype=float)
    q = np.array([q_dict.get(k, 0.0) for k in keys], dtype=float)
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum == 0 or q_sum == 0:
        return None
    p = p / p_sum
    q = q / q_sum
    return float(jensenshannon(p, q) ** 2)

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone_like": re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}\b"),
    "ip_v4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "credit_card_like": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
}


# =========================
# Thresholds + presets
# =========================

@dataclass(frozen=True)
class Thresholds:
    # Quality
    drift_ks_threshold: float
    pii_hit_rate_threshold: float
    perceptual_dup_hamming_threshold: int
    blur_laplacian_like_threshold: float
    min_resolution_short_side: int
    label_error_rate_warn: float
    cross_split_leakage_warn: float
    exact_duplicate_rate_warn: float
    near_duplicate_rate_warn: float
    metadata_completeness_warn: float
    schema_consistency_warn: float
    # Reliability
    annotator_agreement_kappa_warn: float
    # Robustness
    worst_bin_coverage_warn: float
    outlier_rate_warn: float
    # Fairness
    representation_ratio_low: float
    representation_ratio_high: float
    annotation_quality_parity_gap_warn: float
    missingness_disparity_warn: float
    # Transparency
    traceability_coverage_warn: float
    # Security
    integrity_coverage_warn: float
    suspicious_sample_rate_warn: float
    source_concentration_hhi_warn: float


PRESETS: Dict[str, Thresholds] = {
    "Balanced (recommended)": Thresholds(
        drift_ks_threshold=0.30,
        pii_hit_rate_threshold=0.01,
        perceptual_dup_hamming_threshold=6,
        blur_laplacian_like_threshold=25.0,
        min_resolution_short_side=224,
        label_error_rate_warn=0.02,
        cross_split_leakage_warn=0.001,
        exact_duplicate_rate_warn=0.01,
        near_duplicate_rate_warn=0.02,
        metadata_completeness_warn=0.90,
        schema_consistency_warn=0.95,
        annotator_agreement_kappa_warn=0.80,
        worst_bin_coverage_warn=0.03,
        outlier_rate_warn=0.05,
        representation_ratio_low=0.80,
        representation_ratio_high=1.25,
        annotation_quality_parity_gap_warn=0.05,
        missingness_disparity_warn=0.10,
        traceability_coverage_warn=0.95,
        integrity_coverage_warn=1.0,
        suspicious_sample_rate_warn=0.02,
        source_concentration_hhi_warn=0.25,
    ),
    "Strict": Thresholds(
        drift_ks_threshold=0.20,
        pii_hit_rate_threshold=0.005,
        perceptual_dup_hamming_threshold=4,
        blur_laplacian_like_threshold=35.0,
        min_resolution_short_side=256,
        label_error_rate_warn=0.01,
        cross_split_leakage_warn=0.0,
        exact_duplicate_rate_warn=0.005,
        near_duplicate_rate_warn=0.01,
        metadata_completeness_warn=0.95,
        schema_consistency_warn=0.98,
        annotator_agreement_kappa_warn=0.85,
        worst_bin_coverage_warn=0.05,
        outlier_rate_warn=0.03,
        representation_ratio_low=0.85,
        representation_ratio_high=1.18,
        annotation_quality_parity_gap_warn=0.03,
        missingness_disparity_warn=0.05,
        traceability_coverage_warn=0.98,
        integrity_coverage_warn=1.0,
        suspicious_sample_rate_warn=0.01,
        source_concentration_hhi_warn=0.20,
    ),
    "Lenient": Thresholds(
        drift_ks_threshold=0.40,
        pii_hit_rate_threshold=0.02,
        perceptual_dup_hamming_threshold=8,
        blur_laplacian_like_threshold=18.0,
        min_resolution_short_side=160,
        label_error_rate_warn=0.05,
        cross_split_leakage_warn=0.005,
        exact_duplicate_rate_warn=0.02,
        near_duplicate_rate_warn=0.05,
        metadata_completeness_warn=0.80,
        schema_consistency_warn=0.90,
        annotator_agreement_kappa_warn=0.70,
        worst_bin_coverage_warn=0.02,
        outlier_rate_warn=0.08,
        representation_ratio_low=0.70,
        representation_ratio_high=1.40,
        annotation_quality_parity_gap_warn=0.10,
        missingness_disparity_warn=0.15,
        traceability_coverage_warn=0.90,
        integrity_coverage_warn=0.99,
        suspicious_sample_rate_warn=0.05,
        source_concentration_hhi_warn=0.35,
    ),
}


# =========================
# Metric documentation registry
# =========================

THRESHOLD_DOCS: Dict[str, Dict[str, str]] = {
    "drift_ks_threshold": {"label": "Drift KS threshold", "description": "Warn when the Kolmogorov-Smirnov distance between the first and last slice exceeds this value."},
    "pii_hit_rate_threshold": {"label": "PII hit-rate threshold", "description": "Warn when PII-like pattern hits in text-derived fields exceed this rate."},
    "perceptual_dup_hamming_threshold": {"label": "Perceptual duplicate Hamming threshold", "description": "Maximum Hamming distance between perceptual hashes to count two images as near-duplicates."},
    "blur_laplacian_like_threshold": {"label": "Blur threshold", "description": "Images below this Laplacian-variance proxy are treated as blur-prone."},
    "min_resolution_short_side": {"label": "Minimum short side", "description": "Images below this short-side resolution are flagged as low resolution."},
    "label_error_rate_warn": {"label": "Label error-rate warning", "description": "Reference warning level for observed label-quality issues."},
    "cross_split_leakage_warn": {"label": "Cross-split leakage warning", "description": "Warn when the fraction of shared content across splits exceeds this value."},
    "exact_duplicate_rate_warn": {"label": "Exact duplicate-rate warning", "description": "Warn when SHA256-identical duplicates exceed this rate."},
    "near_duplicate_rate_warn": {"label": "Near duplicate-rate warning", "description": "Warn when perceptual near-duplicates exceed this rate."},
    "metadata_completeness_warn": {"label": "Metadata completeness warning", "description": "Warn when required metadata coverage falls below this value."},
    "schema_consistency_warn": {"label": "Schema consistency warning", "description": "Warn when parsed annotation validity falls below this value."},
    "annotator_agreement_kappa_warn": {"label": "Annotator agreement warning", "description": "Warn when mean Cohen kappa falls below this value."},
    "worst_bin_coverage_warn": {"label": "Worst-bin coverage warning", "description": "Warn when the least-covered condition bin falls below this share."},
    "outlier_rate_warn": {"label": "Outlier-rate warning", "description": "Warn when outlier prevalence exceeds this value."},
    "representation_ratio_low": {"label": "Representation ratio low", "description": "Lower acceptable bound for subgroup representation relative to uniform coverage."},
    "representation_ratio_high": {"label": "Representation ratio high", "description": "Upper acceptable bound for subgroup representation relative to uniform coverage."},
    "annotation_quality_parity_gap_warn": {"label": "Annotation parity-gap warning", "description": "Warn when annotation quality gaps between groups exceed this value."},
    "missingness_disparity_warn": {"label": "Missingness disparity warning", "description": "Warn when group-wise missingness gaps exceed this value."},
    "traceability_coverage_warn": {"label": "Traceability coverage warning", "description": "Warn when traceability or provenance coverage falls below this value."},
    "integrity_coverage_warn": {"label": "Integrity coverage warning", "description": "Target integrity coverage across scanned samples."},
    "suspicious_sample_rate_warn": {"label": "Suspicious-sample warning", "description": "Warn when suspicious sample prevalence exceeds this rate."},
    "source_concentration_hhi_warn": {"label": "Source concentration warning", "description": "Warn when source concentration, measured with HHI, exceeds this value."},
}

METRIC_DOCS: Dict[str, Dict[str, Any]] = {
    "quality.readability": {"title": "Q1: Readability rate", "section": "Quality", "description": "Fraction of scanned images that could be opened and decoded successfully.", "why_it_matters": "Unreadable files break training, evaluation, and reproducibility workflows.", "method": "Counts images with open_ok=True after attempted decode with Pillow.", "threshold_key": None, "interpretation": "Values near 1.0 are expected for production-grade datasets.", "report_path": "quality.readability"},
    "quality.annotation_linkage": {"title": "Q2: Annotation linkage", "section": "Quality", "description": "Fraction of images that can be paired with an annotation file sharing the same basename.", "why_it_matters": "Unlinked images reduce label usability and annotation traceability.", "method": "Matches image and annotation stems inside the ZIP.", "threshold_key": None, "interpretation": "Higher is better. Low linkage suggests packaging or naming issues.", "report_path": "quality.annotation_linkage"},
    "quality.format_conformance": {"title": "Q3: Format conformance", "section": "Quality", "description": "Rate of files whose extension belongs to the supported image format set.", "why_it_matters": "Unexpected file types complicate ingestion and downstream processing.", "method": "Checks file extension membership in IMG_EXTS.", "threshold_key": None, "interpretation": "Higher is better.", "report_path": "quality.format_conformance"},
    "quality.metadata_completeness": {"title": "Q4: Metadata completeness", "section": "Quality", "description": "Average non-missingness across selected metadata fields.", "why_it_matters": "Missing metadata weakens traceability, fairness checks, and analysis coverage.", "method": "Computes per-row completeness over present fields and averages across rows.", "threshold_key": "metadata_completeness_warn", "interpretation": "Below-threshold values suggest metadata remediation.", "report_path": "quality.metadata_completeness"},
    "quality.split_coverage": {"title": "Q5: Split coverage", "section": "Quality", "description": "Fraction of records assigned to a train, validation, test, or equivalent split.", "why_it_matters": "Undefined splits hinder reproducible experimentation.", "method": "Measures non-null coverage of the configured split column.", "threshold_key": None, "interpretation": "Higher is better.", "report_path": "quality.split_coverage"},
    "quality.bbox_validity": {"title": "Q7: Bounding-box validity", "section": "Quality", "description": "Fraction of parsed bounding boxes that satisfy basic geometric validity rules.", "why_it_matters": "Invalid boxes degrade object detection training and evaluation.", "method": "Checks VOC and YOLO-style boxes for valid coordinate structure.", "threshold_key": None, "interpretation": "Higher is better.", "report_path": "quality.bbox_validity"},
    "quality.annotation_schema_consistency": {"title": "Q9: Annotation schema consistency", "section": "Quality", "description": "Fraction of annotation files that parse successfully under the supported schema heuristics.", "why_it_matters": "Inconsistent annotation structure blocks automation and quality assurance.", "method": "Parses JSON, XML, and TXT annotations and records validity.", "threshold_key": "schema_consistency_warn", "interpretation": "Below-threshold values suggest schema drift or malformed annotations.", "report_path": "quality.annotation_schema_consistency"},
    "quality.low_resolution": {"title": "Low-resolution rate", "section": "Quality", "description": "Share of images whose shortest side is below the configured minimum resolution.", "why_it_matters": "Low-resolution images may be unsuitable for model training or inspection.", "method": "Compares image short_side against the configured threshold.", "threshold_key": "min_resolution_short_side", "interpretation": "Lower is better.", "report_path": "quality.low_resolution"},
    "quality.blur_proxy": {"title": "Blur proxy", "section": "Quality", "description": "Blur-proneness estimated from Laplacian variance on grayscale images.", "why_it_matters": "Blur can reduce label quality, recognition accuracy, and dataset utility.", "method": "Computes a Laplacian-variance proxy from local grayscale second derivatives.", "threshold_key": "blur_laplacian_like_threshold", "interpretation": "Lower blur-flag rates are better.", "report_path": "quality.blur_proxy"},
    "quality.duplicates": {"title": "Q12-Q13: Duplicates", "section": "Quality", "description": "Measures exact duplicates with SHA256 and near-duplicates with perceptual hashing.", "why_it_matters": "Duplicates can inflate confidence, reduce diversity, and leak content across splits.", "method": "Uses SHA256 for exact equality and perceptual hash Hamming distance for similarity.", "threshold_key": "exact_duplicate_rate_warn", "interpretation": "Lower is better. Near-duplicate handling also depends on the perceptual hash threshold.", "report_path": "quality.duplicates"},
    "quality.cross_split_leakage": {"title": "Q14: Cross-split leakage", "section": "Quality", "description": "Fraction of items whose exact content appears in more than one split.", "why_it_matters": "Leakage inflates validation and test performance estimates.", "method": "Groups SHA256 hashes and checks whether the split column has more than one value per hash.", "threshold_key": "cross_split_leakage_warn", "interpretation": "Near zero is expected.", "report_path": "quality.cross_split_leakage"},
    "quality.conflicting_duplicate_labels": {"title": "Q15: Conflicting duplicate labels", "section": "Quality", "description": "Fraction of exact duplicate groups that carry more than one label.", "why_it_matters": "Conflicting labels point to annotation disagreement or versioning errors.", "method": "Groups identical SHA256 hashes and counts unique labels within each group.", "threshold_key": "label_error_rate_warn", "interpretation": "Lower is better.", "report_path": "quality.conflicting_duplicate_labels"},
    "quality.class_balance": {"title": "Q16-Q18: Class balance", "section": "Quality", "description": "Entropy-based measures of class distribution balance.", "why_it_matters": "Severe imbalance can bias training and weaken minority-class performance.", "method": "Computes entropy, normalized entropy, and effective number of classes from label counts.", "threshold_key": None, "interpretation": "Higher normalized entropy usually means better balance.", "report_path": "quality.class_balance"},
    "reliability.annotator_agreement": {"title": "R1-R3: Annotator agreement", "section": "Reliability", "description": "Agreement between multiple annotators using pairwise Cohen kappa.", "why_it_matters": "Low agreement often signals ambiguous labels or inadequate annotation guidance.", "method": "Computes pairwise Cohen kappa across configured annotator columns.", "threshold_key": "annotator_agreement_kappa_warn", "interpretation": "Higher is better.", "report_path": "reliability.annotator_agreement"},
    "reliability.provenance_coverage": {"title": "R10: Provenance coverage", "section": "Reliability", "description": "Fraction of rows with at least one populated source or provenance field.", "why_it_matters": "Provenance supports repeatability, accountability, and auditability.", "method": "Checks whether any configured source column is non-null for each row.", "threshold_key": "traceability_coverage_warn", "interpretation": "Higher is better.", "report_path": "reliability.provenance_coverage"},
    "reliability.feature_drift": {"title": "Feature drift", "section": "Reliability", "description": "Change in low-level image feature distributions across slices.", "why_it_matters": "Drift may indicate temporal shift, split mismatch, or unstable data collection.", "method": "Compares first and last slice distributions using the KS statistic.", "threshold_key": "drift_ks_threshold", "interpretation": "Lower is better.", "report_path": "reliability.feature_drift_ks_first_last"},
    "robustness.condition_coverage": {"title": "RB1-RB2: Condition coverage", "section": "Robustness", "description": "Coverage of configured acquisition or environment condition bins.", "why_it_matters": "Poor coverage weakens generalization to underrepresented conditions.", "method": "Computes proportions per condition value and tracks the least-covered bin.", "threshold_key": "worst_bin_coverage_warn", "interpretation": "Higher worst-bin coverage is better.", "report_path": "robustness.condition_coverage"},
    "robustness.distribution_divergence": {"title": "RB3: Distribution divergence", "section": "Robustness", "description": "Divergence between observed condition coverage and a reference distribution.", "why_it_matters": "Large divergence can indicate skewed sampling.", "method": "Computes Jensen-Shannon divergence against a uniform baseline.", "threshold_key": None, "interpretation": "Lower is better.", "report_path": "robustness.distribution_divergence"},
    "robustness.image_feature_outliers_mad": {"title": "RB9: Image feature outliers", "section": "Robustness", "description": "Share of rows flagged as outliers from robust MAD-based feature scoring.", "why_it_matters": "Outliers may represent corruption, distribution tails, or rare acquisition setups.", "method": "Builds robust deviation scores over image-level numeric features.", "threshold_key": "outlier_rate_warn", "interpretation": "Lower is better, unless rare edge cases are intentionally included.", "report_path": "robustness.image_feature_outliers_mad"},
    "robustness.label_conditional_outlier_rate": {"title": "RB10: Label-conditional outlier rate", "section": "Robustness", "description": "Outlier prevalence broken down by label.", "why_it_matters": "One class may carry most anomalous samples, indicating data quality asymmetry.", "method": "Computes MAD-based outliers, then aggregates rates by label.", "threshold_key": "outlier_rate_warn", "interpretation": "Large between-label differences deserve review.", "report_path": "robustness.label_conditional_outlier_rate"},
    "fairness.group_checks": {"title": "Fairness group checks", "section": "Fairness", "description": "Group-wise representation, parity, missingness, and dependence diagnostics.", "why_it_matters": "These checks reveal subgroup coverage gaps and potential data bias.", "method": "Aggregates several subgroup diagnostics over configured group columns.", "threshold_key": None, "interpretation": "Balanced coverage and small gaps are preferred.", "report_path": "fairness.group_checks"},
    "fairness.intersectional_coverage": {"title": "F12: Intersectional coverage", "section": "Fairness", "description": "Coverage of combinations across multiple group columns.", "why_it_matters": "Balanced marginal groups can still hide sparse intersections.", "method": "Measures support and entropy over multi-column subgroup combinations.", "threshold_key": None, "interpretation": "Higher coverage and support are better.", "report_path": "fairness.intersectional_coverage"},
    "transparency.datasheet_completeness": {"title": "T1: Datasheet completeness", "section": "Transparency", "description": "Coverage of core documentation fields about the dataset and analysis.", "why_it_matters": "Good documentation supports governance and reproducibility.", "method": "Checks whether expected documentation fields are populated.", "threshold_key": None, "interpretation": "Higher is better.", "report_path": "transparency.datasheet_completeness"},
    "transparency.traceability_coverage": {"title": "T5: Traceability coverage", "section": "Transparency", "description": "Coverage of identifiers, hashes, paths, and related traceability artifacts.", "why_it_matters": "Traceability helps users reproduce and audit findings.", "method": "Measures the presence of traceability-friendly metadata and integrity signals.", "threshold_key": "traceability_coverage_warn", "interpretation": "Higher is better.", "report_path": "transparency.traceability_coverage"},
    "transparency.source_attribution_coverage": {"title": "T8: Source attribution coverage", "section": "Transparency", "description": "Fraction of rows with source attribution information.", "why_it_matters": "Attribution supports licensing, provenance, and governance review.", "method": "Checks populated source columns and source metadata.", "threshold_key": "traceability_coverage_warn", "interpretation": "Higher is better.", "report_path": "transparency.source_attribution_coverage"},
    "transparency.observability_rate": {"title": "T11: Observability rate", "section": "Transparency", "description": "How much of the dataset and run configuration is observable in the report output.", "why_it_matters": "Users need to inspect assumptions and active settings.", "method": "Aggregates visible metadata, configuration, and check-registry coverage.", "threshold_key": None, "interpretation": "Higher is better.", "report_path": "transparency.observability_rate"},
    "security.integrity": {"title": "S1: Integrity", "section": "Security", "description": "Integrity information for the package and scanned files.", "why_it_matters": "Integrity checks help detect tampering and accidental corruption.", "method": "Uses SHA256 digests and scan coverage statistics.", "threshold_key": "integrity_coverage_warn", "interpretation": "Higher coverage is better.", "report_path": "security.integrity"},
    "security.suspicious_samples": {"title": "S5: Suspicious samples", "section": "Security", "description": "Heuristic suspicious-sample prevalence based on metadata and file anomalies.", "why_it_matters": "Suspicious records may indicate poisoning, malformed inputs, or packaging problems.", "method": "Aggregates heuristic indicators collected during analysis.", "threshold_key": "suspicious_sample_rate_warn", "interpretation": "Lower is better.", "report_path": "security.suspicious_samples"},
    "security.conflict_duplicate_rate": {"title": "S7: Conflict duplicate rate", "section": "Security", "description": "Rate of exact duplicate content associated with conflicting labels.", "why_it_matters": "Conflicting duplicates can point to poisoning, merge errors, or labeling instability.", "method": "Reuses duplicate groups and label conflict checks.", "threshold_key": "label_error_rate_warn", "interpretation": "Lower is better.", "report_path": "security.conflict_duplicate_rate"},
    "security.exif_privacy": {"title": "EXIF privacy indicators", "section": "Security", "description": "Presence of EXIF metadata, including GPS-bearing images.", "why_it_matters": "GPS and device data may expose sensitive information.", "method": "Samples EXIF metadata and counts privacy-relevant fields.", "threshold_key": None, "interpretation": "Fewer privacy-sensitive fields are usually preferable.", "report_path": "security.exif_privacy"},
    "security.pii_like_in_paths": {"title": "PII-like pattern scan", "section": "Security", "description": "Heuristic scan for emails, phone-like strings, IPs, and similar patterns in text fields.", "why_it_matters": "PII leakage can create legal and privacy risk.", "method": "Applies regex-based pattern matching to selected textual fields.", "threshold_key": "pii_hit_rate_threshold", "interpretation": "Lower is better.", "report_path": "security.pii_like_in_paths"},
}


def format_threshold_value(value: Any) -> str:
    if value is None:
        return "Not defined"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def metric_registry_dataframe(cfg: AssessConfig) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for key, doc in METRIC_DOCS.items():
        thr_key = doc.get("threshold_key")
        thr_val = getattr(cfg.thresholds, thr_key) if thr_key else None
        rows.append({
            "metric_key": key,
            "section": doc.get("section"),
            "title": doc.get("title"),
            "description": doc.get("description"),
            "method": doc.get("method"),
            "threshold_key": thr_key,
            "threshold_value": thr_val,
            "why_it_matters": doc.get("why_it_matters"),
            "interpretation": doc.get("interpretation"),
            "report_path": doc.get("report_path"),
        })
    return pd.DataFrame(rows)


def render_metric_doc(metric_key: str, cfg: AssessConfig) -> None:
    doc = METRIC_DOCS.get(metric_key)
    if not doc:
        return
    thr_key = doc.get("threshold_key")
    thr_val = getattr(cfg.thresholds, thr_key) if thr_key else None
    with st.expander("Method and interpretation", expanded=False):
        st.markdown(f"**Description:** {doc.get('description', 'N/A')}")
        st.markdown(f"**Why it matters:** {doc.get('why_it_matters', 'N/A')}")
        st.markdown(f"**Computation:** {doc.get('method', 'N/A')}")
        st.markdown(f"**Interpretation:** {doc.get('interpretation', 'N/A')}")
        if thr_key:
            st.markdown(f"**Active threshold:** `{thr_key}` = `{format_threshold_value(thr_val)}`")
            if thr_key in THRESHOLD_DOCS:
                st.caption(THRESHOLD_DOCS[thr_key].get('description', ''))


def render_metric_block(title: str, metric_key: str, payload: Any, cfg: AssessConfig) -> None:
    st.write(f"**{title}**")
    render_metric_doc(metric_key, cfg)
    st.json(to_json_safe(payload))


# =========================
# Column guessing for metadata
# =========================

def _name_score(name: str, patterns: List[str]) -> float:
    n = name.lower().strip()
    return sum(1.0 for p in patterns if re.search(p, n))

def guess_columns_meta(df: pd.DataFrame) -> Dict[str, Any]:
    cols = df.columns.tolist()
    nrows = max(1, len(df))
    notes: List[str] = []

    nunique = {c: int(df[c].nunique(dropna=True)) for c in cols}
    uniq_ratio = {c: float(nunique[c] / nrows) for c in cols}

    label_patterns = [r"\blabel\b", r"\btarget\b", r"\boutcome\b", r"\bclass\b", r"\bground[_\s-]?truth\b", r"\bgt\b", r"\bcategory\b"]
    split_patterns = [r"\bsplit\b", r"\bfold\b", r"\bset\b", r"\bpartition\b", r"\btrain[_\s-]?test\b"]
    time_patterns = [r"\btime\b", r"\bdate\b", r"\btimestamp\b", r"\bcreated\b", r"\bupdated\b", r"\bcaptured\b"]
    path_patterns = [r"\bpath\b", r"\bfilepath\b", r"\bfile\b", r"\bimage\b", r"\bimg\b", r"\buri\b", r"\burl\b", r"\bfilename\b"]
    id_patterns = [r"\bid\b", r"\buuid\b", r"\bguid\b", r"\brecord[_\s-]?id\b", r"\bimage[_\s-]?id\b"]
    group_patterns = [r"\bgender\b", r"\bsex\b", r"\bage\b", r"\bregion\b", r"\bcountry\b", r"\bethnicity\b", r"\brace\b", r"\bgroup\b",
                      r"\bskin[_\s-]?tone\b", r"\bphenotype\b"]
    source_patterns = [r"\bsource\b", r"\borigin\b", r"\bprovider\b", r"\bacquisition\b", r"\bcamera\b", r"\bsensor\b", r"\bdevice\b"]
    condition_patterns = [r"\bweather\b", r"\blighting\b", r"\blight\b", r"\billumination\b", r"\bblur\b", r"\bnoise\b",
                          r"\bocclusion\b", r"\bviewpoint\b", r"\bangle\b", r"\bscale\b", r"\bseason\b", r"\btime[_\s-]?of[_\s-]?day\b"]
    annotator_patterns = [r"\bannotator\b", r"\blabeler\b", r"\brater\b", r"\breview\b"]

    def rank_col(c: str, pats: List[str], extra_fn=None) -> float:
        s = _name_score(c, pats) * 3.0
        if extra_fn:
            s += extra_fn(c)
        return s

    def label_extra(c):
        s = 0.0
        if 2 <= nunique[c] <= min(100, int(0.02 * nrows) + 2):
            s += 2.0
        if uniq_ratio[c] < 0.3:
            s += 1.0
        if uniq_ratio[c] > 0.9:
            s -= 2.0
        return s

    def split_extra(c):
        s = 0.0
        if nunique[c] <= 30:
            s += 2.0
        try:
            vals = df[c].dropna().astype("string").str.lower().value_counts().head(12).index.tolist()
            joined = " ".join(vals)
            if any(x in joined for x in ["train", "test", "val", "valid", "dev", "holdout"]):
                s += 2.0
        except Exception:
            pass
        if uniq_ratio[c] > 0.5:
            s -= 2.0
        return s

    def time_extra(c):
        parsed = to_datetime_if_possible(df[c])
        if pd.api.types.is_datetime64_any_dtype(parsed.dtype):
            return float(parsed.notna().mean()) * 2.0
        return 0.0

    def path_extra(c):
        return 1.0 if uniq_ratio[c] > 0.7 else 0.0

    def id_extra(c):
        return 2.5 if uniq_ratio[c] > 0.95 else 0.0

    def group_extra(c):
        s = 0.0
        if 2 <= nunique[c] <= 50:
            s += 2.0
        if uniq_ratio[c] > 0.6:
            s -= 1.0
        return s

    def source_extra(c):
        return 1.5 if 2 <= nunique[c] <= 500 else 0.0

    best = {}
    for key, pats, extra in [
        ("label", label_patterns, label_extra),
        ("split", split_patterns, split_extra),
        ("time", time_patterns, time_extra),
        ("path", path_patterns, path_extra),
    ]:
        ranked = sorted(cols, key=lambda c: rank_col(c, pats, extra), reverse=True)
        val = ranked[0] if ranked and rank_col(ranked[0], pats, extra) >= 2.0 else None
        best[key] = val
        if val:
            notes.append(f"Guessed {key}: {val}")

    # IDs
    ids: List[str] = []
    taken = set(v for v in best.values() if v)
    ranked_id = sorted(cols, key=lambda c: rank_col(c, id_patterns, id_extra), reverse=True)
    for c in ranked_id:
        if c in taken:
            continue
        if rank_col(c, id_patterns, id_extra) >= 3.5:
            ids.append(c)
            taken.add(c)
        if len(ids) >= 3:
            break
    if ids:
        notes.append(f"Guessed ID columns: {', '.join(ids)}")

    # Groups
    groups: List[str] = []
    ranked_group = sorted(cols, key=lambda c: rank_col(c, group_patterns, group_extra), reverse=True)
    for c in ranked_group:
        if c in taken:
            continue
        if rank_col(c, group_patterns, group_extra) >= 2.5:
            groups.append(c)
            taken.add(c)
        if len(groups) >= 5:
            break
    if groups:
        notes.append(f"Guessed group columns: {', '.join(groups)}")

    # Source columns
    sources: List[str] = []
    ranked_src = sorted(cols, key=lambda c: rank_col(c, source_patterns, source_extra), reverse=True)
    for c in ranked_src:
        if c in taken:
            continue
        if rank_col(c, source_patterns, source_extra) >= 2.5:
            sources.append(c)
            taken.add(c)
        if len(sources) >= 3:
            break
    if sources:
        notes.append(f"Guessed source columns: {', '.join(sources)}")

    # Condition columns
    conditions: List[str] = []
    ranked_cond = sorted(cols, key=lambda c: _name_score(c, condition_patterns) * 3.0, reverse=True)
    for c in ranked_cond:
        if c in taken:
            continue
        if _name_score(c, condition_patterns) * 3.0 >= 2.5:
            conditions.append(c)
            taken.add(c)
        if len(conditions) >= 6:
            break
    if conditions:
        notes.append(f"Guessed condition columns: {', '.join(conditions)}")

    # Annotator columns
    annotators: List[str] = []
    ranked_ann = sorted(cols, key=lambda c: _name_score(c, annotator_patterns) * 3.0, reverse=True)
    for c in ranked_ann:
        if c in taken:
            continue
        if _name_score(c, annotator_patterns) * 3.0 >= 2.5:
            annotators.append(c)
            taken.add(c)
        if len(annotators) >= 3:
            break
    if annotators:
        notes.append(f"Guessed annotator columns: {', '.join(annotators)}")

    best["ids"] = ids
    best["groups"] = groups
    best["sources"] = sources
    best["conditions"] = conditions
    best["annotators"] = annotators
    best["notes"] = notes
    return best


# =========================
# Image scanning
# =========================

EXIF_COLUMNS = ["has_exif", "has_gps", "exif_make", "exif_model", "exif_datetime"]

def safe_open_image(data: bytes) -> Tuple[Optional[Image.Image], Optional[str]]:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img, None
    except Exception as e:
        return None, str(e)

def image_features(img: Image.Image) -> Dict[str, Any]:
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    w, h = img.size
    mode = img.mode
    if mode not in ("RGB", "L"):
        img_rgb = img.convert("RGB")
    else:
        img_rgb = img if mode == "RGB" else img.convert("RGB")
    stat = ImageStat.Stat(img_rgb)
    means = stat.mean
    stds = stat.stddev
    gray = img_rgb.convert("L")
    g = np.asarray(gray, dtype=np.float32)
    if g.shape[0] >= 3 and g.shape[1] >= 3:
        center = g[1:-1, 1:-1]
        lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * center
        blur_score = float(np.var(lap))
    else:
        blur_score = float("nan")
    hist = np.bincount(np.clip(g.astype(np.int32), 0, 255).ravel(), minlength=256).astype(np.float64)
    p = hist / max(1.0, hist.sum())
    entropy_val = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    return {
        "width": int(w), "height": int(h),
        "short_side": int(min(w, h)), "long_side": int(max(w, h)),
        "aspect_ratio": float(w / h) if h else None,
        "megapixels": float(w * h / 1e6),
        "mode": str(mode),
        "mean_r": float(means[0]), "mean_g": float(means[1]), "mean_b": float(means[2]),
        "std_r": float(stds[0]), "std_g": float(stds[1]), "std_b": float(stds[2]),
        "brightness_mean": float(np.mean(means)),
        "color_std_mean": float(np.mean(stds)),
        "blur_var_lap": blur_score,
        "entropy": entropy_val,
    }

def exif_flags(img: Image.Image) -> Dict[str, Any]:
    flags: Dict[str, Any] = {}
    try:
        exif = img.getexif()
        if exif is None or len(exif) == 0:
            return {"has_exif": False, "has_gps": False, "exif_make": None, "exif_model": None, "exif_datetime": None}
        flags["has_exif"] = True
        dt_orig = exif.get(36867) or exif.get(306)
        flags["exif_datetime"] = str(dt_orig) if dt_orig else None
        make = exif.get(271)
        model = exif.get(272)
        flags["exif_make"] = str(make) if make else None
        flags["exif_model"] = str(model) if model else None
        gps = exif.get(34853)
        flags["has_gps"] = bool(gps)
    except Exception as e:
        flags["exif_error"] = str(e)
        flags.setdefault("has_exif", False)
        flags.setdefault("has_gps", False)
        flags.setdefault("exif_make", None)
        flags.setdefault("exif_model", None)
        flags.setdefault("exif_datetime", None)
    return flags

def perceptual_hash(img: Image.Image) -> Optional[str]:
    if not IMAGEHASH_OK:
        return None
    try:
        h = imagehash.phash(ImageOps.exif_transpose(img).convert("RGB"))
        return str(h)
    except Exception:
        return None

def hamming_hexhash(a: Optional[str], b: Optional[str]) -> Optional[int]:
    if not a or not b:
        return None
    try:
        ia = int(a, 16)
        ib = int(b, 16)
        return int((ia ^ ib).bit_count())
    except Exception:
        return None


# =========================
# Annotation parsing helpers
# =========================

def _parse_annotation_json(data: bytes) -> Dict[str, Any]:
    """Attempt to parse a JSON annotation file and extract useful info."""
    try:
        obj = json.loads(data.decode("utf-8", errors="replace"))
        info: Dict[str, Any] = {"format": "json", "valid": True}
        # Detect COCO-style
        if isinstance(obj, dict):
            if "annotations" in obj and "images" in obj:
                info["style"] = "coco"
                info["num_annotations"] = len(obj.get("annotations", []))
                info["num_images_ref"] = len(obj.get("images", []))
                cats = obj.get("categories", [])
                info["num_categories"] = len(cats)
                info["category_names"] = [c.get("name", "") for c in cats[:50]] if isinstance(cats, list) else []
            elif "shapes" in obj:
                info["style"] = "labelme"
                info["num_shapes"] = len(obj.get("shapes", []))
            else:
                info["style"] = "generic_json"
                info["top_keys"] = list(obj.keys())[:20]
        elif isinstance(obj, list):
            info["style"] = "json_array"
            info["num_items"] = len(obj)
        return info
    except Exception as e:
        return {"format": "json", "valid": False, "error": str(e)[:200]}

def _parse_annotation_xml(data: bytes) -> Dict[str, Any]:
    """Attempt to parse an XML annotation file (VOC-like)."""
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(data)
        info: Dict[str, Any] = {"format": "xml", "valid": True, "root_tag": root.tag}
        if root.tag == "annotation":
            # Pascal VOC style
            info["style"] = "pascal_voc"
            objects = root.findall("object")
            info["num_objects"] = len(objects)
            names = [o.findtext("name", "") for o in objects]
            info["object_names"] = names[:50]
            # Check bbox validity
            valid_boxes = 0
            for o in objects:
                bnd = o.find("bndbox")
                if bnd is not None:
                    try:
                        xmin = float(bnd.findtext("xmin", "0"))
                        ymin = float(bnd.findtext("ymin", "0"))
                        xmax = float(bnd.findtext("xmax", "0"))
                        ymax = float(bnd.findtext("ymax", "0"))
                        if xmin < xmax and ymin < ymax:
                            valid_boxes += 1
                    except Exception:
                        pass
            info["valid_boxes"] = valid_boxes
            info["total_boxes"] = len(objects)
        else:
            info["style"] = "generic_xml"
        return info
    except Exception as e:
        return {"format": "xml", "valid": False, "error": str(e)[:200]}

def _parse_annotation_txt(data: bytes) -> Dict[str, Any]:
    """Parse a text annotation file (YOLO-style: class x_center y_center w h)."""
    try:
        lines = data.decode("utf-8", errors="replace").strip().split("\n")
        info: Dict[str, Any] = {"format": "txt", "valid": True}
        valid_lines = 0
        classes: List[str] = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cls = parts[0]
                    vals = [float(x) for x in parts[1:5]]
                    if all(0 <= v <= 1.0 for v in vals):
                        valid_lines += 1
                        classes.append(cls)
                except Exception:
                    pass
        if valid_lines > 0:
            info["style"] = "yolo"
        else:
            info["style"] = "generic_txt"
        info["num_lines"] = len(lines)
        info["valid_yolo_lines"] = valid_lines
        info["classes_found"] = list(set(classes))[:50]
        return info
    except Exception as e:
        return {"format": "txt", "valid": False, "error": str(e)[:200]}


# =========================
# Config + assessment
# =========================

@dataclass
class AssessConfig:
    path_col: Optional[str]
    label_col: Optional[str]
    split_col: Optional[str]
    time_col: Optional[str]
    group_cols: List[str]
    id_cols: List[str]
    source_cols: List[str]
    condition_cols: List[str]
    annotator_cols: List[str]

    metadata: Dict[str, Any]
    mode: str = "Quick Scan"
    thresholds: Thresholds = field(default_factory=lambda: PRESETS["Balanced (recommended)"])
    random_state: int = 7

    max_images: int = 3000
    sample_for_perceptual_dups: int = 1500
    sample_for_exif: int = 2000
    max_pairs_for_near_dups: int = 200000


# =========================
# ZIP ingestion
# =========================

def read_zip_images(zip_bytes: bytes, cfg: AssessConfig) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[str]]:
    """Read images (and annotation files) from ZIP. Returns (img_df, annotation_records, warnings)."""
    warnings: List[str] = []

    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    names = [n for n in zf.namelist() if not n.endswith("/")]

    img_names: List[str] = []
    ann_names: List[str] = []
    for n in names:
        ext = os.path.splitext(n)[1].lower()
        if ext in IMG_EXTS:
            img_names.append(n)
        elif ext in ANN_EXTS:
            ann_names.append(n)

    if not img_names:
        return pd.DataFrame(), [], ["No supported images found in ZIP."]

    if len(img_names) > cfg.max_images:
        warnings.append(f"Image count {len(img_names)} exceeds cap {cfg.max_images}. Scanning a sample.")
        rng = np.random.RandomState(cfg.random_state)
        img_names = list(rng.choice(img_names, size=cfg.max_images, replace=False))

    rows: List[Dict[str, Any]] = []
    rng = np.random.RandomState(cfg.random_state)
    sample_phash = set()
    if IMAGEHASH_OK:
        sample_n = min(len(img_names), cfg.sample_for_perceptual_dups)
        sample_phash = set(rng.choice(img_names, size=sample_n, replace=False))
    sample_exif = set(rng.choice(img_names, size=min(len(img_names), cfg.sample_for_exif), replace=False))

    # Infer folder-based labels
    folder_labels: Dict[str, str] = {}
    for n in img_names:
        parts = n.replace("\\", "/").split("/")
        if len(parts) >= 2:
            folder_labels[n] = parts[-2]

    for n in img_names:
        data = zf.read(n)
        row: Dict[str, Any] = {
            "path_in_zip": n,
            "filename": os.path.basename(n),
            "byte_size": int(len(data)),
            "sha256": sha256_bytes(data),
            "file_ext": os.path.splitext(n)[1].lower(),
        }

        # Folder label
        if n in folder_labels:
            row["folder_label"] = folder_labels[n]

        img, err = safe_open_image(data)
        if img is None:
            row["open_ok"] = False
            row["open_error"] = err
            row.update({c: None for c in EXIF_COLUMNS})
            row["phash"] = None
            rows.append(row)
            continue

        row["open_ok"] = True
        row["open_error"] = None
        row.update(image_features(img))

        if n in sample_exif:
            ex = exif_flags(img)
            for c in EXIF_COLUMNS:
                ex.setdefault(c, None)
            row.update(ex)
        else:
            row.update({c: None for c in EXIF_COLUMNS})

        if IMAGEHASH_OK and n in sample_phash:
            row["phash"] = perceptual_hash(img)
        else:
            row["phash"] = None

        rows.append(row)

    img_df = pd.DataFrame(rows)
    for c in EXIF_COLUMNS + ["phash"]:
        if c not in img_df.columns:
            img_df[c] = None

    # Parse annotation files
    annotation_records: List[Dict[str, Any]] = []
    max_ann = min(len(ann_names), 5000)
    for n in ann_names[:max_ann]:
        ext = os.path.splitext(n)[1].lower()
        try:
            data = zf.read(n)
        except Exception:
            annotation_records.append({"path": n, "format": ext, "valid": False, "error": "read_failed"})
            continue

        if ext == ".json":
            rec = _parse_annotation_json(data)
        elif ext == ".xml":
            rec = _parse_annotation_xml(data)
        elif ext == ".txt":
            rec = _parse_annotation_txt(data)
        else:
            rec = {"format": ext, "valid": True, "style": "unknown"}
        rec["path"] = n
        rec["byte_size"] = len(data)
        rec["sha256"] = sha256_bytes(data)
        annotation_records.append(rec)

    if len(ann_names) > max_ann:
        warnings.append(f"Only first {max_ann} annotation files parsed out of {len(ann_names)}.")

    return img_df, annotation_records, warnings


def load_metadata_file(uploaded_meta) -> Optional[pd.DataFrame]:
    if uploaded_meta is None:
        return None
    data = uploaded_meta.getvalue()
    name = (uploaded_meta.name or "").lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(data))
        if name.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(data))
        st.warning("Unsupported metadata type. Use CSV or Parquet.")
        return None
    except Exception as e:
        st.error(f"Failed to load metadata file: {e}")
        return None

def join_meta(img_df: pd.DataFrame, meta_df: Optional[pd.DataFrame], cfg: AssessConfig) -> Optional[pd.DataFrame]:
    if meta_df is None:
        return None
    if cfg.path_col and cfg.path_col in meta_df.columns:
        m = meta_df.copy()
        m = m.rename(columns={cfg.path_col: "path_in_zip"})
        merged = m.merge(img_df, on="path_in_zip", how="left")
        return merged
    if "filename" in meta_df.columns and "path_in_zip" in img_df.columns:
        m = meta_df.copy()
        m["_base"] = m["filename"].astype(str).apply(lambda x: os.path.basename(x))
        tmp = img_df.copy()
        tmp["_base"] = tmp["path_in_zip"].astype(str).apply(lambda x: os.path.basename(x))
        merged = m.merge(tmp.drop(columns=["path_in_zip"], errors="ignore"), on="_base", how="left")
        merged = merged.drop(columns=["_base"], errors="ignore")
        return merged
    return meta_df


# =========================
# A. QUALITY METRICS
# =========================

def _exact_duplicates(img_df: pd.DataFrame) -> Dict[str, Any]:
    ok = img_df["open_ok"].astype(bool)
    d = img_df.loc[ok, "sha256"]
    if d.empty:
        return {"exact_duplicate_rate": None, "num_unique_sha256": 0, "top_duplicate_sha256": {}}
    counts = d.value_counts()
    dup = counts[counts > 1]
    dup_files = int((d.duplicated()).sum())
    dup_rate = float(dup_files / max(1, len(d)))
    top = dup.head(10).to_dict()
    return {
        "exact_duplicate_rate": float(dup_rate),
        "exact_duplicate_files": dup_files,
        "num_unique_sha256": int(counts.shape[0]),
        "num_duplicate_sha256": int(dup.shape[0]),
        "top_duplicate_sha256": {str(k): int(v) for k, v in top.items()},
    }

def _near_duplicates_phash(img_df: pd.DataFrame, cfg: AssessConfig) -> Dict[str, Any]:
    if not IMAGEHASH_OK:
        return {"note": "imagehash not installed. Near-duplicate check disabled."}
    dfp = img_df[img_df["phash"].notna()].copy()
    if dfp.empty:
        return {"note": "No perceptual hashes available."}
    dfp["bucket"] = dfp["phash"].astype(str).str[:4]
    buckets = dfp.groupby("bucket")
    pairs_checked = 0
    near_pairs = 0
    examples: List[Dict[str, Any]] = []
    thr = int(cfg.thresholds.perceptual_dup_hamming_threshold)
    for _, g in buckets:
        if len(g) < 2:
            continue
        ph = g["phash"].astype(str).to_list()
        paths = g["path_in_zip"].astype(str).to_list()
        n = len(ph)
        for i in range(n):
            for j in range(i + 1, n):
                pairs_checked += 1
                if pairs_checked > cfg.max_pairs_for_near_dups:
                    return {
                        "perceptual_hash_rows": int(len(dfp)),
                        "pairs_checked": int(pairs_checked),
                        "perceptual_near_duplicate_pairs": int(near_pairs),
                        "hamming_threshold": thr,
                        "examples": examples[:20],
                        "note": "Pair cap reached.",
                    }
                hd = hamming_hexhash(ph[i], ph[j])
                if hd is not None and hd <= thr:
                    near_pairs += 1
                    if len(examples) < 20:
                        examples.append({"path_a": paths[i], "path_b": paths[j], "hamming": int(hd)})
    return {
        "perceptual_hash_rows": int(len(dfp)),
        "pairs_checked": int(pairs_checked),
        "perceptual_near_duplicate_pairs": int(near_pairs),
        "hamming_threshold": thr,
        "examples": examples[:20],
    }

def _cross_split_leakage(img_df: pd.DataFrame, meta_joined: Optional[pd.DataFrame], cfg: AssessConfig) -> Dict[str, Any]:
    """Q14: Cross-split leakage rate."""
    if meta_joined is None or not cfg.split_col or cfg.split_col not in meta_joined.columns:
        return {"note": "No split column. Cross-split leakage not computed."}
    tmp = meta_joined[["sha256", cfg.split_col]].copy()
    tmp = tmp[tmp["sha256"].notna() & tmp[cfg.split_col].notna()]
    if tmp.empty:
        return {"note": "No valid data for cross-split leakage."}
    distinct_splits = tmp.groupby("sha256")[cfg.split_col].nunique(dropna=False)
    leaked_hashes = distinct_splits[distinct_splits > 1]
    leaked_items = tmp[tmp["sha256"].isin(leaked_hashes.index)]
    rate = float(len(leaked_items)) / max(1, len(tmp))
    return {
        "cross_split_leakage_rate": rate,
        "leaked_unique_hashes": int(len(leaked_hashes)),
        "leaked_items": int(len(leaked_items)),
        "total_items_with_split": int(len(tmp)),
    }

def _conflicting_duplicate_labels(img_df: pd.DataFrame, meta_joined: Optional[pd.DataFrame], cfg: AssessConfig) -> Dict[str, Any]:
    """Q15: Near-duplicate images with conflicting labels."""
    if meta_joined is None or not cfg.label_col or cfg.label_col not in meta_joined.columns:
        return {"note": "No label column. Conflicting duplicate check skipped."}
    tmp = meta_joined[["sha256", cfg.label_col]].dropna()
    if tmp.empty:
        return {"note": "No data."}
    groups = tmp.groupby("sha256")[cfg.label_col].nunique()
    conflict = groups[groups > 1]
    conflict_rate = float(len(conflict)) / max(1, len(groups))
    return {
        "conflict_duplicate_rate": conflict_rate,
        "conflict_groups": int(len(conflict)),
        "total_hash_groups": int(len(groups)),
        "examples": conflict.head(10).to_dict(),
    }

def _class_balance(meta_joined: Optional[pd.DataFrame], cfg: AssessConfig, img_df: pd.DataFrame) -> Dict[str, Any]:
    """Q16-Q18: Class balance entropy, normalized entropy, effective number of classes."""
    # Try metadata label first, then folder labels
    labels = None
    source = "none"
    if meta_joined is not None and cfg.label_col and cfg.label_col in meta_joined.columns:
        labels = meta_joined[cfg.label_col].dropna()
        source = f"metadata:{cfg.label_col}"
    elif "folder_label" in img_df.columns:
        labels = img_df["folder_label"].dropna()
        source = "folder_structure"

    if labels is None or labels.empty:
        return {"note": "No label information available for class balance."}

    counts = labels.value_counts().to_dict()
    h, h_norm, n_eff = normalized_entropy(counts)

    return {
        "source": source,
        "num_classes": int(len(counts)),
        "entropy": float(h),
        "normalized_entropy": float(h_norm),
        "effective_number_of_classes": float(n_eff),
        "top_20_class_counts": dict(list(sorted(counts.items(), key=lambda kv: -kv[1]))[:20]),
        "min_class_count": int(min(counts.values())) if counts else 0,
        "max_class_count": int(max(counts.values())) if counts else 0,
    }

def _format_conformance(img_df: pd.DataFrame, allowed_formats: Optional[Set[str]] = None) -> Dict[str, Any]:
    """Q3: Format conformance rate."""
    if allowed_formats is None:
        allowed_formats = IMG_EXTS
    if "file_ext" not in img_df.columns:
        return {"note": "No file extension info."}
    conformant = img_df["file_ext"].isin(allowed_formats).sum()
    total = len(img_df)
    return {
        "format_conformance_rate": float(conformant / max(1, total)),
        "conformant": int(conformant),
        "total": total,
        "allowed_formats": sorted(allowed_formats),
        "non_conformant_formats": img_df.loc[~img_df["file_ext"].isin(allowed_formats), "file_ext"].value_counts().to_dict(),
    }

def _annotation_linkage(img_df: pd.DataFrame, annotation_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Q2: Annotation linkage rate — fraction of images with matching annotation files."""
    if not annotation_records:
        return {"annotation_linkage_rate": None, "note": "No annotation files found in ZIP."}

    img_stems = set()
    for _, row in img_df.iterrows():
        p = str(row.get("path_in_zip", ""))
        stem = os.path.splitext(os.path.basename(p))[0]
        img_stems.add(stem)

    ann_stems = set()
    for rec in annotation_records:
        p = str(rec.get("path", ""))
        stem = os.path.splitext(os.path.basename(p))[0]
        ann_stems.add(stem)

    matched = img_stems & ann_stems
    linkage_rate = float(len(matched)) / max(1, len(img_stems))
    return {
        "annotation_linkage_rate": linkage_rate,
        "images_with_annotation": int(len(matched)),
        "total_images": int(len(img_stems)),
        "total_annotation_files": int(len(annotation_records)),
        "unmatched_images_sample": sorted(list(img_stems - ann_stems))[:20],
    }

def _annotation_schema_consistency(annotation_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Q9: Schema consistency — fraction of annotations passing parsing/schema."""
    if not annotation_records:
        return {"schema_consistency_rate": None, "note": "No annotation files."}
    valid = sum(1 for r in annotation_records if r.get("valid", False))
    total = len(annotation_records)
    styles = Counter(r.get("style", "unknown") for r in annotation_records)
    return {
        "schema_consistency_rate": float(valid / max(1, total)),
        "valid_annotations": valid,
        "total_annotations": total,
        "styles_detected": dict(styles),
    }

def _bbox_validity(annotation_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Q7: Bounding-box validity rate (from parsed annotations)."""
    total_boxes = 0
    valid_boxes = 0
    for rec in annotation_records:
        if rec.get("style") == "pascal_voc":
            total_boxes += rec.get("total_boxes", 0)
            valid_boxes += rec.get("valid_boxes", 0)
        elif rec.get("style") == "yolo":
            total_boxes += rec.get("valid_yolo_lines", 0) + (rec.get("num_lines", 0) - rec.get("valid_yolo_lines", 0))
            valid_boxes += rec.get("valid_yolo_lines", 0)
    if total_boxes == 0:
        return {"bbox_validity_rate": None, "note": "No bounding boxes found in annotations."}
    return {
        "bbox_validity_rate": float(valid_boxes / max(1, total_boxes)),
        "valid_boxes": valid_boxes,
        "total_boxes": total_boxes,
    }

def _metadata_completeness(meta_joined: Optional[pd.DataFrame], required_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """Q4: Metadata completeness."""
    if meta_joined is None:
        return {"metadata_completeness": None, "note": "No metadata."}
    if required_fields is None:
        required_fields = meta_joined.columns.tolist()
    if not required_fields:
        return {"metadata_completeness": 1.0, "note": "No required fields specified."}
    present_fields = [f for f in required_fields if f in meta_joined.columns]
    if not present_fields:
        return {"metadata_completeness": 0.0, "missing_fields": required_fields}
    per_item = meta_joined[present_fields].notna().mean(axis=1)
    overall = float(per_item.mean())
    per_field = meta_joined[present_fields].notna().mean().to_dict()
    return {
        "metadata_completeness": overall,
        "per_field_completeness": per_field,
        "fields_checked": present_fields,
        "worst_5_fields": dict(sorted(per_field.items(), key=lambda kv: kv[1])[:5]),
    }

def _split_coverage(meta_joined: Optional[pd.DataFrame], cfg: AssessConfig) -> Dict[str, Any]:
    """Q5: Split assignment coverage."""
    if meta_joined is None or not cfg.split_col or cfg.split_col not in meta_joined.columns:
        return {"split_coverage": None, "note": "No split column."}
    assigned = meta_joined[cfg.split_col].notna().sum()
    total = len(meta_joined)
    splits_found = meta_joined[cfg.split_col].dropna().unique().tolist()
    return {
        "split_coverage": float(assigned / max(1, total)),
        "assigned": int(assigned),
        "total": total,
        "splits_found": sorted([str(s) for s in splits_found]),
        "split_counts": meta_joined[cfg.split_col].value_counts(dropna=False).to_dict(),
    }


def assess_quality(img_df: pd.DataFrame, meta_joined: Optional[pd.DataFrame],
                   annotation_records: List[Dict[str, Any]], cfg: AssessConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    n = int(len(img_df))

    # Q1: Readability rate
    open_ok = img_df["open_ok"].astype(bool)
    corrupt = int((~open_ok).sum())
    out["readability"] = {
        "readability_rate": float(1.0 - corrupt / max(1, n)),
        "images_scanned": n,
        "corrupt_count": corrupt,
        "corrupt_rate": float(corrupt / max(1, n)),
        "example_errors": img_df.loc[~open_ok, ["path_in_zip", "open_error"]].head(10).to_dict(orient="records"),
    }

    # Q2: Annotation linkage
    out["annotation_linkage"] = _annotation_linkage(img_df, annotation_records)

    # Q3: Format conformance
    out["format_conformance"] = _format_conformance(img_df)

    # Q4: Metadata completeness
    out["metadata_completeness"] = _metadata_completeness(meta_joined)

    # Q5: Split coverage
    out["split_coverage"] = _split_coverage(meta_joined, cfg)

    # Q7: BBox validity
    out["bbox_validity"] = _bbox_validity(annotation_records)

    # Q9: Schema consistency
    out["annotation_schema_consistency"] = _annotation_schema_consistency(annotation_records)

    # Missingness
    if meta_joined is not None and len(meta_joined) > 0:
        miss = meta_joined.isna().mean().sort_values(ascending=False)
        out["missingness"] = {
            "overall_missing_rate": float(meta_joined.isna().mean().mean()),
            "top_10_columns_missing_rate": miss.head(10).to_dict(),
        }
    else:
        out["missingness"] = {"overall_missing_rate": 0.0, "top_10_columns_missing_rate": {}}

    # Resolution
    ok_df = img_df.loc[open_ok].copy()
    if ok_df.empty:
        out["low_resolution"] = {"note": "No images opened successfully."}
        out["blur_proxy"] = {"note": "No images opened successfully."}
        out["duplicates"] = {"note": "No images opened successfully."}
        out["class_balance"] = {"note": "No images opened successfully."}
        out["cross_split_leakage"] = {"note": "No images opened successfully."}
        out["conflicting_duplicate_labels"] = {"note": "No images opened successfully."}
        return out

    min_short = int(cfg.thresholds.min_resolution_short_side)
    low_res = ok_df["short_side"].astype(float) < min_short
    out["low_resolution"] = {
        "min_short_side": min_short,
        "low_res_count": int(low_res.sum()),
        "low_res_rate": float(low_res.mean()),
        "short_side_stats": {
            "min": float(ok_df["short_side"].min()),
            "p10": float(ok_df["short_side"].quantile(0.10)),
            "median": float(ok_df["short_side"].median()),
            "p90": float(ok_df["short_side"].quantile(0.90)),
            "max": float(ok_df["short_side"].max()),
        },
    }

    # Blur proxy
    blur_thr = float(cfg.thresholds.blur_laplacian_like_threshold)
    blur = pd.to_numeric(ok_df["blur_var_lap"], errors="coerce")
    blur_low = blur < blur_thr
    out["blur_proxy"] = {
        "threshold": blur_thr,
        "low_blur_count": int(blur_low.sum(skipna=True)),
        "low_blur_rate": float(blur_low.mean(skipna=True)) if blur.notna().any() else None,
    }

    # Q12-Q13: Duplicates
    exact = _exact_duplicates(img_df)
    near = _near_duplicates_phash(img_df, cfg)
    out["duplicates"] = {
        **exact,
        "near_duplicates": near,
        "perceptual_near_duplicate_pairs": int(near.get("perceptual_near_duplicate_pairs", 0)),
        "imagehash_available": bool(IMAGEHASH_OK),
    }

    # Q14: Cross-split leakage
    out["cross_split_leakage"] = _cross_split_leakage(img_df, meta_joined, cfg)

    # Q15: Conflicting duplicate labels
    out["conflicting_duplicate_labels"] = _conflicting_duplicate_labels(img_df, meta_joined, cfg)

    # Q16-Q18: Class balance
    out["class_balance"] = _class_balance(meta_joined, cfg, img_df)

    # For shared HTML report compatibility
    out["corrupt_images"] = out["readability"]
    out["exact_duplicate_row_rate"] = exact.get("exact_duplicate_rate", 0.0)

    return out


# =========================
# B. RELIABILITY METRICS
# =========================

def _annotator_agreement(meta_joined: Optional[pd.DataFrame], cfg: AssessConfig) -> Dict[str, Any]:
    """R1-R3: Inter-annotator agreement (if multiple annotator columns present)."""
    if meta_joined is None or len(cfg.annotator_cols) < 2:
        return {"note": "Need ≥2 annotator columns for agreement computation."}
    if not SKLEARN_OK:
        return {"note": "scikit-learn required for Cohen's kappa. Not installed."}

    results: Dict[str, Any] = {}
    ann_cols = [c for c in cfg.annotator_cols if c in meta_joined.columns]
    if len(ann_cols) < 2:
        return {"note": "Fewer than 2 annotator columns found in metadata."}

    # Pairwise Cohen's kappa
    kappas: List[Dict[str, Any]] = []
    for i in range(len(ann_cols)):
        for j in range(i + 1, len(ann_cols)):
            c1, c2 = ann_cols[i], ann_cols[j]
            mask = meta_joined[c1].notna() & meta_joined[c2].notna()
            if mask.sum() < 10:
                continue
            try:
                k = cohen_kappa_score(meta_joined.loc[mask, c1], meta_joined.loc[mask, c2])
                kappas.append({"annotator_a": c1, "annotator_b": c2, "kappa": float(k), "n_items": int(mask.sum())})
            except Exception:
                pass

    if kappas:
        mean_kappa = float(np.mean([x["kappa"] for x in kappas]))
        results["pairwise_kappas"] = kappas
        results["mean_kappa"] = mean_kappa
    else:
        results["note"] = "Could not compute kappa (insufficient overlap or data)."
    return results


def assess_reliability(img_df: pd.DataFrame, meta_joined: Optional[pd.DataFrame], cfg: AssessConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # R1-R3: Annotator agreement
    out["annotator_agreement"] = _annotator_agreement(meta_joined, cfg)

    # Drift: first vs last slice
    slices = None
    if meta_joined is not None:
        if cfg.split_col and cfg.split_col in meta_joined.columns:
            slices = meta_joined[cfg.split_col].astype("string")
            out["slice_type"] = "split"
            out["slice_col"] = cfg.split_col
        elif cfg.time_col and cfg.time_col in meta_joined.columns:
            t = to_datetime_if_possible(meta_joined[cfg.time_col])
            ok = t.notna()
            if ok.sum() > 0:
                slices = t.dt.to_period("M").astype("string")
                out["slice_type"] = "time_month"
                out["slice_col"] = cfg.time_col

    if slices is not None and meta_joined is not None:
        df = meta_joined.copy()
        df["_slice"] = slices
        feat_cols = [c for c in ["short_side", "aspect_ratio", "brightness_mean", "color_std_mean", "blur_var_lap", "entropy"]
                     if c in df.columns]
        if feat_cols:
            # Missing by slice
            miss_by_slice: Dict[str, float] = {}
            for s_val, g in df.groupby("_slice", dropna=False):
                miss_by_slice[str(s_val)] = float(g[feat_cols].isna().mean().mean())
            out["missing_rate_by_slice_features"] = miss_by_slice

            uniq_sorted = sorted(map(str, df["_slice"].dropna().unique()))
            if len(uniq_sorted) >= 2:
                s_first, s_last = uniq_sorted[0], uniq_sorted[-1]
                g1 = df[df["_slice"].astype("string") == s_first]
                g2 = df[df["_slice"].astype("string") == s_last]
                drift: Dict[str, float] = {}
                for c in feat_cols:
                    d = ks_statistic(
                        pd.to_numeric(g1[c], errors="coerce").to_numpy(),
                        pd.to_numeric(g2[c], errors="coerce").to_numpy(),
                    )
                    if d is not None:
                        drift[c] = d
                out["feature_drift_ks_first_last"] = {
                    "first_slice": s_first, "last_slice": s_last,
                    "first_slice_rows": int(len(g1)), "last_slice_rows": int(len(g2)),
                    "top_10_ks": dict(sorted(drift.items(), key=lambda kv: kv[1], reverse=True)[:10]),
                }
    else:
        out["drift_note"] = "No split/time column available for drift analysis."

    # R10: Provenance coverage
    if meta_joined is not None and cfg.source_cols:
        src_cols = [c for c in cfg.source_cols if c in meta_joined.columns]
        if src_cols:
            prov_coverage = float(meta_joined[src_cols].notna().any(axis=1).mean())
            out["provenance_coverage"] = {
                "coverage_rate": prov_coverage,
                "source_columns_used": src_cols,
            }
        else:
            out["provenance_coverage"] = {"note": "Source columns not found in metadata."}
    else:
        out["provenance_coverage"] = {"note": "No source columns configured."}

    return out


# =========================
# C. ROBUSTNESS METRICS
# =========================

def assess_robustness(img_df: pd.DataFrame, meta_joined: Optional[pd.DataFrame], cfg: AssessConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    df = meta_joined if meta_joined is not None else img_df

    # RB1-RB2: Condition-bin coverage and worst-bin coverage
    condition_coverage: Dict[str, Any] = {}
    for cond_col in cfg.condition_cols:
        if cond_col not in df.columns:
            continue
        vc = df[cond_col].value_counts(dropna=True)
        total = vc.sum()
        if total == 0:
            continue
        proportions = (vc / total).to_dict()
        worst_bin = float(min(proportions.values())) if proportions else 0.0
        condition_coverage[cond_col] = {
            "bin_proportions": proportions,
            "worst_bin_coverage": worst_bin,
            "num_bins": int(len(proportions)),
        }
    out["condition_coverage"] = condition_coverage if condition_coverage else {"note": "No condition columns configured."}

    # RB3: Distribution divergence (if target distributions provided via metadata)
    # For now, compute against uniform distribution as a baseline
    if condition_coverage:
        divergence: Dict[str, Any] = {}
        for cond_col, info in condition_coverage.items():
            props = info.get("bin_proportions", {})
            if len(props) >= 2:
                uniform = {k: 1.0 / len(props) for k in props}
                jsd_val = jsd_distributions(props, uniform)
                divergence[cond_col] = {"jsd_vs_uniform": jsd_val}
        out["distribution_divergence"] = divergence

    # RB5: Sensor/domain diversity
    if meta_joined is not None and cfg.source_cols:
        for src_col in cfg.source_cols:
            if src_col in meta_joined.columns:
                vc = meta_joined[src_col].value_counts(dropna=True)
                if not vc.empty:
                    h, h_norm, n_eff = normalized_entropy(vc.to_dict())
                    out[f"source_diversity_{src_col}"] = {
                        "entropy": h, "normalized_entropy": h_norm,
                        "effective_number": n_eff,
                        "num_sources": int(len(vc)),
                        "top_10_sources": vc.head(10).to_dict(),
                    }

    # Camera/device diversity from EXIF
    if "exif_model" in img_df.columns:
        models = img_df["exif_model"].dropna()
        if not models.empty:
            vc = models.value_counts()
            h, h_norm, n_eff = normalized_entropy(vc.to_dict())
            out["camera_model_diversity"] = {
                "entropy": h, "normalized_entropy": h_norm,
                "effective_number": n_eff,
                "num_models": int(len(vc)),
                "top_10_models": vc.head(10).to_dict(),
            }

    # RB9-RB10: Outlier detection (MAD-based on image features)
    feat_cols = [c for c in ["short_side", "aspect_ratio", "brightness_mean", "color_std_mean", "blur_var_lap", "entropy"]
                 if c in df.columns]
    if feat_cols:
        X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
        med = X.median(axis=0, skipna=True)
        mad = (X - med).abs().median(axis=0, skipna=True).replace(0, np.nan)
        z = (X - med).abs().divide(mad)
        row_score = z.mean(axis=1, skipna=True)
        outlier_thr = 3.0
        outlier_mask = row_score > outlier_thr
        outlier_rate = float(outlier_mask.mean())
        out["image_feature_outliers_mad"] = {
            "features": feat_cols,
            "outlier_threshold_mad": outlier_thr,
            "outlier_rate": outlier_rate,
            "outlier_count": int(outlier_mask.sum()),
            "mean_score": float(row_score.mean(skipna=True)),
            "p95": float(row_score.quantile(0.95)),
            "p99": float(row_score.quantile(0.99)),
            "top_20_row_indices": row_score.sort_values(ascending=False).head(20).index.tolist(),
        }

        # RB10: Label-conditional outlier rate
        labels = None
        if meta_joined is not None and cfg.label_col and cfg.label_col in meta_joined.columns:
            labels = meta_joined[cfg.label_col]
        elif "folder_label" in img_df.columns:
            labels = img_df["folder_label"]
        if labels is not None:
            label_outlier: Dict[str, Any] = {}
            for lbl, grp in df.assign(_lbl=labels).groupby("_lbl", dropna=True):
                grp_scores = row_score.loc[grp.index]
                grp_outliers = (grp_scores > outlier_thr).mean()
                if len(grp) >= 5:
                    label_outlier[str(lbl)] = {"outlier_rate": float(grp_outliers), "count": int(len(grp))}
            out["label_conditional_outlier_rate"] = label_outlier
    else:
        out["image_feature_outliers_mad"] = {"note": "No numeric features available."}

    # Rare category label concentration
    if meta_joined is not None and cfg.label_col and cfg.label_col in meta_joined.columns:
        y = meta_joined[cfg.label_col]
        cat_cols = categorical_cols(meta_joined,
                                    exclude=[cfg.label_col, cfg.split_col, cfg.time_col]
                                    + (cfg.group_cols or []) + (cfg.id_cols or []))
        suspicious: List[Dict[str, Any]] = []
        for c in cat_cols[:50]:
            vc = meta_joined[c].value_counts(dropna=True)
            rare_vals = vc[vc <= max(5, int(0.001 * len(meta_joined)))].index.tolist()
            for v in rare_vals[:200]:
                mask = meta_joined[c] == v
                if int(mask.sum()) < 5:
                    continue
                dist = y[mask].value_counts(normalize=True, dropna=True)
                if len(dist) >= 1:
                    top_share = float(dist.iloc[0])
                    if top_share >= 0.95:
                        suspicious.append({
                            "column": c, "value": str(v), "count": int(mask.sum()),
                            "top_label": str(dist.index[0]), "top_label_share": top_share,
                        })
        out["rare_category_label_concentration"] = {
            "num_findings": len(suspicious),
            "top_findings": sorted(suspicious, key=lambda d: (-d["top_label_share"], -d["count"]))[:20],
        }

    return out


# =========================
# D. FAIRNESS METRICS
# =========================

def assess_fairness(img_df: pd.DataFrame, meta_joined: Optional[pd.DataFrame], cfg: AssessConfig) -> Dict[str, Any]:
    if meta_joined is None or not cfg.group_cols:
        return {"note": "Upload metadata and select group columns to compute fairness checks."}

    df = meta_joined
    out: Dict[str, Any] = {}
    label_ok = bool(cfg.label_col and cfg.label_col in df.columns)

    per_groupcol: Dict[str, Any] = {}
    for gcol in [c for c in cfg.group_cols if c in df.columns]:
        counts = df[gcol].value_counts(dropna=False)
        total = max(1, counts.sum())
        shares = counts / total
        stats: Dict[str, Any] = {
            "num_groups": int(len(counts)),
            "min_group_share": float(shares.min()) if len(shares) else None,
            "max_group_share": float(shares.max()) if len(shares) else None,
        }

        # F1: Representation ratio (against uniform)
        if len(counts) >= 2:
            uniform_share = 1.0 / len(counts)
            rep_ratios = {str(k): float(v / uniform_share) for k, v in shares.items() if pd.notna(k)}
            stats["representation_ratio_vs_uniform"] = rep_ratios
            stats["max_representation_disparity"] = float(shares.max() / max(shares.min(), 1e-12))

        # F4: Representation JSD vs uniform
        if len(counts) >= 2 and SCIPY_OK:
            observed = {str(k): float(v) for k, v in shares.items() if pd.notna(k)}
            uniform_dist = {k: 1.0 / len(observed) for k in observed}
            jsd_val = jsd_distributions(observed, uniform_dist)
            stats["representation_jsd_vs_uniform"] = jsd_val

        stats["representation_share_top10"] = shares.sort_values(ascending=False).head(10).to_dict()

        # F5: Minimum subgroup support
        stats["min_subgroup_support"] = int(counts.min()) if len(counts) else 0

        # F12: Intersectional coverage (entropy)
        h, h_norm, n_eff = normalized_entropy(counts.to_dict())
        stats["coverage_entropy"] = h
        stats["coverage_normalized_entropy"] = h_norm

        # F8: Missingness disparity on features + label
        feat_cols = [c for c in ["short_side", "aspect_ratio", "brightness_mean", "color_std_mean", "blur_var_lap", "entropy"]
                     if c in df.columns]
        miss_disp: Dict[str, float] = {}
        for c in feat_cols + ([cfg.label_col] if label_ok else []):
            mr = df.groupby(gcol)[c].apply(lambda s: s.isna().mean())
            if mr.shape[0] >= 2:
                miss_disp[c] = float(mr.max() - mr.min())
        stats["missingness_disparity_top10"] = dict(sorted(miss_disp.items(), key=lambda kv: kv[1], reverse=True)[:10])
        stats["max_missingness_disparity"] = float(max(miss_disp.values())) if miss_disp else 0.0

        if label_ok:
            y = df[cfg.label_col]
            stats["label_missingness_by_group"] = df.groupby(gcol)[cfg.label_col].apply(
                lambda s: s.isna().mean()).to_dict()

            # F6: Conditional label distribution parity
            label_dist_by_group: Dict[str, Dict[str, float]] = {}
            for grp_val, grp_df in df.groupby(gcol, dropna=True):
                ld = grp_df[cfg.label_col].value_counts(normalize=True, dropna=True)
                label_dist_by_group[str(grp_val)] = ld.to_dict()
            stats["label_distribution_by_group"] = label_dist_by_group

            # Compute max parity gap per label
            all_labels = y.dropna().unique()
            max_gap = 0.0
            parity_gaps: Dict[str, float] = {}
            for lbl in all_labels[:50]:
                rates = {}
                for grp_val, ld in label_dist_by_group.items():
                    rates[grp_val] = ld.get(lbl, 0.0)
                if len(rates) >= 2:
                    gap = max(rates.values()) - min(rates.values())
                    parity_gaps[str(lbl)] = float(gap)
                    max_gap = max(max_gap, gap)
            stats["label_parity_gaps"] = dict(sorted(parity_gaps.items(), key=lambda kv: -kv[1])[:10])
            stats["max_label_parity_gap"] = float(max_gap)

            # F11: Group-label mutual information
            if SKLEARN_OK:
                valid_mask = df[gcol].notna() & df[cfg.label_col].notna()
                if valid_mask.sum() > 10:
                    mi = mutual_info_score(df.loc[valid_mask, gcol].astype(str),
                                           df.loc[valid_mask, cfg.label_col].astype(str))
                    stats["group_label_mutual_information"] = float(mi)

            # Binary label positive-rate disparity
            if y.dropna().nunique() == 2:
                tmp = df.copy()
                tmp["_y_enc"], _ = pd.factorize(tmp[cfg.label_col])
                pos = tmp[tmp[cfg.label_col].notna()].groupby(gcol)["_y_enc"].mean()
                if len(pos) >= 2:
                    stats["positive_rate_by_group"] = pos.to_dict()
                    stats["positive_rate_disparity"] = float(pos.max() - pos.min())

        # F10: Group-context mutual information (group vs condition columns)
        if SKLEARN_OK and cfg.condition_cols:
            ctx_mi: Dict[str, float] = {}
            for ctx_col in cfg.condition_cols:
                if ctx_col in df.columns:
                    valid = df[gcol].notna() & df[ctx_col].notna()
                    if valid.sum() > 10:
                        mi = mutual_info_score(df.loc[valid, gcol].astype(str),
                                               df.loc[valid, ctx_col].astype(str))
                        ctx_mi[ctx_col] = float(mi)
            if ctx_mi:
                stats["group_context_mutual_information"] = ctx_mi

        per_groupcol[gcol] = stats

    out["group_checks"] = per_groupcol

    # Intersectional analysis (if multiple group cols)
    if len(cfg.group_cols) >= 2:
        valid_gcols = [c for c in cfg.group_cols if c in df.columns]
        if len(valid_gcols) >= 2:
            # Create intersection column
            inter = df[valid_gcols].astype(str).agg(" × ".join, axis=1)
            inter_counts = inter.value_counts(dropna=True)
            h, h_norm, n_eff = normalized_entropy(inter_counts.to_dict())
            out["intersectional_coverage"] = {
                "num_subgroups": int(len(inter_counts)),
                "min_subgroup_support": int(inter_counts.min()) if len(inter_counts) else 0,
                "coverage_entropy": h,
                "coverage_normalized_entropy": h_norm,
                "bottom_10_subgroups": inter_counts.sort_values().head(10).to_dict(),
            }

    return out


# =========================
# E. TRANSPARENCY METRICS
# =========================

def assess_transparency(img_df: pd.DataFrame, meta_joined: Optional[pd.DataFrame],
                        annotation_records: List[Dict[str, Any]], cfg: AssessConfig,
                        zip_bytes: bytes) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # T1: Datasheet completeness (heuristic)
    datasheet_sections = [
        ("purpose", bool(cfg.metadata.get("purpose"))),
        ("source_description", bool(cfg.source_cols)),
        ("annotation_process", len(annotation_records) > 0),
        ("splits_defined", cfg.split_col is not None),
        ("label_definition", cfg.label_col is not None),
        ("group_attributes", len(cfg.group_cols) > 0),
        ("condition_attributes", len(cfg.condition_cols) > 0),
        ("metadata_provided", meta_joined is not None),
        ("known_limitations", bool(cfg.metadata.get("known_limitations"))),
        ("intended_use", bool(cfg.metadata.get("intended_use"))),
    ]
    completed = sum(1 for _, ok in datasheet_sections if ok)
    out["datasheet_completeness"] = {
        "completeness_rate": float(completed / max(1, len(datasheet_sections))),
        "completed": completed,
        "total_sections": len(datasheet_sections),
        "section_status": {name: ok for name, ok in datasheet_sections},
    }

    # T5: Traceability coverage
    traceability_fields = []
    if meta_joined is not None:
        for col in cfg.source_cols + cfg.id_cols:
            if col in meta_joined.columns:
                traceability_fields.append(col)
    if traceability_fields and meta_joined is not None:
        cov = float(meta_joined[traceability_fields].notna().all(axis=1).mean())
        out["traceability_coverage"] = {
            "coverage_rate": cov,
            "traceability_fields": traceability_fields,
        }
    else:
        out["traceability_coverage"] = {"coverage_rate": None, "note": "No traceability fields (source/id cols)."}

    # T8: Source attribution coverage
    if meta_joined is not None and cfg.source_cols:
        src_cols_present = [c for c in cfg.source_cols if c in meta_joined.columns]
        if src_cols_present:
            src_cov = float(meta_joined[src_cols_present].notna().any(axis=1).mean())
            out["source_attribution_coverage"] = {"coverage_rate": src_cov, "source_columns": src_cols_present}
        else:
            out["source_attribution_coverage"] = {"note": "Source columns not found."}
    else:
        out["source_attribution_coverage"] = {"note": "No source columns configured."}

    # T11: Observability rate
    observable_attributes = [
        "class_frequencies", "acquisition_sources", "conditions",
        "subgroup_coverage", "duplicates", "corruption_profile",
    ]
    observed = 0
    if cfg.label_col:
        observed += 1  # class_frequencies
    if cfg.source_cols:
        observed += 1  # acquisition_sources
    if cfg.condition_cols:
        observed += 1  # conditions
    if cfg.group_cols:
        observed += 1  # subgroup_coverage
    observed += 1  # duplicates always computed
    observed += 1  # corruption profile always computed
    out["observability_rate"] = {
        "rate": float(observed / max(1, len(observable_attributes))),
        "observed": observed,
        "total_expected": len(observable_attributes),
    }

    # Dataset identity
    sha_zip = sha256_bytes(zip_bytes)
    out["dataset_identity"] = {
        "zip_sha256": sha_zip,
        "zip_byte_size": int(len(zip_bytes)),
        "total_images": int(len(img_df)),
        "total_annotation_files": int(len(annotation_records)),
    }

    # Check registry
    checks_ran = []
    checks_skipped = []
    check_list = [
        ("Readability rate (Q1)", True),
        ("Annotation linkage (Q2)", len(annotation_records) > 0),
        ("Format conformance (Q3)", True),
        ("Metadata completeness (Q4)", meta_joined is not None),
        ("Split coverage (Q5)", cfg.split_col is not None),
        ("Label error rate audit (Q6)", False),  # requires external audit
        ("BBox validity (Q7)", len(annotation_records) > 0),
        ("Schema consistency (Q9)", len(annotation_records) > 0),
        ("Exact duplicate rate (Q12)", True),
        ("Near-duplicate rate (Q13)", IMAGEHASH_OK),
        ("Cross-split leakage (Q14)", cfg.split_col is not None),
        ("Conflicting duplicate labels (Q15)", cfg.label_col is not None),
        ("Class balance entropy (Q16-18)", cfg.label_col is not None or "folder_label" in img_df.columns),
        ("Coverage divergence (Q20)", len(cfg.condition_cols) > 0),
        ("Annotator agreement (R1-3)", len(cfg.annotator_cols) >= 2),
        ("Feature drift KS (R)", cfg.split_col is not None or cfg.time_col is not None),
        ("Provenance coverage (R10)", len(cfg.source_cols) > 0),
        ("Condition-bin coverage (RB1-2)", len(cfg.condition_cols) > 0),
        ("Outlier rate MAD (RB9)", True),
        ("Label-conditional outlier rate (RB10)", cfg.label_col is not None),
        ("Representation ratio (F1)", len(cfg.group_cols) > 0),
        ("Representation JSD (F4)", len(cfg.group_cols) > 0),
        ("Min subgroup support (F5)", len(cfg.group_cols) > 0),
        ("Label parity gap (F6)", len(cfg.group_cols) > 0 and cfg.label_col is not None),
        ("Missingness disparity (F8)", len(cfg.group_cols) > 0),
        ("Group-context MI (F10)", len(cfg.group_cols) > 0 and len(cfg.condition_cols) > 0),
        ("Intersectional coverage (F12)", len(cfg.group_cols) >= 2),
        ("Datasheet completeness (T1)", True),
        ("Traceability coverage (T5)", len(cfg.source_cols + cfg.id_cols) > 0),
        ("Observability rate (T11)", True),
        ("Integrity coverage (S1)", True),
        ("Source concentration HHI (S8)", len(cfg.source_cols) > 0),
        ("EXIF privacy scan", True),
        ("PII pattern scan", True),
    ]
    for name, ran in check_list:
        if ran:
            checks_ran.append(name)
        else:
            checks_skipped.append(name)
    out["check_registry"] = {"ran": checks_ran, "skipped": checks_skipped,
                              "ran_count": len(checks_ran), "skipped_count": len(checks_skipped)}

    return out


# =========================
# F. SECURITY METRICS
# =========================

def assess_security(img_df: pd.DataFrame, meta_joined: Optional[pd.DataFrame],
                    cfg: AssessConfig, zip_bytes: bytes) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # S1: Integrity coverage (all files have SHA-256)
    total = len(img_df)
    has_hash = img_df["sha256"].notna().sum()
    out["integrity"] = {
        "sha256_zip": sha256_bytes(zip_bytes),
        "zip_byte_size": int(len(zip_bytes)),
        "integrity_coverage": float(has_hash / max(1, total)),
        "items_with_hash": int(has_hash),
        "total_items": total,
    }

    # S5: Suspicious sample rate (heuristic: extreme outliers + metadata inconsistencies)
    suspicious_ids: List[str] = []
    # Files that can't open are suspicious
    corrupt_mask = ~img_df["open_ok"].astype(bool)
    suspicious_ids.extend(img_df.loc[corrupt_mask, "path_in_zip"].tolist())
    # Very small files (< 100 bytes) are suspicious
    tiny_mask = img_df["byte_size"] < 100
    suspicious_ids.extend(img_df.loc[tiny_mask, "path_in_zip"].tolist())
    suspicious_ids = list(set(suspicious_ids))
    out["suspicious_samples"] = {
        "suspicious_sample_rate": float(len(suspicious_ids) / max(1, total)),
        "suspicious_count": int(len(suspicious_ids)),
        "total": total,
        "reasons": {
            "corrupt_or_unreadable": int(corrupt_mask.sum()),
            "tiny_files_under_100_bytes": int(tiny_mask.sum()),
        },
        "examples": suspicious_ids[:20],
    }

    # S7: Conflict duplicate rate (same hash, different labels)
    if meta_joined is not None and cfg.label_col and cfg.label_col in meta_joined.columns:
        tmp = meta_joined[["sha256", cfg.label_col]].dropna()
        if not tmp.empty:
            groups = tmp.groupby("sha256")[cfg.label_col].nunique()
            conflicts = groups[groups > 1]
            out["conflict_duplicate_rate"] = {
                "rate": float(len(conflicts) / max(1, len(groups))),
                "conflict_groups": int(len(conflicts)),
                "total_groups": int(len(groups)),
            }

    # S8: Source concentration index (HHI)
    if meta_joined is not None and cfg.source_cols:
        for src_col in cfg.source_cols:
            if src_col in meta_joined.columns:
                vc = meta_joined[src_col].value_counts(dropna=True)
                total_src = vc.sum()
                if total_src > 0:
                    props = vc / total_src
                    hhi = float((props ** 2).sum())
                    out[f"source_concentration_{src_col}"] = {
                        "hhi": hhi,
                        "num_sources": int(len(vc)),
                        "top_source_share": float(props.max()),
                        "top_5_sources": vc.head(5).to_dict(),
                        "interpretation": "high_concentration" if hhi > cfg.thresholds.source_concentration_hhi_warn else "acceptable",
                    }

    # EXIF privacy indicators
    gps_count = int((img_df.get("has_gps", pd.Series(dtype=bool)) == True).sum())
    exif_count = int((img_df.get("has_exif", pd.Series(dtype=bool)) == True).sum())
    out["exif_privacy"] = {
        "exif_rows_sampled": int(img_df[EXIF_COLUMNS].notna().any(axis=1).sum()),
        "exif_images_count": exif_count,
        "gps_images_count": gps_count,
        "gps_rate": float(gps_count / max(1, len(img_df))),
        "make_top10": img_df["exif_make"].dropna().astype(str).value_counts().head(10).to_dict() if "exif_make" in img_df.columns else {},
        "model_top10": img_df["exif_model"].dropna().astype(str).value_counts().head(10).to_dict() if "exif_model" in img_df.columns else {},
    }

    # S14-S17: Privacy / PII scan
    columns_with_hits: Dict[str, Dict[str, float]] = {}

    def scan_series(name: str, s: pd.Series, max_rows: int = 3000) -> None:
        s = s.dropna().astype("string")
        if s.empty:
            return
        rng = np.random.RandomState(cfg.random_state)
        if len(s) > max_rows:
            s = s.sample(n=max_rows, random_state=int(rng.randint(0, 1_000_000)))
        col_hits: Dict[str, float] = {}
        for pat_name, pat in PII_PATTERNS.items():
            rate = float(s.str.contains(pat, regex=True).mean())
            if rate >= cfg.thresholds.pii_hit_rate_threshold:
                col_hits[pat_name] = rate
        if col_hits:
            columns_with_hits[name] = col_hits

    if "path_in_zip" in img_df.columns:
        scan_series("path_in_zip", img_df["path_in_zip"], max_rows=5000)
    if meta_joined is not None:
        text_cols = categorical_cols(meta_joined, exclude=[])
        for c in text_cols[:12]:
            scan_series(f"meta:{c}", meta_joined[c], max_rows=2000)

    out["pii_like_in_paths"] = {
        "threshold_hit_rate": float(cfg.thresholds.pii_hit_rate_threshold),
        "columns_with_hits": columns_with_hits,
        "note": "Heuristic scan for PII-like patterns.",
    }
    # For HTML report compat
    out["confidentiality_pii_heuristics"] = out["pii_like_in_paths"]

    return out


# =========================
# FULL ASSESSMENT
# =========================

def assess_all(img_df: pd.DataFrame, meta_joined: Optional[pd.DataFrame],
               annotation_records: List[Dict[str, Any]],
               cfg: AssessConfig, zip_bytes: bytes) -> Dict[str, Any]:
    return {
        "quality": assess_quality(img_df, meta_joined, annotation_records, cfg),
        "reliability": assess_reliability(img_df, meta_joined, cfg),
        "robustness": assess_robustness(img_df, meta_joined, cfg),
        "fairness": assess_fairness(img_df, meta_joined, cfg),
        "transparency": assess_transparency(img_df, meta_joined, annotation_records, cfg, zip_bytes),
        "security": assess_security(img_df, meta_joined, cfg, zip_bytes=zip_bytes),
        "notes": {
            "mode": cfg.mode,
            "thresholds": {k: getattr(cfg.thresholds, k) for k in cfg.thresholds.__dataclass_fields__},
            "imagehash_available": bool(IMAGEHASH_OK),
            "sklearn_available": bool(SKLEARN_OK),
            "scipy_available": bool(SCIPY_OK),
            "metadata": cfg.metadata,
        },
    }


# =========================
# Scoring + verdict
# =========================

def compute_metric_scores(report: Dict[str, Any], cfg: AssessConfig) -> Dict[str, Any]:
    """Compute per-property scores (0-100) and overall DTI."""
    scores: Dict[str, float] = {}
    details: Dict[str, List[str]] = {}

    # QUALITY (35 points max)
    q = report.get("quality", {})
    q_score = 35.0
    q_notes = []
    readability = q.get("readability", {}).get("readability_rate", 1.0)
    if readability is not None and readability < 0.999:
        penalty = min(10, (1 - readability) * 100)
        q_score -= penalty
        q_notes.append(f"Readability {readability:.4f} (penalty {penalty:.1f})")

    dup_rate = q.get("duplicates", {}).get("exact_duplicate_rate", 0.0) or 0.0
    if dup_rate > cfg.thresholds.exact_duplicate_rate_warn:
        q_score -= min(5, dup_rate * 50)
        q_notes.append(f"Exact duplicates {dup_rate:.4f}")

    leakage = q.get("cross_split_leakage", {}).get("cross_split_leakage_rate", 0.0) or 0.0
    if leakage > cfg.thresholds.cross_split_leakage_warn:
        q_score -= min(5, leakage * 100)
        q_notes.append(f"Cross-split leakage {leakage:.4f}")

    schema_rate = q.get("annotation_schema_consistency", {}).get("schema_consistency_rate")
    if schema_rate is not None and schema_rate < cfg.thresholds.schema_consistency_warn:
        q_score -= min(3, (1 - schema_rate) * 30)
        q_notes.append(f"Schema consistency {schema_rate:.4f}")

    meta_comp = q.get("metadata_completeness", {}).get("metadata_completeness")
    if meta_comp is not None and meta_comp < cfg.thresholds.metadata_completeness_warn:
        q_score -= min(3, (1 - meta_comp) * 30)
        q_notes.append(f"Metadata completeness {meta_comp:.4f}")

    cb = q.get("class_balance", {})
    h_norm = cb.get("normalized_entropy")
    if h_norm is not None and h_norm < 0.5:
        q_score -= min(3, (1 - h_norm) * 5)
        q_notes.append(f"Class balance entropy {h_norm:.4f}")

    scores["quality"] = max(0, q_score)
    details["quality"] = q_notes

    # RELIABILITY (20 points max)
    r = report.get("reliability", {})
    r_score = 20.0
    r_notes = []
    kappa = r.get("annotator_agreement", {}).get("mean_kappa")
    if kappa is not None and kappa < cfg.thresholds.annotator_agreement_kappa_warn:
        r_score -= min(8, (cfg.thresholds.annotator_agreement_kappa_warn - kappa) * 20)
        r_notes.append(f"Mean kappa {kappa:.3f}")

    drift = r.get("feature_drift_ks_first_last", {}).get("top_10_ks", {})
    drift_flag = any(v is not None and float(v) > cfg.thresholds.drift_ks_threshold for v in drift.values())
    if drift_flag:
        r_score -= 5
        r_notes.append("Drift above threshold")

    prov = r.get("provenance_coverage", {}).get("coverage_rate")
    if prov is not None and prov < 0.9:
        r_score -= min(3, (1 - prov) * 10)
        r_notes.append(f"Provenance coverage {prov:.4f}")

    scores["reliability"] = max(0, r_score)
    details["reliability"] = r_notes

    # ROBUSTNESS (15 points max)
    ro = report.get("robustness", {})
    ro_score = 15.0
    ro_notes = []
    outlier_rate = ro.get("image_feature_outliers_mad", {}).get("outlier_rate", 0.0) or 0.0
    if outlier_rate > cfg.thresholds.outlier_rate_warn:
        ro_score -= min(5, outlier_rate * 30)
        ro_notes.append(f"Outlier rate {outlier_rate:.4f}")

    cond_cov = ro.get("condition_coverage", {})
    if isinstance(cond_cov, dict) and "note" not in cond_cov:
        for col, info in cond_cov.items():
            wb = info.get("worst_bin_coverage", 1.0)
            if wb < cfg.thresholds.worst_bin_coverage_warn:
                ro_score -= 2
                ro_notes.append(f"Worst-bin {col}: {wb:.4f}")

    scores["robustness"] = max(0, ro_score)
    details["robustness"] = ro_notes

    # FAIRNESS (15 points max)
    f = report.get("fairness", {})
    f_score = 15.0
    f_notes = []
    gc = f.get("group_checks", {})
    for gcol, stats in gc.items():
        max_disp = stats.get("max_missingness_disparity", 0.0)
        if max_disp > cfg.thresholds.missingness_disparity_warn:
            f_score -= 2
            f_notes.append(f"Missingness disparity {gcol}: {max_disp:.4f}")
        jsd_val = stats.get("representation_jsd_vs_uniform")
        if jsd_val is not None and jsd_val > 0.1:
            f_score -= min(3, jsd_val * 10)
            f_notes.append(f"Representation JSD {gcol}: {jsd_val:.4f}")
        parity = stats.get("max_label_parity_gap", 0.0)
        if parity > 0.2:
            f_score -= min(3, parity * 5)
            f_notes.append(f"Label parity gap {gcol}: {parity:.4f}")

    scores["fairness"] = max(0, f_score)
    details["fairness"] = f_notes

    # TRANSPARENCY (8 points max)
    t = report.get("transparency", {})
    t_score = 8.0
    t_notes = []
    ds_comp = t.get("datasheet_completeness", {}).get("completeness_rate", 0.0)
    t_score *= ds_comp
    if ds_comp < 0.5:
        t_notes.append(f"Datasheet completeness {ds_comp:.2f}")
    trace = t.get("traceability_coverage", {}).get("coverage_rate")
    if trace is not None and trace < cfg.thresholds.traceability_coverage_warn:
        t_score -= 2
        t_notes.append(f"Traceability {trace:.4f}")

    scores["transparency"] = max(0, t_score)
    details["transparency"] = t_notes

    # SECURITY (7 points max)
    s = report.get("security", {})
    s_score = 7.0
    s_notes = []
    gps = s.get("exif_privacy", {}).get("gps_images_count", 0)
    if gps > 0:
        s_score -= 2
        s_notes.append(f"EXIF GPS found in {gps} images")
    pii_hits = s.get("pii_like_in_paths", {}).get("columns_with_hits", {})
    if pii_hits:
        s_score -= 2
        s_notes.append("PII-like patterns detected")
    susp = s.get("suspicious_samples", {}).get("suspicious_sample_rate", 0.0)
    if susp > cfg.thresholds.suspicious_sample_rate_warn:
        s_score -= 1
        s_notes.append(f"Suspicious sample rate {susp:.4f}")

    for src_key in [k for k in s if k.startswith("source_concentration_")]:
        hhi = s[src_key].get("hhi", 0.0)
        if hhi > cfg.thresholds.source_concentration_hhi_warn:
            s_score -= 1
            s_notes.append(f"High source concentration HHI={hhi:.3f}")

    scores["security"] = max(0, s_score)
    details["security"] = s_notes

    total = round(min(100, sum(scores.values())))
    grade = "A" if total >= 90 else ("B" if total >= 80 else ("C" if total >= 70 else ("D" if total >= 60 else "F")))

    return {
        "scores": scores,
        "total": total,
        "grade": grade,
        "max_possible": {"quality": 35, "reliability": 20, "robustness": 15, "fairness": 15, "transparency": 8, "security": 7},
        "details": details,
    }


def build_recommendations(report: Dict[str, Any], cfg: AssessConfig) -> List[str]:
    recs: List[str] = []
    q = report.get("quality", {})
    if q.get("readability", {}).get("corrupt_rate", 0.0) > 0.001:
        recs.append("Corrupt images found. Remove or fix them and rescan.")
    if q.get("low_resolution", {}).get("low_res_rate", 0.0) > 0.05:
        recs.append("Many low-resolution images. Consider filtering or resizing policy.")
    if q.get("duplicates", {}).get("exact_duplicate_rate", 0.0) > cfg.thresholds.exact_duplicate_rate_warn:
        recs.append(f"Exact duplicate rate exceeds {cfg.thresholds.exact_duplicate_rate_warn:.1%}. Deduplicate.")
    if q.get("cross_split_leakage", {}).get("cross_split_leakage_rate", 0.0) > cfg.thresholds.cross_split_leakage_warn:
        recs.append("Cross-split leakage detected. Remove leaked items from test/val splits.")
    conf_dup = q.get("conflicting_duplicate_labels", {}).get("conflict_duplicate_rate", 0.0)
    if conf_dup and conf_dup > 0:
        recs.append("Conflicting labels on duplicate images. Audit and resolve.")

    cb = q.get("class_balance", {})
    h_norm = cb.get("normalized_entropy")
    if h_norm is not None and h_norm < 0.5:
        recs.append("Significant class imbalance. Consider resampling or weighting strategies.")

    r = report.get("reliability", {})
    drift = r.get("feature_drift_ks_first_last", {}).get("top_10_ks", {})
    if drift and any(v is not None and float(v) > cfg.thresholds.drift_ks_threshold for v in drift.values()):
        recs.append("Feature drift above threshold. Check pipeline changes or retrain.")

    kappa = r.get("annotator_agreement", {}).get("mean_kappa")
    if kappa is not None and kappa < cfg.thresholds.annotator_agreement_kappa_warn:
        recs.append(f"Inter-annotator agreement (κ={kappa:.3f}) below threshold. Improve annotation guidelines.")

    s = report.get("security", {})
    if s.get("exif_privacy", {}).get("gps_images_count", 0) > 0:
        recs.append("EXIF GPS present. Strip EXIF before release.")
    if s.get("pii_like_in_paths", {}).get("columns_with_hits"):
        recs.append("PII-like patterns in filenames/paths. Mask or rename assets.")

    t = report.get("transparency", {})
    ds_comp = t.get("datasheet_completeness", {}).get("completeness_rate", 0.0)
    if ds_comp < 0.6:
        recs.append("Dataset documentation is incomplete. Fill in datasheet sections.")

    if not recs:
        recs.append("No major red flags under current checks. Save the report and rerun after updates.")
    return recs


def verdict_panel(report: Dict[str, Any], cfg: AssessConfig) -> Tuple[str, str, List[str]]:
    reasons: List[str] = []
    q = report.get("quality", {})
    r = report.get("reliability", {})
    s = report.get("security", {})

    if q.get("readability", {}).get("corrupt_rate", 0.0) > 0.001:
        reasons.append("Corrupt images present.")
    if q.get("duplicates", {}).get("exact_duplicate_rate", 0.0) > cfg.thresholds.exact_duplicate_rate_warn:
        reasons.append("Exact duplicates exceed threshold.")
    if q.get("cross_split_leakage", {}).get("cross_split_leakage_rate", 0.0) > cfg.thresholds.cross_split_leakage_warn:
        reasons.append("Cross-split leakage detected.")
    drift = r.get("feature_drift_ks_first_last", {}).get("top_10_ks", {})
    if any(v is not None and float(v) > cfg.thresholds.drift_ks_threshold for v in drift.values()):
        reasons.append("Potential feature drift detected.")
    if s.get("exif_privacy", {}).get("gps_images_count", 0) > 0:
        reasons.append("EXIF GPS present.")
    if s.get("pii_like_in_paths", {}).get("columns_with_hits"):
        reasons.append("PII-like patterns detected.")
    kappa = r.get("annotator_agreement", {}).get("mean_kappa")
    if kappa is not None and kappa < cfg.thresholds.annotator_agreement_kappa_warn:
        reasons.append(f"Low annotator agreement (κ={kappa:.3f}).")

    if reasons:
        return "Needs review", "warn", reasons
    return "Looks OK (evidence-based)", "ok", ["No major red flags under current checks."]


# =========================
# Header
# =========================

def _data_uri_from_file(p: str) -> str:
    ext = os.path.splitext(p)[1].lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp", ".svg": "image/svg+xml"}.get(ext, "image/png")
    b64 = base64.b64encode(Path(p).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _find_logo_file() -> Optional[str]:
    here = Path(__file__).resolve().parent
    for c in [here / "logo.png", here / "logo.jpg", here / "assets" / "logo.png",
              here.parent / "assets" / "logo.png", here.parent / "logo.png"]:
        if c.exists() and c.is_file():
            return str(c)
    return None

_logo_path = _find_logo_file()
if _logo_path:
    try:
        _logo_html = f'<img src="{_data_uri_from_file(_logo_path)}" style="width:34px;height:34px;object-fit:contain;border-radius:8px;" />'
    except Exception:
        _logo_html = '<span style="font-size:1.8rem;">🔬</span>'
else:
    _logo_html = '<span style="font-size:1.8rem;">🔬</span>'

st.markdown(f"""
<div class="dsa-card" style="padding:24px 28px 18px 28px; margin-bottom:16px;">
  <div style="display:flex; align-items:center; gap:10px;">
    {_logo_html}
    <div>
      <h2 style="margin:0;">Image Dataset Trustworthiness Analyzer <span style="font-size:0.7em; opacity:0.6;">(Experimental)</span></h2>
      <div class="muted">Comprehensive dataset-centric metrics for Quality, Reliability, Robustness, Fairness, Transparency, and Security — aligned with EU AI Act, ISO/IEC, NIST AI RMF, ENISA.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# =========================
# Sidebar: Upload + Config
# =========================

with st.sidebar:
    st.header("Upload")
    zip_up = st.file_uploader("Images ZIP", type=["zip"],
                               help="ZIP containing images and optionally annotation files (JSON/XML/TXT).")
    meta_up = st.file_uploader("Optional metadata (CSV/Parquet)", type=["csv", "parquet"],
                                help="Tabular metadata with columns for labels, splits, groups, sources, conditions, etc.")

if zip_up is None:
    st.info("Upload a ZIP of images to start. Include annotation files inside the ZIP for annotation-level checks.")
    st.stop()

zip_bytes = zip_up.getvalue()
zip_file_name = getattr(zip_up, "name", None) or "images.zip"

meta_df_raw = load_metadata_file(meta_up)
if meta_df_raw is not None and "auto_meta_guesses_exp" not in st.session_state:
    st.session_state["auto_meta_guesses_exp"] = guess_columns_meta(meta_df_raw)
if "use_auto_meta_exp" not in st.session_state:
    st.session_state["use_auto_meta_exp"] = True

with st.sidebar:
    st.header("Run mode")
    mode = st.radio("Mode", ["Quick Scan", "Full Scan"], index=0,
                     help="Full Scan enables all metrics including near-duplicate, condition coverage, fairness, etc.")

    st.header("Thresholds")
    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0, help="Selects the threshold profile used to judge warnings and evidence strength across checks.")
    th = PRESETS[preset_name]

    with st.expander("Show all thresholds"):
        st.write({k: getattr(th, k) for k in th.__dataclass_fields__})
        thr_df = pd.DataFrame([{"threshold_key": k, "value": getattr(th, k), "description": THRESHOLD_DOCS.get(k, {}).get("description", "")} for k in th.__dataclass_fields__])
        st.dataframe(thr_df, use_container_width=True, hide_index=True)

    with st.expander("How to read the analysis"):
        st.markdown("The report combines image-level checks, metadata coverage checks, annotation parsing checks, and subgroup diagnostics. Each documented metric includes its purpose, computation note, and the active threshold when one exists.")

    st.header("Sampling")
    random_state = st.number_input("Random seed", min_value=0, max_value=10000, value=7, step=1, help="Controls reproducible sampling for capped scans.")
    max_images = st.number_input("Max images scanned", min_value=100, max_value=100000, value=3000, step=100, help="Upper bound on scanned images. Larger values improve coverage but increase run cost.")
    sample_phash = st.number_input("Max images with perceptual hash", min_value=0, max_value=100000, value=1500, step=100, help="Caps near-duplicate analysis based on perceptual hashing.")
    sample_exif = st.number_input("Max images with EXIF scan", min_value=0, max_value=100000, value=2000, step=100, help="Caps EXIF-based transparency and privacy checks.")
    max_pairs = st.number_input("Max near-dup pairs compared", min_value=1000, max_value=5000000, value=200000, step=10000, help="Upper bound on pairwise perceptual-hash comparisons.")

    st.header("Metadata columns")
    if meta_df_raw is None:
        st.caption("No metadata uploaded. Only image-only checks will run.")
        path_col = label_col = split_col = time_col = None
        id_cols: List[str] = []
        group_cols: List[str] = []
        source_cols: List[str] = []
        condition_cols: List[str] = []
        annotator_cols: List[str] = []
    else:
        use_auto = st.toggle("Use suggested columns", value=st.session_state["use_auto_meta_exp"])
        st.session_state["use_auto_meta_exp"] = use_auto
        guesses = st.session_state.get("auto_meta_guesses_exp", {})

        if use_auto and guesses.get("notes"):
            st.caption("Suggestions")
            for n in guesses["notes"]:
                st.write("• " + n)

        cols = meta_df_raw.columns.tolist()
        col_filter = st.text_input("Filter columns", value="", help="Type to filter dropdowns.")
        filtered_cols = [c for c in cols if col_filter.lower() in c.lower()] if col_filter else cols

        def pick_one(label: str, auto_value: Optional[str]) -> Optional[str]:
            options = ["(none)"] + filtered_cols
            if use_auto and auto_value in filtered_cols:
                idx = options.index(auto_value)
            else:
                idx = 0
            chosen = st.selectbox(label, options, index=idx)
            return None if chosen == "(none)" else chosen

        path_col = pick_one("Image path column", guesses.get("path"))
        label_col = pick_one("Label column", guesses.get("label"))
        split_col = pick_one("Split column", guesses.get("split"))
        time_col = pick_one("Time column", guesses.get("time"))

        default_ids = guesses.get("ids", []) if use_auto else []
        default_groups = guesses.get("groups", []) if use_auto else []
        default_sources = guesses.get("sources", []) if use_auto else []
        default_conditions = guesses.get("conditions", []) if use_auto else []
        default_annotators = guesses.get("annotators", []) if use_auto else []

        id_cols = st.multiselect("ID columns", filtered_cols, default=[c for c in default_ids if c in filtered_cols])
        group_cols = st.multiselect("Group columns (fairness)", filtered_cols,
                                     default=[c for c in default_groups if c in filtered_cols])
        source_cols = st.multiselect("Source columns (provenance)", filtered_cols,
                                      default=[c for c in default_sources if c in filtered_cols])
        condition_cols = st.multiselect("Condition columns (robustness)", filtered_cols,
                                         default=[c for c in default_conditions if c in filtered_cols])
        annotator_cols = st.multiselect("Annotator columns (reliability)", filtered_cols,
                                         default=[c for c in default_annotators if c in filtered_cols])

    st.divider()

    # Optional documentation fields
    with st.expander("Dataset documentation (optional)"):
        ds_purpose = st.text_area("Purpose / intended use", height=60, value="")
        ds_limitations = st.text_area("Known limitations", height=60, value="")
        ds_intended_use = st.text_area("Intended and non-intended uses", height=60, value="")

    run = st.button("Run analysis", type="primary", use_container_width=True)


# =========================
# Build config & scan
# =========================

md: Dict[str, Any] = {
    "zip_name": getattr(zip_up, "name", None),
    "metadata_name": getattr(meta_up, "name", None) if meta_up is not None else None,
    "purpose": ds_purpose if meta_df_raw is not None else "",
    "known_limitations": ds_limitations if meta_df_raw is not None else "",
    "intended_use": ds_intended_use if meta_df_raw is not None else "",
}

cfg = AssessConfig(
    path_col=path_col,
    label_col=label_col,
    split_col=split_col,
    time_col=time_col,
    group_cols=group_cols if meta_df_raw is not None else [],
    id_cols=id_cols if meta_df_raw is not None else [],
    source_cols=source_cols if meta_df_raw is not None else [],
    condition_cols=condition_cols if meta_df_raw is not None else [],
    annotator_cols=annotator_cols if meta_df_raw is not None else [],
    metadata=md,
    random_state=int(random_state),
    mode=mode,
    thresholds=th,
    max_images=int(max_images),
    sample_for_perceptual_dups=int(sample_phash),
    sample_for_exif=int(sample_exif),
    max_pairs_for_near_dups=int(max_pairs),
)

with st.spinner("Reading ZIP, scanning images and annotations..."):
    img_df, annotation_records, ingest_warnings = read_zip_images(zip_bytes, cfg)

if img_df.empty:
    st.error("No images scanned.")
    for w in ingest_warnings:
        st.warning(w)
    st.stop()

meta_df_joined = None
if meta_df_raw is not None:
    meta_df_joined = join_meta(img_df, meta_df_raw, cfg)


# =========================
# Preview if not run
# =========================

if not run:
    c1, c2, c3, c4, c5 = st.columns(5, gap="large")
    with c1:
        kpi("Images", f"{len(img_df):,}", "Scanned from ZIP")
    with c2:
        ok_cnt = int((img_df["open_ok"].astype(bool) == True).sum())
        kpi("Open OK", f"{ok_cnt:,}", f"Corrupt: {len(img_df) - ok_cnt:,}")
    with c3:
        kpi("Annotations", f"{len(annotation_records):,}", "JSON/XML/TXT files")
    with c4:
        kpi("ZIP SHA-256", sha256_bytes(zip_bytes)[:12] + "…", "Integrity fingerprint")
    with c5:
        kpi("Dependencies", f"{'✓' if IMAGEHASH_OK else '✗'}H {'✓' if SKLEARN_OK else '✗'}SK {'✓' if SCIPY_OK else '✗'}SP", "imagehash / sklearn / scipy")

    if ingest_warnings:
        st.warning(" | ".join([clip_text(w, 140) for w in ingest_warnings]))

    st.write("")
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Preview")
    st.caption("Image table sample")
    st.dataframe(img_df.head(50), use_container_width=True)
    if annotation_records:
        st.caption(f"Annotation files: {len(annotation_records)} parsed")
        ann_df = pd.DataFrame(annotation_records[:50])
        st.dataframe(ann_df, use_container_width=True)
    if meta_df_joined is not None:
        st.caption("Joined metadata sample")
        st.dataframe(meta_df_joined.head(50), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# =========================
# Run analysis
# =========================

with st.spinner("Running comprehensive trustworthiness checks..."):
    report = assess_all(img_df, meta_df_joined, annotation_records, cfg, zip_bytes=zip_bytes)

safe_report = to_json_safe(report)
verdict, kind, reasons = verdict_panel(report, cfg)
recs = build_recommendations(report, cfg)
scoring = compute_metric_scores(report, cfg)
total_score = scoring["total"]
grade = scoring["grade"]
prop_scores = scoring["scores"]
score_details = scoring["details"]

# Verdict card
st.markdown(f"""
<div class="verdict-card">
  <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;">
    <div>
      <div style="font-weight:700; font-size:1.1rem;">Verdict</div>
      <div style="font-size:1.3rem; font-weight:600;">{verdict}</div>
      <div class="muted" style="margin-top:4px;">
        Mode: <span class="code-pill">{cfg.mode}</span>
        &nbsp; Preset: <span class="code-pill">{preset_name}</span>
      </div>
    </div>
    <div style="display:flex; gap:8px; flex-wrap:wrap;">
      {badge(verdict, kind)}
      {badge(f"Score {total_score}/100", 'ok' if total_score>=80 else ('warn' if total_score>=60 else 'bad'))}
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

# KPI row
c1, c2, c3, c4, c5 = st.columns(5, gap="large")
with c1:
    kpi("Images", f"{len(img_df):,}", "Scanned from ZIP")
with c2:
    ok_cnt = int((img_df["open_ok"].astype(bool) == True).sum())
    kpi("Corrupt rate", f"{1.0 - ok_cnt / max(1, len(img_df)):.4f}", f"Corrupt: {len(img_df) - ok_cnt:,}")
with c3:
    kpi("Annotations", f"{len(annotation_records):,}", "Parsed")
with c4:
    kpi("Checks ran", f"{report['transparency'].get('check_registry', {}).get('ran_count', 0)}", "Metric checks")
with c5:
    kpi("ZIP SHA-256", sha256_bytes(zip_bytes)[:12] + "…", "Integrity")

st.write("")

# Property score bar
st.markdown("**Property scores**")
score_cols = st.columns(6, gap="small")
for i, (prop, max_pts) in enumerate(scoring["max_possible"].items()):
    with score_cols[i]:
        sc = prop_scores.get(prop, 0)
        pct = sc / max(1, max_pts) * 100
        color = "rgba(0,200,0,0.7)" if pct >= 80 else ("rgba(255,165,0,0.7)" if pct >= 50 else "rgba(255,0,0,0.7)")
        st.markdown(f"""
        <div style="text-align:center; font-size:0.82rem;">
          <div style="font-weight:700;">{prop.title()}</div>
          <div style="background:rgba(120,120,120,0.1); border-radius:6px; height:8px; margin:4px 0;">
            <div style="width:{pct:.0f}%; height:100%; background:{color}; border-radius:6px;"></div>
          </div>
          <div>{sc:.1f}/{max_pts}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# =========================
# TABS
# =========================

tab_overview, tab_quality, tab_reliability, tab_robustness, tab_fairness, tab_transparency, tab_security, tab_export = st.tabs(
    ["Overview", "Quality", "Reliability", "Robustness", "Fairness", "Transparency", "Security", "Export"]
)

with tab_overview:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Overview")

    q = report["quality"]
    r = report["reliability"]
    sec = report["security"]

    oc1, oc2, oc3, oc4, oc5 = st.columns(5)
    with oc1:
        st.metric("Readability", f"{q['readability']['readability_rate']:.4f}")
    with oc2:
        st.metric("Exact dup rate", f"{q.get('duplicates', {}).get('exact_duplicate_rate', 0.0):.4f}")
    with oc3:
        st.metric("Low-res rate", f"{q.get('low_resolution', {}).get('low_res_rate', 0.0):.4f}")
    with oc4:
        leakage = q.get("cross_split_leakage", {}).get("cross_split_leakage_rate", "N/A")
        st.metric("Split leakage", f"{leakage:.4f}" if isinstance(leakage, float) else str(leakage))
    with oc5:
        st.metric("GPS EXIF", str(sec["exif_privacy"]["gps_images_count"]))

    with st.expander("How to interpret the overview"):
        st.markdown("This summary condenses the strongest indicators from each property. Use the section tabs for metric definitions, computation notes, and raw evidence.")

    # Score details
    st.write("")
    with st.expander("Scoring details"):
        for prop, notes in score_details.items():
            if notes:
                st.write(f"**{prop.title()}**: {'; '.join(notes)}")
            else:
                st.write(f"**{prop.title()}**: No issues detected.")

    st.caption("Configure metadata columns, source, condition, and group columns in the sidebar to unlock more checks.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab_quality:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Quality metrics")

    q = report["quality"]

    col1, col2 = st.columns(2, gap="large")
    with col1:
        render_metric_block("Q1: Readability rate", "quality.readability", q["readability"], cfg)
    with col2:
        render_metric_block("Q3: Format conformance", "quality.format_conformance", q["format_conformance"], cfg)

    col3, col4 = st.columns(2, gap="large")
    with col3:
        render_metric_block("Q2: Annotation linkage", "quality.annotation_linkage", q["annotation_linkage"], cfg)
    with col4:
        render_metric_block("Q4: Metadata completeness", "quality.metadata_completeness", q["metadata_completeness"], cfg)

    render_metric_block("Q5: Split coverage", "quality.split_coverage", q["split_coverage"], cfg)

    render_metric_block("Q7: Bounding-box validity", "quality.bbox_validity", q["bbox_validity"], cfg)

    render_metric_block("Q9: Annotation schema consistency", "quality.annotation_schema_consistency", q["annotation_schema_consistency"], cfg)

    render_metric_block("Low resolution", "quality.low_resolution", q.get("low_resolution", {}), cfg)

    render_metric_block("Blur proxy", "quality.blur_proxy", q.get("blur_proxy", {}), cfg)

    render_metric_block("Q12-Q13: Duplicates", "quality.duplicates", q.get("duplicates", {}), cfg)

    render_metric_block("Q14: Cross-split leakage", "quality.cross_split_leakage", q.get("cross_split_leakage", {}), cfg)

    render_metric_block("Q15: Conflicting duplicate labels", "quality.conflicting_duplicate_labels", q.get("conflicting_duplicate_labels", {}), cfg)

    render_metric_doc("quality.class_balance", cfg)
    st.write("**Q16-Q18: Class balance**")
    cb = q.get("class_balance", {})
    st.json(to_json_safe(cb))
    if "top_20_class_counts" in cb:
        vc = cb["top_20_class_counts"]
        if vc:
            chart_df = pd.DataFrame(list(vc.items()), columns=["class", "count"]).sort_values("count", ascending=False)
            st.bar_chart(chart_df.set_index("class")["count"])

    st.markdown("</div>", unsafe_allow_html=True)

with tab_reliability:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Reliability metrics")

    r = report["reliability"]

    render_metric_block("R1-R3: Annotator agreement", "reliability.annotator_agreement", r.get("annotator_agreement", {}), cfg)

    render_metric_block("R10: Provenance coverage", "reliability.provenance_coverage", r.get("provenance_coverage", {}), cfg)

    if "drift_note" in r:
        st.info(r["drift_note"])

    if "missing_rate_by_slice_features" in r:
        s = pd.Series(r["missing_rate_by_slice_features"]).sort_index()
        st.write("Feature missingness by slice")
        st.line_chart(s)

    if "feature_drift_ks_first_last" in r:
        ks = r["feature_drift_ks_first_last"]["top_10_ks"]
        ks_df = pd.DataFrame(list(ks.items()), columns=["feature", "ks_stat"]).sort_values("ks_stat", ascending=False)
        render_metric_doc("reliability.feature_drift", cfg)
        st.write(f"**Feature drift (KS)** first vs last slice, threshold={cfg.thresholds.drift_ks_threshold}")
        st.dataframe(ks_df, use_container_width=True)
        if not ks_df.empty:
            st.bar_chart(ks_df.set_index("feature")["ks_stat"])

    st.markdown("</div>", unsafe_allow_html=True)

with tab_robustness:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Robustness metrics")

    ro = report["robustness"]

    st.write("**RB1-RB2: Condition-bin coverage**")
    cond_cov = ro.get("condition_coverage", {})
    if isinstance(cond_cov, dict) and "note" in cond_cov:
        st.info(cond_cov["note"])
    else:
        for col, info in cond_cov.items():
            st.write(f"*{col}*: worst-bin = {info.get('worst_bin_coverage', 'N/A'):.4f}, bins = {info.get('num_bins', 0)}")
            props = info.get("bin_proportions", {})
            if props:
                chart_df = pd.DataFrame(list(props.items()), columns=["bin", "proportion"]).sort_values("proportion", ascending=False)
                st.bar_chart(chart_df.set_index("bin")["proportion"])

    st.write("**RB3: Distribution divergence (vs uniform)**")
    st.json(to_json_safe(ro.get("distribution_divergence", {})))

    st.write("**RB5: Source/sensor diversity**")
    for key in [k for k in ro if k.startswith("source_diversity_")] + (["camera_model_diversity"] if "camera_model_diversity" in ro else []):
        st.json(to_json_safe(ro.get(key, {})))

    render_metric_block("RB9: Outlier rate (MAD)", "robustness.image_feature_outliers_mad", ro.get("image_feature_outliers_mad", {}), cfg)

    render_metric_doc("robustness.label_conditional_outlier_rate", cfg)
    st.write("**RB10: Label-conditional outlier rate**")
    lco = ro.get("label_conditional_outlier_rate", {})
    if lco:
        lco_df = pd.DataFrame([{"label": k, **v} for k, v in lco.items()]).sort_values("outlier_rate", ascending=False)
        st.dataframe(lco_df.head(20), use_container_width=True)

    st.write("**Rare category label concentration**")
    st.json(to_json_safe(ro.get("rare_category_label_concentration", {})))

    st.markdown("</div>", unsafe_allow_html=True)

with tab_fairness:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Fairness metrics")

    f_rep = report["fairness"]
    if "note" in f_rep:
        st.warning(f_rep["note"])
    else:
        render_metric_doc("fairness.group_checks", cfg)
        for gcol, stats_ in f_rep.get("group_checks", {}).items():
            st.markdown(f"#### Group column: `{gcol}`")

            col_f1, col_f2 = st.columns(2, gap="large")
            with col_f1:
                rep = stats_.get("representation_share_top10", {})
                rep_df = pd.DataFrame(list(rep.items()), columns=["group", "share"]).sort_values("share", ascending=False)
                st.write("**F1: Representation (top 10)**")
                st.dataframe(rep_df, use_container_width=True)
                if not rep_df.empty:
                    st.bar_chart(rep_df.set_index("group")["share"])

            with col_f2:
                st.write(f"**F4: Representation JSD vs uniform**: {stats_.get('representation_jsd_vs_uniform', 'N/A')}")
                st.write(f"**F5: Min subgroup support**: {stats_.get('min_subgroup_support', 'N/A')}")
                st.write(f"**Max representation disparity**: {stats_.get('max_representation_disparity', 'N/A')}")

            miss = stats_.get("missingness_disparity_top10", {})
            if miss:
                st.caption(f"Active missingness disparity warning: {cfg.thresholds.missingness_disparity_warn}")
                st.write("**F8: Missingness disparity (top 10)**")
                miss_df = pd.DataFrame(list(miss.items()), columns=["column", "max_gap"]).sort_values("max_gap", ascending=False)
                st.dataframe(miss_df, use_container_width=True)

            parity = stats_.get("label_parity_gaps", {})
            if parity:
                st.write("**F6: Label parity gaps (top 10)**")
                pg_df = pd.DataFrame(list(parity.items()), columns=["label", "gap"]).sort_values("gap", ascending=False)
                st.dataframe(pg_df, use_container_width=True)

            mi = stats_.get("group_label_mutual_information")
            if mi is not None:
                st.write(f"**F11: Group-label mutual information**: {mi:.4f}")

            ctx_mi = stats_.get("group_context_mutual_information", {})
            if ctx_mi:
                st.write("**F10: Group-context mutual information**")
                st.json(to_json_safe(ctx_mi))

            if "positive_rate_by_group" in stats_:
                pr = pd.Series(stats_["positive_rate_by_group"]).sort_index()
                st.write("Positive rate by group (binary labels)")
                st.bar_chart(pr)
                st.write({"positive_rate_disparity": stats_.get("positive_rate_disparity")})

            st.write("---")

        # Intersectional
        inter = f_rep.get("intersectional_coverage", {})
        if inter:
            render_metric_doc("fairness.intersectional_coverage", cfg)
            st.write("**F12: Intersectional coverage**")
            st.json(to_json_safe(inter))

    st.markdown("</div>", unsafe_allow_html=True)

with tab_transparency:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Transparency metrics")

    t_rep = report["transparency"]

    # Dataset identity
    st.markdown('<div class="transparency-header">📁 Dataset Identity</div>', unsafe_allow_html=True)
    di = t_rep.get("dataset_identity", {})
    identity_rows = [
        ("ZIP file", zip_file_name),
        ("Total images scanned", f"{di.get('total_images', 0):,}"),
        ("Annotation files parsed", f"{di.get('total_annotation_files', 0):,}"),
        ("ZIP file size", f"{di.get('zip_byte_size', 0) / 1024:.1f} KB"),
        ("ZIP SHA-256", di.get("zip_sha256", "N/A")),
    ]
    html = "<div>"
    for k, v in identity_rows:
        html += f'<div class="config-row"><div class="config-key">{k}</div><div class="config-val mono">{v}</div></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # T1: Datasheet completeness
    ds = t_rep.get("datasheet_completeness", {})
    render_metric_block("T1: Datasheet completeness", "transparency.datasheet_completeness", ds, cfg)

    # T5: Traceability coverage
    render_metric_block("T5: Traceability coverage", "transparency.traceability_coverage", t_rep.get("traceability_coverage", {}), cfg)

    # T8: Source attribution
    render_metric_block("T8: Source attribution coverage", "transparency.source_attribution_coverage", t_rep.get("source_attribution_coverage", {}), cfg)

    # T11: Observability rate
    render_metric_block("T11: Observability rate", "transparency.observability_rate", t_rep.get("observability_rate", {}), cfg)

    # Analysis config
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="transparency-header">⚙️ Analysis Configuration</div>', unsafe_allow_html=True)
    config_rows = [
        ("Scan mode", cfg.mode),
        ("Threshold preset", preset_name),
        ("Max images", str(cfg.max_images)),
        ("Random seed", str(cfg.random_state)),
        ("Label column", cfg.label_col or "(none)"),
        ("Split column", cfg.split_col or "(none)"),
        ("Group columns", ", ".join(cfg.group_cols) if cfg.group_cols else "(none)"),
        ("Source columns", ", ".join(cfg.source_cols) if cfg.source_cols else "(none)"),
        ("Condition columns", ", ".join(cfg.condition_cols) if cfg.condition_cols else "(none)"),
        ("Annotator columns", ", ".join(cfg.annotator_cols) if cfg.annotator_cols else "(none)"),
    ]
    html2 = "<div>"
    for k, v in config_rows:
        html2 += f'<div class="config-row"><div class="config-key">{k}</div><div class="config-val mono">{v}</div></div>'
    html2 += "</div>"
    st.markdown(html2, unsafe_allow_html=True)

    # Check registry
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="transparency-header">📋 Check Registry</div>', unsafe_allow_html=True)
    cr = t_rep.get("check_registry", {})
    ran_list = cr.get("ran", [])
    skipped_list = cr.get("skipped", [])
    check_rows = [(c, "✓ Ran") for c in ran_list] + [(c, "— Skipped") for c in skipped_list]
    check_df = pd.DataFrame(check_rows, columns=["Check", "Status"])
    st.dataframe(check_df, use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="transparency-header">📘 Metric registry</div>', unsafe_allow_html=True)
    reg_df = metric_registry_dataframe(cfg)
    st.dataframe(reg_df[["section", "title", "threshold_key", "threshold_value", "description", "method"]], use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)

with tab_security:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Security metrics")

    sec = report["security"]

    render_metric_block("S1: Integrity", "security.integrity", sec.get("integrity", {}), cfg)

    render_metric_block("S5: Suspicious samples", "security.suspicious_samples", sec.get("suspicious_samples", {}), cfg)

    if "conflict_duplicate_rate" in sec:
        render_metric_block("S7: Conflict duplicate rate", "security.conflict_duplicate_rate", sec["conflict_duplicate_rate"], cfg)

    for key in [k for k in sec if k.startswith("source_concentration_")]:
        st.write(f"**S8: Source concentration ({key.replace('source_concentration_', '')})**")
        st.json(to_json_safe(sec[key]))

    render_metric_block("EXIF privacy indicators", "security.exif_privacy", sec.get("exif_privacy", {}), cfg)

    render_metric_doc("security.pii_like_in_paths", cfg)
    st.write("**PII pattern scan**")
    pii_cols = sec.get("pii_like_in_paths", {}).get("columns_with_hits", {})
    if pii_cols:
        rows = []
        for col, hits in pii_cols.items():
            for k, v in hits.items():
                rows.append({"column": col, "pattern": k, "hit_rate": v})
        st.dataframe(pd.DataFrame(rows).sort_values("hit_rate", ascending=False), use_container_width=True)
    else:
        st.success("No PII-like patterns flagged.")

    st.markdown("</div>", unsafe_allow_html=True)

with tab_export:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Export")
    st.caption("All exports include report data, thresholds, scoring, and configuration.")

    metric_registry_df = metric_registry_dataframe(cfg)

    cfg_dict = {
        "mode": cfg.mode,
        "preset": preset_name,
        "thresholds": to_json_safe({k: getattr(cfg.thresholds, k) for k in cfg.thresholds.__dataclass_fields__}),
        "sampling": {
            "max_images": int(cfg.max_images),
            "sample_for_perceptual_dups": int(cfg.sample_for_perceptual_dups),
            "sample_for_exif": int(cfg.sample_for_exif),
            "max_pairs_for_near_dups": int(cfg.max_pairs_for_near_dups),
            "random_state": int(cfg.random_state),
        },
        "columns": {
            "path_col": cfg.path_col,
            "label_col": cfg.label_col,
            "split_col": cfg.split_col,
            "time_col": cfg.time_col,
            "group_cols": list(cfg.group_cols or []),
            "id_cols": list(cfg.id_cols or []),
            "source_cols": list(cfg.source_cols or []),
            "condition_cols": list(cfg.condition_cols or []),
            "annotator_cols": list(cfg.annotator_cols or []),
        },
        "metadata": cfg.metadata,
        "dependencies": {
            "imagehash_available": bool(IMAGEHASH_OK),
            "sklearn_available": bool(SKLEARN_OK),
            "scipy_available": bool(SCIPY_OK),
        },
    }

    st.markdown("**Metric registry**")
    st.dataframe(metric_registry_df, use_container_width=True, hide_index=True)
    st.download_button("Download metric registry (JSON)", data=json.dumps(to_json_safe(METRIC_DOCS), indent=2), file_name="image_metric_registry.json", mime="application/json", use_container_width=True)

    col_e1, col_e2, col_e3 = st.columns(3, gap="large")

    with col_e1:
        st.markdown("**JSON report**")
        json_payload = {
            "report": safe_report,
            "scoring": to_json_safe(scoring),
            "config": cfg_dict,
            "metric_registry": metric_registry_df.to_dict(orient="records"),
            "metric_docs": to_json_safe(METRIC_DOCS),
        }
        json_bytes = json.dumps(json_payload, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            "⬇ Download JSON",
            data=json_bytes,
            file_name="image_trustworthiness_report.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_e2:
        st.markdown("**Markdown summary**")
        md_lines: List[str] = [
            f"# Image Dataset Trustworthiness Report — {zip_file_name}",
            "",
            f"- **Mode:** {cfg.mode}",
            f"- **Preset:** {preset_name}",
            f"- **Health score:** {total_score}/100 (Grade {grade})",
            f"- **Verdict:** {verdict}",
            "",
            "## Property scores",
            *[f"- **{prop.title()}**: {prop_scores[prop]:.1f}/{scoring['max_possible'][prop]}" for prop in scoring['max_possible']],
            "",
            "## Findings",
            *[f"- {r}" for r in reasons],
            "",
            "## Recommended actions",
            *[f"- {r}" for r in recs],
            "",
            "## Key metrics",
            f"- Images scanned: {len(img_df):,}",
            f"- Annotation files: {len(annotation_records):,}",
            f"- Readability rate: {report['quality']['readability']['readability_rate']:.4f}",
            f"- Exact duplicate rate: {report['quality'].get('duplicates', {}).get('exact_duplicate_rate', 0.0):.4f}",
            f"- Low-res rate: {report['quality'].get('low_resolution', {}).get('low_res_rate', 0.0):.4f}",
            f"- ZIP SHA-256: {sha256_bytes(zip_bytes)}",
            "",
            "## Regulatory alignment",
            "- EU AI Act: Art. 10 (data governance), Art. 13 (transparency), Art. 15 (robustness/cybersecurity)",
            "- Standards: ISO/IEC 5259, TR 24027, 23894, 42001, NIST AI RMF, ENISA",
            "",
            "---",
            "*Heuristic and statistical report. Validate with domain and legal review.*",
        ]
        md_bytes = "\n".join(md_lines).encode("utf-8")
        st.download_button(
            "⬇ Download Markdown",
            data=md_bytes,
            file_name="image_trustworthiness_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col_e3:
        st.markdown("**HTML report**")
        if HAS_UTILS:
            try:
                report_for_html = report.copy()
                sec0 = report_for_html.get("security", {})
                if isinstance(sec0, dict) and "confidentiality_pii_heuristics" not in sec0:
                    sec0["confidentiality_pii_heuristics"] = sec0.get("pii_like_in_paths", {})
                    report_for_html["security"] = sec0
                html_content = build_html_report(
                    df=img_df, report=report_for_html, cfg_dict=cfg_dict,
                    file_name=zip_file_name, file_bytes=zip_bytes,
                    verdict=verdict, reasons=reasons, recs=recs,
                    score=total_score, grade=grade,
                )
                st.download_button(
                    "⬇ Download HTML",
                    data=html_content.encode("utf-8"),
                    file_name="image_trustworthiness_report.html",
                    mime="text/html",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"HTML export unavailable: {e}")
        else:
            st.info("HTML export requires shared utils module.")

    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("Raw JSON (preview)"):
        st.json(json_payload)

    st.markdown("</div>", unsafe_allow_html=True)
