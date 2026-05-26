"""
Shared utilities for Unified Dataset Safety Analyzer.
Import in each page with:
    import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils import *
"""
from __future__ import annotations

import hashlib
import html
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import streamlit as st
except Exception:  # pragma: no cover - enables headless experiment imports
    st = None

# ─────────────────────────────────────────────
# CSS — shared design system
# ─────────────────────────────────────────────

SHARED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Sora:wght@300;400;600;700;800&display=swap');

/* Root tokens */
:root {
  --c-good:   #22c55e;
  --c-warn:   #f59e0b;
  --c-bad:    #ef4444;
  --c-muted:  rgba(148,163,184,0.9);
  --c-border: rgba(148,163,184,0.15);
  --c-card:   rgba(255,255,255,0.03);
  --c-card-hover: rgba(255,255,255,0.06);
  --radius:   16px;
  --font-mono: 'IBM Plex Mono', ui-monospace, monospace;
  --font-main: 'Sora', system-ui, sans-serif;
}

/* Typography */
body, .stApp { font-family: var(--font-main) !important; }
h1 { font-weight: 800 !important; letter-spacing: -0.03em !important; }
h2 { font-weight: 700 !important; letter-spacing: -0.02em !important; }
h3 { font-weight: 600 !important; letter-spacing: -0.015em !important; }

/* Layout */
.block-container { padding-top: 1rem; padding-bottom: 4rem; max-width: 1280px !important; }

/* Shared card */
.dsa-card {
  border: 1px solid var(--c-border);
  border-radius: var(--radius);
  padding: 20px 22px 14px 22px;
  background: var(--c-card);
  margin-bottom: 12px;
  transition: background 0.2s;
}
.dsa-card:hover { background: var(--c-card-hover); }

/* KPI cards */
.kpi-card {
  border: 1px solid var(--c-border);
  border-radius: 14px;
  padding: 16px 18px;
  background: var(--c-card);
  height: 100%;
}
.kpi-title  { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.6; }
.kpi-value  { font-size: 1.7rem; font-weight: 800; margin: 4px 0; letter-spacing: -0.02em; }
.kpi-hint   { font-size: 0.78rem; color: var(--c-muted); }

/* Badges */
.badge {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 11px; border-radius: 999px;
  font-size: 0.8rem; font-weight: 600; font-family: var(--font-mono);
  border: 1px solid var(--c-border);
}
.badge-ok   { background: rgba(34,197,94,0.12);  color: #22c55e; border-color: rgba(34,197,94,0.3); }
.badge-warn { background: rgba(245,158,11,0.12); color: #f59e0b; border-color: rgba(245,158,11,0.3); }
.badge-bad  { background: rgba(239,68,68,0.12);  color: #ef4444; border-color: rgba(239,68,68,0.3); }
.badge-info { background: rgba(99,102,241,0.12); color: #818cf8; border-color: rgba(99,102,241,0.3); }

/* Check status cards in overview */
.check-card {
  border: 1px solid var(--c-border);
  border-radius: 12px;
  padding: 14px 16px;
  background: var(--c-card);
  text-align: center;
}
.check-icon { font-size: 1.6rem; margin-bottom: 4px; }
.check-label { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; opacity: 0.65; margin-bottom: 6px; }
.check-score { font-size: 1.2rem; font-weight: 700; }
.check-good  { border-color: rgba(34,197,94,0.25);  background: rgba(34,197,94,0.04); }
.check-warn  { border-color: rgba(245,158,11,0.25); background: rgba(245,158,11,0.04); }
.check-bad   { border-color: rgba(239,68,68,0.25);  background: rgba(239,68,68,0.04); }

/* Health score ring */
.health-ring-container {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; gap: 8px;
}
.health-ring-label { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.5; }
.health-grade { font-size: 0.85rem; font-weight: 600; font-family: var(--font-mono); }

/* Progress bar */
.prog-row { display: flex; align-items: center; gap: 10px; margin: 5px 0; }
.prog-label { font-size: 0.82rem; flex: 0 0 140px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-family: var(--font-mono); }
.prog-bar { flex: 1; height: 6px; background: rgba(148,163,184,0.12); border-radius: 3px; overflow: hidden; }
.prog-fill { height: 100%; border-radius: 3px; transition: width 0.3s ease; }
.prog-val  { font-size: 0.78rem; font-family: var(--font-mono); flex: 0 0 52px; text-align: right; }

/* Code / mono */
.mono { font-family: var(--font-mono); font-size: 0.82rem; }
.code-pill {
  font-family: var(--font-mono); font-size: 0.8rem;
  padding: 2px 8px; border-radius: 6px;
  background: rgba(148,163,184,0.1); border: 1px solid var(--c-border);
}

/* Verdict banner */
.verdict-card {
  border-radius: var(--radius);
  padding: 20px 24px;
  margin-bottom: 16px;
  border: 1px solid var(--c-border);
  background: var(--c-card);
}
.verdict-title { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.5; }
.verdict-text  { font-size: 1.1rem; font-weight: 700; margin: 4px 0; }

/* Schema table */
.schema-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; font-family: var(--font-mono); }
.schema-table th { text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.08em; padding: 8px 12px; border-bottom: 1px solid var(--c-border); text-align: left; opacity: 0.6; }
.schema-table td { padding: 7px 12px; border-bottom: 1px solid var(--c-border); }
.schema-table tr:last-child td { border-bottom: none; }
.schema-table tr:hover td { background: var(--c-card-hover); }

/* Transparency sections */
.transparency-section { margin-bottom: 24px; }
.transparency-header {
  font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.1em; opacity: 0.5; margin-bottom: 10px;
  padding-bottom: 6px; border-bottom: 1px solid var(--c-border);
}
.config-row { display: flex; align-items: center; gap: 12px; padding: 7px 0; border-bottom: 1px solid var(--c-border); font-size: 0.84rem; }
.config-key { flex: 0 0 200px; opacity: 0.6; }
.config-val { font-family: var(--font-mono); font-weight: 500; }

/* Divider */
hr { border: none; height: 1px; background: var(--c-border); margin: 1rem 0; }

/* Small muted text */
.muted { font-size: 0.8rem; color: var(--c-muted); }

/* ── Hover lift for cards ─────────────────────────────────── */
.dsa-card {
  transition: background 0.2s, transform 0.18s ease, box-shadow 0.18s ease;
}
.dsa-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}

/* ── Feature cards with coloured top-border accent ───────── */
.feature-card {
  border-top-width: 3px !important;
  border-top-style: solid !important;
}
.feature-card-blue   { border-top-color: #3b82f6 !important; }
.feature-card-purple { border-top-color: #a855f7 !important; }
.feature-card-orange { border-top-color: #f97316 !important; }

/* ── Gradient accent line (animated) ─────────────────────── */
@keyframes gradientShift {
  0%   { background-position: 0%   50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0%   50%; }
}
.gradient-accent {
  height: 3px;
  border-radius: 2px;
  background: linear-gradient(90deg, #3b82f6, #a855f7, #f97316, #3b82f6);
  background-size: 300% 300%;
  animation: gradientShift 4s ease infinite;
  margin: 8px 0 18px 0;
}

/* ── Step circles in "How it works" ──────────────────────── */
.step-circle {
  display: inline-flex; align-items: center; justify-content: center;
  width: 40px; height: 40px; border-radius: 50%;
  font-size: 1.1rem; font-weight: 800;
  background: rgba(99,102,241,0.18);
  border: 2px solid rgba(99,102,241,0.35);
  margin-bottom: 8px;
}

/* ── Open-analyzer hint link ──────────────────────────────── */
.open-hint {
  display: block; margin-top: 14px;
  font-size: 0.82rem; font-weight: 600;
  color: rgba(148,163,184,0.75);
  letter-spacing: 0.02em;
}
.open-hint:hover { color: #818cf8; }

/* ── Footer ───────────────────────────────────────────────── */
.dsa-footer {
  border-top: 1px solid var(--c-border);
  margin-top: 36px;
  padding-top: 20px;
  text-align: center;
  font-size: 0.78rem;
  color: var(--c-muted);
  line-height: 1.9;
}
.dsa-footer a { color: rgba(148,163,184,0.85); text-decoration: none; }
.dsa-footer a:hover { color: #818cf8; }

/* ── Responsive feature columns ──────────────────────────── */
@media (max-width: 768px) {
  .feature-cols { grid-template-columns: 1fr !important; }
}
</style>
"""


# ─────────────────────────────────────────────
# JSON safety
# ─────────────────────────────────────────────

def to_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
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


# ─────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def clip_text(s: str, n: int = 80) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + "…"


def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for c in df.columns:
        dt = df[c].dtype
        if pd.api.types.is_numeric_dtype(dt):
            out[c] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(dt):
            out[c] = "datetime"
        elif pd.api.types.is_bool_dtype(dt):
            out[c] = "boolean"
        else:
            out[c] = "categorical"
    return out


def to_datetime_if_possible(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return s
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return s


def numeric_cols(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    ex = set(exclude or [])
    return [c for c in df.columns if c not in ex and pd.api.types.is_numeric_dtype(df[c].dtype)]


def categorical_cols(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    ex = set(exclude or [])
    cols: List[str] = []
    for c in df.columns:
        if c in ex:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c].dtype):
            continue
        if pd.api.types.is_numeric_dtype(df[c].dtype):
            continue
        cols.append(c)
    return cols


def approx_iqr_outlier_rate(x: pd.Series) -> Optional[float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < 8:
        return None
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return float(((x < lo) | (x > hi)).mean())


try:  # SciPy is preferred (faster, returns p-value, handles ties)
    from scipy.stats import ks_2samp as _scipy_ks_2samp  # type: ignore
    _SCIPY_KS_OK = True
except Exception:  # pragma: no cover - optional dependency
    _scipy_ks_2samp = None  # type: ignore
    _SCIPY_KS_OK = False


def ks_statistic(x1: np.ndarray, x2: np.ndarray) -> Optional[float]:
    """Two-sample Kolmogorov–Smirnov statistic.

    Uses ``scipy.stats.ks_2samp`` when available (faster, more accurate on ties);
    falls back to a NumPy implementation when SciPy is not installed.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]
    if len(x1) < 20 or len(x2) < 20:
        return None
    if _SCIPY_KS_OK:
        try:
            return float(_scipy_ks_2samp(x1, x2).statistic)
        except Exception:
            pass  # fall through to numpy implementation
    x1, x2 = np.sort(x1), np.sort(x2)
    all_vals = np.sort(np.unique(np.concatenate([x1, x2])))
    cdf1 = np.searchsorted(x1, all_vals, side="right") / len(x1)
    cdf2 = np.searchsorted(x2, all_vals, side="right") / len(x2)
    return float(np.max(np.abs(cdf1 - cdf2)))


def ks_statistic_with_pvalue(x1: np.ndarray, x2: np.ndarray) -> Optional[Tuple[float, Optional[float]]]:
    """Return (statistic, p_value) for two-sample KS. p_value is None without SciPy."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]
    if len(x1) < 20 or len(x2) < 20:
        return None
    if _SCIPY_KS_OK:
        try:
            res = _scipy_ks_2samp(x1, x2)
            return float(res.statistic), float(res.pvalue)
        except Exception:
            pass
    stat = ks_statistic(x1, x2)
    return (stat, None) if stat is not None else None


# ─────────────────────────────────────────────
# PII patterns
# ─────────────────────────────────────────────

PII_PATTERNS = {
    "email":           re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone_like":      re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}\b"),
    "ip_v4":           re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "credit_card_like":re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
}


# ─────────────────────────────────────────────
# UI primitives
# ─────────────────────────────────────────────

def badge(label: str, kind: str) -> str:
    icons = {"ok": "✓", "warn": "⚠", "bad": "✕", "info": "ℹ"}
    cls_map = {"ok": "badge-ok", "warn": "badge-warn", "bad": "badge-bad", "info": "badge-info"}
    icon = icons.get(kind, "")
    cls = cls_map.get(kind, "badge-info")
    return f'<span class="badge {cls}">{icon} {label}</span>'


def kpi(title: str, value: str, hint: str = "", color: str = "") -> None:
    color_style = f"color:{color};" if color else ""
    st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-title">{title}</div>
  <div class="kpi-value" style="{color_style}">{value}</div>
  <div class="kpi-hint">{hint}</div>
</div>""", unsafe_allow_html=True)


def health_ring_html(score: int, grade: str) -> str:
    """SVG ring showing health score."""
    circumference = 251.3
    fill = max(0, min(1, score / 100)) * circumference
    if score >= 80:
        color = "#22c55e"
    elif score >= 60:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    return f"""
<div class="health-ring-container">
  <div class="health-ring-label">Health Score</div>
  <div style="position:relative;width:130px;height:130px;">
    <svg viewBox="0 0 100 100" style="width:130px;height:130px;transform:rotate(-90deg)">
      <circle cx="50" cy="50" r="40" fill="none" stroke="rgba(148,163,184,0.12)" stroke-width="7"/>
      <circle cx="50" cy="50" r="40" fill="none" stroke="{color}" stroke-width="7"
              stroke-dasharray="{fill:.1f} {circumference:.1f}" stroke-linecap="round"/>
    </svg>
    <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;">
      <div style="font-size:2rem;font-weight:800;line-height:1;color:{color}">{score}</div>
      <div style="font-size:0.7rem;opacity:0.5;margin-top:2px;">/ 100</div>
    </div>
  </div>
  <div class="health-grade" style="color:{color}">Grade {grade}</div>
</div>"""


def progress_bar_html(label: str, value: float, max_val: float = 1.0,
                      reverse: bool = False, fmt: str = ".1%") -> str:
    """Single colored progress bar row."""
    ratio = max(0.0, min(1.0, value / max_val if max_val else 0))
    if reverse:
        color = "#22c55e" if ratio < 0.33 else ("#f59e0b" if ratio < 0.66 else "#ef4444")
    else:
        color = "#ef4444" if ratio < 0.33 else ("#f59e0b" if ratio < 0.66 else "#22c55e")
    pct = ratio * 100
    val_str = format(value, fmt) if fmt == ".1%" else f"{value:{fmt}}"
    return f"""
<div class="prog-row">
  <div class="prog-label mono" title="{label}">{clip_text(label, 22)}</div>
  <div class="prog-bar"><div class="prog-fill" style="width:{pct:.1f}%;background:{color}"></div></div>
  <div class="prog-val">{val_str}</div>
</div>"""


def check_status_card(label: str, icon: str, status: str, detail: str) -> str:
    """One of the 5 dimension check cards."""
    cls = {"ok": "check-good", "warn": "check-warn", "bad": "check-bad"}.get(status, "")
    color = {"ok": "#22c55e", "warn": "#f59e0b", "bad": "#ef4444"}.get(status, "gray")
    return f"""
<div class="check-card {cls}">
  <div class="check-icon">{icon}</div>
  <div class="check-label">{label}</div>
  <div class="check-score" style="color:{color}">{detail}</div>
</div>"""


# ─────────────────────────────────────────────
# Health score
# ─────────────────────────────────────────────

DEFAULT_WEIGHTS: Dict[str, float] = {
    "quality":     35,
    "security":    25,
    "reliability": 20,
    "robustness":  10,
    "fairness":    10,
}


EU_AI_ACT_SOURCE_URL = "https://eur-lex.europa.eu/eli/reg/2024/1689/"

EU_AI_ACT_ARTICLE_REFERENCES: Dict[str, Dict[str, str]] = {
    "Article 9": {
        "title": "Risk management system",
        "aspect": "Identifying, evaluating, and mitigating reasonably foreseeable risks.",
    },
    "Article 10": {
        "title": "Data and data governance",
        "aspect": "Training, validation, and testing data quality, relevance, representativeness, completeness, and bias controls.",
    },
    "Article 11": {
        "title": "Technical documentation",
        "aspect": "Technical evidence that explains system design, testing, limitations, and data assumptions.",
    },
    "Article 12": {
        "title": "Record-keeping",
        "aspect": "Logs and records that support traceability and post-hoc review.",
    },
    "Article 13": {
        "title": "Transparency and provision of information to deployers",
        "aspect": "Clear information about capabilities, limitations, intended use, input data, and expected performance.",
    },
    "Article 15": {
        "title": "Accuracy, robustness and cybersecurity",
        "aspect": "Evidence about accuracy-related data risks, robustness, stability, and security-relevant signals.",
    },
}

ISO_25012_SOURCE_URL = "https://www.iso.org/standard/35736.html"

ISO_25012_CHARACTERISTICS: Dict[str, Dict[str, str]] = {
    "Accuracy": {
        "perspective": "Inherent",
        "aspect": "Degree to which data correctly represent the intended real-world values.",
    },
    "Completeness": {
        "perspective": "Inherent",
        "aspect": "Degree to which expected data values are present.",
    },
    "Consistency": {
        "perspective": "Inherent",
        "aspect": "Degree to which data are free from contradiction across records, fields, or rules.",
    },
    "Credibility": {
        "perspective": "Inherent",
        "aspect": "Degree to which data are regarded as trustworthy and authentic.",
    },
    "Currentness": {
        "perspective": "Inherent",
        "aspect": "Degree to which data are sufficiently up to date for their intended use.",
    },
    "Accessibility": {
        "perspective": "System-dependent",
        "aspect": "Degree to which data can be accessed in the expected context of use.",
    },
    "Compliance": {
        "perspective": "Both",
        "aspect": "Degree to which data meet applicable rules, standards, or conventions.",
    },
    "Confidentiality": {
        "perspective": "Both",
        "aspect": "Degree to which data are protected from unauthorized disclosure.",
    },
    "Efficiency": {
        "perspective": "Both",
        "aspect": "Degree to which data can be processed using appropriate resources.",
    },
    "Precision": {
        "perspective": "Both",
        "aspect": "Degree to which data have the exactness needed for their intended use.",
    },
    "Traceability": {
        "perspective": "Both",
        "aspect": "Degree to which data access, changes, origin, and processing can be traced.",
    },
    "Understandability": {
        "perspective": "Both",
        "aspect": "Degree to which data meaning and context are clear to users.",
    },
    "Availability": {
        "perspective": "System-dependent",
        "aspect": "Degree to which data are available for authorized use when required.",
    },
    "Portability": {
        "perspective": "System-dependent",
        "aspect": "Degree to which data can be moved or reused across environments.",
    },
    "Recoverability": {
        "perspective": "System-dependent",
        "aspect": "Degree to which data can be restored and verified after failure.",
    },
}


def compute_health_score(
    report: Dict[str, Any],
    drift_threshold: float = 0.30,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[int, str, Dict[str, float]]:
    """
    Returns (score 0-100, grade, component breakdown).
    Default weights: Quality 35%, Security 25%, Reliability 20%, Robustness 10%, Fairness 10%.
    Pass a custom ``weights`` dict to override; values are normalised automatically so they
    do not need to sum to exactly 100.
    """
    effective_weights = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        for dim, value in weights.items():
            if dim in effective_weights:
                try:
                    effective_weights[dim] = max(0.0, float(value))
                except (TypeError, ValueError):
                    effective_weights[dim] = 0.0

    # Normalise the complete validated weight map so partial custom weights
    # cannot accidentally reintroduce default points after normalisation.
    w_sum = sum(effective_weights.values())
    if w_sum <= 0:
        effective_weights = dict(DEFAULT_WEIGHTS)
        w_sum = sum(effective_weights.values())
    norm_w: Dict[str, float] = {k: v / w_sum * 100.0 for k, v in effective_weights.items()}

    components: Dict[str, float] = {}

    # Quality
    q = report.get("quality", {})
    miss = float(q.get("missingness", {}).get("overall_missing_rate", 0))
    dup  = float(q.get("duplicates", {}).get("exact_duplicate_row_rate", 0))
    leak = q.get("split_leakage", {}).get("row_hash_cross_split_rate", None)
    q_scores = [
        max(0.0, 1 - miss / 0.20),   # 0% miss → 1.0; 20%+ → 0.0
        max(0.0, 1 - dup / 0.10),    # 0% dup → 1.0; 10%+ → 0.0
    ]
    if leak is not None:
        q_scores.append(0.0 if float(leak) > 0 else 1.0)
    components["quality"] = (sum(q_scores) / len(q_scores)) * norm_w["quality"]

    # Security — graduated by worst PII hit rate observed across columns.
    # Previously a single hit (even a regex false-positive) zeroed the
    # dimension. Now: 0% hit → full credit; >= PII_SEVERE_HIT_RATE → 0;
    # linear in between. This avoids cliff effects from noisy heuristics
    # while still penalising any real signal proportionally.
    s = report.get("security", {})
    pii_hits = s.get("confidentiality_pii_heuristics", {}).get("columns_with_hits", {})
    PII_SEVERE_HIT_RATE = 0.05  # 5%+ of sampled rows match a PII pattern → full penalty
    if not pii_hits:
        s_score = 1.0
    else:
        worst_rate = 0.0
        for col_hits in pii_hits.values():
            if isinstance(col_hits, dict):
                for v in col_hits.values():
                    try:
                        worst_rate = max(worst_rate, float(v))
                    except (TypeError, ValueError):
                        # nested dicts or unexpected shapes — treat as severe
                        worst_rate = max(worst_rate, PII_SEVERE_HIT_RATE)
            else:
                try:
                    worst_rate = max(worst_rate, float(col_hits))
                except (TypeError, ValueError):
                    worst_rate = max(worst_rate, PII_SEVERE_HIT_RATE)
        s_score = max(0.0, 1.0 - worst_rate / PII_SEVERE_HIT_RATE)
    components["security"] = s_score * norm_w["security"]

    # Reliability
    r = report.get("reliability", {})
    drift = r.get("numeric_drift_ks_first_last", {}).get("top_10_ks", {})
    if drift:
        max_ks = max((float(v) for v in drift.values() if v is not None), default=0.0)
        r_score = max(0.0, 1.0 - max_ks / max(drift_threshold, 0.01))
    else:
        r_score = 0.75  # unknown → neutral
    components["reliability"] = r_score * norm_w["reliability"]

    # Robustness
    rb = report.get("robustness", {})
    p99 = rb.get("row_anomaly_score_mad", {}).get("p99", None)
    if p99 is not None:
        rb_score = max(0.0, 1.0 - float(p99) / 20.0)
    else:
        rb_score = 0.75
    components["robustness"] = rb_score * norm_w["robustness"]

    # Fairness
    f = report.get("fairness", {})
    if "note" in f:
        components["fairness"] = 0.75 * norm_w["fairness"]
    else:
        disp_scores = []
        for gcol, stats in f.get("group_checks", {}).items():
            disp = stats.get("positive_rate_disparity", None)
            if disp is not None:
                disp_scores.append(max(0.0, 1.0 - float(disp) / 0.5))
        components["fairness"] = (
            (sum(disp_scores) / len(disp_scores)) * norm_w["fairness"]
            if disp_scores
            else 0.75 * norm_w["fairness"]
        )

    total = round(min(100, max(0, sum(components.values()))))

    if total >= 90:   grade = "A"
    elif total >= 80: grade = "B"
    elif total >= 70: grade = "C"
    elif total >= 60: grade = "D"
    else:             grade = "F"

    return total, grade, components


def get_dimension_status(report: Dict[str, Any], drift_threshold: float) -> Dict[str, Tuple[str, str]]:
    """
    Returns {dimension: (status, summary_text)} for check cards.
    status is 'ok' | 'warn' | 'bad'
    """
    out: Dict[str, Tuple[str, str]] = {}

    # Quality
    q = report.get("quality", {})
    miss = float(q.get("missingness", {}).get("overall_missing_rate", 0))
    dup  = float(q.get("duplicates", {}).get("exact_duplicate_row_rate", 0))
    leak = q.get("split_leakage", {}).get("row_hash_cross_split_rate", None)
    if (leak is not None and float(leak) > 0) or dup > 0.05 or miss > 0.15:
        qs, qt = "bad", f"Miss {miss:.1%} · Dup {dup:.1%}"
    elif miss > 0.05 or dup > 0.01:
        qs, qt = "warn", f"Miss {miss:.1%} · Dup {dup:.1%}"
    else:
        qs, qt = "ok", f"Miss {miss:.1%} · Dup {dup:.1%}"
    out["Quality"] = (qs, qt)

    # Reliability
    r = report.get("reliability", {})
    drift = r.get("numeric_drift_ks_first_last", {}).get("top_10_ks", {})
    if drift:
        max_ks = max((float(v) for v in drift.values() if v is not None), default=0.0)
        if max_ks > drift_threshold:
            out["Reliability"] = ("warn", f"Max KS {max_ks:.3f}")
        else:
            out["Reliability"] = ("ok", f"Max KS {max_ks:.3f}")
    else:
        out["Reliability"] = ("ok", "No drift checked")

    # Robustness
    rb = report.get("robustness", {})
    auc = rb.get("label_predictability_auc", None)
    if isinstance(auc, float):
        if auc > 0.85:
            out["Robustness"] = ("warn", f"AUC {auc:.3f} (high)")
        else:
            out["Robustness"] = ("ok", f"AUC {auc:.3f}")
    else:
        p99 = rb.get("row_anomaly_score_mad", {}).get("p99", None)
        if p99 is not None:
            status = "warn" if float(p99) > 10 else "ok"
            out["Robustness"] = (status, f"P99 anomaly {float(p99):.1f}")
        else:
            out["Robustness"] = ("ok", "No issues found")

    # Fairness
    f = report.get("fairness", {})
    if "note" in f:
        out["Fairness"] = ("ok", "Not evaluated")
    else:
        max_disp = 0.0
        for gcol, stats in f.get("group_checks", {}).items():
            d = stats.get("positive_rate_disparity", None)
            if d is not None:
                max_disp = max(max_disp, float(d))
        if max_disp > 0.2:
            out["Fairness"] = ("warn", f"Disparity {max_disp:.3f}")
        else:
            out["Fairness"] = ("ok", f"Disparity {max_disp:.3f}")

    # Security
    s = report.get("security", {})
    pii = s.get("confidentiality_pii_heuristics", {}).get("columns_with_hits", {})
    if pii:
        out["Security"] = ("bad", f"{len(pii)} col(s) flagged")
    else:
        out["Security"] = ("ok", "No PII detected")

    return out


# ─────────────────────────────────────────────
# Transparency / Data Card
# ─────────────────────────────────────────────

def render_transparency_tab(df: pd.DataFrame, file_name: str, file_bytes: bytes,
                             cfg_dict: Dict[str, Any], report: Dict[str, Any]) -> None:
    """Render a full Data Card for the Transparency tab."""
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)

    # ── Dataset Identity ──────────────────────
    st.markdown('<div class="transparency-section">', unsafe_allow_html=True)
    st.markdown('<div class="transparency-header">📁 Dataset Identity</div>', unsafe_allow_html=True)
    sha = sha256_bytes(file_bytes)
    rows = [
        ("File name",    file_name),
        ("Rows",         f"{df.shape[0]:,}"),
        ("Columns",      f"{df.shape[1]:,}"),
        ("File size",    f"{len(file_bytes) / 1024:.1f} KB ({len(file_bytes):,} bytes)"),
        ("SHA-256",      sha),
        ("SHA-256 (short)", sha[:32] + "…"),
    ]
    html = '<div>'
    for k, v in rows:
        html += f'<div class="config-row"><div class="config-key">{k}</div><div class="config-val mono">{v}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Column Schema ─────────────────────────
    st.markdown('<div class="transparency-section">', unsafe_allow_html=True)
    st.markdown('<div class="transparency-header">🗂 Column Schema</div>', unsafe_allow_html=True)

    schema_rows = []
    for c in df.columns:
        col_data = df[c]
        null_pct = col_data.isna().mean()
        dtype_str = str(col_data.dtype)
        n_unique = int(col_data.nunique(dropna=True))
        role = cfg_dict.get("column_roles", {}).get(c, "—")
        sample_vals = col_data.dropna().head(3).tolist()
        sample_str = ", ".join(str(v)[:20] for v in sample_vals)
        schema_rows.append({
            "Column": c,
            "Type": dtype_str,
            "Null %": f"{null_pct:.1%}",
            "Unique": f"{n_unique:,}",
            "Role": role,
            "Sample values": sample_str,
        })

    schema_df = pd.DataFrame(schema_rows)
    st.dataframe(schema_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Analysis Configuration ─────────────────
    st.markdown('<div class="transparency-section">', unsafe_allow_html=True)
    st.markdown('<div class="transparency-header">⚙️ Analysis Configuration</div>', unsafe_allow_html=True)

    config_rows = [
        ("Scan mode",            cfg_dict.get("mode", "—")),
        ("Threshold preset",     cfg_dict.get("preset", "—")),
        ("Drift KS threshold",   str(cfg_dict.get("drift_ks_threshold", "—"))),
        ("PII hit-rate threshold", str(cfg_dict.get("pii_hit_rate_threshold", "—"))),
        ("Label column",         cfg_dict.get("label_col") or "(none)"),
        ("Split column",         cfg_dict.get("split_col") or "(none)"),
        ("Time column",          cfg_dict.get("time_col") or "(none)"),
        ("ID columns",           ", ".join(cfg_dict.get("id_cols", [])) or "(none)"),
        ("Group columns",        ", ".join(cfg_dict.get("group_cols", [])) or "(none)"),
        ("Random seed",          str(cfg_dict.get("random_state", "—"))),
    ]
    html = '<div>'
    for k, v in config_rows:
        html += f'<div class="config-row"><div class="config-key">{k}</div><div class="config-val mono">{v}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Check Registry ─────────────────────────
    st.markdown('<div class="transparency-section">', unsafe_allow_html=True)
    st.markdown('<div class="transparency-header">📋 Check Registry</div>', unsafe_allow_html=True)

    checks = [
        ("Quality › Missingness",         "Measures the proportion of null/NaN values per column and overall.",           "✓ Ran"),
        ("Quality › Duplicates",          "Detects exact duplicate rows and, if ID columns selected, duplicate IDs.",     "✓ Ran"),
        ("Quality › Outliers (IQR)",      "Flags values outside 1.5×IQR fences for all numeric columns.",                "✓ Ran"),
        ("Quality › Label stats",         "Cardinality, missing rate, and class balance for the label column.",           "✓ Ran" if cfg_dict.get("label_col") else "— Skipped (no label)"),
        ("Quality › Split leakage",       "Detects rows that appear in multiple splits via row hash.",                    "✓ Ran" if cfg_dict.get("split_col") else "— Skipped (no split)"),
        ("Reliability › Drift (KS)",      "Kolmogorov–Smirnov test on numeric columns between first and last slices.",    "✓ Ran" if (cfg_dict.get("split_col") or cfg_dict.get("time_col")) else "— Skipped"),
        ("Reliability › Schema check",    "Verifies constant columns and reports dtype summary.",                         "✓ Ran"),
        ("Robustness › Rare-cat conc.",   "Finds rare categorical values with near-perfect label concentration.",         "✓ Ran" if cfg_dict.get("label_col") else "— Skipped (no label)"),
        ("Robustness › Row anomaly (MAD)","Median Absolute Deviation scoring for numeric rows.",                          "✓ Ran"),
        ("Robustness › AUC predictability","Logistic regression to check if labels are trivially predictable.",           "✓ Ran" if cfg_dict.get("label_col") else "— Skipped (no label)"),
        ("Fairness › Group analysis",     "Representation, missingness disparity, and positive rate by group.",           "✓ Ran" if cfg_dict.get("group_cols") else "— Skipped (no groups)"),
        ("Security › PII scan",           "Regex heuristics for emails, phone numbers, IPv4, and credit card patterns.", "✓ Ran"),
        ("Security › Integrity",          "SHA-256 fingerprint of the uploaded file.",                                   "✓ Ran"),
    ]

    check_df = pd.DataFrame(checks, columns=["Check", "Description", "Status"])
    st.dataframe(check_df, use_container_width=True, hide_index=True,
                 column_config={"Check": st.column_config.TextColumn(width="medium"),
                                "Description": st.column_config.TextColumn(width="large"),
                                "Status": st.column_config.TextColumn(width="small")})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close dsa-card


# ─────────────────────────────────────────────
# HTML Export
# ─────────────────────────────────────────────

def _eu_get_path(payload: Dict[str, Any], path: str) -> Any:
    cur: Any = payload
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _eu_first_path(payload: Dict[str, Any], paths: List[str]) -> Tuple[Optional[str], Any]:
    for path in paths:
        value = _eu_get_path(payload, path)
        if value is not None:
            return path, value
    return None, None


def _eu_as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(value):
        return None
    return value


def _eu_max_numeric(values: Any) -> Optional[float]:
    if not isinstance(values, dict):
        return None
    nums = [_eu_as_float(v) for v in values.values()]
    nums = [v for v in nums if v is not None]
    return max(nums) if nums else None


def _eu_format_value(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return str(value)
    numeric = _eu_as_float(value)
    if numeric is not None:
        return f"{numeric:.2%}" if percent else f"{numeric:.4f}"
    return str(value)


def _eu_status(value: Any, threshold: Optional[float] = None, *, higher_is_risk: bool = True) -> str:
    numeric = _eu_as_float(value)
    if numeric is None or threshold is None:
        return "Evidence"
    risky = numeric > threshold if higher_is_risk else numeric < threshold
    return "Review" if risky else "OK"


def _eu_add_item(
    items: List[Dict[str, Any]],
    *,
    article: str,
    metric: str,
    evidence_path: str,
    value: Any,
    interpretation: str,
    threshold: Optional[float] = None,
    percent: bool = False,
    higher_is_risk: bool = True,
) -> None:
    if value is None:
        return
    ref = EU_AI_ACT_ARTICLE_REFERENCES[article]
    items.append(
        {
            "article": article,
            "article_title": ref["title"],
            "law_aspect": ref["aspect"],
            "metric": metric,
            "evidence_path": evidence_path,
            "observed_value": _eu_format_value(value, percent=percent),
            "raw_value": to_json_safe(value),
            "threshold": None if threshold is None else _eu_format_value(threshold, percent=percent),
            "status": _eu_status(value, threshold, higher_is_risk=higher_is_risk),
            "interpretation": interpretation,
        }
    )


def build_eu_ai_act_evidence(
    *,
    analyzer: str,
    report: Dict[str, Any],
    cfg_dict: Optional[Dict[str, Any]] = None,
    file_name: Optional[str] = None,
    score: Optional[int] = None,
    grade: Optional[str] = None,
    verdict: Optional[str] = None,
    findings: Optional[List[str]] = None,
    recommendations: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Map ASTRID results to selected EU AI Act evidence areas."""
    cfg_dict = cfg_dict or {}
    findings = findings or []
    recommendations = recommendations or []
    thresholds = cfg_dict.get("thresholds", {}) if isinstance(cfg_dict.get("thresholds"), dict) else {}
    drift_threshold = _eu_as_float(
        cfg_dict.get("drift_ks_threshold", thresholds.get("drift_ks_threshold", 0.30))
    )
    items: List[Dict[str, Any]] = []

    _eu_add_item(items, article="Article 9", metric="Overall audit verdict", evidence_path="verdict", value=verdict, interpretation="The verdict summarizes observed risk signals that may warrant mitigation or acceptance decisions.")
    _eu_add_item(items, article="Article 9", metric="Findings count", evidence_path="findings", value=len(findings), interpretation="Findings provide a run-specific list of risks to review in a risk-management workflow.")
    _eu_add_item(items, article="Article 9", metric="Recommended actions count", evidence_path="recommendations", value=len(recommendations), interpretation="Recommendations translate detected issues into remediation evidence.")

    path, value = _eu_first_path(report, ["quality.missingness.overall_missing_rate"])
    _eu_add_item(items, article="Article 10", metric="Overall missingness", evidence_path=path or "quality.missingness.overall_missing_rate", value=value, threshold=0.15, percent=True, interpretation="Completeness evidence for training, validation, or testing data quality review.")
    path, value = _eu_first_path(report, ["quality.duplicates.exact_duplicate_row_rate", "quality.duplicates.exact_duplicate_rate", "quality.exact_duplicate_row_rate"])
    _eu_add_item(items, article="Article 10", metric="Exact duplicate rate", evidence_path=path or "quality.duplicates", value=value, threshold=0.08, percent=True, interpretation="Duplicate evidence supports data quality, leakage, and representativeness review.")
    path, value = _eu_first_path(report, ["quality.metadata_completeness.metadata_completeness", "transparency.datasheet_completeness.completeness_rate"])
    _eu_add_item(items, article="Article 10", metric="Metadata or datasheet completeness", evidence_path=path or "quality.metadata_completeness", value=value, threshold=0.80, percent=True, higher_is_risk=False, interpretation="Metadata completeness supports data governance and traceability of dataset assumptions.")
    path, value = _eu_first_path(report, ["quality.label_stats.label_missing_rate", "quality.annotation_linkage.annotation_linkage_rate"])
    _eu_add_item(items, article="Article 10", metric="Label or annotation coverage", evidence_path=path or "quality.label_stats", value=value, percent=True, interpretation="Label and annotation coverage indicate whether supervised data is usable and sufficiently documented.")

    group_checks = _eu_get_path(report, "fairness.group_checks")
    if isinstance(group_checks, dict):
        disparities: List[float] = []
        for stats in group_checks.values():
            if not isinstance(stats, dict):
                continue
            for key in ("positive_rate_disparity", "max_label_parity_gap", "max_missingness_disparity"):
                val = _eu_as_float(stats.get(key))
                if val is not None:
                    disparities.append(val)
        _eu_add_item(items, article="Article 10", metric="Maximum subgroup disparity", evidence_path="fairness.group_checks", value=max(disparities) if disparities else None, threshold=0.40, percent=True, interpretation="Subgroup disparity evidence supports bias and representativeness review.")

    _eu_add_item(items, article="Article 11", metric="Configuration captured", evidence_path="config", value=bool(cfg_dict), interpretation="The run configuration supports reproducible technical documentation.")
    _eu_add_item(items, article="Article 11", metric="Health score and grade", evidence_path="score", value=f"{score}/100 ({grade})" if score is not None else None, interpretation="The health score is technical evidence of the dataset assessment outcome.")
    _eu_add_item(items, article="Article 12", metric="Integrity fingerprint", evidence_path="security.integrity", value=_eu_first_path(report, ["security.integrity.sha256", "security.integrity.sha256_zip"])[1], interpretation="A cryptographic fingerprint supports record-keeping and later traceability.")
    _eu_add_item(items, article="Article 12", metric="Dataset or file name", evidence_path="dataset_name", value=file_name, interpretation="Dataset identity links the evidence report to a concrete audited artifact.")

    path, value = _eu_first_path(report, ["transparency.dataset_identity.total_images", "reliability.schema_consistency.num_rows", "quality.time_axis_health.time_range.span_days"])
    _eu_add_item(items, article="Article 13", metric="Dataset identity or scope", evidence_path=path or "transparency.dataset_identity", value=value, interpretation="Scope and identity fields help deployers understand the audited input data.")
    path, value = _eu_first_path(report, ["transparency.traceability_coverage.coverage_rate", "transparency.source_attribution_coverage.coverage_rate", "transparency.datasheet_completeness.completeness_rate"])
    _eu_add_item(items, article="Article 13", metric="Traceability or transparency coverage", evidence_path=path or "transparency", value=value, threshold=0.80, percent=True, higher_is_risk=False, interpretation="Transparency evidence helps communicate limitations, provenance, and assumptions to deployers.")

    drift_path, drift_values = _eu_first_path(report, ["reliability.numeric_drift_ks_first_last.top_10_ks", "reliability.feature_drift_ks_first_last.top_10_ks"])
    _eu_add_item(items, article="Article 15", metric="Maximum drift statistic", evidence_path=drift_path or "reliability.*_drift_ks_first_last.top_10_ks", value=_eu_max_numeric(drift_values), threshold=drift_threshold, interpretation="Drift evidence supports robustness and expected performance stability review.")
    path, value = _eu_first_path(report, ["quality.split_leakage.row_hash_cross_split_rate", "quality.cross_split_leakage.cross_split_leakage_rate"])
    _eu_add_item(items, article="Article 15", metric="Cross-split leakage", evidence_path=path or "quality.split_leakage", value=value, threshold=0.01, percent=True, interpretation="Leakage evidence supports robustness and reliable validation review.")
    path, value = _eu_first_path(report, ["robustness.row_anomaly_score_mad.p99", "robustness.image_feature_outliers_mad.outlier_rate"])
    _eu_add_item(items, article="Article 15", metric="Outlier or anomaly signal", evidence_path=path or "robustness", value=value, interpretation="Outlier evidence supports robustness review under unusual or edge-case data.")
    path, value = _eu_first_path(report, ["quality.time_axis_health.time_parse.parse_ok_rate", "quality.time_axis_health.cadence_global.irregularity_ratio"])
    _eu_add_item(items, article="Article 15", metric="Temporal stability signal", evidence_path=path or "quality.time_axis_health", value=value, interpretation="Time-axis quality supports stability and expected performance review for temporal datasets.")
    pii_hits = _eu_get_path(report, "security.confidentiality_pii_heuristics.columns_with_hits")
    _eu_add_item(items, article="Article 15", metric="PII-like fields flagged", evidence_path="security.confidentiality_pii_heuristics.columns_with_hits", value=len(pii_hits) if isinstance(pii_hits, dict) else None, threshold=0, interpretation="Privacy-sensitive patterns are security-relevant signals that require domain and legal review.")
    path, value = _eu_first_path(report, ["security.exif_privacy.gps_images_count", "security.suspicious_samples.suspicious_sample_rate"])
    _eu_add_item(items, article="Article 15", metric="Image security/privacy signal", evidence_path=path or "security.exif_privacy", value=value, interpretation="EXIF, GPS, and suspicious-sample signals support cybersecurity and privacy-risk review.")

    return to_json_safe(
        {
            "title": "EU AI Act Evidence Mapping",
            "source": EU_AI_ACT_SOURCE_URL,
            "legal_disclaimer": "This report maps ASTRID technical evidence to selected EU AI Act requirement areas. It is not a legal opinion, compliance certification, or high-risk classification.",
            "analyzer": analyzer,
            "dataset_name": file_name,
            "summary": {"score": score, "grade": grade, "verdict": verdict, "findings_count": len(findings), "recommendations_count": len(recommendations)},
            "article_references": EU_AI_ACT_ARTICLE_REFERENCES,
            "evidence": items,
        }
    )


def build_eu_ai_act_evidence_markdown(evidence_report: Dict[str, Any]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "# EU AI Act Evidence Mapping",
        "",
        evidence_report.get("legal_disclaimer", ""),
        "",
        f"- **Source:** {evidence_report.get('source', EU_AI_ACT_SOURCE_URL)}",
        f"- **Analyzer:** {evidence_report.get('analyzer', '')}",
        f"- **Dataset:** {evidence_report.get('dataset_name', '')}",
        "",
        "## Run summary",
        "",
    ]
    for key, value in evidence_report.get("summary", {}).items():
        lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
    lines += [
        "",
        "## Evidence map",
        "",
        "| EU AI Act area | Metric | Observed value | Threshold | Status | Evidence path | Interpretation |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for item in evidence_report.get("evidence", []):
        article = f"{item.get('article')} - {item.get('article_title')}"
        lines.append(
            "| "
            + " | ".join(
                [
                    cell(article),
                    cell(item.get("metric", "")),
                    cell(item.get("observed_value", "")),
                    cell(item.get("threshold") or ""),
                    cell(item.get("status", "")),
                    cell(item.get("evidence_path", "")),
                    cell(item.get("interpretation", "")),
                ]
            )
            + " |"
        )
    lines += ["", "## Article reference areas", ""]
    for article, ref in evidence_report.get("article_references", {}).items():
        lines.append(f"- **{article} - {ref.get('title')}:** {ref.get('aspect')}")
    return "\n".join(lines)


def render_eu_ai_act_evidence_section(evidence_report: Dict[str, Any]) -> None:
    if st is None:
        return
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("EU AI Act Evidence")
    st.caption(evidence_report.get("legal_disclaimer", ""))
    st.markdown(f"Official source: [{EU_AI_ACT_SOURCE_URL}]({EU_AI_ACT_SOURCE_URL})")
    summary = evidence_report.get("summary", {})
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Mapped evidence items", len(evidence_report.get("evidence", [])))
    with c2:
        st.metric("Score", "N/A" if summary.get("score") is None else f"{summary.get('score')}/100")
    with c3:
        st.metric("Grade", summary.get("grade") or "N/A")
    rows = [
        {
            "EU AI Act area": f"{item.get('article')} - {item.get('article_title')}",
            "Metric": item.get("metric"),
            "Observed value": item.get("observed_value"),
            "Threshold": item.get("threshold") or "",
            "Status": item.get("status"),
            "Evidence path": item.get("evidence_path"),
            "Interpretation": item.get("interpretation"),
        }
        for item in evidence_report.get("evidence", [])
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No EU AI Act evidence items could be mapped for this report.")
    st.markdown("</div>", unsafe_allow_html=True)


def _iso_add_item(
    items: List[Dict[str, Any]],
    *,
    characteristic: str,
    metric: str,
    evidence_path: str,
    value: Any,
    interpretation: str,
    threshold: Optional[float] = None,
    percent: bool = False,
    higher_is_risk: bool = True,
) -> None:
    if value is None:
        return
    ref = ISO_25012_CHARACTERISTICS[characteristic]
    items.append(
        {
            "characteristic": characteristic,
            "perspective": ref["perspective"],
            "quality_aspect": ref["aspect"],
            "metric": metric,
            "evidence_path": evidence_path,
            "observed_value": _eu_format_value(value, percent=percent),
            "raw_value": to_json_safe(value),
            "threshold": None if threshold is None else _eu_format_value(threshold, percent=percent),
            "status": _eu_status(value, threshold, higher_is_risk=higher_is_risk),
            "interpretation": interpretation,
        }
    )


def build_iso_25012_evidence(
    *,
    analyzer: str,
    report: Dict[str, Any],
    cfg_dict: Optional[Dict[str, Any]] = None,
    file_name: Optional[str] = None,
    score: Optional[int] = None,
    grade: Optional[str] = None,
    verdict: Optional[str] = None,
    findings: Optional[List[str]] = None,
    recommendations: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Map ASTRID results to ISO/IEC 25012 data-quality characteristics."""
    cfg_dict = cfg_dict or {}
    findings = findings or []
    recommendations = recommendations or []
    thresholds = cfg_dict.get("thresholds", {}) if isinstance(cfg_dict.get("thresholds"), dict) else {}
    drift_threshold = _eu_as_float(
        cfg_dict.get("drift_ks_threshold", thresholds.get("drift_ks_threshold", 0.30))
    )
    items: List[Dict[str, Any]] = []

    path, value = _eu_first_path(
        report,
        [
            "quality.annotation_schema_consistency.schema_consistency_rate",
            "quality.bbox_validity.bbox_validity_rate",
            "quality.time_axis_health.time_parse.parse_ok_rate",
            "quality.label_agreement.exact_agreement_rate",
        ],
    )
    _iso_add_item(items, characteristic="Accuracy", metric="Schema, annotation, or timestamp validity", evidence_path=path or "quality", value=value, threshold=0.95, percent=True, higher_is_risk=False, interpretation="Validity and agreement checks provide partial evidence that values or annotations are plausible and correctly structured.")

    path, value = _eu_first_path(report, ["quality.missingness.overall_missing_rate"])
    _iso_add_item(items, characteristic="Completeness", metric="Overall missingness", evidence_path=path or "quality.missingness.overall_missing_rate", value=value, threshold=0.15, percent=True, interpretation="Missingness is direct evidence for whether expected values are present.")
    path, value = _eu_first_path(report, ["quality.metadata_completeness.metadata_completeness", "transparency.datasheet_completeness.completeness_rate"])
    _iso_add_item(items, characteristic="Completeness", metric="Metadata or datasheet completeness", evidence_path=path or "quality.metadata_completeness", value=value, threshold=0.80, percent=True, higher_is_risk=False, interpretation="Metadata and datasheet completeness indicate whether context needed to use the data is present.")

    path, value = _eu_first_path(report, ["quality.duplicates.exact_duplicate_row_rate", "quality.duplicates.exact_duplicate_rate", "quality.exact_duplicate_row_rate"])
    _iso_add_item(items, characteristic="Consistency", metric="Exact duplicate rate", evidence_path=path or "quality.duplicates", value=value, threshold=0.08, percent=True, interpretation="Duplicate rows or assets can indicate inconsistent data collection, merging, or curation.")
    path, value = _eu_first_path(report, ["quality.split_leakage.row_hash_cross_split_rate", "quality.cross_split_leakage.cross_split_leakage_rate", "quality.time_axis_health.duplicate_timestamps.duplicate_rate"])
    _iso_add_item(items, characteristic="Consistency", metric="Cross-split leakage or duplicate timestamp rate", evidence_path=path or "quality", value=value, threshold=0.01, percent=True, interpretation="Leakage and duplicate time points are contradiction-style signals across dataset partitions or temporal keys.")

    path, value = _eu_first_path(report, ["security.integrity.sha256", "security.integrity.sha256_zip"])
    _iso_add_item(items, characteristic="Credibility", metric="Integrity fingerprint present", evidence_path=path or "security.integrity", value=bool(value), interpretation="A file fingerprint supports authenticity and later verification of the audited artifact.")
    path, value = _eu_first_path(report, ["security.suspicious_samples.suspicious_sample_rate"])
    _iso_add_item(items, characteristic="Credibility", metric="Suspicious sample rate", evidence_path=path or "security.suspicious_samples", value=value, threshold=0.05, percent=True, interpretation="Suspicious samples weaken trust in the dataset and should be reviewed.")

    drift_path, drift_values = _eu_first_path(report, ["reliability.numeric_drift_ks_first_last.top_10_ks", "reliability.feature_drift_ks_first_last.top_10_ks"])
    _iso_add_item(items, characteristic="Currentness", metric="Maximum drift statistic", evidence_path=drift_path or "reliability.*_drift_ks_first_last.top_10_ks", value=_eu_max_numeric(drift_values), threshold=drift_threshold, interpretation="Drift across time or slices is evidence that the data distribution may no longer reflect the intended context.")
    path, value = _eu_first_path(report, ["quality.time_axis_health.time_range.span_days"])
    _iso_add_item(items, characteristic="Currentness", metric="Temporal span", evidence_path=path or "quality.time_axis_health.time_range", value=value, interpretation="The observed temporal coverage helps reviewers judge whether the data are current enough for the use case.")

    path, value = _eu_first_path(report, ["quality.readability.readability_rate", "quality.format_conformance.format_conformance_rate"])
    _iso_add_item(items, characteristic="Accessibility", metric="Readability or format conformance", evidence_path=path or "quality.readability", value=value, threshold=0.99, percent=True, higher_is_risk=False, interpretation="Readable and conformant files are evidence that the data can be accessed in the expected processing context.")

    _iso_add_item(items, characteristic="Compliance", metric="Configuration captured", evidence_path="config", value=bool(cfg_dict), interpretation="Captured configuration supports repeatable review against local rules and documented conventions.")
    _iso_add_item(items, characteristic="Compliance", metric="Findings count", evidence_path="findings", value=len(findings), interpretation="Findings provide evidence of checks that may need policy or standard-specific review.")

    pii_hits = _eu_get_path(report, "security.confidentiality_pii_heuristics.columns_with_hits")
    _iso_add_item(items, characteristic="Confidentiality", metric="PII-like fields flagged", evidence_path="security.confidentiality_pii_heuristics.columns_with_hits", value=len(pii_hits) if isinstance(pii_hits, dict) else None, threshold=0, interpretation="PII-like patterns are direct confidentiality evidence and require legal or domain validation.")
    path, value = _eu_first_path(report, ["security.exif_privacy.gps_images_count"])
    _iso_add_item(items, characteristic="Confidentiality", metric="Images with GPS EXIF", evidence_path=path or "security.exif_privacy.gps_images_count", value=value, threshold=0, interpretation="GPS EXIF metadata can reveal sensitive location information.")

    path, value = _eu_first_path(report, ["security.availability_asset_checks.byte_size", "security.integrity.zip_byte_size", "transparency.dataset_identity.zip_byte_size"])
    _iso_add_item(items, characteristic="Efficiency", metric="Audited artifact byte size", evidence_path=path or "security.availability_asset_checks.byte_size", value=value, interpretation="File size and audited scope are resource-planning evidence for processing efficiency.")
    path, value = _eu_first_path(report, ["reliability.schema_consistency.num_rows", "transparency.dataset_identity.total_images"])
    _iso_add_item(items, characteristic="Efficiency", metric="Audited item count", evidence_path=path or "reliability.schema_consistency.num_rows", value=value, interpretation="Row or item count gives operational context for processing and review cost.")

    path, value = _eu_first_path(report, ["robustness.row_anomaly_score_mad.p99", "robustness.image_feature_outliers_mad.outlier_rate", "quality.blur_proxy.low_blur_rate"])
    _iso_add_item(items, characteristic="Precision", metric="Outlier, anomaly, or blur signal", evidence_path=path or "robustness", value=value, interpretation="Outlier and blur signals provide partial evidence about measurement exactness and data fidelity.")

    path, value = _eu_first_path(report, ["transparency.traceability_coverage.coverage_rate", "transparency.source_attribution_coverage.coverage_rate", "security.integrity.sha256", "security.integrity.sha256_zip"])
    _iso_add_item(items, characteristic="Traceability", metric="Traceability, source attribution, or fingerprint", evidence_path=path or "transparency.traceability_coverage", value=value, threshold=0.80 if _eu_as_float(value) is not None else None, percent=_eu_as_float(value) is not None, higher_is_risk=False, interpretation="Traceability evidence links the data to origins, identifiers, or verifiable fingerprints.")

    path, value = _eu_first_path(report, ["transparency.datasheet_completeness.completeness_rate", "quality.metadata_completeness.metadata_completeness"])
    _iso_add_item(items, characteristic="Understandability", metric="Documentation completeness", evidence_path=path or "transparency.datasheet_completeness", value=value, threshold=0.80, percent=True, higher_is_risk=False, interpretation="Documentation and metadata make the data meaning, limitations, and intended use easier to understand.")
    _iso_add_item(items, characteristic="Understandability", metric="Metric registry/configuration captured", evidence_path="config", value=bool(cfg_dict), interpretation="Stored configuration and metric registries help reviewers understand how the audit was produced.")

    path, value = _eu_first_path(report, ["quality.readability.corrupt_rate", "security.suspicious_samples.suspicious_sample_rate"])
    _iso_add_item(items, characteristic="Availability", metric="Corrupt or suspicious sample rate", evidence_path=path or "quality.readability.corrupt_rate", value=value, threshold=0.01, percent=True, interpretation="Unreadable or suspicious samples reduce data availability for authorized use.")
    path, value = _eu_first_path(report, ["quality.readability.readability_rate"])
    _iso_add_item(items, characteristic="Availability", metric="Readability rate", evidence_path=path or "quality.readability.readability_rate", value=value, threshold=0.99, percent=True, higher_is_risk=False, interpretation="Readability is direct evidence that data items are available to the analysis pipeline.")

    path, value = _eu_first_path(report, ["quality.format_conformance.format_conformance_rate", "reliability.schema_consistency.num_cols"])
    _iso_add_item(items, characteristic="Portability", metric="Format conformance or schema width", evidence_path=path or "quality.format_conformance", value=value, threshold=0.99 if path and "format_conformance" in path else None, percent=bool(path and "format_conformance" in path), higher_is_risk=False, interpretation="Format conformance and schema description support movement across tools and environments.")

    path, value = _eu_first_path(report, ["security.integrity.sha256", "security.integrity.sha256_zip"])
    _iso_add_item(items, characteristic="Recoverability", metric="Recovery verification fingerprint", evidence_path=path or "security.integrity", value=bool(value), interpretation="A fingerprint does not recover data by itself, but it supports verification after restore or transfer.")

    covered = {item["characteristic"] for item in items}
    gaps = [
        {
            "characteristic": name,
            "perspective": ref["perspective"],
            "quality_aspect": ref["aspect"],
            "gap": "No direct ASTRID evidence was mapped for this run.",
        }
        for name, ref in ISO_25012_CHARACTERISTICS.items()
        if name not in covered
    ]
    return to_json_safe(
        {
            "title": "ISO/IEC 25012 Evidence Mapping",
            "source": ISO_25012_SOURCE_URL,
            "standards_disclaimer": "This report maps ASTRID technical evidence to ISO/IEC 25012 data-quality characteristics. It is not a formal conformity assessment or certification.",
            "analyzer": analyzer,
            "dataset_name": file_name,
            "summary": {"score": score, "grade": grade, "verdict": verdict, "findings_count": len(findings), "recommendations_count": len(recommendations)},
            "characteristic_references": ISO_25012_CHARACTERISTICS,
            "evidence": items,
            "coverage_gaps": gaps,
        }
    )


def build_iso_25012_evidence_markdown(evidence_report: Dict[str, Any]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "# ISO/IEC 25012 Evidence Mapping",
        "",
        evidence_report.get("standards_disclaimer", ""),
        "",
        f"- **Source:** {evidence_report.get('source', ISO_25012_SOURCE_URL)}",
        f"- **Analyzer:** {evidence_report.get('analyzer', '')}",
        f"- **Dataset:** {evidence_report.get('dataset_name', '')}",
        "",
        "## Run summary",
        "",
    ]
    for key, value in evidence_report.get("summary", {}).items():
        lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
    lines += [
        "",
        "## Evidence map",
        "",
        "| ISO/IEC 25012 characteristic | Perspective | Metric | Observed value | Threshold | Status | Evidence path | Interpretation |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for item in evidence_report.get("evidence", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    cell(item.get("characteristic", "")),
                    cell(item.get("perspective", "")),
                    cell(item.get("metric", "")),
                    cell(item.get("observed_value", "")),
                    cell(item.get("threshold") or ""),
                    cell(item.get("status", "")),
                    cell(item.get("evidence_path", "")),
                    cell(item.get("interpretation", "")),
                ]
            )
            + " |"
        )
    gaps = evidence_report.get("coverage_gaps", [])
    if gaps:
        lines += ["", "## Coverage gaps", ""]
        for gap in gaps:
            lines.append(f"- **{gap.get('characteristic')} ({gap.get('perspective')}):** {gap.get('gap')}")
    lines += ["", "## Characteristic reference areas", ""]
    for name, ref in evidence_report.get("characteristic_references", {}).items():
        lines.append(f"- **{name} ({ref.get('perspective')}):** {ref.get('aspect')}")
    return "\n".join(lines)


def render_iso_25012_evidence_section(evidence_report: Dict[str, Any]) -> None:
    if st is None:
        return
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("ISO/IEC 25012 Evidence")
    st.caption(evidence_report.get("standards_disclaimer", ""))
    st.markdown(f"Official source: [{ISO_25012_SOURCE_URL}]({ISO_25012_SOURCE_URL})")
    summary = evidence_report.get("summary", {})
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Mapped evidence items", len(evidence_report.get("evidence", [])))
    with c2:
        st.metric("Coverage gaps", len(evidence_report.get("coverage_gaps", [])))
    with c3:
        st.metric("Grade", summary.get("grade") or "N/A")
    rows = [
        {
            "Characteristic": item.get("characteristic"),
            "Perspective": item.get("perspective"),
            "Metric": item.get("metric"),
            "Observed value": item.get("observed_value"),
            "Threshold": item.get("threshold") or "",
            "Status": item.get("status"),
            "Evidence path": item.get("evidence_path"),
            "Interpretation": item.get("interpretation"),
        }
        for item in evidence_report.get("evidence", [])
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No ISO/IEC 25012 evidence items could be mapped for this report.")
    gaps = evidence_report.get("coverage_gaps", [])
    if gaps:
        with st.expander("Coverage gaps"):
            st.dataframe(pd.DataFrame(gaps), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def build_markdown_report(df, report, cfg_dict, file_name, file_bytes,
                           verdict, reasons, recs, score, grade):
    """Build a portable Markdown report — suitable for archiving in git, PR comments, or wikis."""
    sha = sha256_bytes(file_bytes)
    miss_rate = report["quality"]["missingness"]["overall_missing_rate"]
    dup_rate = report["quality"]["duplicates"]["exact_duplicate_row_rate"]
    pii_cols = report.get("security", {}).get("confidentiality_pii_heuristics", {}).get("columns_with_hits", {})
    drift = report.get("reliability", {}).get("numeric_drift_ks_first_last", {}).get("top_10_ks", {})
    miss_top = report["quality"]["missingness"]["top_10_columns_missing_rate"]

    lines = []
    lines.append(f"# Dataset Safety Report — {file_name}\n")
    lines.append(f"**Rows × Columns:** {df.shape[0]:,} × {df.shape[1]:,}  ")
    lines.append(f"**Mode:** {cfg_dict.get('mode','—')}  ")
    lines.append(f"**Preset:** {cfg_dict.get('preset','—')}\n")
    lines.append(f"## Overall\n")
    lines.append(f"- **Score:** {score} / 100 — Grade **{grade}**")
    lines.append(f"- **Verdict:** {verdict}")
    lines.append(f"- **Missingness:** {miss_rate:.2%}")
    lines.append(f"- **Duplicates:** {dup_rate:.2%}\n")

    if reasons:
        lines.append("## Findings\n")
        for r in reasons:
            lines.append(f"- {r}")
        lines.append("")

    if recs:
        lines.append("## Recommended Actions\n")
        for r in recs:
            lines.append(f"- {r}")
        lines.append("")

    if miss_top:
        lines.append("## Missingness — Top Columns\n")
        lines.append("| Column | Missing Rate |")
        lines.append("| --- | ---: |")
        for col, val in miss_top.items():
            lines.append(f"| {col} | {val:.2%} |")
        lines.append("")

    if drift:
        thr = cfg_dict.get("drift_ks_threshold", 0.3)
        lines.append("## Numeric Drift (KS, first vs last slice)\n")
        lines.append("| Column | KS Statistic | Status |")
        lines.append("| --- | ---: | --- |")
        for col, val in drift.items():
            if val is None:
                continue
            status = "Above threshold" if float(val) > thr else "OK"
            lines.append(f"| {col} | {float(val):.4f} | {status} |")
        lines.append("")

    if pii_cols:
        lines.append("## PII Findings\n")
        lines.append("| Column | Pattern | Hit Rate |")
        lines.append("| --- | --- | ---: |")
        for col, hits in pii_cols.items():
            if isinstance(hits, dict):
                for pattern, rate in hits.items():
                    try:
                        lines.append(f"| {col} | {pattern} | {float(rate):.2%} |")
                    except (TypeError, ValueError):
                        lines.append(f"| {col} | {pattern} | {rate} |")
        lines.append("")

    lines.append("## Integrity\n")
    lines.append(f"- **SHA-256:** `{sha}`\n")
    lines.append("---\n*Generated by ASTRID — heuristic report. Validate with domain and legal review.*\n")
    return "\n".join(lines)


def build_html_report(df, report, cfg_dict, file_name, file_bytes,
                       verdict, reasons, recs, score, grade):
    """Generate a standalone HTML report."""
    sha = sha256_bytes(file_bytes)
    quality = report.get("quality", {}) if isinstance(report, dict) else {}
    reliability = report.get("reliability", {}) if isinstance(report, dict) else {}
    security = report.get("security", {}) if isinstance(report, dict) else {}
    thresholds = cfg_dict.get("thresholds", {}) if isinstance(cfg_dict.get("thresholds"), dict) else {}

    pii_cols = security.get("confidentiality_pii_heuristics", {}).get("columns_with_hits", {})
    miss_rate = quality.get("missingness", {}).get("overall_missing_rate", 0.0) or 0.0
    dup_info = quality.get("duplicates", {})
    dup_rate = (
        dup_info.get("exact_duplicate_row_rate")
        if isinstance(dup_info, dict)
        else None
    )
    if dup_rate is None and isinstance(dup_info, dict):
        dup_rate = dup_info.get("exact_duplicate_rate")
    if dup_rate is None:
        dup_rate = quality.get("exact_duplicate_row_rate", 0.0)
    dup_rate = dup_rate or 0.0

    def esc(value: Any) -> str:
        return html.escape(str(value), quote=True)

    score_color = "#22c55e" if score >= 80 else ("#f59e0b" if score >= 60 else "#ef4444")
    reasons_html = "".join(f"<li>{esc(r)}</li>" for r in reasons)
    recs_html = "".join(f"<li>{esc(r)}</li>" for r in recs)
    is_image_report = (
        "path_in_zip" in getattr(df, "columns", [])
        or bool(report.get("transparency", {}).get("dataset_identity"))
    )

    drift = reliability.get("numeric_drift_ks_first_last", {}).get(
        "top_10_ks",
        reliability.get("feature_drift_ks_first_last", {}).get("top_10_ks", {}),
    )
    thr = cfg_dict.get("drift_ks_threshold", thresholds.get("drift_ks_threshold", 0.3))
    drift_rows = "".join(
        f"<tr><td>{esc(col)}</td><td>{float(val):.4f}</td>"
        f"<td style='color:{('#f59e0b' if float(val) > thr else '#22c55e')}'>"
        f"{('Above threshold' if float(val) > thr else 'OK')}</td></tr>"
        for col, val in drift.items() if val is not None
    )

    miss_top = quality.get("missingness", {}).get("top_10_columns_missing_rate", {})
    miss_rows = "".join(
        f"<tr><td>{esc(col)}</td><td>{val:.2%}</td></tr>" for col, val in miss_top.items()
    )

    pii_rows = ""
    for col, hits in pii_cols.items():
        if isinstance(hits, dict):
            for pattern, rate in hits.items():
                try:
                    pii_rows += (
                        f"<tr><td>{esc(col)}</td><td>{esc(pattern)}</td>"
                        f"<td>{float(rate):.2%}</td></tr>"
                    )
                except (TypeError, ValueError):
                    pii_rows += (
                        f"<tr><td>{esc(col)}</td><td>{esc(pattern)}</td>"
                        f"<td>{esc(rate)}</td></tr>"
                    )

    drift_section = (f"<h2>Numeric Drift (KS)</h2><table><tr><th>Column</th><th>KS Statistic</th><th>Status</th></tr>{drift_rows}</table>" if drift_rows else "")
    pii_section = (f"<h2>PII Findings</h2><table><tr><th>Column</th><th>Pattern</th><th>Hit Rate</th></tr>{pii_rows}</table>" if pii_rows else "")
    image_section = ""
    if is_image_report:
        def pct(value):
            try:
                return f"{float(value):.2%}"
            except (TypeError, ValueError):
                return "N/A"

        image_rows = [
            ("Readability", pct(quality.get("readability", {}).get("readability_rate"))),
            ("Corrupt image rate", pct(quality.get("readability", {}).get("corrupt_rate"))),
            ("Low-resolution rate", pct(quality.get("low_resolution", {}).get("low_res_rate"))),
            ("Exact duplicate rate", pct(dup_rate)),
            ("Metadata completeness", pct(quality.get("metadata_completeness", {}).get("metadata_completeness"))),
        ]
        image_section = (
            "<h2>Image Quality Signals</h2><table><tr><th>Metric</th><th>Value</th></tr>"
            + "".join(f"<tr><td>{esc(name)}</td><td>{esc(value)}</td></tr>" for name, value in image_rows)
            + "</table>"
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Dataset Safety Report — {esc(file_name)}</title>
<style>
body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin:0; padding:0; background:#0f1117; color:#e2e8f0; }}
.container {{ max-width: 1100px; margin:0 auto; padding: 40px 24px; }}
h1 {{ font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 4px; }}
h2 {{ font-size: 1.1rem; font-weight: 700; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px; margin-top: 32px; }}
.meta {{ color: #94a3b8; font-size: 0.85rem; margin-bottom: 32px; }}
.score-ring {{ display:inline-flex; align-items:center; gap:16px; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1); border-radius:16px; padding:20px 28px; margin-bottom:24px; }}
.score-num {{ font-size: 3rem; font-weight: 900; color: {score_color}; line-height:1; }}
.score-label {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color:#94a3b8; }}
.verdict {{ font-size: 1.1rem; font-weight: 700; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 24px 0; }}
.kpi {{ background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 14px 16px; }}
.kpi-t {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color:#94a3b8; }}
.kpi-v {{ font-size: 1.5rem; font-weight: 800; margin: 4px 0; }}
ul {{ padding-left: 1.2em; }}
li {{ margin: 6px 0; line-height: 1.5; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 12px; }}
th {{ background: rgba(255,255,255,0.06); padding: 8px 12px; text-align: left; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.07em; color:#94a3b8; }}
td {{ padding: 7px 12px; border-bottom: 1px solid rgba(255,255,255,0.06); }}
.ok {{ color: #22c55e; }} .warn {{ color: #f59e0b; }} .bad {{ color: #ef4444; }}
.sha {{ font-family: monospace; font-size: 0.75rem; word-break: break-all; color: #64748b; }}
footer {{ margin-top: 48px; color: #475569; font-size: 0.78rem; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 16px; }}
</style></head><body>
<div class="container">
<h1>Dataset Safety Report</h1>
<div class="meta">File: <strong>{esc(file_name)}</strong> &nbsp;·&nbsp; {df.shape[0]:,} rows × {df.shape[1]:,} columns &nbsp;·&nbsp; Mode: {esc(cfg_dict.get('mode','—'))} &nbsp;·&nbsp; Preset: {esc(cfg_dict.get('preset','—'))}</div>
<div class="score-ring">
  <div><div class="score-num">{esc(score)}</div><div style="color:{score_color};font-weight:700;font-size:0.9rem;">Grade {esc(grade)}</div></div>
  <div><div class="score-label">Overall verdict</div><div class="verdict">{esc(verdict)}</div></div>
</div>
<div class="kpis">
<div class="kpi"><div class="kpi-t">Rows</div><div class="kpi-v">{df.shape[0]:,}</div></div>
<div class="kpi"><div class="kpi-t">Columns</div><div class="kpi-v">{df.shape[1]:,}</div></div>
<div class="kpi"><div class="kpi-t">Missingness</div><div class="kpi-v {('warn' if miss_rate>0.05 else 'ok')}">{miss_rate:.2%}</div></div>
<div class="kpi"><div class="kpi-t">Duplicates</div><div class="kpi-v {('warn' if dup_rate>0.01 else 'ok')}">{dup_rate:.2%}</div></div>
</div>
<h2>Findings</h2><ul>{reasons_html}</ul>
<h2>Recommended Actions</h2><ul>{recs_html}</ul>
{image_section}
<h2>Missingness — Top Columns</h2><table><tr><th>Column</th><th>Missing Rate</th></tr>{miss_rows}</table>
{drift_section}
{pii_section}
<h2>Integrity</h2><div class="sha">SHA-256: {sha}</div>
<footer>Generated by ASTRID — heuristic report. Validate with domain and legal review.</footer>
</div></body></html>"""
