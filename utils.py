"""
Shared utilities for Unified Dataset Safety Analyzer.
Import in each page with:
    import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils import *
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

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


def ks_statistic(x1: np.ndarray, x2: np.ndarray) -> Optional[float]:
    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]
    if len(x1) < 20 or len(x2) < 20:
        return None
    x1, x2 = np.sort(x1), np.sort(x2)
    all_vals = np.sort(np.unique(np.concatenate([x1, x2])))
    cdf1 = np.searchsorted(x1, all_vals, side="right") / len(x1)
    cdf2 = np.searchsorted(x2, all_vals, side="right") / len(x2)
    return float(np.max(np.abs(cdf1 - cdf2)))


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
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Normalise supplied weights so they always sum to 100
    w_sum = sum(weights.values())
    if w_sum <= 0:
        w_sum = 1.0
    norm_w: Dict[str, float] = {k: v / w_sum * 100.0 for k, v in weights.items()}

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
    components["quality"] = (sum(q_scores) / len(q_scores)) * norm_w.get("quality", 35)

    # Security
    s = report.get("security", {})
    pii_hits = s.get("confidentiality_pii_heuristics", {}).get("columns_with_hits", {})
    components["security"] = 0.0 if pii_hits else norm_w.get("security", 25)

    # Reliability
    r = report.get("reliability", {})
    drift = r.get("numeric_drift_ks_first_last", {}).get("top_10_ks", {})
    if drift:
        max_ks = max((float(v) for v in drift.values() if v is not None), default=0.0)
        r_score = max(0.0, 1.0 - max_ks / max(drift_threshold, 0.01))
    else:
        r_score = 0.75  # unknown → neutral
    components["reliability"] = r_score * norm_w.get("reliability", 20)

    # Robustness
    rb = report.get("robustness", {})
    p99 = rb.get("row_anomaly_score_mad", {}).get("p99", None)
    if p99 is not None:
        rb_score = max(0.0, 1.0 - float(p99) / 20.0)
    else:
        rb_score = 0.75
    components["robustness"] = rb_score * norm_w.get("robustness", 10)

    # Fairness
    f = report.get("fairness", {})
    if "note" in f:
        components["fairness"] = 0.75 * norm_w.get("fairness", 10)
    else:
        disp_scores = []
        for gcol, stats in f.get("group_checks", {}).items():
            disp = stats.get("positive_rate_disparity", None)
            if disp is not None:
                disp_scores.append(max(0.0, 1.0 - float(disp) / 0.5))
        components["fairness"] = (
            (sum(disp_scores) / len(disp_scores)) * norm_w.get("fairness", 10)
            if disp_scores
            else 0.75 * norm_w.get("fairness", 10)
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

def build_html_report(df: pd.DataFrame, report: Dict[str, Any], cfg_dict: Dict[str, Any],
                      file_name: str, file_bytes: bytes, verdict: str, reasons: List[str],
                      recs: List[str], score: int, grade: str) -> str:
    """Generate a standalone HTML report."""
    sha = sha256_bytes(file_bytes)
    pii_cols = report["security"]["confidentiality_pii_heuristics"].get("columns_with_hits", {})
    miss_rate = report["quality"]["missingness"]["overall_missing_rate"]
    dup_rate  = report["quality"]["duplicates"]["exact_duplicate_row_rate"]

    score_color = "#22c55e" if score >= 80 else ("#f59e0b" if score >= 60 else "#ef4444")

    reasons_html = "".join(f"<li>{r}</li>" for r in reasons)
    recs_html    = "".join(f"<li>{r}</li>" for r in recs)

    drift = report.get("reliability", {}).get("numeric_drift_ks_first_last", {}).get("top_10_ks", {})
    drift_rows = "".join(
        f"<tr><td>{col}</td><td>{val:.4f}</td>"
        f"<td style='color:{'#f59e0b' if float(val) > cfg_dict.get('drift_ks_threshold', 0.3) else '#22c55e'}'>"
        f"{'Above threshold' if float(val) > cfg_dict.get('drift_ks_threshold', 0.3) else 'OK'}</td></tr>"
        for col, val in drift.items() if val is not None
    )

    miss_top = report["quality"]["missingness"]["top_10_columns_missing_rate"]
    miss_rows = "".join(f"<tr><td>{col}</td><td>{val:.2%}</td></tr>" for col, val in miss_top.items())

    pii_rows = ""
    for col, hits in pii_cols.items():
        for pattern, rate in hits.items():
            pii_rows += f"<tr><td>{col}</td><td>{pattern}</td><td>{rate:.2%}</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Dataset Safety Report — {file_name}</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; padding: 0; background: #0f1117; color: #e2e8f0; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 40px 24px; }}
  h1 {{ font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 4px; }}
  h2 {{ font-size: 1.1rem; font-weight: 700; letter-spacing: -0.01em; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px; margin-top: 32px; }}
  .meta {{ color: #94a3b8; font-size: 0.85rem; margin-bottom: 32px; }}
  .score-ring {{ display: inline-flex; align-items: center; gap: 16px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; padding: 20px 28px; margin-bottom: 24px; }}
  .score-num {{ font-size: 3rem; font-weight: 900; color: {score_color}; line-height: 1; }}
  .score-label {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; }}
  .verdict {{ font-size: 1.1rem; font-weight: 700; }}
  .kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 24px 0; }}
  .kpi {{ background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 14px 16px; }}
  .kpi-t {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; }}
  .kpi-v {{ font-size: 1.5rem; font-weight: 800; margin: 4px 0; }}
  ul {{ padding-left: 1.2em; }}
  li {{ margin: 6px 0; line-height: 1.5; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 12px; }}
  th {{ background: rgba(255,255,255,0.06); padding: 8px 12px; text-align: left; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.07em; color: #94a3b8; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid rgba(255,255,255,0.06); }}
  .ok {{ color: #22c55e; }} .warn {{ color: #f59e0b; }} .bad {{ color: #ef4444; }}
  .sha {{ font-family: monospace; font-size: 0.75rem; word-break: break-all; color: #64748b; }}
  footer {{ margin-top: 48px; color: #475569; font-size: 0.78rem; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 16px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Dataset Safety Report</h1>
  <div class="meta">File: <strong>{file_name}</strong> &nbsp;·&nbsp; {df.shape[0]:,} rows × {df.shape[1]:,} columns &nbsp;·&nbsp; Mode: {cfg_dict.get('mode','—')} &nbsp;·&nbsp; Preset: {cfg_dict.get('preset','—')}</div>

  <div class="score-ring">
    <div>
      <div class="score-num">{score}</div>
      <div style="color:{score_color};font-weight:700;font-size:0.9rem;">Grade {grade}</div>
    </div>
    <div>
      <div class="score-label">Overall verdict</div>
      <div class="verdict">{verdict}</div>
    </div>
  </div>

  <div class="kpis">
    <div class="kpi"><div class="kpi-t">Rows</div><div class="kpi-v">{df.shape[0]:,}</div></div>
    <div class="kpi"><div class="kpi-t">Columns</div><div class="kpi-v">{df.shape[1]:,}</div></div>
    <div class="kpi"><div class="kpi-t">Missingness</div><div class="kpi-v {'warn' if miss_rate > 0.05 else 'ok'}">{miss_rate:.2%}</div></div>
    <div class="kpi"><div class="kpi-t">Duplicates</div><div class="kpi-v {'warn' if dup_rate > 0.01 else 'ok'}">{dup_rate:.2%}</div></div>
  </div>

  <h2>Findings</h2>
  <ul>{reasons_html}</ul>

  <h2>Recommended Actions</h2>
  <ul>{recs_html}</ul>

  <h2>Missingness — Top Columns</h2>
  <table><tr><th>Column</th><th>Missing Rate</th></tr>{miss_rows}</table>

  {"<h2>Numeric Drift (KS)</h2><table><tr><th>Column</th><th>KS Statistic</th><th>Status</th></tr>" + drift_rows + "</table>" if drift_rows else ""}

  {"<h2>PII Findings</h2><table><tr><th>Column</th><th>Pattern</th><th>Hit Rate</th></tr>" + pii_rows + "</table>" if pii_rows else ""}

  <h2>Integrity</h2>
  <div class="sha">SHA-256: {sha}</div>

  <footer>Generated by Unified Dataset Safety Analyzer &nbsp;·&nbsp; Heuristic report — validate with domain and legal review.</footer>
</div>
</body>
</html>"""
