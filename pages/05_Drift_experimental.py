# 05_Drift_Experimental.py
# Streamlit app: interactive concept-drift simulation & analysis dashboard.
#
# Methods compared:
#   - Static          — no supervision, baseline
#   - SAL             — sliding-window quantile active learning
#   - ADWIN-SAL       — drift-aware budgeted active learning
#   - Symbiosis-Edge  — two-threshold routing (oracle + human)
#
# Install:
#   pip install streamlit pandas numpy matplotlib
#
# Run:
#   streamlit run 05_Drift_Experimental.py

from __future__ import annotations

import io
import json
import math
import os
import sys
import base64
import hashlib
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Shared styling (optional import from sibling utils)
# ──────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from utils import SHARED_CSS
    HAS_UTILS = True
except Exception:
    HAS_UTILS = False
    SHARED_CSS = ""

st.set_page_config(
    page_title="Drift Simulation Analyzer (Experimental)",
    page_icon="🌊",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 3rem; max-width: 1300px; }
h1, h2, h3 { letter-spacing: -0.02em; }
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
.badge-ok   { background: rgba(0,200,0,0.12); }
.badge-warn { background: rgba(255,165,0,0.12); }
.badge-bad  { background: rgba(255,0,0,0.10); }
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
.config-key { font-weight:600; min-width:220px; }
.config-val { opacity:0.85; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:0.85rem; }
.muted { color: rgba(128,128,128,0.85); font-size:0.85rem; }
hr { border:none; height:1px; background:rgba(120,120,120,0.22); margin:1.2rem 0; }
.code-pill {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.85rem; padding: 2px 8px; border-radius: 999px;
  border: 1px solid rgba(120,120,120,0.25);
}
</style>
"""
if HAS_UTILS:
    st.markdown(SHARED_CSS, unsafe_allow_html=True)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tiny helpers
# ──────────────────────────────────────────────────────────────────────────────

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


def badge(label: str, kind: str) -> str:
    cls = {"ok": "badge badge-ok", "warn": "badge badge-warn",
           "bad": "badge badge-bad"}.get(kind, "badge")
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


def clip_text(s: str, n: int = 90) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Schema:
    dataset: str = "dataset"
    t: str = "t"
    method: str = "method"
    y_true: str = "y_true"
    y_pred: str = "y_pred"
    q_human: str = "q_human"
    q_oracle: str = "q_oracle"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    drift_t: Optional[int]


# ──────────────────────────────────────────────────────────────────────────────
# Rolling / statistics helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    return pd.Series(x).rolling(window=window, min_periods=1).mean().to_numpy()


def _auto_ylim(series: List[np.ndarray], pad: float = 0.10,
               min_top: float = 1.0) -> Tuple[float, float]:
    vmax = 0.0
    for s in series:
        if s is None or len(s) == 0:
            continue
        vmax = max(vmax, float(np.nanmax(s)))
    return 0.0, max(min_top, vmax * (1.0 + pad))


def _draw_drift(ax: plt.Axes, drift_t: Optional[int]) -> None:
    if drift_t is not None:
        ax.axvline(drift_t, linestyle="--", linewidth=1.1, color="grey",
                   alpha=0.55, label="drift")


def _keep_post_drift(df: pd.DataFrame, *, t_col: str,
                     drift_t: Optional[int]) -> pd.DataFrame:
    if drift_t is None:
        return df.copy()
    return df[df[t_col].astype(int) >= int(drift_t)].copy()


def _collapse_last(df: pd.DataFrame, *, t_col: str) -> pd.DataFrame:
    return df.sort_values(t_col).groupby(t_col, as_index=False).tail(1)


def _collapse_max(df: pd.DataFrame, *, t_col: str,
                  val_col: str) -> pd.DataFrame:
    return (df.sort_values(t_col)
              .groupby(t_col, as_index=False)[val_col].max())


# ──────────────────────────────────────────────────────────────────────────────
# Uncertainty primitives
# ──────────────────────────────────────────────────────────────────────────────

def _probs_from_pcorrect(p_correct: float, k: int) -> np.ndarray:
    p_correct = float(np.clip(p_correct, 1e-8, 1.0 - 1e-8))
    rest = (1.0 - p_correct) / max(1, k - 1)
    p = np.full(k, rest, dtype=float)
    p[0] = p_correct
    return p / p.sum()


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def _margin(p: np.ndarray) -> float:
    ps = np.sort(p)[::-1]
    return float(ps[0] - ps[1]) if ps.size >= 2 else float(ps[0])


def _uncertainty_score(p_correct: float, *, k: int,
                       alpha: float) -> float:
    p = _probs_from_pcorrect(p_correct, k)
    return float(_entropy(p) + alpha * (1.0 - _margin(p)))


def _quantile_threshold(values: np.ndarray, budget: float) -> float:
    if values.size == 0:
        return float("inf")
    return float(np.quantile(values, 1.0 - np.clip(budget, 0.0, 1.0)))


def _symbiosis_thresholds(
    window_u: np.ndarray, *, b_oracle: float, b_human: float,
) -> Tuple[float, float]:
    tau2 = _quantile_threshold(window_u, b_human)
    tau1 = _quantile_threshold(window_u, b_human + b_oracle)
    if tau1 > tau2:
        tau1 = tau2
    return tau1, tau2


# ──────────────────────────────────────────────────────────────────────────────
# ADWIN drift detector
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SimpleADWIN:
    max_window: int = 300
    min_window: int = 30
    delta: float = 0.08
    values: List[float] = field(default_factory=list)

    def update(self, x: float) -> bool:
        self.values.append(float(x))
        if len(self.values) > self.max_window:
            self.values.pop(0)
        n = len(self.values)
        if n < self.min_window:
            return False
        half = max(5, self.min_window // 2)
        for cut in range(half, n - half + 1):
            left = np.asarray(self.values[:cut], dtype=float)
            right = np.asarray(self.values[cut:], dtype=float)
            if left.size < half or right.size < half:
                continue
            gap = abs(left.mean() - right.mean())
            eps = np.sqrt(2.0 * np.log(2.0 / self.delta)
                          * (1.0 / left.size + 1.0 / right.size))
            if gap > eps:
                self.values = self.values[len(self.values) // 2:]
                return True
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Supervision update
# ──────────────────────────────────────────────────────────────────────────────

def _apply_supervision(
    state: float, *, rng: np.random.Generator,
    annotator_acc: float, lr_correct: float, lr_wrong: float,
) -> Tuple[float, bool]:
    correct = bool(rng.random() < annotator_acc)
    delta = lr_correct if correct else -lr_wrong
    return float(np.clip(state + delta, 0.05, 0.999)), correct


# ──────────────────────────────────────────────────────────────────────────────
# SimParams
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SimParams:
    n: int = 2000
    k_classes: int = 4
    drift_t: int = 500
    window_w: int = 200

    # Budgets
    b_sal: float = 0.12
    b_adwin_base: float = 0.10
    b_adwin_alarm: float = 0.28
    b_oracle: float = 0.12
    b_human: float = 0.05

    # ADWIN
    adwin_delta: float = 0.08
    adwin_max_window: int = 300
    adwin_min_window: int = 30
    adwin_alarm_window: int = 220
    adwin_lr_boost: float = 0.004

    # Uncertainty
    alpha_margin: float = 0.6
    u_floor: float = 0.0

    # Learning rates
    lr_edge: float = 0.001
    lr_oracle: float = 0.010
    lr_human: float = 0.016
    lr_oracle_wrong: float = 0.012
    lr_human_wrong: float = 0.010

    # Annotator reliability
    oracle_acc: float = 0.95
    human_acc: float = 0.99

    # Environment targets
    pre_acc_static: float = 0.92
    post_acc_static: float = 0.60
    pre_acc_sal: float = 0.93
    post_acc_sal: float = 0.55
    pre_acc_adwin: float = 0.93
    post_acc_adwin: float = 0.60
    pre_acc_sym: float = 0.93
    post_acc_sym: float = 0.55
    post_noise: float = 0.02

    # Cost model
    cost_edge_step: float = 0.0
    cost_oracle: float = 1.0
    cost_human: float = 10.0


# ──────────────────────────────────────────────────────────────────────────────
# Preset scenarios
# ──────────────────────────────────────────────────────────────────────────────

SCENARIO_PRESETS: Dict[str, Dict[str, SimParams]] = {
    "Paper defaults": {
        "SYNTHETIC": SimParams(
            n=2000, drift_t=500,
            b_sal=0.12, b_adwin_base=0.10, b_adwin_alarm=0.30,
            b_oracle=0.12, b_human=0.05,
            adwin_alarm_window=240, oracle_acc=0.95, human_acc=1.00,
            pre_acc_static=0.93, post_acc_static=0.60,
            pre_acc_sal=0.93, post_acc_sal=0.55,
            pre_acc_adwin=0.93, post_acc_adwin=0.61,
            pre_acc_sym=0.93, post_acc_sym=0.55,
        ),
        "SECOM": SimParams(
            n=2000, drift_t=500,
            b_sal=0.12, b_adwin_base=0.10, b_adwin_alarm=0.28,
            b_oracle=0.12, b_human=0.05,
            adwin_alarm_window=220, oracle_acc=0.95, human_acc=1.00,
            pre_acc_static=0.92, post_acc_static=0.62,
            pre_acc_sal=0.93, post_acc_sal=0.58,
            pre_acc_adwin=0.93, post_acc_adwin=0.61,
            pre_acc_sym=0.94, post_acc_sym=0.58,
        ),
        "APS": SimParams(
            n=2000, drift_t=500,
            b_sal=0.12, b_adwin_base=0.10, b_adwin_alarm=0.28,
            b_oracle=0.12, b_human=0.05,
            adwin_alarm_window=220, oracle_acc=0.95, human_acc=1.00,
            pre_acc_static=0.90, post_acc_static=0.60,
            pre_acc_sal=0.92, post_acc_sal=0.56,
            pre_acc_adwin=0.92, post_acc_adwin=0.60,
            pre_acc_sym=0.93, post_acc_sym=0.56,
        ),
    },
    "Severe drift": {
        "SYNTHETIC": SimParams(
            n=2000, drift_t=500,
            b_sal=0.15, b_adwin_base=0.12, b_adwin_alarm=0.35,
            b_oracle=0.15, b_human=0.08,
            adwin_alarm_window=280, oracle_acc=0.95, human_acc=1.00,
            pre_acc_static=0.93, post_acc_static=0.40,
            pre_acc_sal=0.93, post_acc_sal=0.35,
            pre_acc_adwin=0.93, post_acc_adwin=0.42,
            pre_acc_sym=0.93, post_acc_sym=0.35,
            post_noise=0.03,
        ),
    },
    "Gradual drift": {
        "SYNTHETIC": SimParams(
            n=3000, drift_t=800,
            b_sal=0.10, b_adwin_base=0.08, b_adwin_alarm=0.22,
            b_oracle=0.10, b_human=0.04,
            adwin_alarm_window=300, oracle_acc=0.95, human_acc=0.99,
            pre_acc_static=0.92, post_acc_static=0.70,
            pre_acc_sal=0.93, post_acc_sal=0.65,
            pre_acc_adwin=0.93, post_acc_adwin=0.72,
            pre_acc_sym=0.93, post_acc_sym=0.65,
            post_noise=0.015,
        ),
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Simulator
# ──────────────────────────────────────────────────────────────────────────────

METHODS = ("Static", "SAL", "ADWIN-SAL", "Symbiosis-Edge")

def simulate_one_run(*, dataset: str, seed: int,
                     params: SimParams) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n, k = params.n, params.k_classes
    y_true = rng.integers(0, k, size=n)

    state = {
        "Static": float(params.pre_acc_static),
        "SAL": float(params.pre_acc_sal),
        "ADWIN-SAL": float(params.pre_acc_adwin),
        "Symbiosis-Edge": float(params.pre_acc_sym),
    }

    u_hist: Dict[str, List[float]] = {m: [] for m in METHODS}
    rows: List[dict] = []

    adwin = SimpleADWIN(max_window=params.adwin_max_window,
                        min_window=params.adwin_min_window,
                        delta=params.adwin_delta)
    adwin_alarm_until = -1
    adwin_u_ema = None
    ema_alpha = 0.12

    for t in range(n):
        is_post = t >= params.drift_t
        noise = params.post_noise if is_post else 0.0

        targets = {
            "Static":         params.post_acc_static if is_post else params.pre_acc_static,
            "SAL":            params.post_acc_sal    if is_post else params.pre_acc_sal,
            "ADWIN-SAL":      params.post_acc_adwin  if is_post else params.pre_acc_adwin,
            "Symbiosis-Edge": params.post_acc_sym    if is_post else params.pre_acc_sym,
        }

        for method in METHODS:
            p = state[method]
            p = 0.995 * p + 0.005 * targets[method]
            p = float(np.clip(p + rng.normal(0.0, noise), 0.05, 0.999))
            state[method] = p

            u = _uncertainty_score(p, k=k, alpha=params.alpha_margin)
            u_hist[method].append(u)

            q_oracle = q_human = False
            oracle_ok = human_ok = np.nan

            # ── query policy ──
            if method == "SAL":
                w = np.array(u_hist[method][-params.window_w:])
                if u > _quantile_threshold(w, params.b_sal):
                    q_oracle = True

            elif method == "ADWIN-SAL":
                w = np.array(u_hist[method][-params.window_w:])
                b = params.b_adwin_alarm if t <= adwin_alarm_until \
                    else params.b_adwin_base
                if u > _quantile_threshold(w, b):
                    q_oracle = True

            elif method == "Symbiosis-Edge":
                w = np.array(u_hist[method][-params.window_w:])
                tau1, tau2 = _symbiosis_thresholds(
                    w, b_oracle=params.b_oracle, b_human=params.b_human)
                if u > tau2:
                    q_human = True
                elif u > tau1:
                    q_oracle = True

            # ── prediction ──
            if rng.random() < state[method]:
                y_pred = int(y_true[t])
            else:
                others = [c for c in range(k) if c != int(y_true[t])]
                y_pred = int(rng.choice(others))

            # ── ADWIN detector ──
            if method == "ADWIN-SAL":
                if adwin_u_ema is None:
                    adwin_u_ema = u
                else:
                    adwin_u_ema = (1 - ema_alpha) * adwin_u_ema + ema_alpha * u
                if adwin.update(adwin_u_ema):
                    adwin_alarm_until = max(adwin_alarm_until,
                                            t + params.adwin_alarm_window)

            # ── state updates ──
            if method == "SAL":
                if q_oracle:
                    state[method], oc = _apply_supervision(
                        state[method], rng=rng,
                        annotator_acc=params.oracle_acc,
                        lr_correct=params.lr_oracle,
                        lr_wrong=params.lr_oracle_wrong)
                    oracle_ok = bool(oc)

            elif method == "ADWIN-SAL":
                if q_oracle:
                    lr = params.lr_oracle + (params.adwin_lr_boost
                                             if t <= adwin_alarm_until else 0)
                    state[method], oc = _apply_supervision(
                        state[method], rng=rng,
                        annotator_acc=params.oracle_acc,
                        lr_correct=lr,
                        lr_wrong=params.lr_oracle_wrong)
                    oracle_ok = bool(oc)
                elif is_post and t <= adwin_alarm_until:
                    state[method] = float(np.clip(
                        state[method] + 0.5 * params.lr_edge, 0.05, 0.999))

            elif method == "Symbiosis-Edge":
                if q_human:
                    state[method], hc = _apply_supervision(
                        state[method], rng=rng,
                        annotator_acc=params.human_acc,
                        lr_correct=params.lr_human,
                        lr_wrong=params.lr_human_wrong)
                    human_ok = bool(hc)
                elif q_oracle:
                    state[method], oc = _apply_supervision(
                        state[method], rng=rng,
                        annotator_acc=params.oracle_acc,
                        lr_correct=params.lr_oracle,
                        lr_wrong=params.lr_oracle_wrong)
                    oracle_ok = bool(oc)
                elif is_post and u >= params.u_floor:
                    state[method] = float(np.clip(
                        state[method] + params.lr_edge, 0.05, 0.999))

            rows.append({
                "dataset": dataset, "t": t, "method": method,
                "y_true": int(y_true[t]), "y_pred": y_pred,
                "q_oracle": bool(q_oracle), "q_human": bool(q_human),
                "oracle_correct": oracle_ok, "human_correct": human_ok,
                "uncertainty": float(u),
            })

    return pd.DataFrame(rows)


def simulate_all(*, datasets: Sequence[DatasetConfig],
                 params_map: Dict[str, SimParams],
                 seed: int = 0) -> pd.DataFrame:
    return pd.concat(
        [simulate_one_run(dataset=ds.name, seed=seed, params=params_map[ds.name])
         for ds in datasets],
        ignore_index=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────────────────────

LABEL_MAP = {
    "Static": "Static", "SAL": "SAL",
    "ADWIN-SAL": "ADWIN-SAL", "Symbiosis-Edge": "Symbiosis",
}

def _method_summary(df: pd.DataFrame, *, schema: Schema,
                    drift_t: Optional[int],
                    params: SimParams) -> pd.DataFrame:
    d = _keep_post_drift(df, t_col=schema.t, drift_t=drift_t)
    if d.empty:
        return pd.DataFrame()
    d["__acc__"] = (d[schema.y_pred].to_numpy()
                    == d[schema.y_true].to_numpy()).astype(float)
    d["__qo__"] = d[schema.q_oracle].fillna(False).astype(int)
    d["__qh__"] = d[schema.q_human].fillna(False).astype(int)
    d["__qa__"] = np.maximum(d["__qo__"].to_numpy(), d["__qh__"].to_numpy())

    rows = []
    static_acc = None
    for m in METHODS:
        dm = d[d[schema.method] == m]
        if dm.empty:
            continue
        n_steps = int(dm[schema.t].nunique())
        gq = dm[[schema.t, "__qa__", "__qo__", "__qh__"]].groupby(
            schema.t, as_index=False).max()
        q_o = int(gq["__qo__"].sum())
        q_h = int(gq["__qh__"].sum())
        q_a = int(gq["__qa__"].sum())
        acc = float(dm["__acc__"].mean())
        if m == "Static":
            cost = 0.0; static_acc = acc
        elif m in ("SAL", "ADWIN-SAL"):
            cost = params.cost_human * q_o
        else:
            cost = (params.cost_edge_step * n_steps
                    + params.cost_oracle * q_o
                    + params.cost_human * q_h)
        rows.append({"method": LABEL_MAP[m], "acc": acc, "queries_oracle": q_o,
                      "queries_human": q_h, "queries_total": q_a,
                      "total_cost": cost, "n_steps": n_steps})
    out = pd.DataFrame(rows)
    if static_acc is not None and not out.empty:
        out["lift"] = out["acc"] - static_acc
        out["roi"] = np.where(out["total_cost"] > 0,
                              out["lift"] / out["total_cost"], np.nan)
    else:
        out["lift"] = 0.0
        out["roi"] = np.nan
    return out


def _annotator_obs(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    om = df["q_oracle"].fillna(False).astype(bool)
    hm = df["q_human"].fillna(False).astype(bool)
    if om.any():
        out["oracle_observed_acc"] = float(
            pd.to_numeric(df.loc[om, "oracle_correct"], errors="coerce").mean())
        out["oracle_queries"] = int(om.sum())
    if hm.any():
        out["human_observed_acc"] = float(
            pd.to_numeric(df.loc[hm, "human_correct"], errors="coerce").mean())
        out["human_queries"] = int(hm.sum())
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Matplotlib figures
# ──────────────────────────────────────────────────────────────────────────────

_PAPER_RC = {
    "figure.dpi": 200, "savefig.dpi": 300,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "lines.linewidth": 1.7, "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22,
    "grid.linestyle": "-", "grid.linewidth": 0.6,
    "text.usetex": False,
}

def _fig_to_buf(fig: plt.Figure, fmt: str = "png") -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_accuracy_and_queries(
    df_ds: pd.DataFrame, *, schema: Schema,
    smooth: int = 25, drift_t: Optional[int] = None,
    title: str = "", x_max: int = 2000,
) -> plt.Figure:
    with plt.rc_context(_PAPER_RC):
        fig = plt.figure(figsize=(7.0, 4.0))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3.0, 1.35], hspace=0.06)
        ax_top = fig.add_subplot(gs[0])
        ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

        df = df_ds.sort_values([schema.method, schema.t]).copy()
        df[schema.t] = df[schema.t].astype(int)
        color_map: Dict[str, str] = {}

        # ── top: rolling accuracy ──
        for method, gm in df.groupby(schema.method):
            g2 = _collapse_last(gm[[schema.t, schema.y_true, schema.y_pred]].dropna(),
                                t_col=schema.t)
            if g2.empty:
                continue
            correct = (g2[schema.y_pred].to_numpy()
                       == g2[schema.y_true].to_numpy()).astype(float)
            y = _rolling_mean(correct, smooth)
            line = ax_top.plot(g2[schema.t].to_numpy(int), y,
                               label=str(method))[0]
            color_map[str(method)] = line.get_color()

        _draw_drift(ax_top, drift_t)
        if title:
            ax_top.set_title(title)
        ax_top.set_ylabel(f"Rolling acc. (w={smooth})")
        ax_top.tick_params(labelbottom=False)
        ax_top.set_xlim(0, x_max)
        ax_top.legend(frameon=False, loc="lower left", handlelength=2.4)

        # ── bottom: cumulative queries ──
        t_support = np.arange(0, x_max + 1, dtype=int)
        sym_handles, sym_labels = [], []

        def _cum(m: str, col: str = "__qa__", ls: str = "-",
                 label: Optional[str] = None,
                 color: Optional[str] = None) -> None:
            dm = df[df[schema.method] == m].copy()
            if dm.empty:
                return
            dm["__qo__"] = dm[schema.q_oracle].fillna(False).astype(int)
            dm["__qh__"] = dm[schema.q_human].fillna(False).astype(int)
            dm["__qa__"] = np.maximum(dm["__qo__"].to_numpy(),
                                       dm["__qh__"].to_numpy())
            tmp = _collapse_max(dm[[schema.t, col]], t_col=schema.t,
                                val_col=col).sort_values(schema.t)
            full = pd.DataFrame({"t": t_support}).merge(
                tmp, on="t", how="left").fillna(0)
            cum = np.cumsum(full[col].to_numpy(int)).astype(float)
            c = color or color_map.get(m)
            h = ax_bot.plot(t_support, cum, color=c, linestyle=ls)[0]
            if label:
                sym_handles.append(h); sym_labels.append(label)

        for m in ("Static", "SAL", "ADWIN-SAL"):
            _cum(m)

        # Symbiosis detail
        dm_s = df[df[schema.method] == "Symbiosis-Edge"].copy()
        if not dm_s.empty:
            dm_s["__qo__"] = dm_s[schema.q_oracle].fillna(False).astype(int)
            dm_s["__qh__"] = dm_s[schema.q_human].fillna(False).astype(int)
            dm_s["__qa__"] = np.maximum(dm_s["__qo__"].to_numpy(),
                                         dm_s["__qh__"].to_numpy())
            _cum("Symbiosis-Edge", "__qa__", "-", "S-E: total", "tab:red")
            _cum("Symbiosis-Edge", "__qo__", ":", "S-E: oracle", "tab:red")
            _cum("Symbiosis-Edge", "__qh__", "--", "S-E: human", "tab:red")

        _draw_drift(ax_bot, drift_t)
        ax_bot.set_xlabel("Time step $t$")
        ax_bot.set_ylabel("Cum. queries")
        ax_bot.set_xlim(0, x_max)
        if sym_handles:
            ax_bot.legend(handles=sym_handles, labels=sym_labels,
                          frameon=False, loc="upper left",
                          bbox_to_anchor=(0.015, 0.98))

        for a in (ax_top, ax_bot):
            a.locator_params(axis="y", nbins=4)
        ax_bot.locator_params(axis="x", nbins=6)
        fig.subplots_adjust(left=0.10, right=0.99, top=0.92, bottom=0.14)
    return fig


def plot_cost_vs_accuracy(
    summary: pd.DataFrame, *, title: str = "",
) -> plt.Figure:
    with plt.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=(5.8, 3.7))
        for _, r in summary.iterrows():
            ax.scatter([r["total_cost"]], [r["acc"]], label=r["method"],
                       s=85, zorder=5)
        if title:
            ax.set_title(title)
        ax.set_xlabel("Total cost (post-drift)")
        ax.set_ylabel("Mean accuracy (post-drift)")
        ax.legend(frameon=False)
        fig.tight_layout()
    return fig


def plot_uncertainty_over_time(
    df_ds: pd.DataFrame, *, schema: Schema, smooth: int = 40,
    drift_t: Optional[int] = None, title: str = "",
    x_max: int = 2000,
) -> plt.Figure:
    with plt.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=(7.0, 2.5))
        df = df_ds.sort_values([schema.method, schema.t]).copy()
        for method, gm in df.groupby(schema.method):
            g2 = gm[[schema.t, "uncertainty"]].dropna().sort_values(schema.t)
            if g2.empty:
                continue
            y = _rolling_mean(g2["uncertainty"].to_numpy(), smooth)
            ax.plot(g2[schema.t].to_numpy(int), y, label=str(method))
        _draw_drift(ax, drift_t)
        ax.set_xlabel("Time step $t$")
        ax.set_ylabel(f"Uncertainty (w={smooth})")
        ax.set_xlim(0, x_max)
        ax.legend(frameon=False, loc="upper left")
        if title:
            ax.set_title(title)
        fig.tight_layout()
    return fig


def plot_query_rate_over_time(
    df_ds: pd.DataFrame, *, schema: Schema, bin_size: int = 50,
    drift_t: Optional[int] = None, title: str = "",
    x_max: int = 2000,
) -> plt.Figure:
    with plt.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=(7.0, 2.5))
        df = df_ds.copy()
        df["__qa__"] = np.maximum(
            df[schema.q_oracle].fillna(False).astype(int).to_numpy(),
            df[schema.q_human].fillna(False).astype(int).to_numpy())
        df["__bin__"] = (df[schema.t].astype(int) // bin_size) * bin_size
        for method, gm in df.groupby(schema.method):
            rates = gm.groupby("__bin__")["__qa__"].mean()
            ax.plot(rates.index, rates.values, label=str(method))
        _draw_drift(ax, drift_t)
        ax.set_xlabel("Time step $t$")
        ax.set_ylabel(f"Query rate (bin={bin_size})")
        ax.set_xlim(0, x_max)
        ax.legend(frameon=False, loc="upper left")
        if title:
            ax.set_title(title)
        fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX table generator
# ──────────────────────────────────────────────────────────────────────────────

def _make_latex_table(summary: pd.DataFrame, *, dataset: str,
                      params: SimParams) -> str:
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Method & \#queries & Total cost & Mean acc. & Lift & ROI \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        roi_s = "--" if (isinstance(r["roi"], float) and np.isnan(r["roi"])) \
                else f"{r['roi']:.6f}"
        lines.append(
            f"{r['method']} & {int(r['queries_total'])} & "
            f"{r['total_cost']:.2f} & {r['acc']:.4f} & "
            f"{r['lift']:.4f} & {roi_s} \\\\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        rf"\caption{{Cost summary on {dataset} (post-drift). "
        rf"Edge cost {params.cost_edge_step}, oracle {params.cost_oracle}, "
        rf"human {params.cost_human}. ROI = accuracy lift / total cost.}}",
        rf"\label{{tab:cost_{dataset.lower()}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────────

def _score_simulation(summary: pd.DataFrame) -> Dict[str, Any]:
    """Quick heuristic scoring of the simulation run."""
    if summary.empty:
        return {"total": 0, "grade": "F", "notes": ["No data."]}
    sym = summary[summary["method"] == "Symbiosis"]
    notes = []
    score = 50  # base

    if not sym.empty:
        acc = float(sym["acc"].iloc[0])
        lift = float(sym["lift"].iloc[0])
        roi = float(sym["roi"].iloc[0]) if not np.isnan(float(sym["roi"].iloc[0])) else 0
        if acc > 0.80:
            score += 15; notes.append(f"Symbiosis acc {acc:.4f} > 0.80")
        elif acc > 0.70:
            score += 8;  notes.append(f"Symbiosis acc {acc:.4f} > 0.70")
        if lift > 0.10:
            score += 15; notes.append(f"Symbiosis lift {lift:.4f} > 0.10")
        elif lift > 0.05:
            score += 8;  notes.append(f"Symbiosis lift {lift:.4f} > 0.05")
        if roi > 0.001:
            score += 10; notes.append(f"Positive ROI {roi:.6f}")
    else:
        notes.append("Symbiosis method not found.")

    # Check all methods have data
    methods_found = set(summary["method"].tolist())
    for m in ("Static", "SAL", "ADWIN-SAL", "Symbiosis"):
        if m not in methods_found:
            score -= 5
            notes.append(f"Missing method: {m}")

    score = max(0, min(100, score))
    grade = ("A" if score >= 90 else "B" if score >= 80
             else "C" if score >= 70 else "D" if score >= 60 else "F")
    return {"total": score, "grade": grade, "notes": notes}


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ──

def _find_logo() -> Optional[str]:
    here = Path(__file__).resolve().parent
    for c in (here / "logo.png", here / "assets" / "logo.png",
              here.parent / "assets" / "logo.png", here.parent / "logo.png"):
        if c.exists():
            return str(c)
    return None

_lp = _find_logo()
if _lp:
    try:
        _b64 = base64.b64encode(Path(_lp).read_bytes()).decode("ascii")
        _logo_html = f'<img src="data:image/png;base64,{_b64}" style="width:34px;height:34px;object-fit:contain;border-radius:8px;" />'
    except Exception:
        _logo_html = '<span style="font-size:1.8rem;">🌊</span>'
else:
    _logo_html = '<span style="font-size:1.8rem;">🌊</span>'

st.markdown(f"""
<div class="dsa-card" style="padding:24px 28px 18px 28px; margin-bottom:16px;">
  <div style="display:flex; align-items:center; gap:10px;">
    {_logo_html}
    <div>
      <h2 style="margin:0;">Drift Simulation Analyzer
        <span style="font-size:0.7em; opacity:0.6;">(Experimental)</span></h2>
      <div class="muted">Interactive concept-drift simulation comparing Static,
      SAL, ADWIN-SAL, and Symbiosis-Edge active-learning strategies.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──

with st.sidebar:
    st.header("Scenario")
    preset_name = st.selectbox("Preset", list(SCENARIO_PRESETS.keys()))
    preset_params = SCENARIO_PRESETS[preset_name]
    datasets_in_preset = sorted(preset_params.keys())

    selected_datasets = st.multiselect(
        "Datasets to simulate",
        datasets_in_preset,
        default=datasets_in_preset,
    )
    if not selected_datasets:
        st.warning("Select at least one dataset.")
        st.stop()

    seed = st.number_input("Random seed", 0, 99999, 0)

    st.header("Overrides")
    st.caption("Override the preset for the first selected dataset. "
               "Leave defaults to use the preset values.")

    ref = preset_params[selected_datasets[0]]

    with st.expander("Environment"):
        n_steps = st.number_input("Total time steps", 500, 50000,
                                   int(ref.n), step=100)
        drift_t = st.number_input("Drift time", 50, 40000,
                                   int(ref.drift_t), step=50)
        k_classes = st.number_input("Num classes", 2, 20,
                                     int(ref.k_classes), step=1)
        post_noise = st.slider("Post-drift noise σ", 0.0, 0.10,
                                float(ref.post_noise), 0.005)

    with st.expander("Pre/post accuracy targets"):
        col_a, col_b = st.columns(2)
        with col_a:
            pre_static = st.slider("Pre-static", 0.5, 1.0,
                                    float(ref.pre_acc_static), 0.01)
            pre_sal = st.slider("Pre-SAL", 0.5, 1.0,
                                 float(ref.pre_acc_sal), 0.01)
            pre_adwin = st.slider("Pre-ADWIN", 0.5, 1.0,
                                   float(ref.pre_acc_adwin), 0.01)
            pre_sym = st.slider("Pre-Symbiosis", 0.5, 1.0,
                                 float(ref.pre_acc_sym), 0.01)
        with col_b:
            post_static = st.slider("Post-static", 0.1, 1.0,
                                     float(ref.post_acc_static), 0.01)
            post_sal = st.slider("Post-SAL", 0.1, 1.0,
                                  float(ref.post_acc_sal), 0.01)
            post_adwin = st.slider("Post-ADWIN", 0.1, 1.0,
                                    float(ref.post_acc_adwin), 0.01)
            post_sym = st.slider("Post-Symbiosis", 0.1, 1.0,
                                  float(ref.post_acc_sym), 0.01)

    with st.expander("Budgets"):
        b_sal = st.slider("SAL budget", 0.01, 0.40,
                           float(ref.b_sal), 0.01)
        b_adwin_base = st.slider("ADWIN base budget", 0.01, 0.40,
                                  float(ref.b_adwin_base), 0.01)
        b_adwin_alarm = st.slider("ADWIN alarm budget", 0.05, 0.60,
                                   float(ref.b_adwin_alarm), 0.01)
        b_oracle = st.slider("Symbiosis oracle budget", 0.01, 0.40,
                              float(ref.b_oracle), 0.01)
        b_human = st.slider("Symbiosis human budget", 0.01, 0.30,
                             float(ref.b_human), 0.01)

    with st.expander("Learning rates & annotator reliability"):
        lr_oracle = st.slider("LR oracle", 0.001, 0.05,
                               float(ref.lr_oracle), 0.001,
                               format="%.3f")
        lr_human = st.slider("LR human", 0.001, 0.05,
                              float(ref.lr_human), 0.001, format="%.3f")
        lr_edge = st.slider("LR edge", 0.0001, 0.01,
                             float(ref.lr_edge), 0.0005, format="%.4f")
        oracle_acc = st.slider("Oracle accuracy", 0.5, 1.0,
                                float(ref.oracle_acc), 0.01)
        human_acc = st.slider("Human accuracy", 0.5, 1.0,
                               float(ref.human_acc), 0.01)

    with st.expander("ADWIN parameters"):
        adwin_delta = st.slider("ADWIN δ", 0.01, 0.30,
                                 float(ref.adwin_delta), 0.01)
        adwin_alarm_win = st.number_input("ADWIN alarm window",
                                           50, 1000,
                                           int(ref.adwin_alarm_window), 10)
        adwin_lr_boost = st.slider("ADWIN LR boost", 0.0, 0.02,
                                    float(ref.adwin_lr_boost), 0.001,
                                    format="%.3f")

    with st.expander("Cost model"):
        cost_edge = st.number_input("Edge cost / step", 0.0, 100.0,
                                     float(ref.cost_edge_step), 0.1)
        cost_oracle_val = st.number_input("Oracle cost / query", 0.0, 100.0,
                                           float(ref.cost_oracle), 0.5)
        cost_human_val = st.number_input("Human cost / query", 0.0, 500.0,
                                          float(ref.cost_human), 1.0)

    smooth_window = st.number_input("Smoothing window", 1, 200, 25, 5)
    query_bin = st.number_input("Query-rate bin size", 10, 500, 50, 10)

    st.divider()
    run = st.button("Run simulation", type="primary",
                     use_container_width=True)


# ── Build params ──

def _build_params(ref_p: SimParams) -> SimParams:
    return SimParams(
        n=n_steps, k_classes=k_classes, drift_t=drift_t,
        window_w=int(ref_p.window_w),
        b_sal=b_sal, b_adwin_base=b_adwin_base,
        b_adwin_alarm=b_adwin_alarm,
        b_oracle=b_oracle, b_human=b_human,
        adwin_delta=adwin_delta,
        adwin_max_window=int(ref_p.adwin_max_window),
        adwin_min_window=int(ref_p.adwin_min_window),
        adwin_alarm_window=adwin_alarm_win,
        adwin_lr_boost=adwin_lr_boost,
        alpha_margin=float(ref_p.alpha_margin),
        u_floor=float(ref_p.u_floor),
        lr_edge=lr_edge, lr_oracle=lr_oracle, lr_human=lr_human,
        lr_oracle_wrong=float(ref_p.lr_oracle_wrong),
        lr_human_wrong=float(ref_p.lr_human_wrong),
        oracle_acc=oracle_acc, human_acc=human_acc,
        pre_acc_static=pre_static, post_acc_static=post_static,
        pre_acc_sal=pre_sal, post_acc_sal=post_sal,
        pre_acc_adwin=pre_adwin, post_acc_adwin=post_adwin,
        pre_acc_sym=pre_sym, post_acc_sym=post_sym,
        post_noise=post_noise,
        cost_edge_step=cost_edge,
        cost_oracle=cost_oracle_val,
        cost_human=cost_human_val,
    )


params_map: Dict[str, SimParams] = {}
ds_configs: List[DatasetConfig] = []
for ds_name in selected_datasets:
    ref_p = preset_params[ds_name]
    if ds_name == selected_datasets[0]:
        p = _build_params(ref_p)
    else:
        # Use preset for non-first datasets but propagate cost model
        p = SimParams(
            **{**{k: getattr(ref_p, k) for k in ref_p.__dataclass_fields__},
               "cost_edge_step": cost_edge, "cost_oracle": cost_oracle_val,
               "cost_human": cost_human_val}
        )
    params_map[ds_name] = p
    ds_configs.append(DatasetConfig(ds_name, p.drift_t))


# ── Preview before run ──

if not run:
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        kpi("Datasets", str(len(selected_datasets)),
            ", ".join(selected_datasets))
    with c2:
        kpi("Time steps", f"{n_steps:,}", f"Drift at t={drift_t}")
    with c3:
        kpi("Methods", str(len(METHODS)), ", ".join(METHODS))
    with c4:
        kpi("Seed", str(seed), "Reproducible run")

    st.info("Configure parameters in the sidebar, then click **Run simulation**.")
    st.stop()


# ── Simulate ──

with st.spinner("Running simulation…"):
    df_all = simulate_all(datasets=ds_configs, params_map=params_map,
                          seed=int(seed))

schema = Schema()

# ── Per-dataset summaries ──

summaries: Dict[str, pd.DataFrame] = {}
scores: Dict[str, Dict[str, Any]] = {}
for ds in ds_configs:
    df_ds = df_all[df_all[schema.dataset] == ds.name].copy()
    s = _method_summary(df_ds, schema=schema, drift_t=ds.drift_t,
                        params=params_map[ds.name])
    summaries[ds.name] = s
    scores[ds.name] = _score_simulation(s)

# best/worst
best_ds = max(scores, key=lambda d: scores[d]["total"])
total_score = scores[best_ds]["total"]
grade = scores[best_ds]["grade"]

# ── Verdict ──

reasons: List[str] = []
recs: List[str] = []
for ds_name, s in summaries.items():
    sym = s[s["method"] == "Symbiosis"]
    if not sym.empty:
        acc = float(sym["acc"].iloc[0])
        lift = float(sym["lift"].iloc[0])
        if lift > 0.05:
            reasons.append(f"{ds_name}: Symbiosis lifts accuracy by {lift:.4f}.")
        else:
            reasons.append(f"{ds_name}: Symbiosis lift only {lift:.4f}.")
        if acc < 0.70:
            recs.append(f"{ds_name}: Post-drift accuracy low ({acc:.4f}). "
                        "Increase budgets or human involvement.")
    adw = s[s["method"] == "ADWIN-SAL"]
    if not adw.empty:
        adw_lift = float(adw["lift"].iloc[0])
        if adw_lift > 0.10:
            reasons.append(f"{ds_name}: ADWIN-SAL effective (lift {adw_lift:.4f}).")

if not reasons:
    reasons = ["Simulation completed. Review per-dataset results."]
if not recs:
    recs = ["Run with different seeds to check stability. "
            "Export results for detailed analysis."]

verdict_text = "Strong recovery" if total_score >= 80 else (
    "Partial recovery" if total_score >= 60 else "Weak recovery")
verdict_kind = "ok" if total_score >= 80 else (
    "warn" if total_score >= 60 else "bad")

st.markdown(f"""
<div class="verdict-card">
  <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;">
    <div>
      <div style="font-weight:700; font-size:1.1rem;">Verdict</div>
      <div style="font-size:1.3rem; font-weight:600;">{verdict_text}</div>
      <div class="muted" style="margin-top:4px;">
        Preset: <span class="code-pill">{preset_name}</span>
        &nbsp; Seed: <span class="code-pill">{seed}</span>
        &nbsp; Datasets: <span class="code-pill">{', '.join(selected_datasets)}</span>
      </div>
    </div>
    <div style="display:flex; gap:8px; flex-wrap:wrap;">
      {badge(verdict_text, verdict_kind)}
      {badge(f"Score {total_score}/100", 'ok' if total_score>=80 else ('warn' if total_score>=60 else 'bad'))}
      {badge(f"Grade {grade}", 'ok' if grade in ('A','B') else ('warn' if grade=='C' else 'bad'))}
    </div>
  </div>
  <hr>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
    <div><div style="font-weight:700; font-size:0.85rem; margin-bottom:6px;">Findings</div>
         <ul style="margin:0; padding-left:1.1em; font-size:0.83rem; line-height:1.7;">
         {''.join(f'<li>{clip_text(r,200)}</li>' for r in reasons)}</ul></div>
    <div><div style="font-weight:700; font-size:0.85rem; margin-bottom:6px;">Recommendations</div>
         <ul style="margin:0; padding-left:1.1em; font-size:0.83rem; line-height:1.7;">
         {''.join(f'<li>{clip_text(r,200)}</li>' for r in recs)}</ul></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ──

c1, c2, c3, c4, c5 = st.columns(5, gap="large")
with c1:
    kpi("Datasets", str(len(selected_datasets)),
        ", ".join(selected_datasets))
with c2:
    kpi("Steps/dataset", f"{n_steps:,}", f"Drift at t={drift_t}")
with c3:
    kpi("Total rows", f"{len(df_all):,}", "All methods × steps")
with c4:
    kpi("Methods", str(len(METHODS)), "Compared")
with c5:
    best_acc = ""
    for s in summaries.values():
        sym = s[s["method"] == "Symbiosis"]
        if not sym.empty:
            best_acc = f"{float(sym['acc'].iloc[0]):.4f}"
            break
    kpi("Best Symbiosis acc", best_acc or "N/A", "Post-drift")

st.write("")

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────

tab_names = ["Overview"]
for ds in ds_configs:
    tab_names.append(f"📈 {ds.name}")
tab_names += ["Cost analysis", "Diagnostics", "Transparency", "Export"]
tabs = st.tabs(tab_names)

# ── Overview ──
with tabs[0]:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Overview — all datasets")

    for ds in ds_configs:
        s = summaries[ds.name]
        sc = scores[ds.name]
        st.markdown(f"**{ds.name}** — Score {sc['total']}/100 (Grade {sc['grade']})")
        if not s.empty:
            disp = s[["method", "acc", "lift", "queries_total",
                       "total_cost", "roi"]].copy()
            disp.columns = ["Method", "Mean acc", "Lift", "Queries",
                            "Total cost", "ROI"]
            st.dataframe(disp, use_container_width=True, hide_index=True)
        if sc["notes"]:
            with st.expander("Scoring notes"):
                for n in sc["notes"]:
                    st.write(f"• {n}")
        st.write("---")

    st.markdown("</div>", unsafe_allow_html=True)

# ── Per-dataset tabs ──
for idx, ds in enumerate(ds_configs):
    with tabs[1 + idx]:
        st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
        st.subheader(f"{ds.name} — Detailed results")

        df_ds = df_all[df_all[schema.dataset] == ds.name].copy()
        p = params_map[ds.name]

        # Accuracy + queries
        fig_aq = plot_accuracy_and_queries(
            df_ds, schema=schema, smooth=smooth_window,
            drift_t=ds.drift_t,
            title=f"{ds.name}: accuracy + cumulative queries",
            x_max=p.n)
        st.pyplot(fig_aq)

        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            # Uncertainty
            fig_u = plot_uncertainty_over_time(
                df_ds, schema=schema, smooth=smooth_window,
                drift_t=ds.drift_t,
                title=f"{ds.name}: uncertainty over time", x_max=p.n)
            st.pyplot(fig_u)

        with col_b:
            # Query rate
            fig_qr = plot_query_rate_over_time(
                df_ds, schema=schema, bin_size=query_bin,
                drift_t=ds.drift_t,
                title=f"{ds.name}: query rate over time", x_max=p.n)
            st.pyplot(fig_qr)

        # Summary table
        st.write("**Post-drift summary**")
        s = summaries[ds.name]
        if not s.empty:
            st.dataframe(s, use_container_width=True, hide_index=True)

        # Annotator observations
        ann = _annotator_obs(df_ds)
        if ann:
            st.write("**Observed annotator accuracy**")
            st.json(to_json_safe(ann))

        st.markdown("</div>", unsafe_allow_html=True)

# ── Cost analysis ──
with tabs[1 + len(ds_configs)]:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Cost analysis")

    for ds in ds_configs:
        s = summaries[ds.name]
        if s.empty:
            continue

        st.markdown(f"#### {ds.name}")

        col1, col2 = st.columns(2, gap="large")
        with col1:
            fig_cv = plot_cost_vs_accuracy(
                s, title=f"{ds.name}: cost vs accuracy")
            st.pyplot(fig_cv)

        with col2:
            disp = s[["method", "acc", "queries_total", "total_cost",
                       "lift", "roi"]].copy()
            disp.columns = ["Method", "Acc", "Queries", "Cost",
                            "Lift", "ROI"]
            st.dataframe(disp, use_container_width=True, hide_index=True)

        # LaTeX
        with st.expander(f"LaTeX table — {ds.name}"):
            tex = _make_latex_table(s, dataset=ds.name,
                                    params=params_map[ds.name])
            st.code(tex, language="latex")

        st.write("---")

    st.markdown("</div>", unsafe_allow_html=True)

# ── Diagnostics ──
with tabs[2 + len(ds_configs)]:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Diagnostics")

    for ds in ds_configs:
        df_ds = df_all[df_all[schema.dataset] == ds.name].copy()
        st.markdown(f"#### {ds.name}")

        # Pre vs post accuracy
        pre = df_ds[df_ds["t"] < ds.drift_t]
        post = df_ds[df_ds["t"] >= ds.drift_t]
        pre_acc = {}; post_acc = {}
        for m in METHODS:
            pm = pre[pre["method"] == m]
            pom = post[post["method"] == m]
            if not pm.empty:
                pre_acc[m] = float((pm["y_pred"].to_numpy()
                                    == pm["y_true"].to_numpy()).mean())
            if not pom.empty:
                post_acc[m] = float((pom["y_pred"].to_numpy()
                                     == pom["y_true"].to_numpy()).mean())

        diag_rows = []
        for m in METHODS:
            diag_rows.append({
                "Method": m,
                "Pre-drift acc": f"{pre_acc.get(m, 0):.4f}",
                "Post-drift acc": f"{post_acc.get(m, 0):.4f}",
                "Drop": f"{pre_acc.get(m, 0) - post_acc.get(m, 0):.4f}",
            })
        st.dataframe(pd.DataFrame(diag_rows), use_container_width=True,
                      hide_index=True)

        # Annotator stats
        ann = _annotator_obs(df_ds)
        if ann:
            st.write("Annotator observations")
            st.json(to_json_safe(ann))

        # ADWIN alarm events
        st.write(f"ADWIN parameters: δ={params_map[ds.name].adwin_delta}, "
                 f"alarm window={params_map[ds.name].adwin_alarm_window}")
        st.write("---")

    st.markdown("</div>", unsafe_allow_html=True)

# ── Transparency ──
with tabs[3 + len(ds_configs)]:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Transparency — Configuration & Check Registry")

    # Config table
    st.markdown('<div style="font-weight:700; margin-bottom:8px;">⚙️ Simulation configuration</div>',
                unsafe_allow_html=True)
    for ds in ds_configs:
        p = params_map[ds.name]
        st.markdown(f"**{ds.name}**")
        cfg_rows = [
            ("Time steps", str(p.n)),
            ("Drift time", str(p.drift_t)),
            ("Classes", str(p.k_classes)),
            ("SAL budget", f"{p.b_sal:.2f}"),
            ("ADWIN base/alarm budget", f"{p.b_adwin_base:.2f} / {p.b_adwin_alarm:.2f}"),
            ("Oracle/human budget", f"{p.b_oracle:.2f} / {p.b_human:.2f}"),
            ("LR oracle/human/edge", f"{p.lr_oracle:.3f} / {p.lr_human:.3f} / {p.lr_edge:.4f}"),
            ("Oracle/human accuracy", f"{p.oracle_acc:.2f} / {p.human_acc:.2f}"),
            ("Pre-acc (Static/SAL/ADWIN/Sym)",
             f"{p.pre_acc_static}/{p.pre_acc_sal}/{p.pre_acc_adwin}/{p.pre_acc_sym}"),
            ("Post-acc (Static/SAL/ADWIN/Sym)",
             f"{p.post_acc_static}/{p.post_acc_sal}/{p.post_acc_adwin}/{p.post_acc_sym}"),
            ("Cost model (edge/oracle/human)",
             f"{p.cost_edge_step} / {p.cost_oracle} / {p.cost_human}"),
        ]
        html_c = "<div>"
        for k, v in cfg_rows:
            html_c += (f'<div class="config-row"><div class="config-key">{k}</div>'
                       f'<div class="config-val mono">{v}</div></div>')
        html_c += "</div>"
        st.markdown(html_c, unsafe_allow_html=True)
        st.write("---")

    # Check registry
    st.markdown('<div style="font-weight:700; margin-bottom:8px;">📋 Check registry</div>',
                unsafe_allow_html=True)
    checks = [
        ("Accuracy simulation (4 methods)", "✓ Ran"),
        ("Uncertainty tracking (entropy + margin)", "✓ Ran"),
        ("Query budget enforcement (quantile)", "✓ Ran"),
        ("ADWIN drift detection", "✓ Ran"),
        ("Symbiosis two-threshold routing", "✓ Ran"),
        ("Annotator accuracy simulation", "✓ Ran"),
        ("Post-drift cost accounting", "✓ Ran"),
        ("ROI computation", "✓ Ran"),
    ]
    st.dataframe(pd.DataFrame(checks, columns=["Check", "Status"]),
                  use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ── Export ──
with tabs[4 + len(ds_configs)]:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Export")

    col_e1, col_e2, col_e3 = st.columns(3, gap="large")

    # JSON
    with col_e1:
        st.markdown("**JSON report**")
        export_payload = {
            "preset": preset_name,
            "seed": int(seed),
            "datasets": [ds.name for ds in ds_configs],
            "scores": to_json_safe(scores),
            "summaries": {ds: to_json_safe(s.to_dict(orient="records"))
                          for ds, s in summaries.items()},
            "params": {ds: to_json_safe(asdict(p))
                       for ds, p in params_map.items()},
            "verdict": verdict_text,
            "reasons": reasons,
            "recommendations": recs,
        }
        json_bytes = json.dumps(export_payload, indent=2,
                                ensure_ascii=False).encode("utf-8")
        st.download_button("⬇ Download JSON", data=json_bytes,
                           file_name="drift_simulation_report.json",
                           mime="application/json",
                           use_container_width=True)

    # CSV
    with col_e2:
        st.markdown("**Raw simulation CSV**")
        csv_bytes = df_all.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download CSV", data=csv_bytes,
                           file_name="drift_simulation_raw.csv",
                           mime="text/csv",
                           use_container_width=True)

    # Markdown
    with col_e3:
        st.markdown("**Markdown summary**")
        md_lines = [
            "# Drift Simulation Report", "",
            f"- **Preset:** {preset_name}",
            f"- **Seed:** {seed}",
            f"- **Verdict:** {verdict_text} (Score {total_score}/100, Grade {grade})",
            "",
        ]
        for ds in ds_configs:
            s = summaries[ds.name]
            sc = scores[ds.name]
            md_lines += [
                f"## {ds.name}", "",
                f"- Score: {sc['total']}/100 (Grade {sc['grade']})",
                f"- Drift at t={ds.drift_t}, n={params_map[ds.name].n}", "",
            ]
            if not s.empty:
                md_lines.append("| Method | Acc | Lift | Queries | Cost | ROI |")
                md_lines.append("|--------|-----|------|---------|------|-----|")
                for _, r in s.iterrows():
                    roi_s = f"{r['roi']:.6f}" if not np.isnan(r['roi']) else "--"
                    md_lines.append(
                        f"| {r['method']} | {r['acc']:.4f} | "
                        f"{r['lift']:.4f} | {int(r['queries_total'])} | "
                        f"{r['total_cost']:.2f} | {roi_s} |")
                md_lines.append("")
        md_lines += [
            "## Findings",
            *[f"- {r}" for r in reasons], "",
            "## Recommendations",
            *[f"- {r}" for r in recs], "",
            "---",
            "*Simulated data. Validate with real deployment.*",
        ]
        md_bytes = "\n".join(md_lines).encode("utf-8")
        st.download_button("⬇ Download Markdown", data=md_bytes,
                           file_name="drift_simulation_report.md",
                           mime="text/markdown",
                           use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Figures download
    st.markdown("**Figures**")
    fig_cols = st.columns(len(ds_configs), gap="small")
    for idx_d, ds in enumerate(ds_configs):
        with fig_cols[idx_d]:
            df_ds = df_all[df_all[schema.dataset] == ds.name].copy()
            fig_dl = plot_accuracy_and_queries(
                df_ds, schema=schema, smooth=smooth_window,
                drift_t=ds.drift_t,
                title=f"{ds.name}", x_max=params_map[ds.name].n)
            png = _fig_to_buf(fig_dl, "png")
            st.download_button(
                f"⬇ {ds.name} (PNG)",
                data=png,
                file_name=f"{ds.name.lower()}_accuracy_queries.png",
                mime="image/png",
                use_container_width=True,
            )

    with st.expander("Raw JSON (preview)"):
        st.json(export_payload)

    st.markdown("</div>", unsafe_allow_html=True)