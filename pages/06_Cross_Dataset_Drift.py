"""Cross-Dataset Drift — compare a baseline file against a candidate file.

Most ASTRID drift analysis looks at the first vs. last slice *within* a single
file. The real production question is usually different: did the data we just
got drift from the data we trained on? This page answers that — upload two
files (e.g. ``train.csv`` and ``prod.csv``) and the page reports per-column KS
statistics, missingness deltas, schema differences, and a verdict.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    SHARED_CSS, badge, kpi, sha256_bytes, numeric_cols, categorical_cols,
    ks_statistic, ks_statistic_with_pvalue, to_json_safe,
)


st.set_page_config(page_title="ASTRID — Cross-Dataset Drift", page_icon="🔀", layout="wide")
st.markdown(SHARED_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="dsa-card" style="padding:24px 28px 18px 28px; margin-bottom:16px;">
  <div style="display:flex; align-items:center; gap:10px;">
    <span style="font-size:1.8rem;">🔀</span>
    <div>
      <h2 style="margin:0;">Cross-Dataset Drift</h2>
      <div class="muted">Upload a baseline and a candidate dataset to measure drift between them.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Cached file readers ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=8)
def _read_tabular(_bytes: bytes, name: str) -> Tuple[pd.DataFrame, str]:
    lower = name.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(BytesIO(_bytes)), "CSV"
    if lower.endswith(".parquet"):
        return pd.read_parquet(BytesIO(_bytes)), "Parquet"
    if lower.endswith((".xls", ".xlsx")):
        return pd.read_excel(BytesIO(_bytes)), "Excel"
    raise ValueError(f"Unsupported file type: {name}")


# ─── Sidebar uploads + thresholds ─────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Files")
    base_up = st.file_uploader("Baseline (reference)", type=["csv", "parquet", "xls", "xlsx"], key="base")
    cand_up = st.file_uploader("Candidate (new)",      type=["csv", "parquet", "xls", "xlsx"], key="cand")
    st.divider()
    st.header("🎛 Threshold")
    ks_threshold = st.slider("KS warn threshold", 0.05, 0.80, 0.30, step=0.05,
                             help="Columns with a KS statistic above this value are flagged as drifted.")
    max_cols = st.number_input("Max numeric cols to compare", 5, 500, 100, step=5)
    st.divider()
    run = st.button("🔬 Compare datasets", type="primary", use_container_width=True)


if base_up is None or cand_up is None:
    st.info("⬆️ Upload a baseline file and a candidate file in the sidebar to begin.")
    st.stop()


try:
    base_df, base_kind = _read_tabular(base_up.getvalue(), base_up.name)
    cand_df, cand_kind = _read_tabular(cand_up.getvalue(), cand_up.name)
except Exception as e:
    st.error(f"Failed to read one of the files: {e}")
    st.stop()


# ─── Pre-run preview ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4, gap="large")
with c1: kpi("Baseline rows",  f"{base_df.shape[0]:,}", base_up.name)
with c2: kpi("Baseline cols",  f"{base_df.shape[1]:,}", base_kind)
with c3: kpi("Candidate rows", f"{cand_df.shape[0]:,}", cand_up.name)
with c4: kpi("Candidate cols", f"{cand_df.shape[1]:,}", cand_kind)


if not run:
    st.markdown("<br>", unsafe_allow_html=True)
    cprev1, cprev2 = st.columns(2, gap="large")
    with cprev1:
        st.subheader("Baseline preview")
        st.dataframe(base_df.head(20), use_container_width=True)
    with cprev2:
        st.subheader("Candidate preview")
        st.dataframe(cand_df.head(20), use_container_width=True)
    st.stop()


# ─── Schema comparison ────────────────────────────────────────────────────────
base_cols = set(base_df.columns)
cand_cols = set(cand_df.columns)
shared = sorted(base_cols & cand_cols)
only_base = sorted(base_cols - cand_cols)
only_cand = sorted(cand_cols - base_cols)


def _dtype_changes(shared_cols: List[str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for c in shared_cols:
        if str(base_df[c].dtype) != str(cand_df[c].dtype):
            out.append({"column": c,
                        "baseline_dtype": str(base_df[c].dtype),
                        "candidate_dtype": str(cand_df[c].dtype)})
    return out


dtype_changes = _dtype_changes(shared)


# ─── Per-column drift (KS) ────────────────────────────────────────────────────
num_shared = [c for c in shared if pd.api.types.is_numeric_dtype(base_df[c].dtype)
              and pd.api.types.is_numeric_dtype(cand_df[c].dtype)]
if max_cols and len(num_shared) > max_cols:
    # Rank by baseline variance to focus on informative columns first
    variances = [(c, float(pd.to_numeric(base_df[c], errors="coerce").var(skipna=True))) for c in num_shared]
    num_shared = [c for c, _ in sorted(variances, key=lambda kv: kv[1], reverse=True)[:max_cols]]


drift_rows: List[Dict[str, Any]] = []
for c in num_shared:
    x1 = pd.to_numeric(base_df[c], errors="coerce").to_numpy()
    x2 = pd.to_numeric(cand_df[c], errors="coerce").to_numpy()
    result = ks_statistic_with_pvalue(x1, x2)
    if result is None:
        continue
    ks, pval = result
    drift_rows.append({
        "column": c,
        "ks": float(ks),
        "p_value": (None if pval is None else float(pval)),
        "baseline_mean": float(np.nanmean(x1)) if len(x1) else None,
        "candidate_mean": float(np.nanmean(x2)) if len(x2) else None,
        "mean_delta": (float(np.nanmean(x2) - np.nanmean(x1)) if len(x1) and len(x2) else None),
        "flagged": float(ks) > ks_threshold,
    })

drift_rows.sort(key=lambda r: r["ks"], reverse=True)
flagged = [r for r in drift_rows if r["flagged"]]


# ─── Categorical / missingness drift ──────────────────────────────────────────
cat_shared = [c for c in shared if c not in num_shared
              and not pd.api.types.is_datetime64_any_dtype(base_df[c].dtype)]

miss_rows: List[Dict[str, Any]] = []
for c in shared:
    m1 = float(base_df[c].isna().mean())
    m2 = float(cand_df[c].isna().mean())
    if abs(m2 - m1) > 0.01:
        miss_rows.append({"column": c, "baseline_missing": m1,
                          "candidate_missing": m2, "delta": m2 - m1})
miss_rows.sort(key=lambda r: abs(r["delta"]), reverse=True)


# ─── Verdict banner ───────────────────────────────────────────────────────────
max_ks = max((r["ks"] for r in drift_rows), default=0.0)
if only_base or only_cand or dtype_changes:
    verdict_text = "Schema changed"
    vkind = "warn"
elif flagged:
    verdict_text = f"Drift detected ({len(flagged)} cols above KS {ks_threshold:.2f})"
    vkind = "warn"
elif max_ks > 0:
    verdict_text = f"No drift above threshold (max KS {max_ks:.3f})"
    vkind = "ok"
else:
    verdict_text = "No comparable numeric columns"
    vkind = "warn"

st.markdown(f"""
<div class="verdict-card">
  <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;">
    <div>
      <div class="verdict-title">Verdict</div>
      <div class="verdict-text">{verdict_text}</div>
      <div class="muted" style="margin-top:4px;">
        KS threshold: <span class="code-pill">{ks_threshold:.2f}</span>
        &nbsp; Shared cols: <span class="code-pill">{len(shared)}</span>
        &nbsp; Numeric compared: <span class="code-pill">{len(num_shared)}</span>
      </div>
    </div>
    <div>{badge(verdict_text, vkind)}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Schema delta panel ───────────────────────────────────────────────────────
with st.expander(f"🗂 Schema delta ({len(only_base)} dropped, {len(only_cand)} added, {len(dtype_changes)} dtype changes)",
                 expanded=bool(only_base or only_cand or dtype_changes)):
    cs1, cs2, cs3 = st.columns(3, gap="large")
    with cs1:
        st.markdown("**Only in baseline**")
        st.write(only_base or "_(none)_")
    with cs2:
        st.markdown("**Only in candidate**")
        st.write(only_cand or "_(none)_")
    with cs3:
        st.markdown("**Dtype changes**")
        if dtype_changes:
            st.dataframe(pd.DataFrame(dtype_changes), use_container_width=True, hide_index=True)
        else:
            st.write("_(none)_")


# ─── Drift table ──────────────────────────────────────────────────────────────
st.subheader("Numeric drift — Kolmogorov–Smirnov per column")
if drift_rows:
    drift_df = pd.DataFrame(drift_rows)
    st.dataframe(drift_df, use_container_width=True, hide_index=True,
                 column_config={
                     "ks": st.column_config.NumberColumn("KS", format="%.4f"),
                     "p_value": st.column_config.NumberColumn("p-value", format="%.4f"),
                     "baseline_mean": st.column_config.NumberColumn("Baseline mean", format="%.4f"),
                     "candidate_mean": st.column_config.NumberColumn("Candidate mean", format="%.4f"),
                     "mean_delta": st.column_config.NumberColumn("Δ mean", format="%.4f"),
                 })
else:
    st.info("No shared numeric columns had enough data for a KS comparison (need >=20 non-null values on each side).")


# ─── Missingness shift ────────────────────────────────────────────────────────
st.subheader("Missingness shift")
if miss_rows:
    st.dataframe(pd.DataFrame(miss_rows), use_container_width=True, hide_index=True,
                 column_config={
                     "baseline_missing": st.column_config.NumberColumn("Baseline missing", format="%.2%%"),
                     "candidate_missing": st.column_config.NumberColumn("Candidate missing", format="%.2%%"),
                     "delta": st.column_config.NumberColumn("Δ missing", format="%.2%%"),
                 })
else:
    st.success("No column shifted its missingness rate by more than 1 pp.")


# ─── Export ───────────────────────────────────────────────────────────────────
st.divider()
payload = {
    "baseline": {"name": base_up.name, "rows": int(base_df.shape[0]),
                 "cols": int(base_df.shape[1]), "sha256": sha256_bytes(base_up.getvalue())},
    "candidate": {"name": cand_up.name, "rows": int(cand_df.shape[0]),
                  "cols": int(cand_df.shape[1]), "sha256": sha256_bytes(cand_up.getvalue())},
    "ks_threshold": ks_threshold,
    "schema_delta": {
        "only_in_baseline": only_base,
        "only_in_candidate": only_cand,
        "dtype_changes": dtype_changes,
    },
    "drift": drift_rows,
    "missingness_shift": miss_rows,
    "verdict": verdict_text,
}
st.download_button(
    "⬇ Download drift report (JSON)",
    data=json.dumps(to_json_safe(payload), indent=2, ensure_ascii=False).encode("utf-8"),
    file_name="cross_dataset_drift.json",
    mime="application/json",
)
