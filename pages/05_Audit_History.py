"""Audit history, report comparison, and policy gate."""
from __future__ import annotations

import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from audit_history import (  # noqa: E402
    DEFAULT_POLICY,
    compare_reports,
    evaluate_policy,
    load_audit_runs,
    summarize_run,
)
from utils import SHARED_CSS  # noqa: E402

st.set_page_config(page_title="Audit History", page_icon="🧭", layout="wide")
st.markdown(SHARED_CSS, unsafe_allow_html=True)


def _run_label(run: dict) -> str:
    created = str(run.get("created_at_utc", "unknown"))
    analyzer = str(run.get("analyzer", "analyzer"))
    dataset = str(run.get("dataset_name", "dataset"))
    score = run.get("score", "NA")
    grade = run.get("grade", "NA")
    run_id = str(run.get("run_id", ""))[-18:]
    return f"{created} | {analyzer} | {dataset} | {score}/100 {grade} | {run_id}"


def _format_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


st.markdown(
    """
<div class="dsa-card" style="padding:24px 28px 18px 28px; margin-bottom:16px;">
  <div style="display:flex; align-items:center; gap:10px;">
    <span style="font-size:1.8rem;">🧭</span>
    <div>
      <h2 style="margin:0;">Audit History</h2>
      <div class="muted">Compare dataset audits, inspect policy gates, and track reliability over time.</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

runs = load_audit_runs()
if not runs:
    st.info("No saved audit runs yet. Run the Tabular, Time Series, or Image analyzer to create history.")
    st.stop()

with st.sidebar:
    st.header("Policy Gate")
    st.caption("Adjust the default deployment gate for this review.")
    policy = {
        "min_health_score": st.number_input(
            "Minimum health score",
            min_value=0,
            max_value=100,
            value=int(DEFAULT_POLICY["min_health_score"]),
            step=1,
        ),
        "max_missingness": st.number_input(
            "Max missingness",
            min_value=0.0,
            max_value=1.0,
            value=float(DEFAULT_POLICY["max_missingness"]),
            step=0.01,
            format="%.3f",
        ),
        "max_duplicate_rate": st.number_input(
            "Max duplicate rate",
            min_value=0.0,
            max_value=1.0,
            value=float(DEFAULT_POLICY["max_duplicate_rate"]),
            step=0.01,
            format="%.3f",
        ),
        "max_split_leakage": st.number_input(
            "Max split leakage",
            min_value=0.0,
            max_value=1.0,
            value=float(DEFAULT_POLICY["max_split_leakage"]),
            step=0.001,
            format="%.4f",
        ),
        "max_drift_ks": st.number_input(
            "Max drift KS",
            min_value=0.0,
            max_value=1.0,
            value=float(DEFAULT_POLICY["max_drift_ks"]),
            step=0.01,
            format="%.3f",
        ),
        "allow_pii": st.toggle("Allow PII flags", value=bool(DEFAULT_POLICY["allow_pii"])),
    }

summary_df = pd.DataFrame([summarize_run(run) for run in runs])

tab_overview, tab_gate, tab_compare, tab_export = st.tabs(
    ["Overview", "Policy Gate", "Compare Runs", "Export"]
)

with tab_overview:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Saved Runs")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    latest = runs[0]
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        st.metric("Latest score", f"{latest.get('score', 'NA')}/100")
    with c2:
        st.metric("Latest grade", latest.get("grade", "NA"))
    with c3:
        st.metric("Saved runs", len(runs))
    with c4:
        st.metric("Analyzers", summary_df["analyzer"].nunique())

with tab_gate:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Run Gate Review")
    selected_label = st.selectbox("Audit run", [_run_label(run) for run in runs])
    selected = runs[[_run_label(run) for run in runs].index(selected_label)]
    result = evaluate_policy(selected, policy=policy)

    status = result["status"]
    st.markdown(
        f"### {'PASS' if status == 'PASS' else 'FAIL'}"
    )
    checks = pd.DataFrame(result["checks"])
    checks["value"] = checks["value"].map(_format_value)
    st.dataframe(checks, use_container_width=True, hide_index=True)
    if result["violations"]:
        st.error("Policy violations need remediation before this dataset should pass the gate.")
        for violation in result["violations"]:
            st.write(f"- {violation['name']}: {violation['value']} against {violation['limit']}")
    else:
        st.success("This audit passes the active policy gate.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab_compare:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Before / After Comparison")
    labels = [_run_label(run) for run in runs]
    if len(runs) < 2:
        st.info("At least two saved runs are needed for comparison.")
    else:
        before_label = st.selectbox("Before", labels, index=min(1, len(labels) - 1))
        after_label = st.selectbox("After", labels, index=0)
        before = runs[labels.index(before_label)]
        after = runs[labels.index(after_label)]
        comparison = compare_reports(before, after)

        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            delta = comparison["score_delta"]
            st.metric("Score delta", "N/A" if delta is None else f"{delta:+.0f}")
        with c2:
            st.metric("Grade before", comparison["grade_before"] or "N/A")
        with c3:
            st.metric("Grade after", comparison["grade_after"] or "N/A")

        metric_df = pd.DataFrame(comparison["metric_deltas"])
        st.dataframe(metric_df, use_container_width=True, hide_index=True)

        c_left, c_right = st.columns(2, gap="large")
        with c_left:
            st.markdown("**New findings**")
            st.write(comparison["new_findings"] or ["None"])
            st.markdown("**New recommendations**")
            st.write(comparison["new_recommendations"] or ["None"])
        with c_right:
            st.markdown("**Resolved findings**")
            st.write(comparison["resolved_findings"] or ["None"])
            st.markdown("**Resolved recommendations**")
            st.write(comparison["resolved_recommendations"] or ["None"])
    st.markdown("</div>", unsafe_allow_html=True)

with tab_export:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Export History")
    st.download_button(
        "Download audit history JSON",
        data=json.dumps(runs, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="astrid_audit_history.json",
        mime="application/json",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

