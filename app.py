import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import SHARED_CSS

st.set_page_config(
    page_title="Dataset Safety Analyzer",
    page_icon="🛡",
    layout="wide",
)

st.markdown(SHARED_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="dsa-card" style="padding:36px 40px 28px 40px; margin-bottom: 24px;">
  <div style="display:flex; align-items:center; gap:14px; margin-bottom:10px;">
    <span style="font-size:2.4rem;">🛡</span>
    <div>
      <h1 style="margin:0; font-size:2.2rem;">Unified Dataset Safety Analyzer</h1>
      <div class="muted" style="margin-top:4px;">
        AI/ML dataset auditing &nbsp;·&nbsp; Quality · Reliability · Robustness · Fairness · Security
      </div>
    </div>
  </div>
  <hr>
  <div class="muted">
    Choose an analyzer from the sidebar. Each page handles a different dataset modality and runs
    a full suite of heuristic safety checks, returning an overall health score, actionable
    recommendations, and a downloadable report.
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown("""
<div class="dsa-card" style="padding:24px;">
  <div style="font-size:2rem; margin-bottom:10px;">📊</div>
  <h3 style="margin:0 0 8px 0;">Tabular</h3>
  <div class="muted">CSV · Parquet · Excel</div>
  <hr>
  <ul style="margin:0; padding-left:1.1em; font-size:0.85rem; line-height:1.9;">
    <li>Missingness &amp; duplicate detection</li>
    <li>IQR outlier rates per column</li>
    <li>Split leakage via row hashing</li>
    <li>KS drift across splits / time</li>
    <li>Rare-category label concentration</li>
    <li>PII heuristic scan</li>
    <li>Fairness proxies by group</li>
    <li>Label predictability AUC</li>
  </ul>
</div>
""", unsafe_allow_html=True)

with c2:
    st.markdown("""
<div class="dsa-card" style="padding:24px;">
  <div style="font-size:2rem; margin-bottom:10px;">📈</div>
  <h3 style="margin:0 0 8px 0;">Time Series</h3>
  <div class="muted">Temporal CSV · Parquet</div>
  <hr>
  <ul style="margin:0; padding-left:1.1em; font-size:0.85rem; line-height:1.9;">
    <li>Timestamp parsing &amp; validity</li>
    <li>Cadence irregularity scoring</li>
    <li>Gap detection (10× median rule)</li>
    <li>Duplicate timestamp detection</li>
    <li>Time-sliced missingness trends</li>
    <li>KS drift first vs last slice</li>
    <li>Entity-level analysis</li>
    <li>PII scan &amp; identifier audit</li>
  </ul>
</div>
""", unsafe_allow_html=True)

with c3:
    st.markdown("""
<div class="dsa-card" style="padding:24px;">
  <div style="font-size:2rem; margin-bottom:10px;">🖼</div>
  <h3 style="margin:0 0 8px 0;">Images</h3>
  <div class="muted">ZIP folder of images</div>
  <hr>
  <ul style="margin:0; padding-left:1.1em; font-size:0.85rem; line-height:1.9;">
    <li>Corrupt / unreadable file detection</li>
    <li>Resolution &amp; aspect ratio analysis</li>
    <li>Brightness / contrast statistics</li>
    <li>Exact duplicate detection (SHA-256)</li>
    <li>Perceptual duplicate detection</li>
    <li>EXIF metadata extraction</li>
    <li>Label distribution (from folders)</li>
    <li>Class imbalance detection</li>
  </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="dsa-card">
  <h3 style="margin-top:0;">How it works</h3>
  <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:20px; text-align:center; padding: 8px 0;">
    <div>
      <div style="font-size:1.6rem;">⬆️</div>
      <div style="font-weight:700; margin:6px 0 4px;">1. Upload</div>
      <div class="muted">Drop your dataset in the page sidebar</div>
    </div>
    <div>
      <div style="font-size:1.6rem;">⚙️</div>
      <div style="font-weight:700; margin:6px 0 4px;">2. Configure</div>
      <div class="muted">Accept auto-guessed columns or adjust</div>
    </div>
    <div>
      <div style="font-size:1.6rem;">🔬</div>
      <div style="font-weight:700; margin:6px 0 4px;">3. Analyze</div>
      <div class="muted">Click Run — checks complete in seconds</div>
    </div>
    <div>
      <div style="font-size:1.6rem;">📥</div>
      <div style="font-weight:700; margin:6px 0 4px;">4. Export</div>
      <div class="muted">Download JSON, Markdown, or HTML report</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="muted" style="text-align:center; margin-top: 20px; font-size:0.75rem;">
  All checks are heuristic. Results are evidence for, not proof of, safety.
  Always validate with domain expertise and legal review before deployment.
</div>
""", unsafe_allow_html=True)
