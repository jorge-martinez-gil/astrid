import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import SHARED_CSS

st.set_page_config(
    page_title="ASTRID — Dataset Safety Analyzer",
    page_icon="🛡",
    layout="wide",
)

st.markdown(SHARED_CSS, unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)

st.sidebar.markdown(
    "<div class='muted' style='text-align:center; padding: 4px 0 12px 0;'>"
    "Advanced Software Tools for the<br>Reliability of Industrial Datasets"
    "</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Navigate**", help="Select an analyzer from the pages below")
st.sidebar.markdown(
    "<div class='muted' style='font-size:0.78rem; line-height:1.7;'>"
    "📊 &nbsp;<b>Tabular</b> — CSV / Parquet / Excel<br>"
    "📈 &nbsp;<b>Time Series</b> — Temporal datasets<br>"
    "🖼 &nbsp;<b>Images</b> — ZIP of image files<br>"
    "🔬 &nbsp;<b>Drift</b> — Experimental drift engine"
    "</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

with st.sidebar.expander("🚀 Quick Start"):
    st.markdown(
        "1. **Choose** an analyzer from the pages list above.\n"
        "2. **Upload** your dataset using the sidebar on that page.\n"
        "3. **Configure** columns (or accept auto-guesses).\n"
        "4. **Click Run** — checks finish in seconds.\n"
        "5. **Export** a JSON / Markdown / HTML report.",
    )

# ── Hero section ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="dsa-card" style="
  padding: 40px 44px 32px 44px;
  margin-bottom: 8px;
  background: linear-gradient(135deg, rgba(59,130,246,0.08) 0%, rgba(168,85,247,0.08) 50%, rgba(249,115,22,0.06) 100%);
  border-color: rgba(99,102,241,0.25);
">
  <div style="display:flex; align-items:center; gap:14px; margin-bottom:6px;">
    <span style="font-size:2.6rem;">🛡</span>
    <div style="flex:1;">
      <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
        <h1 style="margin:0; font-size:2.2rem; letter-spacing:-0.03em;">ASTRID</h1>
        <span class="badge badge-info" style="font-size:0.7rem; padding:2px 9px;">v1.0</span>
        <a href="https://github.com/jorge-martinez-gil/astrid" target="_blank"
           style="text-decoration:none;">
          <span class="badge" style="font-size:0.7rem; padding:2px 9px; background:rgba(255,255,255,0.06);">
            ⭐ GitHub
          </span>
        </a>
      </div>
      <div class="muted" style="margin-top:3px; font-size:0.82rem;">
        Dataset Safety Analyzer &nbsp;·&nbsp; Advanced Software Tools for the Reliability of Industrial Datasets
      </div>
    </div>
  </div>
  <div class="gradient-accent"></div>
  <div style="font-size:1.05rem; font-weight:600; margin-bottom:8px;">
    Enterprise-grade dataset auditing for AI/ML pipelines
  </div>
  <div class="muted">
    Choose an analyzer from the sidebar. Each module runs a full suite of heuristic safety checks
    and returns an overall health score, actionable recommendations, and a downloadable report.
    &nbsp;·&nbsp; Quality &nbsp;·&nbsp; Reliability &nbsp;·&nbsp; Robustness &nbsp;·&nbsp; Fairness &nbsp;·&nbsp; Security
  </div>
</div>
""", unsafe_allow_html=True)

# ── Feature cards ──────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown("""
<div class="dsa-card feature-card feature-card-blue" style="padding:24px; height:100%;">
  <div style="font-size:2rem; margin-bottom:10px;">📊</div>
  <h3 style="margin:0 0 6px 0;">Tabular</h3>
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
  <span class="open-hint">→ Open Analyzer</span>
</div>
""", unsafe_allow_html=True)

with c2:
    st.markdown("""
<div class="dsa-card feature-card feature-card-purple" style="padding:24px; height:100%;">
  <div style="font-size:2rem; margin-bottom:10px;">📈</div>
  <h3 style="margin:0 0 6px 0;">Time Series</h3>
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
  <span class="open-hint">→ Open Analyzer</span>
</div>
""", unsafe_allow_html=True)

with c3:
    st.markdown("""
<div class="dsa-card feature-card feature-card-orange" style="padding:24px; height:100%;">
  <div style="font-size:2rem; margin-bottom:10px;">🖼</div>
  <h3 style="margin:0 0 6px 0;">Images</h3>
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
  <span class="open-hint">→ Open Analyzer</span>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── How it works ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="dsa-card" style="padding:28px 32px;">
  <h3 style="margin-top:0; margin-bottom:20px;">⚡ How it works</h3>
  <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:20px; text-align:center; padding: 8px 0;" class="feature-cols">
    <div>
      <div class="step-circle">⬆️</div>
      <div style="font-weight:700; margin:6px 0 4px;">1. Upload</div>
      <div class="muted">Drop your dataset in the page sidebar</div>
    </div>
    <div>
      <div class="step-circle">⚙️</div>
      <div style="font-weight:700; margin:6px 0 4px;">2. Configure</div>
      <div class="muted">Accept auto-guessed columns or adjust</div>
    </div>
    <div>
      <div class="step-circle">🔬</div>
      <div style="font-weight:700; margin:6px 0 4px;">3. Analyze</div>
      <div class="muted">Click Run — checks complete in seconds</div>
    </div>
    <div>
      <div class="step-circle">📥</div>
      <div style="font-weight:700; margin:6px 0 4px;">4. Export</div>
      <div class="muted">Download JSON, Markdown, or HTML report</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Disclaimer + Footer ────────────────────────────────────────────────────────
st.markdown("""
<div class="muted" style="text-align:center; margin-top: 16px; font-size:0.75rem;">
  All checks are heuristic. Results are evidence for, not proof of, safety.
  Always validate with domain expertise and legal review before deployment.
</div>
<div class="dsa-footer">
  <div>
    <strong>ASTRID</strong> — Advanced Software Tools for the Reliability of Industrial Datasets
  </div>
  <div>
    © 2024 <a href="https://github.com/jorge-martinez-gil" target="_blank">Jorge Martinez-Gil</a>
    &nbsp;·&nbsp;
    <a href="https://github.com/jorge-martinez-gil/astrid" target="_blank">GitHub Repository</a>
    &nbsp;·&nbsp;
    Powered by <a href="https://streamlit.io" target="_blank">Streamlit</a>
  </div>
</div>
""", unsafe_allow_html=True)
