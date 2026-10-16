"""
app.py — Main Streamlit entry point

Online Grooming Detection UI
Implementation of Street et al. (2025)

NOTE: Page modules live in ui/_pages/ (not ui/pages/) so Streamlit does NOT
auto-generate a second navigation bar from them. All navigation is handled
by the custom sidebar radio below.
"""

import sys
from pathlib import Path

# Make grooming_detection modules importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Online Grooming Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar nav ──────────────────────────────────────────────────────────────
st.sidebar.title("🔍 Grooming Detection")
st.sidebar.caption("Street et al. (2025) · PJ Dataset")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📊 Dataset Overview",
        "🤖 Run Experiments",
        "📈 Results & Metrics",
        "💬 Live Classifier",
        "📄 Paper Mapping",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset:** [PervertedJustice](https://github.com/helenkoutli/PervertedJusticeDataset)  \n"
    "**Paper:** Street et al. (2025)  \n"
    "**Repo:** [chptt/GroomingDetection](https://github.com/chptt/GroomingDetection)"
)

# ── Route to page ─────────────────────────────────────────────────────────────
if page == "🏠 Home":
    from _pages import home
    home.show()
elif page == "📊 Dataset Overview":
    from _pages import dataset_overview
    dataset_overview.show()
elif page == "🤖 Run Experiments":
    from _pages import run_experiments_ui
    run_experiments_ui.show()
elif page == "📈 Results & Metrics":
    from _pages import results_viewer
    results_viewer.show()
elif page == "💬 Live Classifier":
    from _pages import live_classifier
    live_classifier.show()
elif page == "📄 Paper Mapping":
    from _pages import paper_mapping
    paper_mapping.show()
