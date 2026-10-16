"""
app.py — Main Streamlit entry point

Online Grooming Detection UI
Implementation of Street et al. (2025)
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

# ── Sidebar nav ─────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/260px-Scikit_learn_logo_small.svg.png",
    width=80,
)
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
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset:** [PervertedJustice](https://github.com/helenkoutli/PervertedJusticeDataset)  \n"
    "**Paper:** Street et al. (2025)"
)

# ── Route to page ────────────────────────────────────────────────────────────
if page == "🏠 Home":
    from pages import home
    home.show()
elif page == "📊 Dataset Overview":
    from pages import dataset_overview
    dataset_overview.show()
elif page == "🤖 Run Experiments":
    from pages import run_experiments_ui
    run_experiments_ui.show()
elif page == "📈 Results & Metrics":
    from pages import results_viewer
    results_viewer.show()
elif page == "💬 Live Classifier":
    from pages import live_classifier
    live_classifier.show()
elif page == "📄 Paper Mapping":
    from pages import paper_mapping
    paper_mapping.show()
