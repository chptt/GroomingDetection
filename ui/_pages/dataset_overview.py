"""Dataset Overview page."""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

DATASET_ROOT = ROOT.parent / "PervertedJusticeDataset"
PRED_DIR = DATASET_ROOT / "predators"
OFF_DIR  = DATASET_ROOT / "officers"


@st.cache_data(show_spinner="Loading dataset statistics…")
def load_stats():
    """Compute per-file utterance counts and pairing info."""
    pred_files = sorted(PRED_DIR.glob("*.txt"))
    off_files  = sorted(OFF_DIR.glob("*.txt"))

    def count_lines(path):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0

    pred_data = [
        {"file": p.name, "conv_id": p.stem.replace("predator_", ""), "role": "Predator", "utterances": count_lines(p)}
        for p in pred_files
    ]
    off_data = [
        {"file": o.name, "conv_id": o.stem.replace("officer_", ""), "role": "Officer", "utterances": count_lines(o)}
        for o in off_files
    ]

    pred_df = pd.DataFrame(pred_data)
    off_df  = pd.DataFrame(off_data)

    # Pairing
    pred_ids = set(pred_df["conv_id"])
    off_ids  = set(off_df["conv_id"])
    paired   = pred_ids & off_ids
    pred_only = pred_ids - off_ids
    off_only  = off_ids - pred_ids

    return pred_df, off_df, paired, pred_only, off_only


def show():
    st.title("📊 Dataset Overview")

    if not DATASET_ROOT.exists():
        st.error(
            f"Dataset not found at `{DATASET_ROOT}`.  \n"
            "Run: `git clone https://github.com/helenkoutli/PervertedJusticeDataset.git`"
        )
        return

    pred_df, off_df, paired, pred_only, off_only = load_stats()

    # ── Top metrics ──────────────────────────────────────────────────────────
    st.subheader("Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Predator Files",      len(pred_df))
    m2.metric("Officer Files",       len(off_df))
    m3.metric("Paired Conversations", len(paired))
    m4.metric("Total Pred. Utterances", f"{pred_df['utterances'].sum():,}")
    m5.metric("Total Off. Utterances",  f"{off_df['utterances'].sum():,}")

    st.markdown("---")

    # ── Utterance distribution ────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Utterances per File — Predators")
        fig = px.histogram(
            pred_df, x="utterances", nbins=50,
            color_discrete_sequence=["#E74C3C"],
            labels={"utterances": "Utterances per file"},
        )
        fig.update_layout(margin=dict(t=20, b=20), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Utterances per File — Officers")
        fig = px.histogram(
            off_df, x="utterances", nbins=50,
            color_discrete_sequence=["#3498DB"],
            labels={"utterances": "Utterances per file"},
        )
        fig.update_layout(margin=dict(t=20, b=20), height=300)
        st.plotly_chart(fig, use_container_width=True)

    # ── Role balance ─────────────────────────────────────────────────────────
    st.subheader("Role Balance")
    total_pred = int(pred_df["utterances"].sum())
    total_off  = int(off_df["utterances"].sum())

    fig_pie = px.pie(
        values=[total_pred, total_off],
        names=["Predator (Adult)", "Officer (Child)"],
        color_discrete_sequence=["#E74C3C", "#3498DB"],
        hole=0.45,
    )
    fig_pie.update_layout(height=320, margin=dict(t=20, b=20))
    st.plotly_chart(fig_pie, use_container_width=True)

    # ── Pairing analysis ─────────────────────────────────────────────────────
    st.subheader("File Pairing Analysis")
    pa1, pa2, pa3 = st.columns(3)
    pa1.metric("Paired (both files exist)", len(paired), delta=None)
    pa2.metric("Predator-only (no officer)", len(pred_only),
               delta=f"-{len(pred_only)}", delta_color="inverse")
    pa3.metric("Officer-only (no predator)", len(off_only),
               delta=f"-{len(off_only)}", delta_color="inverse")

    with st.expander("Show predator-only conversation IDs"):
        st.write(sorted(pred_only))

    with st.expander("Show officer-only conversation IDs (multi-officer conversations have _1, _2 suffixes)"):
        st.write(sorted(off_only))

    # ── Per-file stats table ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Per-File Utterance Table")

    tab_pred, tab_off = st.tabs(["Predator Files", "Officer Files"])

    with tab_pred:
        st.dataframe(
            pred_df.sort_values("utterances", ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=400,
        )

    with tab_off:
        st.dataframe(
            off_df.sort_values("utterances", ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=400,
        )

    # ── Dataset notes ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📝 Dataset Notes")
    st.markdown(
        """
        - **Timestamps removed** during cleaning — chronological interleaving cannot be reconstructed.
        - **12 conversations** have multiple officer files (same predator, different officer).  
          These appear with `_1`, `_2` suffixes in the officer folder.
        - **2 predator files** were merged in the cleaning process (explains slight count mismatch).
        - **Encoding:** ASCII (verified via `chardet`).
        - **All conversations are OG-positive** — every transcript is a confirmed grooming interaction.
        """
    )
