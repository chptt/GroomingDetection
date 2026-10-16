"""Results & Metrics page — display saved experiment outputs."""

import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"

# Paper reference results (display only — never embedded in experiment outputs)
PAPER_MLA = {
    "SVM":             {"TP": 76.0, "IA": 13.5, "IC": 10.5, "O": 0.0},
    "NB":              {"TP": 74.7, "IA": 12.0, "IC": 13.3, "O": 0.0},
    "BERT (t=0.45)":   {"TP": 87.3, "IA":  7.1, "IC":  5.6, "O": 30.0},
    "RoBERTa (t=0.45)":{"TP": 88.8, "IA":  5.3, "IC":  5.9, "O": 30.4},
}
PAPER_CTX_AST05 = {
    "SVM":     {"TP": 118, "FN": 35, "F1": 0.871},
    "NB":      {"TP": 116, "FN": 37, "F1": 0.862},
    "BERT":    {"TP": 143, "FN": 10, "F1": 0.966},
    "RoBERTa": {"TP": 145, "FN":  8, "F1": 0.973},
}


def _load_mla():
    """Load all saved MLA JSON files."""
    rows = []
    for p in RESULTS_DIR.glob("mla_*.json"):
        with open(p) as f:
            d = json.load(f)
        model = p.stem.replace("mla_", "").upper()
        rows.append({"Model": model, **d})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _load_ctx():
    """Load all saved context CSVs."""
    dfs = []
    for p in RESULTS_DIR.glob("context_*.csv"):
        dfs.append(pd.read_csv(p))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def show():
    st.title("📈 Results & Metrics")

    if not RESULTS_DIR.exists() or not any(RESULTS_DIR.glob("mla_*.json")):
        st.warning(
            "No results found yet.  \n"
            "Go to **🤖 Run Experiments** and run an experiment first."
        )
        return

    mla_df = _load_mla()
    ctx_df = _load_ctx()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 MLA Metrics",
        "🔬 Context Determination",
        "🖼️ Confusion Matrices",
        "📎 vs. Paper",
    ])

    # ── Tab 1: MLA ────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Message-Level Analysis (MLA)")
        st.caption(
            "Task: Classify each utterance as Adult (1) or Child (0) language.  "
            "**Not** grooming message detection."
        )

        if mla_df.empty:
            st.info("No MLA results found.")
        else:
            st.dataframe(
                mla_df.set_index("Model")[["TP", "IA", "IC", "O"]],
                use_container_width=True,
            )

            # Grouped bar chart
            metrics = ["TP", "IA", "IC", "O"]
            fig = go.Figure()
            colors = {"TP": "#2ECC71", "IA": "#E74C3C", "IC": "#E67E22", "O": "#95A5A6"}
            for m in metrics:
                if m in mla_df.columns:
                    fig.add_trace(go.Bar(
                        name=m,
                        x=mla_df["Model"],
                        y=mla_df[m],
                        marker_color=colors[m],
                        text=mla_df[m].round(1),
                        textposition="outside",
                    ))
            fig.update_layout(
                barmode="group",
                title="MLA Metrics by Model (%)",
                yaxis_title="Percentage (%)",
                yaxis_range=[0, 110],
                legend_title="Metric",
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Metric definitions"):
                st.markdown(
                    """
                    - **TP%** — Correct Adult/Child determinations (out of kept messages)
                    - **IA%** — Incorrect Adult: predicted Adult but was Child
                    - **IC%** — Incorrect Child: predicted Child but was Adult
                    - **O%** — Omissions: messages excluded by threshold *t* (out of ALL messages)
                    - Note: TP% + IA% + IC% ≈ 100% (over kept messages); O% is separate
                    """
                )

    # ── Tab 2: Context ────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Context Determination Results")
        st.caption(
            "Task: Detect Adult–Child conversation context using Actor Significance Threshold (AST).  "
            "PJ dataset is all-positive → FP=0, TN=0 by design."
        )

        if ctx_df.empty:
            st.info("No context determination results found.")
        else:
            # Filter by model
            models_available = ctx_df["Model"].unique().tolist() if "Model" in ctx_df.columns else []
            selected_model = st.selectbox("Filter by model", ["All"] + models_available)

            df_show = ctx_df if selected_model == "All" else ctx_df[ctx_df["Model"] == selected_model]
            st.dataframe(df_show, use_container_width=True)

            # F1 heatmap over AST × model
            if "AST" in ctx_df.columns and "F1" in ctx_df.columns and "Model" in ctx_df.columns:
                pivot = ctx_df.pivot_table(values="F1", index="Model", columns="AST", aggfunc="max")
                fig_heat = px.imshow(
                    pivot,
                    text_auto=".3f",
                    color_continuous_scale="Blues",
                    title="F1 Score — Model × AST",
                    labels={"color": "F1"},
                )
                fig_heat.update_layout(height=350)
                st.plotly_chart(fig_heat, use_container_width=True)

    # ── Tab 3: Confusion matrices ─────────────────────────────────────────────
    with tab3:
        st.subheader("Confusion Matrices")
        st.caption("Speaker-role classification: Officer (Child) vs. Predator (Adult).")

        cm_files = sorted(RESULTS_DIR.glob("confusion_matrix_*.png"))
        if not cm_files:
            st.info("No confusion matrix images found. Run an experiment first.")
        else:
            cols = st.columns(min(len(cm_files), 3))
            for idx, cm_path in enumerate(cm_files):
                model_name = cm_path.stem.replace("confusion_matrix_", "").upper()
                cols[idx % 3].image(str(cm_path), caption=model_name, use_container_width=True)

    # ── Tab 4: vs. Paper ─────────────────────────────────────────────────────
    with tab4:
        st.subheader("Our Results vs. Paper Published Results")
        st.info(
            "Paper results are **reference targets only** — never embedded in experiment outputs.  "
            "Differences are expected due to dataset version, sample size, and implementation."
        )

        # MLA comparison
        if not mla_df.empty:
            st.markdown("#### MLA Comparison (TP%)")
            fig = go.Figure()
            our_models = mla_df["Model"].tolist()

            fig.add_trace(go.Bar(
                name="Our Results",
                x=our_models,
                y=mla_df["TP"].tolist(),
                marker_color="#3498DB",
            ))
            paper_tp = [PAPER_MLA.get(m, PAPER_MLA.get(m.upper(), {})).get("TP") for m in our_models]
            paper_tp_clean = [v for v in paper_tp if v is not None]
            paper_models   = [m for m, v in zip(our_models, paper_tp) if v is not None]
            if paper_models:
                fig.add_trace(go.Bar(
                    name="Paper (reference)",
                    x=paper_models,
                    y=paper_tp_clean,
                    marker_color="#E67E22",
                    opacity=0.7,
                ))

            fig.update_layout(
                barmode="group",
                title="TP% Comparison",
                yaxis_range=[0, 100],
                height=360,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Paper context table
        st.markdown("#### Paper Context Determination (AST=0.05, PJ dataset)")
        st.dataframe(pd.DataFrame(PAPER_CTX_AST05).T, use_container_width=True)
        st.caption("Source: Street et al. (2025), Tables 4–7")
