"""Run Experiments page — train models interactively."""

import sys
import json
import time
import threading
from pathlib import Path
from io import StringIO

import streamlit as st
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR  = ROOT / "results"
DATASET_ROOT = ROOT.parent / "PervertedJusticeDataset"


def show():
    st.title("🤖 Run Experiments")
    st.markdown(
        "Train SVM or Naive Bayes on the PervertedJustice dataset. "
        "Results are saved to `results/` and visible in the **Results & Metrics** page."
    )

    if not DATASET_ROOT.exists():
        st.error("Dataset not found. See the Home page for setup instructions.")
        return

    st.markdown("---")

    # ── Configuration ────────────────────────────────────────────────────────
    st.subheader("⚙️ Experiment Configuration")

    col1, col2 = st.columns(2)

    with col1:
        model_choice = st.selectbox(
            "Model",
            ["SVM", "Naive Bayes", "Both (SVM + NB)"],
            help="BERT/RoBERTa require TensorFlow and a GPU — run via CLI for those.",
        )

        test_size = st.slider(
            "Test split size",
            min_value=0.10, max_value=0.40, value=0.20, step=0.05,
            help="Fraction of CONVERSATIONS (not utterances) held out for testing.",
        )

        random_seed = st.number_input(
            "Random seed", min_value=0, max_value=9999, value=42, step=1
        )

    with col2:
        fast_mode = st.checkbox(
            "Fast mode (50 conversations)",
            value=True,
            help="Uses a 50-conversation sample. Uncheck for full dataset (~10+ min for SVM).",
        )
        fast_n = st.slider(
            "Conversations in fast mode",
            min_value=20, max_value=200, value=50, step=10,
            disabled=not fast_mode,
        )

        ast_values = st.multiselect(
            "AST values to sweep",
            options=[0.05, 0.01, 0.001],
            default=[0.05, 0.01, 0.001],
            help="Actor Significance Threshold values for context determination.",
        )

    st.markdown("---")

    # ── Warning ───────────────────────────────────────────────────────────────
    st.warning(
        "**Reminder:** These experiments classify *speaker roles* (Adult vs. Child language).  "
        "They do **not** classify individual messages as grooming content."
    )

    # ── Run button ────────────────────────────────────────────────────────────
    run_clicked = st.button("▶ Run Experiment", type="primary", use_container_width=True)

    if run_clicked:
        _run_experiment(
            model_choice=model_choice,
            test_size=test_size,
            random_seed=int(random_seed),
            fast_mode=fast_mode,
            fast_n=fast_n,
            ast_values=ast_values,
        )


def _run_experiment(model_choice, test_size, random_seed, fast_mode, fast_n, ast_values):
    """Execute training and evaluation, streaming progress to the UI."""
    from data_loader import load_dataset
    from preprocessing import apply_minimal_cleaning, group_train_test_split, extract_labels_and_texts
    from message_classifier import (
        SVMClassifier, NaiveBayesClassifier,
        apply_message_threshold, compute_mla_metrics,
    )
    from actor_classification import attach_predictions, sweep_ast_t_values, T_VALUES_TRADITIONAL
    from evaluation import (
        compute_standard_metrics, plot_confusion_matrix,
        save_mla_results, save_context_results,
    )

    progress = st.progress(0, text="Loading dataset…")
    log_box = st.empty()
    logs = []

    def log(msg):
        logs.append(msg)
        log_box.code("\n".join(logs[-20:]))

    try:
        # Stage 1 — Load
        log("📂 Loading dataset…")
        df = load_dataset()
        df = apply_minimal_cleaning(df)
        log(f"   ✓ {len(df):,} utterances, {df['conversation_id'].nunique()} conversations")
        progress.progress(10, text="Dataset loaded")

        # Stage 2 — Sample
        if fast_mode:
            sampled = df["conversation_id"].drop_duplicates().sample(fast_n, random_state=random_seed)
            df = df[df["conversation_id"].isin(sampled)].reset_index(drop=True)
            log(f"   ✓ Fast mode: {len(df):,} utterances from {df['conversation_id'].nunique()} conversations")

        # Stage 3 — Split
        progress.progress(20, text="Splitting data…")
        train_df, test_df = group_train_test_split(df, test_size=test_size, random_state=random_seed)
        X_train, y_train = extract_labels_and_texts(train_df)
        X_test,  y_test  = extract_labels_and_texts(test_df)
        log(f"   ✓ Train: {len(train_df):,} utterances ({train_df['conversation_id'].nunique()} convs)")
        log(f"   ✓ Test:  {len(test_df):,} utterances ({test_df['conversation_id'].nunique()} convs)")

        # Determine which models to run
        models_to_run = []
        if model_choice in ("SVM", "Both (SVM + NB)"):
            models_to_run.append(("SVM", SVMClassifier()))
        if model_choice in ("Naive Bayes", "Both (SVM + NB)"):
            models_to_run.append(("NB", NaiveBayesClassifier()))

        all_mla   = {}
        all_ctx   = {}
        step_size = 60 // max(len(models_to_run), 1)
        current_p = 20

        for name, clf in models_to_run:
            log(f"\n🤖 Training {name}…")
            progress.progress(current_p, text=f"Training {name}…")
            clf.fit(X_train, y_train)
            log(f"   ✓ {name} trained")

            # MLA
            y_pred = clf.predict(X_test)
            mla = compute_mla_metrics(y_test, y_pred)
            all_mla[name] = mla
            log(f"   MLA → TP={mla['TP']}%  IA={mla['IA']}%  IC={mla['IC']}%  O={mla['O']}%")
            save_mla_results(mla, name, RESULTS_DIR)

            # Confusion matrix
            cm_path = RESULTS_DIR / f"confusion_matrix_{name.lower()}.png"
            plot_confusion_matrix(y_test, y_pred, name, save_path=cm_path)

            # Context determination
            log(f"   Running context determination (AST sweep)…")
            test_with_proba = attach_predictions(test_df, clf)
            ctx = sweep_ast_t_values(
                test_with_proba,
                model_name=name,
                t_values=T_VALUES_TRADITIONAL,
                ast_values=ast_values if ast_values else [0.05],
            )
            all_ctx[name] = ctx
            save_context_results(ctx, name, RESULTS_DIR)
            best = ctx.loc[ctx["F1"].idxmax()]
            log(f"   Context → best F1={best['F1']:.3f} at AST={best['AST']}")

            current_p += step_size

        progress.progress(90, text="Saving results…")

        # ── Show inline results ──────────────────────────────────────────────
        st.markdown("---")
        st.subheader("✅ Results")
        st.caption("(Speaker-role classification — Adult vs. Child language, NOT grooming detection)")

        # MLA table
        mla_rows = []
        for name, m in all_mla.items():
            mla_rows.append({"Model": name, "TP%": m["TP"], "IA%": m["IA"], "IC%": m["IC"], "O%": m["O"]})
        st.dataframe(pd.DataFrame(mla_rows).set_index("Model"), use_container_width=True)

        # Context table
        for name, ctx in all_ctx.items():
            st.markdown(f"**{name} — Context Determination**")
            st.dataframe(ctx[["AST", "TP", "FN", "FP", "TN", "F1"]], use_container_width=True)

        # Confusion matrix images
        img_cols = st.columns(len(models_to_run))
        for idx, (name, _) in enumerate(models_to_run):
            cm_path = RESULTS_DIR / f"confusion_matrix_{name.lower()}.png"
            if cm_path.exists():
                img_cols[idx].image(str(cm_path), caption=f"{name} Confusion Matrix", use_container_width=True)

        progress.progress(100, text="Done ✓")
        st.success("Experiment complete. View full results in **Results & Metrics**.")

    except Exception as e:
        st.error(f"Error during experiment: {e}")
        st.exception(e)
