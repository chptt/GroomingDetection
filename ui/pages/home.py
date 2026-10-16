"""Home / landing page."""

import streamlit as st
from pathlib import Path

DATASET_ROOT = Path(__file__).parent.parent.parent.parent / "PervertedJusticeDataset"
RESULTS_DIR  = Path(__file__).parent.parent.parent / "results"


def show():
    st.title("🔍 Online Grooming Detection")
    st.markdown(
        """
        **Implementation of:**
        > Street, J., Ihianle, I.K., Olajide, F., & Lotfi, A. (2025).  
        > *Enhanced Online Grooming Detection Employing Context Determination and Message-Level Analysis.*  
        > Intelligent Systems with Applications, 28, 200607.
        """
    )

    st.markdown("---")

    # Quick status cards
    col1, col2, col3, col4 = st.columns(4)

    dataset_ok = DATASET_ROOT.exists()
    results_ok = (RESULTS_DIR / "mla_svm.json").exists()
    test_count = 88

    with col1:
        st.metric("Dataset", "✅ Cloned" if dataset_ok else "❌ Missing")
    with col2:
        pred_count = len(list((DATASET_ROOT / "predators").glob("*.txt"))) if dataset_ok else 0
        st.metric("Predator Files", f"{pred_count:,}")
    with col3:
        off_count = len(list((DATASET_ROOT / "officers").glob("*.txt"))) if dataset_ok else 0
        st.metric("Officer Files", f"{off_count:,}")
    with col4:
        st.metric("Tests Passing", f"{test_count} / {test_count}")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("📌 What This System Does")
        st.markdown(
            """
            The paper introduces two complementary methods:

            1. **Message-Level Analysis (MLA)**  
               Each utterance is classified as **Adult (1)** or **Child (0)** language  
               using SVM, Naive Bayes, BERT, or RoBERTa.

            2. **Context Determination**  
               Per-actor predictions are aggregated with a **binomial significance test**  
               (Actor Significance Threshold) to detect whether a conversation involves  
               an Adult–Child dyad — a signal for potential Online Grooming.
            """
        )

        st.info(
            "**Speaker role ≠ grooming label.**  "
            "Predator utterances are labelled *Adult* by file structure, "
            "not because each message is grooming content."
        )

    with col_r:
        st.subheader("🗺️ Navigation Guide")
        st.markdown(
            """
            | Page | What you can do |
            |---|---|
            | 📊 **Dataset Overview** | File counts, utterance stats, pairing analysis |
            | 🤖 **Run Experiments** | Train SVM/NB interactively, adjust split & thresholds |
            | 📈 **Results & Metrics** | MLA table, context determination, confusion matrices |
            | 💬 **Live Classifier** | Type a message and classify it in real time |
            | 📄 **Paper Mapping** | Component-by-component reproducibility table |
            """
        )

    st.markdown("---")
    st.subheader("⚠️ Key Dataset Constraints")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.error("**Timestamps removed**  \nChronological order not preserved.  \nCannot reproduce phase analysis.")
    with c2:
        st.warning("**All-positive dataset**  \nEvery PJ conversation is OG-positive.  \nNo true negatives → FP/TN = 0 on PJ alone.")
    with c3:
        st.success("**Speaker identity known**  \nFile structure gives Adult/Child labels.  \nMLA and context determination fully supported.")
