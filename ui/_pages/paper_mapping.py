"""Paper Mapping page — component-by-component reproducibility table."""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT / "results"


def show():
    st.title("📄 Paper-to-Dataset Mapping")
    st.markdown(
        "**Paper:** Street et al. (2025) — *Enhanced Online Grooming Detection "
        "Employing Context Determination and Message-Level Analysis*  \n"
        "**Dataset:** PervertedJustice (Koutli, 2023)"
    )

    # ── Read the mapping file if it exists ───────────────────────────────────
    mapping_path = RESULTS_DIR / "paper_dataset_mapping.md"
    if mapping_path.exists():
        with open(mapping_path, encoding="utf-8") as f:
            content = f.read()
        with st.expander("📝 Full Mapping Document (Markdown)", expanded=False):
            st.markdown(content)

    st.markdown("---")

    # ── Reproducibility table ─────────────────────────────────────────────────
    st.subheader("Reproducibility at a Glance")

    mapping_data = [
        {
            "Component": "Message-Level Analysis (MLA)",
            "Required Inputs": "Per-message text + Adult/Child label",
            "Available?": "✅ Yes",
            "Method": "Predator file = Adult(1), Officer file = Child(0)",
            "Reproducibility": "🟡 Approx.",
        },
        {
            "Component": "Message Threshold t (Eq. 1)",
            "Required Inputs": "P(Adult) probability per message",
            "Available?": "✅ Yes",
            "Method": "Omit if (0.5+t) > n > (0.5-t); t ∈ {0.2, 0.3, 0.4, 0.45}",
            "Reproducibility": "🟢 Exact",
        },
        {
            "Component": "SVM Classifier",
            "Required Inputs": "Labelled utterances",
            "Available?": "✅ Yes",
            "Method": "sklearn Pipeline + CountVectorizer",
            "Reproducibility": "🟡 Approx.",
        },
        {
            "Component": "Naive Bayes Classifier",
            "Required Inputs": "Labelled utterances",
            "Available?": "✅ Yes",
            "Method": "sklearn MultinomialNB + CountVectorizer",
            "Reproducibility": "🟡 Approx.",
        },
        {
            "Component": "BERT Classifier",
            "Required Inputs": "Labelled utterances + GPU",
            "Available?": "✅ Yes (needs TF)",
            "Method": "bert-base-uncased fine-tuned (TensorFlow)",
            "Reproducibility": "🟡 Approx.",
        },
        {
            "Component": "RoBERTa Classifier",
            "Required Inputs": "Labelled utterances + GPU",
            "Available?": "✅ Yes (needs TF)",
            "Method": "roberta-base fine-tuned (TensorFlow)",
            "Reproducibility": "🟡 Approx.",
        },
        {
            "Component": "Actor Significance (Binomial Test)",
            "Required Inputs": "Per-actor predictions + AST",
            "Available?": "✅ Yes",
            "Method": "scipy.stats.binomtest; AST ∈ {0.05, 0.01, 0.001}",
            "Reproducibility": "🟢 Exact",
        },
        {
            "Component": "AC Context Determination",
            "Required Inputs": "Two actor determinations per conversation",
            "Available?": "✅ Yes",
            "Method": "(A XOR B Adult) AND (A XOR B Child) — paper Figure 3",
            "Reproducibility": "🟢 Exact",
        },
        {
            "Component": "80/20 Conversation Split",
            "Required Inputs": "conversation_id grouping",
            "Available?": "✅ Yes",
            "Method": "GroupShuffleSplit by conversation_id",
            "Reproducibility": "🟢 Exact",
        },
        {
            "Component": "MLA Metrics (TP, IA, IC, O)",
            "Required Inputs": "Per-message predictions",
            "Available?": "✅ Yes",
            "Method": "Custom computation (paper Tables 2/3)",
            "Reproducibility": "🟢 Exact",
        },
        {
            "Component": "Context Metrics (TP, FP, TN, FN, F1)",
            "Required Inputs": "Transcript-level ground truth",
            "Available?": "⚠️ PJ only (all positive)",
            "Method": "FP=0, TN=0 on PJ; F1 valid on PJ for TP/FN",
            "Reproducibility": "🟡 Partial",
        },
        {
            "Component": "Grooming Phase Analysis",
            "Required Inputs": "Chronological message order",
            "Available?": "❌ No",
            "Method": "Timestamps removed during cleaning",
            "Reproducibility": "🔴 Not possible",
        },
        {
            "Component": "Cross-dataset eval (PAN12)",
            "Required Inputs": "PAN12 dataset (separate)",
            "Available?": "❌ Not included",
            "Method": "PAN12 must be downloaded separately",
            "Reproducibility": "🔴 Not reproduced",
        },
        {
            "Component": "Turn-taking / response time features",
            "Required Inputs": "Interleaved conversation + timestamps",
            "Available?": "❌ No",
            "Method": "Both timestamps and interleaving removed",
            "Reproducibility": "🔴 Not possible",
        },
    ]

    df = pd.DataFrame(mapping_data)
    st.dataframe(df, use_container_width=True, height=530)

    # ── Legend ────────────────────────────────────────────────────────────────
    st.markdown(
        "🟢 **Exact** — fully reproducible as described in the paper  \n"
        "🟡 **Approx.** — reproducible with minor differences (dataset version, implementation details)  \n"
        "⚠️ **Partial** — reproducible but with dataset-imposed limitations (all-positive PJ)  \n"
        "🔴 **Not possible / Not reproduced** — missing data or external dataset required"
    )

    st.markdown("---")

    # ── Three tasks clarification ─────────────────────────────────────────────
    st.subheader("⚡ Three Tasks — Must Not Be Conflated")

    t1, t2, t3 = st.tabs([
        "Task 1: Speaker-Role Classification",
        "Task 2: Actor Significance",
        "Task 3: Conversation-Level AC Context",
    ])

    with t1:
        st.markdown(
            """
            **What:** Classify each utterance as Adult (1) or Child (0) language.

            **Ground truth source:** File structure — all lines in `predators/` are Adult,
            all lines in `officers/` are Child.

            **Metrics:** TP%, IA%, IC%, O% (paper Tables 2/3)

            **Important:** This is NOT grooming detection. It measures whether the language
            *style* matches an Adult or Child speaker. A predator saying "hello" is still
            labelled Adult because of which file it came from, not the content.
            """
        )

    with t2:
        st.markdown(
            """
            **What:** Use a binomial test to determine whether an actor's aggregate message
            predictions significantly indicate Adult (A), Child (C), or Not Significant (NS).

            **Method:** 
            1. Collect all P(Adult) values for one actor in one conversation
            2. Apply threshold *t* to omit uncertain messages
            3. Run `binomtest(n_adult_votes, n_kept, p=0.5, alternative='two-sided')`
            4. If `p_value < AST`: classify as A (mean > 0.5) or C (mean ≤ 0.5)
            5. If `p_value ≥ AST`: Not Significant

            **Inputs:** Task 1 probabilities + AST value
            """
        )

    with t3:
        st.markdown(
            """
            **What:** Determine if a conversation involves an Adult–Child dyad.

            **Rule (paper Figure 3):**
            ```
            AC context if:
              (Actor A is Adult XOR Actor B is Adult)
              AND
              (Actor A is Child XOR Actor B is Child)
            ```
            In plain English: exactly one actor is significantly Adult, exactly one is significantly Child.
            If either is NS, the context is non-AC.

            **Metrics:** TP, FP, TN, FN, F1 (paper Tables 4–7)

            **PJ constraint:** All PJ conversations are OG-positive → FP=0, TN=0 on this dataset.
            Cross-dataset evaluation (PAN12) is needed to measure false positives.
            """
        )

    st.markdown("---")

    # ── Dataset inspection report ─────────────────────────────────────────────
    st.subheader("📋 Dataset Inspection Report")
    report_path = RESULTS_DIR / "dataset_inspection_report.txt"
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with st.expander("Show full inspection report"):
            st.code(report, language="text")
    else:
        st.info("Run `python inspect_dataset.py` to generate the inspection report.")
