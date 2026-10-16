"""Live Classifier page — classify typed messages in real time."""

import sys
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR  = ROOT / "results"
DATASET_ROOT = ROOT.parent / "PervertedJusticeDataset"


@st.cache_resource(show_spinner="Training quick classifier on dataset…")
def get_trained_classifier(model_name: str, n_convs: int, seed: int):
    """Train a classifier on a sample of the dataset and cache it."""
    from data_loader import load_dataset
    from preprocessing import apply_minimal_cleaning, group_train_test_split, extract_labels_and_texts
    from message_classifier import SVMClassifier, NaiveBayesClassifier

    df = load_dataset()
    df = apply_minimal_cleaning(df)

    # Sample to keep startup fast
    sampled_convs = df["conversation_id"].drop_duplicates().sample(n_convs, random_state=seed)
    df_sample = df[df["conversation_id"].isin(sampled_convs)].reset_index(drop=True)

    train_df, _ = group_train_test_split(df_sample, test_size=0.2, random_state=seed)
    X_train, y_train = extract_labels_and_texts(train_df)

    if model_name == "SVM":
        clf = SVMClassifier()
    else:
        clf = NaiveBayesClassifier()

    clf.fit(X_train, y_train)
    return clf


def show():
    st.title("💬 Live Classifier")
    st.markdown(
        "Type a message below and see how the model classifies its language as "
        "**Adult (predator)** or **Child (officer)** style."
    )

    st.warning(
        "⚠️ **Important:** This classifier identifies *Adult vs. Child language patterns*, "
        "not whether a message is grooming content. It is a speaker-role detector."
    )

    if not DATASET_ROOT.exists():
        st.error("Dataset not found. See the Home page for setup instructions.")
        return

    st.markdown("---")

    # ── Classifier settings ───────────────────────────────────────────────────
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        model_choice = st.selectbox(
            "Classifier",
            ["SVM", "Naive Bayes"],
            help="BERT/RoBERTa require GPU — use CLI for those.",
        )
    with col_s2:
        n_convs = st.slider(
            "Training conversations",
            min_value=20, max_value=150, value=80, step=10,
            help="More conversations → better accuracy, slower startup.",
        )

    load_btn = st.button("🔄 Load / Retrain Classifier", type="secondary")

    if load_btn:
        st.cache_resource.clear()

    # ── Load classifier ───────────────────────────────────────────────────────
    with st.spinner(f"Loading {model_choice} classifier…"):
        try:
            clf = get_trained_classifier(model_choice, n_convs=n_convs, seed=42)
            st.success(f"✅ {model_choice} classifier ready (trained on {n_convs} conversations)")
        except Exception as e:
            st.error(f"Could not load classifier: {e}")
            return

    st.markdown("---")

    # ── Single message input ──────────────────────────────────────────────────
    st.subheader("🔤 Classify a Single Message")

    user_input = st.text_area(
        "Enter a message:",
        placeholder="e.g.  hey whats up, how old r u?",
        height=100,
        key="single_input",
    )

    if st.button("🔍 Classify", type="primary") and user_input.strip():
        _show_prediction(clf, user_input.strip(), model_choice)

    st.markdown("---")

    # ── Batch input ───────────────────────────────────────────────────────────
    st.subheader("📋 Classify Multiple Messages")
    st.caption("Enter one message per line.")

    batch_input = st.text_area(
        "Messages (one per line):",
        height=180,
        placeholder="hello\nhow are you?\nim 13/f\nwanna meet up?\nstudying rn",
        key="batch_input",
    )

    if st.button("🔍 Classify All", type="primary") and batch_input.strip():
        messages = [m.strip() for m in batch_input.strip().split("\n") if m.strip()]
        _show_batch_predictions(clf, messages, model_choice)

    st.markdown("---")

    # ── Conversation demo ─────────────────────────────────────────────────────
    st.subheader("🗨️ Demo Conversation")
    st.caption("Simulates a conversation and classifies each turn.")

    demo_convs = {
        "Example A — Typical interaction": [
            ("Predator", "hey whats up"),
            ("Officer",  "nm just chillin lol"),
            ("Predator", "how old r u?"),
            ("Officer",  "13/f u?"),
            ("Predator", "25/m :) u r cute"),
            ("Officer",  "thx lol"),
            ("Predator", "wanna meet up sometime?"),
            ("Officer",  "idk maybe"),
        ],
        "Example B — Casual chat": [
            ("Officer",  "what do u do for fun"),
            ("Predator", "play guitar mostly"),
            ("Officer",  "thats cool i like music too"),
            ("Predator", "what grade r u in"),
            ("Officer",  "8th hbu"),
            ("Predator", "i graduated lol im older"),
        ],
    }

    selected_demo = st.selectbox("Choose a demo", list(demo_convs.keys()))
    if st.button("▶ Run Demo"):
        _show_demo_conversation(clf, demo_convs[selected_demo], model_choice)


def _show_prediction(clf, text: str, model_name: str):
    """Show prediction for a single message."""
    proba = float(clf.predict_proba([text])[0])
    label = "Predator / Adult" if proba >= 0.5 else "Officer / Child"
    color = "#E74C3C" if proba >= 0.5 else "#3498DB"
    confidence = proba if proba >= 0.5 else 1 - proba

    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.markdown(
            f"<div style='background:{color};padding:18px;border-radius:8px;"
            f"color:white;font-size:1.3em;text-align:center;'>"
            f"<b>{label}</b><br/>"
            f"<span style='font-size:0.8em;'>Confidence: {confidence:.1%}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_b:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            title={"text": "P(Adult / Predator)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#E74C3C"},
                "steps": [
                    {"range": [0, 50], "color": "#D6EAF8"},
                    {"range": [50, 100], "color": "#FADBD8"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
            number={"suffix": "%"},
        ))
        fig.update_layout(height=220, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)


def _show_batch_predictions(clf, messages: list[str], model_name: str):
    """Show predictions for a list of messages."""
    probas = clf.predict_proba(messages)
    results = []
    for msg, p in zip(messages, probas):
        label = "Predator/Adult" if p >= 0.5 else "Officer/Child"
        conf  = p if p >= 0.5 else 1 - p
        results.append({
            "Message": msg[:80] + ("…" if len(msg) > 80 else ""),
            "Prediction": label,
            "P(Adult)": round(float(p), 3),
            "Confidence": f"{conf:.1%}",
        })

    df = pd.DataFrame(results)
    st.dataframe(
        df.style.apply(
            lambda row: [
                "background-color: #FADBD8" if row["Prediction"] == "Predator/Adult"
                else "background-color: #D6EAF8"
            ] * len(row),
            axis=1,
        ),
        use_container_width=True,
        height=min(40 * len(df) + 60, 500),
    )

    # Mini bar chart of P(Adult) per message
    fig = go.Figure(go.Bar(
        x=[f"Msg {i+1}" for i in range(len(messages))],
        y=[round(float(p), 3) for p in probas],
        marker_color=["#E74C3C" if p >= 0.5 else "#3498DB" for p in probas],
        text=[f"{p:.2f}" for p in probas],
        textposition="outside",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="black", annotation_text="Threshold = 0.5")
    fig.update_layout(
        title="P(Adult) per Message",
        yaxis_range=[0, 1.15],
        yaxis_title="P(Adult)",
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)


def _show_demo_conversation(clf, turns: list[tuple], model_name: str):
    """Classify and display a demo conversation turn by turn."""
    st.markdown("#### Conversation Classification")

    texts  = [t[1] for t in turns]
    roles  = [t[0] for t in turns]
    probas = clf.predict_proba(texts)

    for (speaker, message), proba in zip(turns, probas):
        pred_label = "Adult" if proba >= 0.5 else "Child"
        true_label = "Adult" if speaker == "Predator" else "Child"
        correct = pred_label == true_label
        icon = "✅" if correct else "❌"

        color = "#FADBD8" if speaker == "Predator" else "#D6EAF8"
        align = "right" if speaker == "Predator" else "left"
        badge_color = "#E74C3C" if pred_label == "Adult" else "#3498DB"

        st.markdown(
            f"<div style='text-align:{align};margin:6px 0;'>"
            f"<span style='font-size:0.8em;color:#777;'>{speaker}</span><br/>"
            f"<span style='background:{color};padding:8px 14px;border-radius:12px;"
            f"display:inline-block;max-width:70%;'>{message}</span>"
            f"<span style='background:{badge_color};color:white;padding:3px 8px;"
            f"border-radius:8px;font-size:0.75em;margin-left:8px;'>"
            f"{icon} {pred_label} ({proba:.2f})</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
