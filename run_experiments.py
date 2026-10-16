"""
run_experiments.py

Orchestrates the full experimental pipeline for the Online Grooming Detection
implementation of Street et al. (2025).

Pipeline stages:
  1. Load and inspect dataset
  2. Preprocess (minimal, matching paper)
  3. Group-based 80/20 train/test split (by conversation)
  4. Train SVM and Naive Bayes → MLA evaluation
  5. Train BERT (if available) → MLA + context evaluation with t values
  6. Train RoBERTa (if available) → MLA + context evaluation with t values
  7. Context Determination (Actor Significance + AST sweep)
  8. Save all results

Usage:
    python run_experiments.py [--skip-transformers] [--debug]

Flags:
    --skip-transformers : Run only SVM and NB (faster, no GPU needed)
    --debug             : Verbose logging
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_dataset, summarise_dataset, get_paired_conversations
from preprocessing import (
    apply_minimal_cleaning,
    group_train_test_split,
    extract_labels_and_texts,
)
from message_classifier import (
    SVMClassifier,
    NaiveBayesClassifier,
    BERTClassifier,
    RoBERTaClassifier,
    apply_message_threshold,
    compute_mla_metrics,
)
from actor_classification import (
    attach_predictions,
    sweep_ast_t_values,
    T_VALUES_TRANSFORMER,
    T_VALUES_TRADITIONAL,
    AST_VALUES,
)
from evaluation import (
    compute_standard_metrics,
    plot_confusion_matrix,
    format_mla_table,
    format_context_table,
    save_mla_results,
    save_context_results,
    print_paper_comparison,
    PAPER_REFERENCE_RESULTS,
)

logger = logging.getLogger(__name__)
RESULTS_DIR = Path(__file__).parent / "results"


def run_traditional_model(
    clf,
    train_texts: list[str],
    y_train: np.ndarray,
    test_texts: list[str],
    y_test: np.ndarray,
    test_df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Train, evaluate, and run context determination for SVM or NB."""
    model_name = clf.name
    print(f"\n{'='*60}")
    print(f"  {model_name} Classifier")
    print(f"{'='*60}")

    # Train
    print(f"Training {model_name}...")
    clf.fit(train_texts, y_train)

    # MLA evaluation (no omissions for SVM/NB)
    y_pred = clf.predict(test_texts)
    mla_metrics = compute_mla_metrics(y_test, y_pred)

    print(f"\n--- MLA Results ({model_name}) ---")
    print(format_mla_table({model_name: mla_metrics}))

    # Standard metrics
    std_metrics = compute_standard_metrics(y_test, y_pred)
    print(f"\nStandard metrics (speaker-role classification, NOT grooming detection):")
    print(std_metrics["classification_report"])

    # Confusion matrix
    cm_path = RESULTS_DIR / f"confusion_matrix_{model_name.lower()}.png"
    plot_confusion_matrix(y_test, y_pred, model_name, save_path=cm_path)

    # Save MLA results
    save_mla_results(mla_metrics, model_name)

    # Context determination
    print(f"\n--- Context Determination ({model_name}) ---")
    test_with_proba = attach_predictions(test_df, clf, proba_col="pred_proba")
    ctx_results = sweep_ast_t_values(
        test_with_proba,
        model_name=model_name,
        t_values=T_VALUES_TRADITIONAL,
        ast_values=AST_VALUES,
    )

    print(format_context_table(ctx_results))
    save_context_results(ctx_results, model_name)

    return mla_metrics, ctx_results


def run_transformer_model(
    clf,
    train_texts: list[str],
    y_train: np.ndarray,
    test_texts: list[str],
    y_test: np.ndarray,
    test_df: pd.DataFrame,
    epochs: int = 3,
    batch_size: int = 16,
) -> tuple[dict, pd.DataFrame]:
    """Train, evaluate, and run context determination for BERT or RoBERTa."""
    model_name = clf.name
    print(f"\n{'='*60}")
    print(f"  {model_name} Classifier")
    print(f"{'='*60}")

    # Train
    print(f"Training {model_name} (epochs={epochs}, batch={batch_size})...")
    try:
        clf.fit(train_texts, y_train, epochs=epochs, batch_size=batch_size)
    except RuntimeError as e:
        print(f"ERROR: Could not train {model_name}: {e}")
        print(f"Run with --skip-transformers to skip BERT/RoBERTa")
        return {}, pd.DataFrame()

    # MLA evaluation with t value sweep
    mla_all = {}
    print(f"\n--- MLA Results ({model_name}) ---")

    for t in [0.2, 0.3, 0.4, 0.45]:
        proba = clf.predict_proba(test_texts)
        kept_mask, _ = apply_message_threshold(proba, t)
        y_pred_kept = (proba[kept_mask] >= 0.5).astype(int)
        y_true_kept = y_test[kept_mask]
        metrics = compute_mla_metrics(y_true_kept, y_pred_kept, n_total=len(y_test))
        mla_all[f"{model_name}_t={t}"] = metrics

    print(format_mla_table(mla_all))

    # Save MLA results for each t
    for label, metrics in mla_all.items():
        save_mla_results(metrics, label)

    # Confusion matrix (t=0.45 predictions)
    proba = clf.predict_proba(test_texts)
    kept_mask, _ = apply_message_threshold(proba, 0.45)
    y_pred_full = (proba >= 0.5).astype(int)
    cm_path = RESULTS_DIR / f"confusion_matrix_{model_name.lower()}.png"
    plot_confusion_matrix(y_test, y_pred_full, model_name, save_path=cm_path)

    # Standard metrics
    std_metrics = compute_standard_metrics(y_test, y_pred_full)
    print(f"\nStandard metrics (t=0, speaker-role classification):")
    print(std_metrics["classification_report"])

    # Context determination sweep
    print(f"\n--- Context Determination ({model_name}) ---")
    test_with_proba = attach_predictions(test_df, clf, proba_col="pred_proba")
    ctx_results = sweep_ast_t_values(
        test_with_proba,
        model_name=model_name,
        t_values=T_VALUES_TRANSFORMER,
        ast_values=AST_VALUES,
    )

    print(format_context_table(ctx_results))
    save_context_results(ctx_results, model_name)

    return mla_all, ctx_results


def main(skip_transformers: bool = False, debug: bool = False, fast: bool = False):
    """Main experimental pipeline."""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    print("\n" + "="*70)
    print("  Online Grooming Detection — Experimental Pipeline")
    print("  Paper: Street et al. (2025) — PJ Dataset")
    if fast:
        print("  [FAST MODE: 20k utterance sample for quick verification]")
    print("="*70)
    print()
    print("IMPORTANT CLARIFICATIONS:")
    print("  • This experiment classifies SPEAKER ROLES (Adult vs. Child language)")
    print("  • It does NOT classify individual messages as 'grooming' or 'non-grooming'")
    print("  • All PJ conversations are OG-positive — no true negatives exist here")
    print("  • Results cannot be compared to binary grooming detection accuracy")
    print()

    # ------------------------------------------------------------------ Stage 1
    print("--- Stage 1: Load Dataset ---")
    df = load_dataset()
    df = apply_minimal_cleaning(df)
    summarise_dataset(df)

    # Fast mode: sample 20k utterances (balanced), keeping group integrity
    if fast:
        print("[FAST MODE] Sampling 50 conversations for quick run...")
        sampled_convs = df["conversation_id"].drop_duplicates().sample(50, random_state=42)
        df = df[df["conversation_id"].isin(sampled_convs)].reset_index(drop=True)
        print(f"Sampled: {len(df):,} utterances from {df['conversation_id'].nunique()} conversations")
        print()

    # Check paired conversations
    convs = get_paired_conversations()
    n_paired = sum(1 for v in convs.values() if v["paired"])
    print(f"Paired conversations (both predator+officer files): {n_paired}")
    print()

    # ------------------------------------------------------------------ Stage 2
    print("--- Stage 2: Group-Based 80/20 Split ---")
    print("Split unit: CONVERSATION (not individual utterances)")
    print("No data leakage: same conversation always stays in same split")
    print()

    train_df, test_df = group_train_test_split(df, test_size=0.2, random_state=42)

    X_train_texts, y_train = extract_labels_and_texts(train_df)
    X_test_texts, y_test = extract_labels_and_texts(test_df)

    print(f"Train: {len(train_df):,} utterances in {train_df['conversation_id'].nunique()} conversations")
    print(f"Test:  {len(test_df):,} utterances in {test_df['conversation_id'].nunique()} conversations")
    print(f"Train label distribution: {int(y_train.sum())} Adult / {int((y_train==0).sum())} Child")
    print(f"Test  label distribution: {int(y_test.sum())} Adult / {int((y_test==0).sum())} Child")
    print()

    all_results = {}

    # ------------------------------------------------------------------ Stage 3: SVM
    svm = SVMClassifier()
    mla_svm, ctx_svm = run_traditional_model(
        svm, X_train_texts, y_train, X_test_texts, y_test, test_df
    )
    all_results["SVM"] = {"mla": mla_svm, "context": ctx_svm.to_dict() if not ctx_svm.empty else {}}

    # ------------------------------------------------------------------ Stage 4: NB
    nb = NaiveBayesClassifier()
    mla_nb, ctx_nb = run_traditional_model(
        nb, X_train_texts, y_train, X_test_texts, y_test, test_df
    )
    all_results["NB"] = {"mla": mla_nb, "context": ctx_nb.to_dict() if not ctx_nb.empty else {}}

    # ------------------------------------------------------------------ Stage 5: BERT
    if not skip_transformers:
        bert = BERTClassifier()
        mla_bert, ctx_bert = run_transformer_model(
            bert, X_train_texts, y_train, X_test_texts, y_test, test_df,
            epochs=3, batch_size=16,
        )
        all_results["BERT"] = {"mla": mla_bert, "context": ctx_bert.to_dict() if not ctx_bert.empty else {}}

        # ---------------------------------------------------------------- Stage 6: RoBERTa
        roberta = RoBERTaClassifier()
        mla_roberta, ctx_roberta = run_transformer_model(
            roberta, X_train_texts, y_train, X_test_texts, y_test, test_df,
            epochs=3, batch_size=16,
        )
        all_results["RoBERTa"] = {"mla": mla_roberta, "context": ctx_roberta.to_dict() if not ctx_roberta.empty else {}}
    else:
        print("\n[--skip-transformers] Skipping BERT and RoBERTa training.")
        print("To run transformers: python run_experiments.py")

    # ------------------------------------------------------------------ Stage 7: Summary
    print("\n" + "="*70)
    print("  EXPERIMENT SUMMARY")
    print("="*70)
    print()
    print("Task: Speaker-Role Classification (Adult=predator vs. Child=officer)")
    print("NOT: Grooming message detection, grooming conversation detection")
    print()

    # Print comparison with paper (reference only)
    for model_key in ["SVM", "NB"]:
        if model_key in all_results and all_results[model_key]["mla"]:
            print_paper_comparison(all_results[model_key]["mla"], model_key)

    # Save full summary
    summary_path = RESULTS_DIR / "experiment_summary.json"
    with open(summary_path, 'w') as f:
        # Serialize DataFrames to dict for JSON
        serializable = {}
        for k, v in all_results.items():
            serializable[k] = {
                "mla": v["mla"] if isinstance(v["mla"], dict) else {},
                "note": "See individual CSV/JSON files for full results"
            }
        json.dump(serializable, f, indent=2)
    print(f"\nFull results saved to: {RESULTS_DIR}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Online Grooming Detection experiments")
    parser.add_argument(
        "--skip-transformers",
        action="store_true",
        help="Skip BERT and RoBERTa (run SVM and NB only)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Sample 50 conversations for a quick verification run",
    )
    args = parser.parse_args()
    main(skip_transformers=args.skip_transformers, debug=getattr(args, 'debug', False), fast=args.fast)
