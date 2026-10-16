"""
actor_classification.py

Actor-level classification and significance analysis.

This module bridges message_classifier.py (utterance-level predictions)
and context_determination.py (transcript-level AC context).

It handles the full pipeline:
  1. Get per-utterance P(Adult) probabilities from a trained classifier
  2. Attach those probabilities back to the DataFrame
  3. Run context determination for each conversation
  4. Sweep AST and t values as described in the paper
  5. Return results tables matching paper Tables 4-7 format

For PJ dataset notes:
  - All conversations are OG-positive (AC context is the ground truth)
  - No False Positives exist on PJ alone → FP column is always 0 on this dataset
  - PAN12 is needed for FP/TN analysis but is not included in this repo
"""

from pathlib import Path
from typing import Optional
import logging

import numpy as np
import pandas as pd

from context_determination import (
    run_context_determination,
    compute_context_metrics,
    ActorDetermination,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

# AST values tested in the paper
AST_VALUES = [0.05, 0.01, 0.001]

# t values tested in paper (for BERT/RoBERTa)
T_VALUES_TRANSFORMER = [0.2, 0.3, 0.4, 0.45]

# SVM and NB have no omissions (t=0)
T_VALUES_TRADITIONAL = [0.0]


def attach_predictions(
    df: pd.DataFrame,
    classifier,
    proba_col: str = "pred_proba",
) -> pd.DataFrame:
    """
    Run a trained classifier on all utterances and attach probabilities to DataFrame.

    Args:
        df         : DataFrame with 'utterance' column
        classifier : trained SVM, NB, BERT, or RoBERTa classifier with predict_proba()
        proba_col  : name of the new column to add

    Returns:
        DataFrame with proba_col added
    """
    df = df.copy()
    texts = df["utterance"].tolist()

    logger.info(f"Getting predictions from {getattr(classifier, 'name', type(classifier).__name__)}...")
    proba = classifier.predict_proba(texts)
    df[proba_col] = proba
    return df


def sweep_ast_t_values(
    test_df_with_proba: pd.DataFrame,
    model_name: str,
    t_values: Optional[list[float]] = None,
    ast_values: Optional[list[float]] = None,
    proba_col: str = "pred_proba",
) -> pd.DataFrame:
    """
    Sweep over all combinations of t and AST values and collect context metrics.

    Returns a DataFrame with rows matching paper Tables 4-7 format.

    Args:
        test_df_with_proba: test utterances with pred_proba column attached
        model_name        : string label for the model (e.g. 'BERT', 'RoBERTa')
        t_values          : list of message threshold values to sweep
        ast_values        : list of AST p-value thresholds to sweep
        proba_col         : column name for P(Adult) probabilities

    Returns:
        DataFrame with columns: model, t, ast, TP, FP, TN, FN, F1, n_transcripts
    """
    if t_values is None:
        # Determine based on model type
        if model_name.upper() in ("SVM", "NB"):
            t_values = T_VALUES_TRADITIONAL
        else:
            t_values = T_VALUES_TRANSFORMER

    if ast_values is None:
        ast_values = AST_VALUES

    rows = []
    for t in t_values:
        for ast in ast_values:
            ctx_df = run_context_determination(
                test_df_with_proba,
                pred_proba_col=proba_col,
                t=t,
                ast=ast,
            )
            metrics = compute_context_metrics(ctx_df)
            rows.append({
                "Model": model_name,
                "t": t,
                "AST": ast,
                **metrics,
            })
            logger.debug(f"{model_name} | t={t} | AST={ast} | F1={metrics['F1']:.3f}")

    return pd.DataFrame(rows)


def run_full_actor_analysis(
    test_df: pd.DataFrame,
    classifiers: dict,
    proba_col: str = "pred_proba",
) -> dict[str, pd.DataFrame]:
    """
    Run the complete actor classification and context determination pipeline
    for all trained classifiers.

    Args:
        test_df    : test DataFrame (utterances, speaker_role, conversation_id)
        classifiers: dict mapping model name -> trained classifier instance
        proba_col  : column name for probabilities

    Returns:
        dict mapping model_name -> results DataFrame (format matches paper tables)
    """
    all_results = {}

    for model_name, clf in classifiers.items():
        logger.info(f"\n--- Running context determination for {model_name} ---")

        # Attach probabilities
        test_with_proba = attach_predictions(test_df, clf, proba_col=proba_col)

        # Determine t_values based on model type
        is_transformer = model_name.upper() not in ("SVM", "NB")
        t_values = T_VALUES_TRANSFORMER if is_transformer else T_VALUES_TRADITIONAL

        # Sweep AST and t values
        results = sweep_ast_t_values(
            test_with_proba,
            model_name=model_name,
            t_values=t_values,
            ast_values=AST_VALUES,
            proba_col=proba_col,
        )

        all_results[model_name] = results

    return all_results


def format_results_table(results_df: pd.DataFrame) -> str:
    """Format results into a table matching paper style (Tables 4-7)."""
    lines = []
    lines.append(f"\n{'Model':<15} {'t':>5} {'AST':>8} {'TP':>6} {'FN':>6} {'FP':>6} {'TN':>7} {'F1':>7}")
    lines.append("-" * 60)

    for _, row in results_df.iterrows():
        lines.append(
            f"{row['Model']:<15} {row['t']:>5.2f} {row['AST']:>8.3f} "
            f"{row['TP']:>6} {row['FN']:>6} {row['FP']:>6} {row['TN']:>7} "
            f"{row['F1']:>7.3f}"
        )

    return "\n".join(lines)


def save_results(results_df: pd.DataFrame, path: Path) -> None:
    """Save results to CSV."""
    path.parent.mkdir(exist_ok=True)
    results_df.to_csv(path, index=False)
    logger.info(f"Results saved to {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("actor_classification.py — import and use with trained classifiers")
    print("See run_experiments.py for full pipeline execution")
