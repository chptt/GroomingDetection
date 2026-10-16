"""
evaluation.py

Evaluation metrics and visualization for the Online Grooming Detection system.

Two evaluation tracks, which must NOT be conflated:
  1. Message-Level Analysis (MLA): speaker-role classification performance
     Metrics: TP%, IA%, IC%, O% (paper Tables 2/3)
     - This measures how well the model classifies Adult vs. Child LANGUAGE
     - NOT grooming detection performance

  2. Context Determination: transcript-level Adult-Child context detection
     Metrics: TP, FP, TN, FN, F1 (paper Tables 4/5/6/7)
     - This measures whether a conversation is flagged as Adult-Child interaction
     - On PJ dataset: all-positive set (no true negatives exist)
     - FP/TN require PAN12 (not included in this dataset)

CRITICAL: Results are labeled as "speaker-role classification" and
"Adult-Child context determination" — never as "grooming detection accuracy".
The published paper results are used as comparison targets ONLY.
"""

from pathlib import Path
import json
import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

# Paper's published results for comparison reference ONLY
# Source: Street et al. (2025), Tables 2/4
# These are never embedded in experiment outputs — only displayed as reference
PAPER_REFERENCE_RESULTS = {
    "MLA_PJ_sample": {
        "SVM":            {"TP": 76.0, "IA": 13.5, "IC": 10.5, "O": 0.0},
        "NB":             {"TP": 74.7, "IA": 12.0, "IC": 13.3, "O": 0.0},
        "BERT_t=0.2":     {"TP": 82.2, "IA":  9.3, "IC":  8.5, "O": 9.2},
        "BERT_t=0.3":     {"TP": 83.7, "IA":  8.4, "IC":  7.7, "O": 15.3},
        "BERT_t=0.4":     {"TP": 85.6, "IA":  7.8, "IC":  6.6, "O": 22.7},
        "BERT_t=0.45":    {"TP": 87.3, "IA":  7.1, "IC":  5.6, "O": 30.0},
        "RoBERTa_t=0.2":  {"TP": 83.3, "IA":  7.7, "IC":  9.0, "O": 8.8},
        "RoBERTa_t=0.3":  {"TP": 84.9, "IA":  6.9, "IC":  8.2, "O": 14.7},
        "RoBERTa_t=0.4":  {"TP": 87.2, "IA":  6.0, "IC":  8.9, "O": 23.6},
        "RoBERTa_t=0.45": {"TP": 88.8, "IA":  5.3, "IC":  5.9, "O": 30.4},
    },
    "Context_PJ_AST_0.05": {
        "SVM":            {"TP": 118, "FN": 35, "F1": 0.871},
        "NB":             {"TP": 116, "FN": 37, "F1": 0.862},
        "BERT_t=0.2":     {"TP": 143, "FN": 10, "F1": 0.966},
        "RoBERTa_t=0.45": {"TP": 145, "FN":  8, "F1": 0.973},
    }
}


# ---------------------------------------------------------------------------
# Standard classification metrics (used alongside paper-specific metrics)
# ---------------------------------------------------------------------------

def compute_standard_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list = [0, 1],
    target_names: list = ["Officer/Child", "Predator/Adult"],
) -> dict:
    """
    Compute standard sklearn classification metrics.

    These complement the paper's custom TP/IA/IC/O metrics.
    They are labeled as 'speaker-role classification' metrics —
    never as grooming detection metrics.
    """
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "precision_adult": round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "recall_adult": round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "f1_adult": round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "classification_report": classification_report(
            y_true, y_pred, target_names=target_names, zero_division=0
        ),
    }


# ---------------------------------------------------------------------------
# Confusion matrix plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    labels: list = [0, 1],
    target_names: list = ["Officer\n(Child)", "Predator\n(Adult)"],
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot and save confusion matrix for speaker-role classification.
    
    LABELING NOTE: Axes are labeled 'Officer (Child)' and 'Predator (Adult)'
    to reflect the actual classification task. This is speaker-role classification,
    NOT grooming message classification.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
        cbar=True,
    )

    ax.set_xlabel("Predicted Speaker Role", fontsize=12)
    ax.set_ylabel("True Speaker Role", fontsize=12)
    ax.set_title(
        f"{model_name}\nSpeaker-Role Classification Confusion Matrix\n"
        f"(NOT grooming detection)",
        fontsize=11
    )

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# MLA results table (paper format)
# ---------------------------------------------------------------------------

def format_mla_table(results: dict[str, dict]) -> str:
    """
    Format MLA results in the paper's Table 2/3 style.
    
    Header: Model | TP | IA | IC | O
    
    NOTE: These are speaker-role classification metrics (Adult vs. Child),
    not grooming detection metrics.
    """
    lines = [
        "\n=== Message-Level Analysis Results ===",
        "(Task: Adult vs. Child speaker-role classification — NOT grooming detection)",
        "",
        f"{'Model':<20} {'TP%':>7} {'IA%':>7} {'IC%':>7} {'O%':>7}",
        "-" * 50,
    ]

    for model_label, metrics in results.items():
        lines.append(
            f"{model_label:<20} {metrics['TP']:>7.1f} {metrics['IA']:>7.1f} "
            f"{metrics['IC']:>7.1f} {metrics['O']:>7.1f}"
        )

    lines.append("")
    lines.append("TP = True Positive (correct Adult/Child determination)")
    lines.append("IA = Incorrect Adult Determination (predicted Adult, was Child)")
    lines.append("IC = Incorrect Child Determination (predicted Child, was Adult)")
    lines.append("O  = Omissions (excluded by message threshold t; % of all messages)")
    lines.append("Note: TP+IA+IC percentages are out of kept messages; O is out of all messages.")
    return "\n".join(lines)


def format_context_table(results_df: pd.DataFrame) -> str:
    """
    Format context determination results in paper's Tables 4-7 style.

    NOTE: On PJ dataset, FP=0 and TN=0 (all-positive set).
    This limitation must be displayed.
    """
    lines = [
        "\n=== Context Determination Results ===",
        "(Task: Detect Adult-Child conversation context)",
        "(PJ dataset is all-positive: FP=0, TN=0 by design — cross-dataset testing with PAN12 needed for FP/TN)",
        "",
        f"{'Model':<15} {'t':>5} {'AST':>8} {'TP':>5} {'FN':>5} {'FP':>5} {'TN':>5} {'F1':>7}",
        "-" * 60,
    ]

    for _, row in results_df.iterrows():
        lines.append(
            f"{str(row['Model']):<15} {row['t']:>5.2f} {row['AST']:>8.3f} "
            f"{row['TP']:>5} {row['FN']:>5} {row['FP']:>5} {row['TN']:>5} "
            f"{row['F1']:>7.3f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def save_mla_results(
    results: dict,
    model_name: str,
    save_dir: Optional[Path] = None,
) -> None:
    """Save MLA metrics to JSON and print formatted table."""
    save_dir = save_dir or RESULTS_DIR
    save_dir.mkdir(exist_ok=True, parents=True)

    path = save_dir / f"mla_{model_name.lower().replace(' ', '_')}.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"MLA results saved to {path}")


def save_context_results(
    results_df: pd.DataFrame,
    model_name: str,
    save_dir: Optional[Path] = None,
) -> None:
    """Save context determination metrics to CSV."""
    save_dir = save_dir or RESULTS_DIR
    save_dir.mkdir(exist_ok=True, parents=True)

    path = save_dir / f"context_{model_name.lower().replace(' ', '_')}.csv"
    results_df.to_csv(path, index=False)
    logger.info(f"Context results saved to {path}")


def print_paper_comparison(
    our_results: dict,
    model_name: str,
) -> None:
    """
    Print side-by-side comparison with published paper results.
    
    IMPORTANT: Paper results are reference targets only.
    Differences are expected due to:
      - Different dataset version (this repo is a cleaned version)
      - Different splits
      - Potential differences in implementations
    """
    print(f"\n=== Comparison with Paper Published Results: {model_name} ===")
    print("(Paper results are for REFERENCE ONLY — not embedded in experiment outputs)")
    print()

    paper_ref = PAPER_REFERENCE_RESULTS.get("MLA_PJ_sample", {})
    paper_key = model_name.replace(" ", "_")

    if paper_key in paper_ref:
        print(f"{'Metric':<10} {'Our Result':>15} {'Paper Result':>15}")
        print("-" * 42)
        paper = paper_ref[paper_key]
        for key in ["TP", "IA", "IC", "O"]:
            our_val = our_results.get(key, "N/A")
            paper_val = paper.get(key, "N/A")
            print(f"{key:<10} {str(our_val):>15} {str(paper_val):>15}")
    else:
        print(f"No paper reference found for '{model_name}'")
        print(f"Our results: {our_results}")
