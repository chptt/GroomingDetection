"""
context_determination.py

Context Determination approach from Section 3.2 of the paper.

This module implements the full transcript context determination algorithm
that determines whether a conversation has an Adult-Child (AC) context,
indicating a potential Online Grooming (OG) interaction.

Algorithm (from paper Figure 3):
  For each conversation (transcript):
    1. Collect all message predictions (probabilities) per actor
    2. Apply Message Threshold t: omit messages where (0.5+t) > n > (0.5-t)
       (Only for BERT/RoBERTa; SVM/NB have no omissions i.e. t=0)
    3. For each actor (predator and officer):
       a. Compute the mean of remaining predictions
       b. Compute the binomial test p-value
       c. Apply Actor Significance Threshold (AST):
          - If p_value < AST:
            * If mean > 0.5 → Actor is Adult (A)
            * If mean <= 0.5 → Actor is Child (C)
          - If p_value >= AST → Actor is Not Significant (NS)
    4. Determine transcript context:
       Context = AC if:
         (Actor A is Adult XOR Actor B is Adult) AND
         (Actor A is Child XOR Actor B is Child)
       Otherwise: Context = Non-AC (NS)

AST values tested in paper: 0.05, 0.01, 0.001
t values tested in paper: 0.2, 0.3, 0.4, 0.45

IMPORTANT:
  - All PJ transcripts are OG-positive (adult-child interactions confirmed)
  - Therefore on PJ: only TP and FN exist (no FP or TN possible)
  - PAN12 (not included here) provides negative examples for FP/TN
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import logging

import numpy as np
import pandas as pd
from scipy.stats import binomtest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Actor significance determination
# ---------------------------------------------------------------------------

@dataclass
class ActorDetermination:
    """Result of the actor significance determination for one actor."""
    actor_id: str            # conversation_id or actor label
    role: str                # 'predator' or 'officer'
    n_messages: int          # messages after omission
    n_omitted: int           # messages omitted by threshold t
    mean_prediction: float   # mean P(Adult) over kept messages
    p_value: float           # binomial test p-value
    determination: str       # 'A' (Adult), 'C' (Child), or 'NS' (Not Significant)


def determine_actor_significance(
    predictions: np.ndarray,
    actor_id: str,
    role: str,
    t: float = 0.0,
    ast: float = 0.05,
) -> ActorDetermination:
    """
    Determine if an actor is significantly an Adult (A), Child (C),
    or Not Significant (NS).

    Args:
        predictions: array of P(Adult) values for this actor's messages
        actor_id: identifier (e.g. conversation_id)
        role: 'predator' or 'officer'
        t: message threshold; omit messages where (0.5+t) > n > (0.5-t)
        ast: Actor Significance Threshold (p-value threshold, e.g. 0.05)

    Returns:
        ActorDetermination with fields filled in
    """
    # Step 1: Apply message threshold t (Equation 1)
    if t > 0.0:
        lower = 0.5 - t
        upper = 0.5 + t
        omit_mask = (predictions > lower) & (predictions < upper)
        kept = predictions[~omit_mask]
        n_omitted = int(omit_mask.sum())
    else:
        kept = predictions
        n_omitted = 0

    if len(kept) == 0:
        # No messages remain after omission → Not Significant
        return ActorDetermination(
            actor_id=actor_id,
            role=role,
            n_messages=0,
            n_omitted=n_omitted,
            mean_prediction=float('nan'),
            p_value=float('nan'),
            determination="NS",
        )

    # Step 2: Convert P(Adult) to binary for binomial test
    # A prediction >= 0.5 is classified as Adult
    adult_votes = (kept >= 0.5).sum()
    n_kept = len(kept)

    # Step 3: Binomial test
    # H0: the actor's messages are equally likely to be Adult or Child (p=0.5)
    # H1: there is a significant tendency toward Adult or Child
    # The AST is a p-value threshold; lower AST = stricter significance
    result = binomtest(int(adult_votes), n_kept, p=0.5, alternative='two-sided')
    p_value = result.pvalue
    mean_pred = float(kept.mean())

    # Step 4: Determine actor category
    if p_value < ast:
        # Significant: determine direction
        if mean_pred > 0.5:
            determination = "A"   # Significantly Adult
        else:
            determination = "C"   # Significantly Child
    else:
        determination = "NS"      # Not Significant

    return ActorDetermination(
        actor_id=actor_id,
        role=role,
        n_messages=n_kept,
        n_omitted=n_omitted,
        mean_prediction=mean_pred,
        p_value=p_value,
        determination=determination,
    )


# ---------------------------------------------------------------------------
# Transcript context determination
# ---------------------------------------------------------------------------

@dataclass
class TranscriptContext:
    """Full transcript context determination result."""
    conversation_id: str
    actor_pred: ActorDetermination   # predator actor
    actor_off: ActorDetermination    # officer actor
    context: str                     # 'AC' (Adult-Child) or 'non-AC'
    is_ac: bool                      # True if Adult-Child context detected


def determine_transcript_context(
    pred_determination: ActorDetermination,
    off_determination: ActorDetermination,
    conversation_id: str,
) -> TranscriptContext:
    """
    Combine two actor determinations into an overall transcript context.

    Paper rule (from Figure 3):
        Transcript is AC if:
            (Actor A is Adult XOR Actor B is Adult)
            AND
            (Actor A is Child XOR Actor B is Child)
        
        In other words, exactly one actor is Adult AND exactly one is Child.
        If either actor is NS, the transcript is non-AC.

    Args:
        pred_determination: ActorDetermination for predator
        off_determination: ActorDetermination for officer
        conversation_id: identifier string

    Returns:
        TranscriptContext
    """
    a = pred_determination.determination
    b = off_determination.determination

    # Paper logic: (A xor B is Adult) AND (A xor B is Child)
    # i.e., one must be 'A' and the other must be 'C'
    one_adult = (a == "A") ^ (b == "A")   # XOR: exactly one is Adult
    one_child = (a == "C") ^ (b == "C")   # XOR: exactly one is Child

    # NS in either position breaks the AC context
    either_ns = (a == "NS") or (b == "NS")

    is_ac = (one_adult and one_child) and (not either_ns)
    context = "AC" if is_ac else "non-AC"

    return TranscriptContext(
        conversation_id=conversation_id,
        actor_pred=pred_determination,
        actor_off=off_determination,
        context=context,
        is_ac=is_ac,
    )


# ---------------------------------------------------------------------------
# Batch context determination
# ---------------------------------------------------------------------------

def run_context_determination(
    test_df: pd.DataFrame,
    pred_proba_col: str = "pred_proba",
    t: float = 0.0,
    ast: float = 0.05,
) -> pd.DataFrame:
    """
    Run context determination over all conversations in a test DataFrame.

    The DataFrame must have columns:
        - conversation_id : identifies which conversation the utterance belongs to
        - speaker_role    : 1=predator/Adult, 0=officer/Child
        - pred_proba      : P(Adult) from the classifier

    For PJ dataset: all conversations are truly AC, so ground-truth label is always 1.
    The function sets true_ac=True for all (since all PJ conversations are OG-positive).

    Args:
        test_df       : DataFrame with utterance-level predictions
        pred_proba_col: column name containing P(Adult) probabilities
        t             : message threshold value
        ast           : Actor Significance Threshold

    Returns:
        DataFrame with one row per conversation, columns:
            conversation_id, pred_is_ac, true_ac,
            pred_det, off_det, pred_mean, off_mean,
            pred_pval, off_pval, pred_n, off_n
    """
    if pred_proba_col not in test_df.columns:
        raise ValueError(f"Column '{pred_proba_col}' not found in DataFrame")

    results = []

    for conv_id, conv_df in test_df.groupby("conversation_id"):
        # Split by speaker role
        pred_rows = conv_df[conv_df["speaker_role"] == 1]
        off_rows = conv_df[conv_df["speaker_role"] == 0]

        if len(pred_rows) == 0 or len(off_rows) == 0:
            logger.debug(f"Skipping {conv_id}: missing predator or officer utterances")
            continue

        pred_proba = pred_rows[pred_proba_col].values
        off_proba = off_rows[pred_proba_col].values

        # Determine actor significance
        pred_det = determine_actor_significance(pred_proba, conv_id, "predator", t=t, ast=ast)
        off_det = determine_actor_significance(off_proba, conv_id, "officer", t=t, ast=ast)

        # Determine transcript context
        ctx = determine_transcript_context(pred_det, off_det, conv_id)

        results.append({
            "conversation_id": conv_id,
            "pred_is_ac": ctx.is_ac,
            "true_ac": True,          # All PJ conversations are OG-positive (Adult-Child)
            "pred_determination": pred_det.determination,
            "off_determination": off_det.determination,
            "pred_mean": round(pred_det.mean_prediction, 4) if not np.isnan(pred_det.mean_prediction) else None,
            "off_mean": round(off_det.mean_prediction, 4) if not np.isnan(off_det.mean_prediction) else None,
            "pred_pval": round(pred_det.p_value, 6) if not np.isnan(pred_det.p_value) else None,
            "off_pval": round(off_det.p_value, 6) if not np.isnan(off_det.p_value) else None,
            "pred_n_kept": pred_det.n_messages,
            "pred_n_omitted": pred_det.n_omitted,
            "off_n_kept": off_det.n_messages,
            "off_n_omitted": off_det.n_omitted,
            "t_value": t,
            "ast_value": ast,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Context determination metrics
# ---------------------------------------------------------------------------

def compute_context_metrics(ctx_df: pd.DataFrame) -> dict:
    """
    Compute TP, FP, TN, FN, F1 for context determination.

    For PJ dataset (all positives): only TP and FN are meaningful.
    FP and TN require negative examples (e.g., from PAN12).

    Args:
        ctx_df: DataFrame from run_context_determination()

    Returns:
        dict with TP, FP, TN, FN, F1 counts and F1 score
    """
    tp = int(((ctx_df["pred_is_ac"] == True) & (ctx_df["true_ac"] == True)).sum())
    fp = int(((ctx_df["pred_is_ac"] == True) & (ctx_df["true_ac"] == False)).sum())
    tn = int(((ctx_df["pred_is_ac"] == False) & (ctx_df["true_ac"] == False)).sum())
    fn = int(((ctx_df["pred_is_ac"] == False) & (ctx_df["true_ac"] == True)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1": round(f1, 3),
        "n_transcripts": len(ctx_df),
    }
