"""
preprocessing.py

Preprocessing pipeline for the PervertedJustice dataset.

Important note from the paper (Section 4.2):
    "Similar to Villatoro-Tello et al. (2012), preprocessing methods will not
     be used in our approach, beyond the exclusion of transcripts shorter than
     10 messages in length for PAN12 and the exclusion of email transcripts
     within the PJ dataset."

Therefore the paper's approach is intentionally MINIMAL preprocessing.
Punctuation removal, stopword removal, etc. are deliberately avoided as
they may remove signals useful for Adult/Child language classification.

This module provides:
  - Minimal preprocessing matching the paper's approach (raw text)
  - CountVectorizer-based features for SVM/NB (paper uses CountVectorizer)
  - BERT/RoBERTa tokenization for transformer models
  - Group-based train/test splitting by conversation
"""

from pathlib import Path
from typing import Optional
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal text cleaning (paper approach: minimal / no preprocessing)
# ---------------------------------------------------------------------------

def minimal_clean(text: str) -> str:
    """
    Minimal cleaning matching the paper's approach.
    Only strip leading/trailing whitespace; preserve punctuation and casing.
    
    The paper deliberately avoids punctuation removal and stopword removal
    because these may remove useful OG classification signals (emoticons,
    abbreviations, "textspeak").
    """
    return text.strip()


def apply_minimal_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply minimal cleaning to the utterance column."""
    df = df.copy()
    df["utterance"] = df["utterance"].apply(minimal_clean)
    # Remove any rows that became empty after cleaning
    before = len(df)
    df = df[df["utterance"].str.len() > 0].reset_index(drop=True)
    if len(df) < before:
        logger.info(f"Removed {before - len(df)} empty utterances after cleaning")
    return df


# ---------------------------------------------------------------------------
# Group-based train/test splitting (conversation-level, never utterance-level)
# ---------------------------------------------------------------------------

def group_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test sets keeping all utterances from the same
    conversation in the same split.

    The paper uses 80/20 split. The grouping unit is conversation_id so that
    predator and officer utterances from the same conversation always end up
    in the same set — preventing data leakage.

    Args:
        df: DataFrame with 'conversation_id' column
        test_size: fraction for test set (default 0.2 as per paper)
        random_state: for reproducibility

    Returns:
        (train_df, test_df) — no leakage between splits
    """
    if "conversation_id" not in df.columns:
        raise ValueError("DataFrame must have 'conversation_id' column")

    groups = df["conversation_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    train_idx, test_idx = next(gss.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Verify no conversation leakage
    train_convs = set(train_df["conversation_id"].unique())
    test_convs = set(test_df["conversation_id"].unique())
    overlap = train_convs & test_convs
    assert len(overlap) == 0, f"Data leakage: {len(overlap)} conversations in both splits!"

    logger.info(
        f"Split: train={len(train_df)} utterances ({train_df['conversation_id'].nunique()} convs), "
        f"test={len(test_df)} utterances ({test_df['conversation_id'].nunique()} convs)"
    )

    return train_df, test_df


# ---------------------------------------------------------------------------
# Feature extraction for SVM / Naive Bayes
# ---------------------------------------------------------------------------

def build_count_vectorizer(
    train_texts: list[str],
    **kwargs,
) -> tuple[CountVectorizer, np.ndarray]:
    """
    Build a CountVectorizer on training texts (paper uses CountVectorizer for SVM/NB).
    
    The paper uses sklearn's 'pipeline' with a 'countVectoriser' approach.
    No custom preprocessing — CountVectorizer defaults are used.

    Returns:
        (fitted_vectorizer, train_matrix)
    """
    vectorizer = CountVectorizer(**kwargs)
    X_train = vectorizer.fit_transform(train_texts)
    return vectorizer, X_train


def build_tfidf_vectorizer(
    train_texts: list[str],
    **kwargs,
) -> tuple[TfidfVectorizer, np.ndarray]:
    """
    Alternative: TF-IDF vectorizer (mentioned in related work; not paper's primary method).
    Provided for comparison purposes.
    """
    vectorizer = TfidfVectorizer(**kwargs)
    X_train = vectorizer.fit_transform(train_texts)
    return vectorizer, X_train


# ---------------------------------------------------------------------------
# BERT / RoBERTa tokenization
# ---------------------------------------------------------------------------

def tokenize_for_transformer(
    texts: list[str],
    tokenizer,
    max_length: int = 128,
) -> dict:
    """
    Tokenize texts using a HuggingFace tokenizer for BERT or RoBERTa.

    Args:
        texts: list of utterance strings
        tokenizer: a loaded HuggingFace tokenizer
        max_length: token length limit (128 is standard for short messages)

    Returns:
        dict with 'input_ids', 'attention_mask', (and 'token_type_ids' for BERT)
    """
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="tf",  # Paper uses TensorFlow
    )
    return encodings


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def extract_labels_and_texts(
    df: pd.DataFrame,
) -> tuple[list[str], np.ndarray]:
    """Extract (texts, labels) from a DataFrame.
    
    Returns:
        texts: list of utterance strings
        labels: numpy array of speaker_role (1=predator/Adult, 0=officer/Child)
    """
    texts = df["utterance"].tolist()
    labels = df["speaker_role"].values.astype(int)
    return texts, labels


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_dataset, summarise_dataset

    logging.basicConfig(level=logging.INFO)
    df = load_dataset()
    df = apply_minimal_cleaning(df)
    summarise_dataset(df)

    train_df, test_df = group_train_test_split(df)
    print(f"Train: {len(train_df)} utterances, {train_df['conversation_id'].nunique()} conversations")
    print(f"Test:  {len(test_df)} utterances, {test_df['conversation_id'].nunique()} conversations")

    X_train_texts, y_train = extract_labels_and_texts(train_df)
    vec, X_train = build_count_vectorizer(X_train_texts)
    print(f"CountVectorizer vocabulary size: {len(vec.vocabulary_)}")
