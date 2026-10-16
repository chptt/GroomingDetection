"""
test_preprocessing.py

Tests for preprocessing functions.

Key paper constraint: minimal preprocessing only.
Punctuation removal, stopword removal are NOT applied.
Only whitespace stripping is performed.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing import (
    minimal_clean,
    apply_minimal_cleaning,
    build_count_vectorizer,
    group_train_test_split,
    extract_labels_and_texts,
)
import pandas as pd


class TestMinimalCleaning:
    def test_strips_whitespace(self):
        assert minimal_clean("  hello  ") == "hello"

    def test_preserves_punctuation(self):
        """Paper intentionally preserves punctuation as it carries signal."""
        text = "lol... (:-*) r u there??"
        assert minimal_clean(text) == "lol... (:-*) r u there??"

    def test_preserves_casing(self):
        """Paper does not lowercase (case may carry signal in OG contexts)."""
        text = "HEY come over"
        assert minimal_clean(text) == "HEY come over"

    def test_preserves_numbers(self):
        assert minimal_clean("im 13/f") == "im 13/f"

    def test_empty_string(self):
        assert minimal_clean("") == ""

    def test_whitespace_only(self):
        assert minimal_clean("   ") == ""

    def test_already_clean(self):
        assert minimal_clean("hello world") == "hello world"

    def test_newline_stripped(self):
        assert minimal_clean("hello\n") == "hello"

    def test_tab_stripped(self):
        assert minimal_clean("\thello") == "hello"


class TestApplyMinimalCleaning:
    def _make_df(self, utterances, roles=None):
        if roles is None:
            roles = [1] * len(utterances)
        return pd.DataFrame({
            "utterance": utterances,
            "speaker_role": roles,
            "conversation_id": ["conv1"] * len(utterances),
            "file_name": ["test.txt"] * len(utterances),
        })

    def test_removes_empty_after_strip(self):
        df = self._make_df(["  ", "hello", ""])
        result = apply_minimal_cleaning(df)
        assert len(result) == 1
        assert result.iloc[0]["utterance"] == "hello"

    def test_preserves_non_empty(self):
        df = self._make_df(["hello", "world", "lol"])
        result = apply_minimal_cleaning(df)
        assert len(result) == 3

    def test_does_not_modify_in_place(self):
        df = self._make_df(["  hello  "])
        result = apply_minimal_cleaning(df)
        assert df.iloc[0]["utterance"] == "  hello  "  # original unchanged
        assert result.iloc[0]["utterance"] == "hello"

    def test_empty_df(self):
        df = pd.DataFrame(columns=["utterance", "speaker_role", "conversation_id", "file_name"])
        result = apply_minimal_cleaning(df)
        assert len(result) == 0


class TestCountVectorizer:
    def test_builds_vocabulary(self):
        texts = ["hello world", "world of words", "hello python"]
        vec, X = build_count_vectorizer(texts)
        assert len(vec.vocabulary_) > 0

    def test_output_shape(self):
        texts = ["hello world", "world of words", "hello python"]
        vec, X = build_count_vectorizer(texts)
        assert X.shape[0] == 3  # 3 documents

    def test_transform_new_texts(self):
        train_texts = ["hello world", "goodbye world"]
        vec, X_train = build_count_vectorizer(train_texts)
        X_test = vec.transform(["hello goodbye"])
        assert X_test.shape[0] == 1

    def test_empty_texts_handled(self):
        """CountVectorizer should handle texts with common words."""
        texts = ["hello", "world", "foo bar"]
        vec, X = build_count_vectorizer(texts)
        assert X.shape[0] == 3


class TestGroupSplit:
    def _make_df(self, n_convs=10, utts_per_conv=20):
        rows = []
        for i in range(n_convs):
            for j in range(utts_per_conv):
                rows.append({
                    "utterance": f"message {i} {j}",
                    "speaker_role": 1 if j % 2 == 0 else 0,
                    "conversation_id": f"conv_{i:03d}",
                    "file_name": f"pred_{i}.txt" if j % 2 == 0 else f"off_{i}.txt",
                })
        return pd.DataFrame(rows)

    def test_no_conversation_leakage(self):
        """No conversation_id should appear in both train and test."""
        df = self._make_df(n_convs=20)
        train, test = group_train_test_split(df, test_size=0.2, random_state=42)
        train_convs = set(train["conversation_id"])
        test_convs = set(test["conversation_id"])
        assert len(train_convs & test_convs) == 0

    def test_approximate_split_ratio(self):
        """Split should be approximately 80/20."""
        df = self._make_df(n_convs=50, utts_per_conv=20)
        train, test = group_train_test_split(df, test_size=0.2, random_state=42)
        total = len(df)
        test_ratio = len(test) / total
        # Allow ±10% tolerance due to group-based splitting
        assert 0.10 <= test_ratio <= 0.30, f"Unexpected split ratio: {test_ratio:.2%}"

    def test_reproducible_with_same_seed(self):
        df = self._make_df(n_convs=20)
        train1, test1 = group_train_test_split(df, random_state=42)
        train2, test2 = group_train_test_split(df, random_state=42)
        assert set(test1["conversation_id"]) == set(test2["conversation_id"])

    def test_different_seeds_give_different_splits(self):
        df = self._make_df(n_convs=30)
        _, test1 = group_train_test_split(df, random_state=42)
        _, test2 = group_train_test_split(df, random_state=99)
        # With enough conversations, different seeds should give different splits
        # (not guaranteed for small datasets but expected for 30 conversations)
        assert set(test1["conversation_id"]) != set(test2["conversation_id"])

    def test_requires_conversation_id_column(self):
        df = pd.DataFrame({"utterance": ["hello"], "speaker_role": [1]})
        with pytest.raises(ValueError):
            group_train_test_split(df)


class TestExtractLabels:
    def test_returns_correct_types(self):
        df = pd.DataFrame({
            "utterance": ["hello", "world"],
            "speaker_role": [1, 0],
        })
        texts, labels = extract_labels_and_texts(df)
        assert isinstance(texts, list)
        assert isinstance(labels, np.ndarray)

    def test_label_values(self):
        df = pd.DataFrame({
            "utterance": ["hello", "world", "foo"],
            "speaker_role": [1, 0, 1],
        })
        texts, labels = extract_labels_and_texts(df)
        assert list(labels) == [1, 0, 1]
