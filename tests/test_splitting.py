"""
test_splitting.py

Tests for group-based splitting and context determination thresholds.

Validates:
  - Group integrity (no conversation leakage between train/test)
  - 80/20 split ratio
  - Message Threshold (Equation 1) correctness
  - Actor Significance Threshold logic
  - Context determination logic (AC vs non-AC)
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing import group_train_test_split
from message_classifier import apply_message_threshold, compute_mla_metrics
from context_determination import (
    determine_actor_significance,
    determine_transcript_context,
    ActorDetermination,
)


# ---------------------------------------------------------------------------
# Group-based splitting
# ---------------------------------------------------------------------------

def make_df(n_convs: int = 20, utts_per_conv: int = 30) -> pd.DataFrame:
    rows = []
    for i in range(n_convs):
        for j in range(utts_per_conv):
            rows.append({
                "utterance": f"text {i} {j}",
                "speaker_role": 1 if j % 2 == 0 else 0,
                "conversation_id": f"conv_{i:04d}",
                "file_name": f"file_{i}.txt",
            })
    return pd.DataFrame(rows)


class TestGroupSplitting:
    def test_zero_conversation_overlap(self):
        df = make_df(n_convs=50)
        train, test = group_train_test_split(df)
        train_ids = set(train["conversation_id"])
        test_ids = set(test["conversation_id"])
        assert train_ids & test_ids == set(), "Conversation IDs leaked between splits"

    def test_covers_all_data(self):
        df = make_df(n_convs=20)
        train, test = group_train_test_split(df)
        assert len(train) + len(test) == len(df)

    def test_split_at_conversation_boundary(self):
        """All utterances from one conversation must be in the SAME split."""
        df = make_df(n_convs=30)
        train, test = group_train_test_split(df, random_state=42)

        for conv_id in df["conversation_id"].unique():
            in_train = conv_id in train["conversation_id"].values
            in_test = conv_id in test["conversation_id"].values
            # Must be in one or the other, not both
            assert in_train != in_test, f"{conv_id} appears in both splits"

    def test_both_labels_in_train(self):
        """Train set should have both Adult and Child utterances."""
        df = make_df(n_convs=30)
        train, _ = group_train_test_split(df)
        assert 0 in train["speaker_role"].values
        assert 1 in train["speaker_role"].values

    def test_both_labels_in_test(self):
        """Test set should have both Adult and Child utterances."""
        df = make_df(n_convs=30)
        _, test = group_train_test_split(df)
        assert 0 in test["speaker_role"].values
        assert 1 in test["speaker_role"].values


# ---------------------------------------------------------------------------
# Message Threshold (Equation 1)
# ---------------------------------------------------------------------------

class TestMessageThreshold:
    def test_no_omission_at_t_zero(self):
        proba = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        kept, omit = apply_message_threshold(proba, t=0.0)
        # t=0: lower=0.5, upper=0.5 — band is empty, nothing omitted
        # Actually: (0.5+0) > n > (0.5-0) → 0.5 > n > 0.5 → impossible
        assert omit.sum() == 0
        assert kept.sum() == len(proba)

    def test_omits_midrange_at_t_0_2(self):
        """t=0.2: omit if 0.7 > n > 0.3"""
        proba = np.array([0.1, 0.35, 0.5, 0.65, 0.9])
        kept, omit = apply_message_threshold(proba, t=0.2)
        # 0.35 is in (0.3, 0.7) → omitted
        # 0.5  is in (0.3, 0.7) → omitted
        # 0.65 is in (0.3, 0.7) → omitted
        # 0.1  is NOT in (0.3, 0.7) → kept
        # 0.9  is NOT in (0.3, 0.7) → kept
        assert omit[0] == False  # 0.1 kept
        assert omit[1] == True   # 0.35 omitted
        assert omit[2] == True   # 0.5 omitted
        assert omit[3] == True   # 0.65 omitted
        assert omit[4] == False  # 0.9 kept

    def test_omits_more_at_higher_t(self):
        """Higher t → more messages omitted."""
        proba = np.array([0.4, 0.45, 0.5, 0.55, 0.6])
        _, omit_02 = apply_message_threshold(proba, t=0.2)
        _, omit_045 = apply_message_threshold(proba, t=0.45)
        assert omit_045.sum() >= omit_02.sum()

    def test_boundary_values_not_omitted(self):
        """Values exactly at boundary (0.5+t) are NOT omitted (strict inequality)."""
        t = 0.2
        proba = np.array([0.3, 0.7])  # exactly at lower and upper
        _, omit = apply_message_threshold(proba, t=t)
        # (0.5+0.2) > n > (0.5-0.2) → 0.7 > n > 0.3
        # 0.3 and 0.7 are NOT strictly inside → NOT omitted
        assert omit[0] == False
        assert omit[1] == False


# ---------------------------------------------------------------------------
# MLA Metrics
# ---------------------------------------------------------------------------

class TestMLAMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        m = compute_mla_metrics(y_true, y_pred)
        assert m["TP"] == 100.0
        assert m["IA"] == 0.0
        assert m["IC"] == 0.0
        assert m["O"] == 0.0

    def test_all_wrong(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1])
        m = compute_mla_metrics(y_true, y_pred)
        assert m["TP"] == 0.0

    def test_omission_percentage(self):
        """O% should reflect omitted / total (not kept)."""
        y_true_kept = np.array([1, 0])   # 2 kept
        y_pred_kept = np.array([1, 0])
        m = compute_mla_metrics(y_true_kept, y_pred_kept, n_total=10)
        assert m["O"] == 80.0  # 8 omitted out of 10

    def test_ia_metric(self):
        """IA = predicted Adult but was Child."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1])  # both children predicted as adult
        m = compute_mla_metrics(y_true, y_pred)
        assert m["IA"] == 50.0  # 2 IA out of 4 kept

    def test_ic_metric(self):
        """IC = predicted Child but was Adult."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])  # both adults predicted as child
        m = compute_mla_metrics(y_true, y_pred)
        assert m["IC"] == 50.0  # 2 IC out of 4 kept


# ---------------------------------------------------------------------------
# Actor Significance and Context Determination
# ---------------------------------------------------------------------------

class TestActorSignificance:
    def test_adult_determination(self):
        """High P(Adult) across all messages → Actor is Adult."""
        predictions = np.array([0.95, 0.98, 0.92, 0.97, 0.99] * 10)
        result = determine_actor_significance(predictions, "test", "predator", t=0.0, ast=0.05)
        assert result.determination == "A"

    def test_child_determination(self):
        """Low P(Adult) across all messages → Actor is Child."""
        predictions = np.array([0.02, 0.05, 0.01, 0.03, 0.04] * 10)
        result = determine_actor_significance(predictions, "test", "officer", t=0.0, ast=0.05)
        assert result.determination == "C"

    def test_not_significant_mixed(self):
        """Mixed predictions → Not Significant."""
        predictions = np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.5, 0.5])
        result = determine_actor_significance(predictions, "test", "predator", t=0.0, ast=0.05)
        # Mixed equally between Adult and Child → should be NS
        assert result.determination == "NS"

    def test_empty_after_omission(self):
        """If all messages omitted, should be NS."""
        # All in (0.3, 0.7) with t=0.2
        predictions = np.array([0.4, 0.45, 0.5, 0.55, 0.6])
        result = determine_actor_significance(predictions, "test", "predator", t=0.2, ast=0.05)
        assert result.determination == "NS"
        assert result.n_messages == 0

    def test_stricter_ast_more_ns(self):
        """Lower AST (stricter) → fewer significant determinations."""
        predictions = np.array([0.9] * 5 + [0.1] * 3)  # Marginally significant
        result_loose = determine_actor_significance(predictions, "t", "p", t=0.0, ast=0.05)
        result_strict = determine_actor_significance(predictions, "t", "p", t=0.0, ast=0.001)
        # Strict may result in NS where loose gives A/C
        # At minimum, strict should not be MORE specific than loose
        if result_strict.determination != "NS":
            assert result_loose.determination == result_strict.determination


class TestTranscriptContext:
    def _make_det(self, determination: str, conv_id: str = "test", role: str = "predator"):
        return ActorDetermination(
            actor_id=conv_id,
            role=role,
            n_messages=20,
            n_omitted=0,
            mean_prediction=0.9 if determination == "A" else 0.1,
            p_value=0.001,
            determination=determination,
        )

    def test_ac_context_detected(self):
        """One Adult + one Child → AC context."""
        pred_det = self._make_det("A", role="predator")
        off_det = self._make_det("C", role="officer")
        ctx = determine_transcript_context(pred_det, off_det, "conv1")
        assert ctx.is_ac == True
        assert ctx.context == "AC"

    def test_ca_context_also_ac(self):
        """Child predator + Adult officer → still AC (just reversed roles)."""
        pred_det = self._make_det("C", role="predator")
        off_det = self._make_det("A", role="officer")
        ctx = determine_transcript_context(pred_det, off_det, "conv1")
        assert ctx.is_ac == True

    def test_both_adult_not_ac(self):
        """Two Adults → not AC."""
        pred_det = self._make_det("A", role="predator")
        off_det = self._make_det("A", role="officer")
        ctx = determine_transcript_context(pred_det, off_det, "conv1")
        assert ctx.is_ac == False

    def test_both_child_not_ac(self):
        """Two Children → not AC."""
        pred_det = self._make_det("C", role="predator")
        off_det = self._make_det("C", role="officer")
        ctx = determine_transcript_context(pred_det, off_det, "conv1")
        assert ctx.is_ac == False

    def test_ns_in_either_position_not_ac(self):
        """NS in either actor → not AC."""
        pred_det = self._make_det("NS", role="predator")
        off_det = self._make_det("C", role="officer")
        ctx = determine_transcript_context(pred_det, off_det, "conv1")
        assert ctx.is_ac == False

    def test_both_ns_not_ac(self):
        """Both NS → not AC."""
        pred_det = self._make_det("NS", role="predator")
        off_det = self._make_det("NS", role="officer")
        ctx = determine_transcript_context(pred_det, off_det, "conv1")
        assert ctx.is_ac == False
