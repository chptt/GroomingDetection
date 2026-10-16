"""
test_role_labels.py

Tests that speaker-role labels are correctly assigned from file structure.

CRITICAL: Labels must come from file structure (which folder the file is in),
NOT from message content. The following must hold:
  - All utterances from predators/ → speaker_role = 1 (Adult)
  - All utterances from officers/  → speaker_role = 0 (Child)

These are SPEAKER IDENTITY labels, not grooming-message labels.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import load_dataset, DATASET_ROOT

DATASET_AVAILABLE = DATASET_ROOT.exists()


@pytest.mark.skipif(not DATASET_AVAILABLE, reason="Dataset not cloned")
class TestSpeakerRoleLabels:

    @pytest.fixture(scope="class")
    def df(self):
        return load_dataset()

    def test_only_binary_labels(self, df):
        """Labels must be exactly 0 or 1."""
        unique_labels = set(df["speaker_role"].unique())
        assert unique_labels == {0, 1}, (
            f"Expected {{0, 1}}, got {unique_labels}"
        )

    def test_predator_files_get_label_1(self, df):
        """All utterances from predator_*.txt files must have speaker_role=1."""
        pred_rows = df[df["file_name"].str.startswith("predator_")]
        assert len(pred_rows) > 0, "No predator rows found"
        wrong = pred_rows[pred_rows["speaker_role"] != 1]
        assert len(wrong) == 0, (
            f"{len(wrong)} predator rows have wrong label: {wrong['file_name'].unique()[:5]}"
        )

    def test_officer_files_get_label_0(self, df):
        """All utterances from officer_*.txt files must have speaker_role=0."""
        off_rows = df[df["file_name"].str.startswith("officer_")]
        assert len(off_rows) > 0, "No officer rows found"
        wrong = off_rows[off_rows["speaker_role"] != 0]
        assert len(wrong) == 0, (
            f"{len(wrong)} officer rows have wrong label: {wrong['file_name'].unique()[:5]}"
        )

    def test_label_source_is_file_structure_not_content(self, df):
        """
        Verify labels are file-structure-based.
        A predator file with peaceful content still gets label=1.
        An officer file with aggressive content still gets label=0.
        """
        # Sample 100 predator rows and verify all are label=1
        pred_sample = df[df["speaker_role"] == 1].sample(min(100, len(df[df["speaker_role"]==1])), random_state=42)
        assert (pred_sample["file_name"].str.startswith("predator_")).all()

        # Sample 100 officer rows and verify all are label=0
        off_sample = df[df["speaker_role"] == 0].sample(min(100, len(df[df["speaker_role"]==0])), random_state=42)
        assert (off_sample["file_name"].str.startswith("officer_")).all()

    def test_both_roles_present(self, df):
        """Dataset must have both predator and officer utterances."""
        n_pred = (df["speaker_role"] == 1).sum()
        n_off = (df["speaker_role"] == 0).sum()
        assert n_pred > 0, "No predator utterances found"
        assert n_off > 0, "No officer utterances found"
        # Rough balance check — neither should be less than 30% of total
        total = len(df)
        assert n_pred / total > 0.30, f"Too few predator utterances: {n_pred/total:.1%}"
        assert n_off / total > 0.30, f"Too few officer utterances: {n_off/total:.1%}"

    def test_conversation_id_present(self, df):
        """Every row must have a non-empty conversation_id."""
        assert df["conversation_id"].notna().all()
        assert (df["conversation_id"].str.len() > 0).all()

    def test_utterance_not_empty(self, df):
        """No utterance should be an empty string after loading."""
        assert (df["utterance"].str.len() > 0).all(), "Empty utterances found in loaded dataset"

    def test_total_utterance_count(self, df):
        """Total utterances should be in reasonable range for this dataset."""
        # Based on inspection: ~706k predator + ~663k officer = ~1.37M total
        assert len(df) > 1_000_000, f"Unexpectedly few utterances: {len(df)}"
        assert len(df) < 2_000_000, f"Unexpectedly many utterances: {len(df)}"

    def test_label_docstring_warning_not_violated(self, df):
        """
        Programmatic check: speaker_role column description in DataFrame
        should not be confused with a grooming label.
        This test documents the distinction and ensures the column name
        'speaker_role' is used (not 'is_grooming' or 'label_grooming').
        """
        assert "speaker_role" in df.columns, "Column must be named 'speaker_role'"
        assert "is_grooming" not in df.columns, (
            "'is_grooming' column should not exist — all PJ transcripts are OG-positive"
        )
