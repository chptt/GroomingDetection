"""
test_data_loader.py

Tests for the data_loader module.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import (
    _extract_conv_id,
    _read_utterances,
    load_dataset,
    summarise_dataset,
    DATASET_ROOT,
)

DATASET_AVAILABLE = DATASET_ROOT.exists()


class TestReadUtterances:
    def test_reads_lines(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello\nworld\nfoo\n")
        result = _read_utterances(f)
        assert result == ["hello", "world", "foo"]

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello\n\nworld\n\n")
        result = _read_utterances(f)
        assert result == ["hello", "world"]

    def test_nonexistent_file_returns_empty(self, tmp_path):
        result = _read_utterances(tmp_path / "nonexistent.txt")
        assert result == []

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        result = _read_utterances(f)
        assert result == []


class TestLoadDatasetUnit:
    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(dataset_root=tmp_path / "nonexistent")

    def test_loads_from_tmp_directory(self, tmp_path):
        """Test with a synthetic mini-dataset."""
        pred_dir = tmp_path / "predators"
        off_dir = tmp_path / "officers"
        pred_dir.mkdir()
        off_dir.mkdir()

        (pred_dir / "predator_testuser.txt").write_text("hello\ncome meet me\n")
        (off_dir / "officer_testuser.txt").write_text("who r u\nim 13\n")

        df = load_dataset(dataset_root=tmp_path)
        assert len(df) == 4  # 2 pred + 2 off utterances
        assert set(df["speaker_role"].unique()) == {0, 1}
        assert set(df["conversation_id"].unique()) == {"testuser"}

    def test_predator_gets_label_1(self, tmp_path):
        pred_dir = tmp_path / "predators"
        off_dir = tmp_path / "officers"
        pred_dir.mkdir()
        off_dir.mkdir()

        (pred_dir / "predator_user1.txt").write_text("hello\n")
        (off_dir / "officer_user1.txt").write_text("hi\n")

        df = load_dataset(dataset_root=tmp_path)
        pred_rows = df[df["file_name"] == "predator_user1.txt"]
        assert (pred_rows["speaker_role"] == 1).all()

    def test_officer_gets_label_0(self, tmp_path):
        pred_dir = tmp_path / "predators"
        off_dir = tmp_path / "officers"
        pred_dir.mkdir()
        off_dir.mkdir()

        (pred_dir / "predator_user1.txt").write_text("hello\n")
        (off_dir / "officer_user1.txt").write_text("hi\n")

        df = load_dataset(dataset_root=tmp_path)
        off_rows = df[df["file_name"] == "officer_user1.txt"]
        assert (off_rows["speaker_role"] == 0).all()

    def test_min_utterances_filter(self, tmp_path):
        pred_dir = tmp_path / "predators"
        off_dir = tmp_path / "officers"
        pred_dir.mkdir()
        off_dir.mkdir()

        # Conv A: 3 utterances total (should be excluded with min=10)
        (pred_dir / "predator_convA.txt").write_text("hi\n")
        (off_dir / "officer_convA.txt").write_text("hello\ngoodbye\n")

        # Conv B: 12 utterances total (should be kept)
        (pred_dir / "predator_convB.txt").write_text("\n".join([f"msg{i}" for i in range(6)]))
        (off_dir / "officer_convB.txt").write_text("\n".join([f"rep{i}" for i in range(6)]))

        df_full = load_dataset(dataset_root=tmp_path, min_utterances_per_conv=0)
        df_filtered = load_dataset(dataset_root=tmp_path, min_utterances_per_conv=10)

        assert "convA" in df_full["conversation_id"].values
        assert "convA" not in df_filtered["conversation_id"].values
        assert "convB" in df_filtered["conversation_id"].values

    def test_multi_officer_conversation(self, tmp_path):
        """Officer files with _1, _2 suffixes should load independently."""
        pred_dir = tmp_path / "predators"
        off_dir = tmp_path / "officers"
        pred_dir.mkdir()
        off_dir.mkdir()

        (pred_dir / "predator_predA.txt").write_text("hello\n")
        (off_dir / "officer_predA_1.txt").write_text("hi officer 1\n")
        (off_dir / "officer_predA_2.txt").write_text("hi officer 2\n")

        df = load_dataset(dataset_root=tmp_path)
        assert len(df) == 3  # 1 pred + 2 officer utterances
        off_rows = df[df["speaker_role"] == 0]
        assert len(off_rows) == 2


@pytest.mark.skipif(not DATASET_AVAILABLE, reason="Dataset not cloned")
class TestLoadDatasetIntegration:
    @pytest.fixture(scope="class")
    def df(self):
        return load_dataset()

    def test_loads_without_error(self, df):
        assert df is not None
        assert len(df) > 0

    def test_required_columns_present(self, df):
        for col in ["utterance", "speaker_role", "conversation_id", "file_name"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_null_values(self, df):
        assert df["utterance"].notna().all()
        assert df["speaker_role"].notna().all()
        assert df["conversation_id"].notna().all()

    def test_conversation_count(self, df):
        """Should have ~609+ unique conversations (paired ones)."""
        assert df["conversation_id"].nunique() >= 600

    def test_total_utterances(self, df):
        """Total should be ~1.37M utterances."""
        assert len(df) > 1_000_000

    def test_summarise_runs_without_error(self, df, capsys):
        summarise_dataset(df)
        captured = capsys.readouterr()
        assert "Total utterances" in captured.out
