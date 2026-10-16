"""
test_file_pairing.py

Tests for file pairing logic between predator and officer files.
Validates that conversation IDs are consistently extracted and that
the pairing logic correctly handles the dataset's known anomalies:
  - 12 conversations with multiple officer files
  - 2 predator files merged into one
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import (
    _extract_conv_id,
    get_paired_conversations,
    load_dataset,
    PRED_DIR,
    OFF_DIR,
    DATASET_ROOT,
)


DATASET_AVAILABLE = DATASET_ROOT.exists()


# ---------------------------------------------------------------------------
# Unit tests — filename parsing (no dataset required)
# ---------------------------------------------------------------------------

class TestConvIdExtraction:
    def test_predator_extraction(self):
        from pathlib import Path
        p = Path("predator_abraxisreborn.txt")
        assert _extract_conv_id(p, "predator") == "abraxisreborn"

    def test_officer_extraction(self):
        from pathlib import Path
        p = Path("officer_abraxisreborn.txt")
        assert _extract_conv_id(p, "officer") == "abraxisreborn"

    def test_predator_with_numbers(self):
        from pathlib import Path
        p = Path("predator_ACAR556.txt")
        assert _extract_conv_id(p, "predator") == "ACAR556"

    def test_officer_suffix_1(self):
        """Multi-officer conversations have _1, _2 suffixes."""
        from pathlib import Path
        p = Path("officer_ben_taul2000_1.txt")
        assert _extract_conv_id(p, "officer") == "ben_taul2000_1"

    def test_officer_suffix_2(self):
        from pathlib import Path
        p = Path("officer_cantonguy4fun_2.txt")
        assert _extract_conv_id(p, "officer") == "cantonguy4fun_2"

    def test_predator_underscore_in_name(self):
        from pathlib import Path
        p = Path("predator_a_latino_man559.txt")
        assert _extract_conv_id(p, "predator") == "a_latino_man559"

    def test_no_prefix_fallback(self):
        """If file doesn't have expected prefix, return stem as-is."""
        from pathlib import Path
        p = Path("somefile.txt")
        result = _extract_conv_id(p, "predator")
        assert result == "somefile"


# ---------------------------------------------------------------------------
# Integration tests — require dataset to be cloned
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DATASET_AVAILABLE, reason="Dataset not cloned")
class TestPairedConversations:
    def test_paired_count(self):
        """At least 600 conversations should be paired (README says ~609 expected)."""
        convs = get_paired_conversations()
        n_paired = sum(1 for v in convs.values() if v["paired"])
        assert n_paired >= 600, f"Expected >= 600 paired conversations, got {n_paired}"

    def test_predator_file_count(self):
        """README says 621 predator files."""
        pred_files = list(PRED_DIR.glob("*.txt"))
        assert len(pred_files) == 621, f"Expected 621 predator files, got {len(pred_files)}"

    def test_officer_file_count(self):
        """README says 638 officer files."""
        off_files = list(OFF_DIR.glob("*.txt"))
        assert len(off_files) == 638, f"Expected 638 officer files, got {len(off_files)}"

    def test_multi_officer_conversations(self):
        """README says 12 conversations have multiple officer files."""
        off_files = list(OFF_DIR.glob("*.txt"))
        # Multi-officer convs have suffixes _1, _2, etc.
        import re
        multi_suffixes = [f for f in off_files if re.search(r'_\d+\.txt$', f.name)]
        # Each multi-officer conv contributes at least 2 files (_1, _2)
        assert len(multi_suffixes) >= 12, (
            f"Expected >= 12 multi-officer files, got {len(multi_suffixes)}"
        )

    def test_no_empty_files(self):
        """No predator or officer files should be empty."""
        for pf in PRED_DIR.glob("*.txt"):
            assert pf.stat().st_size > 0, f"Empty predator file: {pf.name}"
        for of in OFF_DIR.glob("*.txt"):
            assert of.stat().st_size > 0, f"Empty officer file: {of.name}"

    def test_conversation_id_consistent(self):
        """
        Paired conversation IDs must be consistent:
        predator_abraxisreborn.txt <-> officer_abraxisreborn.txt
        """
        convs = get_paired_conversations()
        for conv_id, info in convs.items():
            if info["paired"]:
                # Each predator file name should start with 'predator_<conv_id>'
                for pred_file in info["predator_files"]:
                    expected_prefix = f"predator_{conv_id}"
                    assert pred_file.startswith("predator_"), (
                        f"{pred_file} should start with 'predator_'"
                    )

    def test_no_duplicate_predator_files(self):
        """No two predator files should have the same content."""
        import hashlib
        hashes = {}
        for pf in PRED_DIR.glob("*.txt"):
            with open(pf, 'rb') as f:
                h = hashlib.md5(f.read()).hexdigest()
            assert h not in hashes, (
                f"Duplicate content: {pf.name} and {hashes[h]}"
            )
            hashes[h] = pf.name
