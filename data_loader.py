"""
data_loader.py

Loads utterances from the PervertedJustice dataset into a structured DataFrame.

Key design decisions based on dataset inspection:
- Each predator file = one speaker's utterances (Adult / speaker_role=1)
- Each officer file  = one speaker's utterances (Child  / speaker_role=0)
- Conversation ID = username extracted from filename
- Files are paired by conversation ID (username)
- 12 conversations have multiple officer files (same conversation_id, different file)
- NO per-utterance random splits; always group by conversation_id
- NOT every predator utterance is a grooming message; label is speaker-role only
"""

from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

DATASET_ROOT = Path(__file__).parent.parent / "PervertedJusticeDataset"
PRED_DIR = DATASET_ROOT / "predators"
OFF_DIR = DATASET_ROOT / "officers"


def _extract_conv_id(filepath: Path, role: str) -> str:
    """Extract conversation ID (username) from filename.
    
    predator_abraxisreborn.txt -> abraxisreborn
    officer_abraxisreborn.txt  -> abraxisreborn
    """
    stem = filepath.stem  # e.g. 'predator_abraxisreborn'
    prefix = f"{role}_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


def _read_utterances(filepath: Path) -> list[str]:
    """Read non-empty lines from a file as individual utterances.
    
    Important: timestamps and usernames were removed during cleaning.
    Each line is one utterance. Order within the file may be
    chronological for that speaker only — no cross-speaker ordering.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines
    except Exception as e:
        logger.warning(f"Could not read {filepath}: {e}")
        return []


def load_dataset(dataset_root: Path | str | None = None,
                 min_utterances_per_conv: int = 0) -> pd.DataFrame:
    """
    Load all predator and officer utterances into a DataFrame.

    Args:
        dataset_root: Path to the PervertedJusticeDataset directory.
                      Defaults to ../PervertedJusticeDataset relative to this file.
        min_utterances_per_conv: Minimum total utterances (predator+officer) for a
                                 conversation to be included. The paper excludes
                                 transcripts shorter than 10 messages for PAN12;
                                 for PJ no explicit minimum is mentioned.

    Returns:
        DataFrame with columns:
            - utterance       : str, the message text
            - speaker_role    : int, 1=predator/Adult, 0=officer/Child
            - conversation_id : str, username key linking predator+officer files
            - file_name       : str, source filename (for traceability)

    IMPORTANT:
        speaker_role is a SPEAKER IDENTITY label, not a grooming-message label.
        Predator=Adult=1, Officer=Child=0 (honeypot actor, not a real child).
        Do not interpret speaker_role=1 as "this message is grooming content".
    """
    if dataset_root is not None:
        pred_dir = Path(dataset_root) / "predators"
        off_dir = Path(dataset_root) / "officers"
    else:
        pred_dir = PRED_DIR
        off_dir = OFF_DIR

    if not pred_dir.exists() or not off_dir.exists():
        raise FileNotFoundError(
            f"Dataset directories not found.\n"
            f"Expected:\n  {pred_dir}\n  {off_dir}\n"
            f"Clone the dataset first:\n"
            f"  git clone https://github.com/helenkoutli/PervertedJusticeDataset.git"
        )

    records = []

    # Load predator utterances (speaker_role = 1 = Adult)
    for pf in sorted(pred_dir.glob("*.txt")):
        conv_id = _extract_conv_id(pf, "predator")
        utterances = _read_utterances(pf)
        for utt in utterances:
            records.append({
                "utterance": utt,
                "speaker_role": 1,
                "conversation_id": conv_id,
                "file_name": pf.name,
            })

    # Load officer utterances (speaker_role = 0 = Child)
    for of in sorted(off_dir.glob("*.txt")):
        conv_id = _extract_conv_id(of, "officer")
        utterances = _read_utterances(of)
        for utt in utterances:
            records.append({
                "utterance": utt,
                "speaker_role": 0,
                "conversation_id": conv_id,
                "file_name": of.name,
            })

    df = pd.DataFrame(records)

    if df.empty:
        logger.warning("No utterances loaded. Check dataset path.")
        return df

    # Optionally filter short conversations
    if min_utterances_per_conv > 0:
        conv_sizes = df.groupby("conversation_id").size()
        valid_convs = conv_sizes[conv_sizes >= min_utterances_per_conv].index
        before = len(df)
        df = df[df["conversation_id"].isin(valid_convs)].reset_index(drop=True)
        logger.info(f"Filtered {before - len(df)} utterances from short conversations")

    logger.info(
        f"Loaded {len(df)} utterances from {df['conversation_id'].nunique()} conversations "
        f"({df['speaker_role'].sum()} predator, {(df['speaker_role']==0).sum()} officer)"
    )

    return df


def get_conversation_ids(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique conversation IDs."""
    return sorted(df["conversation_id"].unique().tolist())


def get_paired_conversations(dataset_root: Path | str | None = None) -> dict[str, dict]:
    """
    Return a dict mapping conversation_id -> {predator_files, officer_files,
    pred_utterances, off_utterances}.

    NOTE: 12 conversations have multiple officer files. This function preserves
    that structure rather than collapsing it.
    """
    if dataset_root is not None:
        pred_dir = Path(dataset_root) / "predators"
        off_dir = Path(dataset_root) / "officers"
    else:
        pred_dir = PRED_DIR
        off_dir = OFF_DIR

    from collections import defaultdict

    convs: dict[str, dict] = defaultdict(lambda: {
        "predator_files": [],
        "officer_files": [],
        "pred_utterances": 0,
        "off_utterances": 0,
        "paired": False,
    })

    for pf in sorted(pred_dir.glob("*.txt")):
        conv_id = _extract_conv_id(pf, "predator")
        utts = _read_utterances(pf)
        convs[conv_id]["predator_files"].append(pf.name)
        convs[conv_id]["pred_utterances"] += len(utts)

    for of in sorted(off_dir.glob("*.txt")):
        conv_id = _extract_conv_id(of, "officer")
        utts = _read_utterances(of)
        convs[conv_id]["officer_files"].append(of.name)
        convs[conv_id]["off_utterances"] += len(utts)

    for conv_id, info in convs.items():
        info["paired"] = (
            len(info["predator_files"]) > 0 and len(info["officer_files"]) > 0
        )

    return dict(convs)


def summarise_dataset(df: pd.DataFrame) -> None:
    """Print a concise dataset summary."""
    print("=== Dataset Summary ===")
    print(f"Total utterances    : {len(df)}")
    print(f"Predator utterances : {(df['speaker_role'] == 1).sum()}")
    print(f"Officer utterances  : {(df['speaker_role'] == 0).sum()}")
    print(f"Unique conversations: {df['conversation_id'].nunique()}")
    print(f"Unique files        : {df['file_name'].nunique()}")

    sizes = df.groupby("conversation_id").size()
    print(f"Utterances/conv — min: {sizes.min()}, max: {sizes.max()}, mean: {sizes.mean():.1f}")
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_dataset()
    summarise_dataset(df)
    print(df.head(10))
