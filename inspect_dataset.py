"""
Dataset Inspection Script for PervertedJustice Dataset

This script analyzes the structure, content, and pairing of predator/officer files
from the PervertedJustice dataset to determine what aspects of the research paper
can be reproduced with this cleaned, speaker-separated data.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
import chardet

# Dataset paths
DATASET_ROOT = Path(__file__).parent.parent / "PervertedJusticeDataset"
PRED_DIR = DATASET_ROOT / "predators"
OFF_DIR = DATASET_ROOT / "officers"
META_FILE = DATASET_ROOT / "meta.tsv"
URL_FILE = DATASET_ROOT / "url_name.tsv"
RESULTS_DIR = Path(__file__).parent / "results"

def main():
    print("=" * 80)
    print("PERVERTED JUSTICE DATASET INSPECTION")
    print("=" * 80)
    print()
    
    if not DATASET_ROOT.exists():
        print(f"ERROR: Dataset not found at {DATASET_ROOT}")
        print("Please clone the repository first:")
        print("  git clone https://github.com/helenkoutli/PervertedJusticeDataset.git")
        sys.exit(1)
    
    results = []
    results.append("=" * 80)
    results.append("PERVERTED JUSTICE DATASET INSPECTION REPORT")
    results.append("=" * 80)
    results.append("")
    
    # 1. File counts
    pred_files = sorted(PRED_DIR.glob("*.txt"))
    off_files = sorted(OFF_DIR.glob("*.txt"))
    
    results.append(f"## 1. FILE COUNTS")
    results.append(f"Predator files: {len(pred_files)}")
    results.append(f"Officer files:  {len(off_files)}")
    results.append("")
    
    # 2. File naming patterns
    results.append("## 2. FILE NAMING PATTERNS")
    pred_names = [f.stem.replace("predator_", "") for f in pred_files]
    off_names = [f.stem.replace("officer_", "") for f in off_files]
    
    results.append(f"Predator prefix pattern: 'predator_<username>.txt'")
    results.append(f"Officer prefix pattern:  'officer_<username>.txt'")
    results.append(f"Sample predator names: {pred_names[:5]}")
    results.append(f"Sample officer names:  {off_names[:5]}")
    results.append("")
    
    # 3. Pairing validation
    results.append("## 3. FILE PAIRING ANALYSIS")
    pred_set = set(pred_names)
    off_set = set(off_names)
    
    # Files with matching pairs
    paired = pred_set & off_set
    pred_only = pred_set - off_set
    off_only = off_set - pred_set
    
    results.append(f"Conversations with both predator and officer files: {len(paired)}")
    results.append(f"Predator files without matching officer: {len(pred_only)}")
    results.append(f"Officer files without matching predator: {len(off_only)}")
    
    if pred_only:
        results.append(f"  Predator-only files (first 10): {sorted(pred_only)[:10]}")
    if off_only:
        results.append(f"  Officer-only files (first 10): {sorted(off_only)[:10]}")
    results.append("")
    
    # 4. Text encoding and format
    results.append("## 4. TEXT ENCODING AND FILE FORMAT")
    
    sample_pred = pred_files[0]
    sample_off = off_files[0]
    
    with open(sample_pred, 'rb') as f:
        pred_encoding = chardet.detect(f.read())
    with open(sample_off, 'rb') as f:
        off_encoding = chardet.detect(f.read())
    
    results.append(f"Sample predator file encoding: {pred_encoding['encoding']} (confidence: {pred_encoding['confidence']:.2f})")
    results.append(f"Sample officer file encoding:  {off_encoding['encoding']} (confidence: {off_encoding['confidence']:.2f})")
    results.append("")
    
    # 5. Utterance counts
    results.append("## 5. UTTERANCE COUNTS PER FILE")
    
    pred_counts = []
    off_counts = []
    empty_pred = []
    empty_off = []
    
    for pf in pred_files:
        with open(pf, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.strip() for line in f if line.strip()]
            pred_counts.append(len(lines))
            if len(lines) == 0:
                empty_pred.append(pf.name)
    
    for of in off_files:
        with open(of, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.strip() for line in f if line.strip()]
            off_counts.append(len(lines))
            if len(lines) == 0:
                empty_off.append(of.name)
    
    results.append(f"Predator utterances:")
    results.append(f"  Min: {min(pred_counts)}, Max: {max(pred_counts)}, Mean: {sum(pred_counts) / len(pred_counts):.1f}, Total: {sum(pred_counts)}")
    results.append(f"Officer utterances:")
    results.append(f"  Min: {min(off_counts)}, Max: {max(off_counts)}, Mean: {sum(off_counts) / len(off_counts):.1f}, Total: {sum(off_counts)}")
    results.append("")
    
    # 6. Empty files
    results.append("## 6. EMPTY FILES")
    results.append(f"Empty predator files: {len(empty_pred)}")
    if empty_pred:
        results.append(f"  {empty_pred}")
    results.append(f"Empty officer files: {len(empty_off)}")
    if empty_off:
        results.append(f"  {empty_off}")
    results.append("")
    
    # 7. Metadata inspection
    results.append("## 7. METADATA FILES")
    
    if META_FILE.exists():
        with open(META_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            meta_lines = f.readlines()
        results.append(f"meta.tsv: {len(meta_lines)} lines (including header)")
        results.append(f"  Columns: {meta_lines[0].strip()}")
    else:
        results.append("meta.tsv: NOT FOUND")
    
    if URL_FILE.exists():
        with open(URL_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            url_lines = f.readlines()
        results.append(f"url_name.tsv: {len(url_lines)} lines")
    else:
        results.append("url_name.tsv: NOT FOUND")
    results.append("")
    
    # 8. Conversation identifier
    results.append("## 8. CONVERSATION IDENTIFIER")
    results.append("The username (extracted from filename after 'predator_' or 'officer_')")
    results.append("serves as the conversation ID. This maps predator/officer file pairs.")
    results.append("")
    results.append("NOTE: According to the README, 12 conversations have multiple officers")
    results.append("and 2 predator files were merged. This explains the file count mismatch.")
    results.append("")
    
    # 9. Message order preservation
    results.append("## 9. MESSAGE ORDER AND CHRONOLOGICAL CONTEXT")
    results.append("✗ MISSING: Timestamps were removed during cleaning")
    results.append("✗ MISSING: Original message order (chronological interleaving) is lost")
    results.append("✗ MISSING: Cannot reconstruct turn-by-turn conversation flow")
    results.append("")
    results.append("Each file contains all utterances from ONE speaker, but:")
    results.append("- We cannot determine which predator message came before/after which officer message")
    results.append("- We cannot measure response times or detect grooming phase progression")
    results.append("- Message sequence within each file MAY preserve that speaker's chronological order,")
    results.append("  but this is not documented and should not be assumed.")
    results.append("")
    
    # 10. Paper reproducibility assessment
    results.append("=" * 80)
    results.append("## 10. PAPER REPRODUCIBILITY ASSESSMENT")
    results.append("=" * 80)
    results.append("")
    results.append("### What CAN be reproduced:")
    results.append("")
    results.append("✓ Message-Level Analysis (MLA) for Adult vs. Child classification")
    results.append("  - Each line is an utterance from a known speaker (predator=Adult, officer=Child)")
    results.append("  - Train SVM, NB, BERT, RoBERTa to classify individual messages")
    results.append("  - Metrics: TP, IA (Incorrect Adult), IC (Incorrect Child), O (Omissions)")
    results.append("  - Message Threshold (t) omission: Eq. (1) from paper applies")
    results.append("")
    results.append("✓ Context Determination per conversation")
    results.append("  - Aggregate per-actor message predictions using binomial test")
    results.append("  - Actor Significance Threshold (AST) p-value test")
    results.append("  - Transcript-level Adult-Child (AC) context determination")
    results.append("  - Metrics: TP, FP, TN, FN, F1 for detecting AC interactions")
    results.append("")
    results.append("✓ Group-based train/test splits (80/20 by conversation, not message)")
    results.append("")
    results.append("### What CANNOT be reproduced exactly:")
    results.append("")
    results.append("✗ Phase-based analysis (O'Connell 2003 grooming stages)")
    results.append("  - Requires chronological message order, which is not preserved")
    results.append("")
    results.append("✗ Turn-level or sequential context features")
    results.append("  - Cannot measure response times, turn-taking patterns, or topic shifts")
    results.append("")
    results.append("✗ Early detection experiments")
    results.append("  - Requires progressive message disclosure in original order")
    results.append("")
    results.append("### Important clarifications:")
    results.append("")
    results.append("• Speaker role ≠ Message-level grooming label")
    results.append("  - Predator utterances are from adults, but not every message is 'grooming'")
    results.append("  - Officer utterances are from honeypot actors, not real children")
    results.append("  - MLA classifies Adult vs. Child language, NOT grooming vs. non-grooming")
    results.append("")
    results.append("• Context Determination determines if conversation involves Adult+Child")
    results.append("  - All PJ transcripts are OG-positive by design (no true negatives)")
    results.append("  - Cross-dataset evaluation (PAN12) provides negative examples")
    results.append("")
    results.append("• No ground-truth per-message grooming labels exist in this dataset")
    results.append("")
    
    # Duplicate detection
    results.append("## 11. DUPLICATE DETECTION")
    
    # Check for exact duplicate content
    pred_hashes = defaultdict(list)
    off_hashes = defaultdict(list)
    
    for pf in pred_files[:100]:  # Sample to avoid long runtime
        with open(pf, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            pred_hashes[hash(content)].append(pf.name)
    
    for of in off_files[:100]:
        with open(of, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            off_hashes[hash(content)].append(of.name)
    
    pred_dupes = {k: v for k, v in pred_hashes.items() if len(v) > 1}
    off_dupes = {k: v for k, v in off_hashes.items() if len(v) > 1}
    
    results.append(f"Duplicate predator files (sampled 100): {len(pred_dupes)}")
    results.append(f"Duplicate officer files (sampled 100): {len(off_dupes)}")
    results.append("")
    
    # Write report
    RESULTS_DIR.mkdir(exist_ok=True)
    report_path = RESULTS_DIR / "dataset_inspection_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(results))
    
    # Print to console
    print("\n".join(results))
    print()
    print(f"Report saved to: {report_path}")
    print()

if __name__ == "__main__":
    main()
