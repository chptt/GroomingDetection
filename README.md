# Online Grooming Detection — Implementation of Street et al. (2025)

Implementation of:

> Street, J., Ihianle, I.K., Olajide, F., & Lotfi, A. (2025).  
> **Enhanced Online Grooming Detection Employing Context Determination and Message-Level Analysis.**  
> *Intelligent Systems with Applications*, 28, 200607.  
> https://doi.org/10.1016/j.iswa.2025.200607

Using the PervertedJustice dataset:
> Koutli, E. (2023). PervertedJusticeDataset.  
> https://github.com/helenkoutli/PervertedJusticeDataset

---

## What This Implements

The paper introduces two complementary approaches:

1. **Message-Level Analysis (MLA):** Classify each utterance as Adult (1) or Child (0) language using SVM, Naive Bayes, BERT, and RoBERTa. Applies a Message Threshold *t* to omit low-confidence predictions.

2. **Context Determination:** Aggregate per-actor predictions via a binomial test with an Actor Significance Threshold (AST) to determine if a conversation involves an Adult-Child dyad (AC context).

**Critical distinction:** These methods classify *speaker roles* and *conversation context*, not individual grooming messages. All PJ conversations are OG-positive; there are no negative examples in this dataset.

---

## Dataset Constraints

The PervertedJustice cleaned dataset:
- Contains **621 predator files** and **638 officer files** (~609 paired conversations)
- Has **timestamps and usernames removed** — chronological ordering is lost
- Is **speaker-separated**: each file contains one actor's utterances only
- Is **all-positive**: every conversation is a confirmed grooming interaction

This means:
- ✓ Message-level Adult/Child classification can be reproduced
- ✓ Actor significance + context determination can be reproduced  
- ✗ Grooming phase analysis (requires chronological order) cannot be reproduced
- ✗ Cross-dataset evaluation (requires PAN12) is not implemented here
- ✗ False positive rate (requires negative examples) cannot be computed from PJ alone

---

## Project Structure

```
grooming_detection/
├── inspect_dataset.py        # Dataset statistics and inspection
├── data_loader.py            # Load utterances into DataFrame
├── preprocessing.py          # Minimal cleaning + group-based splitting
├── message_classifier.py     # SVM, NB, BERT, RoBERTa classifiers + MLA metrics
├── context_determination.py  # Binomial test + Actor Significance + AC context
├── actor_classification.py   # Pipeline: predictions → context sweep over AST/t
├── evaluation.py             # Metrics, plots, paper comparison
├── run_experiments.py        # Main orchestration script
├── requirements.txt
├── README.md
├── tests/
│   ├── test_file_pairing.py   # File pairing and conversation ID tests
│   ├── test_role_labels.py    # Speaker-role label correctness tests
│   ├── test_preprocessing.py  # Cleaning, vectorizer, split tests
│   ├── test_data_loader.py    # Loader unit + integration tests
│   └── test_splitting.py     # Group split, message threshold, context logic tests
└── results/
    ├── dataset_inspection_report.txt
    ├── paper_dataset_mapping.md
    ├── mla_svm.json / mla_nb.json / ...
    ├── context_svm.csv / context_nb.csv / ...
    ├── confusion_matrix_svm.png / ...
    └── experiment_summary.json
```

The original dataset stays at `../PervertedJusticeDataset/` — never modified.

---

## Installation

```bash
# 1. Clone the dataset (already done if you see PervertedJusticeDataset/)
git clone https://github.com/helenkoutli/PervertedJusticeDataset.git

# 2. Install dependencies
pip install -r grooming_detection/requirements.txt
```

Python 3.10+ required (uses `list[str]` type hints).

---

## Running the Experiments

### Step 1 — Inspect the dataset
```bash
python grooming_detection/inspect_dataset.py
```
Produces `results/dataset_inspection_report.txt`.

### Step 2 — Run SVM + NB baselines (fast, ~2 minutes on full dataset)
```bash
python grooming_detection/run_experiments.py --skip-transformers
```

### Step 3 — Quick verification (50 conversations, ~30 seconds)
```bash
python grooming_detection/run_experiments.py --skip-transformers --fast
```

### Step 4 — Full run with BERT and RoBERTa (requires GPU, ~hours)
```bash
python grooming_detection/run_experiments.py
```

### Step 5 — Run tests
```bash
python -m pytest grooming_detection/tests/ -v
```
Expected: **88 passed** (all unit + integration tests).

---

## Experimental Results (Fast Mode — 50 conversations)

These are indicative results on a 50-conversation sample. Full dataset results will differ.

### Message-Level Analysis (Speaker-Role Classification)

| Model | TP% | IA% | IC% | O% |
|---|---|---|---|---|
| SVM | 61.8 | 28.7 | 9.5 | 0.0 |
| NB  | 57.0 | 35.2 | 7.8 | 0.0 |

**Paper reference (full PJ dataset, Table 2):** SVM 76.0%, NB 74.7%  
The gap is expected: the paper trains on ~219k messages; fast mode uses ~85k from 40 conversations.

**These are speaker-role classification metrics — NOT grooming detection accuracy.**

### Context Determination (AC Context Detection on PJ)

All PJ conversations are OG-positive, so FP=0 and TN=0 by dataset design.

| Model | AST | TP | FN | F1 |
|---|---|---|---|---|
| SVM | 0.05 | 4 | 3 | 0.727 |
| NB  | 0.05 | 5 | 2 | 0.833 |

**Paper reference (full PJ, Table 4, AST=0.05):** SVM F1=0.871 (118 TP / 35 FN), NB F1=0.862  
Fast mode uses only 10 test conversations — counts are small but the logic is correct.

---

## Paper Comparison Notes

| Paper Result | Status |
|---|---|
| SVM MLA TP=76.0% on PJ | Approximately reproducible (lower on fast mode due to smaller training set) |
| RoBERTa MLA TP=88.8% (t=0.45) on PJ | Requires BERT/RoBERTa training (GPU recommended) |
| Context Det. F1=0.973 (RoBERTa, AST=0.05) on PJ | Requires full dataset + transformer training |
| Cross-dataset F1=0.960 on PAN12 | Not implemented — PAN12 not in this repository |

Differences from paper are expected because:
1. The paper's PJ dataset was parsed directly from HTML (perverted-justice.com); this repo is a pre-cleaned version
2. The paper used 622 conversations over a different date range
3. This implementation uses `sklearn` + HuggingFace; the paper used TensorFlow throughout

---

## Ethical Note

This project works with transcripts of real online grooming interactions between adult investigators and convicted predators. The data is handled in accordance with the Perverted Justice foundation's published records and is used solely for research purposes.

- Do not republish transcript content
- Do not attempt to identify individuals from the data
- Results from this system should never be used as standalone evidence in legal proceedings

---

## Module Reference

| Module | Purpose |
|---|---|
| `data_loader.py` | Load utterances; extract `speaker_role` from file structure |
| `preprocessing.py` | Minimal cleaning (whitespace only); group-based 80/20 split |
| `message_classifier.py` | SVM, NB, BERT, RoBERTa + MLA metrics (TP/IA/IC/O) + message threshold Eq.(1) |
| `context_determination.py` | Binomial test → Actor A/C/NS; AC context logic from Fig.3 |
| `actor_classification.py` | Sweep AST × t values; attach proba → context pipeline |
| `evaluation.py` | Confusion matrix, classification report, paper comparison display |
| `run_experiments.py` | Full pipeline orchestration |
| `inspect_dataset.py` | Standalone dataset statistics |
