# Project Report: Online Grooming Detection System

**Implementation of:**  
Street, J., Ihianle, I.K., Olajide, F., & Lotfi, A. (2025).  
*Enhanced Online Grooming Detection Employing Context Determination and Message-Level Analysis.*  
Intelligent Systems with Applications, 28, 200607.  
https://doi.org/10.1016/j.iswa.2025.200607

**Dataset:**  
Koutli, E. (2023). PervertedJusticeDataset.  
https://github.com/helenkoutli/PervertedJusticeDataset

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Acquisition and Inspection](#2-dataset-acquisition-and-inspection)
3. [Paper Methodology Summary](#3-paper-methodology-summary)
4. [Paper-to-Dataset Reproducibility Mapping](#4-paper-to-dataset-reproducibility-mapping)
5. [Implementation Architecture](#5-implementation-architecture)
6. [Module Descriptions](#6-module-descriptions)
7. [Experimental Results](#7-experimental-results)
8. [Web UI](#8-web-ui)
9. [Test Suite](#9-test-suite)
10. [Known Limitations](#10-known-limitations)
11. [Reproduction Instructions](#11-reproduction-instructions)

---

## 1. Project Overview

This project implements the grooming detection system described in Street et al. (2025) using the publicly available PervertedJustice (PJ) dataset. The system performs:

1. **Message-Level Analysis (MLA):** Classify each chat utterance as Adult (1) or Child (0) language using four classifiers — SVM, Naive Bayes, BERT, and RoBERTa.
2. **Context Determination:** Aggregate per-actor predictions via a binomial significance test to determine whether a full conversation involves an Adult–Child (AC) dyad — a key signal for Online Grooming (OG).

The implementation is faithful to the paper wherever the dataset allows, and all deviations from the paper's methodology are explicitly documented with justification.

### Key Clarifications (enforced throughout)

| Common misconception | Correct interpretation |
|---|---|
| "Predator utterances are grooming messages" | Predator utterances carry the label *Adult* based on file structure, not message content |
| "TP% = grooming detection accuracy" | TP% measures speaker-role classification accuracy (Adult vs. Child language) |
| "We can detect grooming in new chats" | This system classifies language style; it does not flag individual sentences as grooming |
| "FP=0 means perfect classification" | FP=0 is a dataset property — all PJ conversations are OG-positive, so no false positive can arise |

---

## 2. Dataset Acquisition and Inspection

### 2.1 Repository

The dataset was cloned directly from GitHub:

```
git clone https://github.com/helenkoutli/PervertedJusticeDataset.git
```

### 2.2 Structure

```
PervertedJusticeDataset/
├── predators/    # 621 .txt files, one per predator username
├── officers/     # 638 .txt files, one per officer username
├── meta.tsv      # 623 rows: name, date, place, comment, URLs
└── url_name.tsv  # 1244 rows: URL ↔ HTML filename mapping
```

### 2.3 Inspection Findings

| Property | Value |
|---|---|
| Predator files | 621 |
| Officer files | 638 |
| Paired conversations (both files exist) | 609 |
| Predator-only (no matching officer) | 12 |
| Officer-only (multi-officer convs, _1/_2 suffixes) | 29 |
| Total predator utterances | 706,214 |
| Total officer utterances | 662,925 |
| Grand total utterances | 1,369,139 |
| Utterances per file — min / max / mean | 8 / 27,543 / 2,106 |
| Text encoding | ASCII |
| Empty files | 0 |
| Duplicate files | 0 |

### 2.4 File Naming Convention

- `predators/predator_<username>.txt` — all utterances by the predator
- `officers/officer_<username>.txt` — all utterances by the officer(s) in that conversation
- Conversations with multiple officers: `officer_<username>_1.txt`, `officer_<username>_2.txt`
- Conversation ID = `<username>` (extracted from filename)

### 2.5 Critical Dataset Constraints

- **Timestamps removed** during cleaning → original chronological interleaving is lost
- **Usernames removed** → actors cannot be re-identified
- **PJ comments/interventions removed** → clean utterance-only format
- **All conversations are OG-positive** → every transcript is a confirmed grooming interaction; no negative examples exist in this repository
- **Officers are adult volunteers** posing as children, not real minors

---

## 3. Paper Methodology Summary

### 3.1 Message-Level Analysis (MLA)

Each utterance is classified as Adult (1) or Child (0).

**Models:**
- SVM — `sklearn` Pipeline with `CountVectorizer` + `LinearSVC` (calibrated)
- Naive Bayes — `sklearn` `MultinomialNB` with `CountVectorizer`
- BERT — `bert-base-uncased` fine-tuned with TensorFlow
- RoBERTa — `roberta-base` fine-tuned with TensorFlow

**Message Threshold t (Equation 1):**  
Messages with prediction probability `n` satisfying `(0.5 + t) > n > (0.5 - t)` are *omitted* (low confidence excluded).  
Values tested: `t ∈ {0.2, 0.3, 0.4, 0.45}` (BERT/RoBERTa only; SVM/NB have no omissions)

**MLA Metrics:**
- **TP%** — correct Adult/Child determinations (% of kept messages)
- **IA%** — Incorrect Adult: predicted Adult, was Child
- **IC%** — Incorrect Child: predicted Child, was Adult
- **O%** — Omissions (% of ALL messages excluded by threshold t)

### 3.2 Context Determination

For each conversation, two actors are evaluated:

1. Collect per-actor P(Adult) predictions from the MLA model
2. Apply threshold `t` to omit uncertain messages
3. Run binomial test: `binomtest(n_adult_votes, n_kept, p=0.5, alternative='two-sided')`
4. Apply Actor Significance Threshold (AST):
   - `p_value < AST` and `mean > 0.5` → Actor is **Adult (A)**
   - `p_value < AST` and `mean ≤ 0.5` → Actor is **Child (C)**
   - `p_value ≥ AST` → Actor is **Not Significant (NS)**
5. AC Context Rule (paper Figure 3):
   ```
   is_AC = (A XOR B is Adult) AND (A XOR B is Child)
   ```
   i.e., exactly one actor is A and exactly one is C; NS in either position → non-AC

**AST values tested:** `{0.05, 0.01, 0.001}`

**Context metrics:** TP, FP, TN, FN, F1

### 3.3 Train/Test Split

80% training / 20% testing, split **by conversation** (not by utterance).  
Same conversation's predator and officer files always land in the same split, preventing data leakage.

### 3.4 Preprocessing

Per paper Section 4.2: **no preprocessing** beyond excluding transcripts shorter than 10 messages (for PAN12; not applied to PJ in this implementation). Punctuation, stopwords, and casing are preserved deliberately — they carry signal for Adult/Child language distinction.

---

## 4. Paper-to-Dataset Reproducibility Mapping

| Component | Status | Notes |
|---|---|---|
| MLA — SVM | 🟡 Approx. | Reproducible; lower TP on small sample due to training set size |
| MLA — Naive Bayes | 🟡 Approx. | Same as SVM |
| MLA — BERT | 🟡 Approx. | Requires TensorFlow + GPU; fully implemented |
| MLA — RoBERTa | 🟡 Approx. | Requires TensorFlow + GPU; fully implemented |
| Message Threshold t (Eq. 1) | 🟢 Exact | Implemented precisely as stated |
| Actor Significance (binomial test) | 🟢 Exact | `scipy.stats.binomtest` with two-sided alternative |
| AC Context Rule (Fig. 3) | 🟢 Exact | XOR logic exactly as described |
| MLA metrics (TP, IA, IC, O) | 🟢 Exact | Matches paper Tables 2/3 format |
| Context metrics (TP, FP, TN, FN, F1) | 🟡 Partial | FP=0, TN=0 on PJ (all-positive); valid for TP/FN/F1 |
| 80/20 conversation split | 🟢 Exact | GroupShuffleSplit by conversation_id |
| Minimal preprocessing | 🟢 Exact | Whitespace strip only |
| Grooming phase analysis | 🔴 Not possible | Timestamps removed; chronological order lost |
| Cross-dataset eval (PAN12) | 🔴 Not reproduced | PAN12 not included; requires separate download |
| Turn-taking features | 🔴 Not possible | Speakers separated; interleaving lost |

---

## 5. Implementation Architecture

```
BTP-demo/
├── PervertedJusticeDataset/        # Cloned dataset (never modified)
│   ├── predators/   (621 files)
│   ├── officers/    (638 files)
│   ├── meta.tsv
│   └── url_name.tsv
│
└── grooming_detection/             # All implementation code
    ├── inspect_dataset.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── message_classifier.py
    ├── context_determination.py
    ├── actor_classification.py
    ├── evaluation.py
    ├── run_experiments.py
    ├── requirements.txt
    ├── README.md
    ├── REPORT.md
    ├── ui/
    │   ├── app.py                  # Streamlit entry point
    │   └── pages/
    │       ├── home.py
    │       ├── dataset_overview.py
    │       ├── run_experiments_ui.py
    │       ├── results_viewer.py
    │       ├── live_classifier.py
    │       └── paper_mapping.py
    ├── tests/
    │   ├── test_file_pairing.py
    │   ├── test_role_labels.py
    │   ├── test_preprocessing.py
    │   ├── test_data_loader.py
    │   └── test_splitting.py
    └── results/
        ├── dataset_inspection_report.txt
        ├── paper_dataset_mapping.md
        ├── mla_svm.json
        ├── mla_nb.json
        ├── context_svm.csv
        ├── context_nb.csv
        ├── confusion_matrix_svm.png
        ├── confusion_matrix_nb.png
        └── experiment_summary.json
```

---

## 6. Module Descriptions

### `inspect_dataset.py`
Standalone script that analyzes the raw dataset and produces `results/dataset_inspection_report.txt`. Reports file counts, encoding, utterance statistics, pairing validation, duplicate detection, and reproducibility assessment.

### `data_loader.py`
Core data access layer. Loads all utterances from both speaker folders into a `pandas.DataFrame` with columns:
- `utterance` — message text
- `speaker_role` — `1` (predator/Adult) or `0` (officer/Child); assigned from **file structure only**, never from content
- `conversation_id` — username extracted from filename; groups predator+officer utterances from same conversation
- `file_name` — source file for traceability

### `preprocessing.py`
- `minimal_clean()` — whitespace strip only (paper §4.2)
- `group_train_test_split()` — 80/20 split by conversation using `GroupShuffleSplit`; guarantees zero leakage between splits
- `build_count_vectorizer()` — fits `CountVectorizer` on training texts (paper's feature extraction method)
- `extract_labels_and_texts()` — splits DataFrame into `(texts, labels)` arrays

### `message_classifier.py`
Four classifiers implementing the paper's MLA:
- `SVMClassifier` — `LinearSVC` with calibration inside a Pipeline
- `NaiveBayesClassifier` — `MultinomialNB` inside a Pipeline
- `BERTClassifier` — `bert-base-uncased` via TensorFlow/HuggingFace
- `RoBERTaClassifier` — `roberta-base` via TensorFlow/HuggingFace

Also contains:
- `apply_message_threshold(proba, t)` — implements Equation (1) from the paper
- `compute_mla_metrics(y_true, y_pred, ...)` — computes TP%, IA%, IC%, O%

### `context_determination.py`
Implements the paper's Figure 3 algorithm:
- `determine_actor_significance()` — runs `scipy.stats.binomtest` and applies AST threshold; returns `A`, `C`, or `NS`
- `determine_transcript_context()` — applies the XOR logic to produce `AC` or `non-AC`
- `run_context_determination()` — processes a full test DataFrame conversation by conversation
- `compute_context_metrics()` — computes TP, FP, TN, FN, F1

### `actor_classification.py`
Pipeline bridge between message classifiers and context determination:
- `attach_predictions()` — runs a trained classifier over all test utterances and attaches probabilities
- `sweep_ast_t_values()` — sweeps all combinations of AST × t and returns a results DataFrame
- `run_full_actor_analysis()` — runs the full sweep for all classifiers

### `evaluation.py`
- `compute_standard_metrics()` — accuracy, precision, recall, F1 via sklearn
- `plot_confusion_matrix()` — saves labeled confusion matrix PNG (labeled as speaker-role, not grooming)
- `format_mla_table()` / `format_context_table()` — human-readable results tables
- `print_paper_comparison()` — side-by-side with published results (reference only)
- `PAPER_REFERENCE_RESULTS` — paper's published numbers stored for comparison display; never injected into experiment outputs

### `run_experiments.py`
Full pipeline orchestration with CLI flags:
- `--skip-transformers` — run SVM and NB only
- `--fast` — sample N conversations for quick verification
- `--debug` — verbose logging

### `ui/app.py` + `ui/pages/`
Six-page Streamlit web application (see Section 8).

---

## 7. Experimental Results

### 7.1 Fast-Mode Baseline (50 conversations sample)

These results use a 50-conversation sample for quick verification. Full dataset results will differ — see Section 7.2 for paper reference.

#### Message-Level Analysis

| Model | TP% | IA% | IC% | O% |
|---|---|---|---|---|
| SVM | 61.8 | 28.7 | 9.5 | 0.0 |
| NB  | 57.0 | 35.2 | 7.8 | 0.0 |

- **TP%** = correct speaker-role determinations (Adult/Child language classification)
- Lower TP vs. paper is expected: fast mode trains on ~85k utterances; paper uses ~219k+
- **These are NOT grooming detection metrics**

#### Context Determination (10 test conversations)

| Model | AST | TP | FN | FP | TN | F1 |
|---|---|---|---|---|---|---|
| SVM | 0.05 | 4 | 3 | 0 | 0 | 0.727 |
| SVM | 0.01 | 4 | 3 | 0 | 0 | 0.727 |
| SVM | 0.001 | 4 | 3 | 0 | 0 | 0.727 |
| NB  | 0.05 | 5 | 2 | 0 | 0 | 0.833 |
| NB  | 0.01 | 5 | 2 | 0 | 0 | 0.833 |
| NB  | 0.001 | 5 | 2 | 0 | 0 | 0.833 |

FP=0, TN=0 is correct and expected — all PJ conversations are OG-positive.

### 7.2 Paper Published Results (Reference Only)

These values are from Street et al. (2025) and are displayed for comparison only. They are never hardcoded into experiment outputs.

#### MLA (Table 2, PJ dataset, inter-set)

| Model | t | TP% | IA% | IC% | O% |
|---|---|---|---|---|---|
| SVM | — | 76.0 | 13.5 | 10.5 | 0.0 |
| NB | — | 74.7 | 12.0 | 13.3 | 0.0 |
| BERT | 0.2 | 82.2 | 9.3 | 8.5 | 9.2 |
| BERT | 0.45 | 87.3 | 7.1 | 5.6 | 30.0 |
| RoBERTa | 0.2 | 83.3 | 7.7 | 9.0 | 8.8 |
| RoBERTa | 0.45 | 88.8 | 5.3 | 5.9 | 30.4 |

#### Context Determination (Table 4, PJ, AST=0.05)

| Model | TP | FN | F1 |
|---|---|---|---|
| SVM | 118 | 35 | 0.871 |
| NB | 116 | 37 | 0.862 |
| BERT (t=0.2) | 143 | 10 | 0.966 |
| RoBERTa (t=0.45) | 145 | 8 | 0.973 |

### 7.3 Why Our Results Differ from the Paper

1. **Dataset version:** The paper parsed directly from perverted-justice.com HTML; this repo is a pre-cleaned subset of 622 conversations
2. **Training set size:** Fast mode uses 50 conversations; paper uses full dataset
3. **Framework:** Paper uses TensorFlow throughout; SVM/NB here use sklearn (consistent with paper's sklearn mention)
4. **Random seed differences:** Split randomness affects small samples significantly

---

## 8. Web UI

A six-page Streamlit application provides interactive access to all functionality.

### Pages

| Page | Description |
|---|---|
| **🏠 Home** | Status dashboard — dataset availability, file counts, test pass rate, constraint warnings |
| **📊 Dataset Overview** | Interactive histograms of utterances per file, role balance pie chart, pairing analysis, full searchable file table |
| **🤖 Run Experiments** | Interactive training controls — model selection, test split slider, fast mode, AST sweep; streams live progress and displays results inline |
| **📈 Results & Metrics** | MLA grouped bar chart, context F1 heatmap (AST × model), confusion matrix images, side-by-side comparison with paper reference |
| **💬 Live Classifier** | Real-time message classification with gauge chart; batch classification with color-coded table; demo conversation with chat-bubble UI |
| **📄 Paper Mapping** | Full reproducibility table, three-task explanation, raw dataset inspection report |

### Running the UI

```bash
cd grooming_detection
python -m streamlit run ui/app.py
# Opens at http://localhost:8501
```

---

## 9. Test Suite

**88 tests, all passing.**

```
python -m pytest grooming_detection/tests/ -v
# 88 passed in 13.60s
```

### Test Files

| File | Tests | What is verified |
|---|---|---|
| `test_file_pairing.py` | 14 | Filename parsing, predator/officer counts, multi-officer suffix handling, no duplicates, no empty files |
| `test_role_labels.py` | 9 | Labels from file structure not content; both roles present; no `is_grooming` column |
| `test_preprocessing.py` | 18 | Minimal cleaning preserves punctuation/casing; group split has zero conversation leakage; reproducibility with fixed seed |
| `test_data_loader.py` | 11 | Temp directory loading; multi-officer files; min-utterance filter; null check; conversation count |
| `test_splitting.py` | 20 | Group integrity; message threshold Eq.(1) boundaries; MLA metrics correctness; binomial test direction; AC/non-AC XOR logic |

### Key Safety Tests

- **No conversation leakage** — `test_zero_conversation_overlap` verifies train/test conversation IDs are disjoint
- **Label source** — `test_label_source_is_file_structure_not_content` verifies labels come from folder, not message text
- **No grooming column** — `test_label_docstring_warning_not_violated` asserts no `is_grooming` column exists
- **Threshold boundaries** — `test_boundary_values_not_omitted` verifies strict inequality in Equation (1)
- **XOR logic** — `test_both_adult_not_ac`, `test_ns_in_either_position_not_ac` verify context rule correctly rejects

---

## 10. Known Limitations

### Dataset Limitations

| Limitation | Impact | Workaround |
|---|---|---|
| Timestamps removed | Cannot reproduce phase-based analysis or early detection | Document and skip those paper components |
| All-positive dataset | FP=0, TN=0 — cannot measure false positive rate | Add PAN12 dataset for cross-dataset evaluation |
| Officers are adult volunteers | Child language model may not generalize to real children | Acknowledge in all reporting |
| 622 conversations only | Smaller than full PJ corpus (paper may have used more) | Use full dataset for final runs |

### Implementation Limitations

| Limitation | Impact | Path to resolve |
|---|---|---|
| BERT/RoBERTa need TensorFlow + GPU | Transformer experiments not runnable on CPU-only machines | Install `tensorflow==2.17.0` and run on GPU |
| PAN12 not included | Cannot reproduce cross-dataset F1 scores from paper Tables 5–7 | Download PAN12 separately and add `pan12_loader.py` |
| Fast mode results differ from paper | Expected — not all 622 conversations used | Run without `--fast` for full results |

---

## 11. Reproduction Instructions

### Prerequisites

```bash
# Python 3.10+
pip install -r grooming_detection/requirements.txt

# For BERT/RoBERTa (optional, GPU recommended):
pip install tensorflow==2.17.0
```

### Step-by-Step

```bash
# 1. Clone dataset
git clone https://github.com/helenkoutli/PervertedJusticeDataset.git

# 2. Inspect dataset
python grooming_detection/inspect_dataset.py

# 3. Run tests
python -m pytest grooming_detection/tests/ -v

# 4. Quick verification (50 conversations, ~30 seconds)
python grooming_detection/run_experiments.py --skip-transformers --fast

# 5. Full SVM + NB on complete dataset
python grooming_detection/run_experiments.py --skip-transformers

# 6. Full pipeline with BERT + RoBERTa (GPU required)
python grooming_detection/run_experiments.py

# 7. Launch web UI
python -m streamlit run grooming_detection/ui/app.py
```

### Expected Outputs

After running Step 5 (full SVM + NB):

```
results/
├── dataset_inspection_report.txt    # dataset statistics
├── paper_dataset_mapping.md         # reproducibility table
├── mla_svm.json                     # SVM MLA metrics
├── mla_nb.json                      # NB MLA metrics
├── context_svm.csv                  # SVM context determination sweep
├── context_nb.csv                   # NB context determination sweep
├── confusion_matrix_svm.png         # SVM confusion matrix
├── confusion_matrix_nb.png          # NB confusion matrix
└── experiment_summary.json          # combined summary
```

---

*Report generated for BTP-demo project.*  
*Dataset: PervertedJusticeDataset (Koutli, 2023) — used for academic research only.*
