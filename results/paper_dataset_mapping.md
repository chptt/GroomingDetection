# Paper-to-Dataset Mapping

**Paper:** "Enhanced Online Grooming Detection Employing Context Determination and Message-Level Analysis"  
**Authors:** Jake Street, Isibor Kennedy Ihianle, Funminiyi Olajide, Ahmad Lotfi  
**Published:** Intelligent Systems with Applications 28 (2025) 200607  
**Dataset:** PervertedJustice (PJ) — cleaned version by Eleni Koutli (2023)

---

## Dataset Inspection Findings

| Property | Finding |
|---|---|
| Predator files | 621 |
| Officer files | 638 |
| Paired conversations | 609 |
| Predator-only files | 12 (multiple-officer conversations) |
| Officer-only files | 29 (same 12 convs with _1, _2 suffixes + 17 extra) |
| Total predator utterances | 706,214 |
| Total officer utterances | 662,925 |
| Timestamps | **Removed during cleaning** |
| Usernames | **Removed during cleaning** |
| Chronological interleaving | **Not preserved** |
| Encoding | ASCII |
| Empty files | 0 |

---

## Paper Component Mapping

| Paper Component | Required Inputs | Available in Dataset | Implementation Method | Reproducibility |
|---|---|---|---|---|
| **Message-Level Analysis (MLA)** | Per-message text + Adult/Child label | ✓ Predator files = Adult (1), Officer files = Child (0) | Train SVM/NB/BERT/RoBERTa to classify utterances | **Approximately** |
| **Message Threshold t (Eq. 1)** | Per-message probability n, t value | ✓ Classifier outputs probabilities | Omit if (0.5+t) > n > (0.5-t); t ∈ {0.2, 0.3, 0.4, 0.45} | **Exactly** |
| **Actor Significance (AST)** | Per-actor message predictions + p-value threshold | ✓ Group predictions by conversation_id + speaker_role | Binomial test; AST ∈ {0.05, 0.01, 0.001} | **Approximately** |
| **Context Determination** | Two actors per conversation with significance determinations | ✓ Predator+officer pair per conversation | AC if (one Adult XOR one Adult) AND (one Child XOR one Child) | **Approximately** |
| **Inter-set training (PJ → PJ)** | 80% PJ train, 20% PJ test | ✓ Group-based split by conversation | GroupShuffleSplit(test_size=0.2) | **Approximately** |
| **Cross-set training (PJ → PAN12)** | PJ training set + PAN12 test set | ✗ PAN12 not in this repository | Not implemented (PAN12 requires separate download) | **Not reproduced** |
| **MLA metrics (TP, IA, IC, O)** | Per-message true/predicted labels + omissions | ✓ | Computed from classifier outputs | **Exactly** |
| **Context metrics (TP, FP, TN, FN, F1)** | Transcript-level AC ground truth | ⚠ PJ = all-positive; no TN/FP possible on PJ alone | F1 computed; FP/TN = 0 on PJ (as in paper Tables 4–6) | **Exactly for PJ** |
| **Preprocessing (paper §4.2)** | Raw utterances | ✓ | Whitespace strip only; no punctuation/stopword removal | **Exactly** |
| **Grooming phase progression** | Chronological message order | ✗ Timestamps removed; order lost | **Cannot be reproduced** | **Not at all** |
| **Turn-taking analysis** | Interleaved conversation flow | ✗ Speakers separated into different files | **Cannot be reproduced** | **Not at all** |
| **Early detection** | Progressive message disclosure | ✗ No chronological ordering | **Cannot be reproduced** | **Not at all** |
| **PAN12 cross-dataset evaluation** | PAN12 positive+negative transcripts | ✗ Not available in this repo | Not implemented | **Not reproduced** |

---

## Three Distinct Tasks (Must Not Be Conflated)

### Task 1: Speaker-Role Classification (MLA)
- **What it is:** Classify each utterance as Adult language (1) or Child language (0)
- **Ground truth:** File structure — predator file = Adult, officer file = Child
- **What it measures:** How well models distinguish adult vs. child linguistic patterns
- **What it does NOT measure:** Whether a message is "grooming content"
- **Metrics used:** TP%, IA%, IC%, O% (paper Tables 2/3)

### Task 2: Actor Significance Determination
- **What it is:** Statistical test to determine if an actor is significantly an Adult, Child, or NS
- **Inputs:** Aggregate P(Adult) predictions for all messages from one actor
- **Method:** Binomial test against p=0.5; AST threshold determines significance
- **Important:** This aggregates Task 1 predictions — it does not re-classify at message level

### Task 3: Conversation-Level AC Context Detection
- **What it is:** Determine if a conversation involves an Adult-Child dyad
- **Inputs:** Two actor significance determinations (predator + officer)
- **Rule:** AC if exactly one actor is significantly Adult AND exactly one is significantly Child
- **Metrics:** TP, FP, TN, FN, F1 (paper Tables 4–7)
- **Dataset constraint:** On PJ alone, all conversations are OG-positive → FP=0, TN=0

---

## What Cannot Be Reproduced (and Why)

| Missing Element | Reason | Impact |
|---|---|---|
| Cross-dataset experiments (PAN12) | PAN12 not included in this repository | Cannot evaluate FP/TN or cross-dataset robustness |
| Grooming phase analysis | Chronological order not preserved | Cannot detect early vs. late grooming stages |
| Negative OG transcripts | All PJ conversations are OG-positive | Cannot compute false positive rate on PJ alone |
| Real child language validation | Officers are adult honeypot actors | Child language model may not reflect real child speech |
| Email transcript exclusion | Not identified in cleaned dataset | Minor preprocessing difference |

---

## Important Caveats

1. **Speaker role ≠ grooming label.** The paper classifies Adult vs. Child *language patterns*, not grooming messages. A predator's message "how are you?" is labeled Adult (1) by structure, not because it is grooming.

2. **Officers are not real children.** The honeypot children in PJ are adult volunteers. The paper acknowledges: *"the child messages were sent by an adult pretending to be a child, therefore the language used may not be truly representative of how a child would interact."*

3. **All PJ conversations are OG-positive.** There are no negative examples in this dataset. The paper's FP/TN metrics (Tables 7) come from PAN12 evaluation. Our context determination results on PJ (Tables 4–6 equivalent) will always show FP=0, TN=0.

4. **TP+IA+IC ≠ 100%.** The paper explicitly notes these percentages are over *kept* messages; O% is over *all* messages. This is by design.

5. **Our results differ from paper results.** Expected reasons:
   - The paper uses the full PJ HTML dataset (parsed from perverted-justice.com); this repo is a cleaned subset
   - The paper's PJ dataset includes transcripts up to time of publication; this repo covers 622 conversations
   - The paper's context determination uses the full 80% training set; fast-mode uses 50 conversations
