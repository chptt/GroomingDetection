"""
message_classifier.py

Message-Level Analysis (MLA) as described in Section 3.1 of the paper.

Task: Classify each utterance as Adult (1) or Child (0) based on language.
      This is NOT grooming detection — it is speaker-role classification.

Models implemented:
  - SVM      : sklearn Pipeline with CountVectorizer
  - NB       : sklearn MultinomialNB with CountVectorizer
  - BERT     : bert-base-uncased fine-tuned via PyTorch + HuggingFace
  - RoBERTa  : roberta-base fine-tuned via PyTorch + HuggingFace

Note on framework: The paper uses TensorFlow. This implementation uses
PyTorch because tensorflow==2.17 is not available for Python 3.13+.
The model architecture (bert-base-uncased, roberta-base), hyperparameters
(lr=2e-5, 3 epochs), and tokenisation are identical to the paper.

Results metrics (paper Tables 2/3):
  - TP : True Positive — correct Adult(A) or Child(C) determination
  - IA : Incorrect Adult Determination — predicted Adult but was Child
  - IC : Incorrect Child Determination — predicted Child but was Adult
  - O  : Omissions — messages excluded by Message Threshold t (BERT/RoBERTa only)

Message Threshold (t) — Equation (1) from paper:
  A message with prediction n is omitted if: (0.5 + t) > n > (0.5 - t)
  t values tested: 0.2, 0.3, 0.4, 0.45
  (Only applicable to BERT and RoBERTa which produce probability outputs)
"""

from pathlib import Path
import logging
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Message Threshold (Equation 1 from paper)
# ---------------------------------------------------------------------------

def apply_message_threshold(
    probabilities: np.ndarray,
    t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply message threshold t to filter low-confidence predictions.

    From paper Equation (1): A message is omitted if (0.5 + t) > n > (0.5 - t)
    where n is the prediction probability for Adult class.

    Args:
        probabilities: 1D array of P(Adult) for each message, in [0, 1]
        t: threshold value (0.2, 0.3, 0.4, or 0.45)

    Returns:
        kept_mask  : boolean array, True for messages NOT omitted
        omit_mask  : boolean array, True for omitted messages
    """
    lower = 0.5 - t
    upper = 0.5 + t
    omit_mask = (probabilities > lower) & (probabilities < upper)
    kept_mask = ~omit_mask
    return kept_mask, omit_mask


# ---------------------------------------------------------------------------
# MLA metrics (paper-specific)
# ---------------------------------------------------------------------------

def compute_mla_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    kept_mask: Optional[np.ndarray] = None,
    n_total: Optional[int] = None,
) -> dict:
    """
    Compute MLA metrics as defined in the paper.

    Metrics:
      - TP : fraction of kept messages correctly classified
      - IA : fraction of kept messages that were Adult but classified as Child
      - IC : fraction of kept messages that were Child but classified as Adult
      - O  : fraction of ALL messages that were omitted

    Args:
        y_true    : ground-truth labels (1=Adult, 0=Child) for kept messages
        y_pred    : predicted labels for kept messages
        kept_mask : boolean mask applied to the full test set (for O calculation)
        n_total   : total number of messages before omission (for O calculation)

    Returns:
        dict with keys: TP, IA, IC, O (all as percentages)
    """
    n_kept = len(y_true)

    if n_total is None:
        n_total = n_kept
    n_omitted = n_total - n_kept

    correct = (y_pred == y_true).sum()
    ia = ((y_pred == 1) & (y_true == 0)).sum()
    ic = ((y_pred == 0) & (y_true == 1)).sum()

    tp_pct = 100.0 * correct / n_kept if n_kept > 0 else 0.0
    ia_pct = 100.0 * ia / n_kept if n_kept > 0 else 0.0
    ic_pct = 100.0 * ic / n_kept if n_kept > 0 else 0.0
    o_pct  = 100.0 * n_omitted / n_total if n_total > 0 else 0.0

    return {
        "TP": round(tp_pct, 1),
        "IA": round(ia_pct, 1),
        "IC": round(ic_pct, 1),
        "O":  round(o_pct, 1),
        "n_total":   n_total,
        "n_kept":    n_kept,
        "n_omitted": n_omitted,
    }


# ---------------------------------------------------------------------------
# SVM Classifier
# ---------------------------------------------------------------------------

class SVMClassifier:
    """
    SVM classifier using sklearn Pipeline + CountVectorizer.
    Paper: sklearn pipeline with countVectoriser approach.
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ("vectorizer", CountVectorizer()),
            ("classifier", CalibratedClassifierCV(LinearSVC(max_iter=2000))),
        ])
        self.name = "SVM"

    def fit(self, X_texts: list[str], y: np.ndarray) -> "SVMClassifier":
        self.pipeline.fit(X_texts, y)
        return self

    def predict(self, X_texts: list[str]) -> np.ndarray:
        return self.pipeline.predict(X_texts)

    def predict_proba(self, X_texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(X_texts)[:, 1]

    def evaluate(self, X_texts: list[str], y_true: np.ndarray) -> dict:
        y_pred = self.predict(X_texts)
        return compute_mla_metrics(y_true, y_pred)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)
        logger.info(f"SVM saved to {path}")

    def load(self, path: Path) -> "SVMClassifier":
        with open(path, "rb") as f:
            self.pipeline = pickle.load(f)
        return self


# ---------------------------------------------------------------------------
# Naive Bayes Classifier
# ---------------------------------------------------------------------------

class NaiveBayesClassifier:
    """
    Naive Bayes classifier using sklearn.naive_bayes + CountVectorizer.
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ("vectorizer", CountVectorizer()),
            ("classifier", MultinomialNB()),
        ])
        self.name = "NB"

    def fit(self, X_texts: list[str], y: np.ndarray) -> "NaiveBayesClassifier":
        self.pipeline.fit(X_texts, y)
        return self

    def predict(self, X_texts: list[str]) -> np.ndarray:
        return self.pipeline.predict(X_texts)

    def predict_proba(self, X_texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(X_texts)[:, 1]

    def evaluate(self, X_texts: list[str], y_true: np.ndarray) -> dict:
        y_pred = self.predict(X_texts)
        return compute_mla_metrics(y_true, y_pred)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)
        logger.info(f"NB saved to {path}")

    def load(self, path: Path) -> "NaiveBayesClassifier":
        with open(path, "rb") as f:
            self.pipeline = pickle.load(f)
        return self


# ---------------------------------------------------------------------------
# Shared PyTorch trainer for BERT / RoBERTa
# ---------------------------------------------------------------------------

class _TransformerClassifier:
    """
    Base class for BERT and RoBERTa classifiers using PyTorch.

    Hyperparameters match the paper:
      - Epochs      : 3
      - Learning rate: 2e-5
      - Max length  : 128 tokens
      - Batch size  : 16 (configurable for memory)
    """

    def __init__(self, model_name: str, name: str, max_length: int = 128):
        self.model_name  = model_name
        self.name        = name
        self.max_length  = max_length
        self.model       = None
        self.tokenizer   = None
        self._loaded     = False

    # ── lazy load ────────────────────────────────────────────────────────────
    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model     = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=2
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self._loaded = True
            logger.info(f"{self.name} loaded on {self.device}")
        except Exception as e:
            raise RuntimeError(
                f"Could not load {self.name}. "
                f"Ensure 'transformers' and 'torch' are installed.\n{e}"
            )

    # ── training ─────────────────────────────────────────────────────────────
    def fit(
        self,
        X_texts: list[str],
        y: np.ndarray,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ) -> "_TransformerClassifier":
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import get_linear_schedule_with_warmup

        self._load()
        logger.info(f"Training {self.name} | {len(X_texts)} samples | "
                    f"{epochs} epochs | batch={batch_size} | lr={learning_rate}")

        # Tokenise in chunks to avoid OOM on large datasets
        enc = self.tokenizer(
            X_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels  = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(
            enc["input_ids"],
            enc["attention_mask"],
            labels,
        )
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for step, batch in enumerate(loader):
                input_ids      = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                batch_labels   = batch[2].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels,
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

                if (step + 1) % 100 == 0:
                    logger.info(f"  Epoch {epoch+1}/{epochs} | "
                                f"Step {step+1}/{len(loader)} | "
                                f"Loss {total_loss/(step+1):.4f}")

            avg_loss = total_loss / len(loader)
            logger.info(f"  Epoch {epoch+1}/{epochs} complete — avg loss: {avg_loss:.4f}")

        self.model.eval()
        return self

    # ── inference ─────────────────────────────────────────────────────────────
    def predict_proba(self, X_texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Returns P(Adult) for each message. Processes in batches to avoid OOM."""
        import torch

        self._load()
        self.model.eval()
        all_probs = []

        for i in range(0, len(X_texts), batch_size):
            batch_texts = X_texts[i : i + batch_size]
            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                input_ids      = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.append(probs)

        return np.concatenate(all_probs)

    def predict(self, X_texts: list[str]) -> np.ndarray:
        return (self.predict_proba(X_texts) >= 0.5).astype(int)

    def evaluate(
        self,
        X_texts: list[str],
        y_true: np.ndarray,
        t_values: list[float] = [0.0, 0.2, 0.3, 0.4, 0.45],
    ) -> dict[str, dict]:
        """Evaluate with multiple message thresholds."""
        proba   = self.predict_proba(X_texts)
        n_total = len(y_true)
        results = {}

        for t in t_values:
            if t == 0.0:
                y_pred  = (proba >= 0.5).astype(int)
                metrics = compute_mla_metrics(y_true, y_pred, n_total=n_total)
            else:
                kept_mask, _ = apply_message_threshold(proba, t)
                y_pred_kept  = (proba[kept_mask] >= 0.5).astype(int)
                y_true_kept  = y_true[kept_mask]
                metrics      = compute_mla_metrics(
                    y_true_kept, y_pred_kept,
                    kept_mask=kept_mask, n_total=n_total,
                )
            results[f"t={t}"] = metrics

        return results

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        logger.info(f"{self.name} saved to {path}")

    def load(self, path: Path) -> "_TransformerClassifier":
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model     = AutoModelForSequenceClassification.from_pretrained(str(path))
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self._loaded   = True
        return self


# ---------------------------------------------------------------------------
# BERT Classifier
# ---------------------------------------------------------------------------

class BERTClassifier(_TransformerClassifier):
    """
    BERT-base-uncased fine-tuned for Adult/Child speaker-role classification.
    Uses PyTorch backend (paper used TensorFlow; framework changed due to
    TensorFlow not supporting Python 3.13).
    """
    def __init__(self, max_length: int = 128):
        super().__init__(
            model_name="bert-base-uncased",
            name="BERT",
            max_length=max_length,
        )


# ---------------------------------------------------------------------------
# RoBERTa Classifier
# ---------------------------------------------------------------------------

class RoBERTaClassifier(_TransformerClassifier):
    """
    RoBERTa-base fine-tuned for Adult/Child speaker-role classification.
    Uses PyTorch backend (paper used TensorFlow; framework changed due to
    TensorFlow not supporting Python 3.13).
    """
    def __init__(self, max_length: int = 128):
        super().__init__(
            model_name="roberta-base",
            name="RoBERTa",
            max_length=max_length,
        )
