"""
message_classifier.py

Message-Level Analysis (MLA) as described in Section 3.1 of the paper.

Task: Classify each utterance as Adult (1) or Child (0) based on language.
      This is NOT grooming detection — it is speaker-role classification.

Models implemented:
  - SVM  : sklearn Pipeline with CountVectorizer (paper: sklearn pipeline + countVectoriser)
  - NB   : sklearn MultinomialNB with CountVectorizer (paper: sklearn.naive_bayes)
  - BERT : bert-base-uncased fine-tuned via TensorFlow (paper uses TF)
  - RoBERTa: roberta-base fine-tuned via TensorFlow (paper uses RoBERTa-base)

Results metrics (paper Table 2/3):
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
    # Omit if prediction is in the low-confidence band
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
      - O  : fraction of ALL messages that were omitted (not part of TP+IA+IC)

    Args:
        y_true     : ground-truth labels (1=Adult, 0=Child) for kept messages
        y_pred     : predicted labels for kept messages
        kept_mask  : boolean mask applied to the full test set (for O calculation)
        n_total    : total number of messages before omission (for O calculation)

    Returns:
        dict with keys: TP, IA, IC, O (all as percentages, matching paper Tables 2/3)
    """
    n_kept = len(y_true)

    if n_total is None:
        n_total = n_kept
    n_omitted = n_total - n_kept

    # Among kept messages
    correct = (y_pred == y_true).sum()
    # IA: predicted Adult (1) but true was Child (0)
    ia = ((y_pred == 1) & (y_true == 0)).sum()
    # IC: predicted Child (0) but true was Adult (1)
    ic = ((y_pred == 0) & (y_true == 1)).sum()

    tp_pct = 100.0 * correct / n_kept if n_kept > 0 else 0.0
    ia_pct = 100.0 * ia / n_kept if n_kept > 0 else 0.0
    ic_pct = 100.0 * ic / n_kept if n_kept > 0 else 0.0
    o_pct = 100.0 * n_omitted / n_total if n_total > 0 else 0.0

    return {
        "TP": round(tp_pct, 1),
        "IA": round(ia_pct, 1),
        "IC": round(ic_pct, 1),
        "O": round(o_pct, 1),
        "n_total": n_total,
        "n_kept": n_kept,
        "n_omitted": n_omitted,
    }


# ---------------------------------------------------------------------------
# SVM Classifier
# ---------------------------------------------------------------------------

class SVMClassifier:
    """
    SVM classifier using sklearn Pipeline + CountVectorizer.
    Paper: "For the SVM model, sklearn module was used alongside the 'pipeline'
            with a 'countVectoriser' approach."
    
    LinearSVC is calibrated to produce probabilities (not used for omissions
    since SVM does not natively produce calibrated probabilities — the paper
    reports 0% omissions for SVM/NB, confirming no threshold is applied).
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
        """Returns P(Adult) for each message."""
        proba = self.pipeline.predict_proba(X_texts)
        return proba[:, 1]  # probability of class 1 (Adult)

    def evaluate(self, X_texts: list[str], y_true: np.ndarray) -> dict:
        """Evaluate with no omissions (t=0 for SVM, per paper)."""
        y_pred = self.predict(X_texts)
        metrics = compute_mla_metrics(y_true, y_pred)
        return metrics

    def save(self, path: Path) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        logger.info(f"SVM saved to {path}")

    def load(self, path: Path) -> "SVMClassifier":
        with open(path, 'rb') as f:
            self.pipeline = pickle.load(f)
        return self


# ---------------------------------------------------------------------------
# Naive Bayes Classifier
# ---------------------------------------------------------------------------

class NaiveBayesClassifier:
    """
    Naive Bayes classifier using sklearn.naive_bayes.
    Paper: "For NB the 'sklearn.naive_bayes' module was used alongside the
            feature extraction module."
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
        """Returns P(Adult) for each message."""
        proba = self.pipeline.predict_proba(X_texts)
        return proba[:, 1]  # probability of class 1 (Adult)

    def evaluate(self, X_texts: list[str], y_true: np.ndarray) -> dict:
        """Evaluate with no omissions (t=0 for NB, per paper)."""
        y_pred = self.predict(X_texts)
        metrics = compute_mla_metrics(y_true, y_pred)
        return metrics

    def save(self, path: Path) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        logger.info(f"NB saved to {path}")

    def load(self, path: Path) -> "NaiveBayesClassifier":
        with open(path, 'rb') as f:
            self.pipeline = pickle.load(f)
        return self


# ---------------------------------------------------------------------------
# BERT Classifier
# ---------------------------------------------------------------------------

class BERTClassifier:
    """
    BERT-based binary classifier for Adult/Child message classification.
    Paper: "For both transformer-based models (BERT and RoBERTa) 'tensorflow' was
            utilised. For BERT, the BERT tokenizer within tensorflow was used."

    Uses HuggingFace transformers with TensorFlow backend.
    Model: bert-base-uncased (standard BERT)
    """

    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.name = "BERT"
        self._available = False

    def _load_model(self) -> bool:
        """Lazy-load model to avoid import at module level."""
        try:
            import tensorflow as tf
            from transformers import TFBertForSequenceClassification, BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = TFBertForSequenceClassification.from_pretrained(
                self.model_name, num_labels=2
            )
            self._available = True
            return True
        except ImportError as e:
            logger.warning(f"Cannot load BERT: {e}. Install tensorflow and transformers.")
            return False

    def fit(
        self,
        X_texts: list[str],
        y: np.ndarray,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ) -> "BERTClassifier":
        if not self._available:
            if not self._load_model():
                raise RuntimeError("BERT dependencies not available")

        import tensorflow as tf

        encodings = self.tokenizer(
            X_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="tf",
        )
        labels = tf.convert_to_tensor(y, dtype=tf.int32)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

        dataset = tf.data.Dataset.from_tensor_slices((
            {"input_ids": encodings["input_ids"],
             "attention_mask": encodings["attention_mask"],
             "token_type_ids": encodings["token_type_ids"]},
            labels
        )).batch(batch_size)

        self.model.fit(dataset, epochs=epochs)
        return self

    def predict_proba(self, X_texts: list[str]) -> np.ndarray:
        """Returns P(Adult) for each message."""
        import tensorflow as tf

        encodings = self.tokenizer(
            X_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="tf",
        )
        logits = self.model(encodings).logits
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        return probs[:, 1]  # P(Adult)

    def predict(self, X_texts: list[str]) -> np.ndarray:
        proba = self.predict_proba(X_texts)
        return (proba >= 0.5).astype(int)

    def evaluate(
        self,
        X_texts: list[str],
        y_true: np.ndarray,
        t_values: list[float] = [0.0, 0.2, 0.3, 0.4, 0.45],
    ) -> dict[str, dict]:
        """Evaluate with multiple t thresholds."""
        proba = self.predict_proba(X_texts)
        n_total = len(y_true)
        results = {}

        for t in t_values:
            if t == 0.0:
                # No omission
                y_pred = (proba >= 0.5).astype(int)
                metrics = compute_mla_metrics(y_true, y_pred, n_total=n_total)
                results[f"t={t}"] = metrics
            else:
                kept_mask, omit_mask = apply_message_threshold(proba, t)
                y_pred_kept = (proba[kept_mask] >= 0.5).astype(int)
                y_true_kept = y_true[kept_mask]
                metrics = compute_mla_metrics(
                    y_true_kept, y_pred_kept, kept_mask=kept_mask, n_total=n_total
                )
                results[f"t={t}"] = metrics

        return results

    def get_predictions_with_proba(
        self,
        X_texts: list[str],
        y_true: np.ndarray,
    ) -> pd.DataFrame:
        """Return DataFrame with text, true label, and prediction probability."""
        proba = self.predict_proba(X_texts)
        return pd.DataFrame({
            "utterance": X_texts,
            "true_label": y_true,
            "pred_proba": proba,
        })

    def save(self, path: Path) -> None:
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))

    def load(self, path: Path) -> "BERTClassifier":
        from transformers import TFBertForSequenceClassification, BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(str(path))
        self.model = TFBertForSequenceClassification.from_pretrained(str(path))
        self._available = True
        return self


# ---------------------------------------------------------------------------
# RoBERTa Classifier
# ---------------------------------------------------------------------------

class RoBERTaClassifier:
    """
    RoBERTa-based binary classifier for Adult/Child message classification.
    Paper: "For RoBERTa the 'RoBERTa-base' tokenizer was used."
    """

    def __init__(self, model_name: str = "roberta-base", max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.name = "RoBERTa"
        self._available = False

    def _load_model(self) -> bool:
        try:
            import tensorflow as tf
            from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
            self.model = TFRobertaForSequenceClassification.from_pretrained(
                self.model_name, num_labels=2
            )
            self._available = True
            return True
        except ImportError as e:
            logger.warning(f"Cannot load RoBERTa: {e}. Install tensorflow and transformers.")
            return False

    def fit(
        self,
        X_texts: list[str],
        y: np.ndarray,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ) -> "RoBERTaClassifier":
        if not self._available:
            if not self._load_model():
                raise RuntimeError("RoBERTa dependencies not available")

        import tensorflow as tf

        encodings = self.tokenizer(
            X_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="tf",
        )
        labels = tf.convert_to_tensor(y, dtype=tf.int32)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

        dataset = tf.data.Dataset.from_tensor_slices((
            {"input_ids": encodings["input_ids"],
             "attention_mask": encodings["attention_mask"]},
            labels
        )).batch(batch_size)

        self.model.fit(dataset, epochs=epochs)
        return self

    def predict_proba(self, X_texts: list[str]) -> np.ndarray:
        """Returns P(Adult) for each message."""
        import tensorflow as tf

        encodings = self.tokenizer(
            X_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="tf",
        )
        logits = self.model(encodings).logits
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        return probs[:, 1]

    def predict(self, X_texts: list[str]) -> np.ndarray:
        proba = self.predict_proba(X_texts)
        return (proba >= 0.5).astype(int)

    def evaluate(
        self,
        X_texts: list[str],
        y_true: np.ndarray,
        t_values: list[float] = [0.0, 0.2, 0.3, 0.4, 0.45],
    ) -> dict[str, dict]:
        """Evaluate with multiple t thresholds."""
        proba = self.predict_proba(X_texts)
        n_total = len(y_true)
        results = {}

        for t in t_values:
            if t == 0.0:
                y_pred = (proba >= 0.5).astype(int)
                metrics = compute_mla_metrics(y_true, y_pred, n_total=n_total)
                results[f"t={t}"] = metrics
            else:
                kept_mask, omit_mask = apply_message_threshold(proba, t)
                y_pred_kept = (proba[kept_mask] >= 0.5).astype(int)
                y_true_kept = y_true[kept_mask]
                metrics = compute_mla_metrics(
                    y_true_kept, y_pred_kept, kept_mask=kept_mask, n_total=n_total
                )
                results[f"t={t}"] = metrics

        return results

    def get_predictions_with_proba(
        self,
        X_texts: list[str],
        y_true: np.ndarray,
    ) -> pd.DataFrame:
        proba = self.predict_proba(X_texts)
        return pd.DataFrame({
            "utterance": X_texts,
            "true_label": y_true,
            "pred_proba": proba,
        })

    def save(self, path: Path) -> None:
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))

    def load(self, path: Path) -> "RoBERTaClassifier":
        from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(str(path))
        self.model = TFRobertaForSequenceClassification.from_pretrained(str(path))
        self._available = True
        return self
