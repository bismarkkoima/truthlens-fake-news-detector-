"""Loading, predicting and explaining with the trained model bundle.

The bundle (written by ``train.py``) is a dict:

    {
        "vectorizer": fitted TfidfVectorizer,
        "models":     {"logreg": LogisticRegression, "nb": ComplementNB},
        "metrics":    {"logreg": {...}, "nb": {...}},
        "trained_at": ISO timestamp,
        "samples":    int,
    }

Both models are linear in the TF-IDF features, so we can explain a verdict
exactly (no sampling like LIME): each word's contribution is its TF-IDF weight
times the model's weight for that word.
"""

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from .preprocessing import clean_text

LABELS = ("FAKE", "REAL")  # class 0, class 1

MODEL_NAMES = {
    "logreg": "Logistic Regression",
    "nb": "Complement Naive Bayes",
}


@dataclass
class Verdict:
    label: str
    confidence: float  # 0-100, probability of the predicted label
    real_probability: float  # 0-1
    model_key: str
    evidence: list = field(default_factory=list)  # [(word, signed weight)]

    @property
    def model_name(self) -> str:
        return MODEL_NAMES.get(self.model_key, self.model_key)


class Detector:
    def __init__(self, bundle: dict):
        self.vectorizer = bundle["vectorizer"]
        self.models = bundle["models"]
        self.metrics = bundle.get("metrics", {})
        self.trained_at = bundle.get("trained_at")
        self.samples = bundle.get("samples")
        self._vocab = self.vectorizer.get_feature_names_out()
        self._word_weights = {k: self._linear_weights(m) for k, m in self.models.items()}

    @classmethod
    def load(cls, path) -> "Detector":
        return cls(joblib.load(Path(path)))

    @staticmethod
    def _linear_weights(model) -> np.ndarray:
        """Per-feature weight; positive pushes towards REAL."""
        if hasattr(model, "coef_"):
            return np.asarray(model.coef_).ravel()
        # Naive Bayes: log-likelihood ratio between the classes.
        log_prob = model.feature_log_prob_
        return log_prob[1] - log_prob[0]

    @property
    def available_models(self):
        return list(self.models)

    def predict(self, text: str, model_key: str = "logreg", top_k: int = 10) -> Verdict:
        if model_key not in self.models:
            raise KeyError(f"Unknown model '{model_key}'")
        model = self.models[model_key]
        features = self.vectorizer.transform([clean_text(text)])

        real_p = float(model.predict_proba(features)[0][1])
        label = LABELS[int(real_p >= 0.5)]
        confidence = round((real_p if label == "REAL" else 1 - real_p) * 100, 1)

        row = features.tocoo()
        contributions = row.data * self._word_weights[model_key][row.col]
        order = np.argsort(-np.abs(contributions))[:top_k]
        evidence = [(str(self._vocab[row.col[i]]), float(contributions[i])) for i in order]

        return Verdict(label, confidence, real_p, model_key, evidence)
