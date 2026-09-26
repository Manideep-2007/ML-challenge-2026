"""
Probability calibration fitted on the internal dev split (never on the Stage 2
validation set). Kept only if it improves validation macro F0.5.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class Calibrator:
    def __init__(self, method: str):
        self.method = method
        self.model = None

    def fit(self, probability: np.ndarray, label: np.ndarray, weight: np.ndarray | None = None):
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.model.fit(probability, label, sample_weight=weight)
        elif self.method == "platt":
            self.model = LogisticRegression()
            self.model.fit(_logit(probability).reshape(-1, 1), label, sample_weight=weight)
        else:
            raise ValueError(self.method)
        return self

    def transform(self, probability: np.ndarray) -> np.ndarray:
        if self.method == "isotonic":
            return self.model.predict(probability).astype(np.float32)
        return self.model.predict_proba(_logit(probability).reshape(-1, 1))[:, 1].astype(np.float32)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))
