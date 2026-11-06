"""Líneas base contra las que se compara el Naive Bayes."""
from __future__ import annotations

import numpy as np


class PersistenceBaseline:
    """La clase de mañana es la clase de hoy (fuerte en series autocorrelacionadas)."""

    def fit(self, X=None, y=None):
        return self

    @staticmethod
    def predict(class_today: np.ndarray) -> np.ndarray:
        return np.asarray(class_today, dtype=np.int64)


class MajorityBaseline:
    """Predice siempre la clase más frecuente en entrenamiento."""

    def __init__(self):
        self.majority_: int | None = None

    def fit(self, X, y):
        self.majority_ = int(np.bincount(np.asarray(y, dtype=np.int64)).argmax())
        return self

    def predict(self, X) -> np.ndarray:
        if self.majority_ is None:
            raise RuntimeError("Baseline no entrenado")
        return np.full(len(X), self.majority_, dtype=np.int64)
