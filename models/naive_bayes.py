"""Gaussian Naive Bayes implementado desde cero con NumPy.

Modelo
------
Para K clases y d atributos continuos, Naive Bayes supone independencia
condicional de los atributos dada la clase:

    P(y=k | x) ∝ P(y=k) · Π_j N(x_j ; μ_kj, σ²_kj)

Se trabaja en log-espacio para evitar underflow:

    log P(y=k | x) = log π_k − ½ Σ_j [ log(2π σ²_kj) + (x_j − μ_kj)² / σ²_kj ] − log Z(x)

con  log Z(x) = logsumexp_k(·)  (normalización estable numéricamente).

Estimación (máxima verosimilitud)
---------------------------------
    μ_kj  = media de x_j en la clase k
    σ²_kj = varianza (ddof=0) de x_j en la clase k  +  ε
    ε     = var_smoothing · max_j Var(x_j)     (evita varianzas nulas)
    π_k   ∝ (n_k + α)^λ                           (Laplace; λ=1 empírico, λ=0 uniforme)

El exponente λ ("prior_power") aplana los priors cuando las clases están muy
desbalanceadas: equivale a una regla de decisión de Bayes sensible al costo que
penaliza más no detectar las clases raras (Prevención/Crítico).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

LOG_2PI = np.log(2.0 * np.pi)


def logsumexp(a: np.ndarray, axis: int = 1) -> np.ndarray:
    """log Σ exp(a) estable: log Σ exp(a − m) + m."""
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m).squeeze(axis)


class GaussianNaiveBayes:
    def __init__(self, n_classes: int = 3, priors: str | np.ndarray = "empirical",
                 var_smoothing: float = 1e-9, alpha: float = 1.0, prior_power: float = 1.0):
        if not 0.0 <= prior_power <= 1.0:
            raise ValueError("prior_power debe estar en [0, 1]")
        self.n_classes = n_classes
        self.priors = priors
        self.prior_power = prior_power
        self.var_smoothing = var_smoothing
        self.alpha = alpha
        self.feature_names: list[str] = []
        self.class_count_: np.ndarray | None = None
        self.class_prior_: np.ndarray | None = None
        self.theta_: np.ndarray | None = None   # medias  (K, d)
        self.var_: np.ndarray | None = None     # varianzas (K, d)

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X debe ser (n, d) y y de longitud n")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contiene NaN/inf; limpie antes de entrenar")
        if y.min() < 0 or y.max() >= self.n_classes:
            raise ValueError(f"Etiquetas fuera de 0..{self.n_classes - 1}")
        K, d = self.n_classes, X.shape[1]
        self.feature_names = list(feature_names) if feature_names else [f"x{j}" for j in range(d)]
        eps = self.var_smoothing * float(np.max(np.var(X, axis=0)))
        global_mu, global_var = X.mean(axis=0), X.var(axis=0)

        self.theta_ = np.zeros((K, d))
        self.var_ = np.zeros((K, d))
        self.class_count_ = np.zeros(K)
        for k in range(K):
            Xk = X[y == k]
            self.class_count_[k] = len(Xk)
            if len(Xk) == 0:
                # Clase ausente: se usa la distribución global; su prior suavizado
                # (α) la mantiene posible pero improbable.
                self.theta_[k], self.var_[k] = global_mu, global_var
            else:
                self.theta_[k] = Xk.mean(axis=0)
                self.var_[k] = Xk.var(axis=0)
        self.var_ += eps

        if isinstance(self.priors, str) and self.priors == "uniform":
            self.class_prior_ = np.full(K, 1.0 / K)
        elif isinstance(self.priors, str) and self.priors == "empirical":
            w = (self.class_count_ + self.alpha) ** self.prior_power
            self.class_prior_ = w / w.sum()
        else:
            p = np.asarray(self.priors, dtype=np.float64)
            if p.shape != (K,) or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
                raise ValueError("priors debe ser 'empirical', 'uniform' o un vector que sume 1")
            self.class_prior_ = p
        return self

    # -------------------------------------------------------------- predict
    def _check_fitted(self):
        if self.theta_ is None:
            raise RuntimeError("El modelo no está entrenado")

    def joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """log π_k + Σ_j log N(x_j; μ_kj, σ²_kj)  ->  matriz (n, K)."""
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        # Broadcasting: (n,1,d) − (1,K,d) -> (n,K,d)
        diff2 = (X[:, None, :] - self.theta_[None, :, :]) ** 2
        log_gauss = -0.5 * (LOG_2PI + np.log(self.var_)[None] + diff2 / self.var_[None])
        with np.errstate(divide="ignore"):
            log_prior = np.log(self.class_prior_)
        return log_prior[None, :] + log_gauss.sum(axis=2)

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        jll = self.joint_log_likelihood(X)
        return jll - logsumexp(jll, axis=1)[:, None]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.joint_log_likelihood(X), axis=1)

    # ---------------------------------------------------------- persistencia
    def save(self, path: Path) -> None:
        self._check_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, theta=self.theta_, var=self.var_, prior=self.class_prior_,
                 count=self.class_count_, feature_names=np.array(self.feature_names),
                 var_smoothing=self.var_smoothing, alpha=self.alpha)

    @classmethod
    def load(cls, path: Path) -> "GaussianNaiveBayes":
        z = np.load(path, allow_pickle=False)
        m = cls(n_classes=z["theta"].shape[0], priors=z["prior"],
                var_smoothing=float(z["var_smoothing"]), alpha=float(z["alpha"]))
        m.theta_, m.var_, m.class_prior_ = z["theta"], z["var"], z["prior"]
        m.class_count_ = z["count"]
        m.feature_names = [str(s) for s in z["feature_names"]]
        return m
