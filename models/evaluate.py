"""Métricas y pruebas estadísticas.

scikit-learn se usa EXCLUSIVAMENTE para métricas (permitido por el enunciado).
Bootstrap y McNemar se implementan con NumPy / math.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

import config

LABELS = list(range(len(config.CLASS_NAMES)))


def present_labels(y_true) -> list[int]:
    """Clases con soporte > 0. Los promedios macro se calculan sobre ellas: si una
    clase no aparece en la prueba (p. ej. ningún día Crítico en temporada de
    lluvias), incluirla sumaría un 0 artificial que penaliza a todos los modelos."""
    return sorted(set(np.asarray(y_true).tolist()))


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    kw = dict(labels=LABELS, zero_division=0)
    macro = dict(labels=present_labels(y_true), zero_division=0, average="macro")
    return {
        "n": int(len(y_true)),
        "clases_evaluadas": [config.CLASS_NAMES[c] for c in present_labels(y_true)],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, **macro)),
        "precision_macro": float(precision_score(y_true, y_pred, **macro)),
        "recall_macro": float(recall_score(y_true, y_pred, **macro)),
        "f1_por_clase": dict(zip(config.CLASS_NAMES,
                                 map(float, f1_score(y_true, y_pred, average=None, **kw)))),
        "recall_por_clase": dict(zip(config.CLASS_NAMES,
                                     map(float, recall_score(y_true, y_pred, average=None, **kw)))),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "soporte_por_clase": dict(zip(config.CLASS_NAMES,
                                      map(int, np.bincount(y_true, minlength=len(LABELS))))),
        "binaria_38": binary_metrics(y_true, y_pred),
    }


def binary_metrics(y_true, y_pred) -> dict:
    """Tarea operativa: ¿el día t+1 alcanza >= 38 µg/m³ (Prevención o Crítico)?

    Las 3 clases se conservan; esta vista colapsa {Prevención, Crítico} en "alerta".
    Es la decisión que importa para activar medidas y sí tiene ejemplos suficientes
    (Crítico tiene 1 solo día en los datos abiertos de SIATA).
    """
    t = np.asarray(y_true) >= 1
    p = np.asarray(y_pred) >= 1
    tp, fp, fn = int(np.sum(t & p)), int(np.sum(~t & p)), int(np.sum(t & ~p))
    return {
        "soporte_alerta": int(t.sum()),
        "precision": float(precision_score(t, p, zero_division=0)),
        "recall": float(recall_score(t, p, zero_division=0)),
        "f1": float(f1_score(t, p, zero_division=0)),
        "tp": tp, "fp": fp, "fn": fn,
    }


def bootstrap_ci_binary(y_true, y_pred, n_boot: int = config.BOOTSTRAP_SAMPLES,
                        alpha: float = 0.05, seed: int = config.RANDOM_SEED) -> tuple[float, float]:
    """IC percentil del F1 de la clase "alerta" (>= 38 µg/m³)."""
    t, p = np.asarray(y_true) >= 1, np.asarray(y_pred) >= 1
    rng = np.random.default_rng(seed)
    n, scores = len(t), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        tb, pb = t[idx], p[idx]
        tp = np.sum(tb & pb)
        denom = 2 * tp + np.sum(~tb & pb) + np.sum(tb & ~pb)
        scores[b] = 2 * tp / denom if denom else 0.0
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def fast_f1_macro(y_true, y_pred, labels: list[int] | None = None) -> float:
    """F1 macro vectorizado en NumPy (idéntico a sklearn; ver tests).

    Se usa en los bucles internos (selección de atributos, bootstrap), donde se
    evalúa miles de veces y sklearn sería el cuello de botella.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = present_labels(y_true) if labels is None else labels
    K = len(LABELS)
    cm = np.bincount(y_true * K + y_pred, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(axis=0) + cm.sum(axis=1)          # (tp+fp) + (tp+fn)
    f1 = np.divide(2 * tp, denom, out=np.zeros(K), where=denom > 0)
    return float(f1[labels].mean()) if labels else 0.0


def bootstrap_ci(y_true, y_pred, n_boot: int = config.BOOTSTRAP_SAMPLES,
                 alpha: float = 0.05, seed: int = config.RANDOM_SEED) -> tuple[float, float]:
    """IC percentil (1−α) del F1 macro remuestreando observaciones con reemplazo."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = present_labels(y_true)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        scores[b] = fast_f1_macro(y_true[idx], y_pred[idx], labels)
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Prueba exacta de McNemar (binomial bilateral) sobre aciertos/errores pareados.

    b = A acierta y B falla ; c = A falla y B acierta.  H0: b y c igual de probables.
    """
    y_true = np.asarray(y_true)
    a_ok, b_ok = np.asarray(pred_a) == y_true, np.asarray(pred_b) == y_true
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"b": b, "c": c, "p_value": float(min(1.0, p))}


def blocked_time_folds(dates: np.ndarray, k: int, purge_days: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """K-fold por bloques contiguos de fechas con purga (Bergmeir & Benítez, 2012).

    Se eliminan del entrenamiento las filas a menos de ``purge_days`` de cada
    bloque de prueba para que los rezagos/objetivo no filtren información.
    """
    dates = np.asarray(dates, dtype="datetime64[D]")
    uniq = np.unique(dates)
    blocks = np.array_split(uniq, k)
    folds = []
    gap = np.timedelta64(purge_days, "D")
    for blk in blocks:
        lo, hi = blk.min(), blk.max()
        test = (dates >= lo) & (dates <= hi)
        train = (dates < lo - gap) | (dates > hi + gap)
        folds.append((np.where(train)[0], np.where(test)[0]))
    return folds
