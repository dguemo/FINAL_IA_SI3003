"""Entrenamiento y evaluación rigurosa del Gaussian Naive Bayes.

Protocolo
---------
1. Holdout temporal: último 25 % de fechas como prueba (simula pronóstico real),
   con purga de 3 días entre entrenamiento y prueba.
2. Validación cruzada por bloques temporales (5 bloques, purga de 3 días) sobre
   todo el período, para que ambas temporadas críticas aparezcan en prueba.
   Los hiperparámetros se eligen con CV interna sólo sobre la parte de
   entrenamiento de cada bloque (CV anidada: sin fuga hacia la prueba).
   Dentro de esa CV interna se hace selección hacia adelante de atributos.
3. Comparación con Persistencia y Clase Mayoritaria; IC bootstrap 95 % del F1
   macro y prueba exacta de McNemar NB vs Persistencia.
4. Ablación de atributos (solo PM2.5 / + calendario / + meteorología).
5. Modelo final: hiperparámetros elegidos por CV en todos los datos, entrenado
   con todos los datos y guardado en ``models/artifacts/gnb_params.npz``.

Uso:  python -m models.train
"""
from __future__ import annotations

import itertools
import json
import logging
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import config
from models.baselines import MajorityBaseline, PersistenceBaseline
from models.evaluate import (blocked_time_folds, bootstrap_ci, bootstrap_ci_binary, compute_metrics,
                             fast_f1_macro, mcnemar_exact)
from models.features import FEATURE_SETS, available_feature_set, build_features
from models.naive_bayes import GaussianNaiveBayes

log = logging.getLogger("train")

# λ (prior_power): 1 = priors empíricos ... 0 = uniformes (ver models/naive_bayes.py)
HP_GRID = [{"priors": "empirical", "prior_power": lam, "var_smoothing": v}
           for lam, v in itertools.product([1.0, 0.75, 0.5, 0.25, 0.0], [1e-9, 1e-3])]


def _f1(y, p) -> float:
    return compute_metrics(y, p)["f1_macro"]


def _cv_score(X, y, folds, cols: list[int], hp: dict) -> float:
    """F1 macro AGREGADO fuera de pliegue (no el promedio por pliegue).

    Con clases raras, muchos pliegues no tienen días de alerta: ahí "predecir siempre
    Normal" obtiene F1=1.0 y el promedio por pliegue premia no alertar nunca. Juntar
    las predicciones de todos los pliegues evita ese sesgo (Forman & Scholz, 2010).
    """
    oof = np.full(len(y), -1)
    for tr, te in folds:
        if len(tr) == 0 or len(te) == 0:
            continue
        m = GaussianNaiveBayes(**hp).fit(X[tr][:, cols], y[tr])
        oof[te] = m.predict(X[te][:, cols])
    mask = oof >= 0
    return fast_f1_macro(y[mask], oof[mask]) if mask.any() else -1.0


def forward_selection(X, y, folds, hp: dict, tol: float = 1e-3) -> list[int]:
    """Selección hacia adelante: agrega el atributo que más sube el F1 de CV.

    Motivo: NB asume independencia condicional; los rezagos de PM2.5 están muy
    correlacionados y, si entran todos, el modelo "cuenta varias veces" la misma
    evidencia. La selección por CV elige un subconjunto menos redundante.
    """
    selected, best = [], -1.0
    remaining = list(range(X.shape[1]))
    while remaining:
        score, j = max((_cv_score(X, y, folds, selected + [j], hp), j) for j in remaining)
        if score <= best + tol:
            break
        selected.append(j)
        remaining.remove(j)
        best = score
    return selected


def select_model(X, y, dates) -> dict:
    """Para cada hiperparámetro hace selección hacia adelante; gana el mejor (CV por bloques)."""
    folds = blocked_time_folds(dates, config.CV_FOLDS, config.CV_PURGE_DAYS)
    best, best_score = None, -np.inf
    for hp in HP_GRID:
        cols = forward_selection(X, y, folds, hp)
        score = _cv_score(X, y, folds, cols, hp)
        if score > best_score:
            best, best_score = {"cols": cols, **hp}, score
    return best


def run_models(Xtr, ytr, dtr, Xte, yte, ct_te, tune: bool = True) -> dict[str, np.ndarray]:
    sel = select_model(Xtr, ytr, dtr) if tune else {"cols": list(range(Xtr.shape[1])), **HP_GRID[0]}
    cols, hp = sel["cols"], {k: v for k, v in sel.items() if k != "cols"}
    nb = GaussianNaiveBayes(**hp).fit(Xtr[:, cols], ytr)
    return {
        "naive_bayes": nb.predict(Xte[:, cols]),
        "persistencia": PersistenceBaseline().predict(ct_te),
        "clase_mayoritaria": MajorityBaseline().fit(Xtr, ytr).predict(Xte),
    }, sel


def holdout(df: pd.DataFrame, feats: list[str]) -> dict:
    dates = df["fecha"].values.astype("datetime64[D]")
    uniq = np.unique(dates)
    cutoff = uniq[int(len(uniq) * (1 - config.TEST_FRACTION))]
    gap = np.timedelta64(config.CV_PURGE_DAYS + 1, "D")  # +1: el objetivo es t+1
    tr = dates < cutoff - gap
    te = dates >= cutoff
    X, y, ct = df[feats].to_numpy(float), df["y"].to_numpy(), df["clase_t"].to_numpy()
    preds, sel = run_models(X[tr], y[tr], dates[tr], X[te], y[te], ct[te])
    out = {"cutoff": str(cutoff), "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "seleccion": {"atributos": [feats[c] for c in sel["cols"]],
                         "prior_power": sel["prior_power"], "var_smoothing": sel["var_smoothing"]},
           "modelos": {}}
    for name, p in preds.items():
        m = compute_metrics(y[te], p)
        m["f1_macro_ic95"] = bootstrap_ci(y[te], p)
        m["binaria_38"]["f1_ic95"] = bootstrap_ci_binary(y[te], p)
        out["modelos"][name] = m
    out["mcnemar_nb_vs_persistencia"] = mcnemar_exact(
        y[te], preds["naive_bayes"], preds["persistencia"])
    return out


def cross_validation(df: pd.DataFrame, feats: list[str], tune: bool = True) -> dict:
    dates = df["fecha"].values.astype("datetime64[D]")
    X, y, ct = df[feats].to_numpy(float), df["y"].to_numpy(), df["clase_t"].to_numpy()
    folds = blocked_time_folds(dates, config.CV_FOLDS, config.CV_PURGE_DAYS)
    per_fold = {k: [] for k in ("naive_bayes", "persistencia", "clase_mayoritaria")}
    oof = {k: np.full(len(y), -1) for k in per_fold}
    for tr, te in folds:
        preds, _ = run_models(X[tr], y[tr], dates[tr], X[te], y[te], ct[te], tune=tune)
        for name, p in preds.items():
            per_fold[name].append(_f1(y[te], p))
            oof[name][te] = p
    out = {"folds": config.CV_FOLDS, "purga_dias": config.CV_PURGE_DAYS, "modelos": {}}
    for name in per_fold:
        m = compute_metrics(y, oof[name])
        m["f1_macro_folds"] = per_fold[name]
        m["f1_macro_media"] = float(np.mean(per_fold[name]))
        m["f1_macro_std"] = float(np.std(per_fold[name]))
        m["f1_macro_ic95"] = bootstrap_ci(y, oof[name])
        m["binaria_38"]["f1_ic95"] = bootstrap_ci_binary(y, oof[name])
        out["modelos"][name] = m
    out["mcnemar_nb_vs_persistencia"] = mcnemar_exact(y, oof["naive_bayes"], oof["persistencia"])
    return out


def ablation(daily: pd.DataFrame) -> dict:
    res = {}
    sets = dict(FEATURE_SETS)
    sets["completo"] = available_feature_set(daily)  # solo variables meteorológicas disponibles
    for name, feats in sets.items():
        try:
            df = build_features(daily, feats)
        except ValueError:
            continue  # p. ej. sin meteorología
        cv = cross_validation(df, feats, tune=False)
        nb = cv["modelos"]["naive_bayes"]
        res[name] = {"n": nb["n"], "f1_macro_media": nb["f1_macro_media"],
                     "f1_macro_std": nb["f1_macro_std"], "accuracy": nb["accuracy"],
                     "f1_alerta_38": nb["binaria_38"]["f1"]}
    return res


def write_report(metrics: dict) -> None:
    lines = ["# Reporte del modelo clásico (generado por `models/train.py`)", "",
             f"Generado: {metrics['generado']}  ", f"Atributos: `{', '.join(metrics['features'])}`  ",
             f"Filas: {metrics['dataset']['filas']} | estaciones: {metrics['dataset']['estaciones']} | "
             f"rango: {metrics['dataset']['desde']} a {metrics['dataset']['hasta']}", ""]
    for section, title in (("cv", "Validación cruzada por bloques temporales (fuera de pliegue)"),
                           ("holdout", "Holdout temporal (último 25 % de fechas; pocos días en alerta, ver soporte)")):
        s = metrics[section]
        lines += [f"## {title}", "",
                  "### 3 clases (Normal / Prevención / Crítico)", "",
                  "| Modelo | F1 macro | IC95 % F1 | Accuracy | Precision | Recall |",
                  "|---|---|---|---|---|---|"]
        for name, m in s["modelos"].items():
            lo, hi = m["f1_macro_ic95"]
            lines.append(f"| {name} | {m['f1_macro']:.3f} | [{lo:.3f}, {hi:.3f}] | {m['accuracy']:.3f} | "
                         f"{m['precision_macro']:.3f} | {m['recall_macro']:.3f} |")
        lines += ["", "### Tarea operativa: alerta (≥ 38 µg/m³) al día siguiente", "",
                  "| Modelo | F1 alerta | IC95 % | Precision | Recall | TP | FP | FN |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, m in s["modelos"].items():
            b = m["binaria_38"]
            lo, hi = b["f1_ic95"]
            lines.append(f"| {name} | {b['f1']:.3f} | [{lo:.3f}, {hi:.3f}] | {b['precision']:.3f} | "
                         f"{b['recall']:.3f} | {b['tp']} | {b['fp']} | {b['fn']} |")
        mc = s["mcnemar_nb_vs_persistencia"]
        lines += ["", f"McNemar NB vs Persistencia: b={mc['b']}, c={mc['c']}, p={mc['p_value']:.4f}", "",
                  "Matriz de confusión Naive Bayes (filas=real, columnas=predicho; "
                  "Normal, Prevención, Crítico):", "",
                  "```", *[str(r) for r in s["modelos"]["naive_bayes"]["confusion_matrix"]], "```",
                  f"Soporte: {s['modelos']['naive_bayes']['soporte_por_clase']}", ""]
        if section == "cv":
            nb = s["modelos"]["naive_bayes"]
            lines += [f"F1 macro por pliegue NB: {[round(x, 3) for x in nb['f1_macro_folds']]} "
                      f"(media {nb['f1_macro_media']:.3f} ± {nb['f1_macro_std']:.3f})", ""]
    lines += ["## Ablación de atributos (CV por bloques, hiperparámetros por defecto)", "",
              "| Conjunto | F1 macro (media ± std) | F1 alerta ≥38 | Accuracy |", "|---|---|---|---|"]
    for name, a in metrics["ablacion"].items():
        lines.append(f"| {name} | {a['f1_macro_media']:.3f} ± {a['f1_macro_std']:.3f} | "
                     f"{a['f1_alerta_38']:.3f} | {a['accuracy']:.3f} |")
    mf = metrics["modelo_final"]
    lines += ["", f"Modelo final — atributos seleccionados: `{', '.join(mf['atributos'])}`; "
                  f"hiperparámetros: `{mf['hparams']}`", "",
              "Nota: los promedios macro se calculan sobre las clases presentes en la "
              "prueba (`clases_evaluadas` en metrics.json)."]
    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.EVAL_RESULTS_DIR / "model_report.md").write_text("\n".join(lines), encoding="utf-8")


def train() -> dict:
    if not config.PM25_DAILY_PARQUET.exists():
        raise FileNotFoundError("Falta el dataset procesado. Ejecute: python -m data.build_dataset")
    daily = pd.read_parquet(config.PM25_DAILY_PARQUET)
    feats = available_feature_set(daily)
    df = build_features(daily, feats)
    log.info("Filas de entrenamiento: %d | atributos: %s", len(df), feats)

    metrics = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "features": feats,
        "dataset": {"filas": int(len(df)), "estaciones": int(df["codigo"].nunique()),
                    "desde": str(df["fecha"].min().date()), "hasta": str(df["fecha"].max().date()),
                    "distribucion_y": dict(zip(config.CLASS_NAMES,
                                               map(int, np.bincount(df["y"], minlength=3))))},
        "holdout": holdout(df, feats),
        "cv": cross_validation(df, feats),
        "ablacion": ablation(daily),
    }
    X, y = df[feats].to_numpy(float), df["y"].to_numpy()
    sel = select_model(X, y, df["fecha"].values.astype("datetime64[D]"))
    cols = sel.pop("cols")
    chosen = [feats[c] for c in cols]
    final = GaussianNaiveBayes(**sel).fit(X[:, cols], y, feature_names=chosen)
    final.save(config.GNB_PARAMS)
    metrics["modelo_final"] = {"hparams": sel, "atributos": chosen, "entrenado_con": int(len(y)),
                               "priors": final.class_prior_.tolist()}
    config.MODEL_METRICS.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(metrics)

    for sec in ("cv", "holdout"):
        for name, m in metrics[sec]["modelos"].items():
            log.info("[%s] %-18s F1 macro=%.3f  acc=%.3f", sec, name, m["f1_macro"], m["accuracy"])
    log.info("Modelo guardado en %s", config.GNB_PARAMS)
    return metrics


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        train()
    except FileNotFoundError as err:
        log.error(err)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
