"""Ingeniería de atributos para predecir la clase de PM2.5 del día t+1.

Todos los atributos usan información disponible al cierre del día t (sin fuga
de información del futuro). Se aplica log1p al PM2.5 porque su distribución es
asimétrica (aprox. log-normal) y el modelo asume normalidad por clase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PM25_FEATURES = ["log_pm25_t", "log_pm25_t1", "log_pm25_t2", "log_media3", "delta_1d"]
CALENDAR_FEATURES = ["sin_anual", "cos_anual", "sin_semestral", "cos_semestral"]
METEO_FEATURES = ["log_precip_t", "capa_limite_t", "viento_t", "hr_t"]
METEO_SOURCE = {"log_precip_t": "precip_mm", "capa_limite_t": "capa_limite_m",
                "viento_t": "viento_kmh", "hr_t": "hr_pct"}

FEATURE_SETS = {
    "solo_pm25": PM25_FEATURES,
    "pm25+calendario": PM25_FEATURES + CALENDAR_FEATURES,
    "completo": PM25_FEATURES + CALENDAR_FEATURES + METEO_FEATURES,
}
# (la ablación usa available_feature_set para "completo" cuando falta alguna variable)


def available_feature_set(daily: pd.DataFrame, min_coverage: float = 0.8) -> list[str]:
    """Incluye cada variable meteorológica solo si cubre >= 80 % de los días.

    (p. ej. ERA5 no trae capa límite para ene–jun 2024: esa variable se omite,
    pero lluvia, viento y humedad se conservan).
    """
    meteo = [f for f, col in METEO_SOURCE.items()
             if col in daily.columns and daily[col].notna().mean() >= min_coverage]
    return FEATURE_SETS["pm25+calendario"] + meteo


def _station_frame(g: pd.DataFrame) -> pd.DataFrame:
    """Reindexa a calendario diario continuo para que los rezagos sean días reales."""
    g = g.set_index("fecha").sort_index()
    full = pd.date_range(g.index.min(), g.index.max(), freq="D")
    g = g.reindex(full)
    g.index.name = "fecha"
    out = pd.DataFrame(index=g.index)
    pm = g["pm25"]
    out["pm25_t"] = pm
    out["clase_t"] = g["clase"]
    out["log_pm25_t"] = np.log1p(pm)
    out["log_pm25_t1"] = np.log1p(pm.shift(1))
    out["log_pm25_t2"] = np.log1p(pm.shift(2))
    out["log_media3"] = np.log1p(pm.rolling(3, min_periods=3).mean())
    out["delta_1d"] = pm - pm.shift(1)
    doy = out.index.day_of_year.to_numpy()
    out["sin_anual"] = np.sin(2 * np.pi * doy / 365.25)
    out["cos_anual"] = np.cos(2 * np.pi * doy / 365.25)
    # Armónico semestral: el valle tiene dos temporadas críticas (feb–abr y oct–nov)
    out["sin_semestral"] = np.sin(4 * np.pi * doy / 365.25)
    out["cos_semestral"] = np.cos(4 * np.pi * doy / 365.25)
    for feat, col in METEO_SOURCE.items():
        if col in g.columns:
            v = g[col]
            out[feat] = np.log1p(v.clip(lower=0)) if feat == "log_precip_t" else v
    out["y"] = g["clase"].shift(-1)           # objetivo: clase del día t+1
    out["pm25_t_mas_1"] = pm.shift(-1)
    return out.reset_index()


def build_features(daily: pd.DataFrame, features: list[str],
                   require_target: bool = True) -> pd.DataFrame:
    """Devuelve filas (estación, día t) con atributos completos y, si aplica, objetivo."""
    frames = []
    for codigo, g in daily.groupby("codigo"):
        f = _station_frame(g)
        f.insert(0, "codigo", codigo)
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Atributos no disponibles en el dataset: {missing}")
    needed = features + (["y"] if require_target else [])
    df = df.dropna(subset=needed + ["clase_t"]).reset_index(drop=True)
    if require_target:
        df["y"] = df["y"].astype(np.int64)
    df["clase_t"] = df["clase_t"].astype(np.int64)
    return df


def latest_feature_row(daily: pd.DataFrame, codigo: int, features: list[str]) -> pd.Series | None:
    """Última fila con atributos completos de una estación (para predecir mañana)."""
    sub = daily[daily["codigo"] == codigo]
    if sub.empty:
        return None
    df = build_features(sub, features, require_target=False)
    return None if df.empty else df.iloc[-1]
