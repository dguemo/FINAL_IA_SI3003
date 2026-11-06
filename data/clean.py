"""Limpieza y agregación diaria de PM2.5.

Reglas (documentadas en el informe):
  * ``-9999`` -> faltante.
  * Bandera de calidad SIATA > 2.5 -> faltante (criterio de validación de SIATA).
  * Valores negativos o > 500 µg/m³ -> faltante (físicamente implausibles).
  * Duplicados (estación, hora) -> se conserva el último.
  * Un día es válido solo si tiene >= 18 horas válidas (75 %, representatividad
    temporal exigida para promedios de 24 h).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def clean_hourly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fecha"] = pd.to_datetime(out["fecha"], errors="coerce")
    out = out.dropna(subset=["fecha", "codigo"])
    out["codigo"] = out["codigo"].astype(int)
    pm = pd.to_numeric(out["pm25"], errors="coerce")
    calidad = pd.to_numeric(out.get("calidad"), errors="coerce")
    invalid = (
        (pm == config.SIATA_MISSING)
        | (pm < 0)
        | (pm > config.PM25_PHYSICAL_MAX)
        | (calidad > config.SIATA_MAX_QUALITY_FLAG)
    )
    out["pm25"] = pm.mask(invalid)
    out = out.drop_duplicates(["codigo", "fecha"], keep="last")
    return out[["codigo", "fecha", "pm25"]].sort_values(["codigo", "fecha"])


def to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    h = hourly.copy()
    h["dia"] = h["fecha"].dt.normalize()
    daily = (h.groupby(["codigo", "dia"])["pm25"]
               .agg(pm25="mean", horas_validas="count")
               .reset_index()
               .rename(columns={"dia": "fecha"}))
    daily = daily[daily["horas_validas"] >= config.MIN_VALID_HOURS_PER_DAY].copy()
    daily["pm25"] = daily["pm25"].round(2)
    daily["clase"] = daily["pm25"].map(config.pm25_to_class).astype(np.int8)
    daily["clase_nombre"] = daily["clase"].map(lambda c: config.CLASS_NAMES[c])
    return daily.reset_index(drop=True)


def add_official_daily(daily: pd.DataFrame, official: pd.DataFrame | None) -> pd.DataFrame:
    """Agrega los promedios diarios oficiales de SIATA (geoportal, últimos 30 días).

    Donde coinciden con un día calculado desde datos horarios, prevalece el oficial
    (validado por SIATA). ``horas_validas`` queda vacío porque SIATA no lo publica.
    """
    if official is None or official.empty:
        daily["fuente"] = "siata_horario"
        return daily
    o = official.copy()
    o["fecha"] = pd.to_datetime(o["fecha"])
    o["codigo"] = o["codigo"].astype(int)
    o["horas_validas"] = pd.NA
    o["fuente"] = "siata_diario_oficial"
    daily = daily.assign(fuente="siata_horario")
    allrows = pd.concat([daily[["codigo", "fecha", "pm25", "horas_validas", "fuente"]],
                         o[["codigo", "fecha", "pm25", "horas_validas", "fuente"]]], ignore_index=True)
    allrows = allrows.drop_duplicates(["codigo", "fecha"], keep="last")
    allrows["horas_validas"] = allrows["horas_validas"].astype("Int16")
    allrows["clase"] = allrows["pm25"].map(config.pm25_to_class).astype(np.int8)
    allrows["clase_nombre"] = allrows["clase"].map(lambda c: config.CLASS_NAMES[c])
    return allrows.sort_values(["codigo", "fecha"]).reset_index(drop=True)


def attach_meteo(daily: pd.DataFrame, meteo: pd.DataFrame | None) -> pd.DataFrame:
    if meteo is None or meteo.empty:
        return daily
    m = meteo.copy()
    m["fecha"] = pd.to_datetime(m["fecha"])
    return daily.merge(m, on="fecha", how="left")
