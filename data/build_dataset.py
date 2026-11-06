"""Construye el dataset procesado (Parquet) a partir de ``data/raw/`` (Plan B).

Uso:  python -m data.build_dataset
"""
from __future__ import annotations

import logging
import sys

import pandas as pd

import config
from data.clean import add_official_daily, attach_meteo, clean_hourly, to_daily

log = logging.getLogger("build")


def build() -> pd.DataFrame:
    if not config.RAW_PM25_HOURLY.exists():
        raise FileNotFoundError(
            f"Falta {config.RAW_PM25_HOURLY}. Ejecute primero: python -m data.download_siata")
    raw = pd.read_csv(config.RAW_PM25_HOURLY)
    hourly = clean_hourly(raw)
    daily = to_daily(hourly)
    official = (pd.read_csv(config.RAW_PM25_DAILY_RECENT)
                if config.RAW_PM25_DAILY_RECENT.exists() else None)
    daily = add_official_daily(daily, official)
    meteo = pd.read_csv(config.RAW_METEO) if config.RAW_METEO.exists() else None
    daily = attach_meteo(daily, meteo)

    stations = pd.read_csv(config.RAW_STATIONS)
    for col in ("nombre", "nombre_corto", "municipio"):
        stations[col] = stations[col].fillna("").astype(str).str.strip()
    stations = stations[stations["codigo"].isin(daily["codigo"].unique())]
    stations = stations.dropna(subset=["latitud", "longitud"])
    daily = daily[daily["codigo"].isin(stations["codigo"])]

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(config.PM25_DAILY_PARQUET, index=False)
    stations.to_parquet(config.STATIONS_PARQUET, index=False)

    log.info("Horas crudas: %d | horas válidas: %d (%.1f %%)", len(raw),
             hourly["pm25"].notna().sum(), 100 * hourly["pm25"].notna().mean())
    log.info("Días-estación válidos: %d | estaciones: %d | rango: %s a %s",
             len(daily), daily["codigo"].nunique(),
             daily["fecha"].min().date(), daily["fecha"].max().date())
    log.info("Días por fuente: %s", daily["fuente"].value_counts().to_dict())
    log.info("Distribución de clases: %s",
             daily["clase_nombre"].value_counts().to_dict())
    return daily


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        build()
    except FileNotFoundError as err:
        log.error(err)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
