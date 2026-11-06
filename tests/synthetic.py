"""Datos SINTÉTICOS solo para pruebas automáticas (nunca se usan en la app).

Imitan el formato de ``data/raw/`` (mismo esquema que produce
``data.download_siata``) para poder probar el pipeline sin internet.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_raw(raw_dir: Path, n_stations: int = 6, start="2023-07-31", end="2024-07-31",
             seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(start, end, freq="D")
    doy = days.day_of_year.to_numpy()
    # Estacionalidad bimodal (picos en marzo y octubre) y capa límite anticorrelacionada
    season = 0.5 + 0.5 * np.cos(4 * np.pi * (doy - 75) / 365.25)
    blh = 900 - 400 * season + rng.normal(0, 80, len(days))
    precip = rng.gamma(0.6, 6, len(days))
    meteo = pd.DataFrame({
        "fecha": days.strftime("%Y-%m-%d"),
        "temp_c": 22 + rng.normal(0, 1.2, len(days)),
        "hr_pct": 70 + 10 * season + rng.normal(0, 4, len(days)),
        "precip_mm": precip.round(1),
        "viento_kmh": 6 + rng.normal(0, 1, len(days)),
        "capa_limite_m": blh,
    })
    meteo.to_csv(raw_dir / "meteo_daily.csv", index=False)

    stations, rows = [], []
    for k in range(n_stations):
        codigo = 10 + k * 3
        stations.append({"codigo": codigo, "nombre": f"Estacion Sintetica {k + 1}",
                         "nombre_corto": f"SIN-{k + 1:02d}", "municipio": "Medellin",
                         "latitud": 6.20 + 0.02 * k, "longitud": -75.58 + 0.01 * k})
        offset = rng.normal(0, 4)
        ar = 0.0
        daily = []
        for i in range(len(days)):
            ar = 0.7 * ar + rng.normal(0, 6)
            val = 18 + offset + 30 * season[i] + (700 - blh[i]) * 0.03 - 0.4 * precip[i] + ar
            daily.append(max(val, 2))
        hours = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
        diurnal = 1 + 0.25 * np.sin(2 * np.pi * (hours.hour.to_numpy() - 4) / 24)
        base = np.repeat(daily, 24)[: len(hours)] * diurnal
        vals = np.asarray(base + rng.normal(0, 3, len(hours)), dtype=float)
        calidad = np.ones(len(hours))
        miss = rng.random(len(hours)) < 0.06
        vals[miss] = -9999
        bad = rng.random(len(hours)) < 0.01
        calidad[bad] = 4.0
        for ts, v, q in zip(hours.strftime("%Y-%m-%d %H:%M:%S"), vals, calidad):
            rows.append((codigo, ts, round(float(v), 1), q))
    pd.DataFrame(stations).to_csv(raw_dir / "stations.csv", index=False)
    pd.DataFrame(rows, columns=["codigo", "fecha", "pm25", "calidad"]).to_csv(
        raw_dir / "pm25_hourly.csv.gz", index=False, compression="gzip")


if __name__ == "__main__":
    import sys
    make_raw(Path(sys.argv[1]))
