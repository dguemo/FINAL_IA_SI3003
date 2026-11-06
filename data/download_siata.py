"""Adquisición de datos reales: SIATA (PM2.5), geoportal SIATA/AMVA y meteorología.

Fuentes:
  1. Histórico horario PM2.5 de SIATA (EntregaData1, ~2023-07 a 2024-07).
  2. Geoportal SIATA/AMVA: catálogo de estaciones oficiales, últimas 72 h horarias y
     últimos 30 promedios diarios oficiales (cubren el período reciente).
  3. Open-Meteo (ERA5 + pronóstico): meteorología diaria del valle, incluida la
     altura de la capa límite, que controla la dispersión de contaminantes.

Plan B: todo se guarda en ``data/raw/`` (CSV versionados en git). Si una fuente
falla, se conserva lo que ya existe en disco y el pipeline continúa. El paso
``build_dataset`` lee SOLO de ``data/raw/``, así que la app funciona sin internet.

Uso:  python -m data.download_siata
      python -m data.download_siata --meteo-json historico.json reciente.json
          (importa JSON de Open-Meteo descargados a mano desde el navegador, p. ej. si
           la API devuelve 429 por límite diario; no descarga nada más)
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests

import config

log = logging.getLogger("download")

HOURLY_COLS = ["codigo", "fecha", "pm25", "calidad"]
DAILY_COLS = ["codigo", "fecha", "pm25"]
STATION_COLS = ["codigo", "nombre", "nombre_corto", "municipio", "latitud", "longitud"]
METEO_HOURLY = ["temperature_2m", "relative_humidity_2m", "precipitation",
                "wind_speed_10m", "boundary_layer_height"]


# ----------------------------------------------------------------------------
# HTTP con reintentos
# ----------------------------------------------------------------------------
def http_get_json(url: str, params: dict | None = None, attempts: int = 3) -> Any:
    """GET con reintentos y backoff exponencial ante errores transitorios."""
    last_error: Exception | None = None
    for i in range(attempts):
        try:
            resp = requests.get(url, params=params, timeout=config.HTTP_TIMEOUT_S,
                                headers={"User-Agent": config.HTTP_USER_AGENT})
            if resp.status_code == 429:  # límite de tasa: esperar más que en otros errores
                last_error = requests.HTTPError("HTTP 429")
                log.warning("Límite de tasa en %s. Espera %ss", url, 30 * (i + 1))
                time.sleep(30 * (i + 1))
                continue
            if resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, json.JSONDecodeError) as err:
            last_error = err
            wait = 2 ** i
            log.warning("Fallo %s (%s). Reintento en %ss", url, err, wait)
            time.sleep(wait)
    raise ConnectionError(f"No se pudo descargar {url}: {last_error}")


def _first(d: dict, *keys: str, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


# ----------------------------------------------------------------------------
# Parsers (separados de la red para poder probarlos)
# ----------------------------------------------------------------------------
def parse_historical(payload: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convierte el JSON EntregaData1 en (horario, estaciones)."""
    if not isinstance(payload, list) or not payload:
        raise ValueError("Formato inesperado: se esperaba una lista de estaciones")
    rows, stations = [], []
    for st in payload:
        codigo = int(_first(st, "codigoSerial", "codigo"))
        nombre = str(_first(st, "nombre", default=str(codigo)))
        nombre = re.sub(r"^_?OFF-", "", nombre).strip()  # prefijo de estaciones apagadas
        stations.append({
            "codigo": codigo,
            "nombre": nombre,
            "nombre_corto": _first(st, "nombreCorto", default=""),
            "municipio": _first(st, "municipio", default=""),
            "latitud": float(_first(st, "latitud", "lat", "latitude", default="nan")),
            "longitud": float(_first(st, "longitud", "lon", "longitude", default="nan")),
        })
        for d in st.get("datos", []):
            rows.append((codigo, str(d["fecha"]), float(d["valor"]),
                         float(d["calidad"]) if d.get("calidad") is not None else float("nan")))
    hourly = pd.DataFrame(rows, columns=HOURLY_COLS)
    return hourly, pd.DataFrame(stations, columns=STATION_COLS)


def parse_geoportal_stations(payload: dict) -> pd.DataFrame:
    """GeoJSON del geoportal -> catálogo de estaciones."""
    out = []
    for f in payload["features"]:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"][:2]
        out.append({
            "codigo": int(p["codigo"]),
            "nombre": _first(p, "nombreEstacion", "estacion", default=str(p["codigo"])),
            "nombre_corto": _first(p, "estacion", default=""),
            "municipio": _first(p, "Municipio", default=""),
            "latitud": float(lat),
            "longitud": float(lon),
        })
    return pd.DataFrame(out, columns=STATION_COLS)


def parse_series_72h(payload: dict, codigo: int) -> pd.DataFrame:
    """Serie de 72 h del geoportal. Solo trae la hora final; se reconstruye hacia atrás."""
    info = payload["info"]
    end = pd.Timestamp(info["Hora_fin"])
    values = info["PM25_72H"]
    n = len(values)
    rows = []
    for i, v in enumerate(values):
        ts = end - pd.Timedelta(hours=n - 1 - i)
        rows.append((codigo, ts.strftime("%Y-%m-%d %H:%M:%S"),
                     float(config.SIATA_MISSING if v is None else v), float("nan")))
    return pd.DataFrame(rows, columns=HOURLY_COLS)


def parse_series_30d(payload: dict, codigo: int) -> pd.DataFrame:
    """Últimos 30 promedios diarios oficiales (ya validados por SIATA)."""
    info = payload["info"]
    rows = [(codigo, str(d)[:10], float(v)) for d, v in zip(info["Dias"], info["PM25_Diario"], strict=True)
            if v is not None and float(v) >= 0]
    return pd.DataFrame(rows, columns=DAILY_COLS)


def parse_open_meteo(payload: dict) -> pd.DataFrame:
    """Meteorología horaria -> diaria (promedios; precipitación acumulada)."""
    h = pd.DataFrame(payload["hourly"])
    h["fecha"] = pd.to_datetime(h["time"]).dt.date
    daily = h.groupby("fecha").agg(
        temp_c=("temperature_2m", "mean"),
        hr_pct=("relative_humidity_2m", "mean"),
        precip_mm=("precipitation", "sum"),
        viento_kmh=("wind_speed_10m", "mean"),
        capa_limite_m=("boundary_layer_height", "mean"),
    ).reset_index()
    daily["fecha"] = daily["fecha"].astype(str)
    return daily


# ----------------------------------------------------------------------------
# Descargas
# ----------------------------------------------------------------------------
def download_historical() -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Descargando histórico SIATA (puede tardar, ~20 MB)...")
    return parse_historical(http_get_json(config.SIATA_HISTORICAL_URL))


def download_recent() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info("Descargando catálogo, últimas 72 h y últimos 30 días del geoportal SIATA/AMVA...")
    stations = parse_geoportal_stations(http_get_json(config.SIATA_STATIONS_URL))
    hourly, daily = [], []
    for codigo in stations["codigo"]:
        for url, parser, out, label in ((config.SIATA_SERIES_72H_URL, parse_series_72h, hourly, "72 h"),
                                         (config.SIATA_SERIES_30D_URL, parse_series_30d, daily, "30 d")):
            try:
                out.append(parser(http_get_json(url.format(codigo=codigo)), codigo))
            except (ConnectionError, KeyError, ValueError, TypeError) as err:
                log.warning("Estación %s sin serie %s: %s", codigo, label, err)
            time.sleep(0.3)  # cortesía con el servidor público
    hourly = pd.concat(hourly, ignore_index=True) if hourly else pd.DataFrame(columns=HOURLY_COLS)
    daily = pd.concat(daily, ignore_index=True) if daily else pd.DataFrame(columns=DAILY_COLS)
    return hourly, stations, daily


def contiguous_spans(dates: pd.Series, max_gap_days: int = 30) -> list[tuple[date, date]]:
    """Agrupa fechas en tramos continuos (para no pedir meteorología de los huecos)."""
    d = pd.Series(sorted(pd.to_datetime(pd.Series(dates).astype(str).str[:10]).unique()))
    if d.empty:
        return []
    breaks = d.diff().dt.days.fillna(0) > max_gap_days
    return [(g.min().date(), g.max().date()) for _, g in d.groupby(breaks.cumsum())]


def download_meteo(spans: list[tuple[date, date]]) -> pd.DataFrame:
    """ERA5 (archivo, ~5 días de rezago) por tramo + pronóstico con días pasados para lo reciente."""
    frames = []
    archive_limit = date.today() - timedelta(days=6)
    recent_from = date.today() - timedelta(days=90)
    for start, end in spans:
        a_end = min(end, archive_limit)
        if start <= a_end and a_end < recent_from:
            frames.append(parse_open_meteo(http_get_json(config.OPEN_METEO_ARCHIVE_URL, {
                **config.VALLEY_LOCATION, "hourly": ",".join(METEO_HOURLY),
                "start_date": start.isoformat(), "end_date": a_end.isoformat()})))
            time.sleep(2)
    if any(end >= recent_from for _, end in spans):
        frames.append(parse_open_meteo(http_get_json(config.OPEN_METEO_FORECAST_URL, {
            **config.VALLEY_LOCATION, "hourly": ",".join(METEO_HOURLY),
            "past_days": 92, "forecast_days": 1})))
    daily = pd.concat(frames, ignore_index=True)
    return daily.drop_duplicates("fecha", keep="first").sort_values("fecha")


# ----------------------------------------------------------------------------
# Fusión con el Plan B en disco
# ----------------------------------------------------------------------------
def _merge_save(new: pd.DataFrame, path, keys: list[str], **csv_kwargs) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
    merged = new.drop_duplicates(keys, keep="last").sort_values(keys)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False, **csv_kwargs)
    return merged


def _merge_stations(*frames: pd.DataFrame) -> pd.DataFrame:
    """Une catálogos; para cada campo prefiere el valor no vacío más reciente."""
    allst = pd.concat([f for f in frames if f is not None and len(f)], ignore_index=True)
    allst = allst.replace("", pd.NA)
    merged = allst.groupby("codigo", as_index=False).last()
    return merged[STATION_COLS]


def import_meteo_json(paths: list[str]) -> pd.DataFrame:
    frames = [parse_open_meteo(json.loads(open(p, encoding="utf-8").read())) for p in paths]
    daily = pd.concat(frames, ignore_index=True).drop_duplicates("fecha", keep="first")
    return _merge_save(daily, config.RAW_METEO, ["fecha"])


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--meteo-json", nargs="+", metavar="ARCHIVO",
                    help="Importa JSON horarios de Open-Meteo descargados a mano")
    args, _ = ap.parse_known_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.meteo_json:
        merged = import_meteo_json(args.meteo_json)
        log.info("Meteorología importada: %d días (%s a %s)", len(merged),
                 merged["fecha"].min(), merged["fecha"].max())
        return 0
    ok_sources = 0
    station_frames = []
    if config.RAW_STATIONS.exists():
        station_frames.append(pd.read_csv(config.RAW_STATIONS))

    for name, fn in [("histórico SIATA", download_historical),
                     ("geoportal reciente", download_recent)]:
        try:
            hourly, stations, *daily = fn()
            _merge_save(hourly, config.RAW_PM25_HOURLY, ["codigo", "fecha"], compression="gzip")
            if daily and len(daily[0]):
                _merge_save(daily[0], config.RAW_PM25_DAILY_RECENT, ["codigo", "fecha"])
            station_frames.append(stations)
            ok_sources += 1
            log.info("OK %s: %d filas horarias", name, len(hourly))
        except (ConnectionError, ValueError, KeyError) as err:
            log.error("Falló %s -> se usa Plan B en disco. Detalle: %s", name, err)

    if station_frames:
        st = _merge_stations(*station_frames)
        st.to_csv(config.RAW_STATIONS, index=False)

    if config.RAW_PM25_HOURLY.exists():
        fechas = pd.read_csv(config.RAW_PM25_HOURLY, usecols=["fecha"])["fecha"]
        if config.RAW_PM25_DAILY_RECENT.exists():
            fechas = pd.concat([fechas, pd.read_csv(config.RAW_PM25_DAILY_RECENT)["fecha"]])
        spans = contiguous_spans(fechas)
        log.info("Tramos con datos de PM2.5 para meteorología: %s", spans)
        try:
            meteo = download_meteo(spans)
            _merge_save(meteo, config.RAW_METEO, ["fecha"])
            ok_sources += 1
            log.info("OK meteorología: %d días", len(meteo))
        except (ConnectionError, ValueError, KeyError) as err:
            log.error("Falló Open-Meteo -> se usa Plan B en disco. Detalle: %s", err)

    if not config.RAW_PM25_HOURLY.exists():
        log.error("No hay datos en línea ni Plan B en %s. Ejecute con internet al menos "
                  "una vez y versione data/raw/.", config.RAW_DIR)
        return 1
    log.info("Fuentes en línea exitosas: %d/3. Datos en %s", ok_sources, config.RAW_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
