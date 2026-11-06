"""Acceso a datos con DuckDB: SQL directo sobre los Parquet, sin servidor.

Todas las consultas son parametrizadas (sin interpolar texto del usuario/LLM).
"""
from __future__ import annotations

import threading
import unicodedata
from functools import lru_cache

import duckdb
import pandas as pd

import config

_lock = threading.Lock()


@lru_cache(maxsize=1)
def _connection() -> duckdb.DuckDBPyConnection:
    for p in (config.PM25_DAILY_PARQUET, config.STATIONS_PARQUET):
        if not p.exists():
            raise FileNotFoundError(f"Falta {p}. Ejecute: python pipeline.py")
    con = duckdb.connect(database=":memory:")
    con.execute(f"CREATE VIEW pm25_daily AS SELECT * FROM read_parquet('{config.PM25_DAILY_PARQUET.as_posix()}')")
    con.execute(f"CREATE VIEW stations AS SELECT * FROM read_parquet('{config.STATIONS_PARQUET.as_posix()}')")
    return con


def reset_connection() -> None:
    """Invalida la conexión (p. ej. tras reconstruir el dataset o en pruebas)."""
    _connection.cache_clear()


def query(sql: str, params: list | None = None) -> pd.DataFrame:
    with _lock:  # Streamlit ejecuta en hilos; DuckDB comparte una conexión
        return _connection().execute(sql, params or []).df()


def stations() -> pd.DataFrame:
    return query("SELECT * FROM stations ORDER BY codigo")


def date_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    r = query("SELECT MIN(fecha) AS a, MAX(fecha) AS b FROM pm25_daily")
    return pd.Timestamp(r.at[0, "a"]), pd.Timestamp(r.at[0, "b"])


def series(codigos: list[int], start, end) -> pd.DataFrame:
    return query(
        """SELECT d.*, s.nombre FROM pm25_daily d JOIN stations s USING (codigo)
           WHERE list_contains(?, d.codigo) AND d.fecha BETWEEN ? AND ?
           ORDER BY d.codigo, d.fecha""",
        [list(map(int, codigos)), pd.Timestamp(start), pd.Timestamp(end)])


def station_summary(start, end) -> pd.DataFrame:
    return query(
        """SELECT s.codigo, s.nombre, s.municipio, s.latitud, s.longitud,
                  ROUND(AVG(d.pm25), 1) AS pm25_prom, ROUND(MAX(d.pm25), 1) AS pm25_max,
                  COUNT(d.pm25) AS dias,
                  SUM(CASE WHEN d.clase >= 1 THEN 1 ELSE 0 END) AS dias_sobre_37
           FROM stations s LEFT JOIN pm25_daily d
             ON d.codigo = s.codigo AND d.fecha BETWEEN ? AND ?
           GROUP BY ALL ORDER BY s.codigo""",
        [pd.Timestamp(start), pd.Timestamp(end)])


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return " ".join(t.lower().replace("-", " ").replace("_", " ").split())


def resolve_station(estacion: str | int) -> dict:
    """Encuentra una estación por código, nombre corto o parte del nombre.

    Lanza ``ValueError`` con las opciones válidas si no hay coincidencia única.
    """
    st = stations()
    key = str(estacion).strip()
    if key.isdigit() and int(key) in set(st["codigo"]):
        return st[st["codigo"] == int(key)].iloc[0].to_dict()
    nk = _norm(key)
    exact = st[(st["nombre"].map(_norm) == nk) | (st["nombre_corto"].fillna("").map(_norm) == nk)]
    if len(exact) == 1:
        return exact.iloc[0].to_dict()
    tokens = nk.split()
    hay = (st["nombre"].fillna("") + " " + st["nombre_corto"].fillna("") + " "
           + st["municipio"].fillna("")).map(_norm)
    partial = st[hay.map(lambda h: all(t in h for t in tokens))]
    if len(partial) == 1:
        return partial.iloc[0].to_dict()
    opciones = "; ".join(f"{r.codigo}={r.nombre}" for r in
                         (partial if len(partial) > 1 else st).itertuples())
    motivo = "es ambigua" if len(partial) > 1 else "no existe"
    raise ValueError(f"La estación '{estacion}' {motivo}. Opciones: {opciones}")


def pm25_on(codigo: int, fecha) -> pd.DataFrame:
    return query("SELECT * FROM pm25_daily WHERE codigo = ? AND fecha = ?",
                 [int(codigo), pd.Timestamp(fecha)])


def nearest_dates(codigo: int, fecha, n: int = 3) -> list[str]:
    r = query("""SELECT fecha FROM pm25_daily WHERE codigo = ?
                 ORDER BY ABS(DATEDIFF('day', fecha, ?)) LIMIT ?""",
              [int(codigo), pd.Timestamp(fecha), n])
    return [pd.Timestamp(x).date().isoformat() for x in r["fecha"]]


def history(codigo: int, until=None, days: int = 10) -> pd.DataFrame:
    until = pd.Timestamp(until) if until is not None else pd.Timestamp("2100-01-01")
    return query("""SELECT * FROM pm25_daily WHERE codigo = ? AND fecha <= ?
                    ORDER BY fecha DESC LIMIT ?""", [int(codigo), until, days]
                 ).sort_values("fecha").reset_index(drop=True)
