"""Herramientas invocables por el agente (tool calling) y su despachador.

Cada herramienta devuelve un dict serializable. ``execute_tool`` NUNCA lanza
excepciones: cualquier fallo se devuelve como ``{"error": ...}`` para que el
LLM lo reciba como observación y decida cómo seguir.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

import config
from data import db
from models.features import latest_feature_row
from models.naive_bayes import GaussianNaiveBayes
from rag import retriever

log = logging.getLogger("tools")

LATEST_WORDS = {"ultima", "última", "ultimo", "último", "latest", "reciente", "hoy"}


def _station_info(st: dict) -> dict:
    return {"codigo": int(st["codigo"]), "nombre": st["nombre"],
            "municipio": st.get("municipio") or ""}


def _parse_date(text: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(date.fromisoformat(str(text).strip()[:10]))
    except ValueError as err:
        raise ValueError(f"Fecha inválida '{text}'. Use el formato AAAA-MM-DD") from err


def _round(x, nd=1):
    return None if x is None or pd.isna(x) else round(float(x), nd)


# ----------------------------------------------------------------------------
# 1) get_pm25
# ----------------------------------------------------------------------------
def get_pm25(estacion: str, fecha: str, fecha_fin: str | None = None) -> dict:
    """PM2.5 promedio 24 h de una estación en una fecha (o resumen de un rango)."""
    st = db.resolve_station(estacion)
    codigo = int(st["codigo"])
    lo, hi = db.date_range()
    if str(fecha).strip().lower() in LATEST_WORDS:
        h = db.history(codigo, days=1)
        if h.empty:
            raise LookupError(f"La estación {st['nombre']} no tiene datos válidos")
        fecha = h["fecha"].iloc[-1].date().isoformat()
    d0 = _parse_date(fecha)

    if fecha_fin:
        d1 = _parse_date(fecha_fin)
        if d1 < d0:
            d0, d1 = d1, d0
        s = db.series([codigo], d0, d1)
        if s.empty:
            raise LookupError(f"Sin datos entre {d0.date()} y {d1.date()} para {st['nombre']}. "
                              f"Rango del dataset: {lo.date()} a {hi.date()}")
        counts = s["clase"].value_counts().reindex(range(3), fill_value=0)
        return {
            "estacion": _station_info(st), "desde": str(d0.date()), "hasta": str(d1.date()),
            "dias_con_dato": int(len(s)),
            "pm25_promedio": _round(s["pm25"].mean()),
            "pm25_maximo": _round(s["pm25"].max()),
            "fecha_maximo": str(s.loc[s["pm25"].idxmax(), "fecha"].date()),
            "dias_por_clase": {config.CLASS_NAMES[k]: int(v) for k, v in counts.items()},
            "unidad": "µg/m³ (promedio 24 h)",
        }

    row = db.pm25_on(codigo, d0)
    if row.empty:
        cerca = db.nearest_dates(codigo, d0)
        raise LookupError(f"No hay dato válido para {st['nombre']} el {d0.date()} (día faltante o con "
                          f"<{config.MIN_VALID_HOURS_PER_DAY} h válidas). Fechas cercanas con dato: "
                          f"{cerca}. Rango del dataset: {lo.date()} a {hi.date()}")
    r = row.iloc[0]
    out = {
        "estacion": _station_info(st), "fecha": str(d0.date()),
        "pm25_promedio_24h": _round(r["pm25"], 2), "unidad": "µg/m³",
        "horas_validas": None if pd.isna(r["horas_validas"]) else int(r["horas_validas"]),
        "fuente_dato": ("promedio diario oficial SIATA" if r.get("fuente") == "siata_diario_oficial"
                        else "promedio de datos horarios SIATA"),
        "clase": config.CLASS_NAMES[int(r["clase"])],
        "umbrales_clases": "Normal 0–37, Prevención 38–55, Crítico ≥56 µg/m³",
    }
    meteo = {k: _round(r[k]) for k in ("precip_mm", "capa_limite_m", "viento_kmh", "hr_pct", "temp_c")
             if k in r.index}
    if meteo:
        out["meteorologia_del_dia"] = meteo
    return out


# ----------------------------------------------------------------------------
# 2) predict_tomorrow
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _model() -> GaussianNaiveBayes:
    if not config.GNB_PARAMS.exists():
        raise FileNotFoundError("Modelo no entrenado. Ejecute: python -m models.train")
    return GaussianNaiveBayes.load(config.GNB_PARAMS)


def reset_caches() -> None:
    _model.cache_clear()


def predict_tomorrow(estacion: str) -> dict:
    """Clase de PM2.5 para el día siguiente al último dato disponible (Naive Bayes propio)."""
    st = db.resolve_station(estacion)
    codigo = int(st["codigo"])
    model = _model()
    hist = db.history(codigo, days=15)
    row = latest_feature_row(hist, codigo, model.feature_names)
    if row is None:
        raise LookupError(f"No hay 3 días consecutivos recientes con datos válidos en "
                          f"{st['nombre']} para construir los atributos del modelo")
    x = row[model.feature_names].to_numpy(dtype=float)[None, :]
    proba = model.predict_proba(x)[0]
    k = int(np.argmax(proba))
    base = pd.Timestamp(row["fecha"])
    target = base + pd.Timedelta(days=1)
    out = {
        "estacion": _station_info(st),
        "modelo": ("Gaussian Naive Bayes implementado por el equipo del proyecto (académico; "
                   "NO es un pronóstico oficial de SIATA)"),
        "fecha_base_ultimo_dato": str(base.date()),
        "fecha_pronosticada": str(target.date()),
        "pm25_ultimo_dia": _round(row["pm25_t"], 2),
        "clase_ultimo_dia": config.CLASS_NAMES[int(row["clase_t"])],
        "clase_predicha": config.CLASS_NAMES[k],
        "probabilidades": {c: round(float(p), 3) for c, p in zip(config.CLASS_NAMES, proba)},
        "prediccion_persistencia": config.CLASS_NAMES[int(row["clase_t"])],
        "atributos_usados": {f: _round(row[f], 3) for f in model.feature_names},
    }
    today = pd.Timestamp(date.today())
    lag = (today - base).days
    if lag > 2:
        out["advertencia"] = (f"El último dato disponible es de hace {lag} días; la predicción "
                              f"corresponde al {target.date()}, no a mañana calendario.")
    elif target <= today:
        out["nota"] = (f"SIATA publica el promedio diario con un día de retraso: la fecha "
                       f"pronosticada ({target.date()}) es hoy, no mañana calendario.")
    return out


# ----------------------------------------------------------------------------
# 3) search_docs
# ----------------------------------------------------------------------------
def search_docs(query: str, k: int = config.RAG_TOP_K) -> dict:
    """Busca fragmentos de la normativa (Res. 2254/2017, POECA) con fuente y página."""
    results = retriever.search(query, int(k))
    for r in results:
        r["texto"] = r["texto"][:900]
        r.pop("archivo", None)
    return {"consulta": query, "resultados": results,
            "instruccion_cita": "Cite como [Fuente, p. N] y use comillas para texto literal."}


# ----------------------------------------------------------------------------
# Esquemas JSON (formato OpenAI tools) y despachador
# ----------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_pm25",
        "description": ("Consulta el PM2.5 promedio 24 h medido (histórico) de una estación del Valle "
                        "de Aburrá en una fecha, o un resumen si se da fecha_fin. Úsela para cualquier "
                        "dato medido. fecha='ultima' devuelve el día más reciente disponible."),
        "parameters": {"type": "object", "properties": {
            "estacion": {"type": "string", "description": "Código numérico o nombre de la estación"},
            "fecha": {"type": "string", "description": "AAAA-MM-DD o 'ultima'"},
            "fecha_fin": {"type": "string", "description": "Opcional, AAAA-MM-DD para resumen de rango"},
        }, "required": ["estacion", "fecha"]}}},
    {"type": "function", "function": {
        "name": "predict_tomorrow",
        "description": ("Predice con el modelo Naive Bayes del equipo la clase de PM2.5 (Normal/"
                        "Prevención/Crítico) del día siguiente al último dato de la estación, con "
                        "probabilidades. Úsela para toda pregunta de pronóstico."),
        "parameters": {"type": "object", "properties": {
            "estacion": {"type": "string", "description": "Código numérico o nombre de la estación"},
        }, "required": ["estacion"]}}},
    {"type": "function", "function": {
        "name": "search_docs",
        "description": ("Búsqueda semántica en la normativa oficial (Resolución 2254 de 2017 y POECA "
                        "del Valle de Aburrá). Devuelve fragmentos con fuente y página para citar."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Consulta en español"},
            "k": {"type": "integer", "description": "Número de fragmentos (1–8)", "default": 4},
        }, "required": ["query"]}}},
]

TOOLS = {"get_pm25": get_pm25, "predict_tomorrow": predict_tomorrow, "search_docs": search_docs}


def execute_tool(name: str, raw_args: str | dict) -> dict:
    """Ejecuta una herramienta y devuelve SIEMPRE un dict (resultado o {'error': ...})."""
    if name not in TOOLS:
        return {"error": f"Herramienta desconocida '{name}'. Disponibles: {list(TOOLS)}"}
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
        if not isinstance(args, dict):
            raise TypeError("los argumentos deben ser un objeto JSON")
    except (json.JSONDecodeError, TypeError) as err:
        return {"error": f"Argumentos JSON inválidos para {name}: {err}"}
    try:
        return TOOLS[name](**args)
    except TypeError as err:
        return {"error": f"Parámetros incorrectos para {name}: {err}"}
    except (ValueError, LookupError, FileNotFoundError) as err:
        return {"error": str(err)}
    except Exception as err:  # noqa: BLE001  — el loop nunca debe caerse por una herramienta
        log.exception("Fallo inesperado en %s", name)
        return {"error": f"Fallo interno en {name}: {type(err).__name__}: {err}"}
