"""Configuración central del proyecto: rutas, umbrales y parámetros.

Todo módulo importa sus constantes desde aquí para que no haya números mágicos
repartidos por el código.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# ----------------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_PM25_HOURLY = RAW_DIR / "pm25_hourly.csv.gz"  # Plan B (versionado en git)
RAW_STATIONS = RAW_DIR / "stations.csv"           # Plan B (versionado en git)
RAW_METEO = RAW_DIR / "meteo_daily.csv"           # Plan B (versionado en git)
RAW_PM25_DAILY_RECENT = RAW_DIR / "pm25_daily_recent.csv"  # Plan B: promedios oficiales SIATA

PM25_DAILY_PARQUET = PROCESSED_DIR / "pm25_daily.parquet"
STATIONS_PARQUET = PROCESSED_DIR / "stations.parquet"

MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
GNB_PARAMS = ARTIFACTS_DIR / "gnb_params.npz"
MODEL_METRICS = ARTIFACTS_DIR / "metrics.json"

RAG_DIR = ROOT / "rag"
DOCS_DIR = RAG_DIR / "docs"
CHROMA_DIR = RAG_DIR / "chroma_db"
CHROMA_COLLECTION = "normativa_aire"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

EVAL_DIR = ROOT / "evaluation"
EVAL_RESULTS_DIR = EVAL_DIR / "results"

# ----------------------------------------------------------------------------
# Fuentes de datos (SIATA / AMVA / Open-Meteo)
# ----------------------------------------------------------------------------
SIATA_HISTORICAL_URL = "https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_pm25.json"
SIATA_GEOPORTAL = "https://geoportal.siata.gov.co/fastgeoapi/geodata"
SIATA_STATIONS_URL = f"{SIATA_GEOPORTAL}/geodataJson/1/pm25_minio"
SIATA_SERIES_72H_URL = f"{SIATA_GEOPORTAL}/geographJson/1/pm25/{{codigo}}"
SIATA_SERIES_30D_URL = f"{SIATA_GEOPORTAL}/geographJson/1/pm25_30d/{{codigo}}"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
VALLEY_LOCATION = {"latitude": 6.2518, "longitude": -75.5636, "timezone": "America/Bogota"}
HTTP_USER_AGENT = "FINAL_IA_SI3003/1.0 (proyecto academico EAFIT)"
HTTP_TIMEOUT_S = 60

# ----------------------------------------------------------------------------
# Limpieza
# ----------------------------------------------------------------------------
SIATA_MISSING = -9999
SIATA_MAX_QUALITY_FLAG = 2.5   # SIATA: calidad <= 2.5 es dato válido
PM25_PHYSICAL_MAX = 500.0      # Valores por encima se tratan como error de sensor
MIN_VALID_HOURS_PER_DAY = 18   # 75 % de representatividad temporal (criterio Res. 2254)

# ----------------------------------------------------------------------------
# Clases de PM2.5 (µg/m³, promedio 24 h) — Res. 2254 de 2017 / POECA
# ----------------------------------------------------------------------------
CLASS_NAMES = ["Normal", "Prevención", "Crítico"]
NORMAL_MAX = 37      # 0–37
PREVENCION_MAX = 55  # 38–55 ; Crítico >= 56

# ----------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------
TEST_FRACTION = 0.25
CV_FOLDS = 5
CV_PURGE_DAYS = 3
RANDOM_SEED = 42
BOOTSTRAP_SAMPLES = 2000

# ----------------------------------------------------------------------------
# Agente
# ----------------------------------------------------------------------------
T_MAX = 4
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_model": "openai/gpt-oss-120b",
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
        "default_model": "meta/llama-3.3-70b-instruct",
    },
}
RAG_TOP_K = 4


def pm25_to_class(value: float) -> int:
    """Clasifica un promedio diario de PM2.5 en 0=Normal, 1=Prevención, 2=Crítico.

    Se redondea al entero más cercano porque los umbrales normativos son enteros
    (37 y 55 µg/m³); así 37.4 es Normal y 37.6 es Prevención.
    """
    v = math.floor(float(value) + 0.5)  # redondeo "half-up" (round() de Python es bancario)
    if v <= NORMAL_MAX:
        return 0
    if v <= PREVENCION_MAX:
        return 1
    return 2
