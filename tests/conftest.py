"""Entorno aislado para las pruebas: datos sintéticos, modelo e índice RAG en tmp.

No usa internet ni API keys: el embedder y el LLM se reemplazan por dobles de prueba.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pymupdf
import pytest

import config


class HashEmbedder:
    """Embedder determinista (bolsa de palabras con hashing) para pruebas."""

    def encode(self, texts):
        out = np.zeros((len(texts), 256))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)


PDF_PAGES = [
    "Artículo 2. Niveles máximos permisibles. Para PM2.5 el nivel máximo permisible es 37 "
    "microgramos por metro cúbico en un tiempo de exposición de 24 horas.",
    "Artículo 10. Niveles de prevención, alerta y emergencia. Prevención PM2.5 entre 38 y 55, "
    "alerta entre 56 y 150, emergencia mayor o igual a 151.",
]


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    from data import build_dataset, db
    from agent import tools
    from models import train
    from rag import embeddings, ingest, retriever
    from tests.synthetic import make_raw

    root = tmp_path_factory.mktemp("proj")
    mp = pytest.MonkeyPatch()
    paths = {
        "RAW_DIR": root / "raw", "PROCESSED_DIR": root / "processed",
        "RAW_PM25_HOURLY": root / "raw" / "pm25_hourly.csv.gz",
        "RAW_STATIONS": root / "raw" / "stations.csv", "RAW_METEO": root / "raw" / "meteo_daily.csv",
        "PM25_DAILY_PARQUET": root / "processed" / "pm25_daily.parquet",
        "STATIONS_PARQUET": root / "processed" / "stations.parquet",
        "ARTIFACTS_DIR": root / "artifacts", "GNB_PARAMS": root / "artifacts" / "gnb.npz",
        "MODEL_METRICS": root / "artifacts" / "metrics.json",
        "EVAL_RESULTS_DIR": root / "results", "CHROMA_DIR": root / "chroma",
        "DOCS_DIR": root / "docs",
    }
    for k, v in paths.items():
        mp.setattr(config, k, v)
    mp.setattr(config, "BOOTSTRAP_SAMPLES", 100)

    make_raw(paths["RAW_DIR"], n_stations=4, start="2023-08-01", end="2024-05-31", seed=1)
    build_dataset.build()
    db.reset_connection()
    train.train()
    tools.reset_caches()

    paths["DOCS_DIR"].mkdir()
    doc = pymupdf.open()
    for txt in PDF_PAGES:
        doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 800), txt, fontsize=10)
    doc.save(paths["DOCS_DIR"] / "Resolucion_2254_2017.pdf")
    embeddings.set_embedder(HashEmbedder())
    ingest.ingest(paths["DOCS_DIR"])
    retriever.reset_cache()

    yield paths
    mp.undo()
    db.reset_connection()
    tools.reset_caches()
    retriever.reset_cache()
    embeddings.set_embedder(None)
