"""Ejecuta todo el flujo reproducible de punta a punta.

    python pipeline.py            # descarga (si hay internet) + dataset + modelo + índice RAG
    python pipeline.py --offline  # usa solo el Plan B versionado en data/raw/

Después:  streamlit run app.py
"""
from __future__ import annotations

import argparse
import logging
import sys

log = logging.getLogger("pipeline")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true", help="No intenta descargar datos")
    ap.add_argument("--skip-rag", action="store_true", help="No reconstruye el índice RAG")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from data import build_dataset, download_siata
    from models import train
    from rag import ingest

    if not args.offline:
        log.info("== 1/4 Descarga de datos (SIATA, geoportal AMVA, Open-Meteo) ==")
        if download_siata.main() != 0:
            return 1
    else:
        log.info("== 1/4 Modo offline: se usa el Plan B de data/raw/ ==")

    log.info("== 2/4 Limpieza y exportación a Parquet ==")
    if build_dataset.main() != 0:
        return 1

    log.info("== 3/4 Entrenamiento y evaluación del Naive Bayes ==")
    if train.main() != 0:
        return 1

    if not args.skip_rag:
        log.info("== 4/4 Índice RAG de normativa (ChromaDB) ==")
        if ingest.main() != 0:
            log.error("Sin índice RAG el agente no podrá citar normativa. Revise rag/docs/README.md")
            return 1

    log.info("Listo. Ejecute:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
