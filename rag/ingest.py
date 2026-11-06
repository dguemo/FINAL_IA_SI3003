"""Indexa los PDF normativos en ChromaDB (persistente en ``rag/chroma_db``).

* Extracción por página con PyMuPDF (se conserva el número de página para citar).
* Fragmentos de ~900 caracteres con 150 de solapamiento, sin cruzar páginas.
* Reconstrucción completa en cada ejecución (idempotente).

Uso:  python -m rag.ingest
"""
from __future__ import annotations

import hashlib
import logging
import re
import sys
from pathlib import Path

import chromadb
import pymupdf

import config
from rag.embeddings import get_embedder

log = logging.getLogger("ingest")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Nombre legible para citar, según palabras clave del nombre de archivo
# (el orden importa: la primera clave contenida en el nombre gana)
SOURCE_NAMES = {
    "2254": "Resolución 2254 de 2017 (MinAmbiente)",
    "corantioquia": "Plan de Acción POECA 2026 – Corantioquia (zona rural Valle de Aburrá)",
    "poeca": "POECA – Área Metropolitana del Valle de Aburrá",
}


def source_name(path: Path) -> str:
    stem = path.stem.lower()
    for key, name in SOURCE_NAMES.items():
        if key in stem:
            return name
    return path.stem.replace("_", " ")


def clean_text(text: str) -> str:
    text = text.replace("­", "")                 # guiones suaves
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)       # palabras cortadas por guion
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Corta en ventanas de ``size`` caracteres, prefiriendo terminar en fin de frase."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", start + size // 2, end), text.rfind("\n", start + size // 2, end))
            if cut > start:
                end = cut + 1
        piece = text[start:end].strip()
        if len(piece) > 40:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def repeated_lines(pages: list[str], min_ratio: float = 0.5) -> set[str]:
    """Líneas que aparecen en >= 50 % de las páginas (encabezados/pies de página)."""
    if len(pages) < 3:
        return set()
    counts: dict[str, int] = {}
    for text in pages:
        for line in {ln.strip() for ln in text.split("\n") if ln.strip()}:
            counts[line] = counts.get(line, 0) + 1
    return {ln for ln, c in counts.items() if c / len(pages) >= min_ratio}


def extract_pdf(path: Path) -> list[dict]:
    records = []
    with pymupdf.open(path) as doc:
        raw_pages = [clean_text(page.get_text("text")) for page in doc]
    boiler = repeated_lines(raw_pages)
    page_num = re.compile(r"^p[áa]gina \d+ de \d+$", re.IGNORECASE)
    for i, raw in enumerate(raw_pages, start=1):
        lines = [ln for ln in raw.split("\n")
                 if ln.strip() not in boiler and not page_num.match(ln.strip())]
        text = "\n".join(lines).strip()
        for j, chunk in enumerate(chunk_text(text)):
            uid = hashlib.sha1(f"{path.name}|{i}|{j}".encode()).hexdigest()[:16]
            records.append({"id": uid, "text": chunk,
                            "meta": {"fuente": source_name(path), "archivo": path.name,
                                     "pagina": i}})
    if not records:
        log.warning("%s no tiene texto extraíble (¿PDF escaneado?). Se omite.", path.name)
    return records


def get_collection(client: chromadb.ClientAPI | None = None, reset: bool = False):
    client = client or chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(config.CHROMA_COLLECTION)
        except Exception:  # noqa: BLE001  (no existe aún)
            pass
    return client.get_or_create_collection(config.CHROMA_COLLECTION,
                                           metadata={"hnsw:space": "cosine"})


# PDF oficiales descargables automáticamente si faltan (el POECA se agrega a mano;
# ver rag/docs/README.md). Se recomienda versionar los PDF en el repositorio.
DOC_URLS = {
    "Resolucion_2254_2017_MinAmbiente.pdf":
        "https://calidadaire.cvc.gov.co/sites/default/files/2021-05/Resoluci%C3%B3n%202254%20de%202017.pdf",
}


def download_missing(docs_dir: Path = config.DOCS_DIR) -> None:
    import requests
    docs_dir.mkdir(parents=True, exist_ok=True)
    for name, url in DOC_URLS.items():
        target = docs_dir / name
        key = name.split("_")[1]
        if target.exists() or any(key in p.name for p in docs_dir.glob("*.pdf")):
            continue
        try:
            r = requests.get(url, timeout=config.HTTP_TIMEOUT_S,
                             headers={"User-Agent": config.HTTP_USER_AGENT})
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise ValueError("la respuesta no es un PDF")
            target.write_bytes(r.content)
            log.info("Descargado %s", name)
        except Exception as err:  # noqa: BLE001
            log.warning("No se pudo descargar %s (%s). Colóquelo a mano en %s", name, err, docs_dir)


def ingest(docs_dir: Path = config.DOCS_DIR) -> int:
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No hay PDF en {docs_dir}. Ver rag/docs/README.md")
    records = [r for p in pdfs for r in extract_pdf(p)]
    if not records:
        raise ValueError("Ningún PDF tuvo texto extraíble")
    col = get_collection(reset=True)
    emb = get_embedder()
    batch = 64
    for i in range(0, len(records), batch):
        part = records[i:i + batch]
        vectors = emb.encode([r["text"] for r in part])
        col.add(ids=[r["id"] for r in part], documents=[r["text"] for r in part],
                metadatas=[r["meta"] for r in part], embeddings=[list(map(float, v)) for v in vectors])
    for p in pdfs:
        log.info("Indexado %s", p.name)
    log.info("Total fragmentos: %d en %s", len(records), config.CHROMA_DIR)
    return len(records)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        download_missing()
        ingest()
    except (FileNotFoundError, ValueError) as err:
        log.error(err)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
