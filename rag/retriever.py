"""Búsqueda semántica sobre la normativa indexada en ChromaDB."""
from __future__ import annotations

from functools import lru_cache

import chromadb

import config
from rag.embeddings import get_embedder


@lru_cache(maxsize=1)
def _collection():
    if not config.CHROMA_DIR.exists():
        raise FileNotFoundError("Índice RAG no construido. Ejecute: python -m rag.ingest")
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    col = client.get_or_create_collection(config.CHROMA_COLLECTION,
                                          metadata={"hnsw:space": "cosine"})
    if col.count() == 0:
        raise FileNotFoundError("El índice RAG está vacío. Ejecute: python -m rag.ingest")
    return col


def reset_cache() -> None:
    _collection.cache_clear()


def search(query: str, k: int = config.RAG_TOP_K) -> list[dict]:
    """Devuelve los k fragmentos más similares con fuente, página y similitud coseno."""
    query = (query or "").strip()
    if len(query) < 3:
        raise ValueError("La consulta está vacía o es demasiado corta")
    col = _collection()
    vec = get_embedder().encode([query])[0]
    res = col.query(query_embeddings=[list(map(float, vec))], n_results=max(1, min(k, 10)))
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"fuente": meta["fuente"], "pagina": int(meta["pagina"]),
                    "archivo": meta.get("archivo", ""), "similitud": round(1 - float(dist), 3),
                    "texto": doc})
    return out
