"""Modelo de embeddings compartido por la ingesta y la búsqueda.

Se carga una sola vez (perezoso). Las pruebas pueden inyectar otro embedder
con ``set_embedder`` para no depender de descargar el modelo.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

import config


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer  # import perezoso (pesado)
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32,
                                 show_progress_bar=False)


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformerEmbedder()
    return _embedder


def set_embedder(embedder: Embedder | None) -> None:
    global _embedder
    _embedder = embedder
