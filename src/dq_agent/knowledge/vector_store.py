"""A small, dependency-light vector store for the RAG layer.

Why not Chroma/pgvector/FAISS directly? Those are the right call in
production (see README), but they either need a running service or a
downloaded embedding model — both awkward for a self-contained
proof-of-concept that has to run offline and deterministically in CI.

`HashingVectorStore` uses scikit-learn's `HashingVectorizer` (a
stateless bag-of-words hash, no vocabulary fitting, no network) to embed
text, and plain cosine similarity for retrieval. It implements the same
`VectorStore` interface a real backend would, so swapping it out later
is a one-class change, not a rewrite of the knowledge/rule layer.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore(ABC):
    @abstractmethod
    def add(self, doc_id: str, text: str, metadata: Optional[dict[str, Any]] = None) -> None: ...

    @abstractmethod
    def query(self, text: str, k: int = 5) -> list[tuple[str, float, dict[str, Any]]]:
        """Return up to `k` (doc_id, similarity_score, metadata) tuples,
        most similar first."""

    @abstractmethod
    def delete(self, doc_id: str) -> None: ...


class HashingVectorStore(VectorStore):
    def __init__(self, persist_path: Optional[str] = None, n_features: int = 4096):
        self._vectorizer = HashingVectorizer(
            n_features=n_features, alternate_sign=False, norm="l2"
        )
        self._ids: list[str] = []
        self._vectors: Optional[np.ndarray] = None  # shape (n_docs, n_features), dense for simplicity
        self._metadata: dict[str, dict[str, Any]] = {}
        self._texts: dict[str, str] = {}
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self._load()

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._vectorizer.transform(texts).toarray()

    def add(self, doc_id: str, text: str, metadata: Optional[dict[str, Any]] = None) -> None:
        vec = self._embed([text])
        if doc_id in self._ids:
            idx = self._ids.index(doc_id)
            self._vectors[idx] = vec[0]
        else:
            self._ids.append(doc_id)
            self._vectors = vec if self._vectors is None else np.vstack([self._vectors, vec])
        self._metadata[doc_id] = metadata or {}
        self._texts[doc_id] = text
        if self.persist_path:
            self._save()

    def query(self, text: str, k: int = 5) -> list[tuple[str, float, dict[str, Any]]]:
        if not self._ids:
            return []
        qvec = self._embed([text])
        sims = cosine_similarity(qvec, self._vectors)[0]
        order = np.argsort(-sims)[:k]
        return [
            (self._ids[i], float(sims[i]), self._metadata[self._ids[i]])
            for i in order
            if sims[i] > 0
        ]

    def delete(self, doc_id: str) -> None:
        if doc_id not in self._ids:
            return
        idx = self._ids.index(doc_id)
        self._ids.pop(idx)
        self._vectors = np.delete(self._vectors, idx, axis=0) if self._vectors is not None else None
        self._metadata.pop(doc_id, None)
        self._texts.pop(doc_id, None)
        if self.persist_path:
            self._save()

    # -- persistence -----------------------------------------------------
    def _save(self) -> None:
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "wb") as f:
            pickle.dump(
                {
                    "ids": self._ids,
                    "vectors": self._vectors,
                    "metadata": self._metadata,
                    "texts": self._texts,
                },
                f,
            )

    def _load(self) -> None:
        with open(self.persist_path, "rb") as f:
            state = pickle.load(f)
        self._ids = state["ids"]
        self._vectors = state["vectors"]
        self._metadata = state["metadata"]
        self._texts = state["texts"]
