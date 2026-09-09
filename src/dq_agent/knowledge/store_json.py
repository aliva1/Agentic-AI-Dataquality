"""JSON-file-backed twin of `KnowledgeRepository`.

Same public API as the SQLite version (`dq_agent.knowledge.store.KnowledgeRepository`):
CRUD for `KnowledgeDoc`s plus semantic retrieval via a `VectorStore`.
Only the structured-storage half moves from SQLite to a plain `.json`
file; the RAG index is still the same `HashingVectorStore`, persisted
to its own pickle file (it already was file-based, not a database) --
so a fully flat-file deployment ends up with two small files
(`knowledge.json`, `knowledge.vec.pkl`) instead of one SQLite database.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from dq_agent.knowledge.vector_store import HashingVectorStore, VectorStore
from dq_agent.models import KnowledgeDoc


def _doc_to_dict(doc: KnowledgeDoc) -> dict:
    return {
        "doc_id": doc.doc_id,
        "table_fq_name": doc.table_fq_name,
        "title": doc.title,
        "content": doc.content,
        "tags": doc.tags,
        "created_at": doc.created_at,
    }


def _dict_to_doc(d: dict) -> KnowledgeDoc:
    return KnowledgeDoc(
        doc_id=d["doc_id"],
        table_fq_name=d["table_fq_name"],
        title=d["title"],
        content=d["content"],
        tags=d.get("tags") or [],
        created_at=d["created_at"],
    )


class JsonKnowledgeRepository:
    """Drop-in alternative to `KnowledgeRepository`, backed by one JSON file
    (`{"docs": {"<doc_id>": {...}, ...}}`) plus the same vector-store pickle.
    """

    def __init__(self, json_path: str, vector_store: Optional[VectorStore] = None):
        self.json_path = Path(json_path)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        if self.json_path.exists():
            self._data = json.loads(self.json_path.read_text() or "{}")
        else:
            self._data = {}
        self._data.setdefault("docs", {})
        self._flush()
        self.vector_store = vector_store or HashingVectorStore(
            persist_path=str(self.json_path.with_suffix(".vec.pkl"))
        )

    def _flush(self) -> None:
        self.json_path.write_text(json.dumps(self._data, indent=2, sort_keys=True))

    def add_doc(self, doc: KnowledgeDoc) -> None:
        self._data["docs"][doc.doc_id] = _doc_to_dict(doc)
        self._flush()
        embed_text = f"{doc.table_fq_name} | {doc.title}\n{doc.content}"
        self.vector_store.add(
            doc.doc_id, embed_text, metadata={"table_fq_name": doc.table_fq_name, "title": doc.title}
        )

    def get_docs_for_table(self, table_fq_name: str) -> list[KnowledgeDoc]:
        return [
            _dict_to_doc(d)
            for d in self._data["docs"].values()
            if d["table_fq_name"] == table_fq_name
        ]

    def search(self, query: str, k: int = 5, table_fq_name: Optional[str] = None) -> list[KnowledgeDoc]:
        hits = self.vector_store.query(query, k=k * 3 if table_fq_name else k)
        docs = []
        for doc_id, _score, meta in hits:
            if table_fq_name and meta.get("table_fq_name") != table_fq_name:
                continue
            doc = self.get_doc(doc_id)
            if doc:
                docs.append(doc)
            if len(docs) >= k:
                break
        return docs

    def get_doc(self, doc_id: str) -> Optional[KnowledgeDoc]:
        d = self._data["docs"].get(doc_id)
        return _dict_to_doc(d) if d else None

    def delete_doc(self, doc_id: str) -> None:
        self._data["docs"].pop(doc_id, None)
        self._flush()
        self.vector_store.delete(doc_id)

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass
