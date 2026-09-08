"""Business/domain knowledge repository, backed by the RAG vector store.

This is the "tell the agent what a table means" side of the system:
users (or an onboarding script) write short notes about a table — what
a column represents, known exceptions, business definitions of
"valid" — and the rule suggester retrieves the most relevant notes when
proposing checks, the same way a human analyst would read the wiki page
before writing a DQ rule.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from dq_agent.knowledge.vector_store import HashingVectorStore, VectorStore
from dq_agent.models import KnowledgeDoc

_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_docs (
    doc_id TEXT PRIMARY KEY,
    table_fq_name TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_knowledge_table ON knowledge_docs(table_fq_name);
"""


class KnowledgeRepository:
    """CRUD for `KnowledgeDoc`s + semantic retrieval via a `VectorStore`.

    Two local files back this by default: a sqlite DB for structured
    CRUD/listing, and a pickle file for the vector index. Both are
    plain, inspectable, and require no external service.
    """

    def __init__(self, db_path: str, vector_store: Optional[VectorStore] = None):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self.vector_store = vector_store or HashingVectorStore(
            persist_path=str(Path(db_path).with_suffix(".vec.pkl"))
        )

    def add_doc(self, doc: KnowledgeDoc) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO knowledge_docs VALUES (?, ?, ?, ?, ?, ?)",
            (
                doc.doc_id,
                doc.table_fq_name,
                doc.title,
                doc.content,
                ",".join(doc.tags),
                doc.created_at,
            ),
        )
        self._conn.commit()
        embed_text = f"{doc.table_fq_name} | {doc.title}\n{doc.content}"
        self.vector_store.add(
            doc.doc_id, embed_text, metadata={"table_fq_name": doc.table_fq_name, "title": doc.title}
        )

    def get_docs_for_table(self, table_fq_name: str) -> list[KnowledgeDoc]:
        rows = self._conn.execute(
            "SELECT doc_id, table_fq_name, title, content, tags, created_at "
            "FROM knowledge_docs WHERE table_fq_name = ?",
            (table_fq_name,),
        ).fetchall()
        return [self._row_to_doc(r) for r in rows]

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
        row = self._conn.execute(
            "SELECT doc_id, table_fq_name, title, content, tags, created_at "
            "FROM knowledge_docs WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        return self._row_to_doc(row) if row else None

    def delete_doc(self, doc_id: str) -> None:
        self._conn.execute("DELETE FROM knowledge_docs WHERE doc_id = ?", (doc_id,))
        self._conn.commit()
        self.vector_store.delete(doc_id)

    @staticmethod
    def _row_to_doc(row) -> KnowledgeDoc:
        doc_id, table_fq_name, title, content, tags, created_at = row
        return KnowledgeDoc(
            doc_id=doc_id,
            table_fq_name=table_fq_name,
            title=title,
            content=content,
            tags=tags.split(",") if tags else [],
            created_at=created_at,
        )

    def close(self) -> None:
        self._conn.close()
