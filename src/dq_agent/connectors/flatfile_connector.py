"""Flat-file connector: CSV / Parquet / JSON on local disk or a URL.

Flat files have no catalog to introspect, so the caller declares the
source system a file logically belongs to (e.g. "sap_export",
"vendor_feed") — this becomes the connector's `source_name` and is what
shows up in lineage/knowledge lookups, mirroring how you'd tag an
ungoverned feed in a real catalog.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from dq_agent.connectors.base import StreamingCapableConnector
from dq_agent.models import ColumnMetadata, TableMetadata

_READERS = {
    ".csv": pd.read_csv,
    ".tsv": lambda p, **kw: pd.read_csv(p, sep="\t", **kw),
    ".parquet": pd.read_parquet,
    ".json": pd.read_json,
    ".jsonl": lambda p, **kw: pd.read_json(p, lines=True, **kw),
}


class FlatFileConnector(StreamingCapableConnector):
    """One connector, one file. `list_tables()` returns the logical name
    the caller chose so this composes cleanly with the registry, which
    expects to enumerate "tables" per source.
    """

    source_type = "flatfile"

    def __init__(self, source_name: str, path: str, logical_table: Optional[str] = None):
        super().__init__(source_name)
        self.path = Path(path)
        self.logical_table = logical_table or self.path.stem
        suffix = self.path.suffix.lower()
        if suffix not in _READERS:
            raise ValueError(f"Unsupported flat-file extension {suffix!r} for {path}")
        self._reader = _READERS[suffix]
        self._df: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._reader(self.path)
        return self._df

    def list_tables(self) -> list[str]:
        return [self.logical_table]

    def get_metadata(self, table: str) -> TableMetadata:
        df = self._load()
        columns = [
            ColumnMetadata(name=c, data_type=str(df[c].dtype), nullable=bool(df[c].isna().any()))
            for c in df.columns
        ]
        return TableMetadata(
            source_name=self.source_name,
            schema=None,
            table=self.logical_table,
            columns=columns,
            row_count_estimate=len(df),
            comment=f"flat file: {self.path}",
        )

    def row_count(self, table: str) -> Optional[int]:
        return len(self._load())

    def fetch_sample(self, table: str, limit: int = 1000) -> pd.DataFrame:
        return self._load().head(limit).copy()

    def fetch_batch(self, table: str, where: Optional[str] = None) -> pd.DataFrame:
        df = self._load()
        if where:
            df = df.query(where)
        return df.copy()

    def stream_micro_batches(
        self, table: str, batch_size: int = 500
    ) -> Iterator[pd.DataFrame]:
        df = self._load()
        for start in range(0, len(df), batch_size):
            yield df.iloc[start : start + batch_size].copy()
