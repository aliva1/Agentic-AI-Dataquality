"""The connector interface every data source plugs into.

This is the seam that makes the agent "plugin-shaped": the orchestrator
only ever talks to a `DataSourceConnector`. Swapping Postgres for
Snowflake, Databricks, or a flat file means instantiating a different
connector class through `registry.create_connector(...)` — nothing else
in the pipeline changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional

import pandas as pd

from dq_agent.models import LineageEdge, TableMetadata


class DataSourceConnector(ABC):
    """Abstract base for any batch-queryable data source."""

    #: set by subclasses; used by the registry and shown in error messages
    source_type: str = "base"

    def __init__(self, source_name: str):
        self.source_name = source_name

    # -- lifecycle ---------------------------------------------------
    def __enter__(self) -> "DataSourceConnector":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> None:  # pragma: no cover - default no-op
        """Open any underlying connection. Optional to override."""

    def close(self) -> None:  # pragma: no cover - default no-op
        """Release any underlying connection. Optional to override."""

    # -- required capabilities ----------------------------------------
    @abstractmethod
    def list_tables(self) -> list[str]:
        """Return fully-qualified table names visible to this connector."""

    @abstractmethod
    def get_metadata(self, table: str) -> TableMetadata:
        """Return column-level metadata for `table`."""

    @abstractmethod
    def fetch_sample(self, table: str, limit: int = 1000) -> pd.DataFrame:
        """Return up to `limit` sample rows, used for profiling."""

    @abstractmethod
    def fetch_batch(self, table: str, where: Optional[str] = None) -> pd.DataFrame:
        """Return the full table (or filtered subset) as a DataFrame.

        For genuinely large tables a real deployment would page this or
        push checks down to SQL; the interface intentionally stays
        simple for the proof-of-concept and can be optimized per
        connector without touching callers.
        """

    def get_lineage(self, table: str) -> list[LineageEdge]:
        """Best-effort lineage lookup. Default: unsupported (empty list).

        Concrete connectors override this where the source exposes
        lineage info (e.g. Postgres view dependencies, Databricks Unity
        Catalog's `system.access.table_lineage`, Snowflake's
        ACCOUNT_USAGE.OBJECT_DEPENDENCIES). Returning an empty list is a
        valid, honest answer for sources that don't support it.
        """
        return []

    def row_count(self, table: str) -> Optional[int]:
        """Best-effort exact/estimated row count. Default: unknown."""
        return None


class StreamingCapableConnector(DataSourceConnector):
    """Optional mixin-style base for sources that can also stream.

    Kept separate from `DataSourceConnector` so a purely-batch source
    (e.g. a flat file) doesn't need to implement `stream_micro_batches`.
    """

    @abstractmethod
    def stream_micro_batches(
        self, table: str, batch_size: int = 500
    ) -> Iterator[pd.DataFrame]:
        """Yield successive micro-batches for near-real-time checking."""
