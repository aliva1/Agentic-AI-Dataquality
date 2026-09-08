"""One connector class, many SQL warehouses.

Postgres, Snowflake, Databricks (Unity Catalog) and SQLite all speak
SQL and all have a SQLAlchemy dialect, so `SQLConnector` implements the
`DataSourceConnector` contract exactly once using `sqlalchemy.inspect()`
for generic metadata, and a small per-dialect strategy for the things
that genuinely differ: lineage lookup, and (for Databricks) streaming.

This is the concrete demonstration of "change source, orchestration
stays the same": constructing a `SQLConnector` with a different
`dialect=` and `connection_string=` is the entire integration surface.

Real credentials are never required to *use* this module — only to
connect to a real warehouse. The bundled demo/tests use the `sqlite`
dialect, which needs nothing but a local file.
"""

from __future__ import annotations

from typing import Iterator, Optional

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from dq_agent.connectors.base import StreamingCapableConnector
from dq_agent.models import ColumnMetadata, LineageEdge, TableMetadata

SUPPORTED_DIALECTS = {"postgresql", "snowflake", "databricks", "sqlite"}


class SQLConnector(StreamingCapableConnector):
    """Generic SQL connector, parameterized by SQLAlchemy dialect.

    Parameters
    ----------
    source_name:
        Logical name for this source, used to build fully-qualified
        table names (e.g. "prod_postgres").
    dialect:
        One of "postgresql", "snowflake", "databricks", "sqlite".
    connection_string:
        A SQLAlchemy connection URL, e.g.
        ``postgresql+psycopg2://user:pass@host:5432/dbname``,
        ``snowflake://user:pass@account/db/schema?warehouse=WH``,
        ``databricks://token:<pat>@<host>?http_path=<path>&catalog=<cat>``,
        ``sqlite:///path/to/file.db``.
    schema:
        Default schema to introspect when the caller doesn't fully
        qualify a table name.
    """

    source_type = "sql"

    def __init__(
        self,
        source_name: str,
        dialect: str,
        connection_string: str,
        schema: Optional[str] = None,
    ):
        super().__init__(source_name)
        if dialect not in SUPPORTED_DIALECTS:
            raise ValueError(
                f"Unsupported dialect {dialect!r}; supported: {sorted(SUPPORTED_DIALECTS)}"
            )
        self.dialect = dialect
        self.connection_string = connection_string
        self.schema = schema
        self._engine: Optional[Engine] = None

    # -- lifecycle ---------------------------------------------------
    def connect(self) -> None:
        if self._engine is None:
            self._engine = create_engine(self.connection_string)

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self.connect()
        return self._engine  # type: ignore[return-value]

    # -- metadata ------------------------------------------------------
    def list_tables(self) -> list[str]:
        insp = inspect(self.engine)
        tables = insp.get_table_names(schema=self.schema)
        views = []
        try:
            views = insp.get_view_names(schema=self.schema)
        except NotImplementedError:  # pragma: no cover - some dialects skip views
            pass
        return sorted(set(tables) | set(views))

    def get_metadata(self, table: str) -> TableMetadata:
        insp = inspect(self.engine)
        schema, table_name = self._split(table)
        cols_raw = insp.get_columns(table_name, schema=schema)
        pk_cols = set()
        try:
            pk = insp.get_pk_constraint(table_name, schema=schema)
            pk_cols = set(pk.get("constrained_columns") or [])
        except Exception:  # pragma: no cover - not all dialects expose PKs
            pass

        columns = [
            ColumnMetadata(
                name=c["name"],
                data_type=str(c["type"]),
                nullable=c.get("nullable", True),
                is_primary_key=c["name"] in pk_cols,
                comment=c.get("comment"),
            )
            for c in cols_raw
        ]
        return TableMetadata(
            source_name=self.source_name,
            schema=schema,
            table=table_name,
            columns=columns,
            row_count_estimate=self.row_count(table),
        )

    def row_count(self, table: str) -> Optional[int]:
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {self._qualify(table)}"))
                return int(result.scalar_one())
        except Exception:
            return None

    def get_lineage(self, table: str) -> list[LineageEdge]:
        strategy = _LINEAGE_STRATEGIES.get(self.dialect)
        if strategy is None:
            return []
        try:
            return strategy(self, table)
        except Exception:
            # Lineage is best-effort by design (requires elevated
            # privileges on some warehouses); never let it break a scan.
            return []

    # -- data --------------------------------------------------------
    def fetch_sample(self, table: str, limit: int = 1000) -> pd.DataFrame:
        query = f"SELECT * FROM {self._qualify(table)} LIMIT {int(limit)}"
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn)

    def fetch_batch(self, table: str, where: Optional[str] = None) -> pd.DataFrame:
        query = f"SELECT * FROM {self._qualify(table)}"
        if where:
            query += f" WHERE {where}"
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn)

    def stream_micro_batches(
        self, table: str, batch_size: int = 500
    ) -> Iterator[pd.DataFrame]:
        """Chunked reads standing in for "streaming" over a SQL table.

        Real streaming sources (Kafka) live in `dq_agent.streaming`; this
        exists so a SQL table can be pushed through the exact same
        streaming-orchestration code path in demos/tests without a
        broker.
        """
        query = f"SELECT * FROM {self._qualify(table)}"
        with self.engine.connect() as conn:
            for chunk in pd.read_sql(text(query), conn, chunksize=batch_size):
                yield chunk

    # -- helpers -------------------------------------------------------
    def _split(self, table: str) -> tuple[Optional[str], str]:
        if "." in table:
            schema, name = table.split(".", 1)
            return schema, name
        return self.schema, table

    def _qualify(self, table: str) -> str:
        schema, name = self._split(table)
        return f"{schema}.{name}" if schema else name


# ---------------------------------------------------------------------
# Per-dialect lineage strategies
# ---------------------------------------------------------------------


def _postgres_lineage(conn: SQLConnector, table: str) -> list[LineageEdge]:
    """View -> base table dependencies via information_schema."""
    schema, name = conn._split(table)
    q = text(
        """
        SELECT view_schema, view_name, table_schema, table_name
        FROM information_schema.view_table_usage
        WHERE table_name = :name AND (:schema IS NULL OR table_schema = :schema)
        """
    )
    edges = []
    with conn.engine.connect() as c:
        rows = c.execute(q, {"name": name, "schema": schema}).fetchall()
    for r in rows:
        downstream = f"{r.view_schema}.{r.view_name}"
        upstream = f"{r.table_schema}.{r.table_name}"
        edges.append(
            LineageEdge(
                upstream=upstream,
                downstream=downstream,
                relationship="view_depends_on",
                source_of_info="information_schema.view_table_usage",
            )
        )
    return edges


def _snowflake_lineage(conn: SQLConnector, table: str) -> list[LineageEdge]:
    """Object dependencies via ACCOUNT_USAGE (requires the ACCOUNTADMIN
    role or an explicit grant — see README)."""
    schema, name = conn._split(table)
    q = text(
        """
        SELECT referencing_database, referencing_schema, referencing_object_name,
               referenced_database, referenced_schema, referenced_object_name
        FROM snowflake.account_usage.object_dependencies
        WHERE referenced_object_name = :name
        """
    )
    edges = []
    with conn.engine.connect() as c:
        rows = c.execute(q, {"name": name.upper()}).fetchall()
    for r in rows:
        upstream = f"{r.referenced_schema}.{r.referenced_object_name}"
        downstream = f"{r.referencing_schema}.{r.referencing_object_name}"
        edges.append(
            LineageEdge(
                upstream=upstream,
                downstream=downstream,
                relationship="object_dependency",
                source_of_info="account_usage.object_dependencies",
            )
        )
    return edges


def _databricks_lineage(conn: SQLConnector, table: str) -> list[LineageEdge]:
    """Table lineage via Unity Catalog's system tables."""
    schema, name = conn._split(table)
    fq = f"{schema}.{name}" if schema else name
    q = text(
        """
        SELECT source_table_full_name, target_table_full_name
        FROM system.access.table_lineage
        WHERE target_table_full_name = :fq OR source_table_full_name = :fq
        """
    )
    edges = []
    with conn.engine.connect() as c:
        rows = c.execute(q, {"fq": fq}).fetchall()
    for r in rows:
        if not r.source_table_full_name or not r.target_table_full_name:
            continue
        edges.append(
            LineageEdge(
                upstream=r.source_table_full_name,
                downstream=r.target_table_full_name,
                relationship="etl_lineage",
                source_of_info="system.access.table_lineage",
            )
        )
    return edges


_LINEAGE_STRATEGIES = {
    "postgresql": _postgres_lineage,
    "snowflake": _snowflake_lineage,
    "databricks": _databricks_lineage,
    # sqlite intentionally omitted: no lineage concept, empty list is correct.
}
