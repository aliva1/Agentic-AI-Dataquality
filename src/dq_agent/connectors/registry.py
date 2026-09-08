"""The plugin registry: turns a config dict into a connector instance.

This is what "the orchestration should be the same, only the source
changes" means in code. Everything upstream of this factory
(`DataQualityAgent`) only ever holds a `DataSourceConnector`; everything
about *which* system that is lives in one small config dict.

Adding a brand-new source type is: write a class implementing
`DataSourceConnector`, then call `register("my_source", MyConnector)`
(or add it to `_BUILTIN` below) — no changes anywhere else in the
codebase.
"""

from __future__ import annotations

from typing import Any, Callable

from dq_agent.connectors.base import DataSourceConnector
from dq_agent.connectors.flatfile_connector import FlatFileConnector
from dq_agent.connectors.sql_connector import SQLConnector

ConnectorFactory = Callable[..., DataSourceConnector]

_REGISTRY: dict[str, ConnectorFactory] = {}


def register(source_type: str, factory: ConnectorFactory) -> None:
    _REGISTRY[source_type] = factory


def _make_sql(dialect: str):
    def factory(source_name: str, **config: Any) -> DataSourceConnector:
        return SQLConnector(
            source_name=source_name,
            dialect=dialect,
            connection_string=config["connection_string"],
            schema=config.get("schema"),
        )

    return factory


def _make_flatfile(source_name: str, **config: Any) -> DataSourceConnector:
    return FlatFileConnector(
        source_name=source_name,
        path=config["path"],
        logical_table=config.get("logical_table"),
    )


def _make_kafka(source_name: str, **config: Any) -> DataSourceConnector:
    # imported lazily: kafka-python is an optional extra, and importing
    # it eagerly would make the whole registry require it.
    from dq_agent.streaming.kafka_source import KafkaStreamSource

    return KafkaStreamSource(
        source_name=source_name,
        bootstrap_servers=config["bootstrap_servers"],
        topic=config["topic"],
        value_schema=config.get("value_schema"),
    )


register("postgresql", _make_sql("postgresql"))
register("snowflake", _make_sql("snowflake"))
register("databricks", _make_sql("databricks"))
register("sqlite", _make_sql("sqlite"))
register("flatfile", _make_flatfile)
register("kafka", _make_kafka)


def create_connector(source_type: str, source_name: str, **config: Any) -> DataSourceConnector:
    """Build a connector for `source_type` from a flat config dict.

    Example
    -------
    >>> create_connector(
    ...     "postgresql", "prod_pg",
    ...     connection_string="postgresql+psycopg2://u:p@host/db",
    ...     schema="cam_ms",
    ... )
    """
    try:
        factory = _REGISTRY[source_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown source_type {source_type!r}; registered types: {sorted(_REGISTRY)}"
        ) from exc
    return factory(source_name=source_name, **config)


def registered_source_types() -> list[str]:
    return sorted(_REGISTRY)
