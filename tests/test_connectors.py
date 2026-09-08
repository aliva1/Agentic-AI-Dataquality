from dq_agent.connectors.registry import create_connector, registered_source_types


def test_registered_source_types_includes_all_planned_sources():
    types = registered_source_types()
    for expected in ("postgresql", "snowflake", "databricks", "sqlite", "flatfile", "kafka"):
        assert expected in types


def test_sql_connector_lists_tables_and_metadata(tmp_sqlite_db):
    conn = create_connector("sqlite", "demo", connection_string=tmp_sqlite_db)
    with conn:
        tables = conn.list_tables()
        assert "customers" in tables
        assert "orders" in tables

        meta = conn.get_metadata("customers")
        assert meta.fq_name == "demo.customers"
        col_names = {c.name for c in meta.columns}
        assert {"customer_id", "email", "signup_date", "country"} <= col_names
        assert meta.row_count_estimate == 5


def test_sql_connector_fetch_sample_and_batch(tmp_sqlite_db):
    conn = create_connector("sqlite", "demo", connection_string=tmp_sqlite_db)
    with conn:
        sample = conn.fetch_sample("customers", limit=3)
        assert len(sample) == 3
        full = conn.fetch_batch("customers")
        assert len(full) == 5


def test_sql_connector_stream_micro_batches(tmp_sqlite_db):
    conn = create_connector("sqlite", "demo", connection_string=tmp_sqlite_db)
    with conn:
        batches = list(conn.stream_micro_batches("customers", batch_size=2))
        assert [len(b) for b in batches] == [2, 2, 1]


def test_sqlite_lineage_is_empty_not_error(tmp_sqlite_db):
    # sqlite has no lineage concept; the honest answer is an empty list,
    # never an exception.
    conn = create_connector("sqlite", "demo", connection_string=tmp_sqlite_db)
    with conn:
        assert conn.get_lineage("customers") == []


def test_unknown_source_type_raises():
    import pytest

    with pytest.raises(ValueError):
        create_connector("carrier_pigeon", "x")


def test_flatfile_connector(tmp_path):
    import pandas as pd

    csv_path = tmp_path / "vendor_feed.csv"
    pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]}).to_csv(csv_path, index=False)

    conn = create_connector("flatfile", "vendor", path=str(csv_path))
    assert conn.list_tables() == ["vendor_feed"]
    meta = conn.get_metadata("vendor_feed")
    assert meta.row_count_estimate == 3
    df = conn.fetch_batch("vendor_feed")
    assert len(df) == 3
    batches = list(conn.stream_micro_batches("vendor_feed", batch_size=2))
    assert [len(b) for b in batches] == [2, 1]
