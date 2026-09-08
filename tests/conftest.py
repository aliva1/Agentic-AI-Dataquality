import sqlite3
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def tmp_sqlite_db(tmp_path: Path) -> str:
    """A tiny SQLite DB with two related tables and a few deliberate
    data quality issues, used across connector/profiler/engine tests."""
    db_path = tmp_path / "demo.db"
    conn = sqlite3.connect(db_path)
    customers = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "email": [
                "a@example.com",
                "b@example.com",
                "not-an-email",
                None,
                "e@example.com",
            ],
            "signup_date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ],
            "country": ["IN", "US", "IN", "US", "IN"],
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": [100, 101, 102, 103],
            "customer_id": [1, 2, 3, 999],  # 999 doesn't exist -> referential violation
            "amount": [10.5, 20.0, -5.0, 15.0],  # negative amount -> range violation
        }
    )
    customers.to_sql("customers", conn, index=False, if_exists="replace")
    orders.to_sql("orders", conn, index=False, if_exists="replace")
    conn.commit()
    conn.close()
    return f"sqlite:///{db_path}"


@pytest.fixture
def workdir(tmp_path: Path) -> str:
    d = tmp_path / "dq_state"
    d.mkdir()
    return str(d)
