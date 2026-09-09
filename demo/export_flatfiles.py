"""Export the same Chinook data `run_demo.py` uses into flat CSV files.

Run with:

    python demo/export_flatfiles.py

Reads `data/chinook.db` (the SQLite demo database) and writes
`data/flatfiles/Customer.csv`, `Track.csv`, and `InvoiceLine.csv` --
byte-for-byte the same rows, just as plain CSV instead of SQL tables.
`run_demo_flatfile.py` then runs the exact same DQ pipeline against
these files via `DirectoryFlatFileConnector`, proving that swapping a
database source for flat files doesn't change the orchestration at all.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
DB_PATH = DATA_DIR / "chinook.db"
OUT_DIR = DATA_DIR / "flatfiles"

TABLES = ["Customer", "Track", "InvoiceLine"]


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"Expected {DB_PATH} — see README for how to fetch the Chinook database.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    try:
        for table in TABLES:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            out_path = OUT_DIR / f"{table}.csv"
            df.to_csv(out_path, index=False)
            print(f"{table}: {len(df)} rows -> {out_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
