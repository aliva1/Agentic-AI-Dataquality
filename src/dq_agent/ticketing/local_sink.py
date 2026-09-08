"""Default, offline ticket sink: a local SQLite table.

Mirrors two things learned the hard way in production ITSM automation:
one ticket per table per incident window rather than one per failed
rule (`CreateDQ_Combined.py`-style grouping), and duplicate-ticket
prevention — if an OPEN ticket already references the same rules for
the same table, reuse it instead of spamming a new one.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from dq_agent.models import CheckResult, Severity, Ticket, TicketStatus
from dq_agent.ticketing.base import TicketSink

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    table_fq_name TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    severity TEXT NOT NULL,
    check_results TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tickets_table ON tickets(table_fq_name);
"""


class LocalTicketSink(TicketSink):
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def create_ticket(
        self,
        table_fq_name: str,
        title: str,
        description: str,
        severity: Severity,
        results: list[CheckResult],
    ) -> Ticket:
        rule_ids = sorted({r.rule_id for r in results})
        existing = self._find_open_duplicate(table_fq_name, rule_ids)
        if existing:
            return existing

        ticket = Ticket(
            table_fq_name=table_fq_name,
            title=title,
            description=description,
            severity=severity,
            check_results=rule_ids,
        )
        self._conn.execute(
            "INSERT INTO tickets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ticket.ticket_id,
                ticket.table_fq_name,
                ticket.title,
                ticket.description,
                ticket.severity.value,
                json.dumps(ticket.check_results),
                ticket.status.value,
                ticket.created_at,
            ),
        )
        self._conn.commit()
        return ticket

    def _find_open_duplicate(self, table_fq_name: str, rule_ids: list[str]) -> Optional[Ticket]:
        rows = self._conn.execute(
            "SELECT * FROM tickets WHERE table_fq_name = ? AND status = ?",
            (table_fq_name, TicketStatus.OPEN.value),
        ).fetchall()
        for row in rows:
            ticket = self._row_to_ticket(row)
            if set(ticket.check_results) == set(rule_ids):
                return ticket
        return None

    def list_tickets(self, table_fq_name: Optional[str] = None) -> list[Ticket]:
        if table_fq_name:
            rows = self._conn.execute(
                "SELECT * FROM tickets WHERE table_fq_name = ?", (table_fq_name,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM tickets").fetchall()
        return [self._row_to_ticket(r) for r in rows]

    def resolve_ticket(self, ticket_id: str) -> None:
        self._conn.execute(
            "UPDATE tickets SET status = ? WHERE ticket_id = ?",
            (TicketStatus.RESOLVED.value, ticket_id),
        )
        self._conn.commit()

    @staticmethod
    def _row_to_ticket(row) -> Ticket:
        ticket_id, table_fq_name, title, description, severity, check_results, status, created_at = row
        return Ticket(
            ticket_id=ticket_id,
            table_fq_name=table_fq_name,
            title=title,
            description=description,
            severity=Severity(severity),
            check_results=json.loads(check_results),
            status=TicketStatus(status),
            created_at=created_at,
        )

    def close(self) -> None:
        self._conn.close()
