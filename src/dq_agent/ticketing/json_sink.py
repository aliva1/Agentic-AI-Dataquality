"""JSON-file-backed twin of `LocalTicketSink`.

Implements the same `TicketSink` interface (`create_ticket`,
`list_tickets`) plus the same `resolve_ticket` convenience the SQLite
sink has, backed by one plain `.json` file instead of a SQLite table.
Same one-ticket-per-incident + duplicate-open-ticket prevention as
`LocalTicketSink` -- only the storage medium changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from dq_agent.models import CheckResult, Severity, Ticket, TicketStatus
from dq_agent.ticketing.base import TicketSink


def _ticket_to_dict(t: Ticket) -> dict:
    return {
        "ticket_id": t.ticket_id,
        "table_fq_name": t.table_fq_name,
        "title": t.title,
        "description": t.description,
        "severity": t.severity.value,
        "check_results": t.check_results,
        "status": t.status.value,
        "created_at": t.created_at,
    }


def _dict_to_ticket(d: dict) -> Ticket:
    return Ticket(
        ticket_id=d["ticket_id"],
        table_fq_name=d["table_fq_name"],
        title=d["title"],
        description=d["description"],
        severity=Severity(d["severity"]),
        check_results=d.get("check_results") or [],
        status=TicketStatus(d["status"]),
        created_at=d["created_at"],
    )


class JsonTicketSink(TicketSink):
    """Drop-in alternative to `LocalTicketSink`, backed by one JSON file
    (`{"tickets": {"<ticket_id>": {...}, ...}}`).
    """

    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        if self.json_path.exists():
            self._data = json.loads(self.json_path.read_text() or "{}")
        else:
            self._data = {}
        self._data.setdefault("tickets", {})
        self._flush()

    def _flush(self) -> None:
        self.json_path.write_text(json.dumps(self._data, indent=2, sort_keys=True))

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
        self._data["tickets"][ticket.ticket_id] = _ticket_to_dict(ticket)
        self._flush()
        return ticket

    def _find_open_duplicate(self, table_fq_name: str, rule_ids: list[str]) -> Optional[Ticket]:
        for d in self._data["tickets"].values():
            if d["table_fq_name"] != table_fq_name or d["status"] != TicketStatus.OPEN.value:
                continue
            if set(d.get("check_results") or []) == set(rule_ids):
                return _dict_to_ticket(d)
        return None

    def list_tickets(self, table_fq_name: Optional[str] = None) -> list[Ticket]:
        tickets = [_dict_to_ticket(d) for d in self._data["tickets"].values()]
        if table_fq_name:
            tickets = [t for t in tickets if t.table_fq_name == table_fq_name]
        return tickets

    def resolve_ticket(self, ticket_id: str) -> None:
        d = self._data["tickets"].get(ticket_id)
        if d is not None:
            d["status"] = TicketStatus.RESOLVED.value
            self._flush()

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass
