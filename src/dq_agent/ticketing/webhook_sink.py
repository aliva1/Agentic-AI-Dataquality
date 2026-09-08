"""Generic REST/webhook ticket sink for plugging into a real ITSM system.

This is intentionally thin — a real integration (ServiceNow, Jira,
Lenovo's internal ITSM bearer-token API, etc.) has enough
vendor-specific quirks (auth, field mapping, approval workflow) that
they belong in a small subclass or config, not baked into this proof of
concept. What matters architecturally is that it satisfies the same
`TicketSink` interface as `LocalTicketSink`, so the orchestrator never
needs to know which one it's talking to.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from dq_agent.models import CheckResult, Severity, Ticket
from dq_agent.ticketing.base import TicketSink


class WebhookTicketSink(TicketSink):
    def __init__(
        self,
        webhook_url: str,
        headers: Optional[dict[str, str]] = None,
        post_fn: Optional[Callable[..., Any]] = None,
    ):
        self.webhook_url = webhook_url
        self.headers = headers or {}
        # `post_fn` is injectable for testing without a live endpoint;
        # defaults to `requests.post`, imported lazily so `requests`
        # stays an optional dependency for users who never touch this sink.
        self._post_fn = post_fn
        self._tickets: list[Ticket] = []  # local mirror, since we can't "list" a remote ITSM here

    def _post(self, payload: dict) -> None:
        if self._post_fn is not None:
            self._post_fn(self.webhook_url, json=payload, headers=self.headers)
            return
        import requests

        requests.post(self.webhook_url, json=payload, headers=self.headers, timeout=10)

    def create_ticket(
        self,
        table_fq_name: str,
        title: str,
        description: str,
        severity: Severity,
        results: list[CheckResult],
    ) -> Ticket:
        ticket = Ticket(
            table_fq_name=table_fq_name,
            title=title,
            description=description,
            severity=severity,
            check_results=[r.rule_id for r in results],
        )
        self._post(
            {
                "ticket_id": ticket.ticket_id,
                "table": table_fq_name,
                "title": title,
                "description": description,
                "severity": severity.value,
                "rule_ids": ticket.check_results,
            }
        )
        self._tickets.append(ticket)
        return ticket

    def list_tickets(self, table_fq_name: Optional[str] = None) -> list[Ticket]:
        if table_fq_name is None:
            return list(self._tickets)
        return [t for t in self._tickets if t.table_fq_name == table_fq_name]
