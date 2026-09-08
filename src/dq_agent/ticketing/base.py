"""Ticket sink interface: where a DQ finding goes to become a work item.

Kept deliberately narrow — `create_ticket` in, a `Ticket` back — so a
real deployment can point this at ServiceNow/Jira/an internal ITSM API
(matching the approval-gated ITSM pattern from Lenovo's own DQ
platform) by implementing this one method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from dq_agent.models import CheckResult, Severity, Ticket


class TicketSink(ABC):
    @abstractmethod
    def create_ticket(
        self,
        table_fq_name: str,
        title: str,
        description: str,
        severity: Severity,
        results: list[CheckResult],
    ) -> Ticket: ...

    @abstractmethod
    def list_tickets(self, table_fq_name: str | None = None) -> list[Ticket]: ...
