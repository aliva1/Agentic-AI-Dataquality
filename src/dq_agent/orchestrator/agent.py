"""`DataQualityAgent`: the orchestration core everything else plugs into.

This class is deliberately the only place that knows about all the
other modules. Everything it depends on is injected (connector,
knowledge repo, rule repo, reasoner, ticket sink, gate, threshold
manager), so:

- Changing data source = pass a different `DataSourceConnector`
  (built via `connectors.registry.create_connector`). Nothing else here
  changes.
- `run_batch()` and `run_stream()` both bottom out in the same private
  `_process_batch()` — that's the "same orchestration for batch and
  streaming" requirement, in one method.
- Swapping the LLM, the ticket system, or the vector store backing the
  knowledge repo never touches this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from dq_agent.checks import engine as check_engine
from dq_agent.connectors.base import DataSourceConnector
from dq_agent.gating.gate import DataGate
from dq_agent.knowledge.rules_repo import RuleRepository
from dq_agent.knowledge.store import KnowledgeRepository
from dq_agent.llm.dq_reasoner import DQReasoner
from dq_agent.models import (
    CheckResult,
    CheckStatus,
    GateResult,
    KnowledgeDoc,
    ProposedRule,
    Rule,
    RuleStatus,
    RuleType,
    Severity,
    TableMetadata,
    TableProfile,
    Ticket,
)
from dq_agent.profiling.profiler import profile_table
from dq_agent.streaming.base import StreamSource
from dq_agent.adaptive.threshold_manager import ThresholdManager
from dq_agent.ticketing.base import TicketSink

_SEVERITY_ORDER = {Severity.INFO: 0, Severity.WARN: 1, Severity.CRITICAL: 2}


@dataclass
class BatchReport:
    table_fq_name: str
    batch_id: str
    row_count: int
    results: list[CheckResult] = field(default_factory=list)
    gate_result: Optional[GateResult] = None
    tickets: list[Ticket] = field(default_factory=list)
    adapted_rules: list[Rule] = field(default_factory=list)


class DataQualityAgent:
    def __init__(
        self,
        connector: DataSourceConnector,
        knowledge_repo: KnowledgeRepository,
        rule_repo: RuleRepository,
        reasoner: DQReasoner,
        ticket_sink: TicketSink,
        gate: Optional[DataGate] = None,
        threshold_manager: Optional[ThresholdManager] = None,
    ):
        self.connector = connector
        self.knowledge_repo = knowledge_repo
        self.rule_repo = rule_repo
        self.reasoner = reasoner
        self.ticket_sink = ticket_sink
        self.gate = gate or DataGate()
        self.threshold_manager = threshold_manager or ThresholdManager()

    # -----------------------------------------------------------------
    # Knowledge management (user-facing: teach the agent about a table)
    # -----------------------------------------------------------------
    def add_business_knowledge(self, table_fq_name: str, title: str, content: str, tags: Optional[list[str]] = None) -> KnowledgeDoc:
        doc = KnowledgeDoc(table_fq_name=table_fq_name, title=title, content=content, tags=tags or [])
        self.knowledge_repo.add_doc(doc)
        return doc

    # -----------------------------------------------------------------
    # Scan: metadata + lineage + profile
    # -----------------------------------------------------------------
    def scan_table(self, table: str, sample_size: int = 5000) -> tuple[TableMetadata, TableProfile]:
        metadata = self.connector.get_metadata(table)
        sample = self.connector.fetch_sample(table, limit=sample_size)
        profile = profile_table(sample, table_fq_name=metadata.fq_name)
        return metadata, profile

    # -----------------------------------------------------------------
    # Rule lifecycle: propose (LLM+stats) -> approve -> activate
    # -----------------------------------------------------------------
    def propose_rules(self, table: str, sample_size: int = 5000) -> list[Rule]:
        _, profile = self.scan_table(table, sample_size=sample_size)
        docs = self.knowledge_repo.get_docs_for_table(profile.table_fq_name)
        proposals: list[ProposedRule] = self.reasoner.suggest_rules(profile, docs)
        return [self.rule_repo.add_rule(p.to_rule()) for p in proposals]

    def pending_rules(self, table: str) -> list[Rule]:
        return self.rule_repo.list_rules(table_fq_name=self._fq(table), status=RuleStatus.PROPOSED)

    def approve_rule(self, rule_id: str) -> Rule:
        return self.rule_repo.approve_rule(rule_id)

    def reject_rule(self, rule_id: str) -> Rule:
        return self.rule_repo.reject_rule(rule_id)

    def activate_approved_rules(self, table: str) -> list[Rule]:
        approved = self.rule_repo.list_rules(table_fq_name=self._fq(table), status=RuleStatus.APPROVED)
        return [self.rule_repo.activate_rule(r.rule_id) for r in approved]

    def _fq(self, table: str) -> str:
        """Resolve a connector-local table name to the fully-qualified
        name rules/knowledge are keyed by (`source.[schema.]table`).

        Callers may also pass an already-fully-qualified name (as the
        CLI and tests sometimes do); in that case metadata lookup will
        simply fail to find it as a raw table, so we accept it as-is
        when it already starts with the connector's source name.
        """
        if table.startswith(f"{self.connector.source_name}."):
            return table
        return self.connector.get_metadata(table).fq_name

    # -----------------------------------------------------------------
    # Running checks — batch and streaming share `_process_batch`
    # -----------------------------------------------------------------
    def run_batch(self, table: str) -> BatchReport:
        df = self.connector.fetch_batch(table)
        fq = self._fq(table)
        rules = self.rule_repo.list_rules(table_fq_name=fq, status=RuleStatus.ACTIVE)
        return self._process_batch(fq, df, rules)

    def run_dataframe(self, table: str, df: pd.DataFrame) -> BatchReport:
        """Run the ACTIVE rules for `table` against a caller-supplied
        DataFrame instead of pulling one from the connector.

        Useful for testing "what if this batch looked like X" (a
        one-off bad file, a hand-built regression fixture) without
        needing that data to actually exist in the source system.
        """
        fq = self._fq(table)
        rules = self.rule_repo.list_rules(table_fq_name=fq, status=RuleStatus.ACTIVE)
        return self._process_batch(fq, df, rules)

    def run_stream(
        self, table: str, stream_source: StreamSource, max_batches: Optional[int] = None
    ) -> list[BatchReport]:
        fq = self._fq(table)
        rules = self.rule_repo.list_rules(table_fq_name=fq, status=RuleStatus.ACTIVE)
        reports = []
        with stream_source:
            for micro_batch in stream_source.consume(max_batches=max_batches):
                reports.append(self._process_batch(fq, micro_batch, rules))
        return reports

    def _process_batch(self, table: str, df: pd.DataFrame, rules: list[Rule]) -> BatchReport:
        reference_values = self._resolve_reference_values(rules)
        results = check_engine.evaluate_all(rules, df, reference_values_by_rule=reference_values)
        batch_id = results[0].batch_id if results else "empty_batch"

        adapted_rules = []
        for rule, result in zip(rules, results):
            self.threshold_manager.record(result)
            adapted = self.threshold_manager.maybe_adapt(rule, self.rule_repo)
            if adapted:
                adapted_rules.append(adapted)

        gate_result = self.gate.evaluate(table, batch_id, df, results)
        tickets = self._handle_failures(table, results)

        return BatchReport(
            table_fq_name=table,
            batch_id=batch_id,
            row_count=len(df),
            results=results,
            gate_result=gate_result,
            tickets=tickets,
            adapted_rules=adapted_rules,
        )

    def _resolve_reference_values(self, rules: list[Rule]) -> dict[str, set]:
        out: dict[str, set] = {}
        for rule in rules:
            if rule.rule_type != RuleType.REFERENTIAL:
                continue
            ref_table = rule.params.get("ref_table")
            ref_column = rule.params.get("ref_column")
            if not ref_table or not ref_column:
                continue
            try:
                ref_df = self.connector.fetch_batch(ref_table)
                out[rule.rule_id] = set(ref_df[ref_column].dropna())
            except Exception:
                continue  # engine falls back to rule.params["ref_values"] or an empty set
        return out

    def _handle_failures(self, table: str, results: list[CheckResult]) -> list[Ticket]:
        problems = [r for r in results if r.status in (CheckStatus.FAIL, CheckStatus.WARN)]
        if not problems:
            return []

        docs = self.knowledge_repo.get_docs_for_table(table)
        worst = max(problems, key=lambda r: (_SEVERITY_ORDER[r.severity], r.status == CheckStatus.FAIL))
        history = self.rule_repo.get_threshold_history(worst.rule_id)
        rca = self.reasoner.explain_failure(worst, docs, history)

        overall_severity = max((r.severity for r in problems), key=lambda s: _SEVERITY_ORDER[s])
        title, description = self.reasoner.draft_ticket(problems, rca)
        ticket = self.ticket_sink.create_ticket(table, title, description, overall_severity, problems)
        return [ticket]
