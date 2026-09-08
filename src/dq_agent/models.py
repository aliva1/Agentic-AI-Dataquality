"""Shared domain model for the whole agent.

Every module (connectors, profiling, knowledge, checks, adaptive,
ticketing, gating, orchestrator) speaks these dataclasses so that the
pieces stay decoupled and swappable. Nothing in here talks to a database,
an LLM, or a file — it's pure data.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# Metadata & lineage
# --------------------------------------------------------------------------


@dataclass
class ColumnMetadata:
    name: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False
    comment: Optional[str] = None


@dataclass
class TableMetadata:
    source_name: str          # e.g. "prod_postgres", "sales_lakehouse"
    schema: Optional[str]
    table: str
    columns: list[ColumnMetadata] = field(default_factory=list)
    row_count_estimate: Optional[int] = None
    comment: Optional[str] = None

    @property
    def fq_name(self) -> str:
        parts = [p for p in (self.source_name, self.schema, self.table) if p]
        return ".".join(parts)


@dataclass
class LineageEdge:
    """A single upstream -> downstream dependency edge (best-effort)."""

    upstream: str      # fully-qualified object name
    downstream: str    # fully-qualified object name
    relationship: str = "derives_from"  # e.g. view definition, FK, ETL step
    source_of_info: str = "unknown"     # e.g. "information_schema", "unity_catalog", "user_declared"


# --------------------------------------------------------------------------
# Profiling
# --------------------------------------------------------------------------


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    row_count: int
    null_count: int
    distinct_count: int
    min_value: Any = None
    max_value: Any = None
    mean: Optional[float] = None
    stddev: Optional[float] = None
    top_values: list[tuple[Any, int]] = field(default_factory=list)
    inferred_pattern: Optional[str] = None  # e.g. "email", "uuid", "date:%Y-%m-%d", "integer_id"
    is_candidate_key: bool = False

    @property
    def null_rate(self) -> float:
        return 0.0 if self.row_count == 0 else self.null_count / self.row_count

    @property
    def distinct_rate(self) -> float:
        return 0.0 if self.row_count == 0 else self.distinct_count / self.row_count


@dataclass
class TableProfile:
    table_fq_name: str
    row_count: int
    columns: dict[str, ColumnProfile] = field(default_factory=dict)
    candidate_grain: list[str] = field(default_factory=list)  # columns that together look like a grain/PK
    profiled_at: float = field(default_factory=time.time)


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------


class RuleType(str, Enum):
    COMPLETENESS = "completeness"     # null rate <= threshold
    UNIQUENESS = "uniqueness"         # distinct rate >= threshold (or duplicate count <= threshold)
    RANGE = "range"                   # numeric value between min/max
    PATTERN = "pattern"               # regex match rate >= threshold
    FRESHNESS = "freshness"           # max(timestamp col) within threshold of now
    REFERENTIAL = "referential"       # values exist in a reference table/column
    CUSTOM_SQL = "custom_sql"         # user/LLM supplied SQL predicate, threshold = max violation rate


class RuleStatus(str, Enum):
    PROPOSED = "proposed"     # suggested by profiler/LLM, awaiting human approval
    APPROVED = "approved"     # approved, not yet active (e.g. waiting for next run)
    ACTIVE = "active"         # running in production checks
    REJECTED = "rejected"
    RETIRED = "retired"


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


@dataclass
class Rule:
    table_fq_name: str
    column: Optional[str]
    rule_type: RuleType
    threshold: float
    severity: Severity = Severity.WARN
    status: RuleStatus = RuleStatus.PROPOSED
    params: dict[str, Any] = field(default_factory=dict)  # e.g. {"regex": "...", "ref_table": "...", "ref_column": "..."}
    rationale: str = ""            # why this rule/threshold was proposed (from profiler or LLM)
    adaptive: bool = True          # whether ThresholdManager may auto-tune this rule's threshold
    created_by: str = "system"
    rule_id: str = field(default_factory=lambda: _new_id("rule"))
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 1


@dataclass
class ProposedRule:
    """A rule suggestion that hasn't been persisted/approved yet."""

    table_fq_name: str
    column: Optional[str]
    rule_type: RuleType
    threshold: float
    severity: Severity
    rationale: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_rule(self, created_by: str = "dq_reasoner") -> Rule:
        return Rule(
            table_fq_name=self.table_fq_name,
            column=self.column,
            rule_type=self.rule_type,
            threshold=self.threshold,
            severity=self.severity,
            status=RuleStatus.PROPOSED,
            params=dict(self.params),
            rationale=self.rationale,
            created_by=created_by,
        )


# --------------------------------------------------------------------------
# Check execution
# --------------------------------------------------------------------------


class CheckStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    ERROR = "error"  # the check itself couldn't run (missing column, bad SQL, etc.)


@dataclass
class CheckResult:
    rule_id: str
    table_fq_name: str
    column: Optional[str]
    rule_type: RuleType
    status: CheckStatus
    metric_value: Optional[float]
    threshold: float
    severity: Severity
    message: str = ""
    batch_id: str = field(default_factory=lambda: _new_id("batch"))
    checked_at: float = field(default_factory=time.time)
    sample_violations: list[Any] = field(default_factory=list)


# --------------------------------------------------------------------------
# Knowledge
# --------------------------------------------------------------------------


@dataclass
class KnowledgeDoc:
    table_fq_name: str
    title: str
    content: str
    doc_id: str = field(default_factory=lambda: _new_id("kdoc"))
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


# --------------------------------------------------------------------------
# Ticketing & gating
# --------------------------------------------------------------------------


class TicketStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class Ticket:
    table_fq_name: str
    title: str
    description: str
    severity: Severity
    check_results: list[str]  # rule_ids or CheckResult ids referenced
    ticket_id: str = field(default_factory=lambda: _new_id("tkt"))
    status: TicketStatus = TicketStatus.OPEN
    created_at: float = field(default_factory=time.time)


class GateDecision(str, Enum):
    PASS = "pass"
    PASS_WITH_WARNINGS = "pass_with_warnings"
    BLOCK = "block"


@dataclass
class GateResult:
    table_fq_name: str
    batch_id: str
    decision: GateDecision
    reasons: list[str] = field(default_factory=list)
    quarantined_rows: int = 0
