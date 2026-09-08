"""Where the LLM actually earns its keep in this agent.

Three jobs, each combining a deterministic/statistical baseline with an
LLM refinement pass so the pipeline degrades gracefully without a
configured model (see `MockLLMClient`) but gets genuinely smarter with
one:

1. `suggest_rules`   — turn a data profile + retrieved business
   knowledge into proposed DQ rules with thresholds, for human approval.
2. `explain_failure` — root-cause a failed check, with a confidence score.
3. `draft_ticket`    — turn one or more failures into a ticket a human
   (or the real ITSM system) can act on.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from dq_agent.llm.client import LLMClient
from dq_agent.models import (
    CheckResult,
    KnowledgeDoc,
    ProposedRule,
    RuleType,
    Severity,
    TableProfile,
)


@dataclass
class RootCauseAnalysis:
    explanation: str
    confidence: float  # 0.0 - 1.0
    suggested_action: str = ""


class DQReasoner:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    # -----------------------------------------------------------------
    # 1. Rule suggestion
    # -----------------------------------------------------------------
    def suggest_rules(
        self,
        profile: TableProfile,
        knowledge_docs: list[KnowledgeDoc] | None = None,
    ) -> list[ProposedRule]:
        baseline = _statistical_baseline_rules(profile)
        if not baseline:
            return []

        prompt = _build_rule_suggestion_prompt(profile, knowledge_docs or [], baseline)
        raw = self.llm.complete(
            prompt,
            system=(
                "You are a data quality engineer. Review the candidate rules and "
                "business context, then respond with ONLY a JSON array refining "
                "them (you may adjust thresholds/severity/rationale, drop ones that "
                "don't make sense, or add new ones implied by the business notes). "
                "Each item: {table, column, rule_type, threshold, severity, rationale}."
            ),
        )
        refined = _parse_rule_json(raw, profile.table_fq_name)
        return refined if refined else baseline

    # -----------------------------------------------------------------
    # 2. Root-cause analysis on a failed check
    # -----------------------------------------------------------------
    def explain_failure(
        self,
        result: CheckResult,
        knowledge_docs: list[KnowledgeDoc] | None = None,
        recent_history: list[dict] | None = None,
    ) -> RootCauseAnalysis:
        prompt = _build_rca_prompt(result, knowledge_docs or [], recent_history or [])
        raw = self.llm.complete(
            prompt,
            system=(
                "You are investigating a data quality failure. Give the most likely "
                "root cause in 2-3 sentences, then on a new line write "
                "'Confidence: <0-1 number>', then on another line "
                "'Action: <one concrete next step>'."
            ),
        )
        return _parse_rca(raw)

    # -----------------------------------------------------------------
    # 3. Ticket drafting
    # -----------------------------------------------------------------
    def draft_ticket(self, results: list[CheckResult], rca: RootCauseAnalysis | None = None) -> tuple[str, str]:
        prompt = _build_ticket_prompt(results, rca)
        raw = self.llm.complete(
            prompt,
            system=(
                "Draft a concise ticket title (one line) and description for a data "
                "quality incident, for an on-call data engineer audience. Format as:\n"
                "Title: <title>\nDescription: <description>"
            ),
        )
        return _parse_ticket(raw, results)


# =======================================================================
# Statistical baseline (works with zero LLM configuration)
# =======================================================================


def _statistical_baseline_rules(profile: TableProfile) -> list[ProposedRule]:
    rules: list[ProposedRule] = []
    table = profile.table_fq_name

    for col_name, cp in profile.columns.items():
        # Completeness: only propose if the column isn't already 100% null
        # (that's more likely a schema/ETL problem than a threshold to track).
        if 0 <= cp.null_rate < 1.0:
            threshold = round(min(0.99, cp.null_rate + 0.02), 4)
            severity = Severity.CRITICAL if cp.null_rate < 0.01 else Severity.WARN
            rules.append(
                ProposedRule(
                    table_fq_name=table,
                    column=col_name,
                    rule_type=RuleType.COMPLETENESS,
                    threshold=threshold,
                    severity=severity,
                    rationale=(
                        f"Observed null rate {cp.null_rate:.2%}; proposing a "
                        f"max-null-rate threshold of {threshold:.2%} with headroom."
                    ),
                )
            )

        # Uniqueness for candidate keys / grain columns.
        if cp.is_candidate_key:
            rules.append(
                ProposedRule(
                    table_fq_name=table,
                    column=col_name,
                    rule_type=RuleType.UNIQUENESS,
                    threshold=0.999,
                    severity=Severity.CRITICAL,
                    rationale=f"{col_name} looks like a candidate key (all distinct, no nulls).",
                )
            )

        # Range for numeric columns with a sane spread.
        if cp.min_value is not None and cp.max_value is not None and cp.stddev:
            rules.append(
                ProposedRule(
                    table_fq_name=table,
                    column=col_name,
                    rule_type=RuleType.RANGE,
                    threshold=0.0,  # 0 violations tolerated outside [min, max] bounds by default
                    severity=Severity.WARN,
                    rationale=f"Observed range [{cp.min_value}, {cp.max_value}]; flagging values outside it.",
                    params={
                        "min": cp.min_value - abs(cp.stddev) * 3,
                        "max": cp.max_value + abs(cp.stddev) * 3,
                    },
                )
            )

        # Pattern conformance where we detected a consistent shape.
        if cp.inferred_pattern:
            rules.append(
                ProposedRule(
                    table_fq_name=table,
                    column=col_name,
                    rule_type=RuleType.PATTERN,
                    threshold=0.95,
                    severity=Severity.WARN,
                    rationale=f"{col_name} values match the '{cp.inferred_pattern}' pattern in the sample.",
                    params={"pattern_name": cp.inferred_pattern},
                )
            )

    return rules


# =======================================================================
# Prompt building
# =======================================================================


def _profile_summary(profile: TableProfile) -> str:
    lines = [f"Table: {profile.table_fq_name} ({profile.row_count} rows)"]
    for name, cp in profile.columns.items():
        lines.append(
            f"- {name} ({cp.dtype}): null_rate={cp.null_rate:.2%}, "
            f"distinct_rate={cp.distinct_rate:.2%}, pattern={cp.inferred_pattern}, "
            f"candidate_key={cp.is_candidate_key}"
        )
    return "\n".join(lines)


def _knowledge_summary(docs: list[KnowledgeDoc]) -> str:
    if not docs:
        return "(no business knowledge documents on file for this table)"
    return "\n".join(f"- {d.title}: {d.content}" for d in docs)


def _build_rule_suggestion_prompt(
    profile: TableProfile, docs: list[KnowledgeDoc], baseline: list[ProposedRule]
) -> str:
    baseline_json = json.dumps(
        [
            {
                "table": r.table_fq_name,
                "column": r.column,
                "rule_type": r.rule_type.value,
                "threshold": r.threshold,
                "severity": r.severity.value,
                "rationale": r.rationale,
            }
            for r in baseline
        ],
        indent=2,
    )
    return (
        f"Propose data quality rules for this table.\n\n"
        f"DATA PROFILE:\n{_profile_summary(profile)}\n\n"
        f"BUSINESS KNOWLEDGE:\n{_knowledge_summary(docs)}\n\n"
        f"STATISTICAL BASELINE CANDIDATES:\n{baseline_json}\n"
    )


def _build_rca_prompt(result: CheckResult, docs: list[KnowledgeDoc], history: list[dict]) -> str:
    history_str = "\n".join(
        f"- {h['changed_at']}: {h['old_threshold']} -> {h['new_threshold']} ({h['reason']})"
        for h in history[-5:]
    ) or "(no threshold history)"
    return (
        f"A data quality check failed.\n\n"
        f"Table: {result.table_fq_name}\nColumn: {result.column}\n"
        f"Rule type: {result.rule_type.value}\nStatus: {result.status.value}\n"
        f"Metric value: {result.metric_value}\nThreshold: {result.threshold}\n"
        f"Message: {result.message}\n\n"
        f"BUSINESS KNOWLEDGE:\n{_knowledge_summary(docs)}\n\n"
        f"RECENT THRESHOLD HISTORY:\n{history_str}\n"
    )


def _build_ticket_prompt(results: list[CheckResult], rca: RootCauseAnalysis | None) -> str:
    results_str = "\n".join(
        f"- {r.table_fq_name}.{r.column}: {r.rule_type.value} failed "
        f"(value={r.metric_value}, threshold={r.threshold}, severity={r.severity.value})"
        for r in results
    )
    rca_str = f"\nLikely root cause: {rca.explanation} (confidence {rca.confidence:.0%})" if rca else ""
    return f"Failed checks:\n{results_str}\n{rca_str}\n"


# =======================================================================
# Response parsing
# =======================================================================


def _parse_rule_json(raw: str, table_fq_name: str) -> list[ProposedRule]:
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    rules = []
    for item in items:
        try:
            rules.append(
                ProposedRule(
                    table_fq_name=item.get("table", table_fq_name),
                    column=item.get("column"),
                    rule_type=RuleType(item["rule_type"]),
                    threshold=float(item["threshold"]),
                    severity=Severity(item.get("severity", "warn")),
                    rationale=item.get("rationale", "(LLM-suggested)"),
                    params=item.get("params", {}),
                )
            )
        except (KeyError, ValueError):
            continue
    return rules


_CONFIDENCE_RE = re.compile(r"confidence[:\s]+([0-9]*\.?[0-9]+)", re.IGNORECASE)
_ACTION_RE = re.compile(r"action[:\s]+(.+)", re.IGNORECASE)


def _parse_rca(raw: str) -> RootCauseAnalysis:
    conf_match = _CONFIDENCE_RE.search(raw)
    confidence = float(conf_match.group(1)) if conf_match else 0.5
    confidence = max(0.0, min(1.0, confidence if confidence <= 1 else confidence / 100))
    action_match = _ACTION_RE.search(raw)
    action = action_match.group(1).strip() if action_match else ""
    explanation = raw.split("Confidence:")[0].split("confidence:")[0].strip()
    return RootCauseAnalysis(explanation=explanation or raw.strip(), confidence=confidence, suggested_action=action)


def _parse_ticket(raw: str, results: list[CheckResult]) -> tuple[str, str]:
    title_match = re.search(r"Title:\s*(.+)", raw)
    desc_match = re.search(r"Description:\s*(.+)", raw, re.DOTALL)
    if title_match and desc_match:
        return title_match.group(1).strip(), desc_match.group(1).strip()
    # Fallback: build a serviceable ticket ourselves rather than failing.
    tables = ", ".join(sorted({r.table_fq_name for r in results}))
    title = f"Data quality failures on {tables}"
    return title, raw.strip()
