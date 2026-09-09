"""End-to-end demo against the SAME Chinook data, but as a flat-file source.

Run with:

    python demo/export_flatfiles.py     # once, to (re)generate the CSVs
    python demo/run_demo_flatfile.py

This is `run_demo.py`'s twin: identical data (the same three Chinook
tables -- Customer, Track, InvoiceLine -- exported byte-for-byte from
`data/chinook.db` into `data/flatfiles/*.csv` by `export_flatfiles.py`),
identical business knowledge, identical rules, run through the exact
same `DataQualityAgent` orchestrator -- the only thing that changes is
one connector: `DirectoryFlatFileConnector` instead of `SQLConnector`.
That substitution is the whole point: it's the concrete proof that
"change the source, orchestration stays the same" isn't just a design
claim.

No ANTHROPIC_API_KEY is required — this uses `MockLLMClient` by default
so the whole pipeline (profiling -> rule suggestion -> checks ->
adaptive thresholds -> tickets -> gating) runs deterministically
offline.

This script exercises the identical sequence as `run_demo.py`: metadata
scan (flat files have no lineage catalog, so lineage is honestly empty
here -- see README), business knowledge ingestion, LLM-assisted rule
proposal, human approval, batch execution with quarantine-on-critical-
failure, adaptive threshold tuning, and streaming execution of the same
rules via `InMemoryStreamSource`.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

from dq_agent.adaptive.threshold_manager import ThresholdManager
from dq_agent.connectors.registry import create_connector
from dq_agent.gating.gate import DataGate
from dq_agent.knowledge.rules_repo_json import JsonRuleRepository
from dq_agent.knowledge.store_json import JsonKnowledgeRepository
from dq_agent.llm.client import AnthropicClient, MockLLMClient
from dq_agent.llm.dq_reasoner import DQReasoner
from dq_agent.models import Rule, RuleStatus, RuleType, Severity
from dq_agent.orchestrator.agent import DataQualityAgent
from dq_agent.streaming.in_memory_source import InMemoryStreamSource
from dq_agent.ticketing.json_sink import JsonTicketSink

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
FLATFILE_DIR = DATA_DIR / "flatfiles"
STATE_DIR = DATA_DIR / "demo_state_flatfile"
SOURCE_NAME = "csv_export"  # e.g. "the nightly SFTP export from CRM/billing"


def _line(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def _build_reasoner() -> DQReasoner:
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY found — using real Claude for rule suggestion/RCA.")
        return DQReasoner(AnthropicClient())
    print("No ANTHROPIC_API_KEY set — using MockLLMClient (deterministic, offline).")
    return DQReasoner(MockLLMClient())


def build_agent() -> DataQualityAgent:
    # Every piece of state is a plain file here -- source data (CSV),
    # rules (JSON), business knowledge (JSON + a vector-store pickle),
    # tickets (JSON), and quarantine (CSV, same as run_demo.py). No
    # database anywhere in this variant.
    connector = create_connector(
        "flatfile_dir", source_name=SOURCE_NAME, directory=str(FLATFILE_DIR)
    )
    knowledge_repo = JsonKnowledgeRepository(str(STATE_DIR / "knowledge.json"))
    rule_repo = JsonRuleRepository(str(STATE_DIR / "rules.json"))
    ticket_sink = JsonTicketSink(str(STATE_DIR / "tickets.json"))
    gate = DataGate(str(STATE_DIR / "quarantine"))
    reasoner = _build_reasoner()
    return DataQualityAgent(connector, knowledge_repo, rule_repo, reasoner, ticket_sink, gate, ThresholdManager())


def seed_business_knowledge(agent: DataQualityAgent) -> None:
    _line("STEP 1: Teach the agent about these tables (business/domain knowledge -> RAG)")
    knowledge = [
        ("csv_export.Customer", "Email is required",
         "Every customer row must have a valid Email; Fax and Company are legacy fields "
         "and are expected to be null for most rows — do not flag them for completeness."),
        ("csv_export.InvoiceLine", "Referential integrity to Track",
         "Every InvoiceLine.TrackId must reference an existing Track.TrackId. This is the "
         "single most important consistency check on this table; a violation usually means "
         "the ETL loaded invoice lines before the track catalog finished loading."),
        ("csv_export.Track", "Pricing",
         "UnitPrice is in USD and should always be positive; the catalog has used 0.99 and "
         "1.99 historically, so treat 0 or negative prices as a data error, not a valid tier."),
    ]
    for table, title, content in knowledge:
        agent.add_business_knowledge(table, title, content)
        print(f"  + knowledge doc for {table}: {title!r}")


def scan_tables(agent: DataQualityAgent, tables: list[str]) -> None:
    _line("STEP 2: Scan metadata + lineage + profile")
    for table in tables:
        metadata, profile = agent.scan_table(table)
        print(f"\n{metadata.fq_name} (~{metadata.row_count_estimate} rows)")
        for col in metadata.columns:
            cp = profile.columns.get(col.name)
            if cp is None:
                continue
            print(
                f"    {col.name:16s} {col.data_type:10s} null_rate={cp.null_rate:6.2%}  "
                f"distinct_rate={cp.distinct_rate:6.2%}  pattern={cp.inferred_pattern}"
            )
        lineage = agent.connector.get_lineage(table)
        print(f"    lineage: {lineage if lineage else '(none — flat files carry no catalog/lineage metadata at all; this is the honest, expected answer for an ungoverned CSV drop)'}")


def propose_and_approve(agent: DataQualityAgent, tables: list[str]) -> None:
    _line("STEP 3: Profile-driven + LLM-refined rule proposals -> human approval -> activation")
    for table in tables:
        proposed = agent.propose_rules(table)
        print(f"\n{table}: {len(proposed)} rule(s) proposed")
        for r in proposed:
            print(f"    [{r.rule_type.value:12s}] {r.column or '(table)':16s} threshold={r.threshold:<8.4f} {r.rationale}")
        # Simulating a human reviewing and approving every proposal in this demo.
        for r in proposed:
            agent.approve_rule(r.rule_id)
        activated = agent.activate_approved_rules(table)
        print(f"  -> approved and activated {len(activated)} rule(s)")


def add_manual_referential_rule(agent: DataQualityAgent) -> None:
    _line("STEP 4: Add a hand-authored rule the statistical profiler can't infer on its own")
    rule = Rule(
        table_fq_name="csv_export.InvoiceLine",
        column="TrackId",
        rule_type=RuleType.REFERENTIAL,
        threshold=0.0,
        severity=Severity.CRITICAL,
        status=RuleStatus.ACTIVE,
        params={"ref_table": "Track", "ref_column": "TrackId"},
        rationale="Every invoice line must reference a real track (business knowledge doc).",
        created_by="demo_script",
    )
    agent.rule_repo.add_rule(rule)
    agent.rule_repo.activate_rule(rule.rule_id)
    print(f"  + {rule.rule_id}: InvoiceLine.TrackId must exist in Track.TrackId (CRITICAL)")


def run_batches(agent: DataQualityAgent, tables: list[str]) -> None:
    _line("STEP 5: Run batch checks")
    for table in tables:
        report = agent.run_batch(table)
        print(f"\n{table}: {report.row_count} rows checked, batch_id={report.batch_id}")
        for r in report.results:
            print(f"    {r.status.value.upper():5s} {(r.column or '(table)'):16s} {r.rule_type.value:12s} value={r.metric_value} threshold={r.threshold}")
        print(f"  GATE DECISION: {report.gate_result.decision.value} — {report.gate_result.reasons or 'clean'}")
        if report.gate_result.quarantined_rows:
            print(f"  -> {report.gate_result.quarantined_rows} row(s) quarantined, held back from downstream")
        for t in report.tickets:
            print(f"  TICKET [{t.ticket_id}] ({t.severity.value}): {t.title}\n    {t.description}")


def demonstrate_adaptive_threshold(agent: DataQualityAgent) -> None:
    _line("STEP 6: Adaptive thresholds")
    completeness_rules = [
        r for r in agent.rule_repo.list_rules(table_fq_name="csv_export.Customer", status=RuleStatus.ACTIVE)
        if r.rule_type == RuleType.COMPLETENESS and r.column == "Email"
    ]
    if not completeness_rules:
        print("  (no Email completeness rule active — skipping)")
        return
    rule = completeness_rules[0]

    from dq_agent.models import CheckResult, CheckStatus

    print(f"  6a) Gentle organic drift — null rate creeps up a little each run, staying")
    print(f"      inside PASS/WARN territory the whole time.")
    print(f"      Starting threshold for Customer.Email completeness: {rule.threshold:.4f}")
    # A fresh ThresholdManager for this walkthrough, so the observations
    # driving the adaptation are exactly the ones printed below (the real
    # `agent.threshold_manager` already carries history from Step 5).
    tm1 = ThresholdManager()
    for i, value in enumerate([0.018, 0.019, 0.020, 0.021, 0.022, 0.0225, 0.023]):
        result = CheckResult(
            rule_id=rule.rule_id, table_fq_name=rule.table_fq_name, column="Email",
            rule_type=RuleType.COMPLETENESS,
            status=CheckStatus.PASS if value <= rule.threshold else CheckStatus.WARN,
            metric_value=value, threshold=rule.threshold, severity=rule.severity,
        )
        tm1.record(result)
        adapted = tm1.maybe_adapt(rule, agent.rule_repo)
        status_note = ""
        if adapted:
            status_note = f"  -> auto-adjusted threshold to {adapted.threshold:.4f} (status: {adapted.status.value})"
            rule = adapted
        print(f"      run {i+1}: null_rate={value:.2%} status={result.status.value:5s}{status_note}")

    print(f"\n  6b) Sudden shift (synthetic — simulating a business change, e.g. a new market\n"
          f"      segment where Email is optional). One-off jumps like this should NOT\n"
          f"      quietly relax the rule; they should pause it for human re-approval.")
    tm2 = ThresholdManager()
    big_jump_rule = Rule(
        table_fq_name="csv_export.Customer", column="Email", rule_type=RuleType.COMPLETENESS,
        threshold=0.02, severity=Severity.WARN, status=RuleStatus.ACTIVE, adaptive=True,
    )
    agent.rule_repo.add_rule(big_jump_rule)
    agent.rule_repo.activate_rule(big_jump_rule.rule_id)
    for i, value in enumerate([0.30, 0.32, 0.29, 0.31, 0.30, 0.33]):
        result = CheckResult(
            rule_id=big_jump_rule.rule_id, table_fq_name=big_jump_rule.table_fq_name,
            column="Email", rule_type=RuleType.COMPLETENESS, status=CheckStatus.WARN,
            metric_value=value, threshold=big_jump_rule.threshold, severity=Severity.WARN,
        )
        tm2.record(result)
        adapted = tm2.maybe_adapt(big_jump_rule, agent.rule_repo)
        note = ""
        if adapted:
            note = f"  -> threshold moved to {adapted.threshold:.4f}, status demoted to {adapted.status.value} (needs human re-approval)"
            big_jump_rule = adapted
        print(f"      run {i+1}: null_rate={value:.2%}{note}")

    print(f"\n  Threshold history for {big_jump_rule.rule_id}:")
    for h in agent.rule_repo.get_threshold_history(big_jump_rule.rule_id):
        print(f"    {h['old_threshold']:.4f} -> {h['new_threshold']:.4f}  auto={h['auto_adjusted']}  reapproval_needed={h['required_reapproval']}  ({h['reason']})")


def demonstrate_forced_failure_and_ticketing(agent: DataQualityAgent) -> None:
    _line("STEP 6c: Force a CRITICAL failure to prove gating + ticketing end-to-end")
    df = agent.connector.fetch_batch("InvoiceLine")
    corrupted = df.copy()
    # Corrupt enough rows that the violation rate clears the rule engine's
    # warn buffer and actually registers as FAIL, not just WARN.
    corrupted.loc[corrupted.index[:15], "TrackId"] = 999999999  # doesn't exist in Track
    report = agent.run_dataframe("InvoiceLine", corrupted)
    referential_result = next(r for r in report.results if r.rule_type == RuleType.REFERENTIAL)
    print(f"  referential check: status={referential_result.status.value} value={referential_result.metric_value:.4%}")
    print(f"  GATE DECISION: {report.gate_result.decision.value} — {report.gate_result.reasons}")
    print(f"  quarantined rows: {report.gate_result.quarantined_rows}")
    for t in report.tickets:
        print(f"  TICKET [{t.ticket_id}] ({t.severity.value}, status={t.status.value}): {t.title}\n    {t.description}")


def demonstrate_streaming(agent: DataQualityAgent) -> None:
    _line("STEP 7: Same orchestration, streaming source (proves batch/streaming parity)")
    df = agent.connector.fetch_batch("Track")
    stream = InMemoryStreamSource(SOURCE_NAME, "track_topic", df, batch_size=500)
    reports = agent.run_stream("Track", stream, max_batches=None)
    print(f"  Processed {len(reports)} micro-batches, {sum(r.row_count for r in reports)} rows total")
    for r in reports:
        statuses = {res.status.value for res in r.results}
        print(f"    batch {r.batch_id[:14]}...: {r.row_count} rows, statuses={statuses}, gate={r.gate_result.decision.value}")


def main() -> None:
    if not FLATFILE_DIR.exists() or not any(FLATFILE_DIR.iterdir()):
        raise SystemExit(
            f"Expected CSV files at {FLATFILE_DIR} — run `python demo/export_flatfiles.py` first "
            f"(needs data/chinook.db, same as run_demo.py)."
        )

    if STATE_DIR.exists():
        shutil.rmtree(STATE_DIR)  # fresh state each run, so the demo is reproducible
    STATE_DIR.mkdir(parents=True)

    agent = build_agent()
    tables = ["Customer", "Track", "InvoiceLine"]

    seed_business_knowledge(agent)
    scan_tables(agent, tables)
    propose_and_approve(agent, tables)
    add_manual_referential_rule(agent)
    run_batches(agent, tables)
    demonstrate_adaptive_threshold(agent)
    demonstrate_forced_failure_and_ticketing(agent)
    demonstrate_streaming(agent)

    _line("DONE")
    print(f"State persisted under {STATE_DIR} (rules.json, knowledge.json + .vec.pkl, tickets.json, quarantine/*.csv) — every file is plain text/CSV, inspectable with any editor.")
    print(f"Source data read from {FLATFILE_DIR} (plain CSVs).")
    print("No database anywhere in this run -- source, rules, knowledge, and tickets are all flat files.")


if __name__ == "__main__":
    main()
