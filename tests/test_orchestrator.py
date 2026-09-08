from dq_agent.connectors.registry import create_connector
from dq_agent.gating.gate import DataGate
from dq_agent.knowledge.rules_repo import RuleRepository
from dq_agent.knowledge.store import KnowledgeRepository
from dq_agent.llm.client import MockLLMClient
from dq_agent.llm.dq_reasoner import DQReasoner
from dq_agent.models import CheckStatus, GateDecision, RuleStatus, RuleType, Severity
from dq_agent.orchestrator.agent import DataQualityAgent
from dq_agent.streaming.in_memory_source import InMemoryStreamSource
from dq_agent.ticketing.local_sink import LocalTicketSink


def _build_agent(tmp_sqlite_db, workdir, source_name="demo"):
    connector = create_connector("sqlite", source_name, connection_string=tmp_sqlite_db)
    knowledge_repo = KnowledgeRepository(f"{workdir}/knowledge.db")
    rule_repo = RuleRepository(f"{workdir}/rules.db")
    ticket_sink = LocalTicketSink(f"{workdir}/tickets.db")
    gate = DataGate(f"{workdir}/quarantine")
    reasoner = DQReasoner(MockLLMClient())
    return DataQualityAgent(connector, knowledge_repo, rule_repo, reasoner, ticket_sink, gate)


def test_full_lifecycle_scan_propose_approve_activate_run(tmp_sqlite_db, workdir):
    agent = _build_agent(tmp_sqlite_db, workdir)

    metadata, profile = agent.scan_table("customers")
    assert metadata.table == "customers"
    assert profile.row_count == 5

    agent.add_business_knowledge(
        "demo.customers", "PII note", "email is scrubbed for non-prod environments"
    )

    proposed = agent.propose_rules("customers")
    assert len(proposed) > 0
    assert all(r.status == RuleStatus.PROPOSED for r in proposed)

    for r in proposed:
        agent.approve_rule(r.rule_id)
    activated = agent.activate_approved_rules("demo.customers")
    assert len(activated) == len(proposed)

    report = agent.run_batch("customers")
    assert report.row_count == 5
    assert len(report.results) == len(proposed)
    # Proposed thresholds are calibrated with headroom above the profiled
    # sample by design (see reasoner._statistical_baseline_rules), so a
    # freshly-approved rule set is expected to pass against the very data
    # it was derived from — drift is what future runs should catch.
    assert all(r.status == CheckStatus.PASS for r in report.results)

    # Prove drift detection works: tighten one rule past what the data
    # actually looks like and confirm the same pipeline now flags it.
    email_rule = next(r for r in proposed if r.column == "email")
    agent.rule_repo.update_threshold(email_rule.rule_id, 0.0, reason="tightened for test", auto_adjusted=False)
    stricter_report = agent.run_batch("customers")
    email_result = next(r for r in stricter_report.results if r.column == "email")
    assert email_result.status != CheckStatus.PASS


def test_run_batch_with_referential_rule_flags_orphan_order(tmp_sqlite_db, workdir):
    agent = _build_agent(tmp_sqlite_db, workdir)
    from dq_agent.models import Rule

    rule = Rule(
        table_fq_name="demo.orders",
        column="customer_id",
        rule_type=RuleType.REFERENTIAL,
        threshold=0.0,
        severity=Severity.CRITICAL,
        status=RuleStatus.ACTIVE,
        params={"ref_table": "customers", "ref_column": "customer_id"},
    )
    agent.rule_repo.add_rule(rule)
    agent.rule_repo.activate_rule(rule.rule_id)

    report = agent.run_batch("orders")
    assert len(report.results) == 1
    assert report.results[0].status == CheckStatus.FAIL
    assert 999 in report.results[0].sample_violations
    # CRITICAL + FAIL -> gate should block and quarantine
    assert report.gate_result.decision == GateDecision.BLOCK
    assert report.gate_result.quarantined_rows == 4
    assert len(report.tickets) == 1


def test_run_stream_uses_same_orchestration_as_batch(tmp_sqlite_db, workdir):
    agent = _build_agent(tmp_sqlite_db, workdir)
    from dq_agent.models import Rule

    rule = Rule(
        table_fq_name="demo.customers",
        column="email",
        rule_type=RuleType.COMPLETENESS,
        threshold=0.05,
        severity=Severity.WARN,
        status=RuleStatus.ACTIVE,
    )
    agent.rule_repo.add_rule(rule)
    agent.rule_repo.activate_rule(rule.rule_id)

    df = agent.connector.fetch_batch("customers")
    stream = InMemoryStreamSource("demo", "customers_topic", df, batch_size=2)
    reports = agent.run_stream("customers", stream)

    assert len(reports) == 3  # 5 rows / batch_size 2 -> 2, 2, 1
    assert sum(r.row_count for r in reports) == 5
    for r in reports:
        assert len(r.results) == 1
        assert r.results[0].rule_id == rule.rule_id


def test_ticket_deduplication_across_runs(tmp_sqlite_db, workdir):
    agent = _build_agent(tmp_sqlite_db, workdir)
    from dq_agent.models import Rule

    rule = Rule(
        table_fq_name="demo.orders",
        column="customer_id",
        rule_type=RuleType.REFERENTIAL,
        threshold=0.0,
        severity=Severity.CRITICAL,
        status=RuleStatus.ACTIVE,
        params={"ref_table": "customers", "ref_column": "customer_id"},
    )
    agent.rule_repo.add_rule(rule)
    agent.rule_repo.activate_rule(rule.rule_id)

    report1 = agent.run_batch("orders")
    report2 = agent.run_batch("orders")
    assert report1.tickets[0].ticket_id == report2.tickets[0].ticket_id
