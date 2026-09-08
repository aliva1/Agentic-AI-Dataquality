import pandas as pd

from dq_agent.gating.gate import DataGate
from dq_agent.models import CheckResult, CheckStatus, GateDecision, RuleType, Severity
from dq_agent.ticketing.local_sink import LocalTicketSink


def _result(status, severity, column="email") -> CheckResult:
    return CheckResult(
        rule_id="rule_1",
        table_fq_name="t.customers",
        column=column,
        rule_type=RuleType.COMPLETENESS,
        status=status,
        metric_value=0.5,
        threshold=0.1,
        severity=severity,
    )


def test_gate_passes_clean_batch(tmp_path):
    gate = DataGate(str(tmp_path / "quarantine"))
    df = pd.DataFrame({"email": ["a", "b"]})
    result = gate.evaluate("t.customers", "batch_1", df, [_result(CheckStatus.PASS, Severity.WARN)])
    assert result.decision == GateDecision.PASS
    assert result.quarantined_rows == 0


def test_gate_warns_on_non_critical_failure(tmp_path):
    gate = DataGate(str(tmp_path / "quarantine"))
    df = pd.DataFrame({"email": ["a", "b"]})
    result = gate.evaluate("t.customers", "batch_1", df, [_result(CheckStatus.FAIL, Severity.WARN)])
    assert result.decision == GateDecision.PASS_WITH_WARNINGS
    assert result.quarantined_rows == 0


def test_gate_blocks_and_quarantines_on_critical_failure(tmp_path):
    quarantine_dir = tmp_path / "quarantine"
    gate = DataGate(str(quarantine_dir))
    df = pd.DataFrame({"email": ["a", "b", "c"]})
    result = gate.evaluate("t.customers", "batch_42", df, [_result(CheckStatus.FAIL, Severity.CRITICAL)])
    assert result.decision == GateDecision.BLOCK
    assert result.quarantined_rows == 3
    quarantined_files = list(quarantine_dir.rglob("batch_42.csv"))
    assert len(quarantined_files) == 1
    assert len(pd.read_csv(quarantined_files[0])) == 3


def test_ticket_sink_dedupes_open_tickets_for_same_rules(tmp_path):
    sink = LocalTicketSink(str(tmp_path / "tickets.db"))
    results = [_result(CheckStatus.FAIL, Severity.CRITICAL)]
    t1 = sink.create_ticket("t.customers", "title", "desc", Severity.CRITICAL, results)
    t2 = sink.create_ticket("t.customers", "title again", "desc again", Severity.CRITICAL, results)
    assert t1.ticket_id == t2.ticket_id
    assert len(sink.list_tickets("t.customers")) == 1


def test_ticket_sink_resolve_allows_new_ticket_for_same_rules(tmp_path):
    sink = LocalTicketSink(str(tmp_path / "tickets.db"))
    results = [_result(CheckStatus.FAIL, Severity.CRITICAL)]
    t1 = sink.create_ticket("t.customers", "title", "desc", Severity.CRITICAL, results)
    sink.resolve_ticket(t1.ticket_id)
    t2 = sink.create_ticket("t.customers", "title", "desc", Severity.CRITICAL, results)
    assert t1.ticket_id != t2.ticket_id
    assert len(sink.list_tickets("t.customers")) == 2
