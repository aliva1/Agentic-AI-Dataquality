"""Tests for the JSON-file-backed repositories/sink -- the same
behavioral contract as the SQLite versions (see test_knowledge_and_rules.py
and test_gate_and_ticketing.py), proven against a plain JSON file instead
of a database.
"""

import pytest

from dq_agent.knowledge.rules_repo_json import JsonRuleRepository
from dq_agent.knowledge.store_json import JsonKnowledgeRepository
from dq_agent.models import CheckResult, CheckStatus, KnowledgeDoc, Rule, RuleStatus, RuleType, Severity
from dq_agent.ticketing.json_sink import JsonTicketSink


# -- JsonKnowledgeRepository -------------------------------------------------


def test_json_knowledge_repo_add_and_get(workdir):
    repo = JsonKnowledgeRepository(f"{workdir}/knowledge.json")
    doc = KnowledgeDoc(
        table_fq_name="csv_export.Customer",
        title="Grain",
        content="One row per customer; CustomerId is the primary key.",
    )
    repo.add_doc(doc)
    docs = repo.get_docs_for_table("csv_export.Customer")
    assert len(docs) == 1
    assert docs[0].title == "Grain"


def test_json_knowledge_repo_semantic_search_finds_relevant_doc(workdir):
    repo = JsonKnowledgeRepository(f"{workdir}/knowledge.json")
    repo.add_doc(
        KnowledgeDoc(
            table_fq_name="t.orders",
            title="Amount rules",
            content="Order amount must never be negative; refunds are separate rows.",
        )
    )
    repo.add_doc(
        KnowledgeDoc(
            table_fq_name="t.orders",
            title="Timezone",
            content="All timestamps are stored in Beijing time (BJT), not UTC.",
        )
    )
    hits = repo.search("why would order amount be negative", k=1)
    assert len(hits) == 1
    assert hits[0].title == "Amount rules"


def test_json_knowledge_repo_survives_reload(workdir):
    path = f"{workdir}/knowledge.json"
    repo = JsonKnowledgeRepository(path)
    repo.add_doc(KnowledgeDoc(table_fq_name="t.x", title="A", content="B"))

    reloaded = JsonKnowledgeRepository(path)
    docs = reloaded.get_docs_for_table("t.x")
    assert len(docs) == 1
    assert docs[0].title == "A"


def test_json_knowledge_repo_delete_doc(workdir):
    repo = JsonKnowledgeRepository(f"{workdir}/knowledge.json")
    doc = KnowledgeDoc(table_fq_name="t.x", title="A", content="B")
    repo.add_doc(doc)
    repo.delete_doc(doc.doc_id)
    assert repo.get_docs_for_table("t.x") == []


# -- JsonRuleRepository -------------------------------------------------


def test_json_rule_repository_lifecycle(workdir):
    repo = JsonRuleRepository(f"{workdir}/rules.json")
    rule = Rule(
        table_fq_name="t.orders",
        column="amount",
        rule_type=RuleType.RANGE,
        threshold=0.0,
        severity=Severity.WARN,
    )
    repo.add_rule(rule)
    assert repo.get_rule(rule.rule_id).status == RuleStatus.PROPOSED

    repo.approve_rule(rule.rule_id)
    assert repo.get_rule(rule.rule_id).status == RuleStatus.APPROVED

    repo.activate_rule(rule.rule_id)
    active = repo.list_rules(table_fq_name="t.orders", status=RuleStatus.ACTIVE)
    assert len(active) == 1
    assert active[0].rule_id == rule.rule_id


def test_json_rule_repository_threshold_history_and_reapproval_gate(workdir):
    repo = JsonRuleRepository(f"{workdir}/rules.json")
    rule = Rule(
        table_fq_name="t.orders",
        column="amount",
        rule_type=RuleType.COMPLETENESS,
        threshold=0.05,
        severity=Severity.WARN,
        status=RuleStatus.ACTIVE,
    )
    repo.add_rule(rule)

    repo.update_threshold(rule.rule_id, 0.06, reason="drift", auto_adjusted=True, required_reapproval=False)
    updated = repo.get_rule(rule.rule_id)
    assert updated.threshold == 0.06
    assert updated.status == RuleStatus.ACTIVE

    repo.update_threshold(rule.rule_id, 0.5, reason="big jump", auto_adjusted=True, required_reapproval=True)
    updated2 = repo.get_rule(rule.rule_id)
    assert updated2.threshold == 0.5
    assert updated2.status == RuleStatus.APPROVED

    history = repo.get_threshold_history(rule.rule_id)
    assert len(history) == 2
    assert history[-1]["required_reapproval"] is True


def test_json_rule_repository_survives_reload(workdir):
    path = f"{workdir}/rules.json"
    repo = JsonRuleRepository(path)
    rule = Rule(table_fq_name="t.x", column="c", rule_type=RuleType.COMPLETENESS, threshold=0.1)
    repo.add_rule(rule)

    reloaded = JsonRuleRepository(path)
    fetched = reloaded.get_rule(rule.rule_id)
    assert fetched is not None
    assert fetched.threshold == 0.1
    assert fetched.rule_type == RuleType.COMPLETENESS


def test_json_rule_repository_get_rule_missing_raises_keyerror(workdir):
    repo = JsonRuleRepository(f"{workdir}/rules.json")
    with pytest.raises(KeyError):
        repo.approve_rule("does_not_exist")


# -- JsonTicketSink -------------------------------------------------


def _fail_result(rule_id: str) -> CheckResult:
    return CheckResult(
        rule_id=rule_id,
        table_fq_name="t.orders",
        column="amount",
        rule_type=RuleType.RANGE,
        status=CheckStatus.FAIL,
        metric_value=0.2,
        threshold=0.0,
        severity=Severity.CRITICAL,
    )


def test_json_ticket_sink_creates_and_lists(workdir):
    sink = JsonTicketSink(f"{workdir}/tickets.json")
    ticket = sink.create_ticket(
        "t.orders", "Bad amounts", "desc", Severity.CRITICAL, [_fail_result("r1")]
    )
    assert sink.list_tickets("t.orders") == [ticket]
    assert sink.list_tickets("t.other") == []
    assert sink.list_tickets() == [ticket]


def test_json_ticket_sink_deduplicates_open_tickets(workdir):
    sink = JsonTicketSink(f"{workdir}/tickets.json")
    first = sink.create_ticket("t.orders", "Bad amounts", "desc", Severity.CRITICAL, [_fail_result("r1")])
    second = sink.create_ticket("t.orders", "Bad amounts again", "desc2", Severity.CRITICAL, [_fail_result("r1")])
    assert first.ticket_id == second.ticket_id
    assert len(sink.list_tickets("t.orders")) == 1


def test_json_ticket_sink_resolve_then_new_ticket_for_same_rules(workdir):
    sink = JsonTicketSink(f"{workdir}/tickets.json")
    first = sink.create_ticket("t.orders", "Bad amounts", "desc", Severity.CRITICAL, [_fail_result("r1")])
    sink.resolve_ticket(first.ticket_id)
    second = sink.create_ticket("t.orders", "Bad amounts", "desc", Severity.CRITICAL, [_fail_result("r1")])
    assert second.ticket_id != first.ticket_id
    assert len(sink.list_tickets("t.orders")) == 2


def test_json_ticket_sink_survives_reload(workdir):
    path = f"{workdir}/tickets.json"
    sink = JsonTicketSink(path)
    ticket = sink.create_ticket("t.orders", "Bad amounts", "desc", Severity.WARN, [_fail_result("r1")])

    reloaded = JsonTicketSink(path)
    assert reloaded.list_tickets("t.orders") == [ticket]
