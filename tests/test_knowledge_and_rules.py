import pytest

from dq_agent.knowledge.rules_repo import RuleRepository
from dq_agent.knowledge.store import KnowledgeRepository
from dq_agent.models import KnowledgeDoc, Rule, RuleStatus, RuleType, Severity


def test_knowledge_repo_add_and_get(workdir):
    repo = KnowledgeRepository(f"{workdir}/knowledge.db")
    doc = KnowledgeDoc(
        table_fq_name="cam_ms.fact_opportunity_hist",
        title="Grain",
        content="Grain is opportunity_id + snapshot_dt_key; append-only, use MAX(snapshot_dt_key).",
    )
    repo.add_doc(doc)
    docs = repo.get_docs_for_table("cam_ms.fact_opportunity_hist")
    assert len(docs) == 1
    assert docs[0].title == "Grain"


def test_knowledge_repo_semantic_search_finds_relevant_doc(workdir):
    repo = KnowledgeRepository(f"{workdir}/knowledge.db")
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


def test_rule_repository_lifecycle(workdir):
    repo = RuleRepository(f"{workdir}/rules.db")
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


def test_rule_repository_threshold_history_and_reapproval_gate(workdir):
    repo = RuleRepository(f"{workdir}/rules.db")
    rule = Rule(
        table_fq_name="t.orders",
        column="amount",
        rule_type=RuleType.COMPLETENESS,
        threshold=0.05,
        severity=Severity.WARN,
        status=RuleStatus.ACTIVE,
    )
    repo.add_rule(rule)

    # small, auto-adjustable change: stays ACTIVE
    repo.update_threshold(rule.rule_id, 0.06, reason="drift", auto_adjusted=True, required_reapproval=False)
    updated = repo.get_rule(rule.rule_id)
    assert updated.threshold == 0.06
    assert updated.status == RuleStatus.ACTIVE

    # large change: demoted back to APPROVED pending human sign-off
    repo.update_threshold(rule.rule_id, 0.5, reason="big jump", auto_adjusted=True, required_reapproval=True)
    updated2 = repo.get_rule(rule.rule_id)
    assert updated2.threshold == 0.5
    assert updated2.status == RuleStatus.APPROVED

    history = repo.get_threshold_history(rule.rule_id)
    assert len(history) == 2
    assert history[-1]["required_reapproval"] is True


def test_get_rule_missing_raises_keyerror(workdir):
    repo = RuleRepository(f"{workdir}/rules.db")
    with pytest.raises(KeyError):
        repo.approve_rule("does_not_exist")
