import pandas as pd

from dq_agent.llm.client import MockLLMClient
from dq_agent.llm.dq_reasoner import DQReasoner
from dq_agent.models import CheckResult, CheckStatus, KnowledgeDoc, RuleType, Severity
from dq_agent.profiling.profiler import profile_table


def test_suggest_rules_falls_back_to_statistical_baseline_with_no_llm():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "email": ["a@x.com", "b@x.com", "c@x.com", None],
        }
    )
    profile = profile_table(df, "t.customers")
    reasoner = DQReasoner(MockLLMClient())  # default mock returns "[]" for rule proposals
    rules = reasoner.suggest_rules(profile)
    rule_types = {r.rule_type for r in rules}
    assert RuleType.COMPLETENESS in rule_types
    assert RuleType.UNIQUENESS in rule_types  # id is a candidate key


def test_suggest_rules_uses_llm_refinement_when_available():
    df = pd.DataFrame({"id": [1, 2, 3, 4]})
    profile = profile_table(df, "t.customers")
    canned = {
        "propose data quality rules": (
            '[{"table": "t.customers", "column": "id", "rule_type": "uniqueness", '
            '"threshold": 0.999, "severity": "critical", "rationale": "LLM: id is the PK"}]'
        )
    }
    reasoner = DQReasoner(MockLLMClient(canned_responses=canned))
    rules = reasoner.suggest_rules(profile)
    assert len(rules) == 1
    assert rules[0].rationale == "LLM: id is the PK"


def test_explain_failure_parses_confidence_and_action():
    result = CheckResult(
        rule_id="r1",
        table_fq_name="t.orders",
        column="amount",
        rule_type=RuleType.RANGE,
        status=CheckStatus.FAIL,
        metric_value=0.3,
        threshold=0.0,
        severity=Severity.CRITICAL,
    )
    canned = {
        "root cause": (
            "The upstream ETL job likely double-counted refunds.\n"
            "Confidence: 0.82\n"
            "Action: Re-run the refunds dedup step for yesterday's load."
        )
    }
    reasoner = DQReasoner(MockLLMClient(canned_responses=canned))
    rca = reasoner.explain_failure(result)
    assert rca.confidence == 0.82
    assert "refunds dedup" in rca.suggested_action
    assert "double-counted refunds" in rca.explanation


def test_explain_failure_defaults_confidence_when_unparseable():
    result = CheckResult(
        rule_id="r1",
        table_fq_name="t.orders",
        column="amount",
        rule_type=RuleType.RANGE,
        status=CheckStatus.FAIL,
        metric_value=0.3,
        threshold=0.0,
        severity=Severity.CRITICAL,
    )
    reasoner = DQReasoner(MockLLMClient())
    rca = reasoner.explain_failure(result)
    assert 0.0 <= rca.confidence <= 1.0


def test_draft_ticket_falls_back_when_llm_output_unparseable():
    result = CheckResult(
        rule_id="r1",
        table_fq_name="t.orders",
        column="amount",
        rule_type=RuleType.RANGE,
        status=CheckStatus.FAIL,
        metric_value=0.3,
        threshold=0.0,
        severity=Severity.CRITICAL,
    )
    reasoner = DQReasoner(MockLLMClient())
    title, description = reasoner.draft_ticket([result])
    assert "t.orders" in title


def test_knowledge_docs_feed_into_prompt():
    df = pd.DataFrame({"id": [1, 2, 3]})
    profile = profile_table(df, "t.customers")
    docs = [KnowledgeDoc(table_fq_name="t.customers", title="Note", content="ids restart at fiscal year")]
    captured = {}

    class CapturingMock(MockLLMClient):
        def complete(self, prompt, system=None, max_tokens=1024):
            captured["prompt"] = prompt
            return super().complete(prompt, system, max_tokens)

    reasoner = DQReasoner(CapturingMock())
    reasoner.suggest_rules(profile, docs)
    assert "ids restart at fiscal year" in captured["prompt"]
