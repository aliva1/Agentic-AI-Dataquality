import pandas as pd

from dq_agent.checks.engine import evaluate, evaluate_all
from dq_agent.models import CheckStatus, Rule, RuleType, Severity


def _rule(**overrides) -> Rule:
    defaults = dict(
        table_fq_name="t.customers",
        column="email",
        rule_type=RuleType.COMPLETENESS,
        threshold=0.1,
        severity=Severity.WARN,
    )
    defaults.update(overrides)
    return Rule(**defaults)


def test_completeness_pass():
    df = pd.DataFrame({"email": ["a", "b", "c", None]})  # 25% null
    rule = _rule(threshold=0.5)
    result = evaluate(rule, df)
    assert result.status == CheckStatus.PASS
    assert result.metric_value == 0.25


def test_completeness_fail():
    df = pd.DataFrame({"email": ["a", None, None, None]})  # 75% null
    rule = _rule(threshold=0.1)
    result = evaluate(rule, df)
    assert result.status == CheckStatus.FAIL


def test_completeness_warn_zone():
    df = pd.DataFrame({"email": ["a"] * 90 + [None] * 10})  # 10% null
    rule = _rule(threshold=0.10)  # exactly at threshold and within buffer -> pass
    result = evaluate(rule, df)
    assert result.status == CheckStatus.PASS

    df2 = pd.DataFrame({"email": ["a"] * 89 + [None] * 11})  # 11% null, threshold 0.10 -> warn zone
    result2 = evaluate(rule, df2)
    assert result2.status == CheckStatus.WARN


def test_uniqueness():
    df = pd.DataFrame({"id": [1, 2, 3, 3]})
    rule = _rule(column="id", rule_type=RuleType.UNIQUENESS, threshold=0.999)
    result = evaluate(rule, df)
    assert result.status == CheckStatus.FAIL
    assert result.metric_value == 0.75


def test_range_violation():
    df = pd.DataFrame({"amount": [10, 20, -5, 15]})
    rule = _rule(
        column="amount",
        rule_type=RuleType.RANGE,
        threshold=0.0,
        params={"min": 0, "max": 1000},
    )
    result = evaluate(rule, df)
    assert result.status == CheckStatus.FAIL
    assert result.metric_value == 0.25
    assert -5 in result.sample_violations


def test_pattern_match():
    df = pd.DataFrame({"email": ["a@x.com", "b@x.com", "bad-email"]})
    rule = _rule(
        column="email",
        rule_type=RuleType.PATTERN,
        threshold=0.9,
        params={"pattern_name": "email"},
    )
    result = evaluate(rule, df)
    assert result.status == CheckStatus.FAIL  # 66% match rate vs 90% required, outside warn buffer


def test_referential_violation():
    df = pd.DataFrame({"customer_id": [1, 2, 999]})
    rule = _rule(
        column="customer_id",
        rule_type=RuleType.REFERENTIAL,
        threshold=0.0,
    )
    result = evaluate(rule, df, reference_values={1, 2, 3})
    assert result.status == CheckStatus.FAIL
    assert 999 in result.sample_violations


def test_freshness():
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    df = pd.DataFrame({"updated_at": [now - pd.Timedelta(hours=1)]})
    rule = _rule(column="updated_at", rule_type=RuleType.FRESHNESS, threshold=24)
    result = evaluate(rule, df)
    assert result.status == CheckStatus.PASS


def test_custom_sql():
    df = pd.DataFrame({"quantity": [1, 2, -1], "price": [10, 20, 30]})
    rule = _rule(
        column=None,
        rule_type=RuleType.CUSTOM_SQL,
        threshold=0.0,
        params={"valid_expr": "quantity >= 0 and price > 0"},
    )
    result = evaluate(rule, df)
    assert result.status == CheckStatus.FAIL
    assert abs(result.metric_value - (1 / 3)) < 1e-6


def test_missing_column_produces_error_not_crash():
    df = pd.DataFrame({"other_col": [1, 2, 3]})
    rule = _rule(column="does_not_exist", rule_type=RuleType.COMPLETENESS, threshold=0.1)
    result = evaluate(rule, df)
    assert result.status == CheckStatus.ERROR


def test_evaluate_all_shares_batch_id():
    df = pd.DataFrame({"email": ["a", "b", None]})
    rules = [_rule(threshold=0.5), _rule(threshold=0.9)]
    results = evaluate_all(rules, df)
    assert len({r.batch_id for r in results}) == 1
