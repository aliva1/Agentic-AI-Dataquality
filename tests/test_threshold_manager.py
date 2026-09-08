from dq_agent.adaptive.threshold_manager import MIN_OBSERVATIONS, ThresholdManager
from dq_agent.knowledge.rules_repo import RuleRepository
from dq_agent.models import CheckResult, CheckStatus, Rule, RuleStatus, RuleType, Severity


def _make_rule(repo: RuleRepository, threshold=0.05) -> Rule:
    rule = Rule(
        table_fq_name="t.customers",
        column="email",
        rule_type=RuleType.COMPLETENESS,
        threshold=threshold,
        severity=Severity.WARN,
        status=RuleStatus.ACTIVE,
        adaptive=True,
    )
    repo.add_rule(rule)
    return rule


def _result(rule: Rule, value: float, status=CheckStatus.PASS) -> CheckResult:
    return CheckResult(
        rule_id=rule.rule_id,
        table_fq_name=rule.table_fq_name,
        column=rule.column,
        rule_type=rule.rule_type,
        status=status,
        metric_value=value,
        threshold=rule.threshold,
        severity=rule.severity,
    )


def test_no_adaptation_before_min_observations(workdir):
    repo = RuleRepository(f"{workdir}/rules.db")
    rule = _make_rule(repo)
    tm = ThresholdManager()
    for _ in range(MIN_OBSERVATIONS - 1):
        tm.record(_result(rule, 0.20))
    assert tm.maybe_adapt(rule, repo) is None


def test_small_drift_auto_adjusts_and_stays_active(workdir):
    repo = RuleRepository(f"{workdir}/rules.db")
    rule = _make_rule(repo, threshold=0.05)
    tm = ThresholdManager()
    # steady null rate slightly above the original threshold -> small drift
    for _ in range(10):
        tm.record(_result(rule, 0.055))
    updated = tm.maybe_adapt(rule, repo)
    assert updated is not None
    assert updated.status == RuleStatus.ACTIVE
    history = repo.get_threshold_history(rule.rule_id)
    assert history[-1]["required_reapproval"] is False


def test_large_drift_requires_reapproval(workdir):
    repo = RuleRepository(f"{workdir}/rules.db")
    rule = _make_rule(repo, threshold=0.02)
    tm = ThresholdManager()
    # null rate has drifted dramatically higher -> large relative change
    for _ in range(10):
        tm.record(_result(rule, 0.40))
    updated = tm.maybe_adapt(rule, repo)
    assert updated is not None
    assert updated.status == RuleStatus.APPROVED  # paused pending human re-approval
    history = repo.get_threshold_history(rule.rule_id)
    assert history[-1]["required_reapproval"] is True


def test_failed_observations_do_not_influence_baseline(workdir):
    repo = RuleRepository(f"{workdir}/rules.db")
    rule = _make_rule(repo, threshold=0.05)
    tm = ThresholdManager()
    for _ in range(MIN_OBSERVATIONS + 2):
        tm.record(_result(rule, 0.9, status=CheckStatus.FAIL))
    # all observations were FAIL and must be ignored -> nothing to adapt from
    assert tm.observation_count(rule.rule_id) == 0
    assert tm.maybe_adapt(rule, repo) is None


def test_non_adaptive_rule_never_changes(workdir):
    repo = RuleRepository(f"{workdir}/rules.db")
    rule = Rule(
        table_fq_name="t.customers",
        column="email",
        rule_type=RuleType.COMPLETENESS,
        threshold=0.05,
        severity=Severity.WARN,
        status=RuleStatus.ACTIVE,
        adaptive=False,
    )
    repo.add_rule(rule)
    tm = ThresholdManager()
    for _ in range(10):
        tm.record(_result(rule, 0.5))
    assert tm.maybe_adapt(rule, repo) is None
