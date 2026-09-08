"""The deterministic check-execution engine.

This is intentionally the most boring, most tested module in the
codebase: given a `Rule` and a chunk of data (a full table or a
micro-batch — the engine doesn't care which), it computes one metric and
turns it into a `CheckResult`. No LLM calls, no I/O beyond the DataFrame
it's handed. That's what lets `orchestrator/pipeline.py` run the exact
same `evaluate()` call for batch and streaming.

Each rule type maps a raw value to a "badness" metric that should be
<= threshold to pass, except UNIQUENESS and PATTERN where the metric is
a "goodness" rate that should be >= threshold — noted per-branch below.
A configurable warn buffer gives a soft zone between clean pass and
outright failure, so a metric drifting toward its threshold shows up as
WARN before it ever hits FAIL.
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd

from dq_agent.models import CheckResult, CheckStatus, Rule, RuleType

WARN_BUFFER_RATIO = 0.15  # 15% headroom before a passing metric is flagged WARN


def evaluate(
    rule: Rule,
    df: pd.DataFrame,
    reference_values: Optional[set] = None,
    batch_id: Optional[str] = None,
) -> CheckResult:
    try:
        metric, higher_is_better, message, violations = _METRIC_FUNCS[rule.rule_type](
            rule, df, reference_values
        )
    except KeyError:
        return _result(rule, CheckStatus.ERROR, None, f"Unknown rule_type {rule.rule_type}", [], batch_id)
    except Exception as exc:  # column missing, bad expression, etc. — never crash the pipeline
        return _result(rule, CheckStatus.ERROR, None, f"Check errored: {exc}", [], batch_id)

    status = _status_for(metric, rule.threshold, higher_is_better)
    return _result(rule, status, metric, message, violations, batch_id)


def _status_for(metric: float, threshold: float, higher_is_better: bool) -> CheckStatus:
    buffer = max(threshold, 0.01) * WARN_BUFFER_RATIO
    if higher_is_better:
        if metric >= threshold:
            return CheckStatus.PASS
        if metric >= threshold - buffer:
            return CheckStatus.WARN
        return CheckStatus.FAIL
    else:
        if metric <= threshold:
            return CheckStatus.PASS
        if metric <= threshold + buffer:
            return CheckStatus.WARN
        return CheckStatus.FAIL


def _result(rule, status, metric, message, violations, batch_id) -> CheckResult:
    kwargs = dict(
        rule_id=rule.rule_id,
        table_fq_name=rule.table_fq_name,
        column=rule.column,
        rule_type=rule.rule_type,
        status=status,
        metric_value=metric,
        threshold=rule.threshold,
        severity=rule.severity,
        message=message,
        sample_violations=violations[:10],
    )
    if batch_id:
        kwargs["batch_id"] = batch_id
    return CheckResult(**kwargs)


# =======================================================================
# Per-rule-type metric functions
#
# Each returns (metric_value, higher_is_better, message, sample_violations)
# =======================================================================


def _completeness(rule: Rule, df: pd.DataFrame, _ref) -> tuple[float, bool, str, list]:
    series = df[rule.column]
    null_rate = float(series.isna().mean()) if len(series) else 0.0
    return null_rate, False, f"Null rate {null_rate:.2%} (max allowed {rule.threshold:.2%})", []


def _uniqueness(rule: Rule, df: pd.DataFrame, _ref) -> tuple[float, bool, str, list]:
    series = df[rule.column].dropna()
    distinct_rate = float(series.nunique() / len(series)) if len(series) else 1.0
    dupes = series[series.duplicated(keep=False)].unique().tolist()
    return distinct_rate, True, f"Distinct rate {distinct_rate:.2%} (min required {rule.threshold:.2%})", dupes


def _range(rule: Rule, df: pd.DataFrame, _ref) -> tuple[float, bool, str, list]:
    series = pd.to_numeric(df[rule.column], errors="coerce").dropna()
    lo = rule.params.get("min")
    hi = rule.params.get("max")
    if not len(series):
        return 0.0, False, "No non-null numeric values to check", []
    mask = pd.Series(True, index=series.index)
    if lo is not None:
        mask &= series >= lo
    if hi is not None:
        mask &= series <= hi
    violation_rate = float((~mask).mean())
    violations = series[~mask].head(10).tolist()
    return violation_rate, False, f"{violation_rate:.2%} of values outside [{lo}, {hi}]", violations


_PATTERN_REGEX = {
    "email": r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$",
    "uuid": r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
    "date_iso": r"^\d{4}-\d{2}-\d{2}$",
    "datetime_iso": r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}",
    "integer_id": r"^\d+$",
    "phone": r"^\+?[\d\-\(\) ]{7,15}$",
    "zip_code_us": r"^\d{5}(-\d{4})?$",
}


def _pattern(rule: Rule, df: pd.DataFrame, _ref) -> tuple[float, bool, str, list]:
    series = df[rule.column].dropna().astype(str)
    if not len(series):
        return 1.0, True, "No non-null values to check", []
    pattern = rule.params.get("regex") or _PATTERN_REGEX.get(rule.params.get("pattern_name", ""), None)
    if pattern is None:
        raise ValueError(f"No regex/pattern_name configured for rule {rule.rule_id}")
    matches = series.str.match(pattern)
    match_rate = float(matches.mean())
    violations = series[~matches].head(10).tolist()
    return match_rate, True, f"{match_rate:.2%} matched pattern (min required {rule.threshold:.2%})", violations


def _freshness(rule: Rule, df: pd.DataFrame, _ref) -> tuple[float, bool, str, list]:
    series = pd.to_datetime(df[rule.column], errors="coerce").dropna()
    if not len(series):
        return float("inf"), False, "No valid timestamps found", []
    max_ts = series.max()
    now = pd.Timestamp.now(tz="UTC")
    if max_ts.tzinfo is not None:
        now = now.tz_convert(max_ts.tzinfo)
    else:
        now = now.tz_localize(None)
    age_hours = float((now - max_ts).total_seconds() / 3600.0)
    return age_hours, False, f"Most recent {rule.column} is {age_hours:.1f}h old (max allowed {rule.threshold}h)", []


def _referential(rule: Rule, df: pd.DataFrame, ref: Optional[set]) -> tuple[float, bool, str, list]:
    series = df[rule.column].dropna()
    ref_values = ref if ref is not None else set(rule.params.get("ref_values", []))
    if not len(series):
        return 0.0, False, "No non-null values to check", []
    mask = series.isin(ref_values)
    violation_rate = float((~mask).mean())
    violations = series[~mask].unique().tolist()[:10]
    return violation_rate, False, f"{violation_rate:.2%} of values not found in reference set", violations


def _custom_sql(rule: Rule, df: pd.DataFrame, _ref) -> tuple[float, bool, str, list]:
    valid_expr = rule.params.get("valid_expr")
    if not valid_expr:
        raise ValueError(f"No 'valid_expr' configured for custom rule {rule.rule_id}")
    valid_mask = df.eval(valid_expr)
    violation_rate = float((~valid_mask).mean()) if len(df) else 0.0
    sample = df.loc[~valid_mask].head(10).to_dict("records")
    return violation_rate, False, f"{violation_rate:.2%} of rows violate '{valid_expr}'", sample


_METRIC_FUNCS = {
    RuleType.COMPLETENESS: _completeness,
    RuleType.UNIQUENESS: _uniqueness,
    RuleType.RANGE: _range,
    RuleType.PATTERN: _pattern,
    RuleType.FRESHNESS: _freshness,
    RuleType.REFERENTIAL: _referential,
    RuleType.CUSTOM_SQL: _custom_sql,
}


def evaluate_all(
    rules: list[Rule],
    df: pd.DataFrame,
    reference_values_by_rule: Optional[dict[str, set]] = None,
    batch_id: Optional[str] = None,
) -> list[CheckResult]:
    """Convenience wrapper: run every rule against the same DataFrame."""
    reference_values_by_rule = reference_values_by_rule or {}
    batch_id = batch_id or f"batch_{int(time.time() * 1000)}"
    return [
        evaluate(rule, df, reference_values_by_rule.get(rule.rule_id), batch_id=batch_id)
        for rule in rules
    ]
