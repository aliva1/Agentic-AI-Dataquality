"""Adaptive threshold tuning.

The user's ask, verbatim: thresholds start out approved by a human, but
"as the DQ checks continue it should automatically adjust threshold."
The guardrail is just as important as the adaptation: a small drift
auto-tunes quietly (the rule stays ACTIVE); a large jump still gets
written to the rule (so its new value and the reason are visible
immediately) but demotes the rule from ACTIVE back to APPROVED, i.e. it
stops enforcing until a human re-activates it. That way "the agent
adjusts thresholds on its own" never means "the agent silently makes a
rule much easier or much harder to fail without anyone noticing."

The tuning target is only ever computed from PASS/WARN observations —
FAIL/ERROR results are exactly what the rule exists to catch, so they
must never pull a threshold toward masking themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dq_agent.knowledge.rules_repo import RuleRepository
from dq_agent.models import CheckResult, CheckStatus, Rule, RuleType

MIN_OBSERVATIONS = 5
EWMA_ALPHA = 0.3
K_STDDEV = 2.0
AUTO_ADJUST_MAX_RELATIVE_CHANGE = 0.20  # beyond this, require re-approval
MAX_HISTORY = 200

# Which rule types treat a *higher* metric as better (uniqueness, pattern
# match rate) vs. a *lower* metric as better (everything else). Mirrors
# `checks/engine.py`'s per-rule-type semantics without re-deriving it.
_HIGHER_IS_BETTER = {
    RuleType.UNIQUENESS: True,
    RuleType.PATTERN: True,
    RuleType.COMPLETENESS: False,
    RuleType.RANGE: False,
    RuleType.FRESHNESS: False,
    RuleType.REFERENTIAL: False,
    RuleType.CUSTOM_SQL: False,
}


@dataclass
class _RuleStats:
    observations: list[float] = field(default_factory=list)
    ewma_mean: float | None = None
    ewma_var: float = 0.0


class ThresholdManager:
    def __init__(self):
        self._stats: dict[str, _RuleStats] = {}

    def record(self, result: CheckResult) -> None:
        """Feed one check result into the rolling statistics for its rule."""
        if result.status in (CheckStatus.ERROR,) or result.metric_value is None:
            return
        if result.status == CheckStatus.FAIL:
            return  # don't let failures pull the "normal" baseline toward themselves
        stats = self._stats.setdefault(result.rule_id, _RuleStats())
        stats.observations.append(result.metric_value)
        if len(stats.observations) > MAX_HISTORY:
            stats.observations.pop(0)
        if stats.ewma_mean is None:
            stats.ewma_mean = result.metric_value
        else:
            delta = result.metric_value - stats.ewma_mean
            stats.ewma_mean += EWMA_ALPHA * delta
            stats.ewma_var = (1 - EWMA_ALPHA) * (stats.ewma_var + EWMA_ALPHA * delta**2)

    def observation_count(self, rule_id: str) -> int:
        stats = self._stats.get(rule_id)
        return len(stats.observations) if stats else 0

    def maybe_adapt(self, rule: Rule, rule_repo: RuleRepository) -> Rule | None:
        """Possibly adjust `rule`'s threshold based on accumulated history.

        Returns the updated Rule if an adjustment was made, else None.
        Only rules with `adaptive=True` are considered.
        """
        if not rule.adaptive:
            return None
        stats = self._stats.get(rule.rule_id)
        if stats is None or len(stats.observations) < MIN_OBSERVATIONS:
            return None

        higher_is_better = _HIGHER_IS_BETTER.get(rule.rule_type, False)
        std = stats.ewma_var**0.5
        mean = stats.ewma_mean
        assert mean is not None

        if higher_is_better:
            suggested = max(0.0, mean - K_STDDEV * std)
        else:
            suggested = mean + K_STDDEV * std
            if rule.rule_type in (
                RuleType.COMPLETENESS,
                RuleType.RANGE,
                RuleType.REFERENTIAL,
            ):
                suggested = min(1.0, suggested)

        suggested = round(suggested, 6)
        current = rule.threshold
        denom = max(abs(current), 1e-9)
        relative_change = abs(suggested - current) / denom

        # No meaningful drift: nothing to do.
        if relative_change < 0.01:
            return None

        auto_ok = relative_change <= AUTO_ADJUST_MAX_RELATIVE_CHANGE
        reason = (
            f"EWMA-based auto-tune from {len(stats.observations)} observations "
            f"(mean={mean:.4f}, std={std:.4f}); relative_change={relative_change:.1%}"
        )
        updated = rule_repo.update_threshold(
            rule.rule_id,
            new_threshold=suggested,
            reason=reason,
            auto_adjusted=True,
            required_reapproval=not auto_ok,
        )
        return updated
