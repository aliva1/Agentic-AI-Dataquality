"""The pass/hold gate: the "if the data is real bad, don't let it
through" circuit breaker.

One `CRITICAL`-severity rule in `FAIL` status is enough to `BLOCK` a
batch outright — it gets written to a quarantine store instead of being
considered ready for the next layer. Anything short of that but still
failing/warning downgrades to `PASS_WITH_WARNINGS` (downstream consumers
can decide for themselves whether to proceed); a clean batch is `PASS`.

This runs identically whether the batch came from a full table scan or
one streaming micro-batch — it only ever looks at `CheckResult`s.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dq_agent.models import CheckResult, CheckStatus, GateDecision, GateResult, Severity


class DataGate:
    def __init__(self, quarantine_dir: str = "./data/quarantine"):
        self.quarantine_dir = Path(quarantine_dir)

    def evaluate(
        self,
        table_fq_name: str,
        batch_id: str,
        df: pd.DataFrame,
        results: list[CheckResult],
    ) -> GateResult:
        critical_failures = [
            r for r in results if r.severity == Severity.CRITICAL and r.status == CheckStatus.FAIL
        ]
        other_issues = [
            r
            for r in results
            if r.status in (CheckStatus.FAIL, CheckStatus.WARN) and r not in critical_failures
        ]

        if critical_failures:
            decision = GateDecision.BLOCK
            reasons = [f"CRITICAL FAIL: {r.column} {r.rule_type.value} — {r.message}" for r in critical_failures]
            quarantined = self._quarantine(table_fq_name, batch_id, df)
            return GateResult(
                table_fq_name=table_fq_name,
                batch_id=batch_id,
                decision=decision,
                reasons=reasons,
                quarantined_rows=quarantined,
            )

        if other_issues:
            reasons = [f"{r.status.value.upper()}: {r.column} {r.rule_type.value} — {r.message}" for r in other_issues]
            return GateResult(
                table_fq_name=table_fq_name,
                batch_id=batch_id,
                decision=GateDecision.PASS_WITH_WARNINGS,
                reasons=reasons,
            )

        return GateResult(table_fq_name=table_fq_name, batch_id=batch_id, decision=GateDecision.PASS)

    def _quarantine(self, table_fq_name: str, batch_id: str, df: pd.DataFrame) -> int:
        target_dir = self.quarantine_dir / table_fq_name.replace(".", "_")
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{batch_id}.csv"
        df.to_csv(target_path, index=False)
        return len(df)
