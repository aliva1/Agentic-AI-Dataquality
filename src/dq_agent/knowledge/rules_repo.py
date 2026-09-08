"""The DQ rule repository: lifecycle, persistence, and threshold history.

Every rule the agent runs lives here. The lifecycle is deliberately
explicit because the user asked for a human-in-the-loop on new rules
and on their thresholds:

    PROPOSED --approve--> APPROVED --activate--> ACTIVE
       |--reject--> REJECTED
    ACTIVE --retire--> RETIRED

Threshold changes (manual or auto-adjusted by
`dq_agent.adaptive.threshold_manager`) are appended to
`rule_threshold_history` rather than silently overwritten, so every
adjustment is auditable — which matters a lot once thresholds start
moving on their own.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

from dq_agent.models import Rule, RuleStatus, RuleType, Severity

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rules (
    rule_id TEXT PRIMARY KEY,
    table_fq_name TEXT NOT NULL,
    column_name TEXT,
    rule_type TEXT NOT NULL,
    threshold REAL NOT NULL,
    severity TEXT NOT NULL,
    status TEXT NOT NULL,
    params TEXT,
    rationale TEXT,
    adaptive INTEGER NOT NULL,
    created_by TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    version INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rules_table ON rules(table_fq_name);
CREATE INDEX IF NOT EXISTS idx_rules_status ON rules(status);

CREATE TABLE IF NOT EXISTS rule_threshold_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id TEXT NOT NULL,
    old_threshold REAL NOT NULL,
    new_threshold REAL NOT NULL,
    changed_at REAL NOT NULL,
    reason TEXT,
    auto_adjusted INTEGER NOT NULL,
    required_reapproval INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_rule ON rule_threshold_history(rule_id);
"""


class RuleRepository:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- create / read -------------------------------------------------
    def add_rule(self, rule: Rule) -> Rule:
        self._conn.execute(
            "INSERT INTO rules VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rule.rule_id,
                rule.table_fq_name,
                rule.column,
                rule.rule_type.value,
                rule.threshold,
                rule.severity.value,
                rule.status.value,
                json.dumps(rule.params),
                rule.rationale,
                int(rule.adaptive),
                rule.created_by,
                rule.created_at,
                rule.updated_at,
                rule.version,
            ),
        )
        self._conn.commit()
        return rule

    def get_rule(self, rule_id: str) -> Optional[Rule]:
        row = self._conn.execute("SELECT * FROM rules WHERE rule_id = ?", (rule_id,)).fetchone()
        return self._row_to_rule(row) if row else None

    def list_rules(
        self, table_fq_name: Optional[str] = None, status: Optional[RuleStatus] = None
    ) -> list[Rule]:
        query = "SELECT * FROM rules WHERE 1=1"
        params: list = []
        if table_fq_name:
            query += " AND table_fq_name = ?"
            params.append(table_fq_name)
        if status:
            query += " AND status = ?"
            params.append(status.value)
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_rule(r) for r in rows]

    # -- lifecycle -------------------------------------------------------
    def approve_rule(self, rule_id: str) -> Rule:
        return self._set_status(rule_id, RuleStatus.APPROVED)

    def activate_rule(self, rule_id: str) -> Rule:
        return self._set_status(rule_id, RuleStatus.ACTIVE)

    def reject_rule(self, rule_id: str) -> Rule:
        return self._set_status(rule_id, RuleStatus.REJECTED)

    def retire_rule(self, rule_id: str) -> Rule:
        return self._set_status(rule_id, RuleStatus.RETIRED)

    def _set_status(self, rule_id: str, status: RuleStatus) -> Rule:
        now = time.time()
        self._conn.execute(
            "UPDATE rules SET status = ?, updated_at = ? WHERE rule_id = ?",
            (status.value, now, rule_id),
        )
        self._conn.commit()
        rule = self.get_rule(rule_id)
        if rule is None:
            raise KeyError(f"No such rule: {rule_id}")
        return rule

    # -- threshold adjustment -------------------------------------------
    def update_threshold(
        self,
        rule_id: str,
        new_threshold: float,
        reason: str,
        auto_adjusted: bool = False,
        required_reapproval: bool = False,
    ) -> Rule:
        rule = self.get_rule(rule_id)
        if rule is None:
            raise KeyError(f"No such rule: {rule_id}")
        now = time.time()
        self._conn.execute(
            "INSERT INTO rule_threshold_history "
            "(rule_id, old_threshold, new_threshold, changed_at, reason, auto_adjusted, required_reapproval) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rule_id, rule.threshold, new_threshold, now, reason, int(auto_adjusted), int(required_reapproval)),
        )
        new_status = RuleStatus.APPROVED.value if required_reapproval else rule.status.value
        self._conn.execute(
            "UPDATE rules SET threshold = ?, status = ?, updated_at = ?, version = version + 1 "
            "WHERE rule_id = ?",
            (new_threshold, new_status, now, rule_id),
        )
        self._conn.commit()
        updated = self.get_rule(rule_id)
        assert updated is not None
        return updated

    def get_threshold_history(self, rule_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT old_threshold, new_threshold, changed_at, reason, auto_adjusted, required_reapproval "
            "FROM rule_threshold_history WHERE rule_id = ? ORDER BY changed_at ASC",
            (rule_id,),
        ).fetchall()
        return [
            {
                "old_threshold": r[0],
                "new_threshold": r[1],
                "changed_at": r[2],
                "reason": r[3],
                "auto_adjusted": bool(r[4]),
                "required_reapproval": bool(r[5]),
            }
            for r in rows
        ]

    @staticmethod
    def _row_to_rule(row) -> Rule:
        (
            rule_id,
            table_fq_name,
            column_name,
            rule_type,
            threshold,
            severity,
            status,
            params,
            rationale,
            adaptive,
            created_by,
            created_at,
            updated_at,
            version,
        ) = row
        return Rule(
            rule_id=rule_id,
            table_fq_name=table_fq_name,
            column=column_name,
            rule_type=RuleType(rule_type),
            threshold=threshold,
            severity=Severity(severity),
            status=RuleStatus(status),
            params=json.loads(params) if params else {},
            rationale=rationale or "",
            adaptive=bool(adaptive),
            created_by=created_by or "system",
            created_at=created_at,
            updated_at=updated_at,
            version=version,
        )

    def close(self) -> None:
        self._conn.close()
