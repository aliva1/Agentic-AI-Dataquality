"""JSON-file-backed twin of `RuleRepository`.

Same lifecycle, same audit trail, same public API as the SQLite version
(`dq_agent.knowledge.rules_repo.RuleRepository`) -- only the storage
medium changes: one plain, human-readable `.json` file instead of a
SQLite database. This is what "everything in a file" means for the rule
repository: open it in any text editor and you can read every rule and
its full threshold-change history directly.

Not meant to replace the SQLite repository for large rule sets (every
write rewrites the whole file) -- it's the right choice when the whole
point is a fully flat-file, no-database deployment (e.g. alongside a
`DirectoryFlatFileConnector` source), same as `JsonKnowledgeRepository`
and `JsonTicketSink`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from dq_agent.models import Rule, RuleStatus, RuleType, Severity


def _rule_to_dict(rule: Rule) -> dict[str, Any]:
    return {
        "rule_id": rule.rule_id,
        "table_fq_name": rule.table_fq_name,
        "column": rule.column,
        "rule_type": rule.rule_type.value,
        "threshold": rule.threshold,
        "severity": rule.severity.value,
        "status": rule.status.value,
        "params": rule.params,
        "rationale": rule.rationale,
        "adaptive": rule.adaptive,
        "created_by": rule.created_by,
        "created_at": rule.created_at,
        "updated_at": rule.updated_at,
        "version": rule.version,
    }


def _dict_to_rule(d: dict[str, Any]) -> Rule:
    return Rule(
        rule_id=d["rule_id"],
        table_fq_name=d["table_fq_name"],
        column=d["column"],
        rule_type=RuleType(d["rule_type"]),
        threshold=d["threshold"],
        severity=Severity(d["severity"]),
        status=RuleStatus(d["status"]),
        params=d.get("params") or {},
        rationale=d.get("rationale") or "",
        adaptive=bool(d.get("adaptive", True)),
        created_by=d.get("created_by") or "system",
        created_at=d["created_at"],
        updated_at=d["updated_at"],
        version=d.get("version", 1),
    )


class JsonRuleRepository:
    """Drop-in alternative to `RuleRepository`, backed by one JSON file.

    File shape::

        {
          "rules": {"<rule_id>": {...}, ...},
          "threshold_history": {"<rule_id>": [{...}, ...], ...}
        }
    """

    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        if self.json_path.exists():
            self._data = json.loads(self.json_path.read_text() or "{}")
        else:
            self._data = {}
        self._data.setdefault("rules", {})
        self._data.setdefault("threshold_history", {})
        self._flush()

    def _flush(self) -> None:
        self.json_path.write_text(json.dumps(self._data, indent=2, sort_keys=True))

    # -- create / read -------------------------------------------------
    def add_rule(self, rule: Rule) -> Rule:
        self._data["rules"][rule.rule_id] = _rule_to_dict(rule)
        self._flush()
        return rule

    def get_rule(self, rule_id: str) -> Optional[Rule]:
        d = self._data["rules"].get(rule_id)
        return _dict_to_rule(d) if d else None

    def list_rules(
        self, table_fq_name: Optional[str] = None, status: Optional[RuleStatus] = None
    ) -> list[Rule]:
        rules = [_dict_to_rule(d) for d in self._data["rules"].values()]
        if table_fq_name:
            rules = [r for r in rules if r.table_fq_name == table_fq_name]
        if status:
            rules = [r for r in rules if r.status == status]
        return rules

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
        d = self._data["rules"].get(rule_id)
        if d is None:
            raise KeyError(f"No such rule: {rule_id}")
        d["status"] = status.value
        d["updated_at"] = time.time()
        self._flush()
        return _dict_to_rule(d)

    # -- threshold adjustment -------------------------------------------
    def update_threshold(
        self,
        rule_id: str,
        new_threshold: float,
        reason: str,
        auto_adjusted: bool = False,
        required_reapproval: bool = False,
    ) -> Rule:
        d = self._data["rules"].get(rule_id)
        if d is None:
            raise KeyError(f"No such rule: {rule_id}")
        now = time.time()
        history = self._data["threshold_history"].setdefault(rule_id, [])
        history.append(
            {
                "old_threshold": d["threshold"],
                "new_threshold": new_threshold,
                "changed_at": now,
                "reason": reason,
                "auto_adjusted": auto_adjusted,
                "required_reapproval": required_reapproval,
            }
        )
        d["threshold"] = new_threshold
        if required_reapproval:
            d["status"] = RuleStatus.APPROVED.value
        d["updated_at"] = now
        d["version"] = d.get("version", 1) + 1
        self._flush()
        return _dict_to_rule(d)

    def get_threshold_history(self, rule_id: str) -> list[dict]:
        return list(self._data["threshold_history"].get(rule_id, []))

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass
