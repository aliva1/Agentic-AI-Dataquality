"""Command-line entry point.

    dq-agent scan      --source sqlite --conn sqlite:///demo.db --table customers
    dq-agent knowledge-add --table main.customers --title "PII" --content "email is scrubbed in staging"
    dq-agent propose   --source sqlite --conn sqlite:///demo.db --table customers
    dq-agent list-rules --table main.customers --status proposed
    dq-agent approve    --rule-id rule_abc123
    dq-agent activate   --table main.customers
    dq-agent run        --source sqlite --conn sqlite:///demo.db --table customers
    dq-agent tickets    --table main.customers

All state (rule repo, knowledge repo, ticket store) lives under
`--workdir` (default `./data`) as plain SQLite files, so runs are
resumable and inspectable with any SQLite browser.
"""

from __future__ import annotations

import os
from pathlib import Path

import click

from dq_agent.connectors.registry import create_connector
from dq_agent.gating.gate import DataGate
from dq_agent.knowledge.rules_repo import RuleRepository
from dq_agent.knowledge.store import KnowledgeRepository
from dq_agent.llm.client import AnthropicClient, MockLLMClient
from dq_agent.llm.dq_reasoner import DQReasoner
from dq_agent.models import RuleStatus
from dq_agent.orchestrator.agent import DataQualityAgent
from dq_agent.ticketing.local_sink import LocalTicketSink


def _build_agent(workdir: str, source: str, conn: str | None, schema: str | None, path: str | None, table_name: str) -> DataQualityAgent:
    Path(workdir).mkdir(parents=True, exist_ok=True)

    if source == "flatfile":
        connector = create_connector("flatfile", source_name="flatfile", path=path, logical_table=table_name)
    else:
        connector = create_connector(source, source_name=source, connection_string=conn, schema=schema)

    knowledge_repo = KnowledgeRepository(os.path.join(workdir, "knowledge.db"))
    rule_repo = RuleRepository(os.path.join(workdir, "rules.db"))
    ticket_sink = LocalTicketSink(os.path.join(workdir, "tickets.db"))
    gate = DataGate(os.path.join(workdir, "quarantine"))

    llm = AnthropicClient() if os.environ.get("ANTHROPIC_API_KEY") else MockLLMClient()
    reasoner = DQReasoner(llm)

    return DataQualityAgent(connector, knowledge_repo, rule_repo, reasoner, ticket_sink, gate)


_source_option = click.option("--source", required=True, help="postgresql | snowflake | databricks | sqlite | flatfile")
_conn_option = click.option("--conn", default=None, help="SQLAlchemy connection string (for DB sources)")
_schema_option = click.option("--schema", default=None, help="Default schema (for DB sources)")
_path_option = click.option("--path", default=None, help="File path (for flatfile source)")
_table_option = click.option("--table", required=True, help="Table / logical name to operate on")
_workdir_option = click.option("--workdir", default="./data", help="Local state directory")


@click.group()
def main():
    """dq-agent: a pluggable, LLM-assisted data quality agent."""


@main.command()
@_source_option
@_conn_option
@_schema_option
@_path_option
@_table_option
@_workdir_option
def scan(source, conn, schema, path, table, workdir):
    """Scan metadata, lineage, and profile a table."""
    agent = _build_agent(workdir, source, conn, schema, path, table)
    metadata, profile = agent.scan_table(table)
    click.echo(f"Table: {metadata.fq_name} (~{metadata.row_count_estimate} rows)")
    for col in metadata.columns:
        click.echo(f"  - {col.name}: {col.data_type} (nullable={col.nullable}, pk={col.is_primary_key})")
    lineage = agent.connector.get_lineage(table)
    if lineage:
        click.echo("Lineage:")
        for edge in lineage:
            click.echo(f"  {edge.upstream} -> {edge.downstream} ({edge.relationship})")
    else:
        click.echo("Lineage: none found / not supported for this source")
    click.echo(f"Profiled {profile.row_count} sample rows; candidate grain: {profile.candidate_grain}")


@main.command("knowledge-add")
@_table_option
@click.option("--title", required=True)
@click.option("--content", required=True)
@_workdir_option
def knowledge_add(table, title, content, workdir):
    """Add a business-knowledge note about a table (used by rule suggestion + RCA)."""
    knowledge_repo = KnowledgeRepository(os.path.join(workdir, "knowledge.db"))
    from dq_agent.models import KnowledgeDoc

    doc = KnowledgeDoc(table_fq_name=table, title=title, content=content)
    knowledge_repo.add_doc(doc)
    click.echo(f"Added knowledge doc {doc.doc_id} for {table}")


@main.command()
@_source_option
@_conn_option
@_schema_option
@_path_option
@_table_option
@_workdir_option
def propose(source, conn, schema, path, table, workdir):
    """Profile the table and propose DQ rules for human approval."""
    agent = _build_agent(workdir, source, conn, schema, path, table)
    rules = agent.propose_rules(table)
    for r in rules:
        click.echo(f"[{r.rule_id}] {r.column} {r.rule_type.value} threshold={r.threshold} ({r.severity.value}) — {r.rationale}")
    click.echo(f"{len(rules)} rule(s) proposed. Review with `list-rules --status proposed`, then `approve --rule-id ...`.")


@main.command("list-rules")
@_table_option
@click.option("--status", default=None, type=click.Choice([s.value for s in RuleStatus]))
@_workdir_option
def list_rules(table, status, workdir):
    rule_repo = RuleRepository(os.path.join(workdir, "rules.db"))
    rules = rule_repo.list_rules(table_fq_name=table, status=RuleStatus(status) if status else None)
    for r in rules:
        click.echo(f"[{r.rule_id}] {r.status.value:9s} {r.column} {r.rule_type.value} threshold={r.threshold}")


@main.command()
@click.option("--rule-id", required=True)
@_workdir_option
def approve(rule_id, workdir):
    rule_repo = RuleRepository(os.path.join(workdir, "rules.db"))
    rule = rule_repo.approve_rule(rule_id)
    click.echo(f"Approved {rule.rule_id} (threshold={rule.threshold})")


@main.command()
@_table_option
@_workdir_option
def activate(table, workdir):
    rule_repo = RuleRepository(os.path.join(workdir, "rules.db"))
    approved = rule_repo.list_rules(table_fq_name=table, status=RuleStatus.APPROVED)
    for r in approved:
        rule_repo.activate_rule(r.rule_id)
    click.echo(f"Activated {len(approved)} rule(s) for {table}")


@main.command()
@_source_option
@_conn_option
@_schema_option
@_path_option
@_table_option
@_workdir_option
def run(source, conn, schema, path, table, workdir):
    """Run all ACTIVE rules against the table (batch mode)."""
    agent = _build_agent(workdir, source, conn, schema, path, table)
    report = agent.run_batch(table)
    for r in report.results:
        click.echo(f"  {r.status.value:5s} {r.column} {r.rule_type.value} value={r.metric_value} threshold={r.threshold}")
    click.echo(f"Gate decision: {report.gate_result.decision.value} ({report.gate_result.reasons})")
    if report.tickets:
        for t in report.tickets:
            click.echo(f"Ticket created: [{t.ticket_id}] {t.title}")


@main.command()
@_table_option
@_workdir_option
def tickets(table, workdir):
    ticket_sink = LocalTicketSink(os.path.join(workdir, "tickets.db"))
    for t in ticket_sink.list_tickets(table_fq_name=table):
        click.echo(f"[{t.ticket_id}] ({t.status.value}) {t.title}")


if __name__ == "__main__":
    main()
