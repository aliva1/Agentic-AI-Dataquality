from click.testing import CliRunner

from dq_agent.cli import main


def test_cli_scan_propose_approve_activate_run(tmp_sqlite_db, workdir):
    runner = CliRunner()
    common = ["--source", "sqlite", "--conn", tmp_sqlite_db, "--table", "customers", "--workdir", workdir]

    result = runner.invoke(main, ["scan", *common])
    assert result.exit_code == 0, result.output
    assert "customer_id" in result.output

    result = runner.invoke(main, ["propose", *common])
    assert result.exit_code == 0, result.output
    assert "rule(s) proposed" in result.output

    result = runner.invoke(main, ["list-rules", "--table", "sqlite.customers", "--status", "proposed", "--workdir", workdir])
    assert result.exit_code == 0, result.output
    rule_lines = [l for l in result.output.splitlines() if l.strip()]
    assert len(rule_lines) > 0
    rule_id = rule_lines[0].split("]")[0].strip("[")

    result = runner.invoke(main, ["approve", "--rule-id", rule_id, "--workdir", workdir])
    assert result.exit_code == 0, result.output

    result = runner.invoke(main, ["activate", "--table", "sqlite.customers", "--workdir", workdir])
    assert result.exit_code == 0, result.output
    assert "Activated 1 rule(s)" in result.output

    result = runner.invoke(main, ["run", *common])
    assert result.exit_code == 0, result.output
    assert "Gate decision" in result.output

    result = runner.invoke(main, ["knowledge-add", "--table", "sqlite.customers", "--title", "t", "--content", "c", "--workdir", workdir])
    assert result.exit_code == 0, result.output

    result = runner.invoke(main, ["tickets", "--table", "sqlite.customers", "--workdir", workdir])
    assert result.exit_code == 0, result.output
