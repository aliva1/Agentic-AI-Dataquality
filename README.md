# dq-agent

A pluggable, LLM-assisted data quality agent: it scans a table's metadata
and lineage, profiles the data, keeps a business-knowledge + rule
repository backed by a RAG store, proposes DQ rules and thresholds for
human approval, then runs those checks — adapting thresholds over time
— against either a batch source or a live stream, routing failures into
a ticketing system and holding back data that's bad enough to warrant
it.

This is a proof-of-concept built to demonstrate the architecture end to
end, not a production platform — see [Scope & design decisions](#scope--design-decisions)
for exactly where the line is drawn and why.

## Why it's shaped this way

```
                          ┌─────────────────────┐
                          │  Business knowledge  │
                          │   + rule repository   │◄──── human approves
                          │   (RAG-backed, RAG    │      rules & big
                          │   = local hashing     │      threshold jumps
                          │   vector store)        │
                          └─────────┬────────────┘
                                    │ retrieval
                                    ▼
   ┌───────────┐   profile   ┌──────────────┐  suggest   ┌──────────┐
   │ Connector  ├────────────►│   Profiler   ├───────────►│   LLM     │
   │ (plugin)   │             └──────────────┘  rules     │ reasoner  │
   │ Postgres / │                                          └────┬─────┘
   │ Snowflake /│                                               │ proposed rules
   │ Databricks/│                                               ▼
   │ SQLite /   │  batch rows or           ┌───────────────────────────┐
   │ flat file  ├─────────────────────────►│   DataQualityAgent         │
   └─────┬──────┘  streaming micro-batches │   (orchestrator)           │
         │                                  │  same code path for       │
   ┌─────▼──────┐  micro-batches            │  batch & streaming        │
   │ StreamSource│─────────────────────────►│                            │
   │ Kafka /     │                          └───┬───────────┬───────────┘
   │ in-memory   │                              │           │
   └─────────────┘                    ┌─────────▼──┐   ┌────▼─────────┐
                                       │ Rule engine │   │ Threshold    │
                                       │ (deterministic)│  manager     │
                                       └─────┬──────┘   │ (EWMA,       │
                                             results     │ auto-tune,   │
                                             │           │ re-approval  │
                                       ┌─────▼──────┐    │ guardrail)   │
                                       │ Gate        │    └──────────────┘
                                       │ pass / warn │
                                       │ / BLOCK +   │
                                       │ quarantine  │
                                       └─────┬──────┘
                                             │ failures
                                       ┌─────▼──────┐
                                       │ Ticket sink │
                                       │ (local /    │
                                       │ webhook)    │
                                       └────────────┘
```

Every box above is a small, independently-testable module behind an
abstract interface (`connectors.base.DataSourceConnector`,
`streaming.base.StreamSource`, `llm.client.LLMClient`,
`ticketing.base.TicketSink`, `knowledge.vector_store.VectorStore`).
`orchestrator.agent.DataQualityAgent` is the only thing that knows about
all of them, and it's built by dependency injection — that's what makes
"change source, orchestration stays the same" actually true rather than
aspirational.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Run the full pipeline against a real public dataset (Chinook — a
# digital media store) with no external services or API keys required:
python demo/run_demo.py
```

A full transcript of that run is saved at [`demo/sample_output.txt`](demo/sample_output.txt).
It walks through, in order: metadata + lineage scan, business-knowledge
ingestion, LLM-assisted rule proposal, human approval, batch execution
with a hand-authored referential rule, adaptive threshold tuning (both
the "quiet auto-adjust" and "pause for re-approval" paths), a forced
CRITICAL failure that blocks and quarantines a batch and opens a ticket,
and the same rules run again over a streaming source to prove the
orchestration is identical.

### CLI

```bash
dq-agent scan       --source sqlite --conn sqlite:///data/chinook.db --table Customer
dq-agent propose    --source sqlite --conn sqlite:///data/chinook.db --table Customer
dq-agent list-rules  --table sqlite.Customer --status proposed
dq-agent approve     --rule-id <rule_id>
dq-agent activate    --table sqlite.Customer
dq-agent run         --source sqlite --conn sqlite:///data/chinook.db --table Customer
dq-agent tickets     --table sqlite.Customer
```

All state (rules, knowledge, tickets, quarantine) lives under
`--workdir` (default `./data`) as plain SQLite files / CSVs — inspect
them with any SQLite browser.

### Tests

```bash
pip install -r requirements.txt
pytest                                  # 48 tests, fully offline/deterministic
pytest --cov=dq_agent --cov-report=term-missing   # ~87% line coverage
```

## Flat-file variant (no database anywhere)

`demo/run_demo.py` uses SQLite for the source data and SQLite for all
of the agent's own state (rules, knowledge, tickets). There's a second
demo that proves the same orchestration works with *no database
anywhere* -- source data and every piece of agent state as plain files:

```bash
python demo/export_flatfiles.py     # exports Customer/Track/InvoiceLine
                                     # from data/chinook.db into
                                     # data/flatfiles/*.csv (same rows,
                                     # same data, just CSV instead of SQL)
python demo/run_demo_flatfile.py    # runs the identical pipeline against them
```

What changes, and what doesn't:

| | `run_demo.py` | `run_demo_flatfile.py` |
|---|---|---|
| Source data | `data/chinook.db` (SQLite) | `data/flatfiles/*.csv` |
| Connector | `SQLConnector(dialect="sqlite")` | `DirectoryFlatFileConnector` |
| Rule repository | `RuleRepository` → `rules.db` | `JsonRuleRepository` → `rules.json` |
| Knowledge repository | `KnowledgeRepository` → `knowledge.db` | `JsonKnowledgeRepository` → `knowledge.json` |
| Ticket sink | `LocalTicketSink` → `tickets.db` | `JsonTicketSink` → `tickets.json` |
| Quarantine | CSV | CSV (unchanged either way) |
| Orchestrator, rule engine, adaptive thresholds, gating, reasoner | identical | identical |

`DirectoryFlatFileConnector` (`connectors/flatfile_connector.py`) treats
a folder of CSVs the way `FlatFileConnector` treats one file: it's the
"table" enumeration a real flat-file drop needs (an SFTP export, an S3
prefix) -- each file's name becomes its logical table, so a 3-table
source works the same as a 3-table database as far as the orchestrator
is concerned.

`JsonRuleRepository`, `JsonKnowledgeRepository`, and `JsonTicketSink`
(`knowledge/rules_repo_json.py`, `knowledge/store_json.py`,
`ticketing/json_sink.py`) are drop-in alternatives to their SQLite
counterparts -- same public methods, same lifecycle, same audit trail
-- backed by one plain, human-readable `.json` file each instead of a
database table. Open `data/demo_state_flatfile/rules.json` in any text
editor after running the demo and you can read every rule, every
threshold change, every ticket, directly.

## Scope & design decisions

This was built to prove the architecture works end to end, in an
environment with no access to a real Postgres/Snowflake/Databricks
warehouse, no Kafka broker, and no LLM API key. Rather than fake those
things, the design makes the boundary explicit and swappable:

- **One connector class covers Postgres, Snowflake, Databricks, and
  SQLite** (`connectors/sql_connector.py`) via SQLAlchemy + a small
  per-dialect lineage strategy (Postgres: `information_schema.view_table_usage`;
  Snowflake: `account_usage.object_dependencies`; Databricks: Unity
  Catalog's `system.access.table_lineage`). The demo and test suite run
  it against SQLite because that needs no live server — the exact same
  class, with a different `dialect=`/`connection_string=`, is what you'd
  point at a real warehouse. Postgres/Snowflake/Databricks integration
  is therefore code-complete but not exercised against a live warehouse
  in CI; that's the one honest gap in an offline proof-of-concept.
- **Kafka streaming is a real implementation** (`streaming/kafka_source.py`,
  `kafka-python`) that needs a live broker to run. `streaming/in_memory_source.py`
  implements the identical `StreamSource` interface without one, which
  is what lets the orchestrator's streaming code path be genuinely
  tested and demoed. Swapping one for the other in `connectors/registry.py`
  is the entire integration surface.
- **The RAG layer is a small self-contained hashing-vectorizer + cosine-similarity
  store** (`knowledge/vector_store.py`), not Chroma/pgvector, so it needs
  no running service and no model download — it works offline and
  deterministically. It implements the same `VectorStore` interface a
  production backend would, so swapping it in is a one-class change.
- **The LLM is pluggable** (`llm/client.py`): `AnthropicClient` is the
  real thing (needs `ANTHROPIC_API_KEY`); `MockLLMClient` is a
  deterministic offline stand-in used by the test suite and the default
  demo run. The reasoner (`llm/dq_reasoner.py`) always computes a
  statistical baseline first and only *refines* it with the LLM, so the
  pipeline degrades gracefully to "still useful, just less clever"
  with zero configuration, and gets measurably smarter with a real key.
- **Ticketing** defaults to a local SQLite sink with the same
  one-ticket-per-incident + duplicate-prevention pattern used in
  production ITSM integrations; `ticketing/webhook_sink.py` is the
  generic REST shape for pointing at a real system.

None of this is a limitation you'd need to work around later — it's the
seam a real deployment plugs into.

## Rule types

| Type | What it measures | Direction |
|---|---|---|
| `completeness` | null rate | lower is better |
| `uniqueness` | distinct rate | higher is better |
| `range` | fraction of values outside `[min, max]` | lower is better |
| `pattern` | fraction matching a regex (email, UUID, date, phone, zip, ...) | higher is better |
| `freshness` | hours since the max timestamp | lower is better |
| `referential` | fraction of values missing from a reference set/table | lower is better |
| `custom_sql` | fraction of rows violating a pandas-evaluable predicate | lower is better |

Every rule has a threshold, a severity (`info` / `warn` / `critical`),
and an `adaptive` flag. A metric drifting toward its threshold shows up
as `WARN` before it ever hits `FAIL` (a configurable buffer zone).

## Rule lifecycle

```
PROPOSED --approve--> APPROVED --activate--> ACTIVE
   │--reject--> REJECTED
ACTIVE --retire--> RETIRED
```

Only `ACTIVE` rules run. Every threshold change — manual or
auto-adjusted — is appended to an audit trail (`rule_threshold_history`),
never overwritten silently.

## Adaptive thresholds — the actual guardrail

The threshold manager (`adaptive/threshold_manager.py`) only learns from
`PASS`/`WARN` observations — a `FAIL` is exactly what the rule exists to
catch, so failures never pull a threshold toward masking themselves.
After enough observations, it computes an EWMA mean/std and proposes a
new threshold:

- **Small relative change (≤20%)** → applied immediately, rule stays
  `ACTIVE`. This is the "quietly adapts to normal drift" behavior.
- **Large relative change (>20%)** → the new threshold is still written
  (with the reason, in the audit trail) but the rule is demoted back to
  `APPROVED` — it stops enforcing until a human re-activates it. A big,
  sudden shift in what "normal" looks like should never silently change
  what the agent will and won't flag.

## Gating (the "hold bad data" circuit breaker)

`gating/gate.py`: any `CRITICAL`-severity rule in `FAIL` status blocks
the batch outright — the data is written to `data/quarantine/<table>/<batch_id>.csv`
instead of being considered ready for the next layer. Anything short of
that but still failing/warning downgrades to `PASS_WITH_WARNINGS`;
otherwise it's a clean `PASS`. This runs identically for a full batch or
one streaming micro-batch.

## Project layout

```
src/dq_agent/
  models.py                 shared dataclasses (no I/O)
  connectors/                DataSourceConnector + Postgres/Snowflake/Databricks/
                              SQLite (one class) + flat file (single file or a
                              directory of files, DirectoryFlatFileConnector) +
                              plugin registry
  streaming/                  StreamSource + Kafka + in-memory
  profiling/                  column/table statistical profiling
  knowledge/                  business-knowledge repo, rule repo, RAG vector store
                              (SQLite-backed, plus JSON-file-backed twins:
                              rules_repo_json.py, store_json.py)
  llm/                        pluggable LLM client + the reasoner (rules/RCA/tickets)
  checks/                     deterministic rule-evaluation engine
  adaptive/                   EWMA threshold auto-tuning + re-approval guardrail
  ticketing/                  local SQLite sink, a JSON-file sink, + generic webhook sink
  gating/                     pass/warn/block + quarantine
  orchestrator/                DataQualityAgent: wires everything together
  cli.py                      command-line entry point
demo/
  run_demo.py                 end-to-end walkthrough against a public dataset (SQLite)
  run_demo_flatfile.py        the same walkthrough with zero databases (flat files + JSON)
  export_flatfiles.py         exports the demo data to data/flatfiles/*.csv
  sample_output.txt           a saved transcript of the SQLite run
tests/                        63 tests, fully offline/deterministic
```

## Author

Aliva Dash
