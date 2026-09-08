from dq_agent.models import CheckResult, RuleType, Severity, CheckStatus
from dq_agent.ticketing.webhook_sink import WebhookTicketSink


def test_webhook_sink_posts_payload_and_tracks_tickets():
    posted = []

    def fake_post(url, json, headers):
        posted.append((url, json, headers))

    sink = WebhookTicketSink("https://itsm.example.com/hook", headers={"Authorization": "Bearer x"}, post_fn=fake_post)
    result = CheckResult(
        rule_id="r1",
        table_fq_name="t.orders",
        column="amount",
        rule_type=RuleType.RANGE,
        status=CheckStatus.FAIL,
        metric_value=0.2,
        threshold=0.0,
        severity=Severity.CRITICAL,
    )
    ticket = sink.create_ticket("t.orders", "Bad amounts", "desc", Severity.CRITICAL, [result])

    assert len(posted) == 1
    url, payload, headers = posted[0]
    assert url == "https://itsm.example.com/hook"
    assert payload["ticket_id"] == ticket.ticket_id
    assert payload["rule_ids"] == ["r1"]
    assert headers["Authorization"] == "Bearer x"

    assert sink.list_tickets("t.orders") == [ticket]
    assert sink.list_tickets("t.other") == []
    assert sink.list_tickets() == [ticket]
