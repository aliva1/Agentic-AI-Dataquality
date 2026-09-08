import pandas as pd

from dq_agent.profiling.profiler import profile_table


def test_profile_table_basic_stats():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "email": ["a@x.com", "b@x.com", "not-an-email", None, "e@x.com"],
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    profile = profile_table(df, "t.customers")
    assert profile.row_count == 5
    assert profile.columns["id"].is_candidate_key is True
    assert "id" in profile.candidate_grain

    email_profile = profile.columns["email"]
    assert email_profile.null_count == 1
    assert email_profile.null_rate == 0.2

    amount_profile = profile.columns["amount"]
    assert amount_profile.min_value == 10.0
    assert amount_profile.max_value == 50.0
    assert amount_profile.mean == 30.0


def test_profile_table_detects_email_pattern():
    df = pd.DataFrame({"email": [f"user{i}@example.com" for i in range(20)]})
    profile = profile_table(df, "t.users")
    assert profile.columns["email"].inferred_pattern == "email"


def test_profile_table_handles_empty_dataframe():
    df = pd.DataFrame({"a": pd.Series(dtype="float64")})
    profile = profile_table(df, "t.empty")
    assert profile.row_count == 0
    assert profile.columns["a"].null_rate == 0.0
