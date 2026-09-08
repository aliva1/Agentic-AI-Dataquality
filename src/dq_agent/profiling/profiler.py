"""Column- and table-level data profiling.

Pure pandas — no LLM calls here. The profile this produces is what both
the rule suggester (LLM-assisted) and the adaptive threshold manager
(purely statistical) build on top of.
"""

from __future__ import annotations

import re

import pandas as pd

from dq_agent.models import ColumnProfile, TableProfile

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$")),
    ("uuid", re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")),
    ("date_iso", re.compile(r"^\d{4}-\d{2}-\d{2}$")),
    ("datetime_iso", re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")),
    ("integer_id", re.compile(r"^\d+$")),
    ("phone", re.compile(r"^\+?[\d\-\(\) ]{7,15}$")),
    ("zip_code_us", re.compile(r"^\d{5}(-\d{4})?$")),
]

_PATTERN_SAMPLE_SIZE = 200
_PATTERN_MATCH_THRESHOLD = 0.9  # fraction of sampled non-null values that must match


def _infer_pattern(series: pd.Series) -> str | None:
    sample = series.dropna().astype(str)
    if sample.empty:
        return None
    sample = sample.sample(min(len(sample), _PATTERN_SAMPLE_SIZE), random_state=0)
    for name, pattern in _PATTERNS:
        matches = sample.str.match(pattern).sum()
        if matches / len(sample) >= _PATTERN_MATCH_THRESHOLD:
            return name
    return None


def profile_column(series: pd.Series, row_count: int) -> ColumnProfile:
    null_count = int(series.isna().sum())
    non_null = series.dropna()
    distinct_count = int(non_null.nunique())

    min_value = max_value = mean = stddev = None
    if pd.api.types.is_numeric_dtype(series) and not non_null.empty:
        min_value = float(non_null.min())
        max_value = float(non_null.max())
        mean = float(non_null.mean())
        stddev = float(non_null.std()) if len(non_null) > 1 else 0.0
    elif not non_null.empty:
        try:
            min_value = non_null.min()
            max_value = non_null.max()
        except TypeError:
            pass

    top_values: list[tuple[object, int]] = []
    if not non_null.empty:
        vc = non_null.value_counts().head(5)
        top_values = list(vc.items())

    inferred_pattern = None
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        inferred_pattern = _infer_pattern(series)
    elif pd.api.types.is_integer_dtype(series):
        inferred_pattern = "integer_id" if distinct_count == len(non_null) else None

    is_candidate_key = row_count > 0 and distinct_count == row_count and null_count == 0

    return ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        row_count=row_count,
        null_count=null_count,
        distinct_count=distinct_count,
        min_value=min_value,
        max_value=max_value,
        mean=mean,
        stddev=stddev,
        top_values=top_values,
        inferred_pattern=inferred_pattern,
        is_candidate_key=is_candidate_key,
    )


def profile_table(df: pd.DataFrame, table_fq_name: str) -> TableProfile:
    row_count = len(df)
    columns = {col: profile_column(df[col], row_count) for col in df.columns}
    candidate_grain = [name for name, cp in columns.items() if cp.is_candidate_key]
    return TableProfile(
        table_fq_name=table_fq_name,
        row_count=row_count,
        columns=columns,
        candidate_grain=candidate_grain,
    )
