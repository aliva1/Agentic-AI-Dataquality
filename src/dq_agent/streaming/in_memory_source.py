"""An in-memory stand-in for a broker, used by demos and tests.

Splits a DataFrame (or an explicit list of DataFrames) into
micro-batches and yields them exactly like `KafkaStreamSource` would.
This is what proves "same orchestration for batch and streaming"
without needing a running Kafka cluster: swap `KafkaStreamSource` for
`InMemoryStreamSource` and every line of orchestration code is
untouched.
"""

from __future__ import annotations

from typing import Iterator, Optional

import pandas as pd

from dq_agent.streaming.base import StreamSource


class InMemoryStreamSource(StreamSource):
    source_type = "in_memory"

    def __init__(
        self,
        source_name: str,
        topic: str,
        data: pd.DataFrame,
        batch_size: int = 200,
    ):
        super().__init__(source_name, topic)
        self._data = data
        self.batch_size = batch_size

    def consume(self, max_batches: Optional[int] = None) -> Iterator[pd.DataFrame]:
        n = len(self._data)
        batches = 0
        for start in range(0, n, self.batch_size):
            yield self._data.iloc[start : start + self.batch_size].copy()
            batches += 1
            if max_batches is not None and batches >= max_batches:
                return
