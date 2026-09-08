"""Streaming source interface.

The whole point of this module is that `dq_agent.orchestrator.agent`
processes a `StreamSource` with the *exact* method it uses for a batch
connector's micro-batches: `for micro_batch_df in source.consume(): ...`.
There is no separate "streaming orchestrator" — see
`orchestrator/pipeline.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import pandas as pd


class StreamSource(ABC):
    source_type: str = "stream_base"

    def __init__(self, source_name: str, topic: str):
        self.source_name = source_name
        self.topic = topic

    def __enter__(self) -> "StreamSource":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> None:  # pragma: no cover - default no-op
        pass

    def close(self) -> None:  # pragma: no cover - default no-op
        pass

    @abstractmethod
    def consume(self, max_batches: int | None = None) -> Iterator[pd.DataFrame]:
        """Yield micro-batches as DataFrames.

        `max_batches=None` means "run forever" (real streaming); demos
        and tests pass a small integer to get a bounded run.
        """
