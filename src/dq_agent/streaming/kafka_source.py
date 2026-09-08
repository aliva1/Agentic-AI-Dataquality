"""Real Kafka streaming source.

Requires `kafka-python` and a reachable broker — neither is available in
every environment (including the sandbox this was built in), so this
module is imported lazily by the registry and only touched when someone
actually asks for `source_type="kafka"`. For offline development, demos,
and tests, use `InMemoryStreamSource` in this same package instead — it
implements the identical `StreamSource` interface, so orchestration code
never needs to know which one it's talking to.
"""

from __future__ import annotations

import json
from typing import Iterator, Optional

import pandas as pd

from dq_agent.streaming.base import StreamSource


class KafkaStreamSource(StreamSource):
    source_type = "kafka"

    def __init__(
        self,
        source_name: str,
        bootstrap_servers: str,
        topic: str,
        value_schema: Optional[dict] = None,
        batch_size: int = 500,
        poll_timeout_ms: int = 2000,
    ):
        super().__init__(source_name, topic)
        self.bootstrap_servers = bootstrap_servers
        self.value_schema = value_schema  # documents expected JSON shape; not enforced here
        self.batch_size = batch_size
        self.poll_timeout_ms = poll_timeout_ms
        self._consumer = None

    def connect(self) -> None:
        try:
            from kafka import KafkaConsumer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "kafka-python is required for KafkaStreamSource: pip install kafka-python"
            ) from exc
        self._consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            consumer_timeout_ms=self.poll_timeout_ms,
        )

    def close(self) -> None:
        if self._consumer is not None:
            self._consumer.close()
            self._consumer = None

    def consume(self, max_batches: int | None = None) -> Iterator[pd.DataFrame]:
        if self._consumer is None:
            self.connect()
        batches_yielded = 0
        buffer: list[dict] = []
        for message in self._consumer:  # pragma: no cover - needs a live broker
            buffer.append(message.value)
            if len(buffer) >= self.batch_size:
                yield pd.DataFrame(buffer)
                buffer = []
                batches_yielded += 1
                if max_batches is not None and batches_yielded >= max_batches:
                    return
        if buffer:  # flush partial batch on consumer timeout / stream end
            yield pd.DataFrame(buffer)
