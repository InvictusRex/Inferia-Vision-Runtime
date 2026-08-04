from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Telemetry:
    latencies_ms: list[float] = field(default_factory=list)

    def reset(self) -> None:
        self.latencies_ms = []

    def record(self, latency_ms: float) -> None:
        self.latencies_ms.append(float(latency_ms))

    def last_latency_ms(self) -> float:
        return self.latencies_ms[-1] if self.latencies_ms else 0.0

    def mean_latency_ms(self, window: int = 30) -> float:
        w = self.latencies_ms[-window:]
        if not w:
            return 0.0
        return float(np.mean(w))

    def fps(self, window: int = 30) -> float:
        w = self.latencies_ms[-window:]
        if not w:
            return 0.0
        return 1000.0 / float(np.mean(w) + 1e-9)
