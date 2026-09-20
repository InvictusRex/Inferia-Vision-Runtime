from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class GpuSnapshot:
    """Emulated ROCK 5C NPU/shared-memory state for the current action (or None if unset)."""

    npu_util_pct: float = 0.0
    mem_used_mb: float = 0.0
    mem_budget_mb: float = 0.0
    temp_c: float = 0.0
    power_w: float = 0.0
    latency_ms: float = 0.0

    @property
    def mem_frac(self) -> float:
        return float(self.mem_used_mb / self.mem_budget_mb) if self.mem_budget_mb > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "npu_util_pct": self.npu_util_pct,
            "mem_used_mb": self.mem_used_mb,
            "mem_budget_mb": self.mem_budget_mb,
            "temp_c": self.temp_c,
            "power_w": self.power_w,
            "latency_ms": self.latency_ms,
        }


@dataclass
class Telemetry:
    latencies_ms: list[float] = field(default_factory=list)
    gpu: Optional[GpuSnapshot] = None

    def reset(self) -> None:
        self.latencies_ms = []
        self.gpu = None

    def record(self, latency_ms: float) -> None:
        self.latencies_ms.append(float(latency_ms))

    def record_gpu(self, snapshot: GpuSnapshot) -> None:
        self.gpu = snapshot

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
