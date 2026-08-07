from __future__ import annotations

import numpy as np


class LatencyEstimator:
    def __init__(
        self,
        model_latencies: dict[str, float],
        model_gflops: dict[str, float] | None = None,
        jitter_ms: float = 0.0,
    ):
        self.model_latencies = model_latencies
        self.model_gflops = model_gflops or {}
        self.jitter_ms = max(0.0, float(jitter_ms))

    @classmethod
    def from_variants(
        cls,
        variants: dict,
        gflops_by_model: dict[str, float],
        jitter_ms: float = 0.0,
    ) -> LatencyEstimator:
        explicit: dict[str, float] = {}
        for name, variant in variants.items():
            latency = getattr(variant, "edge_latency_ms", 0.0) or 0.0
            if latency > 0:
                explicit[name] = float(latency)
        if explicit:
            return cls(explicit, gflops_by_model, jitter_ms)
        return cls(cls._gflop_proportional(gflops_by_model), gflops_by_model, jitter_ms)

    @staticmethod
    def _gflop_proportional(gflops_by_model: dict[str, float]) -> dict[str, float]:
        if not gflops_by_model:
            return {}
        ref_gflops = min(v for v in gflops_by_model.values() if v > 0)
        base_ms = 18.0
        return {
            name: base_ms * (gflops / ref_gflops)
            for name, gflops in gflops_by_model.items()
            if gflops > 0
        }

    def _latency_of(self, model: str) -> float:
        if model in self.model_latencies:
            return self.model_latencies[model]
        if self.model_gflops and model in self.model_gflops:
            return self._gflop_proportional({model: self.model_gflops[model]}).get(model, 0.0)
        return 0.0

    def latency_ms(self, model: str) -> float:
        latency = self._latency_of(model)
        if latency <= 0:
            return 0.0
        if self.jitter_ms > 0:
            latency += float(np.random.uniform(-self.jitter_ms, self.jitter_ms))
        return max(1e-3, latency)

    def fps(self, model: str) -> float:
        latency = self.latency_ms(model)
        return 1000.0 / latency if latency > 0 else 0.0
