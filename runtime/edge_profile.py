from __future__ import annotations

from dataclasses import dataclass, field

from model_management.runtime_config_space import RuntimeConfig
from .latency_estimator import LatencyEstimator

PRECISION_BYTES = {"fp32": 1.0, "fp16": 0.5, "int8": 0.25}
PRECISION_FACTOR = {"fp32": 1.0, "fp16": 0.6, "int8": 0.35}

# Maps a stress-test constraint key (configs/reward_constrained_bdd.yaml) to the
# EdgeSnapshot/GpuSnapshot metric it bounds. Shared by EdgeProfile.constraint_violations
# and RewardCalculator so the threshold comparison exists in exactly one place.
CONSTRAINT_METRIC_KEYS = {
    "max_npu_util": "npu_util_pct",
    "max_memory_mb": "mem_used_mb",
    "max_temp_c": "temp_c",
    "max_power_w": "power_w",
    "max_latency_ms": "latency_ms",
}


def compute_constraint_violations(metrics: dict, constraints: dict) -> dict:
    """Return {constraint_name: fractional violation} for enabled, exceeded constraints.

    `metrics` maps metric names (npu_util_pct, mem_used_mb, temp_c, power_w,
    latency_ms) to their current values -- either an EdgeSnapshot or a GpuSnapshot
    provides these under the same names. Violation is positive when the metric
    exceeds its limit: (value - limit) / limit. Satisfied or unset constraints are
    omitted.
    """
    violations: dict = {}
    for key, metric_name in CONSTRAINT_METRIC_KEYS.items():
        limit = constraints.get(key)
        if limit is None:
            continue
        value = metrics.get(metric_name)
        if value is None:
            continue
        if value > limit:
            violations[key] = (value - limit) / max(limit, 1e-9)
    return violations


@dataclass
class EdgeSnapshot:
    """Emulated edge-device metrics for one action, for one frame."""

    latency_ms: float = 0.0
    mem_used_mb: float = 0.0
    npu_util_pct: float = 0.0
    power_w: float = 0.0
    temp_c: float = 0.0

    def as_dict(self) -> dict:
        return {
            "latency_ms": self.latency_ms,
            "mem_used_mb": self.mem_used_mb,
            "npu_util_pct": self.npu_util_pct,
            "power_w": self.power_w,
            "temp_c": self.temp_c,
        }


@dataclass
class PhysicalSpec:
    """Vendor-published specification of the real 8 GB ROCK 5C. Not tunable."""

    soc: str = "RK3588S2"
    npu_tops_int8: float = 6.0
    memory_type: str = "LPDDR4X"
    memory_shared: bool = True
    system_memory_mb: float = 8192.0


@dataclass
class SimulatedParams:
    """Synthetic IVR simulator parameters. NOT measured, NOT board specifications."""

    npu_efficiency: float = 0.5
    activation_scale_mb: float = 60.0
    power_idle_w: float = 3.0
    power_envelope_w: float = 15.0
    mem_bandwidth_power_share: float = 0.25
    ambient_temp_c: float = 35.0
    temp_rise_c: float = 40.0
    temp_ewma_alpha: float = 0.2
    thermal_throttle_temp_c: float = 80.0
    thermal_throttle_latency_factor: float = 1.5
    thermal_throttle_enabled: bool = False


@dataclass
class PolicyBudget:
    """IVR runtime budgets: project choices, not physical hardware limits."""

    power_budget_w: float = 40.0
    inference_memory_budget_mb: float = 512.0
    target_fps: float = 30.0


@dataclass
class HardwareProfile:
    """ROCK 5C simulation profile, split into physical / simulated / policy categories."""

    name: str = "rock_5c_rk3588s2"
    physical: PhysicalSpec = field(default_factory=PhysicalSpec)
    simulated: SimulatedParams = field(default_factory=SimulatedParams)
    policy: PolicyBudget = field(default_factory=PolicyBudget)


class EdgeProfile:
    """Emulates per-config edge-device metrics (memory / NPU util / power / temp / latency).

    Models the ROCK 5C (RK3588S2, 6 TOPS INT8 NPU, shared LPDDR4X memory) as a
    synthetic simulator so the RL agent trains against a consistent, device-agnostic
    cost model instead of the noisy live behavior of the dev GPU. `simulated` values
    are temporary IVR parameters, not measurements; they are to be calibrated with
    real ROCK 5C benchmarks in Phase 3 (see `measured:` in configs/hardware.yaml).
    """

    def __init__(
        self,
        profile: HardwareProfile,
        model_gflops: dict[str, float],
        model_weights_mb: dict[str, float],
        latency_estimator: LatencyEstimator,
    ):
        self.profile = profile
        self.model_gflops = model_gflops
        self.model_weights_mb = model_weights_mb
        self.latency = latency_estimator
        self._temp = profile.simulated.ambient_temp_c
        self._last_config: RuntimeConfig | None = None

    @classmethod
    def from_configs(
        cls,
        hw_cfg: dict,
        variants: dict,
        latency_estimator: LatencyEstimator,
    ) -> EdgeProfile:
        phys_cfg = hw_cfg.get("physical", {}) or {}
        sim_cfg = hw_cfg.get("simulated", {}) or {}
        pol_cfg = hw_cfg.get("policy", {}) or {}
        profile = HardwareProfile(
            name=str(hw_cfg.get("name", "rock_5c_rk3588s2")),
            physical=PhysicalSpec(
                soc=str(phys_cfg.get("soc", "RK3588S2")),
                npu_tops_int8=float(phys_cfg.get("npu_tops_int8", 6.0)),
                memory_type=str(phys_cfg.get("memory_type", "LPDDR4X")),
                memory_shared=bool(phys_cfg.get("memory_shared", True)),
                system_memory_mb=float(phys_cfg.get("system_memory_mb", 8192.0)),
            ),
            simulated=SimulatedParams(
                npu_efficiency=float(sim_cfg.get("npu_efficiency", 0.5)),
                activation_scale_mb=float(sim_cfg.get("activation_scale_mb", 60.0)),
                power_idle_w=float(sim_cfg.get("power_idle_w", 3.0)),
                power_envelope_w=float(sim_cfg.get("power_envelope_w", 15.0)),
                mem_bandwidth_power_share=float(sim_cfg.get("mem_bandwidth_power_share", 0.25)),
                ambient_temp_c=float(sim_cfg.get("ambient_temp_c", 35.0)),
                temp_rise_c=float(sim_cfg.get("temp_rise_c", 40.0)),
                temp_ewma_alpha=float(sim_cfg.get("temp_ewma_alpha", 0.2)),
                thermal_throttle_temp_c=float(sim_cfg.get("thermal_throttle_temp_c", 80.0)),
                thermal_throttle_latency_factor=float(
                    sim_cfg.get("thermal_throttle_latency_factor", 1.5)
                ),
                thermal_throttle_enabled=bool(sim_cfg.get("thermal_throttle_enabled", False)),
            ),
            policy=PolicyBudget(
                power_budget_w=float(pol_cfg.get("power_budget_w", 40.0)),
                inference_memory_budget_mb=float(pol_cfg.get("inference_memory_budget_mb", 512.0)),
                target_fps=float(pol_cfg.get("target_fps", 30.0)),
            ),
        )
        gflops = {name: variant.gflops for name, variant in variants.items()}
        weights_mb = {name: variant.edge_weights_mb for name, variant in variants.items()}
        return cls(profile, gflops, weights_mb, latency_estimator)

    @property
    def device(self) -> str:
        return self.profile.name

    @property
    def memory_budget_mb(self) -> float:
        """IVR inference-memory budget (policy value) -- NOT the 8 GB system memory."""
        return self.profile.policy.inference_memory_budget_mb

    def reset(self) -> None:
        self._temp = self.profile.simulated.ambient_temp_c
        self._last_config = None

    def snapshot(self, config: RuntimeConfig) -> EdgeSnapshot:
        phys = self.profile.physical
        sim = self.profile.simulated
        pol = self.profile.policy
        model = config.model
        res = max(1, int(config.resolution))
        precision = str(config.precision)
        res_scale = (res / 640.0) ** 2
        prec_factor = float(PRECISION_FACTOR.get(precision, 1.0))
        prec_bytes = float(PRECISION_BYTES.get(precision, 1.0))

        latency_ms = self.latency.latency_ms(model, res, precision)
        if sim.thermal_throttle_enabled and self._temp > sim.thermal_throttle_temp_c:
            latency_ms *= sim.thermal_throttle_latency_factor

        # Shared-memory working set (unified LPDDR4X, no dedicated VRAM pool):
        # model weights at this precision + an activation working set that scales
        # with resolution. Reported against the `policy` inference-memory budget,
        # never against `physical.system_memory_mb` (8192 MB is context, not a budget).
        base_weights = self.model_weights_mb.get(model, 0.0) * prec_bytes
        activation = sim.activation_scale_mb * res_scale
        mem_used_mb = base_weights + activation

        # NPU compute-capacity pressure: per-second compute demand vs. peak NPU
        # capacity. Not clipped here -- an action whose demand exceeds capacity
        # (e.g. yolo11m @960 fp32) legitimately reports >100%; only the bounded
        # observation dimension clips to 1.0.
        gflops = self.model_gflops.get(model, 0.0)
        demand_gflops_per_s = gflops * res_scale * prec_factor * pol.target_fps
        capacity_gflops_per_s = phys.npu_tops_int8 * 1000.0 * sim.npu_efficiency
        npu_util_pct = (
            100.0 * demand_gflops_per_s / max(capacity_gflops_per_s, 1e-9)
            if capacity_gflops_per_s
            else 0.0
        )

        # Power: compute term + memory-bandwidth term, so power is not a
        # deterministic linear function of npu_util_pct alone (both coefficients
        # are `simulated` parameters, not measurements).
        npu_load01 = min(1.0, npu_util_pct / 100.0)
        mem_load01 = min(1.0, mem_used_mb / max(pol.inference_memory_budget_mb, 1e-9))
        compute_share = 1.0 - sim.mem_bandwidth_power_share
        load01 = compute_share * npu_load01 + sim.mem_bandwidth_power_share * mem_load01
        power_w = sim.power_idle_w + (sim.power_envelope_w - sim.power_idle_w) * load01

        target_temp = sim.ambient_temp_c + sim.temp_rise_c * load01
        alpha = sim.temp_ewma_alpha
        self._temp = self._temp * (1.0 - alpha) + target_temp * alpha

        self._last_config = config
        return EdgeSnapshot(
            latency_ms=latency_ms,
            mem_used_mb=mem_used_mb,
            npu_util_pct=npu_util_pct,
            power_w=power_w,
            temp_c=self._temp,
        )

    def constraint_violations(
        self,
        snapshot: EdgeSnapshot,
        constraints: dict,
    ) -> dict:
        """Return {constraint_name: fractional violation} for enabled constraints."""
        return compute_constraint_violations(snapshot.as_dict(), constraints)
