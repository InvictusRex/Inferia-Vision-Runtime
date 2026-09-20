from __future__ import annotations

import numpy as np

from runtime.system_telemetry import GpuSnapshot, Telemetry
from vision_pipeline.scene_analyzer import SceneFeatures


class FeatureBuilder:
    DIM = 14
    MAX_COUNT = 40.0
    MAX_FPS = 60.0
    MAX_LATENCY = 100.0
    MAX_NPU_UTIL = 100.0

    def __init__(
        self,
        n_actions: int = 3,
        power_budget_w: float = 40.0,
        ambient_temp_c: float = 35.0,
        throttle_temp_c: float = 80.0,
    ):
        self.n_actions = max(2, int(n_actions))
        # Dims 10-13 are normalized against the `simulated`/`policy` hardware
        # values (configs/hardware.yaml), not fixed constants, so the observation
        # tracks whatever budgets the current hardware profile declares.
        self.power_budget_w = max(1e-6, float(power_budget_w))
        self.ambient_temp_c = float(ambient_temp_c)
        self.throttle_temp_c = float(throttle_temp_c)
        self._temp_span = max(1e-6, self.throttle_temp_c - self.ambient_temp_c)

    @property
    def dim(self) -> int:
        return self.DIM

    @staticmethod
    def _gpu(telemetry: Telemetry) -> GpuSnapshot:
        gpu = telemetry.gpu
        return gpu if gpu is not None else GpuSnapshot()

    def build(self, scene: SceneFeatures, telemetry: Telemetry, action: int) -> np.ndarray:
        gpu = self._gpu(telemetry)
        v = np.zeros(self.DIM, dtype=np.float32)
        v[0] = float(np.clip(scene.motion, 0.0, 1.0))
        v[1] = float(np.clip(scene.obj_count / self.MAX_COUNT, 0.0, 1.0))
        v[2] = float(np.clip(scene.mean_conf, 0.0, 1.0))
        v[3] = float(np.clip(scene.mean_box_area, 0.0, 1.0))
        v[4] = float(np.clip(scene.brightness, 0.0, 1.0))
        v[5] = float(np.clip(scene.entropy, 0.0, 1.0))
        v[6] = float(np.clip(scene.temporal_consistency, 0.0, 1.0))
        v[7] = float(np.clip(telemetry.fps() / self.MAX_FPS, 0.0, 1.0))
        v[8] = float(np.clip(telemetry.last_latency_ms() / self.MAX_LATENCY, 0.0, 1.0))
        v[9] = float(np.clip(action / (self.n_actions - 1), 0.0, 1.0))
        # NPU compute-capacity pressure.
        v[10] = float(np.clip(gpu.npu_util_pct / self.MAX_NPU_UTIL, 0.0, 1.0))
        # Shared-memory pressure (fraction of the IVR inference-memory budget).
        v[11] = float(np.clip(gpu.mem_frac, 0.0, 1.0))
        # Thermal state, scaled across the ambient->throttle range.
        v[12] = float(np.clip((gpu.temp_c - self.ambient_temp_c) / self._temp_span, 0.0, 1.0))
        # Power-budget pressure (fraction of the IVR policy power budget).
        v[13] = float(np.clip(gpu.power_w / self.power_budget_w, 0.0, 1.0))
        return v
