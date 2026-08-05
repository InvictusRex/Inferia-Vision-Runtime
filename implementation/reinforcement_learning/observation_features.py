from __future__ import annotations

import numpy as np

from ..runtime.system_telemetry import Telemetry
from ..vision_pipeline.scene_analyzer import SceneFeatures


class FeatureBuilder:
    DIM = 10
    MAX_COUNT = 40.0
    MAX_FPS = 60.0
    MAX_LATENCY = 100.0

    def __init__(self, n_models: int = 3):
        self.n_models = max(1, int(n_models))

    @property
    def dim(self) -> int:
        return self.DIM

    def build(self, scene: SceneFeatures, telemetry: Telemetry, action: int) -> np.ndarray:
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
        v[9] = float(np.clip(action / (self.n_models - 1), 0.0, 1.0))
        return v
