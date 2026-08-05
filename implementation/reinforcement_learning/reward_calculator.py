from __future__ import annotations

from dataclasses import dataclass

from ..runtime.system_telemetry import Telemetry
from ..vision_pipeline.yolo_detector import Detections
from ..vision_pipeline.scene_analyzer import SceneFeatures


@dataclass
class RewardConfig:
    quality_backend: str = "proxy"
    w_quality: float = 1.0
    w_latency: float = 1.0
    w_compute: float = 0.5
    w_switch: float = 0.3
    target_count: float = 10.0
    max_latency_ms: float = 40.0
    min_fps: float = 15.0
    labels_path: str | None = None
    compute_cost: tuple[float, ...] = ()


class RewardCalculator:
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg

    def compute(
        self,
        detections: Detections,
        scene: SceneFeatures,
        telemetry: Telemetry,
        prev_action: int | None,
        action: int,
    ) -> float:
        raise NotImplementedError


class ProxyReward(RewardCalculator):
    def compute(
        self,
        detections: Detections,
        scene: SceneFeatures,
        telemetry: Telemetry,
        prev_action: int | None,
        action: int,
    ) -> float:
        cfg = self.cfg
        count = len(detections)
        mean_conf = float(detections.confs.mean()) if count else 0.0
        quality = mean_conf * min(1.0, count / max(cfg.target_count, 1e-6))
        stability = float(scene.temporal_consistency)
        quality_term = cfg.w_quality * (0.7 * quality + 0.3 * stability)

        latency = telemetry.last_latency_ms()
        latency_term = -cfg.w_latency * min(1.0, latency / max(cfg.max_latency_ms, 1e-6))

        compute_term = 0.0
        if cfg.compute_cost and action < len(cfg.compute_cost):
            compute_term = -cfg.w_compute * cfg.compute_cost[action]

        switch_pen = 0.0
        if prev_action is not None and prev_action != action:
            switch_pen = -cfg.w_switch

        constraint_pen = 0.0
        fps = telemetry.fps()
        if cfg.min_fps > 0 and 0.0 < fps < cfg.min_fps:
            constraint_pen = -0.5 * (1.0 - fps / cfg.min_fps)

        return float(quality_term + latency_term + compute_term + switch_pen + constraint_pen)


class MapReward(RewardCalculator):
    def compute(
        self,
        detections: Detections,
        scene: SceneFeatures,
        telemetry: Telemetry,
        prev_action: int | None,
        action: int,
    ) -> float:
        if not self.cfg.labels_path:
            raise NotImplementedError(
                "MapReward requires labels_path; use ProxyReward for unlabeled videos"
            )
        raise NotImplementedError("mAP reward backend not yet implemented")


def build_reward(cfg: RewardConfig) -> RewardCalculator:
    if cfg.quality_backend == "map":
        return MapReward(cfg)
    return ProxyReward(cfg)
