from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..runtime.edge_profile import compute_constraint_violations
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
    latency_source: str = "live"
    latency_jitter_ms: float = 0.0
    constraints: dict = field(default_factory=dict)
    constraint_weights: dict = field(default_factory=dict)
    constraint_default_weight: float = 2.0
    constraint_cliff: float = 0.0
    map_iou_threshold: float = 0.5


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
        gt_boxes: np.ndarray | None = None,
    ) -> float:
        raise NotImplementedError


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1, iy1 = np.maximum(ax1, bx1), np.maximum(ay1, by1)
    ix2, iy2 = np.minimum(ax2, bx2), np.minimum(ay2, by2)
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-9)


def _detection_f1(pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float) -> float:
    """Greedy IoU-matched precision/recall F1. Class-agnostic, single IoU
    threshold -- a fast, honest quality signal, not a full COCO-mAP protocol."""
    if len(gt_boxes) == 0:
        return 1.0 if len(pred_boxes) == 0 else 0.0
    if len(pred_boxes) == 0:
        return 0.0
    ious = _iou_matrix(pred_boxes, gt_boxes)
    matched_gt: set[int] = set()
    tp = 0
    for i in range(len(pred_boxes)):
        best_j, best_iou = -1, iou_threshold
        for j in range(len(gt_boxes)):
            if j in matched_gt:
                continue
            if ious[i, j] > best_iou:
                best_iou, best_j = ious[i, j], j
        if best_j >= 0:
            matched_gt.add(best_j)
            tp += 1
    precision = tp / len(pred_boxes)
    recall = tp / len(gt_boxes)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


class ProxyReward(RewardCalculator):
    def _quality(
        self, detections: Detections, scene: SceneFeatures, gt_boxes: np.ndarray | None
    ) -> float:
        count = len(detections)
        mean_conf = float(detections.confs.mean()) if count else 0.0
        return mean_conf * (count / max(self.cfg.target_count, 1e-6))

    def compute(
        self,
        detections: Detections,
        scene: SceneFeatures,
        telemetry: Telemetry,
        prev_action: int | None,
        action: int,
        gt_boxes: np.ndarray | None = None,
    ) -> float:
        cfg = self.cfg
        quality = self._quality(detections, scene, gt_boxes)
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
        if cfg.constraints:
            gpu = telemetry.gpu
            if gpu is not None:
                violations = compute_constraint_violations(gpu.as_dict(), cfg.constraints)
                for key, frac in violations.items():
                    w = float(cfg.constraint_weights.get(key, cfg.constraint_default_weight))
                    constraint_pen -= w * frac + cfg.constraint_cliff

        fps = telemetry.fps()
        if cfg.min_fps > 0 and 0.0 < fps < cfg.min_fps:
            constraint_pen -= 0.5 * (1.0 - fps / cfg.min_fps)

        return float(quality_term + latency_term + compute_term + switch_pen + constraint_pen)


class MapReward(ProxyReward):
    """Same reward shape as ProxyReward, but quality is a real IoU-matched
    detection F1 against ground-truth boxes instead of a confidence x count
    proxy. Requires gt_boxes to be passed into compute() each step."""

    def _quality(
        self, detections: Detections, scene: SceneFeatures, gt_boxes: np.ndarray | None
    ) -> float:
        if gt_boxes is None:
            raise ValueError("MapReward.compute() requires gt_boxes (per-frame ground truth)")
        return _detection_f1(detections.boxes, gt_boxes, self.cfg.map_iou_threshold)


def build_reward(cfg: RewardConfig) -> RewardCalculator:
    if cfg.quality_backend == "map":
        return MapReward(cfg)
    return ProxyReward(cfg)
