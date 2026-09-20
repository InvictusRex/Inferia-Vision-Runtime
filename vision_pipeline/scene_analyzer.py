from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .yolo_detector import Detections


@dataclass
class SceneFeatures:
    motion: float = 0.0
    obj_count: int = 0
    mean_conf: float = 0.0
    mean_box_area: float = 0.0
    brightness: float = 0.0
    entropy: float = 0.0
    temporal_consistency: float = 1.0


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


class SceneAnalyzer:
    def __init__(
        self,
        scale: tuple[int, int] = (160, 90),
        entropy_bins: int = 64,
        max_entropy: float = 8.0,
        iou_threshold: float = 0.2,
    ):
        self._scale = scale
        self._entropy_bins = entropy_bins
        self._max_entropy = max_entropy
        self._iou_threshold = iou_threshold
        self._prev_gray: np.ndarray | None = None
        self._prev_boxes: np.ndarray | None = None
        self._prev_cls: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_boxes = None
        self._prev_cls = None

    def analyze(self, frame: np.ndarray, detections: Detections) -> SceneFeatures:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, self._scale)

        motion = 0.0
        if self._prev_gray is not None:
            motion = float(np.abs(gray.astype(np.float32) - self._prev_gray).mean() / 255.0)

        brightness = float(gray.mean() / 255.0)
        hist = cv2.calcHist([gray], [0], None, [self._entropy_bins], [0, 256])
        hist = hist / hist.sum()
        p = hist[hist > 0]
        entropy = float(-(p * np.log2(p)).sum() / self._max_entropy)

        count = len(detections)
        mean_conf = float(detections.confs.mean()) if count else 0.0
        frame_h, frame_w = frame.shape[:2]
        if count and frame_w and frame_h:
            boxes = detections.boxes
            areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
                0.0, boxes[:, 3] - boxes[:, 1]
            )
            mean_box_area = float(areas.mean() / (frame_w * frame_h))
        else:
            mean_box_area = 0.0

        consistency = self._consistency(detections)

        self._prev_gray = gray
        self._prev_boxes = detections.boxes.copy() if count else None
        self._prev_cls = detections.cls_ids.copy() if count else None

        return SceneFeatures(
            motion=motion,
            obj_count=count,
            mean_conf=mean_conf,
            mean_box_area=mean_box_area,
            brightness=brightness,
            entropy=entropy,
            temporal_consistency=consistency,
        )

    def _consistency(self, detections: Detections) -> float:
        prev_boxes = self._prev_boxes
        prev_cls = self._prev_cls
        if prev_boxes is None or len(prev_boxes) == 0:
            return 1.0
        if len(detections) == 0:
            return 0.0
        ious = _iou(prev_boxes, detections.boxes)
        cls_match = prev_cls[:, None] == detections.cls_ids[None, :]
        good = ious >= self._iou_threshold
        matched = bool(((good & cls_match).max(axis=1)).sum())
        return matched / len(prev_boxes)
