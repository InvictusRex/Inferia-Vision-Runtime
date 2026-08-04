from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from ..model_management.runtime_config_space import RuntimeConfig
from ..runtime.system_telemetry import Telemetry


@dataclass
class Detections:
    boxes: np.ndarray
    confs: np.ndarray
    cls_ids: np.ndarray
    names: list[str]
    latency_ms: float

    def __len__(self) -> int:
        return int(self.confs.shape[0])


class Detector:
    def __init__(
        self,
        weight_paths: dict[str, str],
        conf: float = 0.25,
        device: str = "cuda",
        telemetry: Telemetry | None = None,
    ):
        self._weight_paths = weight_paths
        self._conf = conf
        self._device = device
        self._telemetry = telemetry
        self._models: dict[tuple[str, str], YOLO] = {}

    def _get_model(self, model: str, precision: str) -> YOLO:
        key = (model, precision)
        if key not in self._models:
            if model not in self._weight_paths:
                raise KeyError(f"no weights registered for model {model!r}")
            m = YOLO(self._weight_paths[model])
            m.to(self._device)
            if precision == "fp16":
                m = m.half()
            self._models[key] = m
        return self._models[key]

    def detect(self, frame: np.ndarray, config: RuntimeConfig) -> Detections:
        model = self._get_model(config.model, config.precision)
        t0 = time.perf_counter()
        result = model.predict(
            frame,
            conf=self._conf,
            imgsz=config.resolution,
            verbose=False,
            device=self._device,
        )[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if self._telemetry is not None:
            self._telemetry.record(latency_ms)

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            empty = np.zeros((0, 4), dtype=np.float32)
            return Detections(
                boxes=empty,
                confs=np.zeros(0, dtype=np.float32),
                cls_ids=np.zeros(0, dtype=int),
                names=[],
                latency_ms=latency_ms,
            )
        cls_ids = boxes.cls.cpu().numpy().astype(int).ravel()
        return Detections(
            boxes=boxes.xyxy.cpu().numpy().reshape(-1, 4),
            confs=boxes.conf.cpu().numpy().ravel(),
            cls_ids=cls_ids,
            names=[model.names[int(i)] for i in cls_ids],
            latency_ms=latency_ms,
        )
