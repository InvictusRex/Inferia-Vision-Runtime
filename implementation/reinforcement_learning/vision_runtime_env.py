from __future__ import annotations

from typing import Iterator, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..model_management.runtime_config_space import ConfigSpace
from ..runtime.latency_estimator import LatencyEstimator
from ..runtime.system_telemetry import Telemetry
from ..vision_pipeline.video_frame_source import FrameSource
from ..vision_pipeline.yolo_detector import Detector, Detections
from ..vision_pipeline.scene_analyzer import SceneAnalyzer, SceneFeatures
from .observation_features import FeatureBuilder
from .reward_calculator import RewardCalculator


class VisionRuntimeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        frame_source: FrameSource,
        config_space: ConfigSpace,
        detector: Detector,
        analyzer: SceneAnalyzer,
        features: FeatureBuilder,
        reward: RewardCalculator,
        telemetry: Optional[Telemetry] = None,
        frame_step: int = 1,
        latency_estimator: Optional[LatencyEstimator] = None,
    ):
        super().__init__()
        self.frame_source = frame_source
        self.config_space = config_space
        self.detector = detector
        self.analyzer = analyzer
        self.features = features
        self.reward = reward
        self.telemetry = telemetry or Telemetry()
        self.frame_step = max(1, int(frame_step))
        self.latency_estimator = latency_estimator

        self.action_space = spaces.Discrete(self.config_space.n_actions)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.features.dim,), dtype=np.float32
        )

        self._frames: Optional[Iterator[np.ndarray]] = None
        self._frame_idx = 0
        self._current_action = 0
        self._prev_action: Optional[int] = None
        self._scene: Optional[SceneFeatures] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._frames = self.frame_source.iter_frames(step=self.frame_step)
        self._frame_idx = 0
        self._current_action = 0
        self._prev_action = None
        self.telemetry.reset()
        self.analyzer.reset()

        frame = next(self._frames, None)
        if frame is None:
            raise RuntimeError("video has no frames")
        self._frame_idx = 1
        detections = self._infer(frame, 0)
        self._scene = self.analyzer.analyze(frame, detections)
        obs = self.features.build(self._scene, self.telemetry, self._current_action)
        info = self._make_info(detections, obs=obs, reward=0.0)
        return obs, info

    def step(self, action: int):
        action = int(action)
        frame = next(self._frames, None)
        if frame is None:
            obs = self.features.build(
                self._scene or SceneFeatures(),
                self.telemetry,
                self._current_action,
            )
            info = self._make_info(
                Detections(
                    np.zeros((0, 4)),
                    np.zeros(0),
                    np.zeros(0, dtype=int),
                    [],
                    self.telemetry.last_latency_ms(),
                ),
                obs=obs,
                reward=0.0,
            )
            info["truncated"] = True
            return obs, 0.0, False, True, info

        detections = self._infer(frame, action)
        scene = self.analyzer.analyze(frame, detections)
        self._scene = scene
        self._frame_idx += 1

        reward = self.reward.compute(
            detections,
            scene,
            self.telemetry,
            self._prev_action,
            action,
        )
        self._prev_action = self._current_action
        self._current_action = action

        obs = self.features.build(scene, self.telemetry, action)
        info = self._make_info(detections, obs=obs, reward=reward)
        return obs, float(reward), False, False, info

    def _infer(self, frame: np.ndarray, action: int) -> Detections:
        config = self.config_space.action_to_config(action)
        detections = self.detector.detect(frame, config)
        if self.latency_estimator is not None and self.telemetry.latencies_ms:
            latency = self.latency_estimator.latency_ms(config.model)
            if latency > 0:
                self.telemetry.latencies_ms[-1] = latency
        return detections

    def _make_info(self, detections: Detections, obs, reward: float) -> dict:
        scene = self._scene
        return {
            "frame": self._frame_idx,
            "detections": len(detections),
            "obj_count": scene.obj_count if scene else 0,
            "mean_conf": scene.mean_conf if scene else 0.0,
            "latency_ms": self.telemetry.last_latency_ms(),
            "fps": self.telemetry.fps(),
            "action": self._current_action,
            "model": self.config_space.action_to_config(self._current_action).model,
            "reward": float(reward),
            "obs": obs,
        }
