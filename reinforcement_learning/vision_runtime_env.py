from __future__ import annotations

from itertools import islice
from typing import Iterator, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from model_management.runtime_config_space import ConfigSpace
from runtime.edge_profile import EdgeProfile, compute_constraint_violations
from runtime.latency_estimator import LatencyEstimator
from runtime.system_telemetry import GpuSnapshot, Telemetry
from vision_pipeline.video_frame_source import FrameSource
from vision_pipeline.yolo_detector import Detector, Detections
from vision_pipeline.scene_analyzer import SceneAnalyzer, SceneFeatures
from .observation_features import FeatureBuilder
from .reward_calculator import RewardCalculator


def deadline_action_mask(latencies_ms: np.ndarray, deadline_ms: float) -> np.ndarray:
    """Actions whose nominal latency fits the deadline; the fastest one if none does."""
    mask = latencies_ms <= deadline_ms
    if not mask.any():
        mask[int(np.argmin(latencies_ms))] = True
    return mask


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
        max_episode_frames: Optional[int] = None,
        episode_min_frames: int = 150,
        edge_profile: Optional[EdgeProfile] = None,
        deadline_levels_ms: Optional[list[float]] = None,
        deadline_hold_frames: tuple[int, int] = (50, 150),
        deadline_seed: int = 0,
        deadline_curriculum: Optional[list[tuple[int, float]]] = None,
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
        self.edge_profile = edge_profile
        self.max_episode_frames = max_episode_frames
        self.episode_min_frames = max(1, int(episode_min_frames))
        # Time-varying latency deadline: piecewise constant, redrawn from the levels after a
        # random hold. Its own RNG so every policy evaluated on a video sees the same schedule.
        self.deadline_levels_ms = [float(d) for d in deadline_levels_ms or []]
        self.deadline_hold_frames = (int(deadline_hold_frames[0]), int(deadline_hold_frames[1]))
        self._deadline_rng = np.random.default_rng(deadline_seed)
        self._deadline_ms: Optional[float] = None
        self._deadline_left = 0
        # Optional training curriculum: (step, lowest level allowed from that step on), so
        # tight deadlines join only after the agent has seen loose ones.
        self.deadline_curriculum = sorted((int(s), float(d)) for s, d in deadline_curriculum or [])
        self._total_steps = 0

        self.action_space = spaces.Discrete(self.config_space.n_actions)
        self.action_latencies_ms = (
            np.array(
                [
                    edge_profile.latency_ms(self.config_space.action_to_config(a))
                    for a in range(self.config_space.n_actions)
                ]
            )
            if edge_profile is not None
            else None
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.features.dim,), dtype=np.float32
        )

        self._frames: Optional[Iterator[np.ndarray]] = None
        self._frame_idx = 0
        self._current_action = 0
        self._prev_action: Optional[int] = None
        self._scene: Optional[SceneFeatures] = None

    def _advance_deadline(self) -> None:
        if not self.deadline_levels_ms:
            return
        if self._deadline_left <= 0:
            lowest = 0.0
            for step, level in self.deadline_curriculum:
                if self._total_steps >= step:
                    lowest = level
            levels = [d for d in self.deadline_levels_ms if d >= lowest] or self.deadline_levels_ms
            self._deadline_ms = float(self._deadline_rng.choice(levels))
            lo, hi = self.deadline_hold_frames
            self._deadline_left = int(self._deadline_rng.integers(lo, hi + 1))
        self._deadline_left -= 1

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if hasattr(self.frame_source, "random_episode"):
            self.frame_source.random_episode(
                min_frames=self.episode_min_frames,
                max_frames=self.max_episode_frames or self.frame_source.frame_count,
                step=self.frame_step,
            )
        frames = self.frame_source.iter_frames(step=self.frame_step)
        if self.max_episode_frames is not None:
            frames = islice(frames, int(self.max_episode_frames))
        self._frames = frames
        self._frame_idx = 0
        self._current_action = 0
        self._prev_action = None
        self.telemetry.reset()
        self.analyzer.reset()
        if self.edge_profile is not None:
            self.edge_profile.reset()

        frame = next(self._frames, None)
        if frame is None:
            raise RuntimeError("video has no frames")
        self._frame_idx = 1
        self._deadline_left = 0
        detections = self._infer(frame, 0)
        self._scene = self.analyzer.analyze(frame, detections)
        self._advance_deadline()
        obs = self.features.build(
            self._scene, self.telemetry, self._current_action, self._deadline_ms
        )
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
                self._deadline_ms,
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
        self._total_steps += 1

        gt_boxes = ignored = None
        if hasattr(self.frame_source, "gt_for_current_frame"):
            gt_boxes = self.frame_source.gt_for_current_frame()
        if hasattr(self.frame_source, "ignored_for_current_sequence"):
            ignored = self.frame_source.ignored_for_current_sequence()
        # the deadline this action was chosen against (shown in the previous observation)
        deadline_ms = self._deadline_ms
        reward = self.reward.compute(
            detections,
            scene,
            self.telemetry,
            self._prev_action,
            action,
            gt_boxes=gt_boxes,
            deadline_ms=deadline_ms,
            ignored_regions=ignored,
        )
        self._prev_action = action
        self._current_action = action

        self._advance_deadline()
        obs = self.features.build(scene, self.telemetry, action, self._deadline_ms)
        info = self._make_info(detections, obs=obs, reward=reward)
        info["quality"] = float(self.reward.last_quality)
        info["deadline_ms"] = deadline_ms
        info["deadline_miss"] = int(
            deadline_ms is not None and self.telemetry.last_latency_ms() > deadline_ms
        )
        return obs, float(reward), False, False, info

    def action_masks(self) -> np.ndarray:
        """For MaskablePPO: only configurations that fit the deadline of the next frame."""
        if self._deadline_ms is None or self.action_latencies_ms is None:
            return np.ones(self.action_space.n, dtype=bool)
        return deadline_action_mask(self.action_latencies_ms, self._deadline_ms)

    def _infer(self, frame: np.ndarray, action: int) -> Detections:
        config = self.config_space.action_to_config(action)
        detections = self.detector.detect(frame, config)
        snap = self.edge_profile.snapshot(config) if self.edge_profile is not None else None
        if self.latency_estimator is not None and self.telemetry.latencies_ms:
            # The edge profile is the device latency model when present: it carries measured
            # board values and thermal throttling, which the nominal estimator alone does not.
            latency = (
                snap.latency_ms
                if snap is not None
                else self.latency_estimator.latency_ms(
                    config.model, config.resolution, config.precision
                )
            )
            if latency > 0:
                self.telemetry.latencies_ms[-1] = latency
        if snap is not None:
            self.telemetry.record_gpu(
                GpuSnapshot(
                    npu_util_pct=snap.npu_util_pct,
                    mem_used_mb=snap.mem_used_mb,
                    mem_budget_mb=self.edge_profile.memory_budget_mb,
                    temp_c=snap.temp_c,
                    power_w=snap.power_w,
                    latency_ms=snap.latency_ms,
                )
            )
        return detections

    def _make_info(self, detections: Detections, obs, reward: float) -> dict:
        scene = self._scene
        config = self.config_space.action_to_config(self._current_action)
        gpu = self.telemetry.gpu
        constraints = getattr(self.reward.cfg, "constraints", None) or {}
        violations = (
            compute_constraint_violations(gpu.as_dict(), constraints)
            if gpu is not None and constraints
            else {}
        )
        return {
            "frame": self._frame_idx,
            "detections": len(detections),
            "obj_count": scene.obj_count if scene else 0,
            "mean_conf": scene.mean_conf if scene else 0.0,
            "latency_ms": self.telemetry.last_latency_ms(),
            "fps": self.telemetry.fps(),
            "action": self._current_action,
            "model": config.model,
            "resolution": config.resolution,
            "precision": config.precision,
            "npu_util_pct": gpu.npu_util_pct if gpu is not None else 0.0,
            "mem_used_mb": gpu.mem_used_mb if gpu is not None else 0.0,
            "power_w": gpu.power_w if gpu is not None else 0.0,
            "temp_c": gpu.temp_c if gpu is not None else 0.0,
            "constraint_violations": violations,
            "constraint_violation_count": len(violations),
            "video": (
                self.frame_source.current_path.name
                if hasattr(self.frame_source, "current_path")
                else "single"
            ),
            "reward": float(reward),
            "obs": obs,
            "next_deadline_ms": self._deadline_ms,
        }
