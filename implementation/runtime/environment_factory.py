from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import yaml

from ..model_management.runtime_config_space import ConfigSpace
from ..model_management.model_variants import build_variants
from ..runtime.latency_estimator import LatencyEstimator
from ..runtime.system_telemetry import Telemetry
from ..vision_pipeline.video_frame_source import VideoFrameSource
from ..vision_pipeline.yolo_detector import Detector
from ..vision_pipeline.scene_analyzer import SceneAnalyzer
from ..reinforcement_learning.vision_runtime_env import VisionRuntimeEnv
from ..reinforcement_learning.observation_features import FeatureBuilder
from ..reinforcement_learning.reward_calculator import RewardConfig, build_reward


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_env_from_configs(env_cfg: dict, variants_cfg: dict, reward_cfg: dict) -> VisionRuntimeEnv:
    variants = build_variants(variants_cfg)
    weights_dir = variants_cfg.get("weights_dir", "weights")
    weight_paths = {
        name: str(Path(weights_dir) / variant.weights) for name, variant in variants.items()
    }
    config_space = ConfigSpace(
        models=list(variants.keys()),
        resolutions=variants_cfg.get("resolutions", [640]),
        precisions=variants_cfg.get("precisions", ["fp32"]),
    )
    telemetry = Telemetry()
    detector = Detector(
        weight_paths=weight_paths,
        conf=float(env_cfg.get("conf_threshold", 0.25)),
        device=str(env_cfg.get("device", "cuda")),
        telemetry=telemetry,
    )
    analyzer = SceneAnalyzer()
    features = FeatureBuilder(n_models=len(variants))
    reward_fields = {f.name for f in fields(RewardConfig)}
    reward_cfg = RewardConfig(**{k: v for k, v in reward_cfg.items() if k in reward_fields})
    gflops_by_model = {name: variant.gflops for name, variant in variants.items()}
    max_gflops = max(gflops_by_model.values()) or 1.0
    reward_cfg.compute_cost = tuple(
        gflops_by_model.get(config_space.action_to_config(i).model, 1.0) / max_gflops
        for i in range(config_space.n_actions)
    )
    reward = build_reward(reward_cfg)
    estimator = None
    if reward_cfg.latency_source != "live":
        estimator = LatencyEstimator.from_variants(
            variants, gflops_by_model, jitter_ms=reward_cfg.latency_jitter_ms
        )
    source = VideoFrameSource(str(env_cfg["video_path"]))
    frame_step = int(env_cfg.get("frame_step", 1))
    return VisionRuntimeEnv(
        frame_source=source,
        config_space=config_space,
        detector=detector,
        analyzer=analyzer,
        features=features,
        reward=reward,
        telemetry=telemetry,
        frame_step=frame_step,
        latency_estimator=estimator,
    )
