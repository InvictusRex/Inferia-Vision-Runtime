from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import yaml

from model_management.runtime_config_space import ConfigSpace
from model_management.model_variants import build_variants
from runtime.edge_profile import EdgeProfile
from runtime.latency_estimator import LatencyEstimator
from runtime.system_telemetry import Telemetry
from vision_pipeline.detrac_source import DetracFrameSource
from vision_pipeline.video_frame_source import DatasetVideoSource, VideoFrameSource
from vision_pipeline.yolo_detector import Detector
from vision_pipeline.scene_analyzer import SceneAnalyzer
from reinforcement_learning.vision_runtime_env import VisionRuntimeEnv
from reinforcement_learning.observation_features import FeatureBuilder
from reinforcement_learning.reward_calculator import RewardConfig, build_reward


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def detrac_xml_dir(dataset_dir: str, split: str) -> Path:
    folder = "DETRAC-Train-Annotations-XML" if split == "train" else "DETRAC-Test-Annotations-XML"
    return Path(dataset_dir) / folder / folder


def detrac_sequence_dirs(dataset_dir: str, split: str) -> list[str]:
    images_dir = Path(dataset_dir) / "DETRAC-Images" / "DETRAC-Images"
    xml_dir = detrac_xml_dir(dataset_dir, split)
    seq_dirs = sorted(
        str(p) for p in images_dir.iterdir() if p.is_dir() and (xml_dir / f"{p.name}.xml").exists()
    )
    if not seq_dirs:
        raise FileNotFoundError(f"no {split} sequences found under {images_dir}")
    return seq_dirs


def build_env_from_configs(
    env_cfg: dict,
    variants_cfg: dict,
    reward_cfg: dict,
    video_override: str | None = None,
    hardware_cfg: dict | None = None,
) -> VisionRuntimeEnv:
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
    reward_fields = {f.name for f in fields(RewardConfig)}
    reward_cfg = RewardConfig(**{k: v for k, v in reward_cfg.items() if k in reward_fields})
    gflops_by_model = {name: variant.gflops for name, variant in variants.items()}
    max_gflops = max(gflops_by_model.values()) or 1.0
    reward_cfg.compute_cost = tuple(
        (
            gflops_by_model.get(config_space.action_to_config(i).model, 1.0)
            * (config_space.action_to_config(i).resolution / 640.0) ** 2
        )
        / max_gflops
        for i in range(config_space.n_actions)
    )
    reward = build_reward(reward_cfg)
    estimator = None
    if reward_cfg.latency_source != "live":
        estimator = LatencyEstimator.from_variants(
            variants, gflops_by_model, jitter_ms=reward_cfg.latency_jitter_ms
        )

    edge_profile = None
    if hardware_cfg:
        edge_estimator = estimator or LatencyEstimator.from_variants(
            variants, gflops_by_model, jitter_ms=reward_cfg.latency_jitter_ms
        )
        edge_profile = EdgeProfile.from_configs(hardware_cfg, variants, edge_estimator)

    if edge_profile is not None:
        sim = edge_profile.profile.simulated
        features = FeatureBuilder(
            n_actions=config_space.n_actions,
            power_budget_w=edge_profile.profile.policy.power_budget_w,
            ambient_temp_c=sim.ambient_temp_c,
            throttle_temp_c=sim.thermal_throttle_temp_c,
        )
    else:
        features = FeatureBuilder(n_actions=config_space.n_actions)

    if env_cfg.get("dataset_type") == "detrac":
        dataset_dir = env_cfg["dataset_dir"]
        split = str(env_cfg.get("dataset_split", "train"))
        seq_dirs = (
            [str(video_override)] if video_override is not None
            else detrac_sequence_dirs(dataset_dir, split)
        )
        source = DetracFrameSource(
            seq_dirs, str(detrac_xml_dir(dataset_dir, split)), seed=int(env_cfg.get("seed", 0))
        )
    elif video_override is not None:
        source = VideoFrameSource(str(video_override))
    else:
        dataset_dir = env_cfg.get("dataset_dir")
        if dataset_dir:
            split_dir = Path(dataset_dir) / str(env_cfg.get("dataset_split", "training"))
            paths = sorted((split_dir / "camera_videos").glob("*.mp4"))
            if not paths:
                raise FileNotFoundError(
                    f"no camera videos found under {split_dir / 'camera_videos'}"
                )
            source = DatasetVideoSource([str(p) for p in paths], seed=int(env_cfg.get("seed", 0)))
        else:
            source = VideoFrameSource(str(env_cfg["video_path"]))

    frame_step = int(env_cfg.get("frame_step", 1))
    max_episode_frames = env_cfg.get("max_episode_frames")
    episode_min_frames = int(env_cfg.get("episode_min_frames", 150))
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
        max_episode_frames=max_episode_frames,
        episode_min_frames=episode_min_frames,
        edge_profile=edge_profile,
    )
