"""Closed-loop IVR on the ROCK 5C: a trained policy picks the runtime config each frame
and the NPU runs it.

Same loop as VisionRuntimeEnv -- the observation of frame t chooses the config used on
frame t+1 -- built from the repository's own scene analyzer, observation builder,
config space and edge profile, but with the RKNN detectors on the NPU in place of the
PyTorch detector and with real measured latency feeding the observation. Writes a
per-frame trace and an annotated video.

Run on the board, next to rknn_benchmark.py, with a copy of the repository:

    python rknn_live_demo.py --repo ~/ivr/repo --bundle ~/ivr/bundle \\
        --policy ~/ivr/policies/dqn_detrac_board_s0_final.zip --sequence MVI_40714
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

# torch (via SB3) must be imported before RKNN-Toolkit-Lite2, which rewrites the logging
# level table that torch reads at import time
from stable_baselines3 import DQN, PPO  # isort: skip
from rknn_benchmark import NPU_TEMP, infer, letterbox, open_runtime, postprocess, read_temp


@dataclass
class BoardDetections:
    """The slice of vision_pipeline.Detections the scene analyzer reads."""

    boxes: np.ndarray
    confs: np.ndarray
    cls_ids: np.ndarray

    def __len__(self) -> int:
        return int(self.confs.shape[0])


def import_repo(repo: Path) -> None:
    sys.path.insert(0, str(repo))
    # register these as bare packages: their __init__ pulls in the PyTorch detector and
    # the gymnasium env, neither of which exists on the board
    for pkg in ("vision_pipeline", "reinforcement_learning"):
        module = types.ModuleType(pkg)
        module.__path__ = [str(repo / pkg)]
        sys.modules[pkg] = module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--policy", required=True, help="SB3 DQN or PPO checkpoint zip")
    parser.add_argument("--sequence", required=True, help="folder under <bundle>/demo")
    parser.add_argument("--hardware", default="configs/hardware_measured.yaml")
    parser.add_argument("--variants", default="configs/variants.yaml")
    parser.add_argument("--cores", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    repo, bundle = Path(args.repo).expanduser(), Path(args.bundle).expanduser()
    import_repo(repo)
    from model_management.model_variants import build_variants
    from model_management.runtime_config_space import ConfigSpace
    from reinforcement_learning.observation_features import FeatureBuilder
    from runtime.edge_profile import EdgeProfile
    from runtime.latency_estimator import LatencyEstimator
    from runtime.system_telemetry import GpuSnapshot, Telemetry
    from vision_pipeline.scene_analyzer import SceneAnalyzer

    variants_cfg = yaml.safe_load((repo / args.variants).read_text())
    hw_cfg = yaml.safe_load((repo / args.hardware).read_text())
    precision_map = hw_cfg["measured"]["precision_map"]

    variants = build_variants(variants_cfg)
    space = ConfigSpace(
        models=list(variants),
        resolutions=variants_cfg["resolutions"],
        precisions=variants_cfg["precisions"],
    )
    gflops = {name: v.gflops for name, v in variants.items()}
    edge = EdgeProfile.from_configs(
        hw_cfg, variants, LatencyEstimator.from_variants(variants, gflops)
    )
    sim = edge.profile.simulated
    features = FeatureBuilder(
        n_actions=space.n_actions,
        power_budget_w=edge.profile.policy.power_budget_w,
        ambient_temp_c=sim.ambient_temp_c,
        throttle_temp_c=sim.thermal_throttle_temp_c,
    )
    analyzer, telemetry = SceneAnalyzer(), Telemetry()
    policy_path = Path(args.policy).expanduser()
    policy = (PPO if "ppo" in policy_path.stem.lower() else DQN).load(
        policy_path,
        device="cpu",
        # training-only schedules are pickled with the host's Python; inference never uses them
        custom_objects={k: 0.0 for k in ("lr_schedule", "exploration_schedule", "clip_range")},
    )

    runtimes: dict[tuple, object] = {}
    load_ms: dict[str, float] = {}

    def runtime_for(config):
        key = (config.model, config.resolution, precision_map[config.precision])
        if key not in runtimes:
            t0 = time.perf_counter()
            runtimes[key] = open_runtime(
                bundle / "rknn" / f"{key[0]}_{key[1]}_{key[2]}.rknn", args.cores
            )
            load_ms["_".join(map(str, key))] = (time.perf_counter() - t0) * 1000.0
        return runtimes[key]

    frames = sorted((bundle / "demo" / args.sequence).glob("img*.jpg"))[: args.max_frames]
    out_dir = bundle / "demo_out"
    out_dir.mkdir(exist_ok=True)
    stem = f"{policy_path.stem}_{args.sequence}"
    video = None
    if not args.no_video:
        h, w = cv2.imread(str(frames[0])).shape[:2]
        video = cv2.VideoWriter(
            str(out_dir / f"{stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h)
        )

    rows = []
    action = 0  # VisionRuntimeEnv also runs its first frame on action 0
    for path in frames:
        frame = cv2.imread(str(path))
        config = space.action_to_config(action)
        lite = runtime_for(config)

        t0 = time.perf_counter()
        inp, scale, left, top = letterbox(frame, config.resolution)
        dets = postprocess(infer(lite, inp), scale, left, top)
        t1 = time.perf_counter()
        detections = BoardDetections(dets[:, :4], dets[:, 4], dets[:, 5].astype(int))
        latency_ms = (t1 - t0) * 1000.0
        telemetry.record(latency_ms)
        snap = edge.snapshot(config)
        telemetry.record_gpu(
            GpuSnapshot(
                npu_util_pct=snap.npu_util_pct,
                mem_used_mb=snap.mem_used_mb,
                mem_budget_mb=edge.memory_budget_mb,
                temp_c=snap.temp_c,
                power_w=snap.power_w,
                latency_ms=latency_ms,
            )
        )
        scene = analyzer.analyze(frame, detections)
        obs = features.build(scene, telemetry, action)
        next_action = int(policy.predict(obs, deterministic=True)[0])
        t2 = time.perf_counter()

        rows.append(
            {
                "frame": int(path.stem[3:]),
                "model": config.model,
                "resolution": config.resolution,
                "precision": config.precision,
                "board_precision": precision_map[config.precision],
                "detections": len(detections),
                "latency_ms": round(latency_ms, 2),
                "loop_ms": round((t2 - t0) * 1000.0, 2),
                "npu_temp_c": read_temp(NPU_TEMP),
                "switched": int(next_action != action),
            }
        )
        if video is not None:
            for x1, y1, x2, y2 in dets[:, :4].astype(int):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = (
                f"{config.model} {config.resolution}px {precision_map[config.precision]}  "
                f"{latency_ms:5.1f} ms  {len(detections)} vehicles"
            )
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            video.write(frame)
        action = next_action

    if video is not None:
        video.release()
    for lite in runtimes.values():
        lite.release()

    with open(out_dir / f"{stem}.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    loop = np.array([r["loop_ms"] for r in rows])
    configs = {}
    for r in rows:
        key = f"{r['model']}/{r['resolution']}/{r['board_precision']}"
        configs[key] = configs.get(key, 0) + 1
    print(
        f"frames={len(rows)} switches={sum(r['switched'] for r in rows)} "
        f"loop_ms_p50={np.median(loop):.1f} closed_loop_fps={1000.0 / np.median(loop):.1f}"
    )
    print("configs used: " + ", ".join(f"{k}={v}" for k, v in sorted(configs.items())))
    print("model load ms: " + ", ".join(f"{k}={v:.0f}" for k, v in load_ms.items()))
    print(f"written {out_dir / stem}.csv" + ("" if video is None else f" and {stem}.mp4"))


if __name__ == "__main__":
    main()
