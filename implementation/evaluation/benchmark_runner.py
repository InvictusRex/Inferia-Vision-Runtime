from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import numpy as np

from ..reinforcement_learning.vision_runtime_env import VisionRuntimeEnv
from .baseline_schedulers import Policy, default_baselines

DENSITY_BINS = [(0.0, 8.0, "sparse"), (8.0, 16.0, "mid"), (16.0, float("inf"), "dense")]


def _progress_bar(done: int, total: int, width: int = 18) -> str:
    frac = done / total if total else 0.0
    filled = int(width * frac)
    return f"[{'#' * filled}{'.' * (width - filled)}] {done:>5}/{total} ({frac * 100:3.0f}%)"


class EvalProgress:
    """Live console progress for dataset evaluation (mirrors train.py's bar)."""

    def __init__(self, log_every: int = 1):
        self.log_every = log_every
        self.start = 0.0

    def begin(self, total: int, label: str) -> None:
        import time as _t

        self.start = _t.time()
        self.total = total
        self.label = label

    def update(self, done: int, policy_name: str) -> None:
        import time as _t

        elapsed = _t.time() - self.start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else 0.0
        if done == self.total or done % self.log_every == 0:
            prog = _progress_bar(done, self.total)
            print(
                f"{prog} | {self.label} vids={done}/{self.total} ({rate:.2f} vids/s, "
                f"ETA {eta:5.0f}s) | policy={policy_name}",
                flush=True,
            )


def _bin_label(count: float) -> str:
    for lo, hi, label in DENSITY_BINS:
        if lo <= count < hi:
            return label
    return "dense"


def run_episode(env: VisionRuntimeEnv, policy: Policy, max_steps: int | None = None) -> dict:
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    switches = 0
    prev_action = info["action"]
    latencies: list[float] = []
    confs: list[float] = []
    counts: list[int] = []
    rewards: list[float] = []
    fps_samples: list[float] = []
    bin_rewards: dict[str, list[float]] = {"sparse": [], "mid": [], "dense": []}

    done = False
    while not done:
        if max_steps is not None and steps >= max_steps:
            break
        action = policy.choose(obs, info)
        if action != prev_action:
            switches += 1
        obs, reward, terminated, truncated, info = env.step(action)
        policy.update(obs, info, reward, action)
        total_reward += reward
        steps += 1
        latencies.append(float(info["latency_ms"]))
        confs.append(float(info["mean_conf"]))
        counts.append(int(info["obj_count"]))
        rewards.append(float(reward))
        fps_samples.append(float(info["fps"]))
        bin_rewards[_bin_label(float(info["obj_count"]))].append(float(reward))
        prev_action = action
        done = terminated or truncated

    return {
        "policy": policy.name,
        "steps": steps,
        "total_reward": float(total_reward),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_conf": float(np.mean(confs)) if confs else 0.0,
        "mean_count": float(np.mean(counts)) if counts else 0.0,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "mean_fps": float(np.mean(fps_samples)) if fps_samples else 0.0,
        "switches": switches,
        "switch_rate": switches / steps if steps else 0.0,
        "bin_rewards": {k: (float(np.mean(v)) if v else 0.0) for k, v in bin_rewards.items()},
        "bin_frame_counts": {k: len(v) for k, v in bin_rewards.items()},
    }


def compare_policies(
    env_builder: Callable[[], VisionRuntimeEnv],
    policies: list[Policy],
    episodes: int = 1,
    max_steps: int | None = None,
    output_dir: str = "output",
) -> dict[str, list[dict]]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict]] = {}
    for policy in policies:
        per_episode = []
        for _ in range(episodes):
            env = env_builder()
            per_episode.append(run_episode(env, policy, max_steps=max_steps))
        results[policy.name] = per_episode
        row = per_episode[0]
        print(
            f"{row['policy']:<30} steps={row['steps']:5d} reward={row['total_reward']:8.1f} "
            f"conf={row['mean_conf']:.3f} count={row['mean_count']:5.1f} "
            f"lat={row['mean_latency_ms']:6.1f}ms fps={row['mean_fps']:6.1f} "
            f"switches={row['switches']:4d}"
        )

    path = out_dir / "benchmark.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "policy",
                "steps",
                "total_reward",
                "mean_reward",
                "mean_conf",
                "mean_count",
                "mean_latency_ms",
                "mean_fps",
                "switches",
                "switch_rate",
            ],
        )
        writer.writeheader()
        for per_episode in results.values():
            writer.writerows(per_episode)
    print(f"benchmark written to {path}")
    return results


def compare_policies_dataset(
    env_builder: Callable[[str], VisionRuntimeEnv],
    video_paths: list[str],
    policies: list[Policy],
    output_dir: str = "output",
    output_filename: str = "benchmark_dataset.csv",
    progress_every: int | None = None,
) -> dict[str, list[dict]]:
    """Run every policy over a list of videos; aggregate mean +/- std + density bins.

    progress_every: print a live progress line every N videos per policy (train-style
    bar). None disables live progress (still prints the per-policy summary rows).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = EvalProgress(progress_every) if progress_every else None
    results: dict[str, list[dict]] = {}
    n_videos = len(video_paths)
    for p_idx, policy in enumerate(policies):
        if progress:
            progress.begin(n_videos, f"{policy.name}")
        per_video: list[dict] = []
        for v_idx, path in enumerate(video_paths):
            env = env_builder(path)
            row = run_episode(env, policy)
            row["video"] = Path(path).name
            per_video.append(row)
            if progress:
                progress.update(v_idx + 1, policy.name)
        results[policy.name] = per_video

        rewards = np.array([r["total_reward"] for r in per_video])
        confs = np.array([r["mean_conf"] for r in per_video])
        counts = np.array([r["mean_count"] for r in per_video])
        lats = np.array([r["mean_latency_ms"] for r in per_video])
        sw = np.array([r["switches"] for r in per_video])
        bin_rows = []
        for b in ("sparse", "mid", "dense"):
            vals = []
            for r in per_video:
                vals.extend([r["bin_rewards"].get(b, 0.0)] * r["bin_frame_counts"].get(b, 0))
            if vals:
                bin_rows.append(f"{b}={np.mean(vals):+.2f}")
        print(
            f"{policy.name:<30} videos={len(per_video)} "
            f"reward={rewards.mean():8.1f} +/- {rewards.std():6.1f} "
            f"conf={confs.mean():.3f} count={counts.mean():5.1f} "
            f"lat={lats.mean():6.1f}ms "
            f"fps={np.array([r['mean_fps'] for r in per_video]).mean():6.1f} "
            f"switches={sw.mean():6.1f} | " + " ".join(bin_rows)
        )

    csv_path = out_dir / output_filename
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "video",
                "policy",
                "steps",
                "total_reward",
                "mean_reward",
                "mean_conf",
                "mean_count",
                "mean_latency_ms",
                "mean_fps",
                "switches",
                "switch_rate",
                "bin_sparse_reward",
                "bin_mid_reward",
                "bin_dense_reward",
            ],
        )
        writer.writeheader()
        for per_video in results.values():
            for r in per_video:
                writer.writerow(
                    {
                        **{
                            k: v
                            for k, v in r.items()
                            if k not in ("bin_rewards", "bin_frame_counts")
                        },
                        "bin_sparse_reward": r["bin_rewards"]["sparse"],
                        "bin_mid_reward": r["bin_rewards"]["mid"],
                        "bin_dense_reward": r["bin_rewards"]["dense"],
                    }
                )
    print(f"dataset benchmark written to {csv_path}")
    return results


def _load_configs(env_path: str, variants_path: str, reward_path: str):
    from ..runtime.environment_factory import build_env_from_configs, load_yaml

    env_cfg = load_yaml(env_path)
    variants_cfg = load_yaml(variants_path)
    reward_cfg = load_yaml(reward_path)
    return (
        env_cfg,
        variants_cfg,
        reward_cfg,
        build_env_from_configs(env_cfg, variants_cfg, reward_cfg),
    )


def _split_videos(dataset_dir: str, split: str) -> list[str]:
    base = Path(dataset_dir) / split / "camera_videos"
    paths = sorted(p for p in base.glob("*.mp4") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"no camera videos under {base}")
    return [str(p) for p in paths]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IVR baseline comparison (single or dataset)")
    parser.add_argument("--env", default="configs/env_bdd.yaml")
    parser.add_argument("--variants", default="configs/variants.yaml")
    parser.add_argument("--reward", default="configs/reward_bdd.yaml")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dataset", choices=["train", "validation", "test"], default=None)
    parser.add_argument("--n-videos", type=int, default=None, help="limit videos (default: all)")
    parser.add_argument("--model", default=None, help="path to a saved SB3 DQN/PPO zip")
    parser.add_argument("--model-name", default=None, help="label for the loaded model")
    parser.add_argument("--policies", default=None, help="comma-separated baseline names override")
    args = parser.parse_args()

    env_cfg, variants_cfg, reward_cfg, env = _load_configs(args.env, args.variants, args.reward)

    def builder_for(path: str = None):
        from ..runtime.environment_factory import build_env_from_configs

        return build_env_from_configs(env_cfg, variants_cfg, reward_cfg, video_override=path)

    policies = default_baselines(env)
    if args.policies:
        wanted = {p.strip() for p in args.policies.split(",")}
        policies = [p for p in policies if p.name in wanted]
    if args.model:
        from stable_baselines3 import DQN, PPO
        from .baseline_schedulers import SB3Policy

        try:
            model = PPO.load(args.model)
        except Exception:
            model = DQN.load(args.model)
        name = args.model_name or f"ivr_{Path(args.model).stem}"
        policies.append(SB3Policy(model, name=name))

    if args.dataset:
        videos = _split_videos(env_cfg.get("dataset_dir", "../BDDA/BDDA"), args.dataset)
        if args.n_videos:
            videos = videos[: args.n_videos]
        compare_policies_dataset(builder_for, videos, policies)
    else:
        compare_policies(
            lambda: builder_for(None),
            policies,
            episodes=args.episodes,
            max_steps=args.max_steps,
        )
