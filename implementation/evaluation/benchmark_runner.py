from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import numpy as np

from ..reinforcement_learning.vision_runtime_env import VisionRuntimeEnv
from .baseline_schedulers import Policy, build_policy


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
            f"{row['policy']:<22} steps={row['steps']:5d} reward={row['total_reward']:8.1f} "
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IVR baselines comparison")
    parser.add_argument("--env", default="configs/env.yaml")
    parser.add_argument("--variants", default="configs/variants.yaml")
    parser.add_argument("--reward", default="configs/reward.yaml")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--model", default=None, help="path to a saved SB3 DQN zip")
    args = parser.parse_args()

    from ..runtime.environment_factory import build_env_from_configs, load_yaml

    env_cfg = load_yaml(args.env)
    variants_cfg = load_yaml(args.variants)
    reward_cfg = load_yaml(args.reward)

    def builder():
        return build_env_from_configs(env_cfg, variants_cfg, reward_cfg)

    env = builder()
    policies = [
        build_policy("always_0", env),
        build_policy("always_1", env),
        build_policy("always_2", env),
        build_policy("random", env),
        build_policy("rule_based", env),
        build_policy("contextual_bandit", env),
    ]
    if args.model:
        from stable_baselines3 import DQN
        from .baseline_schedulers import SB3Policy

        policies.append(SB3Policy(DQN.load(args.model), name="ivr_dqn"))
    compare_policies(
        builder,
        policies,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )
