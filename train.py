"""Convenience launcher for offline IVR training.

Run from anywhere (no module form needed):

    python train.py --dataset-dir "..\\BDDA\\BDDA"
    python train.py --timesteps 5000
    python train.py --algorithm ppo --dataset-split validation

See docs/RUN_BDD.md for the full workflow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from implementation.runtime.environment_factory import load_yaml  # noqa: E402
from implementation.reinforcement_learning.train_runtime import train  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train the IVR RL agent (offline, no network)")
    parser.add_argument("--env", default="configs/env_bdd.yaml")
    parser.add_argument("--variants", default="configs/variants.yaml")
    parser.add_argument("--reward", default="configs/reward_bdd.yaml")
    parser.add_argument("--training", default="configs/training_bdd.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--algorithm", default=None, choices=["dqn", "ppo"])
    parser.add_argument("--log-every", type=int, default=2000)
    parser.add_argument("--verbose-episodes", action="store_true")
    parser.add_argument(
        "--dataset-dir",
        default=None,
help="override dataset root (absolute or relative to repo root), e.g. "
        "..\\BDDA\\BDDA",
    )
    parser.add_argument("--dataset-split", default=None, choices=["training", "validation", "test"])
    args = parser.parse_args()

    overrides: dict = {}
    if args.timesteps is not None:
        overrides["timesteps"] = args.timesteps
    if args.algorithm is not None:
        overrides["algorithm"] = args.algorithm

    env_cfg = load_yaml(args.env)
    if args.dataset_dir:
        env_cfg["dataset_dir"] = args.dataset_dir
    if args.dataset_split:
        env_cfg["dataset_split"] = args.dataset_split

    train(
        args.env,
        args.variants,
        args.reward,
        args.training,
        overrides=overrides or None,
        env_cfg_override=env_cfg,
        log_every=args.log_every,
        verbose_episodes=args.verbose_episodes,
    )


if __name__ == "__main__":
    main()
