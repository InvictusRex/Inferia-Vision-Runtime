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
    parser.add_argument("--hardware", default="configs/hardware.yaml", help="hardware profile yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--algorithm", default=None, choices=["dqn", "ppo"])
    parser.add_argument(
        "--log-name", default=None, help="override the training config's log_name/checkpoint stem"
    )
    parser.add_argument("--log-every", type=int, default=2000)
    parser.add_argument("--verbose-episodes", action="store_true")
    parser.add_argument("--w-switch", type=float, default=None, help="override reward w_switch")
    parser.add_argument(
        "--constraints",
        choices=["on", "off"],
        default=None,
        help="toggle constraint penalties in the reward (default: yaml value)",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
help="override dataset root (absolute or relative to repo root), e.g. "
"..\\BDDA\\BDDA",
    )
    parser.add_argument("--dataset-split", default=None, choices=["training", "validation", "test"])
    parser.add_argument(
        "--checkpoint-every", type=int, default=None,
        help="save a checkpoint every N steps (e.g. 50000)",
    )
    parser.add_argument(
        "--resume", default=None,
        help="resume training from this checkpoint zip",
    )
    args = parser.parse_args()

    overrides: dict = {}
    if args.timesteps is not None:
        overrides["timesteps"] = args.timesteps
    if args.algorithm is not None:
        overrides["algorithm"] = args.algorithm
    if args.log_name is not None:
        overrides["log_name"] = args.log_name

    env_cfg = load_yaml(args.env)
    if args.dataset_dir:
        env_cfg["dataset_dir"] = args.dataset_dir
    if args.dataset_split:
        env_cfg["dataset_split"] = args.dataset_split

    reward_cfg = load_yaml(args.reward)
    if args.w_switch is not None:
        reward_cfg["w_switch"] = args.w_switch
    if args.constraints is not None:
        if args.constraints == "off":
            reward_cfg["constraints"] = {}
        else:
            base = load_yaml("configs/reward_constrained_bdd.yaml")
            reward_cfg["constraints"] = base.get("constraints", {})
            reward_cfg["constraint_weights"] = base.get("constraint_weights", {})

    train(
        args.env,
        args.variants,
        args.reward,
        args.training,
        overrides=overrides or None,
        env_cfg_override=env_cfg,
        reward_cfg_override=reward_cfg,
        log_every=args.log_every,
        verbose_episodes=args.verbose_episodes,
        hardware_cfg_path=args.hardware,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
