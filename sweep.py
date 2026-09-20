"""Offline launcher for the PPO run + switch-penalty sweep.

Run from anywhere (no module form needed):

    python sweep.py                                # PPO 300k, then w_switch sweep on PPO
    python sweep.py --timesteps 20000              # shorter sanity pass
    python sweep.py --skip-train                   # skip the main PPO run, sweep only
    python sweep.py --sweep-algorithm dqn          # sweep the DQN instead of PPO
    python sweep.py --sweep-values 0.1,0.3,0.5     # custom switch penalties

Everything runs offline (weights + BDD-A local). Each training job prints the same
live progress bar as train.py (steps / ETA / episodes / ep_rew_mean / eps / loss)
and saves to training/{log_name}_final.zip.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from runtime.environment_factory import load_yaml  # noqa: E402
from reinforcement_learning.train_runtime import train  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train PPO on BDD-A + sweep the switch penalty")
    parser.add_argument("--env", default="configs/env_bdd.yaml")
    parser.add_argument("--variants", default="configs/variants.yaml")
    parser.add_argument("--reward", default="configs/reward_bdd.yaml")
    parser.add_argument("--training", default="configs/training_ppo_bdd.yaml")
    parser.add_argument("--hardware", default="configs/hardware.yaml")
    parser.add_argument("--timesteps", type=int, default=None, help="override main run steps")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="skip the main PPO training run, only do the switch sweep",
    )
    parser.add_argument("--sweep-algorithm", default="ppo", choices=["dqn", "ppo"])
    parser.add_argument("--sweep-values", default="0.1,0.5", help="comma-separated w_switch values")
    parser.add_argument(
        "--sweep-timesteps", type=int, default=60000, help="steps per sweep training run"
    )
    parser.add_argument("--log-every", type=int, default=2000)
    parser.add_argument("--verbose-episodes", action="store_true")
    parser.add_argument(
        "--constraints",
        choices=["on", "off"],
        default=None,
        help="toggle constraint penalties in the reward (default: yaml value)",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="override dataset root (absolute or relative to repo root)",
    )
    parser.add_argument("--dataset-split", default=None, choices=["training", "validation", "test"])
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50000,
        help="save a checkpoint every N steps (default 50000)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="resume the main run from this checkpoint zip",
    )
    args = parser.parse_args()

    env_cfg = load_yaml(args.env)
    if args.dataset_dir:
        env_cfg["dataset_dir"] = args.dataset_dir
    if args.dataset_split:
        env_cfg["dataset_split"] = args.dataset_split

    base_reward_cfg = load_yaml(args.reward)
    if args.constraints is not None:
        if args.constraints == "off":
            base_reward_cfg["constraints"] = {}
        else:
            base = load_yaml("configs/reward_constrained_bdd.yaml")
            base_reward_cfg["constraints"] = base.get("constraints", {})
            base_reward_cfg["constraint_weights"] = base.get("constraint_weights", {})

    # --- 1. Main run: PPO 300k (unless --skip-train) -------------------------
    if not args.skip_train:
        print(f"\n{'=' * 72}\nSTEP 1/2: main run ({args.training})\n{'=' * 72}")
        overrides: dict = {}
        if args.timesteps is not None:
            overrides["timesteps"] = args.timesteps
        train(
            args.env,
            args.variants,
            args.reward,
            args.training,
            overrides=overrides or None,
            env_cfg_override=env_cfg,
            reward_cfg_override=dict(base_reward_cfg),
            log_every=args.log_every,
            verbose_episodes=args.verbose_episodes,
            hardware_cfg_path=args.hardware,
            checkpoint_every=args.checkpoint_every,
            resume_from=args.resume,
        )
    else:
        print("--skip-train: skipping the main run.")

    # --- 2. Switch-penalty sweep on the chosen algorithm ----------------------
    sweep_values = [float(v) for v in args.sweep_values.split(",")]
    print(
        f"\n{'=' * 72}\nSTEP 2/2: w_switch sweep on {args.sweep_algorithm} "
        f"-> {sweep_values}\n{'=' * 72}"
    )

    base_training = load_yaml(args.training)
    base_training["algorithm"] = args.sweep_algorithm
    base_training["timesteps"] = args.sweep_timesteps
    log_root = base_training.get("log_name", args.sweep_algorithm)

    summary: list[dict] = []
    for w in sweep_values:
        log_name = f"{log_root}_ws{w}"
        training_cfg = dict(base_training)
        training_cfg["log_name"] = log_name
        reward_cfg = dict(base_reward_cfg)
        reward_cfg["w_switch"] = w

        print(f"\n--- sweep w_switch={w} -> training/{log_name}_final.zip ---")
        train(
            args.env,
            args.variants,
            args.reward,
            args.training,
            overrides=training_cfg,
            env_cfg_override=env_cfg,
            reward_cfg_override=reward_cfg,
            log_every=args.log_every,
            verbose_episodes=args.verbose_episodes,
            hardware_cfg_path=args.hardware,
            checkpoint_every=args.checkpoint_every,
        )
        summary.append(
            {
                "log_name": log_name,
                "w_switch": w,
                "checkpoint": f"training/{log_name}_final.zip",
            }
        )

    print(f"\n{'=' * 72}\nSWEEP DONE\n{'=' * 72}")
    for row in summary:
        print(f"  w_switch={row['w_switch']:>4} -> {row['checkpoint']}")
    print("\nEvaluate each checkpoint with:")
    for row in summary:
        print(f"  python eval.py --model {row['checkpoint']}")


if __name__ == "__main__":
    main()
