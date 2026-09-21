from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import DQN

from runtime.environment_factory import build_env_from_configs, load_yaml


def train(
    env_cfg_path: str,
    variants_cfg_path: str,
    reward_cfg_path: str,
    training_cfg_path: str,
    overrides: dict | None = None,
) -> DQN:
    env_cfg = load_yaml(env_cfg_path)
    variants_cfg = load_yaml(variants_cfg_path)
    reward_cfg = load_yaml(reward_cfg_path)
    training_cfg = load_yaml(training_cfg_path)
    if overrides:
        training_cfg.update(overrides)

    env = build_env_from_configs(env_cfg, variants_cfg, reward_cfg)
    save_dir = Path(training_cfg.get("save_dir", "training"))
    tb_dir = Path(training_cfg.get("tb_dir", "runs/tensorboard"))
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    model = DQN(
        policy=training_cfg.get("policy", "MlpPolicy"),
        env=env,
        learning_rate=float(training_cfg.get("learning_rate", 0.0001)),
        buffer_size=int(training_cfg.get("buffer_size", 100000)),
        batch_size=int(training_cfg.get("batch_size", 64)),
        gamma=float(training_cfg.get("gamma", 0.99)),
        train_freq=int(training_cfg.get("train_freq", 4)),
        gradient_steps=int(training_cfg.get("gradient_steps", 1)),
        target_update_interval=int(training_cfg.get("target_update_interval", 1000)),
        exploration_fraction=float(training_cfg.get("exploration_fraction", 0.2)),
        exploration_final_eps=float(training_cfg.get("exploration_final_eps", 0.05)),
        tensorboard_log=str(tb_dir),
        seed=int(training_cfg.get("seed", 0)),
        verbose=1,
    )

    timesteps = int(training_cfg.get("timesteps", 20000))
    model.learn(total_timesteps=timesteps, tb_log_name=training_cfg.get("log_name", "dqn"))

    log_name = training_cfg.get("log_name", "dqn")
    save_path = save_dir / f"{log_name}_final.zip"
    model.save(str(save_path))
    print(f"model saved to {save_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the IVR RL agent")
    parser.add_argument("--env", default="configs/env/env.yaml")
    parser.add_argument("--variants", default="configs/variants.yaml")
    parser.add_argument("--reward", default="configs/reward/reward.yaml")
    parser.add_argument("--training", default="configs/training/training.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()
    overrides = {"timesteps": args.timesteps} if args.timesteps is not None else None
    train(args.env, args.variants, args.reward, args.training, overrides=overrides)
