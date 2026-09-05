from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from ..runtime.environment_factory import build_env_from_configs, load_yaml


class ProgressCallback(BaseCallback):
    """Prints a training-progress line every `log_every` steps (YOLO-style loop view)."""

    def __init__(self, log_every: int = 2000, verbose_episodes: bool = False):
        super().__init__()
        self.log_every = max(1, int(log_every))
        self.verbose_episodes = bool(verbose_episodes)
        self._last_log = 0
        self._episodes = 0
        self._t0 = time.perf_counter()

    def _on_step(self):
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos and dones:
            self._episodes += int(sum(bool(d) for d in dones))
            if self.verbose_episodes:
                for inf, done in zip(infos, dones):
                    if done and "episode" in inf:
                        ep = inf["episode"]
                        print(
                            f"  ep end | video={inf.get('video', '?')} "
                            f"reward={ep['r']:+8.1f} len={int(ep['l']):4d}"
                        )
        if self.model.num_timesteps - self._last_log >= self.log_every:
            self._log()
            self._last_log = self.model.num_timesteps
        return True

    def _log(self):
        ts = int(self.model.num_timesteps)
        total = int(getattr(self.model, "_total_timesteps", 0) or 0)
        elapsed = time.perf_counter() - self._t0
        fps = ts / max(1e-6, elapsed)
        eta = (total - ts) / max(1e-6, fps)
        pct = 100.0 * ts / total if total else 0.0

        ep = getattr(self.model, "ep_info_buffer", [])
        rews = [e.get("r") for e in ep if "r" in e]
        lens = [e.get("l") for e in ep if "l" in e]
        rew_mean = float(np.mean(rews)) if rews else float("nan")
        len_mean = float(np.mean(lens)) if lens else 0.0

        vals = self.model.logger.name_to_value if self.model.logger else {}
        bits = [
            f"steps {ts:,}/{total:,} ({pct:.0f}%)",
            f"{elapsed:.0f}s elapsed ({fps:.1f} st/s, ETA {eta:.0f}s)",
        ]
        if self._episodes:
            bits.append(f"episodes {self._episodes}")
        if not np.isnan(rew_mean):
            bits.append(f"ep_rew_mean {rew_mean:+.2f} (len {len_mean:.0f})")
        if hasattr(self.model, "exploration_rate"):
            bits.append(f"eps {self.model.exploration_rate:.3f}")
        for key in (
            "train/loss",
            "train/q_loss",
            "train/policy_gradient_loss",
            "train/value_loss",
            "train/entropy_loss",
        ):
            if key in vals:
                bits.append(f"{key.split('/')[-1]} {vals[key]:.4f}")
        bar = "#" * int(20 * ts / max(1, total)) + "." * int(20 * (1 - ts / max(1, total)))
        print(f"[{bar}] " + " | ".join(bits))


def train(
    env_cfg_path: str,
    variants_cfg_path: str,
    reward_cfg_path: str,
    training_cfg_path: str,
    overrides: dict | None = None,
    env_cfg_override: dict | None = None,
    reward_cfg_override: dict | None = None,
    log_every: int = 2000,
    verbose_episodes: bool = False,
    hardware_cfg_path: str | None = None,
    checkpoint_every: int | None = None,
    resume_from: str | None = None,
):
    env_cfg = env_cfg_override if env_cfg_override is not None else load_yaml(env_cfg_path)
    variants_cfg = load_yaml(variants_cfg_path)
    reward_cfg = (
        reward_cfg_override if reward_cfg_override is not None else load_yaml(reward_cfg_path)
    )
    training_cfg = load_yaml(training_cfg_path)
    if overrides:
        training_cfg.update(overrides)

    algorithm = str(training_cfg.get("algorithm", "dqn")).lower()
    hardware_cfg = load_yaml(hardware_cfg_path) if hardware_cfg_path else None
    env = build_env_from_configs(env_cfg, variants_cfg, reward_cfg, hardware_cfg=hardware_cfg)
    print(
        f"env: obs={env.observation_space} actions={env.action_space.n} "
        f"videos={getattr(env.frame_source, 'n_videos', 1)}"
    )
    if env.edge_profile is not None:
        print(
            f"edge profile: {env.edge_profile.device} "
            f"(inference memory budget {env.edge_profile.memory_budget_mb:.0f} MB)"
        )
    if env.reward.cfg.constraints:
        print(f"constraints: {env.reward.cfg.constraints}")
    save_dir = Path(training_cfg.get("save_dir", "training"))
    tb_dir = Path(training_cfg.get("tb_dir", "runs/tensorboard"))
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        policy=training_cfg.get("policy", "MlpPolicy"),
        env=env,
        tensorboard_log=str(tb_dir),
        seed=int(training_cfg.get("seed", 0)),
        verbose=1,
    )

    timesteps = int(training_cfg.get("timesteps", 20000))
    log_name = training_cfg.get("log_name", algorithm)
    checkpoint_every = checkpoint_every or training_cfg.get("checkpoint_every")

    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        model = (PPO if algorithm == "ppo" else DQN).load(
            str(resume_path),
            env=env,
            tensorboard_log=str(tb_dir),
            seed=int(training_cfg.get("seed", 0)),
        )
        print(
            f"resumed {algorithm} from {resume_path} "
            f"(already {model.num_timesteps:,} steps, target {timesteps:,})"
        )
    elif algorithm == "ppo":
        model = PPO(
            **common,
            learning_rate=float(training_cfg.get("learning_rate", 3e-4)),
            n_steps=int(training_cfg.get("n_steps", 2048)),
            batch_size=int(training_cfg.get("batch_size", 64)),
            n_epochs=int(training_cfg.get("n_epochs", 10)),
            gamma=float(training_cfg.get("gamma", 0.99)),
            gae_lambda=float(training_cfg.get("gae_lambda", 0.95)),
            clip_range=float(training_cfg.get("clip_range", 0.2)),
            ent_coef=float(training_cfg.get("ent_coef", 0.0)),
        )
    else:
        model = DQN(
            **common,
            learning_rate=float(training_cfg.get("learning_rate", 0.0001)),
            buffer_size=int(training_cfg.get("buffer_size", 100000)),
            batch_size=int(training_cfg.get("batch_size", 64)),
            gamma=float(training_cfg.get("gamma", 0.99)),
            train_freq=int(training_cfg.get("train_freq", 4)),
            gradient_steps=int(training_cfg.get("gradient_steps", 1)),
            target_update_interval=int(training_cfg.get("target_update_interval", 1000)),
            exploration_fraction=float(training_cfg.get("exploration_fraction", 0.2)),
            exploration_final_eps=float(training_cfg.get("exploration_final_eps", 0.05)),
        )

    callbacks = [ProgressCallback(log_every=log_every, verbose_episodes=verbose_episodes)]
    cb_verbosity = 1 if resume_from else 0
    if checkpoint_every:
        callbacks.append(
            CheckpointCallback(
                save_freq=int(checkpoint_every),
                save_path=str(save_dir),
                name_prefix=log_name,
                verbose=cb_verbosity,
            )
        )
    learn_target = timesteps
    if resume_from:
        remaining = timesteps - int(model.num_timesteps)
        if remaining <= 0:
            print(
                f"already trained {model.num_timesteps:,} >= target "
                f"{timesteps:,}; skipping learn"
            )
        else:
            print(f"resume: {remaining:,} steps remaining to target {timesteps:,}")
        learn_target = max(remaining, 0)
    if learn_target > 0:
        model.learn(
            total_timesteps=learn_target,
            tb_log_name=log_name,
            callback=callbacks,
            reset_num_timesteps=not resume_from,
        )
    else:
        print("no learning needed")

    save_path = save_dir / f"{log_name}_final.zip"
    model.save(str(save_path))
    print(f"model saved to {save_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the IVR RL agent (offline, no network)")
    parser.add_argument("--env", default="configs/env_bdd.yaml")
    parser.add_argument("--variants", default="configs/variants.yaml")
    parser.add_argument("--reward", default="configs/reward_bdd.yaml")
    parser.add_argument("--training", default="configs/training_bdd.yaml")
    parser.add_argument("--hardware", default="configs/hardware.yaml", help="hardware profile yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--algorithm", default=None, choices=["dqn", "ppo"])
    parser.add_argument("--log-every", type=int, default=2000)
    parser.add_argument("--verbose-episodes", action="store_true")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="save a checkpoint every N steps (e.g. 50000)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="resume training from this checkpoint zip",
    )
    args = parser.parse_args()

    overrides: dict = {}
    if args.timesteps is not None:
        overrides["timesteps"] = args.timesteps
    if args.algorithm is not None:
        overrides["algorithm"] = args.algorithm
    train(
        args.env,
        args.variants,
        args.reward,
        args.training,
        overrides=overrides or None,
        log_every=args.log_every,
        verbose_episodes=args.verbose_episodes,
        hardware_cfg_path=args.hardware,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume,
    )
