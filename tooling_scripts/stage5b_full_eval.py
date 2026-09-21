"""Full 306-video evaluation of the new DQN/PPO checkpoints plus the required
reference baselines.

This does not use eval.py's CLI because the required baseline set includes two
specific fixed configs (action 5 and action 3) that eval.py's default_baselines()
does not construct. It reuses the same underlying evaluation infrastructure
(compare_policies_dataset) with an explicitly assembled policy list instead.

Required baselines:
  - Always-yolo11n, Always-yolo11s, Always-yolo11m (canonical 640/fp32 configs)
  - Always-yolo11n/960/fp16 (action 5 -- the exact config PPO previously selected)
  - Always-yolo11n/640/fp16 (action 3 -- the best fixed configuration among all 18
    actions, per the prior 40-video forensic landscape survey)
  - Rule-Based, Random, Contextual Bandit
plus the new DQN and PPO checkpoints.

Output: output/stage5_306v_full_eval_output.csv (+ _stats.csv), never overwrites
any historical output.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import DQN, PPO  # noqa: E402

from evaluation.baseline_schedulers import AlwaysConfig, default_baselines, SB3Policy  # noqa: E402
from evaluation.benchmark_runner import _split_videos, compare_policies_dataset  # noqa: E402
from runtime.environment_factory import build_env_from_configs, load_yaml  # noqa: E402

DQN_CHECKPOINT = "training/dqn_bdd_s5_120k_final.zip"
PPO_CHECKPOINT = "training/ppo_bdd_s5_300k_final.zip"
OUTPUT_NAME = "stage5_306v_full_eval_output.csv"


def main() -> None:
    env_cfg = load_yaml("configs/env/env_bdd.yaml")
    variants_cfg = load_yaml("configs/variants.yaml")
    reward_cfg = load_yaml("configs/reward/reward_bdd.yaml")
    hardware_cfg = load_yaml("configs/hardware.yaml")

    env = build_env_from_configs(env_cfg, variants_cfg, reward_cfg, hardware_cfg=hardware_cfg)

    def builder_for(path=None):
        return build_env_from_configs(
            env_cfg, variants_cfg, reward_cfg, video_override=path, hardware_cfg=hardware_cfg
        )

    policies = default_baselines(env)  # always_n/s/m@640fp32, random, rule_based, contextual_bandit
    policies.append(AlwaysConfig(env, "yolo11n", 960, "fp16"))  # action 5
    policies.append(AlwaysConfig(env, "yolo11n", 640, "fp16"))  # action 3, best fixed config

    dqn_model = DQN.load(DQN_CHECKPOINT)
    ppo_model = PPO.load(PPO_CHECKPOINT)
    policies.append(SB3Policy(dqn_model, name="ivr_dqn_s5_120k"))
    policies.append(SB3Policy(ppo_model, name="ivr_ppo_s5_300k"))

    print(f"policies ({len(policies)}): {[p.name for p in policies]}", flush=True)

    videos = _split_videos(env_cfg.get("dataset_dir", "../BDDA/BDDA"), "test")
    print(f"n videos: {len(videos)} (full test split)", flush=True)

    compare_policies_dataset(
        builder_for,
        videos,
        policies,
        output_dir="output",
        output_filename=OUTPUT_NAME,
        progress_every=20,
        stats_reference="ivr_dqn_s5_120k",
    )
    print("=== stage5b full-306 eval complete ===", flush=True)


if __name__ == "__main__":
    main()
