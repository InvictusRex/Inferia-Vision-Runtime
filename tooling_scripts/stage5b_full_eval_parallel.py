"""Full 306-video evaluation, parallelized across policies.

Replaces the sequential tools/stage5b_full_eval.py: same 10 required policies,
same full 306-video test split, same output filename -- just run with a
multiprocessing.Pool of worker processes (one policy per worker at a time)
instead of one process working through all 10 policies serially. See
tools/stage5b_parallel_common.py for why this is safe to parallelize.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.benchmark_runner import _split_videos  # noqa: E402
from runtime.environment_factory import load_yaml  # noqa: E402
from tooling_scripts.stage5b_parallel_common import run_parallel  # noqa: E402

DQN_CHECKPOINT = "training/dqn_bdd_s5_120k_final.zip"
PPO_CHECKPOINT = "training/ppo_bdd_s5_300k_final.zip"
OUTPUT_NAME = "stage5_306v_full_eval_output.csv"

SPECS = [
    ("always", "yolo11n", 640, "fp32"),
    ("always", "yolo11s", 640, "fp32"),
    ("always", "yolo11m", 640, "fp32"),
    ("random",),
    ("rule_based",),
    ("contextual_bandit",),
    ("always", "yolo11n", 960, "fp16"),  # action 5
    ("always", "yolo11n", 640, "fp16"),  # action 3, best-fixed among the 18 configs
    ("dqn", DQN_CHECKPOINT, "ivr_dqn_s5_120k"),
    ("ppo", PPO_CHECKPOINT, "ivr_ppo_s5_300k"),
]


def main() -> None:
    env_cfg = load_yaml("configs/env_bdd.yaml")
    videos = _split_videos(env_cfg.get("dataset_dir", "../BDDA/BDDA"), "test")
    print(f"n videos: {len(videos)} (full test split)", flush=True)
    print(f"policies ({len(SPECS)}): {SPECS}", flush=True)

    run_parallel(
        SPECS,
        videos,
        output_dir="output",
        output_filename=OUTPUT_NAME,
        stats_reference="ivr_dqn_s5_120k",
    )
    print("=== stage5b full-306 eval (parallel) complete ===", flush=True)


if __name__ == "__main__":
    main()
