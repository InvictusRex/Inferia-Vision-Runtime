"""18-action fixed-policy landscape on a 100-video subsample of the
306-video test split, for the per-video adaptive-headroom oracle.

The full 18x306 sweep (5508 episodes, ~19h at the observed per-episode rate) was
judged disproportionate; this runs the full 18-action sweep on a smaller, fixed,
pre-registered subsample instead. The subsample is chosen by a seeded random
draw (seed=42) BEFORE any of this data exists, so it cannot be cherry-picked
after the fact. DQN, PPO and the other required baselines are evaluated
separately on the COMPLETE 306-video set (see stage5b_full_eval.py) -- headroom
figures computed from this file are explicitly a 100-video-subsample estimate,
never conflated with the full-306 metrics.

Output: output/stage5_oracle_18action_100v.csv, a new artifact name that does
not overwrite anything (including the unrelated 10- and 40-video forensic
landscape CSVs from the earlier investigation).
"""

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.baseline_schedulers import AlwaysConfig  # noqa: E402
from evaluation.benchmark_runner import _split_videos, compare_policies_dataset  # noqa: E402
from runtime.environment_factory import build_env_from_configs, load_yaml  # noqa: E402

SUBSAMPLE_SEED = 42
SUBSAMPLE_SIZE = 100
OUTPUT_NAME = "stage5_oracle_18action_100v.csv"


def main() -> None:
    env_cfg = load_yaml("configs/env_bdd.yaml")
    variants_cfg = load_yaml("configs/variants.yaml")
    reward_cfg = load_yaml("configs/reward_bdd.yaml")
    hardware_cfg = load_yaml("configs/hardware.yaml")

    env = build_env_from_configs(env_cfg, variants_cfg, reward_cfg, hardware_cfg=hardware_cfg)

    def builder_for(path=None):
        return build_env_from_configs(
            env_cfg, variants_cfg, reward_cfg, video_override=path, hardware_cfg=hardware_cfg
        )

    all_videos = _split_videos(env_cfg.get("dataset_dir", "../BDDA/BDDA"), "test")
    rng = random.Random(SUBSAMPLE_SEED)
    subsample = sorted(rng.sample(all_videos, SUBSAMPLE_SIZE))
    print(f"full test split: {len(all_videos)} videos", flush=True)
    print(f"seeded subsample (seed={SUBSAMPLE_SEED}): {len(subsample)} videos", flush=True)
    print(f"subsample: {[Path(v).name for v in subsample]}", flush=True)

    model_names = [m["name"] for m in variants_cfg["models"]]
    resolutions = variants_cfg["resolutions"]
    precisions = variants_cfg["precisions"]
    policies = [
        AlwaysConfig(env, m, r, p)
        for m, r, p in itertools.product(model_names, resolutions, precisions)
    ]
    print(f"policies ({len(policies)}): {[p.name for p in policies]}", flush=True)

    compare_policies_dataset(
        builder_for,
        subsample,
        policies,
        output_dir="output",
        output_filename=OUTPUT_NAME,
        progress_every=20,
    )
    print("=== stage5b 100-video oracle sweep complete ===", flush=True)


if __name__ == "__main__":
    main()
