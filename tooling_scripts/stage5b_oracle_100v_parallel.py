"""18-action fixed-policy landscape on a 100-video subsample of the
306-video test split, parallelized across the 18 actions.

Same seeded (seed=42) subsample-selection logic as the sequential
tools/stage5b_oracle_100v.py, same output filename -- just distributed across
a multiprocessing.Pool of worker processes instead of one process working
through all 18 actions serially.
"""

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.benchmark_runner import _split_videos  # noqa: E402
from runtime.environment_factory import load_yaml  # noqa: E402
from tooling_scripts.stage5b_parallel_common import run_parallel  # noqa: E402

SUBSAMPLE_SEED = 42
SUBSAMPLE_SIZE = 100
OUTPUT_NAME = "stage5_oracle_18action_100v.csv"


def main() -> None:
    env_cfg = load_yaml("configs/env/env_bdd.yaml")
    variants_cfg = load_yaml("configs/variants.yaml")

    all_videos = _split_videos(env_cfg.get("dataset_dir", "../BDDA/BDDA"), "test")
    rng = random.Random(SUBSAMPLE_SEED)
    subsample = sorted(rng.sample(all_videos, SUBSAMPLE_SIZE))
    print(f"full test split: {len(all_videos)} videos", flush=True)
    print(f"seeded subsample (seed={SUBSAMPLE_SEED}): {len(subsample)} videos", flush=True)

    model_names = [m["name"] for m in variants_cfg["models"]]
    resolutions = variants_cfg["resolutions"]
    precisions = variants_cfg["precisions"]
    specs = [
        ("always", m, r, p) for m, r, p in itertools.product(model_names, resolutions, precisions)
    ]
    print(f"policies ({len(specs)}): {specs}", flush=True)

    run_parallel(specs, subsample, output_dir="output", output_filename=OUTPUT_NAME)
    print("=== stage5b 100-video oracle sweep (parallel) complete ===", flush=True)


if __name__ == "__main__":
    main()
