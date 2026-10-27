"""Full UA-DETRAC test-split evaluation, parallelized across policies.

Runs any mix of the 18 fixed configurations (the oracle landscape), the
non-learning baselines and trained checkpoints over every annotated test
sequence, with the real IoU-matched F1 reward, and writes one per-video CSV
plus paired statistics against a reference policy.

    python tooling_scripts/detrac_full_eval_parallel.py --fixed all --baselines ^
        --model ivr_dqn_s0=training/dqn_detrac_final.zip ^
        --model ivr_ppo_s0=training/ppo_detrac_final.zip ^
        --output test_detrac_full_eval_output.csv --stats-reference ivr_dqn_s0
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime.environment_factory import detrac_sequence_dirs, load_yaml  # noqa: E402
from tooling_scripts.stage5b_parallel_common import run_parallel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="configs/env/env_detrac.yaml")
    parser.add_argument("--reward", default="configs/reward/reward_detrac.yaml")
    parser.add_argument("--hardware", default="configs/hardware.yaml")
    parser.add_argument("--variants", default="configs/variants.yaml")
    parser.add_argument(
        "--fixed",
        choices=["none", "640fp32", "all"],
        default="none",
        help="fixed-config policies: none, the three 640/fp32 ones, or all 18",
    )
    parser.add_argument(
        "--baselines",
        action="store_true",
        help="add random, rule-based and contextual-bandit baselines",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="trained checkpoint as name=path.zip (repeatable)",
    )
    parser.add_argument(
        "--deadline-lookup",
        default=None,
        help="add the deadline lookup schedulers; value is a yaml of per model_res accuracy",
    )
    parser.add_argument("--constraints", choices=["on", "off"], default=None)
    parser.add_argument("--n-videos", type=int, default=None)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-reference", default=None)
    args = parser.parse_args()

    configs = {
        "env": load_yaml(args.env),
        "variants": load_yaml(args.variants),
        "reward": load_yaml(args.reward),
        "hardware": load_yaml(args.hardware),
    }
    if args.constraints == "off":
        configs["reward"]["constraints"] = {}
    elif args.constraints == "on":
        stress = load_yaml("configs/reward/reward_constrained_bdd.yaml")
        configs["reward"]["constraints"] = stress.get("constraints", {})
        configs["reward"]["constraint_weights"] = stress.get("constraint_weights", {})

    videos = detrac_sequence_dirs(configs["env"]["dataset_dir"], "test")
    if args.n_videos:
        videos = videos[: args.n_videos]

    variants = configs["variants"]
    if args.fixed == "all":
        grid = itertools.product(
            [m["name"] for m in variants["models"]], variants["resolutions"], variants["precisions"]
        )
    elif args.fixed == "640fp32":
        grid = ((m["name"], 640, "fp32") for m in variants["models"])
    else:
        grid = ()
    specs: list[tuple] = [("always", m, r, p) for m, r, p in grid]
    if args.baselines:
        specs += [("random",), ("rule_based",), ("contextual_bandit",)]
    if args.deadline_lookup:
        specs += [("deadline_lookup", None), ("deadline_lookup", load_yaml(args.deadline_lookup))]
    for item in args.model:
        name, path = item.split("=", 1)
        kind = "ppo" if "ppo" in Path(path).stem.lower() else "dqn"
        specs.append((kind, path, name))
    if not specs:
        raise SystemExit("nothing to evaluate: pass --fixed, --baselines and/or --model")

    print(f"UA-DETRAC test split: {len(videos)} sequences, {len(specs)} policies", flush=True)
    run_parallel(
        specs,
        videos,
        output_dir="output",
        output_filename=args.output,
        stats_reference=args.stats_reference,
        n_workers=args.workers,
        configs=configs,
    )


if __name__ == "__main__":
    main()
