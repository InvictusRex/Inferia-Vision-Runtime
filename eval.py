"""Convenience launcher for offline IVR evaluation.

Run from anywhere (no module form needed):

    python eval.py                                   # full BDD-100K test split, default model
    python eval.py --model training\\dqn_bdd_final.zip
    python eval.py --n-videos 20                     # quick sanity pass
    python eval.py --dataset validation

Runs every baseline + the trained model over each video, writes per-video rows to
output/<dataset>_<model>_eval_output.csv, and prints mean +/- std per policy plus
per-density-bin rewards. See docs/RUN_BDD.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from implementation.runtime.environment_factory import load_yaml  # noqa: E402


def default_output_name(dataset: str, model: str) -> str:
    """Output naming scheme: {dataset}_{model_stem}_eval_output.csv.

    e.g. test_dqn_bdd_final_eval_output.csv. Encodes split + model so future
    evals (other runs/models/splits) get their own file and never clobber.
    """
    stem = Path(model).stem
    return f"{dataset}_{stem}_eval_output.csv"


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained IVR RL agent (offline)")
    parser.add_argument("--env", default="configs/env_bdd.yaml")
    parser.add_argument("--variants", default="configs/variants.yaml")
    parser.add_argument("--reward", default="configs/reward_bdd.yaml")
    parser.add_argument("--hardware", default="configs/hardware.yaml", help="hardware profile yaml")
    parser.add_argument(
        "--dataset",
        default="test",
        choices=["training", "validation", "test"],
        help="BDD-100K split to evaluate (default: test)",
    )
    parser.add_argument("--model", default="training/dqn_bdd_final.zip")
    parser.add_argument("--model-name", default=None, help="label for the loaded model")
    parser.add_argument("--n-videos", type=int, default=None, help="limit videos (default: all)")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="print a live progress line every N videos per policy (0 disables)",
    )
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--output", default=None, help="override the output CSV filename")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="override dataset root (absolute or relative to repo root)",
    )
    parser.add_argument(
        "--policies",
        default=None,
        help="comma-separated baseline names override (default: all baselines + model)",
    )
    parser.add_argument("--w-switch", type=float, default=None, help="override reward w_switch")
    parser.add_argument(
        "--constraints",
        choices=["on", "off"],
        default=None,
        help="toggle constraint penalties in the reward (default: yaml value)",
    )
    args = parser.parse_args()

    from implementation.evaluation.benchmark_runner import (  # noqa: E402
        _split_videos,
        compare_policies_dataset,
    )
    from implementation.evaluation.baseline_schedulers import (  # noqa: E402
        SB3Policy,
        default_baselines,
    )
    from implementation.runtime.environment_factory import build_env_from_configs  # noqa: E402
    from stable_baselines3 import DQN, PPO  # noqa: E402

    env_cfg = load_yaml(args.env)
    if args.dataset_dir:
        env_cfg["dataset_dir"] = args.dataset_dir

    variants_cfg = load_yaml(args.variants)
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
    hardware_cfg = load_yaml(args.hardware)
    env = build_env_from_configs(env_cfg, variants_cfg, reward_cfg, hardware_cfg=hardware_cfg)

    def builder_for(path: str = None):
        return build_env_from_configs(
            env_cfg, variants_cfg, reward_cfg, video_override=path, hardware_cfg=hardware_cfg
        )

    policies = default_baselines(env)
    if args.policies:
        wanted = {p.strip() for p in args.policies.split(",")}
        policies = [p for p in policies if p.name in wanted]

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    try:
        model = PPO.load(model_path)
    except Exception:
        model = DQN.load(model_path)
    model_name = args.model_name or f"ivr_{model_path.stem}"
    policies.append(SB3Policy(model, name=model_name))

    videos = _split_videos(env_cfg.get("dataset_dir", "../BDDA/BDDA"), args.dataset)
    if args.n_videos:
        videos = videos[: args.n_videos]

    output_name = args.output or default_output_name(args.dataset, model_path.name)
    results = compare_policies_dataset(
        builder_for,
        videos,
        policies,
        output_dir=args.output_dir,
        output_filename=output_name,
        progress_every=args.progress_every if args.progress_every else None,
    )

    always = next(
        (r for r in results if r.startswith("always_") and "yolo11s" in r),
        None,
    )
    print("\n=== eval complete ===")
    print(f"model: {model_path}")
    print(f"split: {args.dataset} ({len(videos)} videos)")
    print(f"csv:   {Path(args.output_dir) / output_name}")
    if always and model_name in results:
        dqn_rewards = [r["total_reward"] for r in results[model_name]]
        base_rewards = [r["total_reward"] for r in results[always]]
        dq = sum(dqn_rewards) / len(dqn_rewards)
        ba = sum(base_rewards) / len(base_rewards)
        print(f"claim check: ivr_dqn mean {dq:+.1f} vs {always} {ba:+.1f} -> delta {dq - ba:+.1f}")


if __name__ == "__main__":
    main()
