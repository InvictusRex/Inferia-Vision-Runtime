"""Shared worker/merge logic for the parallelized evaluation scripts.

Parallelizes by POLICY: each worker process builds its own env/detector (its own
CUDA context) and runs ONE policy across every video assigned to it, entirely
independently of every other worker -- there is no shared state between
policies or videos, so this is an embarrassingly-parallel workload. A
multiprocessing.Pool distributes the policy list across N_WORKERS processes.

Policies are described as plain, picklable tuples (never as live objects) so
each worker constructs its own model/policy instance from scratch -- this
sidesteps pickling SB3 models or env-bound Policy objects across the process
boundary entirely.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
from pathlib import Path

from implementation.evaluation.baseline_schedulers import (
    AlwaysConfig,
    ContextualBandit,
    RandomPolicy,
    RuleBasedPolicy,
    SB3Policy,
)
from implementation.evaluation.benchmark_runner import (
    _csv_fieldnames,
    _row_with_breakdown,
    _write_stats_csv,
    compare_policies_stats,
    run_episode,
)
from implementation.runtime.environment_factory import build_env_from_configs, load_yaml

N_WORKERS = 5


def _build_policy(env, spec: tuple):
    kind = spec[0]
    if kind == "always":
        _, model, res, prec = spec
        return AlwaysConfig(env, model, res, prec)
    if kind == "random":
        return RandomPolicy(env.action_space.n, seed=0)
    if kind == "rule_based":
        return RuleBasedPolicy(env)
    if kind == "contextual_bandit":
        return ContextualBandit(env.action_space.n)
    if kind in ("dqn", "ppo"):
        from stable_baselines3 import DQN, PPO

        _, checkpoint, name = spec
        model = (DQN if kind == "dqn" else PPO).load(checkpoint, device="cpu")
        return SB3Policy(model, name=name)
    raise ValueError(f"unknown policy spec: {spec}")


def _policy_name(spec: tuple) -> str:
    if spec[0] == "always":
        _, model, res, prec = spec
        return f"always_{model}_{res}_{prec}"
    if spec[0] in ("dqn", "ppo"):
        return spec[2]
    return spec[0]


def _worker(args) -> tuple[str, list[dict]]:
    spec, videos = args
    env_cfg = load_yaml("configs/env_bdd.yaml")
    variants_cfg = load_yaml("configs/variants.yaml")
    reward_cfg = load_yaml("configs/reward_bdd.yaml")
    hardware_cfg = load_yaml("configs/hardware.yaml")

    def builder_for(path=None):
        return build_env_from_configs(
            env_cfg, variants_cfg, reward_cfg, video_override=path, hardware_cfg=hardware_cfg
        )

    probe_env = builder_for(videos[0])
    policy = _build_policy(probe_env, spec)
    name = policy.name
    print(f"[{name}] starting {len(videos)} videos", flush=True)

    rows = []
    for i, path in enumerate(videos):
        policy.reset()
        env = builder_for(path)
        row = run_episode(env, policy)
        row["video"] = Path(path).name
        rows.append(row)
        if (i + 1) % 20 == 0 or (i + 1) == len(videos):
            print(f"[{name}] {i + 1}/{len(videos)} videos done", flush=True)

    print(f"[{name}] finished all {len(videos)} videos", flush=True)
    return name, rows


def run_parallel(
    specs: list[tuple],
    videos: list[str],
    output_dir: str,
    output_filename: str,
    stats_reference: str | None = None,
    n_workers: int = N_WORKERS,
) -> None:
    variants_cfg = load_yaml("configs/variants.yaml")
    model_names = sorted({m["name"] for m in variants_cfg["models"]})
    resolutions = sorted(variants_cfg["resolutions"])
    precisions = sorted(variants_cfg["precisions"])

    tasks = [(spec, videos) for spec in specs]
    print(f"dispatching {len(tasks)} policies across {n_workers} worker processes "
          f"over {len(videos)} videos each", flush=True)

    with mp.Pool(processes=n_workers) as pool:
        results_list = pool.map(_worker, tasks)

    results: dict[str, list[dict]] = dict(results_list)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames(model_names, resolutions, precisions)
    csv_path = out_dir / output_filename
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for per_video in results.values():
            for r in per_video:
                writer.writerow(_row_with_breakdown(r, model_names, resolutions, precisions))
    print(f"written: {csv_path} ({sum(len(v) for v in results.values())} rows)", flush=True)

    if stats_reference:
        stats_rows = compare_policies_stats(results, stats_reference)
        for row in stats_rows:
            ci = f"[{row['ci95_lo']:+.2f}, {row['ci95_hi']:+.2f}]"
            win_pct = row["win_rate"] * 100
            print(
                f"{row['policy']:<30} vs {row['reference']:<20} "
                f"delta={row['mean_delta']:+7.2f} (95% CI {ci}) "
                f"t={row['t_stat']:+.2f} p={row['p_value']:.2e} win_rate={win_pct:5.1f}%",
                flush=True,
            )
        stats_path = out_dir / f"{Path(output_filename).stem}_stats.csv"
        _write_stats_csv(stats_path, stats_rows)
