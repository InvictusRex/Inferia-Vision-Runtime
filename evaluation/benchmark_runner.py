from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import stats as scipy_stats

from reinforcement_learning.vision_runtime_env import VisionRuntimeEnv
from .baseline_schedulers import Policy, default_baselines

DENSITY_BINS = [(0.0, 8.0, "sparse"), (8.0, 16.0, "mid"), (16.0, float("inf"), "dense")]


def _progress_bar(done: int, total: int, width: int = 18) -> str:
    frac = done / total if total else 0.0
    filled = int(width * frac)
    return f"[{'#' * filled}{'.' * (width - filled)}] {done:>5}/{total} ({frac * 100:3.0f}%)"


class EvalProgress:
    """Live console progress for dataset evaluation (mirrors train.py's bar)."""

    def __init__(self, log_every: int = 1):
        self.log_every = log_every
        self.start = 0.0

    def begin(self, total: int, label: str) -> None:
        import time as _t

        self.start = _t.time()
        self.total = total
        self.label = label

    def update(self, done: int, policy_name: str) -> None:
        import time as _t

        elapsed = _t.time() - self.start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else 0.0
        if done == self.total or done % self.log_every == 0:
            prog = _progress_bar(done, self.total)
            print(
                f"{prog} | {self.label} vids={done}/{self.total} ({rate:.2f} vids/s, "
                f"ETA {eta:5.0f}s) | policy={policy_name}",
                flush=True,
            )


def _bin_label(count: float) -> str:
    for lo, hi, label in DENSITY_BINS:
        if lo <= count < hi:
            return label
    return "dense"


def run_episode(env: VisionRuntimeEnv, policy: Policy, max_steps: int | None = None) -> dict:
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    switches = 0
    prev_action = info["action"]
    latencies: list[float] = []
    confs: list[float] = []
    counts: list[int] = []
    rewards: list[float] = []
    fps_samples: list[float] = []
    npu_utils: list[float] = []
    mems: list[float] = []
    powers: list[float] = []
    temps: list[float] = []
    violation_frames = 0
    model_counts: dict[str, int] = {}
    resolution_counts: dict[int, int] = {}
    precision_counts: dict[str, int] = {}
    bin_rewards: dict[str, list[float]] = {"sparse": [], "mid": [], "dense": []}

    done = False
    while not done:
        if max_steps is not None and steps >= max_steps:
            break
        action = policy.choose(obs, info)
        if action != prev_action:
            switches += 1
        obs, reward, terminated, truncated, info = env.step(action)
        policy.update(obs, info, reward, action)
        total_reward += reward
        steps += 1
        latencies.append(float(info["latency_ms"]))
        confs.append(float(info["mean_conf"]))
        counts.append(int(info["obj_count"]))
        rewards.append(float(reward))
        fps_samples.append(float(info["fps"]))
        npu_utils.append(float(info["npu_util_pct"]))
        mems.append(float(info["mem_used_mb"]))
        powers.append(float(info["power_w"]))
        temps.append(float(info["temp_c"]))
        if info["constraint_violation_count"] > 0:
            violation_frames += 1
        model_counts[info["model"]] = model_counts.get(info["model"], 0) + 1
        resolution_counts[info["resolution"]] = resolution_counts.get(info["resolution"], 0) + 1
        precision_counts[info["precision"]] = precision_counts.get(info["precision"], 0) + 1
        bin_rewards[_bin_label(float(info["obj_count"]))].append(float(reward))
        prev_action = action
        done = terminated or truncated

    return {
        "policy": policy.name,
        "steps": steps,
        "total_reward": float(total_reward),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_conf": float(np.mean(confs)) if confs else 0.0,
        "mean_count": float(np.mean(counts)) if counts else 0.0,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "mean_fps": float(np.mean(fps_samples)) if fps_samples else 0.0,
        "mean_npu_util": float(np.mean(npu_utils)) if npu_utils else 0.0,
        "mean_mem_mb": float(np.mean(mems)) if mems else 0.0,
        "mean_power_w": float(np.mean(powers)) if powers else 0.0,
        "peak_temp_c": float(np.max(temps)) if temps else 0.0,
        "violation_count": int(violation_frames),
        "violation_rate": violation_frames / steps if steps else 0.0,
        "switches": switches,
        "switch_rate": switches / steps if steps else 0.0,
        "model_counts": model_counts,
        "resolution_counts": resolution_counts,
        "precision_counts": precision_counts,
        "bin_rewards": {k: (float(np.mean(v)) if v else 0.0) for k, v in bin_rewards.items()},
        "bin_frame_counts": {k: len(v) for k, v in bin_rewards.items()},
    }


def _config_breakdown_keys(env: VisionRuntimeEnv) -> tuple[list[str], list[int], list[str]]:
    """Canonical (models, resolutions, precisions) for CSV column naming.

    Derived via `action_to_config` over the full action space -- never via
    `ConfigSpace.models`'s positional order, which differs from the action
    enumeration (model-major itertools.product order).
    """
    configs = [env.config_space.action_to_config(i) for i in range(env.action_space.n)]
    models = sorted({c.model for c in configs})
    resolutions = sorted({c.resolution for c in configs})
    precisions = sorted({c.precision for c in configs})
    return models, resolutions, precisions


def _row_with_breakdown(
    r: dict, models: list[str], resolutions: list[int], precisions: list[str]
) -> dict:
    steps = max(1, int(r["steps"]))
    row = {
        k: v
        for k, v in r.items()
        if k
        not in (
            "bin_rewards",
            "bin_frame_counts",
            "model_counts",
            "resolution_counts",
            "precision_counts",
        )
    }
    row["bin_sparse_reward"] = r["bin_rewards"]["sparse"]
    row["bin_mid_reward"] = r["bin_rewards"]["mid"]
    row["bin_dense_reward"] = r["bin_rewards"]["dense"]
    row["bin_sparse_frames"] = r["bin_frame_counts"]["sparse"]
    row["bin_mid_frames"] = r["bin_frame_counts"]["mid"]
    row["bin_dense_frames"] = r["bin_frame_counts"]["dense"]
    for m in models:
        row[f"model_frac_{m}"] = r["model_counts"].get(m, 0) / steps
    for res in resolutions:
        row[f"resolution_frac_{res}"] = r["resolution_counts"].get(res, 0) / steps
    for prec in precisions:
        row[f"precision_frac_{prec}"] = r["precision_counts"].get(prec, 0) / steps
    return row


def _csv_fieldnames(models: list[str], resolutions: list[int], precisions: list[str]) -> list[str]:
    return (
        [
            "video",
            "policy",
            "steps",
            "total_reward",
            "mean_reward",
            "mean_conf",
            "mean_count",
            "mean_latency_ms",
            "mean_fps",
            "mean_npu_util",
            "mean_mem_mb",
            "mean_power_w",
            "peak_temp_c",
            "violation_count",
            "violation_rate",
            "switches",
            "switch_rate",
            "bin_sparse_reward",
            "bin_mid_reward",
            "bin_dense_reward",
            "bin_sparse_frames",
            "bin_mid_frames",
            "bin_dense_frames",
        ]
        + [f"model_frac_{m}" for m in models]
        + [f"resolution_frac_{res}" for res in resolutions]
        + [f"precision_frac_{prec}" for prec in precisions]
    )


def paired_comparison(rewards_a: np.ndarray, rewards_b: np.ndarray) -> dict:
    """Paired per-video comparison of `a` (candidate) vs `b` (reference).

    Assumes both arrays are per-video total_reward over the *same* videos in the
    *same* order (true for compare_policies_dataset, which runs every policy over
    an identical video list).
    """
    diff = rewards_a - rewards_b
    n = len(diff)
    mean_diff = float(np.mean(diff)) if n else 0.0
    if n > 1 and np.std(diff, ddof=1) > 0:
        t_stat, p_value = scipy_stats.ttest_rel(rewards_a, rewards_b)
        sem = float(np.std(diff, ddof=1) / np.sqrt(n))
        t_crit = float(scipy_stats.t.ppf(0.975, df=n - 1))
        ci_lo = mean_diff - t_crit * sem
        ci_hi = mean_diff + t_crit * sem
    else:
        t_stat, p_value = float("nan"), float("nan")
        ci_lo, ci_hi = mean_diff, mean_diff
    win_rate = float(np.mean(diff > 0)) if n else 0.0
    return {
        "n": n,
        "mean_a": float(np.mean(rewards_a)) if n else 0.0,
        "mean_b": float(np.mean(rewards_b)) if n else 0.0,
        "mean_delta": mean_diff,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "ci95_lo": float(ci_lo),
        "ci95_hi": float(ci_hi),
        "win_rate": win_rate,
    }


def compare_policies_stats(results: dict[str, list[dict]], reference: str) -> list[dict]:
    """Paired t-test + win rate + 95% CI of every other policy vs `reference`."""
    if reference not in results:
        return []
    ref_rewards = np.array([r["total_reward"] for r in results[reference]])
    rows = []
    for name, per_video in results.items():
        if name == reference:
            continue
        rewards = np.array([r["total_reward"] for r in per_video])
        if len(rewards) != len(ref_rewards):
            continue
        stat = paired_comparison(rewards, ref_rewards)
        stat["policy"] = name
        stat["reference"] = reference
        rows.append(stat)
    return rows


def _print_stats(rows: list[dict]) -> None:
    if not rows:
        return
    print("\n=== paired comparison vs reference (per-video total_reward) ===")
    for row in rows:
        ci = f"[{row['ci95_lo']:+.2f}, {row['ci95_hi']:+.2f}]"
        print(
            f"{row['policy']:<30} vs {row['reference']:<20} "
            f"delta={row['mean_delta']:+7.2f} (95% CI {ci}) "
            f"t={row['t_stat']:+.2f} p={row['p_value']:.2e} win_rate={row['win_rate'] * 100:5.1f}%"
        )


def _write_stats_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "policy",
                "reference",
                "n",
                "mean_a",
                "mean_b",
                "mean_delta",
                "t_stat",
                "p_value",
                "ci95_lo",
                "ci95_hi",
                "win_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"stats written to {path}")


def compare_policies(
    env_builder: Callable[[], VisionRuntimeEnv],
    policies: list[Policy],
    episodes: int = 1,
    max_steps: int | None = None,
    output_dir: str = "output",
) -> dict[str, list[dict]]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict]] = {}
    for policy in policies:
        per_episode = []
        for _ in range(episodes):
            policy.reset()
            env = env_builder()
            per_episode.append(run_episode(env, policy, max_steps=max_steps))
        results[policy.name] = per_episode
        row = per_episode[0]
        print(
            f"{row['policy']:<30} steps={row['steps']:5d} reward={row['total_reward']:8.1f} "
            f"conf={row['mean_conf']:.3f} count={row['mean_count']:5.1f} "
            f"lat={row['mean_latency_ms']:6.1f}ms fps={row['mean_fps']:6.1f} "
            f"switches={row['switches']:4d}"
        )

    path = out_dir / "benchmark.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "policy",
                "steps",
                "total_reward",
                "mean_reward",
                "mean_conf",
                "mean_count",
                "mean_latency_ms",
                "mean_fps",
                "mean_npu_util",
                "mean_mem_mb",
                "mean_power_w",
                "peak_temp_c",
                "violation_count",
                "violation_rate",
                "switches",
                "switch_rate",
            ],
        )
        writer.writeheader()
        for per_episode in results.values():
            for r in per_episode:
                writer.writerow({k: v for k, v in r.items() if k in writer.fieldnames})
    print(f"benchmark written to {path}")
    return results


def compare_policies_dataset(
    env_builder: Callable[[str], VisionRuntimeEnv],
    video_paths: list[str],
    policies: list[Policy],
    output_dir: str = "output",
    output_filename: str = "benchmark_dataset.csv",
    progress_every: int | None = None,
    stats_reference: str | None = None,
) -> dict[str, list[dict]]:
    """Run every policy over a list of videos; aggregate mean +/- std + density bins.

    progress_every: print a live progress line every N videos per policy (train-style
    bar). None disables live progress (still prints the per-policy summary rows).
    stats_reference: policy name to run paired t-tests / win rates / CIs against
    (default: the last policy in `policies`, which callers append the trained
    model as).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = EvalProgress(progress_every) if progress_every else None
    results: dict[str, list[dict]] = {}
    n_videos = len(video_paths)

    probe_env = env_builder(video_paths[0])
    models, resolutions, precisions = _config_breakdown_keys(probe_env)

    for p_idx, policy in enumerate(policies):
        if progress:
            progress.begin(n_videos, f"{policy.name}")
        per_video: list[dict] = []
        for v_idx, path in enumerate(video_paths):
            policy.reset()
            env = env_builder(path)
            row = run_episode(env, policy)
            row["video"] = Path(path).name
            per_video.append(row)
            if progress:
                progress.update(v_idx + 1, policy.name)
        results[policy.name] = per_video

        rewards = np.array([r["total_reward"] for r in per_video])
        confs = np.array([r["mean_conf"] for r in per_video])
        counts = np.array([r["mean_count"] for r in per_video])
        lats = np.array([r["mean_latency_ms"] for r in per_video])
        sw = np.array([r["switches"] for r in per_video])
        viol = np.array([r["violation_rate"] for r in per_video])
        bin_rows = []
        for b in ("sparse", "mid", "dense"):
            vals = []
            for r in per_video:
                vals.extend([r["bin_rewards"].get(b, 0.0)] * r["bin_frame_counts"].get(b, 0))
            if vals:
                bin_rows.append(f"{b}={np.mean(vals):+.2f}")
        print(
            f"{policy.name:<30} videos={len(per_video)} "
            f"reward={rewards.mean():8.1f} +/- {rewards.std():6.1f} "
            f"conf={confs.mean():.3f} count={counts.mean():5.1f} "
            f"lat={lats.mean():6.1f}ms "
            f"fps={np.array([r['mean_fps'] for r in per_video]).mean():6.1f} "
            f"switches={sw.mean():6.1f} viol_rate={viol.mean() * 100:4.1f}% | " + " ".join(bin_rows)
        )

    csv_path = out_dir / output_filename
    fieldnames = _csv_fieldnames(models, resolutions, precisions)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for per_video in results.values():
            for r in per_video:
                writer.writerow(_row_with_breakdown(r, models, resolutions, precisions))
    print(f"dataset benchmark written to {csv_path}")

    reference = stats_reference or (policies[-1].name if policies else None)
    if reference:
        stats_rows = compare_policies_stats(results, reference)
        _print_stats(stats_rows)
        stats_path = out_dir / f"{Path(output_filename).stem}_stats.csv"
        _write_stats_csv(stats_path, stats_rows)

    return results


def _load_configs(env_path: str, variants_path: str, reward_path: str, hardware_path: str = None):
    from runtime.environment_factory import build_env_from_configs, load_yaml

    env_cfg = load_yaml(env_path)
    variants_cfg = load_yaml(variants_path)
    reward_cfg = load_yaml(reward_path)
    hardware_cfg = load_yaml(hardware_path) if hardware_path else None
    return (
        env_cfg,
        variants_cfg,
        reward_cfg,
        hardware_cfg,
        build_env_from_configs(env_cfg, variants_cfg, reward_cfg, hardware_cfg=hardware_cfg),
    )


def _split_videos(dataset_dir: str, split: str) -> list[str]:
    base = Path(dataset_dir) / split / "camera_videos"
    paths = sorted(p for p in base.glob("*.mp4") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"no camera videos under {base}")
    return [str(p) for p in paths]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IVR baseline comparison (single or dataset)")
    parser.add_argument("--env", default="configs/env_bdd.yaml")
    parser.add_argument("--variants", default="configs/variants.yaml")
    parser.add_argument("--reward", default="configs/reward_bdd.yaml")
    parser.add_argument("--hardware", default="configs/hardware.yaml", help="hardware profile yaml")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dataset", choices=["training", "validation", "test"], default=None)
    parser.add_argument("--n-videos", type=int, default=None, help="limit videos (default: all)")
    parser.add_argument("--model", default=None, help="path to a saved SB3 DQN/PPO zip")
    parser.add_argument("--model-name", default=None, help="label for the loaded model")
    parser.add_argument("--policies", default=None, help="comma-separated baseline names override")
    args = parser.parse_args()

    env_cfg, variants_cfg, reward_cfg, hardware_cfg, env = _load_configs(
        args.env, args.variants, args.reward, args.hardware
    )

    def builder_for(path: str = None):
        from runtime.environment_factory import build_env_from_configs

        return build_env_from_configs(
            env_cfg, variants_cfg, reward_cfg, video_override=path, hardware_cfg=hardware_cfg
        )

    policies = default_baselines(env)
    if args.policies:
        wanted = {p.strip() for p in args.policies.split(",")}
        policies = [p for p in policies if p.name in wanted]
    if args.model:
        from stable_baselines3 import DQN, PPO
        from .baseline_schedulers import SB3Policy

        try:
            model = PPO.load(args.model)
        except Exception:
            model = DQN.load(args.model)
        name = args.model_name or f"ivr_{Path(args.model).stem}"
        policies.append(SB3Policy(model, name=name))

    if args.dataset:
        videos = _split_videos(env_cfg.get("dataset_dir", "../BDDA/BDDA"), args.dataset)
        if args.n_videos:
            videos = videos[: args.n_videos]
        compare_policies_dataset(builder_for, videos, policies)
    else:
        compare_policies(
            lambda: builder_for(None),
            policies,
            episodes=args.episodes,
            max_steps=args.max_steps,
        )
