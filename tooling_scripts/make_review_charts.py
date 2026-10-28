"""Render the review-slide charts from the evaluation and board-profile outputs.

    python tooling_scripts/make_review_charts.py \\
        --naive-int8-profile ../rknn_bundle/superseded_split_concat/rknn_board_profile.csv

Writes PNGs into docs/screenshots/review-slides/. Every chart reads only files under
output/ (plus the optional pre-fix board profile), so rerunning after new evaluations
regenerates the figures.
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "screenshots" / "review-slides"

# validated categorical slots (light surface); neutrals for everything that is context
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CONTEXT, CONTEXT_STRONG = "#c9c8c1", "#898781"
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
MODEL_COLORS = {"yolo11n": BLUE, "yolo11s": ORANGE, "yolo11m": AQUA}
PRECISION_MARKERS = {"fp16": "o", "int8": "s"}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": CONTEXT_STRONG,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "text.color": INK,
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "legend.frameon": False,
    }
)


def log_ms_axis(ax) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 50, 100, 200, 500]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())


def rewards(csv_path: Path) -> dict[str, list[float]]:
    by: dict[str, list[float]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by[row["policy"]].append(float(row["total_reward"]))
    return by


def fixed_label(policy: str) -> str:
    _, model, res, prec = policy.split("_")
    return f"{model.replace('yolo11', 'yolo11-')} {res} {prec}"


def chart_final(eval_csvs: list[Path], out: Path, title: str) -> None:
    per_policy: dict[str, list[float]] = {}
    for path in eval_csvs:
        per_policy.update(rewards(path))
    seeds: dict[str, list[float]] = defaultdict(list)
    bars = []
    for name, values in per_policy.items():
        if "masked" in name:  # deadline-masked PPO gets the latency profile; reported in tables only
            continue
        if name.startswith("ivr_"):
            algo = name.split("_")[1].upper()
            seeds[algo].append(st.mean(values))
        elif name.startswith("always_"):
            bars.append((fixed_label(name), st.mean(values), 0.0, CONTEXT))
        else:
            bars.append((name.replace("_", " "), st.mean(values), 0.0, CONTEXT_STRONG))
    for algo, color in (("DQN", BLUE), ("PPO", ORANGE)):
        if seeds[algo]:
            spread = st.stdev(seeds[algo]) if len(seeds[algo]) > 1 else 0.0
            label = f"IVR-{algo} ({len(seeds[algo])} seeds)"
            bars.append((label, st.mean(seeds[algo]), spread, color))
    bars.sort(key=lambda b: b[1])

    fig, ax = plt.subplots(figsize=(11, 0.36 * len(bars) + 1.6))
    y = range(len(bars))
    ax.barh(
        list(y),
        [b[1] for b in bars],
        color=[b[3] for b in bars],
        height=0.72,
        edgecolor=SURFACE,
        linewidth=2,
    )
    for i, bar in enumerate(bars):
        if bar[2] > 0:
            ax.errorbar(bar[1], i, xerr=bar[2], color=INK, capsize=4, elinewidth=1.2)
    ax.set_yticks(list(y), [b[0] for b in bars])
    for tick, bar in zip(ax.get_yticklabels(), bars):
        if bar[0].startswith("IVR"):
            tick.set_fontweight("bold")
            tick.set_color(INK)
    for i, bar in enumerate(bars):
        if bar[0].startswith("IVR") or i == len(bars) - 1 or bar[3] == CONTEXT_STRONG:
            x = bar[1] + bar[2]
            ax.text(x + 6 if x >= 0 else 6, i, f"{bar[1]:.1f}", va="center", fontsize=10)
    ax.axvline(0, color=CONTEXT_STRONG, linewidth=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean total reward per test sequence (higher is better)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def read_profile(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def chart_latency(profile: list[dict], eval_csv: Path, out: Path) -> None:
    sim = {}
    with open(eval_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["policy"].startswith("always_"):
                sim.setdefault(row["policy"][len("always_") :], float(row["mean_latency_ms"]))
    rows = sorted(profile, key=lambda r: float(r["e2e_ms_p50"]))
    fig, ax = plt.subplots(figsize=(10, 7.5))
    for i, r in enumerate(rows):
        s, m = sim[r["action"]], float(r["e2e_ms_p50"])
        ax.plot([s, m], [i, i], color=GRID, linewidth=2, zorder=1)
    ax.scatter(
        [sim[r["action"]] for r in rows],
        range(len(rows)),
        s=60,
        color=CONTEXT_STRONG,
        label="simulated (old profile)",
        zorder=2,
        edgecolor=SURFACE,
        linewidth=1.5,
    )
    ax.scatter(
        [float(r["e2e_ms_p50"]) for r in rows],
        range(len(rows)),
        s=60,
        color=BLUE,
        marker="s",
        label="measured on ROCK 5C NPU",
        zorder=3,
        edgecolor=SURFACE,
        linewidth=1.5,
    )
    ax.set_yticks(
        range(len(rows)),
        [f"{fixed_label('always_' + r['action'])}  ->  NPU {r['board_precision']}" for r in rows],
    )
    log_ms_axis(ax)
    ax.set_xlabel("Per-frame latency, ms (log scale; measured = letterbox + NPU + decode/NMS)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Simulated vs measured latency, all 18 configurations")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_f1(profile: list[dict], naive: list[dict] | None, out: Path) -> None:
    pairs = sorted({(r["action"].split("_")[0], int(r["action"].split("_")[1])) for r in profile})
    by_action = {r["action"]: r for r in profile}
    naive_by = {r["action"]: r for r in naive} if naive else {}
    labels, gpu, fp16, int8, int8_naive = [], [], [], [], []
    for model, res in pairs:
        hi, lo = by_action[f"{model}_{res}_fp32"], by_action[f"{model}_{res}_fp16"]
        labels.append(f"{model.replace('yolo11', 'yolo11-')} {res}")
        gpu.append(float(hi["f1_gpu"]))
        fp16.append(float(hi["f1_board"]))
        int8.append(float(lo["f1_board"]))
        if naive_by:
            int8_naive.append(float(naive_by[f"{model}_{res}_fp16"]["f1_board"]))
    fig, ax = plt.subplots(figsize=(10, 5.8))
    y = list(range(len(labels)))
    series = [
        (fp16, BLUE, "D", "NPU fp16"),
        (int8, AQUA, "s", "NPU int8, head decoded on CPU (final)"),
    ]
    if int8_naive:
        series.append((int8_naive, ORANGE, "X", "NPU int8, head quantized (first attempt)"))
    for values, color, marker, label in series:
        ax.scatter(
            values,
            y,
            s=70,
            color=color,
            marker=marker,
            label=label,
            zorder=3,
            edgecolor=SURFACE,
            linewidth=1.5,
        )
    ax.scatter(
        gpu,
        y,
        s=170,
        facecolor="none",
        edgecolor=INK_MUTED,
        linewidth=1.8,
        zorder=4,
        label="GPU detector (training reference)",
    )
    ax.set_yticks(y, labels)
    ax.set_xlabel("Detection F1 vs UA-DETRAC ground truth (IoU 0.5), 3 test sequences")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=10)
    ax.set_title("On-device accuracy matches the training detector once int8 is fixed")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_frontier(profile: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.8))
    points = [(float(r["e2e_ms_p50"]), float(r["f1_board"])) for r in profile]
    for model, color in MODEL_COLORS.items():
        for prec, marker in PRECISION_MARKERS.items():
            pts = [
                r for r in profile if r["action"].startswith(model) and r["board_precision"] == prec
            ]
            ax.scatter(
                [float(r["e2e_ms_p50"]) for r in pts],
                [float(r["f1_board"]) for r in pts],
                s=90,
                color=color,
                marker=marker,
                edgecolor=SURFACE,
                linewidth=1.5,
                zorder=3,
                label=f"{model.replace('yolo11', 'yolo11-')} {prec}",
            )
            for r in pts:
                xy = (float(r["e2e_ms_p50"]), float(r["f1_board"]))
                near = [
                    (px, py)
                    for px, py in points
                    if (px, py) != xy and abs(px / xy[0] - 1) < 0.25 and abs(py - xy[1]) < 0.006
                ]
                if any(px > xy[0] for px, _ in near):
                    offset = (-28, 4)  # neighbour on the right: label to the left
                elif near:
                    offset = (7, -13)  # neighbour on the left: label below-right
                else:
                    offset = (7, 4)
                ax.annotate(
                    r["action"].split("_")[1],
                    xy,
                    textcoords="offset points",
                    xytext=offset,
                    fontsize=9,
                    color=INK_MUTED,
                )
    best = min(profile, key=lambda r: float(r["e2e_ms_p50"]))
    ax.annotate(
        "fastest and among the most accurate",
        (float(best["e2e_ms_p50"]), float(best["f1_board"])),
        textcoords="offset points",
        xytext=(25, -55),
        fontsize=10,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED},
    )
    log_ms_axis(ax)
    ax.set_xlabel("Measured per-frame latency on ROCK 5C, ms (log scale)")
    ax.set_ylabel("Detection F1 on the NPU")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=10)
    ax.set_title("UA-DETRAC: a bigger model or resolution never buys accuracy")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_deadline(eval_csvs: list[Path], out: Path) -> None:
    """Accuracy against deadline-miss rate, frame-weighted over the test sequences."""
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for path in eval_csvs:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                t, steps = totals[row["policy"]], float(row["steps"])
                t[0] += float(row["mean_quality"]) * steps
                t[1] += float(row["deadline_misses"])
                t[2] += steps
    points = {name: (100.0 * miss / n, f1 / n) for name, (f1, miss, n) in totals.items()}

    fig, ax = plt.subplots(figsize=(11, 6))
    for name, (miss, f1) in points.items():
        if name.startswith("always_"):
            _, model, res, prec = name.split("_")
            board = "int8" if prec == "fp16" else "fp16"
            ax.scatter(miss, f1, s=70, color=CONTEXT, edgecolor=SURFACE, linewidth=1.5, zorder=2)
            if prec == "fp16":
                ax.annotate(f"{model[6:]}{res} {board}", (miss, f1), xytext=(6, -4),
                            textcoords="offset points", fontsize=9, color=INK_MUTED)
        elif name.startswith("deadline_lookup"):
            ax.scatter(miss, f1, s=110, marker="D", color=CONTEXT_STRONG, edgecolor=SURFACE,
                       linewidth=1.5, zorder=3)
    for algo, color in (("dqn", BLUE), ("ppo", ORANGE)):
        seeds = [p for n, p in points.items() if n.startswith(f"ivr_{algo}") and "masked" not in n]
        if seeds:
            ax.scatter(*zip(*seeds), s=120, color=color, edgecolor=SURFACE, linewidth=1.5,
                       zorder=4, label=f"IVR-{algo.upper()} ({len(seeds)} seeds)")
    ax.scatter([], [], s=110, marker="D", color=CONTEXT_STRONG, label="hand-built deadline lookup")
    ax.scatter([], [], s=70, color=CONTEXT, label="fixed config (board precision)")
    ax.set_xlabel("Frames that missed the latency deadline (%)")
    ax.set_ylabel("Detection F1, official UA-DETRAC scoring")
    ax.set_xlim(-4, 104)
    ax.legend(loc="lower right")
    ax.set_title("Deadline changes every 2-6 s: only adaptive policies are both accurate and on time")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--naive-int8-profile",
        default=None,
        help="board profile CSV from the head-quantized int8 models",
    )
    args = parser.parse_args()

    out = ROOT / "output"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart_final(
        [
            out / "test_detrac_gt_full_eval_output.csv",
            out / "test_detrac_gt_dqn_seeds.csv",
            out / "test_detrac_gt_ppo_seeds.csv",
        ],
        OUT_DIR / "chart_final_gt.png",
        "UA-DETRAC, all 40 test sequences, real IoU-F1 reward (simulated latency)",
    )
    profile = read_profile(out / "rknn_board_profile.csv")
    naive = read_profile(Path(args.naive_int8_profile)) if args.naive_int8_profile else None
    chart_latency(
        profile, out / "test_detrac_gt_full_eval_output.csv", OUT_DIR / "chart_board_latency.png"
    )
    chart_f1(profile, naive, OUT_DIR / "chart_board_f1.png")
    chart_frontier(profile, OUT_DIR / "chart_board_frontier.png")
    board_eval = [out / "test_detrac_gt_board_fixed_eval.csv", out / "test_detrac_gt_board_rl.csv"]
    if all(p.exists() for p in board_eval):
        chart_final(
            board_eval,
            OUT_DIR / "chart_final_board.png",
            "UA-DETRAC, all 40 test sequences, measured ROCK 5C latency",
        )
    deadline_eval = [
        p
        for p in (
            out / "test_detrac_deadline_fixed_eval.csv",
            out / "test_detrac_deadline_lookup_eval.csv",
            out / "test_detrac_deadline_dqn_eval.csv",
            out / "test_detrac_deadline_ppo_short_eval.csv",  # PPO at gamma 0.5
        )
        if p.exists()
    ]
    if deadline_eval:
        chart_deadline(deadline_eval, OUT_DIR / "chart_deadline_tradeoff.png")
        chart_final(
            deadline_eval,
            OUT_DIR / "chart_deadline_reward.png",
            "Time-varying deadline, 40 test sequences, measured ROCK 5C latency",
        )
    print(f"charts written to {OUT_DIR}")


if __name__ == "__main__":
    main()
