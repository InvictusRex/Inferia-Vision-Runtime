"""Closed-loop IVR on the ROCK 5C: a trained policy picks the runtime config each frame
and the NPU runs it.

Same loop as VisionRuntimeEnv -- the observation of frame t chooses the config used on
frame t+1 -- built from the repository's own scene analyzer, observation builder,
config space and edge profile, but with the RKNN detectors on the NPU in place of the
PyTorch detector and with real measured latency feeding the observation. Writes a
per-frame trace and an annotated video.

For a deadline-aware policy (observation extended by the deadline levels) the per-frame latency deadline is set
live: keys 1-5 pick a level, A replays the training-style random schedule. With --show
the annotated stream, a status panel and a latency/deadline timeline open in a window
on the board's desktop (viewable over VNC).

Run on the board, next to rknn_benchmark.py, with a copy of the repository:

    python rknn_live_demo.py --repo ~/ivr/repo --bundle ~/ivr/bundle \\
        --policy ~/ivr/policies/dqn_detrac_deadline_s0_final.zip \\
        --sequence MVI_40714 MVI_40864 --show --loop
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

# torch (via SB3) must be imported before RKNN-Toolkit-Lite2, which rewrites the logging
# level table that torch reads at import time
from stable_baselines3 import DQN, PPO  # isort: skip
from rknn_benchmark import NPU_TEMP, infer, letterbox, open_runtime, postprocess, read_temp

VEHICLE_CLASSES = (2, 5, 7)  # COCO car, bus, truck: what the official scoring counts
# one colour per configuration: the model picks the hue family (n green, s orange, m purple),
# the board precision shifts the hue within it and the resolution sets the shade (480 light, 960 deep)
MODEL_HUE = {"yolo11n": 60, "yolo11s": 12, "yolo11m": 140}  # OpenCV hue, 0-180
PRECISION_HUE_SHIFT = {"int8": 0, "fp16": 14}
RESOLUTION_SV = {480: (140, 255), 640: (205, 235), 960: (255, 185)}
PANEL_W, STRIP_H = 440, 150
INK, MUTED, BG = (241, 245, 249), (148, 163, 184), (42, 23, 15)
GOOD, BAD = (94, 197, 34), (68, 68, 239)
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class BoardDetections:
    """The slice of vision_pipeline.Detections the scene analyzer reads."""

    boxes: np.ndarray
    confs: np.ndarray
    cls_ids: np.ndarray

    def __len__(self) -> int:
        return int(self.confs.shape[0])


def import_repo(repo: Path) -> None:
    sys.path.insert(0, str(repo))
    # register these as bare packages: their __init__ pulls in the PyTorch detector and
    # the gymnasium env, neither of which exists on the board
    for pkg in ("vision_pipeline", "reinforcement_learning"):
        module = types.ModuleType(pkg)
        module.__path__ = [str(repo / pkg)]
        sys.modules[pkg] = module


def _text(img, s, xy, scale=0.6, color=INK, thick=1):
    cv2.putText(img, s, xy, FONT, scale, color, thick, cv2.LINE_AA)


def config_color(model: str, resolution: int, board_precision: str) -> tuple[int, int, int]:
    """BGR colour of one runtime configuration."""
    hue = (MODEL_HUE[model] + PRECISION_HUE_SHIFT[board_precision]) % 180
    sat, val = RESOLUTION_SV[int(resolution)]
    b, g, r = cv2.cvtColor(np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def config_key(row: dict) -> tuple[str, int, str]:
    return row["model"], int(row["resolution"]), row["board_precision"]


def config_label(key: tuple[str, int, str]) -> str:
    return f"{key[0][-1]}{key[1]} {key[2]}"


def draw_frame(
    frame: np.ndarray, dets: np.ndarray, history: list[dict], state: dict, deadline_levels: list[float]
) -> np.ndarray:
    """Annotated frame + status panel (right) + latency/deadline timeline (bottom)."""
    h, w = frame.shape[:2]
    canvas = np.zeros((h + STRIP_H, w + PANEL_W, 3), np.uint8)
    canvas[:] = BG
    view = frame.copy()
    for x1, y1, x2, y2 in dets[:, :4].astype(int):
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 220, 60), 2)
    x, cur = w + 20, history[-1]
    color = config_color(*config_key(cur))
    cv2.rectangle(view, (2, 2), (w - 3, h - 3), color, 5)  # frame border in the running config's colour
    canvas[:h, :w] = view

    _text(canvas, "INFERIA VISION RUNTIME", (x, 34), 0.75, INK, 2)
    _text(canvas, f"policy  {state['policy']}", (x, 60), 0.5, MUTED)
    _text(canvas, f"video   {state['video']}  frame {cur['frame']}", (x, 82), 0.5, MUTED)
    _text(canvas, "NOW RUNNING ON THE NPU", (x, 124), 0.5, MUTED)
    cv2.rectangle(canvas, (x, 136), (x + PANEL_W - 40, 176), color, -1)
    text_ink = (15, 23, 42) if sum(color) > 420 else INK
    _text(canvas, cur["model"].replace("yolo11", "YOLO11"), (x + 10, 166), 0.95, text_ink, 2)
    _text(canvas, f"{cur['resolution']} px  {cur['board_precision']}", (x + 185, 166), 0.8, text_ink, 2)

    y = 214
    _text(canvas, f"latency  {cur['latency_ms']:6.1f} ms", (x, y), 0.65, INK, 1)
    deadline = cur["deadline_ms"]
    if deadline is not None:
        met = cur["latency_ms"] <= deadline
        _text(canvas, f"deadline {deadline:6.0f} ms", (x, y + 28), 0.65, INK, 1)
        _text(canvas, "MET" if met else "MISSED", (x + 270, y + 14), 0.8, GOOD if met else BAD, 2)
        bar_w = PANEL_W - 40
        frac = min(1.0, cur["latency_ms"] / (2 * deadline))
        cv2.rectangle(canvas, (x, y + 40), (x + bar_w, y + 52), (71, 51, 30), -1)
        cv2.rectangle(canvas, (x, y + 40), (x + int(bar_w * frac), y + 52), GOOD if met else BAD, -1)
        cv2.line(canvas, (x + bar_w // 2, y + 36), (x + bar_w // 2, y + 56), INK, 2)
        mode = "auto schedule" if state["auto"] else "set by keys 1-5"
        _text(canvas, f"deadline {mode}", (x, y + 78), 0.5, MUTED)
    y = 318
    _text(canvas, f"loop rate   {state['fps']:5.1f} FPS", (x, y), 0.6)
    _text(canvas, f"vehicles    {cur['detections']}", (x, y + 26), 0.6)
    _text(canvas, f"NPU temp    {cur['npu_temp_c']:.1f} C", (x, y + 52), 0.6)
    _text(canvas, f"switches    {state['switches']}", (x, y + 78), 0.6)
    if deadline is not None:
        _text(canvas, f"deadlines met {100 * state['met'] / max(1, state['n']):5.1f}%", (x, y + 104), 0.6)
    _text(canvas, "1-5 deadline  A auto  N next video", (x, h - 30), 0.48, MUTED)
    _text(canvas, "P pause  Q quit", (x, h - 10), 0.48, MUTED)

    # timeline: per-frame latency (log scale, coloured by configuration) against the deadline,
    # with a marker on every switch
    x0, x1, y0, y1 = 60, w + PANEL_W - 20, h + 34, h + STRIP_H - 14
    lo, hi = np.log(10.0), np.log(600.0)

    def ypix(ms):
        return int(y1 - (np.log(max(ms, 10.0)) - lo) / (hi - lo) * (y1 - y0))

    for ms in (20, 50, 100, 250):
        cv2.line(canvas, (x0, ypix(ms)), (x1, ypix(ms)), (71, 51, 30), 1)
        _text(canvas, f"{ms}", (20, ypix(ms) + 4), 0.4, MUTED)
    span = history[-(x1 - x0) // 2:]
    for i, r in enumerate(span):
        px = x0 + 2 * i
        cv2.line(canvas, (px, y1), (px, ypix(r["latency_ms"])), config_color(*config_key(r)), 2)
        if i and config_key(r) != config_key(span[i - 1]):
            cv2.drawMarker(canvas, (px, y0 - 5), INK, cv2.MARKER_TRIANGLE_DOWN, 8, 2)
        if r["deadline_ms"] is not None and i:
            prev = span[i - 1]["deadline_ms"]
            cv2.line(canvas, (px - 2, ypix(prev)), (px, ypix(r["deadline_ms"])), INK, 2)
    # legend: every configuration visible in the timeline, fastest model first
    seen = sorted({config_key(r) for r in span}, key=lambda k: (k[0][-1] != "n", k[0][-1] == "m", k[1], k[2]))
    lx = x1 - 112 * len(seen)
    for key in seen:
        cv2.rectangle(canvas, (lx, h + 8), (lx + 14, h + 20), config_color(*key), -1)
        _text(canvas, config_label(key), (lx + 18, h + 19), 0.42, INK)
        lx += 112
    if deadline_levels:
        _text(canvas, "white line = deadline, marks = switch", (x0, h + 19), 0.42, INK)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--policy", required=True, help="SB3 DQN or PPO checkpoint zip")
    parser.add_argument("--sequence", required=True, nargs="+", help="folder(s) under <bundle>/demo")
    parser.add_argument("--hardware", default="configs/hardware_measured.yaml")
    parser.add_argument("--variants", default="configs/variants.yaml")
    parser.add_argument("--env", default="configs/env/env_detrac_deadline.yaml",
                        help="deadline levels and hold for a deadline-aware policy")
    parser.add_argument("--deadline-ms", type=float, default=None,
                        help="starting deadline; default is the auto schedule")
    parser.add_argument("--cores", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--show", action="store_true", help="open a live window")
    parser.add_argument("--fullscreen", action="store_true", help="with --show, fill the screen")
    parser.add_argument("--loop", action="store_true", help="restart the video at its end")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    repo, bundle = Path(args.repo).expanduser(), Path(args.bundle).expanduser()
    import_repo(repo)
    from model_management.model_variants import build_variants
    from model_management.runtime_config_space import ConfigSpace
    from reinforcement_learning.observation_features import FeatureBuilder
    from runtime.edge_profile import EdgeProfile
    from runtime.latency_estimator import LatencyEstimator
    from runtime.system_telemetry import GpuSnapshot, Telemetry
    from vision_pipeline.scene_analyzer import SceneAnalyzer

    variants_cfg = yaml.safe_load((repo / args.variants).read_text())
    hw_cfg = yaml.safe_load((repo / args.hardware).read_text())
    precision_map = hw_cfg["measured"]["precision_map"]

    variants = build_variants(variants_cfg)
    space = ConfigSpace(
        models=list(variants),
        resolutions=variants_cfg["resolutions"],
        precisions=variants_cfg["precisions"],
    )
    gflops = {name: v.gflops for name, v in variants.items()}
    edge = EdgeProfile.from_configs(
        hw_cfg, variants, LatencyEstimator.from_variants(variants, gflops)
    )
    policy_path = Path(args.policy).expanduser()
    policy = (PPO if "ppo" in policy_path.stem.lower() else DQN).load(
        policy_path,
        device="cpu",
        # training-only schedules are pickled with the host's Python; inference never uses them
        custom_objects={k: 0.0 for k in ("lr_schedule", "exploration_schedule", "clip_range")},
    )
    env_cfg = yaml.safe_load((repo / args.env).read_text())
    levels = [float(d) for d in env_cfg["deadline_levels_ms"]]
    if policy.observation_space.shape[0] == FeatureBuilder.DIM:
        levels = []  # a policy trained without a deadline
    deadline_aware = bool(levels)
    hold_lo, hold_hi = env_cfg.get("deadline_hold_frames", (50, 150))
    sim = edge.profile.simulated
    features = FeatureBuilder(
        n_actions=space.n_actions,
        power_budget_w=edge.profile.policy.power_budget_w,
        ambient_temp_c=sim.ambient_temp_c,
        throttle_temp_c=sim.thermal_throttle_temp_c,
        deadline_levels_ms=levels,
    )
    analyzer, telemetry = SceneAnalyzer(), Telemetry()

    runtimes: dict[tuple, object] = {}
    load_ms: dict[str, float] = {}

    def runtime_for(config):
        key = (config.model, config.resolution, precision_map[config.precision])
        if key not in runtimes:
            t0 = time.perf_counter()
            runtimes[key] = open_runtime(
                bundle / "rknn" / f"{key[0]}_{key[1]}_{key[2]}.rknn", args.cores
            )
            load_ms["_".join(map(str, key))] = (time.perf_counter() - t0) * 1000.0
        return runtimes[key]

    if args.show:
        # load every model up front so a live switch never stalls on a first load
        for a in range(space.n_actions):
            runtime_for(space.action_to_config(a))
        cv2.namedWindow("IVR", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        if args.fullscreen:
            cv2.setWindowProperty("IVR", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    out_dir = bundle / "demo_out"
    out_dir.mkdir(exist_ok=True)
    stem = f"{policy_path.stem}_{'_'.join(args.sequence)}"
    video = None
    rng = np.random.default_rng(0)
    auto = deadline_aware and args.deadline_ms is None
    deadline = args.deadline_ms if args.deadline_ms is not None else (levels[0] if levels else None)
    hold = 0

    rows, history = [], []
    state = {"policy": policy_path.stem, "auto": auto, "fps": 0.0, "switches": 0, "met": 0, "n": 0}
    action = 0  # VisionRuntimeEnv also runs its first frame on action 0
    seq_i, quit_ = 0, False
    while not quit_:
        seq = args.sequence[seq_i]
        frames = sorted((bundle / "demo" / seq).glob("img*.jpg"))[: args.max_frames]
        state["video"] = seq
        analyzer.reset()
        next_seq = False
        for path in frames:
            frame = cv2.imread(str(path))
            config = space.action_to_config(action)
            lite = runtime_for(config)

            t0 = time.perf_counter()
            inp, scale, left, top = letterbox(frame, config.resolution)
            dets = postprocess(infer(lite, inp), scale, left, top)
            t1 = time.perf_counter()
            detections = BoardDetections(dets[:, :4], dets[:, 4], dets[:, 5].astype(int))
            latency_ms = (t1 - t0) * 1000.0
            telemetry.record(latency_ms)
            snap = edge.snapshot(config)
            telemetry.record_gpu(
                GpuSnapshot(
                    npu_util_pct=snap.npu_util_pct,
                    mem_used_mb=snap.mem_used_mb,
                    mem_budget_mb=edge.memory_budget_mb,
                    temp_c=snap.temp_c,
                    power_w=snap.power_w,
                    latency_ms=latency_ms,
                )
            )
            frame_deadline = deadline  # the deadline this frame's config was chosen against
            if state["auto"]:
                hold -= 1
                if hold <= 0:
                    deadline = float(rng.choice(levels))
                    hold = int(rng.integers(hold_lo, hold_hi + 1))
            scene = analyzer.analyze(frame, detections)
            obs = features.build(scene, telemetry, action, deadline)
            next_action = int(policy.predict(obs, deterministic=True)[0])
            t2 = time.perf_counter()

            vehicles = dets[np.isin(dets[:, 5].astype(int), VEHICLE_CLASSES)]
            row = {
                "video": seq,
                "frame": int(path.stem[3:]),
                "model": config.model,
                "resolution": config.resolution,
                "precision": config.precision,
                "board_precision": precision_map[config.precision],
                "detections": len(vehicles),
                "latency_ms": round(latency_ms, 2),
                "loop_ms": round((t2 - t0) * 1000.0, 2),
                "deadline_ms": frame_deadline,
                "deadline_met": None if frame_deadline is None else int(latency_ms <= frame_deadline),
                "npu_temp_c": read_temp(NPU_TEMP),
                "switched": int(next_action != action),
            }
            rows.append(row)
            history.append(row)
            del history[:-2000]
            state["switches"] += row["switched"]
            if frame_deadline is not None:
                state["n"] += 1
                state["met"] += row["deadline_met"]
            state["fps"] = 0.9 * state["fps"] + 0.1 * 1000.0 / row["loop_ms"] if state["fps"] else 1000.0 / row["loop_ms"]
            action = next_action

            if args.show or not args.no_video:
                canvas = draw_frame(frame, vehicles, history, state, levels)
                if video is None and not args.no_video:
                    ch, cw = canvas.shape[:2]
                    video = cv2.VideoWriter(
                        str(out_dir / f"{stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (cw, ch)
                    )
                if video is not None:
                    video.write(canvas)
            if args.show:
                if len(rows) == 1 and not args.fullscreen:
                    cv2.resizeWindow("IVR", canvas.shape[1], canvas.shape[0])
                cv2.imshow("IVR", canvas)
                key = cv2.waitKey(1) & 0xFF
                while key == ord("p"):
                    key = cv2.waitKey(0) & 0xFF
                    key = 0 if key == ord("p") else key
                if ord("1") <= key < ord("1") + len(levels):
                    deadline, state["auto"] = levels[key - ord("1")], False
                elif key == ord("a") and levels:
                    state["auto"], hold = True, 0
                elif key == ord("n"):
                    next_seq = True
                    break
                elif key in (ord("q"), 27):
                    quit_ = True
                    break
        if quit_:
            break
        seq_i += 1
        if next_seq or args.loop:
            seq_i %= len(args.sequence)
        elif seq_i >= len(args.sequence):
            break

    if video is not None:
        video.release()
    for lite in runtimes.values():
        lite.release()
    cv2.destroyAllWindows()

    with open(out_dir / f"{stem}.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    loop = np.array([r["loop_ms"] for r in rows])
    configs = {}
    for r in rows:
        key = f"{r['model']}/{r['resolution']}/{r['board_precision']}"
        configs[key] = configs.get(key, 0) + 1
    print(
        f"frames={len(rows)} switches={sum(r['switched'] for r in rows)} "
        f"loop_ms_p50={np.median(loop):.1f} closed_loop_fps={1000.0 / np.median(loop):.1f}"
    )
    met = [r["deadline_met"] for r in rows if r["deadline_met"] is not None]
    if met:
        print(f"deadlines met: {100.0 * np.mean(met):.1f}% of {len(met)} frames")
    print("configs used: " + ", ".join(f"{k}={v}" for k, v in sorted(configs.items())))
    print("model load ms: " + ", ".join(f"{k}={v:.0f}" for k, v in load_ms.items()))
    print(f"written {out_dir / stem}.csv" + ("" if video is None else f" and {stem}.mp4"))


if __name__ == "__main__":
    main()
