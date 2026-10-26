"""On-device measurement of every runtime configuration on the RK3588 NPU.

Runs on the ROCK 5C against the RKNN models produced by rknn_convert.py, using
only numpy, OpenCV and RKNN-Toolkit-Lite2 so it needs none of the training stack.

latency mode -- for each of the 18 configs: cold load time, resident memory,
    NPU-only and end-to-end latency (letterbox + NPU + decode/NMS) over real
    UA-DETRAC frames, per-core NPU load and NPU/SoC temperature, with a
    cool-down between configs so one config's heat does not bias the next.

detect mode -- runs every config over the staged evaluation sequences and
    saves raw detections in original-frame coordinates, to be scored against
    ground truth on the host with the same F1 matching the reward uses.

Clocks should be pinned first (perf_mode.sh). NPU load is read from debugfs,
which needs root, so run under sudo for the full set of measurements:

    sudo ~/ivr/venv/bin/python rknn_benchmark.py --bundle ~/ivr/bundle --mode latency --cores 3
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite

# ultralytics predict() defaults, so on-device detections match the training-time detector
CONF_THRESHOLD = 0.25
NMS_IOU = 0.7
MAX_DET = 300
PAD_VALUE = 114
DFL_BINS = 16

CORE_MASKS = {1: RKNNLite.NPU_CORE_0, 3: RKNNLite.NPU_CORE_0_1_2}
NPU_LOAD_PATH = Path("/sys/kernel/debug/rknpu/load")


def thermal_zone(kind: str) -> Path:
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        if (zone / "type").read_text().strip() == kind:
            return zone / "temp"
    raise FileNotFoundError(f"no thermal zone {kind}")


NPU_TEMP = thermal_zone("npu-thermal")
SOC_TEMP = thermal_zone("soc-thermal")


def read_temp(path: Path) -> float:
    return int(path.read_text()) / 1000.0


def read_npu_load() -> list[int] | None:
    try:
        return [int(v) for v in re.findall(r"Core\d+:\s*(\d+)%", NPU_LOAD_PATH.read_text())]
    except OSError:
        return None


def rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    raise RuntimeError("VmRSS missing from /proc/self/status")


def mem_available_mb() -> float:
    """System-wide available memory. The NPU allocates from the same unified LPDDR4X
    as the CPU, so this captures NPU buffers that process RSS does not."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024.0
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = round(w * scale), round(h * scale)
    pad_x, pad_y = (size - nw) / 2, (size - nh) / 2
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    top, left = round(pad_y - 0.1), round(pad_x - 0.1)
    canvas[top : top + nh, left : left + nw] = cv2.resize(
        frame, (nw, nh), interpolation=cv2.INTER_LINEAR
    )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None], scale, left, top


def infer(lite: RKNNLite, inp: np.ndarray) -> np.ndarray:
    """Run the model and decode its raw per-scale head outputs on the CPU.

    Returns the YOLO head layout (1, 4 + n_classes, K) -- cx, cy, w, h in input pixels,
    then class probabilities -- for the K anchors whose best score clears the threshold.
    """
    outs = lite.inference(inputs=[inp])
    size = inp.shape[1]
    boxes = {o.shape[2]: o for o in outs if o.shape[1] == 4 * DFL_BINS}
    logits = {o.shape[2]: o for o in outs if o.shape[1] != 4 * DFL_BINS}
    decoded = []
    for h in sorted(boxes, reverse=True):
        stride = size / h
        cls = 1.0 / (1.0 + np.exp(-logits[h][0].reshape(logits[h].shape[1], -1)))
        keep = np.flatnonzero(cls.max(0) > CONF_THRESHOLD)
        if keep.size == 0:
            continue
        dist = boxes[h][0].reshape(4, DFL_BINS, -1)[:, :, keep]
        dist = np.exp(dist - dist.max(1, keepdims=True))
        dist = (dist / dist.sum(1, keepdims=True) * np.arange(DFL_BINS)[None, :, None]).sum(1)
        ax, ay = keep % h + 0.5, keep // h + 0.5
        x1, y1 = (ax - dist[0]) * stride, (ay - dist[1]) * stride
        x2, y2 = (ax + dist[2]) * stride, (ay + dist[3]) * stride
        xywh = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        decoded.append(np.concatenate([xywh, cls[:, keep]]))
    if not decoded:
        return np.zeros((1, 4 + len(next(iter(logits.values()))[0]), 0), dtype=np.float32)
    return np.concatenate(decoded, axis=1)[None].astype(np.float32)


def postprocess(out: np.ndarray, scale: float, left: float, top: float) -> np.ndarray:
    """(1, 4 + n_classes, anchors) -> rows of [x1, y1, x2, y2, conf, cls] in frame coordinates."""
    pred = out[0]
    scores = pred[4:]
    cls = scores.argmax(0)
    conf = scores[cls, np.arange(scores.shape[1])]
    keep = conf > CONF_THRESHOLD
    if not keep.any():
        return np.zeros((0, 6), dtype=np.float32)
    cx, cy, bw, bh = pred[:4, keep]
    conf, cls = conf[keep], cls[keep]
    xywh = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1)
    idx = cv2.dnn.NMSBoxesBatched(
        xywh.tolist(), conf.tolist(), cls.tolist(), CONF_THRESHOLD, NMS_IOU
    )
    idx = np.asarray(idx, dtype=int).reshape(-1)
    idx = idx[np.argsort(-conf[idx])][:MAX_DET]
    x1 = (xywh[idx, 0] - left) / scale
    y1 = (xywh[idx, 1] - top) / scale
    x2 = x1 + xywh[idx, 2] / scale
    y2 = y1 + xywh[idx, 3] / scale
    return np.stack([x1, y1, x2, y2, conf[idx], cls[idx]], 1).astype(np.float32)


def parse_name(path: Path) -> tuple[str, int, str]:
    model, res, precision = path.stem.split("_")
    return model, int(res), precision


def open_runtime(path: Path, cores: int) -> RKNNLite:
    lite = RKNNLite(verbose=False)
    if lite.load_rknn(str(path)) != 0:
        raise RuntimeError(f"load_rknn failed: {path}")
    if lite.init_runtime(core_mask=CORE_MASKS[cores]) != 0:
        raise RuntimeError(f"init_runtime failed: {path}")
    return lite


def cool_down(target_c: float, timeout_s: float = 120.0) -> None:
    t_end = time.time() + timeout_s
    while read_temp(NPU_TEMP) > target_c and time.time() < t_end:
        time.sleep(2)


def load_frames(eval_dir: Path) -> list[np.ndarray]:
    return [cv2.imread(str(p)) for p in sorted(eval_dir.glob("*/*.jpg"))]


def bench_latency(
    models: list[Path], frames: list[np.ndarray], cores: int, runs: int, out_csv: Path
) -> None:
    baseline_c = read_temp(NPU_TEMP)
    rows = []
    for path in models:
        model, res, precision = parse_name(path)
        cool_down(baseline_c + 2.0)
        rss0, avail0 = rss_mb(), mem_available_mb()
        t0 = time.perf_counter()
        lite = open_runtime(path, cores)
        load_ms = (time.perf_counter() - t0) * 1000.0
        for frame in frames[:20]:
            inp, *_ = letterbox(frame, res)
            infer(lite, inp)
        rss_delta = rss_mb() - rss0
        mem_delta = avail0 - mem_available_mb()

        temp_start = read_temp(NPU_TEMP)
        pre, npu, post, e2e, loads = [], [], [], [], []
        npu_peak = soc_peak = temp_start
        for i in range(runs):
            frame = frames[i % len(frames)]
            t0 = time.perf_counter()
            inp, scale, left, top = letterbox(frame, res)
            t1 = time.perf_counter()
            out = infer(lite, inp)
            t2 = time.perf_counter()
            postprocess(out, scale, left, top)
            t3 = time.perf_counter()
            pre.append((t1 - t0) * 1000.0)
            npu.append((t2 - t1) * 1000.0)
            post.append((t3 - t2) * 1000.0)
            e2e.append((t3 - t0) * 1000.0)
            if i % 10 == 0:
                load = read_npu_load()
                if load is not None:
                    loads.append(load[:cores])
                npu_peak = max(npu_peak, read_temp(NPU_TEMP))
                soc_peak = max(soc_peak, read_temp(SOC_TEMP))
        lite.release()

        row = {
            "model": model,
            "resolution": res,
            "precision": precision,
            "cores": cores,
            "file_mb": round(path.stat().st_size / 1e6, 2),
            "load_ms": round(load_ms, 1),
            "rss_delta_mb": round(rss_delta, 1),
            "mem_delta_mb": round(mem_delta, 1),
            "pre_ms_p50": round(float(np.median(pre)), 2),
            "npu_ms_p50": round(float(np.median(npu)), 2),
            "npu_ms_p95": round(float(np.percentile(npu, 95)), 2),
            "post_ms_p50": round(float(np.median(post)), 2),
            "e2e_ms_p50": round(float(np.median(e2e)), 2),
            "e2e_ms_p95": round(float(np.percentile(e2e, 95)), 2),
            "e2e_fps": round(1000.0 / float(np.median(e2e)), 1),
            "npu_load_pct": round(float(np.mean(loads)), 1) if loads else "",
            "npu_temp_start_c": temp_start,
            "npu_temp_peak_c": npu_peak,
            "soc_temp_peak_c": soc_peak,
        }
        rows.append(row)
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"written {out_csv}", flush=True)


def dump_detections(models: list[Path], eval_dir: Path, cores: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seqs = sorted(p for p in eval_dir.iterdir() if p.is_dir())
    for path in models:
        target = out_dir / f"{path.stem}.npz"
        if target.exists():
            continue
        _, res, _ = parse_name(path)
        lite = open_runtime(path, cores)
        per_seq = {}
        for seq in seqs:
            rows = []
            for img in sorted(seq.glob("img*.jpg")):
                frame_num = int(img.stem[3:])
                inp, scale, left, top = letterbox(cv2.imread(str(img)), res)
                dets = postprocess(infer(lite, inp), scale, left, top)
                rows.append(np.column_stack([np.full(len(dets), frame_num, np.float32), dets]))
            per_seq[seq.name] = np.concatenate(rows) if rows else np.zeros((0, 7), np.float32)
        lite.release()
        np.savez_compressed(target, **per_seq)
        print(
            f"{target.name}: " + ", ".join(f"{k}={len(v)}" for k, v in per_seq.items()), flush=True
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--mode", choices=["latency", "detect"], required=True)
    parser.add_argument("--cores", type=int, choices=sorted(CORE_MASKS), default=3)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()

    bundle = Path(args.bundle).expanduser()
    models = sorted((bundle / "rknn").glob("*.rknn"))
    if not models:
        raise SystemExit(f"no .rknn models under {bundle / 'rknn'}")
    if args.mode == "latency":
        if read_npu_load() is None:
            print(
                "warning: NPU load unreadable (needs root + debugfs); column left empty", flush=True
            )
        frames = load_frames(bundle / "eval")
        bench_latency(
            models, frames, args.cores, args.runs, bundle / f"latency_{args.cores}core.csv"
        )
    else:
        dump_detections(models, bundle / "eval", args.cores, bundle / "detections")


if __name__ == "__main__":
    main()
