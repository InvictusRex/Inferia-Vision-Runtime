from __future__ import annotations

import argparse
import glob
import random
import statistics
from pathlib import Path

import cv2
import numpy as np
import yaml

from model_management.runtime_config_space import RuntimeConfig
from vision_pipeline.yolo_detector import Detector


def main():
    parser = argparse.ArgumentParser(
        description="Survey object density/confidence over a video set (yolo11n @640 fp32)"
    )
    parser.add_argument("--variants", default="configs/variants.yaml")
    parser.add_argument(
        "--dataset-dir",
        default="../BDDA/BDDA/training/camera_videos",
        help="folder of .mp4 files to sample",
    )
    parser.add_argument("--n-videos", type=int, default=40)
    parser.add_argument("--frames-per-video", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.variants, encoding="utf-8") as fh:
        variants_cfg = yaml.safe_load(fh) or {}
    model = "yolo11n"
    weight = None
    for item in variants_cfg.get("models", []):
        if item["name"] == model:
            weight = str(Path(variants_cfg.get("weights_dir", "weights")) / item["file"])
    if not weight:
        raise SystemExit(f"model {model} not found in {args.variants}")

    paths = sorted(glob.glob(str(Path(args.dataset_dir) / "*.mp4")))
    if not paths:
        raise SystemExit(f"no videos found in {args.dataset_dir}")
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    paths = paths[: args.n_videos]

    detector = Detector(weight_paths={model: weight}, conf=0.25, device="cuda")
    cfg = RuntimeConfig(model=model, resolution=640, precision="fp32")
    step = 5

    counts: list[int] = []
    confs: list[float] = []
    for v in paths:
        cap = cv2.VideoCapture(v)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        span = min(args.frames_per_video * step, max(1, n))
        cap.set(cv2.CAP_PROP_POS_FRAMES, rng.randint(0, max(0, n - span)))
        i = 0
        vc: list[int] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                det = detector.detect(frame, cfg)
                c = len(det)
                counts.append(c)
                if c:
                    confs.append(float(det.confs.mean()))
                vc.append(c)
            i += 1
        cap.release()
        print(f"  {Path(v).name}: median_count={statistics.median(vc):.0f}")

    counts_arr = np.array(counts, dtype=float)
    confs_arr = np.array(confs, dtype=float)
    print("\n== summary (yolo11n @640 fp32) ==")
    print(f"frames surveyed: {len(counts_arr)}")
    for p in (10, 25, 50, 75, 90, 95, 99):
        print(f"  count p{p:02d}: {int(np.percentile(counts_arr, p))}")
    print(f"  count mean: {counts_arr.mean():.1f}  std: {counts_arr.std():.1f}")
    print(f"  conf mean: {confs_arr.mean():.3f}  median: {np.median(confs_arr):.3f}")
    print(f"  fraction frames with 0 objects: {(counts_arr == 0).mean():.3f}")
    for lo, hi in ((0, 8), (8, 16), (16, 30), (30, 1000)):
        mask = (counts_arr >= lo) & (counts_arr < hi)
        print(f"  count in [{lo},{hi}): {(mask.mean() * 100):.1f}%")


if __name__ == "__main__":
    main()
