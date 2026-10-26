"""Turn ROCK 5C benchmark output into a measured hardware profile.

Reads what rknn_benchmark.py produced on the board (latency CSVs per core mode
and raw detections), and:

1. scores the on-device detections against UA-DETRAC ground truth with the
   exact F1 matching the reward uses, next to the GPU detector's F1 on the same
   frames, so the int8/fp16 deployment gap is measured rather than assumed
2. maps each runtime action onto its on-device equivalent -- the NPU has no
   fp32, so action precision fp32 runs as NPU fp16 and fp16 runs as NPU int8 --
   picking the faster core mode per config
3. writes output/rknn_board_profile.csv and configs/hardware_measured.yaml
   (configs/hardware.yaml with its `measured:` section filled in)

    python tooling_scripts/rknn_measured_profile.py --bundle ../rknn_bundle
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_management.runtime_config_space import RuntimeConfig  # noqa: E402
from reinforcement_learning.reward_calculator import _detection_f1  # noqa: E402
from runtime.environment_factory import detrac_xml_dir, load_yaml  # noqa: E402
from vision_pipeline.detrac_source import load_detrac_annotations  # noqa: E402

# runtime action precision -> on-device RKNN precision
PRECISION_MAP = {"fp32": "fp16", "fp16": "int8"}


def eval_frames(bundle: Path) -> dict[str, list[Path]]:
    return {
        seq.name: sorted(seq.glob("img*.jpg"))
        for seq in sorted((bundle / "eval").iterdir())
        if seq.is_dir()
    }


def mean_f1(dets_by_frame: dict[int, np.ndarray], frames: list[Path], gt: dict) -> float:
    empty = np.zeros((0, 4), dtype=np.float32)
    scores = []
    for img in frames:
        num = int(img.stem[3:])
        scores.append(
            _detection_f1(dets_by_frame.get(num, empty), gt.get(num, empty), iou_threshold=0.5)
        )
    return float(np.mean(scores))


def board_f1(npz_path: Path, frames: dict[str, list[Path]], gts: dict) -> dict[str, float]:
    data = np.load(npz_path)
    out = {}
    for seq, imgs in frames.items():
        arr = data[seq]
        by_frame = {int(n): arr[arr[:, 0] == n, 1:5] for n in np.unique(arr[:, 0])}
        out[seq] = mean_f1(by_frame, imgs, gts[seq])
    return out


def gpu_f1(detector, config: RuntimeConfig, frames: dict[str, list[Path]], gts: dict) -> dict:
    out = {}
    for seq, imgs in frames.items():
        by_frame = {
            int(p.stem[3:]): detector.detect(cv2.imread(str(p)), config).boxes for p in imgs
        }
        out[seq] = mean_f1(by_frame, imgs, gts[seq])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", required=True, help="bundle dir pulled back from the board")
    parser.add_argument("--env", default="configs/env/env_detrac.yaml")
    parser.add_argument("--variants", default="configs/variants.yaml")
    parser.add_argument("--hardware", default="configs/hardware.yaml")
    parser.add_argument("--out-hardware", default="configs/hardware_measured.yaml")
    parser.add_argument("--skip-gpu", action="store_true", help="skip the GPU reference F1")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    env_cfg = load_yaml(args.env)
    variants_cfg = load_yaml(args.variants)
    hw_cfg = load_yaml(args.hardware)
    target_fps = float(hw_cfg.get("policy", {}).get("target_fps", 30.0))

    frames = eval_frames(bundle)
    gts = {
        seq: load_detrac_annotations(
            str(detrac_xml_dir(env_cfg["dataset_dir"], "test") / f"{seq}.xml")
        )
        for seq in frames
    }

    latency: dict[tuple[str, int, str], dict] = {}
    for path in sorted(bundle.glob("latency_*core.csv")):
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                key = (row["model"], int(row["resolution"]), row["precision"])
                if key not in latency or float(row["e2e_ms_p50"]) < float(
                    latency[key]["e2e_ms_p50"]
                ):
                    latency[key] = row

    detector = None
    if not args.skip_gpu:
        from runtime.system_telemetry import Telemetry
        from vision_pipeline.yolo_detector import Detector

        weights_dir = ROOT / variants_cfg.get("weights_dir", "weights")
        detector = Detector(
            {m["name"]: str(weights_dir / m["file"]) for m in variants_cfg["models"]},
            conf=float(env_cfg.get("conf_threshold", 0.25)),
            device=str(env_cfg.get("device", "cuda")),
            telemetry=Telemetry(),
        )

    rows, measured = [], {}
    for m in variants_cfg["models"]:
        for res in variants_cfg["resolutions"]:
            for action_prec in variants_cfg["precisions"]:
                model, board_prec = m["name"], PRECISION_MAP[action_prec]
                lat = latency[(model, res, board_prec)]
                f1_board = board_f1(
                    bundle / "detections" / f"{model}_{res}_{board_prec}.npz", frames, gts
                )
                row = {
                    "action": f"{model}_{res}_{action_prec}",
                    "board_precision": board_prec,
                    "cores": int(lat["cores"]),
                    "e2e_ms_p50": float(lat["e2e_ms_p50"]),
                    "npu_ms_p50": float(lat["npu_ms_p50"]),
                    "e2e_fps": float(lat["e2e_fps"]),
                    "mem_delta_mb": float(lat["mem_delta_mb"]),
                    "npu_temp_peak_c": float(lat["npu_temp_peak_c"]),
                    "npu_load_pct": lat["npu_load_pct"],
                    "f1_board": round(float(np.mean(list(f1_board.values()))), 4),
                    **{f"f1_board_{seq}": round(v, 4) for seq, v in f1_board.items()},
                }
                if detector is not None:
                    f1_gpu = gpu_f1(detector, RuntimeConfig(model, res, action_prec), frames, gts)
                    row["f1_gpu"] = round(float(np.mean(list(f1_gpu.values()))), 4)
                    row["f1_gap"] = round(row["f1_board"] - row["f1_gpu"], 4)
                rows.append(row)
                print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)

                measured[row["action"]] = {
                    "latency_ms": row["e2e_ms_p50"],
                    # NPU busy share of each second when sustaining the target frame rate;
                    # same compute-pressure meaning as the simulated value, >100% means the
                    # config cannot hold target_fps
                    "npu_util_pct": round(100.0 * row["npu_ms_p50"] * target_fps / 1000.0, 1),
                    "temp_c": row["npu_temp_peak_c"],
                }

    out_csv = ROOT / "output" / "rknn_board_profile.csv"
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"written {out_csv}", flush=True)

    hw_cfg["measured"] = {
        "source": "ROCK 5C RK3588S2 NPU at 1 GHz, RKNN 2.3.0, tooling_scripts/rknn_benchmark.py",
        "precision_map": dict(PRECISION_MAP),
        "configs": measured,
    }
    header = (
        "# ROCK 5C profile with `measured:` filled from on-device benchmarks.\n"
        "# Latency, NPU utilisation and temperature per config are measured. Power has no\n"
        "# on-board sensor and the unified-memory delta is too noisy at these model sizes\n"
        "# (see output/rknn_board_profile.csv), so both stay simulated. Generated by\n"
        "# tooling_scripts/rknn_measured_profile.py -- edit the benchmark, not this file.\n"
    )
    out_hw = ROOT / args.out_hardware
    out_hw.write_text(header + yaml.safe_dump(hw_cfg, sort_keys=False), encoding="utf-8")
    print(f"written {out_hw}", flush=True)


if __name__ == "__main__":
    main()
