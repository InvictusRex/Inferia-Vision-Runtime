"""Prepare everything the ROCK 5C needs for RKNN conversion and benchmarking.

Exports one fixed-shape ONNX graph per (model, resolution) -- RKNN models are
static-shape, so precision is chosen later at conversion time -- and stages
two UA-DETRAC image sets next to them:

- calib/  frames sampled from training sequences, for int8 quantization
- eval/   every Nth frame of a few test sequences spanning sparse to dense
          traffic, for on-device latency and detection-quality measurement

    python tooling_scripts/rknn_export_onnx.py --out ../rknn_bundle
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime.environment_factory import (  # noqa: E402
    detrac_sequence_dirs,
    detrac_xml_dir,
    load_yaml,
)
from vision_pipeline.detrac_source import load_detrac_annotations  # noqa: E402

ONNX_OPSET = 12  # highest opset RKNN-Toolkit2 handles without op fallbacks for YOLO11


def export_onnx(variants_cfg: dict, out_dir: Path) -> None:
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = ROOT / variants_cfg.get("weights_dir", "weights")
    for m in variants_cfg["models"]:
        for res in variants_cfg["resolutions"]:
            target = out_dir / f"{m['name']}_{res}.onnx"
            if target.exists():
                continue
            # ultralytics writes next to the weights file, so export from a scratch copy
            scratch = out_dir / f"_{m['name']}_{res}.pt"
            shutil.copy(weights_dir / m["file"], scratch)
            exported = YOLO(str(scratch)).export(
                format="onnx", imgsz=res, opset=ONNX_OPSET, simplify=True, dynamic=False
            )
            Path(exported).replace(target)
            scratch.unlink()
            print(f"exported {target.name}", flush=True)


def mean_gt_count(dataset_dir: str, seq_dir: str) -> float:
    anns = load_detrac_annotations(
        str(detrac_xml_dir(dataset_dir, "test") / f"{Path(seq_dir).name}.xml")
    )
    return sum(len(b) for b in anns.values()) / max(1, len(anns))


def stage_images(
    dataset_dir: str, out_dir: Path, n_calib: int, eval_seqs: int, eval_step: int
) -> None:
    rng = random.Random(0)

    calib_dir = out_dir / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    train_frames = [
        f
        for seq in detrac_sequence_dirs(dataset_dir, "train")
        for f in sorted(Path(seq).glob("img*.jpg"))
    ]
    for f in rng.sample(train_frames, n_calib):
        shutil.copy(f, calib_dir / f"{f.parent.name}_{f.name}")
    print(f"calib: {n_calib} training frames", flush=True)

    # test sequences at evenly spaced density quantiles: sparse and dense traffic both covered
    test_seqs = sorted(
        detrac_sequence_dirs(dataset_dir, "test"), key=lambda s: mean_gt_count(dataset_dir, s)
    )
    picks = [test_seqs[round(i * (len(test_seqs) - 1) / (eval_seqs - 1))] for i in range(eval_seqs)]
    for seq in picks:
        seq_out = out_dir / "eval" / Path(seq).name
        seq_out.mkdir(parents=True, exist_ok=True)
        frames = sorted(Path(seq).glob("img*.jpg"))[::eval_step]
        for f in frames:
            shutil.copy(f, seq_out / f.name)
        print(
            f"eval: {Path(seq).name} mean_gt={mean_gt_count(dataset_dir, seq):.1f} "
            f"frames={len(frames)}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="bundle directory to copy to the board")
    parser.add_argument("--env", default="configs/env/env_detrac.yaml")
    parser.add_argument("--variants", default="configs/variants.yaml")
    parser.add_argument("--n-calib", type=int, default=150)
    parser.add_argument("--eval-seqs", type=int, default=3)
    parser.add_argument("--eval-step", type=int, default=5)
    args = parser.parse_args()

    out = Path(args.out)
    export_onnx(load_yaml(args.variants), out / "onnx")
    stage_images(
        load_yaml(args.env)["dataset_dir"], out, args.n_calib, args.eval_seqs, args.eval_step
    )


if __name__ == "__main__":
    main()
