"""Convert the exported ONNX graphs into RKNN models for the RK3588 NPU.

Runs on the ROCK 5C (or any host with RKNN-Toolkit2). Each <model>_<res>.onnx
becomes two RKNN models:

- fp16: no quantization, the highest precision the NPU executes
- int8: asymmetric quantization calibrated on the staged UA-DETRAC frames

Both are exported with the raw per-scale head outputs (see SPLIT_OUTPUTS).

The NPU has no fp32 path, so the runtime's fp32/fp16 action axis maps onto
these two on-device precisions.

    python rknn_convert.py --bundle ~/ivr/bundle
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from rknn.api import RKNN

TARGET_PLATFORM = "rk3588"
# The graph is cut where the YOLO11 head's per-scale convolutions end: box-distribution
# logits (64 ch) and class logits (80 ch) for strides 8/16/32. Quantizing the decoded head
# instead puts pixel-scale boxes, DFL softmax and 0..1 scores under shared int8 scales,
# which destroys localization; decoding these six tensors on the CPU keeps int8 usable.
SPLIT_OUTPUTS = [
    f"/model.23/{branch}.{i}/{branch}.{i}.2/Conv_output_0"
    for i in range(3)
    for branch in ("cv2", "cv3")
]


def convert(onnx_path: Path, out_path: Path, quantize: bool, dataset_txt: Path) -> None:
    rknn = RKNN(verbose=False)
    try:
        # the graph expects RGB in [0, 1]; the NPU folds this normalization into its input stage
        rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform=TARGET_PLATFORM,
            quantized_dtype="w8a8",
            quantized_algorithm="normal",
        )
        if rknn.load_onnx(str(onnx_path), outputs=SPLIT_OUTPUTS) != 0:
            raise RuntimeError(f"load_onnx failed: {onnx_path}")
        if (
            rknn.build(do_quantization=quantize, dataset=str(dataset_txt) if quantize else None)
            != 0
        ):
            raise RuntimeError(f"build failed: {onnx_path} quantize={quantize}")
        # export under a temporary name so an interrupted run never leaves a truncated
        # model that a rerun would skip as already converted
        partial = out_path.with_suffix(".partial")
        if rknn.export_rknn(str(partial)) != 0:
            raise RuntimeError(f"export failed: {out_path}")
        partial.replace(out_path)
    finally:
        rknn.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", required=True, help="directory holding onnx/ and calib/")
    args = parser.parse_args()

    bundle = Path(args.bundle).expanduser()
    out_dir = bundle / "rknn"
    out_dir.mkdir(exist_ok=True)
    dataset_txt = bundle / "calib.txt"
    dataset_txt.write_text(
        "\n".join(str(p) for p in sorted((bundle / "calib").glob("*.jpg"))) + "\n"
    )

    for onnx_path in sorted((bundle / "onnx").glob("*.onnx")):
        for precision, quantize in (("fp16", False), ("int8", True)):
            out_path = out_dir / f"{onnx_path.stem}_{precision}.rknn"
            if out_path.exists():
                continue
            t0 = time.time()
            convert(onnx_path, out_path, quantize, dataset_txt)
            print(
                f"{out_path.name}: {time.time() - t0:.0f}s, "
                f"{out_path.stat().st_size / 1e6:.1f} MB",
                flush=True,
            )


if __name__ == "__main__":
    main()
