from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelVariant:
    name: str
    weights: str
    input_size: int = 640
    precision: str = "fp32"
    gflops: float = 0.0


def build_variants(cfg: dict) -> dict[str, ModelVariant]:
    variants: dict[str, ModelVariant] = {}
    for item in cfg.get("models", []):
        name = item["name"]
        variants[name] = ModelVariant(
            name=name,
            weights=str(item["file"]),
            input_size=int(item.get("input_size", 640)),
            precision=str(item.get("precision", "fp32")),
            gflops=float(item.get("gflops", 0.0)),
        )
    return variants
