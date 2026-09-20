from __future__ import annotations

import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    model: str
    resolution: int
    precision: str


class ConfigSpace:
    def __init__(
        self,
        models: list[str],
        resolutions: list[int] = (640,),
        precisions: list[str] = ("fp32",),
    ):
        if not models:
            raise ValueError("ConfigSpace requires at least one model")
        self._actions = list(itertools.product(models, resolutions, precisions))

    @property
    def n_actions(self) -> int:
        return len(self._actions)

    @property
    def models(self) -> list[str]:
        return sorted({a[0] for a in self._actions})

    def action_to_config(self, action: int) -> RuntimeConfig:
        model, resolution, precision = self._actions[int(action)]
        return RuntimeConfig(model=model, resolution=resolution, precision=precision)

    def config_to_action(self, config: RuntimeConfig) -> int:
        return self._actions.index((config.model, config.resolution, config.precision))
