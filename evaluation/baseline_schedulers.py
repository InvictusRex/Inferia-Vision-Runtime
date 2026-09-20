from __future__ import annotations

import numpy as np

from model_management.runtime_config_space import RuntimeConfig
from reinforcement_learning.vision_runtime_env import VisionRuntimeEnv


class Policy:
    name = "policy"

    def __init__(self, n_actions: int):
        self.n_actions = int(n_actions)

    def choose(self, obs: np.ndarray, info: dict) -> int:
        raise NotImplementedError

    def update(self, obs: np.ndarray, info: dict, reward: float, action: int) -> None:
        pass

    def reset(self) -> None:
        """Reset any per-episode/per-video learning state. No-op for fixed policies."""
        pass


class AlwaysModel(Policy):
    def __init__(self, n_actions: int, action: int):
        super().__init__(n_actions)
        self.name = f"always_{action}"
        self.action = int(action)

    def choose(self, obs: np.ndarray, info: dict) -> int:
        return self.action


class AlwaysConfig(Policy):
    """Always run a specific runtime config (model x resolution x precision)."""

    def __init__(self, env: VisionRuntimeEnv, model: str, resolution: int, precision: str):
        super().__init__(env.action_space.n)
        self.action = env.config_space.config_to_action(
            RuntimeConfig(model=model, resolution=resolution, precision=precision)
        )
        self.name = f"always_{model}_{resolution}_{precision}"

    def choose(self, obs: np.ndarray, info: dict) -> int:
        return self.action


class RandomPolicy(Policy):
    def __init__(self, n_actions: int, seed: int = 0):
        super().__init__(n_actions)
        self.name = "random"
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def choose(self, obs: np.ndarray, info: dict) -> int:
        return int(self._rng.integers(0, self.n_actions))

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)


class RuleBasedPolicy(Policy):
    """Object-count thresholds mapped to canonical configs (model x 640 x fp32)."""

    def __init__(
        self,
        env: VisionRuntimeEnv,
        thresholds=((8, "yolo11n"), (16, "yolo11s"), (1e9, "yolo11m")),
    ):
        super().__init__(env.action_space.n)
        self.name = "rule_based"
        self._actions = [
            env.config_space.config_to_action(
                RuntimeConfig(model=m, resolution=640, precision="fp32")
            )
            for _, m in thresholds
        ]
        self._thr = [float(t) for t, _ in thresholds]

    def choose(self, obs: np.ndarray, info: dict) -> int:
        count = float(info.get("obj_count", 0.0))
        for i, thr in enumerate(self._thr):
            if count < thr:
                return self._actions[i]
        return self._actions[-1]


class ContextualBandit(Policy):
    def __init__(self, n_actions: int, n_bins: int = 3, epsilon: float = 0.1, seed: int = 0):
        super().__init__(n_actions)
        self.name = "contextual_bandit"
        self.n_bins = int(n_bins)
        self.epsilon = float(epsilon)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._q = np.zeros((self.n_bins, self.n_actions))
        self._n = np.zeros((self.n_bins, self.n_actions))

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._q = np.zeros((self.n_bins, self.n_actions))
        self._n = np.zeros((self.n_bins, self.n_actions))

    def _bin(self, obs: np.ndarray) -> int:
        count_feat = float(np.clip(obs[1], 0.0, 1.0))
        return int(min(self.n_bins - 1, int(count_feat * self.n_bins)))

    def choose(self, obs: np.ndarray, info: dict) -> int:
        b = self._bin(obs)
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return int(np.argmax(self._q[b]))

    def update(self, obs: np.ndarray, info: dict, reward: float, action: int) -> None:
        b = self._bin(obs)
        self._n[b, action] += 1
        k = self._n[b, action]
        self._q[b, action] += (reward - self._q[b, action]) / k


class SB3Policy(Policy):
    def __init__(self, model, name: str = "ivr_rl"):
        self.model = model
        self.name = name

    def choose(self, obs: np.ndarray, info: dict) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def build_policy(name: str, env: VisionRuntimeEnv, **kwargs) -> Policy:
    n = env.action_space.n
    if name == "random":
        return RandomPolicy(n, seed=int(kwargs.get("seed", 0)))
    if name == "rule_based":
        return RuleBasedPolicy(env)
    if name == "contextual_bandit":
        return ContextualBandit(n)
    if name.startswith("always_"):
        parts = name.split("_")[1:]
        if len(parts) == 3:
            return AlwaysConfig(env, parts[0], int(parts[1]), parts[2])
        return AlwaysModel(n, int(parts[0]))
    raise ValueError(f"unknown policy: {name}")


def default_baselines(env: VisionRuntimeEnv) -> list[Policy]:
    """Canonical fixed pipelines + heuristic schedulers for comparison."""
    return [
        AlwaysConfig(env, "yolo11n", 640, "fp32"),
        AlwaysConfig(env, "yolo11s", 640, "fp32"),
        AlwaysConfig(env, "yolo11m", 640, "fp32"),
        RandomPolicy(env.action_space.n, seed=0),
        RuleBasedPolicy(env),
        ContextualBandit(env.action_space.n),
    ]
