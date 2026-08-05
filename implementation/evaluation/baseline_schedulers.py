from __future__ import annotations

import numpy as np

from ..reinforcement_learning.vision_runtime_env import VisionRuntimeEnv


class Policy:
    name = "policy"

    def __init__(self, n_actions: int):
        self.n_actions = int(n_actions)

    def choose(self, obs: np.ndarray, info: dict) -> int:
        raise NotImplementedError

    def update(self, obs: np.ndarray, info: dict, reward: float, action: int) -> None:
        pass


class AlwaysModel(Policy):
    def __init__(self, n_actions: int, action: int):
        super().__init__(n_actions)
        self.name = f"always_{action}"
        self.action = int(action)

    def choose(self, obs: np.ndarray, info: dict) -> int:
        return self.action


class RandomPolicy(Policy):
    def __init__(self, n_actions: int, seed: int = 0):
        super().__init__(n_actions)
        self.name = "random"
        self._rng = np.random.default_rng(seed)

    def choose(self, obs: np.ndarray, info: dict) -> int:
        return int(self._rng.integers(0, self.n_actions))


class RuleBasedPolicy(Policy):
    def __init__(self, n_actions: int):
        super().__init__(n_actions)
        self.name = "rule_based"

    def choose(self, obs: np.ndarray, info: dict) -> int:
        count = float(info.get("obj_count", 0.0))
        ratio = count / max(1, self.n_actions * 5)
        if ratio < 1.0:
            return 0
        if ratio < 2.0:
            return min(1, self.n_actions - 1)
        return min(2, self.n_actions - 1)


class ContextualBandit(Policy):
    def __init__(self, n_actions: int, n_bins: int = 3, epsilon: float = 0.1, seed: int = 0):
        super().__init__(n_actions)
        self.name = "contextual_bandit"
        self.n_bins = int(n_bins)
        self.epsilon = float(epsilon)
        self._rng = np.random.default_rng(seed)
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
    def __init__(self, model, name: str = "ivr_dqn"):
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
        return RuleBasedPolicy(n)
    if name == "contextual_bandit":
        return ContextualBandit(n)
    if name.startswith("always_"):
        return AlwaysModel(n, int(name.split("_", 1)[1]))
    raise ValueError(f"unknown policy: {name}")
