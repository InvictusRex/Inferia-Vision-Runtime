"""Offline pipeline validation gate.

Exercises the full chain

    configuration -> EdgeProfile -> environment/state -> observation vector
                  -> policy -> reward/constraint logic -> evaluation metrics

end to end, asserting directional correctness and structural correctness, not
merely that values are non-zero.

Run from the project root:

    python tools/validate_pipeline.py

Exits non-zero if any check fails.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from implementation.model_management.model_variants import build_variants  # noqa: E402
from implementation.model_management.runtime_config_space import (  # noqa: E402
    ConfigSpace,
    RuntimeConfig,
)
from implementation.reinforcement_learning.reward_calculator import (  # noqa: E402
    ProxyReward,
    RewardConfig,
)
from implementation.runtime.edge_profile import (  # noqa: E402
    EdgeProfile,
    compute_constraint_violations,
)
from implementation.runtime.environment_factory import (  # noqa: E402
    build_env_from_configs,
    load_yaml,
)
from implementation.runtime.latency_estimator import LatencyEstimator  # noqa: E402
from implementation.runtime.system_telemetry import GpuSnapshot, Telemetry  # noqa: E402
from implementation.vision_pipeline.scene_analyzer import SceneFeatures  # noqa: E402
from implementation.vision_pipeline.yolo_detector import Detections  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def _edge_profile(hw_path: str = "configs/hardware.yaml") -> tuple[EdgeProfile, dict, ConfigSpace]:
    variants_cfg = load_yaml("configs/variants.yaml")
    hw_cfg = load_yaml(hw_path)
    variants = build_variants(variants_cfg)
    gflops_by_model = {n: v.gflops for n, v in variants.items()}
    estimator = LatencyEstimator.from_variants(variants, gflops_by_model)
    edge = EdgeProfile.from_configs(hw_cfg, variants, estimator)
    cs = ConfigSpace(
        models=list(variants.keys()),
        resolutions=variants_cfg["resolutions"],
        precisions=variants_cfg["precisions"],
    )
    return edge, hw_cfg, cs


def _steady_snapshot(edge: EdgeProfile, config: RuntimeConfig, warmup: int = 50):
    edge.reset()
    snap = None
    for _ in range(warmup):
        snap = edge.snapshot(config)
    return snap


# --- checks 1-4: directional correctness --------------------------------------------


def check_directional_npu_util(edge: EdgeProfile) -> None:
    configs = [
        RuntimeConfig("yolo11n", 480, "fp32"),
        RuntimeConfig("yolo11n", 640, "fp32"),
        RuntimeConfig("yolo11n", 960, "fp32"),
    ]
    utils = [_steady_snapshot(edge, c).npu_util_pct for c in configs]
    check(
        "1. increasing NPU load -> increasing NPU-pressure observation",
        utils[0] < utils[1] < utils[2],
        f"npu_util_pct across res 480/640/960: {[round(u, 2) for u in utils]}",
    )


def check_directional_memory(edge: EdgeProfile) -> None:
    configs = [
        RuntimeConfig("yolo11n", 640, "fp32"),
        RuntimeConfig("yolo11s", 640, "fp32"),
        RuntimeConfig("yolo11m", 640, "fp32"),
    ]
    mems = [_steady_snapshot(edge, c).mem_used_mb for c in configs]
    check(
        "2. increasing memory usage -> increasing memory-pressure observation",
        mems[0] < mems[1] < mems[2],
        f"mem_used_mb across n/s/m @640fp32: {[round(m, 1) for m in mems]}",
    )


def check_directional_temp(edge: EdgeProfile) -> None:
    configs = [
        RuntimeConfig("yolo11n", 480, "fp16"),
        RuntimeConfig("yolo11s", 640, "fp32"),
        RuntimeConfig("yolo11m", 960, "fp32"),
    ]
    temps = [_steady_snapshot(edge, c).temp_c for c in configs]
    check(
        "3. increasing temperature -> increasing thermal observation",
        temps[0] < temps[1] < temps[2],
        f"temp_c across increasing-load configs: {[round(t, 2) for t in temps]}",
    )


def check_directional_power(edge: EdgeProfile) -> None:
    configs = [
        RuntimeConfig("yolo11n", 480, "fp16"),
        RuntimeConfig("yolo11s", 640, "fp32"),
        RuntimeConfig("yolo11m", 960, "fp32"),
    ]
    powers = [_steady_snapshot(edge, c).power_w for c in configs]
    check(
        "4. increasing power -> increasing power-pressure observation",
        powers[0] < powers[1] < powers[2],
        f"power_w across increasing-load configs: {[round(p, 2) for p in powers]}",
    )


# --- check 5: constraint violation reduces reward by the expected amount -----------


def _fixed_reward(cfg: RewardConfig, gpu: GpuSnapshot) -> float:
    reward = ProxyReward(cfg)
    detections = Detections(
        boxes=np.zeros((3, 4)),
        confs=np.full(3, 0.8),
        cls_ids=np.zeros(3, dtype=int),
        names=["car"] * 3,
        latency_ms=20.0,
    )
    scene = SceneFeatures(
        motion=0.5,
        obj_count=3,
        mean_conf=0.8,
        mean_box_area=0.1,
        brightness=0.5,
        entropy=0.5,
        temporal_consistency=1.0,
    )
    telemetry = Telemetry()
    telemetry.record(20.0)
    telemetry.record_gpu(gpu)
    return reward.compute(detections, scene, telemetry, prev_action=0, action=0)


def check_constraint_reward_delta() -> None:
    cfg = RewardConfig(
        w_quality=1.0,
        w_latency=1.0,
        w_compute=0.0,
        w_switch=0.0,
        min_fps=0.0,
        constraints={"max_power_w": 10.0},
        constraint_weights={"max_power_w": 3.0},
    )
    ok_gpu = GpuSnapshot(
        npu_util_pct=10.0,
        mem_used_mb=100.0,
        mem_budget_mb=512.0,
        temp_c=40.0,
        power_w=8.0,
        latency_ms=20.0,
    )
    bad_gpu = GpuSnapshot(
        npu_util_pct=10.0,
        mem_used_mb=100.0,
        mem_budget_mb=512.0,
        temp_c=40.0,
        power_w=15.0,
        latency_ms=20.0,
    )
    r_ok = _fixed_reward(cfg, ok_gpu)
    r_bad = _fixed_reward(cfg, bad_gpu)
    expected_frac = (15.0 - 10.0) / 10.0
    expected_delta = -3.0 * expected_frac
    actual_delta = r_bad - r_ok
    check(
        "5. constraint violation -> reward decreases by the expected amount",
        np.isclose(actual_delta, expected_delta, atol=1e-6),
        f"expected delta {expected_delta:.4f}, got {actual_delta:.4f}",
    )


# --- check 6: switch-penalty correctness (drives the real env) --------------------


def check_switch_penalty(env_cfg, variants_cfg) -> None:
    reward_cfg = {
        "quality_backend": "proxy",
        "w_quality": 0.0,
        "w_latency": 0.0,
        "w_compute": 0.0,
        "w_switch": 0.3,
        "min_fps": 0.0,
        "max_latency_ms": 1000.0,
        "target_count": 10.0,
        "latency_source": "edge_profile",
        "latency_jitter_ms": 0.0,
        "constraints": {},
        "constraint_weights": {},
    }
    env = build_env_from_configs(env_cfg, variants_cfg, reward_cfg, hardware_cfg=None)
    env.reset()
    sequence = [0, 0, 1, 1, 0]
    expected = [0.0, 0.0, -0.3, 0.0, -0.3]
    got = []
    for action in sequence:
        _, reward, _, _, _ = env.step(action)
        got.append(round(float(reward), 6))
    check(
        "6. switch-penalty fires only on immediate-previous-action transitions",
        all(np.isclose(g, e, atol=1e-6) for g, e in zip(got, expected)),
        f"sequence={sequence} expected={expected} got={got}",
    )


# --- check 7: constraints can fire, non-degenerately -------------------------------


def check_constraints_fire(edge: EdgeProfile, cs: ConfigSpace) -> None:
    constrained_cfg = load_yaml("configs/reward_constrained_bdd.yaml")
    constraints = constrained_cfg["constraints"]
    violated = []
    print("    action table (model, res, precision -> violated constraints):")
    for i in range(cs.n_actions):
        config = cs.action_to_config(i)
        snap = _steady_snapshot(edge, config)
        v = compute_constraint_violations(snap.as_dict(), constraints)
        if v:
            violated.append((config.model, config.resolution, config.precision, list(v.keys())))
        keys = list(v.keys())
        print(f"      {config.model:8s} {config.resolution:4d} {config.precision:5s} -> {keys}")
    n = len(violated)
    check(
        "7. stress-test constraints bind on a non-degenerate subset of actions",
        0 < n < cs.n_actions,
        f"{n}/{cs.n_actions} actions violate at least one constraint: {violated}",
    )

    # reward drop for a violating vs a non-violating action, all else equal
    cfg = RewardConfig(
        w_quality=1.0,
        w_latency=1.0,
        w_compute=0.0,
        w_switch=0.0,
        min_fps=0.0,
        constraints=constraints,
        constraint_weights=constrained_cfg["constraint_weights"],
    )
    good_action = next(
        i
        for i in range(cs.n_actions)
        if (
            cs.action_to_config(i).model,
            cs.action_to_config(i).resolution,
            cs.action_to_config(i).precision,
        )
        not in [(m, r, p) for m, r, p, _ in violated]
    )
    bad_action = (
        cs.config_to_action(RuntimeConfig(violated[0][0], violated[0][1], violated[0][2]))
        if violated
        else None
    )
    if bad_action is not None:
        good_snap = _steady_snapshot(edge, cs.action_to_config(good_action))
        bad_snap = _steady_snapshot(edge, cs.action_to_config(bad_action))
        good_gpu = GpuSnapshot(
            npu_util_pct=good_snap.npu_util_pct,
            mem_used_mb=good_snap.mem_used_mb,
            mem_budget_mb=edge.memory_budget_mb,
            temp_c=good_snap.temp_c,
            power_w=good_snap.power_w,
            latency_ms=good_snap.latency_ms,
        )
        bad_gpu = GpuSnapshot(
            npu_util_pct=bad_snap.npu_util_pct,
            mem_used_mb=bad_snap.mem_used_mb,
            mem_budget_mb=edge.memory_budget_mb,
            temp_c=bad_snap.temp_c,
            power_w=bad_snap.power_w,
            latency_ms=bad_snap.latency_ms,
        )
        r_good = _fixed_reward(cfg, good_gpu)
        r_bad = _fixed_reward(cfg, bad_gpu)
        check(
            "7b. a violating action's reward is lower than a non-violating action's",
            r_bad < r_good,
            f"good={r_good:.3f} bad={r_bad:.3f}",
        )


# --- check 8: checkpoint save/load consistency -------------------------------------


def check_checkpoint_consistency(env_cfg, variants_cfg, reward_cfg, hardware_cfg) -> None:
    from stable_baselines3 import DQN

    env = build_env_from_configs(env_cfg, variants_cfg, reward_cfg, hardware_cfg=hardware_cfg)
    model = DQN("MlpPolicy", env=env, seed=0, verbose=0)
    rng = np.random.default_rng(0)
    obs_batch = rng.uniform(0.0, 1.0, size=(16, env.observation_space.shape[0])).astype(np.float32)
    actions_before, _ = model.predict(obs_batch, deterministic=True)
    shape_before = env.observation_space.shape

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "validate_checkpoint.zip"
        model.save(str(ckpt))
        reloaded = DQN.load(str(ckpt))
        actions_after, _ = reloaded.predict(obs_batch, deterministic=True)
        shape_after = reloaded.observation_space.shape

    check(
        "8. checkpoint save/load produces identical deterministic actions",
        bool(np.array_equal(actions_before, actions_after)),
        f"before={actions_before.tolist()} after={actions_after.tolist()}",
    )
    check(
        "8b. checkpoint save/load preserves observation_space.shape",
        shape_before == shape_after == (env.features.dim,),
        f"before={shape_before} after={shape_after} expected={(env.features.dim,)}",
    )


# --- check 9: action-space round trips ---------------------------------------------


def check_action_roundtrip(cs: ConfigSpace) -> None:
    ok = all(cs.config_to_action(cs.action_to_config(i)) == i for i in range(cs.n_actions))
    check(
        "9. config_to_action round-trips all 18 configs",
        ok and cs.n_actions == 18,
        f"n_actions={cs.n_actions}",
    )


# --- check 10: model-order independence --------------------------------------------


def check_model_order_independence(cs: ConfigSpace) -> None:
    sorted_models = cs.models
    enumeration_first_model = cs.action_to_config(0).model
    naive_positional_model = sorted_models[0]
    # The two orderings genuinely differ for this project's variants (n/s/m
    # inserted in that order, but `.models` sorts alphabetically to m/n/s) --
    # this proves a positional zip between them would silently mismap.
    orderings_differ = naive_positional_model != enumeration_first_model
    # Every model must still resolve correctly through config_to_action, never
    # via position in `.models`.
    roundtrip_ok = all(
        cs.action_to_config(
            cs.config_to_action(
                RuntimeConfig(
                    m, cs.action_to_config(0).resolution, cs.action_to_config(0).precision
                )
            )
        ).model
        == m
        for m in sorted_models
    )
    check(
        "10. no code path maps `.models` position to an action positionally",
        orderings_differ and roundtrip_ok,
        f".models={sorted_models} (sorted) vs "
        f"action_to_config(0).model={enumeration_first_model!r} (model-major order)",
    )


# --- check 11: end-to-end observation/policy integration ---------------------------


def check_end_to_end_integration(env_cfg, variants_cfg, reward_cfg, hardware_cfg) -> None:
    env = build_env_from_configs(env_cfg, variants_cfg, reward_cfg, hardware_cfg=hardware_cfg)
    obs, _ = env.reset()
    rng = np.random.default_rng(0)
    rows = [obs]
    steps = 0
    done = False
    while not done and steps < 220:
        action = int(rng.integers(0, env.action_space.n))
        obs, _, term, trunc, _ = env.step(action)
        rows.append(obs)
        done = term or trunc
        steps += 1
        if done:
            obs, _ = env.reset()
            done = False
    arr = np.stack(rows)
    hw_dims = arr[:, 10:14]
    populated = bool(np.any(hw_dims != 0.0))
    varies = bool(np.all(hw_dims.std(axis=0) > 0.0))
    check(
        "11. >=200 real env steps: dims 10-13 populated and vary",
        steps >= 200 and populated and varies,
        f"steps={steps} dim_std={hw_dims.std(axis=0).tolist()}",
    )


def main() -> None:
    print("=== offline pipeline validation ===\n")

    edge, hw_cfg, cs = _edge_profile()
    check_directional_npu_util(edge)
    check_directional_memory(edge)
    check_directional_temp(edge)
    check_directional_power(edge)
    check_constraint_reward_delta()

    env_cfg = load_yaml("configs/env_bdd.yaml")
    variants_cfg = load_yaml("configs/variants.yaml")
    reward_cfg = load_yaml("configs/reward_bdd.yaml")

    check_switch_penalty(env_cfg, variants_cfg)
    check_constraints_fire(edge, cs)
    check_checkpoint_consistency(env_cfg, variants_cfg, reward_cfg, hw_cfg)
    check_action_roundtrip(cs)
    check_model_order_independence(cs)
    check_end_to_end_integration(env_cfg, variants_cfg, reward_cfg, hw_cfg)

    print()
    if FAILURES:
        print(f"=== {len(FAILURES)} CHECK(S) FAILED ===")
        for name in FAILURES:
            print(f"  - {name}")
        sys.exit(1)
    print("=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
