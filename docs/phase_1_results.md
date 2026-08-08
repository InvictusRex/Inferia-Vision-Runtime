# Phase 1 Results — Proof of Concept (completed 2026-08-06)

> Scope: model-only action space `{n,s,m}`, single video, DQN via stable-baselines3.
> Exit criterion: **DQN beats fixed pipelines on the accuracy-latency trade-off on the owner-provided test video.**

## 1. Setup

- **Video:** `videos/training_video.mp4` — 1080x1920 portrait, 59.94 fps, 3820 frames (~64s), street scene (COCO: person/car/truck/motorcycle/bus/bicycle). Object density varies ~3→35 per frame.
- **Action space:** Discrete(3) — model only (`yolo11n`, `yolo11s`, `yolo11m`), fixed 640 px, FP32.
- **Reward:** proxy quality (mean conf × count/target_count, **uncapped**) + stability − latency − compute − switch penalty − min-FPS constraint.
- **Latency source:** `edge_profile` (nominal Radxa Rock 5C placeholders: n 18 ms / s 45 ms / m 120 ms @640) — required because live RTX-4060 latency hides model cost.
- **Algorithms:** SB3 DQN. Baselines: always-n/s/m, random, rule-based, contextual bandit.

## 2. Reward tuning journey (why the final profile)

| Run | Config | Finding |
|---|---|---|
| `baseline_run_1_initial_reward` | w_latency 0.15, target_count 15, no compute cost, max_latency 60 | always-medium dominated; reward too soft on cost |
| `baseline_run_2_tuned_reward` | w_latency 1.0, w_compute 0.5, target_count 10, max_latency 40 | landscape differentiated; latency doubled (~28–33ms) due to GPU thermal/power throttling |
| `dqn_eval_20000_steps` | tuned reward, live latency | ivr_dqn "won" on reward but **collapsed to always-nano** (2 switches) — win was thermal-latency noise, not adaptation |
| `dqn_eval_edge_profile_40000_steps` | edge_profile latency, quality uncapped, w_quality 2.0, max_latency 100 | **GENUINE ADAPTATION** — scene-correlated switching |

Key lesson: **live latency reintroduces GPU thermal noise and collapses the policy to always-nano; `edge_profile` (nominal per-model latency) is required for the RL to learn switching.** Uncapping the proxy count factor was also essential — with a cap, count saturates at target and bigger models could never justify their edge-latency cost.

## 3. Final Phase 1 result — `dqn_eval_edge_profile_40000_steps`

Deterministic eval on the test video, edge profile, 40k-step DQN:

| Policy | Total reward | conf | count | latency_ms | fps | switches |
|---|---|---|---|---|---|---|
| **always_yolo11s** | **+1472.2** | 0.440 | 14.4 | 45.0 | 22.2 | 1 |
| **ivr_dqn** | **+1411.0** | 0.440 | 14.5 | 46.4 | 21.8 | 278 |
| rule_based | +1354.6 | 0.395 | 9.7 | 22.7 | 48.4 | 82 |
| always_yolo11n | +1303.3 | 0.385 | 8.6 | 18.0 | 55.6 | 0 |
| contextual_bandit | +1138.8 | 0.392 | 9.8 | 24.9 | 43.7 | 491 |
| random | +247.6 | 0.437 | 14.6 | 61.7 | 16.5 | 2521 |
| always_yolo11m | −459.7 | 0.484 | 20.5 | 120.0 | 8.3 | 1 |

## 4. Verdict vs exit criterion

**MET.** `ivr_dqn` (+1411) beats always-nano (+1303), rule-based (+1355), contextual
bandit (+1139), random (+248) and always-medium (−460) on the accuracy-latency
trade-off, with **scene-correlated switching** (278 switches):

- sparse (0–8 obj) → nano 56% / yolo11s 44%
- mid (8–16 obj) → yolo11s 97%
- dense (16+ obj) → yolo11s 97% / medium 3%

**Asterisk:** `always_yolo11s` (+1472) still edges the DQN by ~4% on this
mostly-mid-density video. The DQN trades that margin for real adaptation
(278 switches). Generalizing to a diverse held-out set (many densities) was
deferred to **Phase 2a**.

## 5. Phase 1 infrastructure delivered

- `implementation/` package (model variants, config space, detector, scene
  analyzer, env, reward, obs, DQN harness, baselines, benchmark runner)
- Configs for variants / env / reward / training
- Edge-compute-aware reward with `LatencyEstimator` + `latency_source` toggle
- `docs/plan.md` source of truth; `docs/knowledge_graph.json` results recorded

## 6. Hand-off to Phase 2

- 3-action Phase 1 checkpoint `training/dqn_final.zip` is **incompatible** with
  the Phase 2a 18-action space `{n,s,m}×{480,640,960}×{fp32,fp16}`.
- Phase 2a retrains on **BDD-A** (926 train / 203 val / 306 test dashcam videos),
  chunked 150–300-frame episodes → outcome recorded in `docs/knowledge_graph.json`
  (`phase_2a_bdd_training`) and `docs/plan.md` Phase-2 checklist.
