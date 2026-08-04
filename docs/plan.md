# Inferia Vision Runtime (IVR) — Project Plan & Source of Truth

> **Status:** Active / Under development
> **Last updated:** 2026-08-04
> **This document is the single source of truth** for the IVR project across all phases.
> Changes to scope, architecture, reward design, or roadmap MUST be reflected here first.

---

## 1. Vision

Inferia Vision Runtime (IVR) is an adaptive vision inference framework that uses
Reinforcement Learning (RL) to dynamically optimize computer vision pipelines in
real time.

Unlike traditional systems that use a fixed inference configuration throughout
deployment, IVR continuously adapts its execution strategy according to scene
complexity, temporal information, and available hardware resources.

The long-term goal is to create a reusable adaptive vision runtime capable of
maximizing perception quality while minimizing latency, computational cost, and
energy consumption on edge devices.

---

## 2. Core Idea

Instead of statically selecting one detector (e.g., YOLO11n, YOLO11s, YOLO11m),
IVR intelligently decides the best runtime configuration for every stage of a
video stream.

The RL agent acts as an orchestration layer sitting between incoming video frames
and the inference engine. Its objective is to **maximize long-term performance**
rather than making independent frame-by-frame decisions.

### Mental model (what the agent learns)

- **Easy / empty scene** → shrink to a smaller model (Nano) to conserve compute.
- **Complex / crowded scene** → reach for a larger model (Small/Medium) for
  accuracy — but only if latency/FPS/energy constraints allow.
- The agent discovers this trade-off automatically from reward, not from
  hand-coded rules.

---

## 3. High-Level Architecture

```
                     Local Video File (or camera later)
                                   │
                                   ▼
                         Scene Analysis Module
                                   │
                 Scene & Temporal Features (observation)
                                   │
                                   ▼
                       RL Decision Agent (IVR)
                                   │
                Runtime Configuration Selection (action)
                                   │
                                   ▼
                        Vision Runtime Execution
                                   │
             ┌─────────────────────┼──────────────────────┐
             ▼                     ▼                      ▼
           Model             Resolution              Precision
                                   │
                                   ▼
                         Vision Model Inference
                                   │
                 Detection Results + Telemetry
                                   │
                                   ▼
                         Reward / Next State
```

---

## 4. Locked-In Design Decisions

These decisions were confirmed with the project owner and are binding:

| Decision | Choice | Notes |
|---|---|---|
| **Model pool (now)** | YOLO only: `yolo11n / yolo11s / yolo11m` | RT-DETR, DINO, SAM are future work |
| **Reward quality signal** | Pluggable: proxy (default, no labels) OR mAP (labels present) | Selected via config; hybrid supported |
| **RL framework** | `gymnasium` (env) + `stable-baselines3` (agents) | New dependencies to install |
| **Video input** | Local video files (path configurable) | Camera-ready abstractions from day one |
| **Hardware telemetry** | Deferred to Phase 2 (`pynvml`) | Phase 1 uses inference latency/FPS only |
| **Weights source** | Official `yolo11n/s/m.pt`, downloaded into `weights/` | Already fetched; re-downloadable on demand |
| **Action space** | Flattened **discrete** combos: model × resolution × precision | DQN-friendly; env designed to extend to hierarchical later |
| **Layout/tooling** | `implementation/` package, ruff + type hints, `pyproject.toml` | Minimal but clean |
| **Phase scope** | Full 4-phase roadmap planned; **implementation is YOLO-only until non-YOLO integration phase** | Model-agnostic interfaces kept throughout |

---

## 5. Environment Facts (current machine)

- **GPU:** NVIDIA GeForce RTX 4060 Laptop — 8 GB VRAM
- **Python:** 3.9.11
- **Installed:** torch 2.7.0+cu118, torchvision 0.22.0+cu118, ultralytics 8.3.140,
  opencv-contrib-python 4.13, opencv-python 4.11, numpy 1.26.4, PyYAML 6.0.2,
  pandas, matplotlib, shapely
- **Missing (to install):** `gymnasium`, `stable-baselines3` (Phase 1); `pynvml` (Phase 2)
- **Dev tooling to add:** `ruff`, `black`

---

## 6. Major Components

### 6.1 Vision Model Pool

The runtime manages multiple vision models. YOLO variants are the initial
implementation. A common `Detector` interface makes future models swappable
without touching the core runtime.

- Initial: `yolo11n`, `yolo11s`, `yolo11m`
- Future: RT-DETR, Grounding DINO, Segment Anything, other perception models

### 6.2 Runtime Configuration Space

The RL agent selects an inference **configuration**, not just a model:

| Parameter | Options (Phase 2 onward) | Phase 1 |
|---|---|---|
| Model | `n / s / m` | ✅ `n / s / m` |
| Input Resolution | `480 / 640 / 960` | Fixed (640) |
| Numerical Precision | `FP32 / FP16` (+ `INT8` via TensorRT in Phase 3) | Fixed (FP32) |
| Frame Skip Rate | Future | — |
| Confidence Threshold | Future | — |
| ROI Processing Strategy | Future | — |

Action space = flattened Cartesian product of enabled parameters, exposed as a
single discrete index to DQN/PPO.

### 6.3 Scene Analyzer

A lightweight module extracts inexpensive scene information before expensive
inference. These signals form the visual portion of the RL observation vector:

- Motion magnitude
- Previous object count
- Average confidence
- Bounding box sizes
- Brightness
- Image entropy
- Temporal consistency

### 6.4 Hardware Monitor

Observes device state (Phase 2+ via `pynvml`):

- GPU utilization, CPU utilization
- VRAM usage, RAM usage
- Temperature, power consumption
- FPS, inference latency

### 6.5 RL Environment

Custom `gymnasium.Env` over a video file. One interaction:

1. Observe current state (scene features + telemetry)
2. Select runtime configuration (action)
3. Execute inference (one frame or frame-group)
4. Measure performance (latency, FPS, quality proxy/mAP)
5. Compute reward (incl. switching penalty)
6. Transition to next frame

### 6.6 RL Agent

- Initial algorithms: **DQN**, **PPO** (via `stable-baselines3`)
- Future exploration: SAC, Hierarchical RL, Multi-objective RL
- Note: the contribution is the adaptive runtime, not a new RL algorithm.

### 6.7 Reward Function (pluggable)

Balances competing objectives:

- **Maximize:** detection quality, stability, resource efficiency
- **Minimize:** latency, energy, switching overhead, memory

Two quality backends, selected by config:

- **Proxy** (default): no ground truth required. Quality ≈ mean confidence,
  object-count stability, box-size consistency.
- **mAP**: requires annotated data. Ground-truth mAP vs. model output.

### 6.8 Priority / Constraint Configuration

Runtime behavior is steered by configurable knobs (the "constrained
optimization" layer from §10 of the original vision):

- **Hard constraints:** `min_fps`, `max_latency_ms`, `max_vram`, `max_power`,
  `max_temp`. Violations incur a large penalty or invalidate the action.
- **Soft priorities (reward weights):** e.g. `accuracy: 0.6`, `energy: 0.2`,
  `stability: 0.2`. A config can demand "keep 30 FPS no matter what" or
  "maximize quality, speed flexible."

### 6.9 Temporal Decision Making

IVR reasons over sequences, learning *when* to switch, *when* to stay stable,
*when* to spend compute, and *when* to conserve resources. This long-horizon
behavior is why RL is the right tool.

### 6.10 Switching Cost Awareness

Model/config switches have overhead. The policy must avoid unnecessary
oscillation while still adapting to changing scenes → switching penalty baked
into the reward.

---

## 7. Acceleration Backends: TensorRT & ONNX (Phase 3)

Clarification of what these are and why they matter:

- A `.pt` file is the **trained weights**. TensorRT and ONNX are **runtime
  engines** that convert those weights into highly optimized executable programs.
- **TensorRT (NVIDIA):** compiles the model into a GPU-tuned engine, fuses
  layers, and unlocks **FP16 / INT8** precision → the real 2–5× speedups on the
  RTX 4060, and the vehicle for precision switching.
- **ONNX Runtime:** cross-vendor inference engine (CPU/GPU), lightweight,
  supports quantization.

Deployment flow: **train/choose weights → export to ONNX → build TensorRT
engines (FP16/INT8 variants) → IVR picks among engines at run time.**

- **Phase 1** uses raw ultralytics PyTorch inference to prove the RL logic.
- **Phase 3** swaps in the engines behind the same `Detector` interface.

---

## 8. Experimental Evaluation

### Baselines

- Always Nano, Always Small, Always Medium
- Random Scheduler
- Rule-Based Scheduler
- Supervised Scheduler
- Contextual Bandit
- **IVR (RL)** ← the candidate

### Metrics

- mAP (or proxy quality score)
- FPS
- Latency
- Power consumption
- GPU utilization
- Memory usage
- Switching frequency
- Cumulative reward

Goal: a better accuracy–latency–energy trade-off than any fixed pipeline.

---

## 9. Proposed Repository Layout

```
pyproject.toml, README.md, .gitignore
weights/            # yolo11n/s/m.pt (official, downloaded)
videos/             # training/eval video files
configs/            # yaml: model variants, config space, env, reward, training
implementation/
  model_management/
    model_variants.py        # ModelVariant registry (repo yolo11n/s/m + gflops)
    runtime_config_space.py  # flatten model×res×precision → discrete action index
  runtime/
    system_telemetry.py      # Phase1: infer fps/latency; Phase2: pynvml hooks
    environment_factory.py   # wires configs → env (weights, detector, reward, obs)
  vision_pipeline/
    yolo_detector.py         # thin ultralytics wrapper → boxes+scores+latency
    scene_analyzer.py        # motion, count/conf/box-size, brightness, entropy
    video_frame_source.py    # video file ingest, frame iterator (camera-ready)
  reinforcement_learning/
    vision_runtime_env.py    # gymnasium.Env over the video stream
    reward_calculator.py     # RewardCalculator: proxy | map | hybrid
    observation_features.py  # observation vector builder
    dqn_training_harness.py  # SB3 DQN (and PPO later) training CLI
  evaluation/
    baseline_schedulers.py   # Always-n/s/m, Random, Rule-based, Contextual bandit
    benchmark_runner.py      # head-to-head comparison vs IVR → output/benchmark.csv
training/           # model checkpoints
runs/               # tensorboard / logs
output/             # metrics
```

CLI entrypoints: `python -m implementation.evaluation.benchmark_runner` (baselines
comparison, `--model` to add a trained DQN) and
`python -m implementation.reinforcement_learning.dqn_training_harness` (train).

---

## 10. Dependencies

**Add (Phase 1):** `gymnasium`, `stable-baselines3`
**Add (Phase 2):** `pynvml`
**Add (Phase 3):** `onnx`, `onnxruntime-gpu`, TensorRT (via NVIDIA package)
**Dev:** `ruff`, `black`

---

## 11. Development Roadmap

### Phase 1 — Proof of Concept
- [ ] Project scaffolding: pyproject, ruff config, `implementation/` skeleton, configs
- [ ] Variants + action space (model only: `{n,s,m}`)
- [ ] Detector wrapper with latency/FPS telemetry
- [ ] Scene Analyzer → observation vector
- [ ] Pluggable reward (proxy default, mAP option) with switching penalty
- [ ] Gymnasium env over a video file
- [ ] SB3 **DQN** agent harness
- [ ] Eval harness: baselines + metrics comparison
- **Exit criteria:** DQN beats fixed pipelines on the accuracy-latency trade-off
  on the owner-provided test video.

### Phase 2 — Adaptive Runtime
- [ ] Expand action space to model × resolution × precision
- [ ] NVML telemetry (VRAM, util, GPU temp, power) into observations
- [ ] **PPO** alongside DQN
- [ ] Constraint-aware rewards (hard constraints + priority weights)
- [ ] Switching-penalty tuning

### Phase 3 — Edge Runtime
- [ ] TensorRT and ONNX Runtime backends behind `Detector` interface
- [ ] FP16 / INT8 precision variants
- [ ] Real hardware benchmarking with measured energy + thermal
- [ ] Constraint solver: min FPS / max latency / max power / max memory / thermal

### Phase 4 — Research
- [ ] Multiple datasets, generalization experiments
- [ ] Ablation studies
- [ ] Pareto frontier analysis
- [ ] Comparison vs non-RL schedulers
- [ ] Paper preparation
- [ ] Non-YOLO detector integration (RT-DETR, DINO, SAM) via the same interface

---

## 12. Application Profiles (Generalized Framework, Specialized Policies)

### 12.1 Philosophy

The framework is **generalized**; the learned policy is **specialized per
application**.

- **Generalized (built once):** the machinery — scene analyzer, Gymnasium env,
  reward plumbing, action space, training harness, eval baselines, config
  system. Application-agnostic and model-agnostic.
- **Specialized (per deployment):** the RL policy is the output of training
  against a specific application's reward, video distribution, and model pool.
  This is analogous to fine-tuning and is how RL systems deploy in practice.

### 12.2 How a user onboards a new application

1. Write an **application profile** (YAML): classes of interest, model pool
   (e.g., custom-trained YOLO `.pt` for `boat / person / lifejacket`),
   priorities + hard constraints, reward weights, reward quality backend
   (`proxy` or `map`).
2. Supply **videos / dataset** (labeled → mAP reward; unlabeled → proxy).
3. Optionally provide **custom-trained weights** for their classes (stock COCO
   weights only detect COCO classes).
4. Run training → produces an **app-specific RL policy**.
5. Deploy that policy with its profile.

### 12.3 Reference example — Maritime Search & Rescue

- Classes: `boat`, `person`, `lifejacket`
- Model pool: custom YOLO weights fine-tuned on those classes
- Reward: **mAP path** (small/rare objects make the proxy weak) + latency +
  switching penalties; priorities tuned for "detect persons/lifejackets at all
  cost, latency secondary"
- Videos: SAR footage with varying sea-state and crowd density

### 12.4 Honest caveats

- **Zero-shot generalization across applications** (one policy for SAR, warehouse,
  and traffic without retraining) is harder research (meta-RL / context-
  conditioned policies) — a Phase 4 thread, not Phase 1.
- **Proxy reward is class-sensitive.** Small/rare objects (lifejackets) degrade
  the mean-confidence proxy → prefer the labeled `map` path there.
- **Custom classes require custom-trained weights.** The framework accepts a
  configurable weight path, but it cannot invent classes the detector never saw.

---

## 13. Long-Term Vision

IVR aims to become a general-purpose adaptive perception runtime capable of
orchestrating vision workloads across heterogeneous hardware by making
intelligent, resource-aware runtime decisions through reinforcement learning.

The framework remains **model-agnostic** so future detectors and perception
models integrate without changing the core runtime architecture.

---

## 14. Open Items / Risks

- Phase 1 smoke test video: `videos/training_video.mp4` — 1080x1920 portrait,
  59.94 fps, 3820 frames (~64s), street scene fully detectable by stock COCO
  weights (person, car, truck, motorcycle, bus, bicycle). Object density varies
  strongly (approx 3→35 per frame), so the RL agent has a real adaptation task.
  Training uses the **full video**, not a sampled subset.
- `gymnasium` + `stable-baselines3` installation responsibility (owner or
  assistant) — confirm at build start.
- True INT8 requires TensorRT in Phase 3; FP16 is available in Phase 2.
- Reward with mAP needs annotated data; otherwise the proxy path is used.
- SB3/torch/py3.9 compatibility must be validated at install time.
