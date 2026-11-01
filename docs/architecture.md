# Inferia Vision Runtime: Architecture

Inferia Vision Runtime (IVR) treats vision inference as a sequential decision problem rather than a fixed pipeline. A learned policy sits between the incoming frames and the inference engine and chooses, frame by frame, which detector configuration to run, according to scene complexity, temporal context and the resources and latency budget currently available. The policy is trained to maximise long-term performance, not to make independent per-frame guesses.

What the policy learns, in practice:

- **Easy or empty scene, or a tight budget**: run a smaller, faster configuration.
- **Crowded scene with a loose budget**: run a larger, more accurate configuration.
- The trade-off is discovered from reward, not from hand-coded rules.

## Data flow

```
                     Video file or camera
                              │
                              ▼
                     Scene analysis module
                              │
            Scene, temporal and hardware features (observation)
                              │
                              ▼
                        Policy (DQN / PPO)
                              │
                Runtime configuration (action)
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
           Model          Resolution        Precision
                              │
                              ▼
                     Detector inference
                              │
                 Detections + telemetry
                              │
                              ▼
                   Reward / next observation
```

## Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Model pool | YOLO11 `n / s / m` | A common `Detector` interface keeps other models (RT-DETR, DINO, SAM) swappable |
| Action space | Flattened discrete product: model × resolution × precision | Works with stock DQN and PPO; can be extended to a hierarchical space |
| Reward quality signal | Pluggable: proxy (no labels) or IoU-matched F1 (labels present) | Selected via config |
| RL framework | `gymnasium` environment + `stable-baselines3` agents (`sb3-contrib` for MaskablePPO) | Standard, well-tested algorithms; the contribution is the runtime, not a new RL algorithm |
| Hardware model | `EdgeProfile`: the target device's cost per configuration, measured values overriding simulated ones | Training on a desktop GPU's live state makes every configuration look free and collapses the policy to the cheapest one |
| Constraint semantics | Reward penalty proportional to the violation fraction; optional deadline action masking | Keeps DQN and PPO unmodified |
| Layout | Flat top-level packages, absolute imports, `pyproject.toml` | Package names match import paths |

## Components

### Model pool

`model_management/model_variants.py` registers the available detectors from `configs/variants.yaml` (weights file, GFLOPs, nominal edge latency and weight size). YOLO11 nano, small and medium are the current pool.

### Runtime configuration space

The policy selects a full inference configuration, not just a model:

| Parameter | Options |
|---|---|
| Model | `n / s / m` |
| Input resolution | `480 / 640 / 960` |
| Precision | `fp32 / fp16` on the host; NPU `fp16 / int8` on the RK3588 |

`model_management/runtime_config_space.py` flattens the Cartesian product into 18 discrete actions. Frame skipping, confidence thresholds and ROI strategies are natural extensions of the same space.

### Scene analyzer

`vision_pipeline/scene_analyzer.py` extracts cheap signals before the detector runs: motion magnitude, previous object count, mean confidence, mean box size, brightness, image entropy and temporal box consistency.

### Telemetry and edge profile

`runtime/system_telemetry.py` tracks loop rate and inference latency. `runtime/edge_profile.py` models the target device (Radxa ROCK 5C, RK3588S2) per configuration:

| Metric | Model |
|---|---|
| Memory | per-model weight size × precision bytes + activations ∝ (res/640)² |
| NPU utilisation | `gflops × (res/640)² × precision_factor / (TOPS × efficiency)` |
| Power | base W + load × per-TOPS factor |
| Temperature | ambient + load × k (optional EWMA time constant) |
| Latency | `LatencyEstimator`, or measured board latency |

The profile lives in `configs/hardware.yaml`; `configs/hardware_measured.yaml` fills its `measured:` section from on-device benchmarks, and any measured value replaces the simulated estimate. Observation dimensions 10 to 13 carry these metrics for the current action, so they are deterministic and consistent with the reward.

### Environment

`reinforcement_learning/vision_runtime_env.py` is a `gymnasium.Env` over videos or annotated sequences. One step:

1. Observe the current state (scene features, telemetry, device metrics, deadline flags).
2. Select a runtime configuration.
3. Run inference on the frame.
4. Measure latency and detection quality.
5. Compute the reward, including the switching cost.
6. Move to the next frame.

Episodes are random 150 to 300 frame chunks of a random video, seeded for reproducibility. The configuration chosen on frame *t* applies to frame *t+1*, as in a real streaming system.

### Reward

`reinforcement_learning/reward_calculator.py` balances detection quality and stability against latency, compute, switching and constraint violations. Two quality backends:

- **Proxy**: no ground truth. Mean confidence × `count / target_count`, blended with count stability and box-size consistency.
- **F1 against ground truth**: predictions greedily matched to labels at IoU 0.5; optionally restricted to vehicle classes with ignored regions masked, matching UA-DETRAC's official scoring.

In deadline mode the continuous latency and compute penalties are replaced by a fixed penalty per frame that exceeds its deadline.

### Constraints and priorities

- **Hard constraints**: `min_fps`, `max_latency_ms`, memory, power, temperature and NPU utilisation budgets, evaluated against the edge profile. Violations are penalised in proportion to how far they exceed the budget.
- **Soft priorities**: reward weights for quality, latency, compute and switching.

### Temporal decisions and switching cost

The policy reasons over sequences: when to switch, when to hold, when to spend compute. Every switch pays a cost in the reward, so the policy does not oscillate between configurations without a reason.

## Acceleration backends

A `.pt` file holds the trained weights; runtime engines compile them into optimised executables. On the RK3588, models are exported to fixed-shape ONNX and converted with RKNN-Toolkit2 to NPU fp16 and calibrated int8, with the detection head decoded on the CPU to keep int8 accuracy. On NVIDIA hardware the same role would be played by TensorRT engines. In every case the runtime chooses among prebuilt engines at run time, behind the same detector interface.

## Evaluation

**Baselines:** every one of the 18 fixed configurations, random, rule based (object-count thresholds), contextual bandit, and deadline lookup tables built from the device profile.

**Metrics:** detection F1 (or proxy quality), latency and deadline misses, loop rate, simulated power, NPU utilisation, memory, switching frequency and cumulative reward, with paired per-sequence statistics against a reference policy.

## Application profiles

The framework is general; the learned policy is specific to a deployment.

- **General, built once:** scene analyzer, environment, reward plumbing, action space, training harness, baselines and configuration system.
- **Specific, per deployment:** the policy, trained against that application's reward, footage, model pool and device profile.

Onboarding a new application:

1. Describe the application in config: classes of interest, model pool, priorities and hard constraints, reward weights and quality backend.
2. Supply footage, labelled for the F1 reward or unlabelled for the proxy reward.
3. Provide custom-trained weights if the classes are not in COCO.
4. Train, which produces an application-specific policy.
5. Deploy the policy with its device profile.

Example, maritime search and rescue: classes `boat`, `person`, `lifejacket`; YOLO weights fine-tuned on them; the F1 reward (small, rare objects make the proxy weak) with priorities tuned to find people at almost any latency cost.

Caveats:

- One policy for several unrelated applications without retraining would need context-conditioned or meta-RL policies.
- The proxy reward is class-sensitive; small or rare objects degrade it, so labelled data is preferred.
- Custom classes need custom-trained weights; the runtime cannot detect classes the detector never learned.

## Data and outputs

- **BDD-A** dashcam footage lives outside the repository at `../BDDA/BDDA/{training,validation,test}/camera_videos/`, referenced by `configs/env/env_bdd.yaml`.
- **UA-DETRAC** images and XML annotations live at the `dataset_dir` in `configs/env/env_detrac*.yaml`.
- `videos/training_video.mp4` is a short street-scene clip (1080x1920, 59.94 fps, 3,820 frames, about 3 to 35 objects per frame) used for quick end-to-end checks.
- Evaluation output is named `output/{split}_{model_stem}_eval_output.csv`, so separate evaluations never overwrite each other; `--output <name>` overrides it.
- A density survey of 40 BDD-A training videos (YOLO11n at 640) gave a median of about 9 objects per frame (p90 14, p99 18), which sets the proxy reward's `target_count: 10` and the observation's `MAX_COUNT: 40`.
