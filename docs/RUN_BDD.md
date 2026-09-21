# RUN_BDD.md — Offline BDD-A training & evaluation

Everything below runs **offline** (no internet): YOLO weights are already in
`weights/`, BDD-A videos are local, and all Python deps are in `.venv`.

Run every command from the **project root** (the folder containing `train.py`).
The BDD-A dataset is a sibling directory referenced as `..\BDDA\BDDA`.

---

## 1. Verify the data is in place

```powershell
Test-Path "../BDDA/BDDA/training/camera_videos"
```

Should print `True`. (926 / 203 / 306 videos across train / val / test.)

## 2. Train the DQN (the main command)

Recommended — run via the **root launcher** (works from any folder, no module form):

```powershell
python train.py
```

Same thing with an explicit dataset path (configurable straight from the CLI):

```powershell
python train.py --dataset-dir "..\BDDA\BDDA"
```

Or the module form from the project root:

```powershell
& .\.venv\Scripts\python.exe -m implementation.reinforcement_learning.train_runtime `
  --env configs\env_bdd.yaml `
  --variants configs\variants.yaml `
  --reward configs\reward_bdd.yaml `
  --training configs\training_bdd.yaml
```

- Defaults: **DQN, 120,000 timesteps**, 18-action space
  (`{n,s,m} × {480,640,960} × {fp32,fp16}`), edge-profile reward, chunked
  150–300-frame episodes over random train videos.
- Expected runtime on the RTX 4060: **~1.5–4 hours** (overnight).
- Checkpoint → `training\dqn_bdd_final.zip`. Logs → `runs\tensorboard\dqn_bdd_1`.
- **Live console progress** prints `steps/ETA/episodes/ep_rew_mean/eps/loss` every
  2000 steps (like YOLO's loop table). Add `--verbose-episodes` to also print a
  line per finished episode (shows the sampled video).

### train.py options

```powershell
python train.py --timesteps 5000                 # shorter run
python train.py --algorithm ppo                  # PPO instead of DQN
python train.py --dataset-split validation       # train on validation split
python train.py --dataset-dir <path>             # point at a different dataset root
python train.py --log-every 1000                 # print progress more often
python train.py --verbose-episodes               # per-episode trace
```

### Optional: shorter run to sanity-check first

```powershell
python train.py --timesteps 5000
```

## 3. (Optional) watch training live

```powershell
& .\.venv\Scripts\tensorboard.exe --logdir runs\tensorboard
```

## 4. Evaluate on the held-out test split (all 306 videos)

Recommended — run via the **root eval launcher** (works from any folder):

```powershell
python eval.py
```

Same thing with an explicit model path (configurable straight from the CLI):

```powershell
python eval.py --model training\dqn_bdd_final.zip
```

Or the module form from the project root:

```powershell
& .\.venv\Scripts\python.exe -m implementation.evaluation.benchmark_runner `
  --env configs\env_bdd.yaml `
  --variants configs\variants.yaml `
  --reward configs\reward_bdd.yaml `
  --dataset test `
  --model training\dqn_bdd_final.zip
```

- Defaults: **`--dataset test`** (BDD-100K held-out split, all 306 videos), model
  `training\dqn_bdd_final.zip`.
- Runs every policy (always nano/small/medium @640-fp32, random, rule-based,
  contextual bandit, and your DQN) over **all 306 test videos**, printing
  mean ± std of reward/conf/count/latency/fps/switches plus per-density-bin
  rewards.
- Writes per-video rows to `output\test_dqn_bdd_final_eval_output.csv`.
- Prints a final **claim check** line: `ivr_dqn` mean reward vs
  `always_yolo11s_640_fp32` (the Phase 1 asterisk we're fixing).
- **Output naming scheme** (so future runs never clobber each other):
  `output/{dataset}_{model_stem}_eval_output.csv`, e.g. `test_dqn_bdd_final_eval_output.csv`,
  `validation_ppo_bdd_eval_output.csv`. Override with `--output <name>`.

### eval.py options

```powershell
python eval.py --dataset validation        # eval on validation split instead
python eval.py --n-videos 50               # quick pass over 50 videos
python eval.py --model <path>              # point at a different checkpoint
python eval.py --dataset-dir <path>        # point at a different dataset root
python eval.py --output <name>             # override the CSV filename
```

### Eval on validation (dev) instead

```powershell
... --dataset validation
```

### Eval a limited number of videos

```powershell
... --dataset test --n-videos 50
```

## 5. Train PPO too (Phase 2b, optional)

```powershell
python train.py --algorithm ppo --timesteps 120000
```

## 6. Phase 2b: PPO 300k + switch-penalty sweep (root launcher)

```powershell
python sweep.py                              # PPO 300k, then w_switch 0.1/0.5 @ 60k
python sweep.py --timesteps 20000            # shorter sanity pass
python sweep.py --skip-train --sweep-algorithm ppo --sweep-values 0.1,0.3,0.5
python sweep.py --constraints on             # train/sweep with hard constraints
python sweep.py --sweep-values 0.1,0.3,0.5 --sweep-timesteps 60000
```

- Step 1 trains `configs/training/training_ppo_bdd.yaml` (PPO, 300k, `training/ppo_bdd_final.zip`).
- Step 2 retrains `{log_root}_ws{w}_final.zip` for each `--sweep-values` entry (default 0.1 and 0.5).
- Each job prints the same live progress bar as `train.py`.

## 7. Phase 2b: edge-emulated hardware constraints

The reward and observation read **nominal Rock 5C values** (never the 4060's
live numbers — that would break VRAM/power feasibility and reintroduce thermal
noise). `configs/hardware.yaml` defines the edge profile (6 TOPS, 2048 MB VRAM,
5 W TDP, 30 fps target); `runtime/edge_profile.py` emulates
VRAM/util/power/temp/latency per config.

- Observation is now **14-dim** (dims 10–13 = emulated util/vram/temp/power).
  The old 10-dim Phase 2a checkpoints are **incompatible** — retrain DQN + PPO.
- Constraints + weights live in `configs/reward/reward_constrained_bdd.yaml`
  (`max_vram_mb 2048`, `max_power_w 5`, `max_gpu_temp_c 80`, `max_gpu_util 90`,
  `max_latency_ms 100`). Violations add a heavy `w*(1+fraction)` penalty.
- Toggle with `--constraints on|off` and override the switch weight with
  `--w-switch <float>` on `train.py`, `eval.py`, and `sweep.py`.

```powershell
python train.py --constraints on --w-switch 0.5     # constrained DQN retrain
python eval.py --constraints on                     # eval under the same constraints
python eval.py --reward configs/reward/reward_constrained_bdd.yaml
```

> **Note:** all EdgeProfile numbers are placeholders pending Phase 3 real
> benchmark measurements on the Radxa Rock 5C.

## 8. Density survey (already done; rerun anytime)

```powershell
& .\.venv\Scripts\python.exe -m tools.survey_density --n-videos 40
```

---

## Notes

- **No internet needed.** If a `ModuleNotFoundError` appears it's a local dep
  issue (tell me, don't reinstall anything without checking).
- `--training` points at the config file that sets hyperparameters and
  `log_name`. The saved file is `training\{log_name}_final.zip`.
- The old Phase 1 checkpoint `training\dqn_final.zip` was trained on the old
  3-action space and is **not** compatible with the 18-action env.
- Nothing here commits anything to git; when you're happy with results, ask me
  to record them in `docs/knowledge_graph.json`.
