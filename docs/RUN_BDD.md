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

## 6. Density survey (already done; rerun anytime)

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
