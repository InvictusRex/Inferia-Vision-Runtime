"""Training sequencer: runs DQN 120k then PPO 300k, back-to-back, unattended.

Both jobs go through tools/run_detached.py exactly as any other detached
experiment would (each gets its own experiment directory with its own
command/config-snapshot/PID/status/log/checkpoint path) -- this script only
adds the "wait for DQN to finish, then launch PPO" sequencing, so the two
training jobs never contend for the GPU concurrently. This script is itself
meant to be launched detached (see tools/run_detached.py), so the sequencing
survives the launching session ending, same as the jobs it sequences.

Usage:
    python tools/run_detached.py --name stage5_sequencer -- \
        <python> tools/stage5_sequencer.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tooling_scripts.run_detached import _now, _read_meta, launch  # noqa: E402

PY = sys.executable

JOBS = [
    {
        "name": "dqn_s5_120k",
        "checkpoint_path": "training/dqn_bdd_s5_120k_final.zip",
        "command": [
            PY, "train.py",
            "--training", "configs/training/training_bdd.yaml",
            "--algorithm", "dqn",
            "--timesteps", "120000",
            "--hardware", "configs/hardware.yaml",
            "--log-name", "dqn_bdd_s5_120k",
            "--checkpoint-every", "20000",
            "--log-every", "5000",
        ],
    },
    {
        "name": "ppo_s5_300k",
        "checkpoint_path": "training/ppo_bdd_s5_300k_final.zip",
        "command": [
            PY, "train.py",
            "--training", "configs/training/training_ppo_bdd.yaml",
            "--algorithm", "ppo",
            "--timesteps", "300000",
            "--hardware", "configs/hardware.yaml",
            "--log-name", "ppo_bdd_s5_300k",
            "--checkpoint-every", "50000",
            "--log-every", "5000",
        ],
    },
]


def wait_for(run_dir: Path, poll_seconds: int = 30) -> dict:
    while True:
        meta = _read_meta(run_dir)
        if meta["status"] in ("finished", "failed"):
            return meta
        time.sleep(poll_seconds)


def main() -> None:
    print(f"=== training sequencer start {_now()} ===", flush=True)
    for job in JOBS:
        print(f"\n--- launching {job['name']} at {_now()} ---", flush=True)
        run_dir = launch(job["name"], job["command"], job["checkpoint_path"])
        print(f"run dir: {run_dir}", flush=True)
        meta = wait_for(run_dir)
        print(f"--- {job['name']} finished at {_now()}: status={meta['status']} "
              f"exit_code={meta['exit_code']} ---", flush=True)
        if meta["status"] != "finished":
            print(f"=== training sequencer ABORTING: {job['name']} did not finish "
                  f"successfully, skipping remaining jobs ===", flush=True)
            return
    print(f"\n=== training sequencer done {_now()}: all jobs finished successfully ===",
          flush=True)


if __name__ == "__main__":
    main()
