"""Evaluation sequencer: waits for the training sequencer to finish
successfully, then runs the full-306 evaluation, then the 100-video oracle
sweep, back to back. Meant to be launched detached (see tools/run_detached.py)
so it survives the launching session ending, same as the jobs it sequences.

Usage:
    python tools/run_detached.py --name eval_sequencer -- \
        <python> tools/stage5b_sequencer.py <path to training sequencer run dir>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.run_detached import _now, _read_meta, launch  # noqa: E402

PY = sys.executable

JOBS = [
    {
        "name": "full_306_eval",
        "checkpoint_path": None,
        "command": [PY, "tools/stage5b_full_eval_parallel.py"],
    },
    {
        "name": "oracle_100v_sweep",
        "checkpoint_path": None,
        "command": [PY, "tools/stage5b_oracle_100v_parallel.py"],
    },
]


def wait_for(run_dir: Path, poll_seconds: int = 30) -> dict:
    while True:
        meta = _read_meta(run_dir)
        if meta["status"] in ("finished", "failed"):
            return meta
        time.sleep(poll_seconds)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: stage5b_sequencer.py <training sequencer run dir>")
    training_dir = Path(sys.argv[1])

    print(f"=== evaluation sequencer start {_now()} ===", flush=True)
    print(f"waiting on training sequencer: {training_dir}", flush=True)
    training_meta = wait_for(training_dir)
    print(f"training sequencer finished at {_now()}: status={training_meta['status']} "
          f"exit_code={training_meta['exit_code']}", flush=True)
    if training_meta["status"] != "finished":
        print("=== evaluation sequencer ABORTING: training did not finish successfully ===",
              flush=True)
        return

    for job in JOBS:
        print(f"\n--- launching {job['name']} at {_now()} ---", flush=True)
        run_dir = launch(job["name"], job["command"], job["checkpoint_path"])
        print(f"run dir: {run_dir}", flush=True)
        meta = wait_for(run_dir)
        print(f"--- {job['name']} finished at {_now()}: status={meta['status']} "
              f"exit_code={meta['exit_code']} ---", flush=True)
        if meta["status"] != "finished":
            print(f"=== evaluation sequencer ABORTING: {job['name']} did not finish "
                  f"successfully, skipping remaining jobs ===", flush=True)
            return

    print(f"\n=== evaluation sequencer done {_now()}: all jobs finished successfully ===",
          flush=True)


if __name__ == "__main__":
    main()
