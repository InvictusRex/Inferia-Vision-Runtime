"""Windows-compatible detached/background experiment runner.

Launches a long-running command (e.g. `python train.py ...`) as a background
process that survives the launching shell/session ending, and persists
everything needed to review or resume the run later:

- the exact command line as invoked
- a snapshot of configs/ as it existed at launch time
- the process PID
- status (starting / running / finished / failed)
- start and finish timestamps
- incrementally-written logs (not buffered until exit)
- the checkpoint path(s) the run is expected to produce (as given by the caller)
- the final exit code

Usage:
    python tools/run_detached.py --name dqn_p2b_val -- ^
        python train.py --algorithm dqn --timesteps 20000 --hardware configs/hardware.yaml

    python tools/run_detached.py --status runs/experiments/<run_dir>

Each run gets its own directory under runs/experiments/<timestamp>_<name>/:
    meta.json          -- status, PID, timestamps, command, exit code
    command.txt         -- the exact command line
    log.txt             -- incrementally-written stdout+stderr of the child
    configs_snapshot/   -- copy of configs/ at launch time
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs" / "experiments"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _write_meta(run_dir: Path, meta: dict) -> None:
    tmp = run_dir / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(run_dir / "meta.json")


def _read_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))


def _detach_kwargs() -> dict:
    if sys.platform == "win32":
        return {
            "creationflags": (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NO_WINDOW
            ),
            "close_fds": True,
        }
    return {"start_new_session": True}


def _worker(run_dir: Path) -> None:
    """Runs inside the detached background process.

    Executes the recorded command, blocks on it, and updates meta.json as it
    progresses. This function's own process lifetime -- not the CLI entry
    point below, which returns immediately -- is what survives the launching
    shell/session ending.
    """
    meta = _read_meta(run_dir)
    cmd = meta["command"]
    log_path = run_dir / "log.txt"
    with open(log_path, "a", buffering=1, encoding="utf-8") as log_fh:
        log_fh.write(f"\n=== worker start {_now()} ===\ncommand: {cmd}\ncwd: {meta['cwd']}\n\n")
        log_fh.flush()
        try:
            proc = subprocess.Popen(
                cmd, cwd=meta["cwd"], stdout=log_fh, stderr=subprocess.STDOUT, **_detach_kwargs()
            )
        except OSError as exc:
            meta["status"] = "failed"
            meta["exit_code"] = None
            meta["end_time"] = _now()
            meta["error"] = str(exc)
            _write_meta(run_dir, meta)
            log_fh.write(f"\n=== failed to start: {exc} ===\n")
            return
        meta["pid"] = proc.pid
        meta["status"] = "running"
        _write_meta(run_dir, meta)
        exit_code = proc.wait()
        meta["exit_code"] = exit_code
        meta["status"] = "finished" if exit_code == 0 else "failed"
        meta["end_time"] = _now()
        _write_meta(run_dir, meta)
        log_fh.write(
            f"\n=== worker end {_now()} exit_code={exit_code} status={meta['status']} ===\n"
        )


def launch(name: str, command: list[str], checkpoint_path: str | None) -> Path:
    if not command:
        raise SystemExit("no command given -- pass it after `--`")
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{name}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    configs_dir = ROOT / "configs"
    if configs_dir.is_dir():
        shutil.copytree(configs_dir, run_dir / "configs_snapshot")

    (run_dir / "command.txt").write_text(" ".join(command), encoding="utf-8")

    meta = {
        "run_id": run_id,
        "name": name,
        "command": command,
        "cwd": str(ROOT),
        "checkpoint_path": checkpoint_path,
        "pid": None,
        "status": "starting",
        "start_time": _now(),
        "end_time": None,
        "exit_code": None,
        "log_path": str(run_dir / "log.txt"),
    }
    _write_meta(run_dir, meta)

    worker_cmd = [sys.executable, str(Path(__file__).resolve()), "--_worker", str(run_dir)]
    subprocess.Popen(
        worker_cmd,
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **_detach_kwargs(),
    )
    return run_dir


def print_status(run_dir: Path) -> None:
    print(json.dumps(_read_meta(run_dir), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch a detached background experiment (Windows-compatible)"
    )
    parser.add_argument("--name", default="run", help="short label, part of the run dir name")
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="checkpoint path this run is expected to produce (recorded in meta.json)",
    )
    parser.add_argument("--status", default=None, help="print meta.json for a run dir and exit")
    parser.add_argument("--_worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command to run, after `--`")
    args = parser.parse_args()

    if args._worker:
        _worker(Path(args._worker))
        return

    if args.status:
        print_status(Path(args.status))
        return

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    run_dir = launch(args.name, command, args.checkpoint_path)
    print(f"launched: {run_dir}")
    print(f"log:      {run_dir / 'log.txt'}")
    print(f"status:   python tools/run_detached.py --status {run_dir}")


if __name__ == "__main__":
    main()
