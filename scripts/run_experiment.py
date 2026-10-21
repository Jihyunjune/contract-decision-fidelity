"""Sequential, resumable reproduction; model stages never overlap in VRAM."""

from pathlib import Path
import argparse, sys, subprocess, time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import append_jsonl, now


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["smoke", "development", "sealed"], required=True)
    a = p.parse_args()
    methods = (
        ["original", "naive", "contextual", "long_30"]
        if a.split == "sealed"
        else ["original", "naive", "contextual", "long_15", "long_30"]
    )
    stages = (
        [("summary", m) for m in ["naive", "contextual"]]
        + [("compress", m) for m in methods if m.startswith("long_")]
        + [("judge", m) for m in methods]
    )
    for stage, method in stages:
        cmd = ["-m", "cdfv2.cli", "run", "--split", a.split, "--stage", stage, "--method", method]
        t = time.perf_counter()
        r = subprocess.run([sys.executable, *cmd], cwd=ROOT)
        append_jsonl(
            ROOT / f"results/{a.split}/execution_history.jsonl",
            {
                "command": cmd,
                "wall_seconds": time.perf_counter() - t,
                "exit_code": r.returncode,
                "time": now(),
            },
        )
        if r.returncode:
            raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
