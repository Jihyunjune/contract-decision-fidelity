"""Run offline checks against staged files alone, without data or model directories."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = (
        subprocess.check_output(["git", "ls-files", "--cached", "-z"], cwd=ROOT)
        .decode()
        .split("\0")[:-1]
    )
    audit = ROOT / ".release_audit"
    audit.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="clean-", dir=audit) as tmp:
        dest = Path(tmp)
        for name in files:
            path = dest / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(subprocess.check_output(["git", "show", ":" + name], cwd=ROOT))
        env = os.environ.copy()
        env["PYTHONPATH"] = str(dest)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        code = "from pathlib import Path; import src,cdfv2; root=Path.cwd(); assert Path(src.__file__).parent.parent==root; assert Path(cdfv2.__file__).parent.parent==root; print('Imports resolved only inside clean export')"
        commands = [
            ["-c", code],
            ["-m", "cdfv2.cli", "--help"],
            ["-m", "cdfv2.cli", "verify"],
            ["-m", "pytest", "-q"],
        ]
        outputs = []
        for args in commands:
            r = subprocess.run(
                [sys.executable, *args],
                cwd=dest,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            outputs.append(
                {"args": args, "exit_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
            )
            if r.returncode:
                print(r.stdout, r.stderr)
                raise RuntimeError("Clean export check failed")
        (audit / "clean_export.json").write_text(
            json.dumps({"passed": True, "files": len(files), "commands": outputs}, indent=2),
            encoding="utf-8",
        )
        print(outputs[-1]["stdout"])
        print("Independent clean export passed; no data/models or sibling imports.")


if __name__ == "__main__":
    main()
