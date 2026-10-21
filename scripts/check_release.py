"""Inspect exactly the staged Git distribution. Never print matched secret values."""

from pathlib import Path
import subprocess, json, re, hashlib, argparse

ROOT = Path(__file__).resolve().parents[1]


def staged_files():
    return (
        subprocess.check_output(["git", "ls-files", "--cached", "-z"], cwd=ROOT)
        .decode()
        .split("\0")[:-1]
    )


def scan():
    files = staged_files()
    findings = []
    total = 0
    patterns = {
        "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "github_token": r"\bgh[pousr]_[A-Za-z0-9]{30,}\b",
        "huggingface_token": r"\bhf_[A-Za-z0-9]{25,}\b",
        "openai_style_key": r"\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b",
        "windows_absolute_path": r"\b[A-Za-z]:[\\/]",
        "posix_home_path": r"/(?:Users|home)/[A-Za-z0-9_.-]+/",
        "credential_assignment": r"""(?i)(?:api_key|access_token|password)\s*[=:]\s*["'][^"'\s]{12,}["']""",
    }
    forbidden = {"data", "models", "runtime", "wheels", ".venv", ".release_audit", "__pycache__"}
    hashes = {}
    for name in files:
        # Inspect index bytes, not only working-tree copies.
        b = subprocess.check_output(["git", "show", ":" + name], cwd=ROOT)
        total += len(b)
        hashes[name] = hashlib.sha256(b).hexdigest()
        if (
            Path(name).parts[0] in forbidden
            or "__pycache__" in Path(name).parts
            or name == ".env"
            or Path(name).suffix.lower() in [".gguf", ".safetensors", ".bin", ".pt", ".pth", ".zip"]
        ):
            findings.append({"file": name, "kind": "forbidden_distribution"})
        if len(b) > 5 * 1024 * 1024:
            findings.append({"file": name, "kind": "over_5_MiB"})
        if name.endswith((".png", ".jpg", ".jpeg", ".gif")):
            continue
        text = b.decode("utf-8")
        for kind, pattern in patterns.items():
            if re.search(pattern, text):
                findings.append({"file": name, "kind": kind})
    probes = [
        ".env",
        ".venv/x",
        "data/raw/train.json",
        "models/model.safetensors",
        "runtime/ollama.exe",
        "wheels/x.whl",
        "results/sealed/original_predictions.jsonl",
        ".release_audit/log.txt",
    ]
    ignored = {
        p: subprocess.run(["git", "check-ignore", "-q", p], cwd=ROOT).returncode == 0
        for p in probes
    }
    return {
        "staged_file_count": len(files),
        "total_bytes": total,
        "findings": findings,
        "ignore_probes": ignored,
        "passed": not findings and all(ignored.values()),
        "files_sha256": hashes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/security_scan.json")
    a = parser.parse_args()
    r = scan()
    dest = ROOT / a.output
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in r.items() if k != "files_sha256"}, indent=2))
    if not r["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
