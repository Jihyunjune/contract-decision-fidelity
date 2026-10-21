"""Verify the published release inventory, excluding its intentionally unhashed self."""

from pathlib import Path
import hashlib, json

ROOT = Path(__file__).resolve().parents[1]


def main():
    r = json.loads((ROOT / "results/manifests/public_files.json").read_text(encoding="utf-8"))
    for name, h in r["files_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != h:
            raise RuntimeError("Release file changed: " + name)
    print(f"Verified {len(r['files_sha256'])} release files; inventory excludes itself.")


if __name__ == "__main__":
    main()
