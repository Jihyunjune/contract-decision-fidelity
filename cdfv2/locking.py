"""A reproduction lock, distinct from the historical experimental lock."""

from cdfv2.common import ROOT, read_json, immutable, sha, digest, now, config


def verify_assets():
    import llmlingua
    from pathlib import Path

    for role, obj in read_json(ROOT / "configs/tokenizers.json").items():
        for name, h in obj["files"].items():
            if sha(ROOT / "data/tokenizers" / role / name) != h:
                raise RuntimeError("Tokenizer asset mismatch")
    obj = read_json(ROOT / "configs/compressor_assets.json")
    for name, h in obj["files_sha256"].items():
        if sha(ROOT / name) != h:
            raise RuntimeError("Compressor asset mismatch")
    library = Path(llmlingua.__file__).parent
    for name, h in obj["library_sha256"].items():
        if sha(library / name) != h:
            raise RuntimeError("Official LLMLingua library changed")


def protected_files():
    paths = []
    for folder in ["src", "cdfv2", "configs", "prompts"]:
        paths.extend(
            p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts
        )
    paths.extend(
        ROOT / p for p in read_json(ROOT / "results/manifests/dataset_seal.json")["files_sha256"]
    )
    paths.extend([ROOT / "results/manifests/dataset_seal.json", ROOT / "requirements.txt"])
    for role, c in read_json(ROOT / "configs/tokenizers.json").items():
        paths.extend(ROOT / "data/tokenizers" / role / name for name in c["files"])
    paths.extend(
        ROOT / p for p in read_json(ROOT / "configs/compressor_assets.json")["files_sha256"]
    )
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def create_lock():
    from cdfv2.runtime import Ollama
    import importlib.metadata

    path = ROOT / "results/manifests/reproduction_lock.json"
    if path.exists():
        raise RuntimeError("Reproduction lock already exists; verify or use a fresh checkout")
    for role in ["summary", "decision"]:
        Ollama(role)
    verify_assets()
    for p, h in read_json(ROOT / "results/manifests/dataset_seal.json")["files_sha256"].items():
        if sha(ROOT / p) != h:
            raise RuntimeError("Data seal mismatch")
    files = protected_files()
    obj = {
        "created_utc": now(),
        "kind": "public packaging reproduction, NOT a new independent evaluation",
        "methods": ["original", "naive", "contextual", "long_30"],
        "files_sha256": files,
        "implementation_set_sha256": digest(files),
        "endpoint": config()["endpoint"],
        "packages": {
            n: importlib.metadata.version(n)
            for n in ["torch", "llmlingua", "transformers", "numpy", "pydantic", "bitsandbytes"]
        },
    }
    immutable(path, obj)
    immutable(ROOT / "results/manifests/reproduction_lock_sha256.json", {"sha256": sha(path)})
    return {"lock_sha256": sha(path)}


def verify_lock():
    import importlib.metadata

    p = ROOT / "results/manifests/reproduction_lock.json"
    r = read_json(p)
    if sha(p) != read_json(ROOT / "results/manifests/reproduction_lock_sha256.json")["sha256"]:
        raise RuntimeError("Reproduction lock changed")
    for name, h in r["files_sha256"].items():
        if sha(ROOT / name) != h:
            raise RuntimeError(f"Protected file changed: {name}")
    for n, v in r["packages"].items():
        if importlib.metadata.version(n) != v:
            raise RuntimeError(f"Package changed: {n}")
    if config()["endpoint"] != r["endpoint"]:
        raise RuntimeError("Runtime endpoint changed")
    verify_assets()
    return r
