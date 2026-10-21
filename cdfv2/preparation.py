"""Official downloads and deterministic reconstruction of the historical partition."""

import io
import zipfile
import requests
from cdfv2.common import ROOT, config, read_json, write_json, write_jsonl, read_jsonl, sha


def download_data():
    c = read_json(ROOT / "configs/dataset_download.json")
    path = ROOT / "data/raw/contract-nli.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        r = requests.get(c["url"], timeout=180)
        r.raise_for_status()
        import hashlib

        if hashlib.sha256(r.content).hexdigest() != c["sha256"]:
            raise RuntimeError("Official archive hash changed; do not silently use a new version")
        path.write_bytes(r.content)
    if sha(path) != c["sha256"]:
        raise RuntimeError("Archive integrity mismatch")
    with zipfile.ZipFile(path) as z:
        for name in ["train.json", "dev.json", "test.json", "LICENSE", "TERMS", "README.md"]:
            members = [n for n in z.namelist() if n.rsplit("/", 1)[-1] == name]
            if len(members) != 1:
                if name in ["TERMS", "README.md"]:
                    continue
                raise RuntimeError(f"Unexpected archive member: {name}")
            destination = path.parent / name
            content = z.read(members[0])
            if destination.exists() and destination.read_bytes() != content:
                raise RuntimeError(f"Refusing to overwrite changed raw file: {name}")
            destination.write_bytes(content)
    return {"archive_sha256": sha(path), "source": c["url"]}


def assets(compressor=False):
    from huggingface_hub import snapshot_download

    for role, c in read_json(ROOT / "configs/tokenizers.json").items():
        folder = ROOT / "data/tokenizers" / role
        snapshot_download(
            c["repository"],
            revision=c["revision"],
            local_dir=folder,
            allow_patterns=list(c["files"]),
            max_workers=2,
        )
        for name, expected in c["files"].items():
            if sha(folder / name) != expected:
                raise RuntimeError("Tokenizer asset mismatch")
    if compressor:
        c = config()["compressor"]
        snapshot_download(
            c["model_name"],
            revision=c["revision"],
            local_dir=ROOT / "models/llama2-chat",
            allow_patterns=["*.safetensors", "*.json", "*.model", "*.md", "LICENSE*"],
            max_workers=2,
        )
        for name, expected in read_json(ROOT / "configs/compressor_assets.json")[
            "files_sha256"
        ].items():
            if sha(ROOT / name) != expected:
                raise RuntimeError("Compressor asset mismatch")


def audit_and_prepare():
    from src.data.audit import audit

    result = audit()
    train = read_jsonl(ROOT / "data/processed/train_inputs.jsonl")
    byid = {d["document_id"]: d for d in train}
    seal = read_json(ROOT / "results/manifests/dataset_seal.json")
    for split in ["development", "sealed"]:
        write_jsonl(ROOT / f"data/{split}_inputs.jsonl", [byid[i] for i in seal[f"{split}_ids"]])
    refs = [
        r
        for r in read_jsonl(ROOT / "data/processed/evaluation_references.jsonl")
        if r["document_id"] in byid
    ]
    write_jsonl(ROOT / "data/evaluation_references.jsonl", refs)
    (ROOT / "data/hypotheses.json").write_bytes(
        (ROOT / "data/processed/hypotheses.json").read_bytes()
    )
    for p, expected in seal["files_sha256"].items():
        if sha(ROOT / p) != expected:
            raise RuntimeError(f"Historical data projection mismatch: {p}")
    return {"documents": result["total_documents"], "historical_partition_hashes_match": True}
