"""Inference-only inputs: never imports evaluation references."""

import random
from cdfv2.common import ROOT, config, read_json, read_jsonl, sha, immutable
from src.data.inputs import InputDocument, Hypothesis


def documents(split):
    if split not in ["development", "sealed", "smoke"]:
        raise ValueError(split)
    name = "development" if split == "smoke" else split
    rows = [InputDocument.model_validate(r) for r in read_jsonl(ROOT / f"data/{name}_inputs.jsonl")]
    seal = read_json(ROOT / "results/manifests/dataset_seal.json")
    p = f"data/{name}_inputs.jsonl"
    if sha(ROOT / p) != seal["files_sha256"][p]:
        raise RuntimeError("Sealed input changed")
    if name == "development":
        rng = random.Random(config()["development_seed"])
        rng.shuffle(rows)
        rows = rows[: config()["development_sample_n"]]
        immutable(
            ROOT / "results/manifests/development_sample.json",
            {
                "seed": config()["development_seed"],
                "ids": [d.document_id for d in rows],
                "pool_n": len(seal["development_ids"]),
                "reason": "Fixed 24-contract development evaluation (408 hypotheses per method) limits per-hypothesis 7B compressor cost; full 100-contract sealed evaluation remains unchanged. Selected before model results.",
            },
        )
    return rows[:2] if split == "smoke" else rows


def hypotheses():
    obj = read_json(ROOT / "data/hypotheses.json")
    return [Hypothesis.model_validate(r) for r in obj]
