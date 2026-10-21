"""Per-item durable checkpoints; inference never reads gold references."""

import json, time, traceback
from cdfv2.common import *
from cdfv2.inputs import documents, hypotheses
from cdfv2.runtime import Ollama
from cdfv2.summaries import summarize
from cdfv2.decision import decide


def fingerprint():
    paths = [
        *[
            ROOT / f"cdfv2/{n}.py"
            for n in [
                "__init__",
                "common",
                "inputs",
                "runtime",
                "summaries",
                "decision",
                "compression",
                "runner",
            ]
        ],
        *sorted((ROOT / "prompts").glob("*.txt")),
        ROOT / "configs/experiment.json",
    ]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def check_start(split, method):
    if split == "sealed":
        from cdfv2.locking import verify_lock

        lock = verify_lock()
        if method not in lock["methods"]:
            raise ValueError("Method was not selected before sealed evaluation")


def run_summaries(split, method):
    check_start(split, method)
    docs = documents(split)
    base = ROOT / f"results/{split}"
    path = base / f"{method}_summaries.jsonl"
    backend = Ollama("summary")
    immutable(
        base / f"{method}_summary_manifest.json",
        {
            "fingerprint": fingerprint(),
            "ids": [d.document_id for d in docs],
            "model": backend.identity,
            "runtime": backend.runtime,
        },
    )
    saved = read_jsonl(path)
    done = {r["document_id"]: r for r in saved}
    if len(saved) != len(done):
        raise RuntimeError("Duplicate checkpoint")
    try:
        for i, d in enumerate(docs):
            if d.document_id in done:
                assert done[d.document_id]["input_hash"] == text_hash(d.original_text)
                continue
            r = summarize(d, method, backend)
            append_jsonl(path, r)
            print(
                json.dumps(
                    {
                        "stage": split + "/" + method + "/summarize",
                        "n": i + 1,
                        "total": len(docs),
                        "status": r["status"],
                        "attempts": len(r["attempts"]),
                    }
                ),
                flush=True,
            )
    finally:
        backend.unload()


def run_compression(split, rate):
    check_start(split, f"long_{int(rate * 100)}")
    from cdfv2.compression import Compressor

    docs = documents(split)
    hyps = hypotheses()
    method = f"long_{int(rate * 100)}"
    base = ROOT / f"results/{split}"
    path = base / f"{method}_contexts.jsonl"
    immutable(
        base / f"{method}_compress_manifest.json",
        {
            "fingerprint": fingerprint(),
            "ids": [d.document_id for d in docs],
            "rate": rate,
            "model": config()["compressor"],
        },
    )
    saved = read_jsonl(path)
    done = {(r["document_id"], r["hypothesis_id"]): r for r in saved}
    if len(saved) != len(done):
        raise RuntimeError("Duplicate compression checkpoint")
    engine = Compressor()
    try:
        for d in docs:
            for h in hyps:
                if (d.document_id, h.hypothesis_id) in done:
                    assert done[d.document_id, h.hypothesis_id]["input_hash"] == text_hash(
                        d.original_text
                    )
                    continue
                try:
                    r = engine.compress(d, h, rate)
                except Exception as e:
                    append_jsonl(
                        base / "execution_errors.jsonl",
                        {
                            "time": now(),
                            "stage": method,
                            "id": d.document_id,
                            "hypothesis": h.hypothesis_id,
                            "error": repr(e),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    raise  # Infrastructure/implementation errors stop, never fabricate contexts.
                append_jsonl(path, r)
                done[d.document_id, h.hypothesis_id] = r
                print(
                    json.dumps(
                        {
                            "stage": split + "/" + method + "/compress",
                            "n": len(done),
                            "total": len(docs) * 17,
                            "seconds": round(r["latency"], 3),
                        }
                    ),
                    flush=True,
                )
    finally:
        engine.close()


def run_decisions(split, method):
    check_start(split, method)
    docs = documents(split)
    hyps = hypotheses()
    base = ROOT / f"results/{split}"
    path = base / f"{method}_predictions.jsonl"
    sums = {}
    contexts = {}
    if method in ["naive", "contextual"]:
        sums = {r["document_id"]: r for r in read_jsonl(base / f"{method}_summaries.jsonl")}
    elif method.startswith("long_"):
        contexts = {
            (r["document_id"], r["hypothesis_id"]): r
            for r in read_jsonl(base / f"{method}_contexts.jsonl")
        }
    elif method != "original":
        raise ValueError(method)
    backend = Ollama("decision")
    immutable(
        base / f"{method}_decision_manifest.json",
        {
            "fingerprint": fingerprint(),
            "ids": [d.document_id for d in docs],
            "model": backend.identity,
            "runtime": backend.runtime,
            "batch_size": 1,
        },
    )
    saved = read_jsonl(path)
    done = {(r["document_id"], r["hypothesis_id"]): r for r in saved}
    if len(saved) != len(done):
        raise RuntimeError("Duplicate prediction checkpoint")
    try:
        for d in docs:
            for h in hyps:
                status = "SUCCESS"
                if method == "original":
                    text = d.original_text
                elif method in ["naive", "contextual"]:
                    s = sums[d.document_id]
                    text = s["summary"]
                    status = "SUCCESS" if s["status"] == "SUCCESS" else "UPSTREAM_SUMMARY_FAILED"
                else:
                    s = contexts[d.document_id, h.hypothesis_id]
                    text = s["compressed_context"]
                    status = (
                        "SUCCESS" if s["status"] == "SUCCESS" else "UPSTREAM_COMPRESSION_FAILED"
                    )
                if (d.document_id, h.hypothesis_id) in done:
                    assert done[d.document_id, h.hypothesis_id]["input_hash"] == text_hash(text)
                    continue
                r = decide(d, text, h, backend, method, status)
                append_jsonl(path, r)
                done[d.document_id, h.hypothesis_id] = r
                print(
                    json.dumps(
                        {
                            "stage": split + "/" + method + "/judge",
                            "n": len(done),
                            "total": len(docs) * 17,
                            "status": r["inference_status"],
                        }
                    ),
                    flush=True,
                )
    finally:
        backend.unload()
