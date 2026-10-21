"""The only experiment module reading original-contract gold references."""

import numpy as np, pandas as pd
from cdfv2.common import *
from cdfv2.inputs import documents, hypotheses
from src.evaluation.metrics import evaluate_arrays


def evaluate(split, methods):
    docs = documents(split)
    hyps = hypotheses()
    ids = [d.document_id for d in docs]
    hids = [h.hypothesis_id for h in hyps]
    expected = {(i, h) for i in ids for h in hids}
    base = ROOT / f"results/{split}"
    allrows = {}
    for m in methods:
        rows = read_jsonl(base / f"{m}_predictions.jsonl")
        keyed = {(r["document_id"], r["hypothesis_id"]): r for r in rows}
        if len(rows) != len(keyed) or set(keyed) != expected:
            raise RuntimeError(f"Incomplete or duplicate ID set: {m}")
        allrows[m] = keyed
    refs = {
        (r["document_id"], r["hypothesis_id"]): r
        for r in read_jsonl(ROOT / "data/evaluation_references.jsonl")
    }
    gold = np.array([[LABELS.index(refs[i, h]["gold_label"]) for h in hids] for i in ids])
    pred = {
        m: np.array(
            [
                [
                    LABELS.index(rows[i, h]["predicted_label"])
                    if rows[i, h]["predicted_label"] in LABELS
                    else -1
                    for h in hids
                ]
                for i in ids
            ]
        )
        for m, rows in allrows.items()
    }
    b = config()["bootstrap_replicates"]
    seed = config()["bootstrap_seed"]
    result = evaluate_arrays(gold, pred, b, seed)
    for m in methods:
        if not m.startswith("long_"):
            continue
        extra = evaluate_arrays(
            gold,
            {"original": pred["original"], "naive": pred["naive"], "contextual": pred[m]},
            b,
            seed,
        )
        result["paired_ci95"].extend(
            [
                {**r, "contrast": r["contrast"].replace("contextual", m)}
                for r in extra["paired_ci95"]
                if r["contrast"].startswith("contextual")
            ]
        )
    result.update(
        split=split,
        documents=len(ids),
        hypotheses=len(hids),
        created_utc=now(),
        dataset_seal_sha256=sha(ROOT / "results/manifests/dataset_seal.json"),
    )
    transitions = []
    history = read_jsonl(base / "execution_history.jsonl")
    for m, rows in allrows.items():
        metric = result["methods"][m]
        rr = list(rows.values())
        attempts = [a for r in rr for a in r["attempts"]]
        metric.update(
            output_failure_rate=metric["failure_n"] / metric["n"],
            nli_latency_seconds=sum(r["latency"] for r in rr),
            nli_input_tokens=sum(r["input_tokens"] for r in rr),
            nli_output_tokens=sum(r["output_tokens"] for r in rr),
            nli_attempts=len(attempts),
            first_attempt_valid_rate=float(
                np.mean([r["attempts"][0]["schema_valid"] for r in rr if r["attempts"]])
            )
            if attempts
            else None,
            exact_quote_valid_rate=float(
                np.mean([r["evidence_valid"] for r in rr if r["evidence_quote"]])
            )
            if any(r["evidence_quote"] for r in rr)
            else None,
        )
        if m == "original":
            ratios = [1.0] * len(docs)
            pre_seconds = 0.0
            pre_in = pre_out = forward = 0
            context_tokens = sum(d.original_token_count for d in docs) * 17
        elif m in ["naive", "contextual"]:
            ss = read_jsonl(base / f"{m}_summaries.jsonl")
            ratios = [s["compression_ratio"] for s in ss]
            aa = [a for s in ss for a in s["attempts"]]
            pre_seconds = sum(s["latency"] for s in ss)
            pre_in = sum(a.get("input_tokens") or 0 for a in aa)
            pre_out = sum(a.get("output_tokens") or 0 for a in aa)
            forward = 0
            context_tokens = sum(s["summary_tokens"] * 17 for s in ss)
            metric.update(
                summary_failures=sum(s["status"] != "SUCCESS" for s in ss),
                summary_attempts=len(aa),
                summary_mean_words=float(np.mean([s["summary_words"] for s in ss])),
                summary_max_words=max(s["summary_words"] for s in ss),
                summary_first_attempt_valid_rate=float(
                    np.mean([not s["attempts"][0]["validation_errors"] for s in ss])
                ),
                summary_errors=dict(
                    __import__("collections").Counter(e for s in ss for e in s["errors"])
                ),
            )
        else:
            cs = read_jsonl(base / f"{m}_contexts.jsonl")
            ratios = [c["compression_ratio"] for c in cs]
            pre_seconds = sum(c["latency"] for c in cs)
            pre_in = pre_out = 0
            forward = sum(c["compressor_forward_tokens"] for c in cs)
            context_tokens = sum(c["compressed_tokens"] for c in cs)
            metric["compressor_peak_gpu_bytes"] = max(c["peak_gpu_allocated_bytes"] for c in cs)
            metric["compressor_load_seconds_per_process"] = cs[0]["model_load_seconds"]
        metric.update(
            compression_ratio={
                "mean": float(np.mean(ratios)),
                "median": float(np.median(ratios)),
                "p25": float(np.percentile(ratios, 25)),
                "p75": float(np.percentile(ratios, 75)),
            },
            preparation_latency_seconds=pre_seconds,
            preparation_input_tokens=pre_in,
            preparation_output_tokens=pre_out,
            compressor_forward_tokens=forward,
            accumulated_context_tokens_all_17=context_tokens,
            total_latency_seconds=pre_seconds + metric["nli_latency_seconds"],
            total_ollama_tokens=pre_in
            + pre_out
            + metric["nli_input_tokens"]
            + metric["nli_output_tokens"],
            mean_nli_latency=float(np.mean([r["latency"] for r in rr])),
            p95_nli_latency=float(np.percentile([r["latency"] for r in rr], 95)),
            external_api_cost_usd=0,
            electricity_cost="not measured",
        )
        stages = [
            x
            for x in history
            if "--method" in x["command"] and x["command"][x["command"].index("--method") + 1] == m
        ]
        metric["observed_stage_wall_seconds_including_process_and_model_load"] = (
            sum(x["wall_seconds"] for x in stages) if stages else None
        )
        # Semantic comparison on identical complete cases, to isolate failure effects.
        complete = (pred["original"] >= 0) & (pred["naive"] >= 0) & (pred[m] >= 0)
        metric["common_valid_with_original_naive"] = {
            "n": int(complete.sum()),
            "accuracy": float((pred[m][complete] == gold[complete]).mean())
            if complete.any()
            else None,
            "naive_accuracy": float((pred["naive"][complete] == gold[complete]).mean())
            if complete.any()
            else None,
            "original_accuracy": float((pred["original"][complete] == gold[complete]).mean())
            if complete.any()
            else None,
        }
        for i in ids:
            for h in hids:
                p = rows[i, h]["predicted_label"]
                o = allrows["original"][i, h]["predicted_label"]
                g = refs[i, h]["gold_label"]
                transitions.append(
                    {
                        "document_id": i,
                        "hypothesis_id": h,
                        "method": m,
                        "gold": g,
                        "original": o,
                        "prediction": p,
                        "flipped": o != p if o is not None and p is not None else None,
                        "error_introduced": o == g and p != g,
                        "error_corrected": o != g and p == g,
                        "status": rows[i, h]["inference_status"],
                    }
                )
    write_json(ROOT / f"results/metrics/{split}.json", result)
    frame = pd.DataFrame(
        [
            {
                "method": m,
                **{
                    k: v[k]
                    for k in [
                        "strict_accuracy",
                        "valid_accuracy",
                        "macro_f1",
                        "coverage",
                        "decision_flip_rate",
                        "error_introduced_rate_all",
                        "output_failure_rate",
                        "total_latency_seconds",
                        "total_ollama_tokens",
                        "compressor_forward_tokens",
                    ]
                },
                "compression_ratio": v["compression_ratio"]["mean"],
            }
            for m, v in result["methods"].items()
        ]
    )
    frame.to_csv(ROOT / f"results/metrics/{split}_comparison.csv", index=False)
    pd.DataFrame(transitions).to_csv(ROOT / f"results/metrics/{split}_transitions.csv", index=False)
    print(frame.to_string(index=False))
    return result
