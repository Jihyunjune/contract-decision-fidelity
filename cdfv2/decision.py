"""Same v1 NLI prompt/labels, one hypothesis per call for every v2 condition."""

import json
from cdfv2.common import config, text_hash
from src.decision.pipeline import messages, schema, parse, quote_check
from src.models.ollama import ContextOverflow


def decide(document, text, hypothesis, backend, method, upstream="SUCCESS"):
    attempts = []
    judgment = None
    status = upstream
    if upstream == "SUCCESS":
        for i in range(config()["decision_attempts"]):
            a = {"attempt": i + 1, "json_valid": False, "schema_valid": False}
            try:
                a.update(backend.generate(messages(text, [hypothesis]), schema([hypothesis])))
                json.loads(a["raw_output"])
                a["json_valid"] = True
                parsed = parse(a["raw_output"], [hypothesis.hypothesis_id])
                if a["thinking_output"].strip() or "<think>" in a["raw_output"]:
                    raise ValueError("Thinking in non-thinking response")
                if a["done_reason"] == "length":
                    raise ValueError("Generation truncated")
                judgment = parsed[hypothesis.hypothesis_id]
                a["schema_valid"] = True
                status = "SUCCESS"
            except ContextOverflow as e:
                status = "CONTEXT_OVERFLOW"
                a["error"] = str(e)
            except ValueError as e:
                status = "OUTPUT_FAILED"
                a["error"] = str(e)[:2000]
            attempts.append(a)
            if status in ["SUCCESS", "CONTEXT_OVERFLOW"]:
                break
    label = judgment.predicted_label if judgment and status == "SUCCESS" else None
    quote = judgment.evidence_quote if judgment and status == "SUCCESS" else ""
    return {
        "document_id": document.document_id,
        "hypothesis_id": hypothesis.hypothesis_id,
        "condition": method,
        "predicted_label": label,
        "evidence_quote": quote,
        "inference_status": status,
        "input_hash": text_hash(text),
        "model_id": backend.model,
        "model_digest": backend.identity["digest"],
        "prompt_version": "v1 unchanged; v2 single-hypothesis batch",
        "attempts": attempts,
        "latency": sum(a.get("latency", 0) for a in attempts),
        "input_tokens": sum(a.get("input_tokens") or 0 for a in attempts),
        "output_tokens": sum(a.get("output_tokens") or 0 for a in attempts),
        **quote_check(quote, text, label),
    }
