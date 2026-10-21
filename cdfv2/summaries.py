"""Schema-first summaries; only presentation normalization, never text deletion."""

import json, re
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field
from cdfv2.common import ROOT, config, words, text_hash
from src.models.tokenization import count
from src.models.ollama import ContextOverflow


class SummaryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    bullets: Annotated[list[str], Field(min_length=1, max_length=20)]


def normalize_bullet(s):
    # Leading presentation markers and line wrapping only. All content words remain.
    s = s.strip()
    s = re.sub(r"^(?:[-*\u2022]\s+|\d+[.)]\s+)", "", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_summary(raw):
    raw = raw.strip()
    if raw.startswith("```") and raw.endswith("```"):
        lines = raw.splitlines()
        if lines[0].lower() in ["```", "```json"]:
            raw = "\n".join(lines[1:-1])
    result = SummaryOutput.model_validate_json(raw)
    bullets = [normalize_bullet(s) for s in result.bullets]
    if any(not s for s in bullets):
        raise ValueError("Empty bullet")
    return "\n".join("- " + s for s in bullets)


def messages(text, method, feedback=None):
    prompt = (ROOT / f"prompts/{method}_summary_v2.txt").read_text(encoding="utf-8")
    if feedback:
        prompt += (
            "\nYour previous attempt violated these formal constraints: "
            + ", ".join(feedback)
            + ". Regenerate from the complete contract, using fewer bullets and at most 220 words. Preserve factual qualifiers; do not append commentary."
        )
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "<contract>\n" + text + "\n</contract>"},
    ]


def summarize(document, method, backend):
    attempts = []
    summary = ""
    errors = []
    for i in range(config()["summary_attempts"]):
        a = {"attempt": i + 1, "json_valid": False, "schema_valid": False}
        feedback = errors.copy()
        try:
            a.update(
                backend.generate(
                    messages(document.original_text, method, feedback),
                    SummaryOutput.model_json_schema(),
                )
            )
            json.loads(a["raw_output"].strip().removeprefix("```json").removesuffix("```").strip())
            a["json_valid"] = True
            summary = parse_summary(a["raw_output"])
            a["schema_valid"] = True
            errors = []
            if len(words(summary)) > config()["max_words"]:
                errors.append("word_budget_exceeded")
            if count(summary) >= count(document.original_text):
                errors.append("not_compressed_tokens")
            if a["done_reason"] == "length":
                errors.append("generation_truncated")
        except ContextOverflow as e:
            errors = ["context_overflow"]
            a["error"] = str(e)
        except ValueError as e:
            errors = ["output_format"]
            a["error"] = str(e)[:2000]
        a.update(validation_errors=errors.copy(), feedback=feedback)
        attempts.append(a)
        if not errors or errors == ["context_overflow"]:
            break
    return {
        "document_id": document.document_id,
        "method": method,
        "summary": summary,
        "status": "SUCCESS" if not errors else "SUMMARY_FAILED",
        "errors": errors,
        "attempts": attempts,
        "input_hash": text_hash(document.original_text),
        "summary_hash": text_hash(summary),
        "original_tokens": count(document.original_text),
        "summary_tokens": count(summary),
        "summary_words": len(words(summary)),
        "compression_ratio": count(summary) / count(document.original_text),
        "latency": sum(a.get("latency", 0) for a in attempts),
        "normalization": "Whitespace and leading bullet markers only; no sentence deletion or string truncation",
        "model_id": backend.model,
        "model_digest": backend.identity["digest"],
    }
