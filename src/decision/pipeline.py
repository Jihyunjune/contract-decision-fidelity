import json
from typing import Literal
from pydantic import BaseModel, ConfigDict
from src.common import ROOT, config, text_hash, digest
from src.data.inputs import InputDocument, Hypothesis
from src.models.ollama import ContextOverflow


class Judgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    hypothesis_id: str
    predicted_label: Literal["Entailment", "Contradiction", "NotMentioned"]
    evidence_quote: str


class DecisionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    predictions: list[Judgment]


def messages(text, hypotheses):
    return [
        {
            "role": "system",
            "content": (ROOT / "prompts/nli_decision_v1.txt").read_text(encoding="utf-8"),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"contract_text": text, "hypotheses": [h.model_dump() for h in hypotheses]},
                ensure_ascii=False,
            ),
        },
    ]


def schema(hypotheses):
    obj = DecisionOutput.model_json_schema()
    obj["$defs"]["Judgment"]["properties"]["hypothesis_id"]["enum"] = [
        h.hypothesis_id for h in hypotheses
    ]
    obj["properties"]["predictions"].update(minItems=len(hypotheses), maxItems=len(hypotheses))
    return obj


def parse(raw, ids):
    obj = DecisionOutput.model_validate_json(raw)
    found = [r.hypothesis_id for r in obj.predictions]
    if len(found) != len(set(found)) or set(found) != set(ids):
        raise ValueError("Duplicate, missing or foreign hypothesis ID")
    return {r.hypothesis_id: r for r in obj.predictions}


def quote_check(quote, text, label):
    if not quote:
        return {
            "evidence_valid": label == "NotMentioned",
            "evidence_error": None if label == "NotMentioned" else "missing_quote",
            "start_char": None,
            "end_char": None,
        }
    start = text.find(quote)
    return {
        "evidence_valid": start >= 0,
        "evidence_error": None if start >= 0 else "quote_not_substring",
        "start_char": start if start >= 0 else None,
        "end_char": start + len(quote) if start >= 0 else None,
    }
