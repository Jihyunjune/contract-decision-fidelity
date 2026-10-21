"""Inference-facing data only. No annotation or evidence-reference loader."""

from pydantic import BaseModel, ConfigDict
from src.common import ROOT, read_json, read_jsonl


class InputDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    document_id: int
    original_text: str
    split: str
    original_token_count: int
    original_text_sha256: str


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    hypothesis_id: str
    hypothesis_text: str
