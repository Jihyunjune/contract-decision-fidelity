import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from cdfv2.common import ROOT, read_json
from src.data.inputs import InputDocument, Hypothesis
from src.decision.pipeline import parse, quote_check, messages as judgment_messages
from src.evaluation.metrics import (
    confusion,
    from_confusion,
    pair_counts,
    from_pair_counts,
    evaluate_arrays,
)
from cdfv2.summaries import parse_summary, messages, summarize
from cdfv2.decision import decide

H = Hypothesis(
    hypothesis_id="synthetic-h", hypothesis_text="The recipient must not disclose information."
)


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        '{"bullets":[]}',
        '{"bullets":[""]}',
        '{"bullets":[1]}',
        '{"bullets":["ok"],"label":"Entailment"}',
    ],
)
def test_invalid_summary(raw):
    with pytest.raises(ValueError):
        parse_summary(raw)


@pytest.mark.parametrize("fence", [False, True])
def test_format_normalization(fence):
    raw = json.dumps(
        {"bullets": ["* No copying except for the Purpose.", "2. Duties survive termination."]}
    )
    if fence:
        raw = "```json\n" + raw + "\n```"
    assert (
        parse_summary(raw) == "- No copying except for the Purpose.\n- Duties survive termination."
    )
    text = " ".join(["word"] * 300)
    assert len(parse_summary(json.dumps({"bullets": [text]})).split()) == 301


def test_input_boundary():
    with pytest.raises(ValueError):
        Hypothesis(hypothesis_id="h", hypothesis_text="text", gold_label="Entailment")
    assert H.hypothesis_text not in str(messages("SYNTHETIC CONTRACT", "contextual"))
    assert json.loads(judgment_messages("SYNTHETIC CONTRACT", [H])[-1]["content"])[
        "hypotheses"
    ] == [H.model_dump()]
    for name in ["compression", "summaries", "decision", "runner", "runtime", "inputs"]:
        code = (ROOT / f"cdfv2/{name}.py").read_text(encoding="utf-8")
        assert "evaluation_references" not in code and "gold_label" not in code


@pytest.mark.parametrize(
    "predictions",
    [
        [{"hypothesis_id": "h", "predicted_label": "unknown", "evidence_quote": ""}],
        [{"hypothesis_id": "wrong", "predicted_label": "NotMentioned", "evidence_quote": ""}],
        [],
        [{"hypothesis_id": "h", "predicted_label": "NotMentioned", "evidence_quote": ""}] * 2,
    ],
)
def test_invalid_decision(predictions):
    with pytest.raises(ValueError):
        parse(json.dumps({"predictions": predictions}), ["h"])


def test_quote_check():
    assert quote_check("abc", "xxabc", "Entailment")["start_char"] == 2
    invalid = quote_check("Invented", "Actual", "Entailment")
    assert not invalid["evidence_valid"] and "predicted_label" not in invalid


def test_metrics_hand_computed():
    g = np.array([0, 1, 2, 0])
    o = np.array([0, 1, 2, 2])
    p = np.array([0, -1, 0, 0])
    r = from_confusion(confusion(g, p))
    assert r["strict_accuracy"] == 0.5
    assert r["valid_accuracy"] == pytest.approx(2 / 3) and r["macro_f1"] == pytest.approx(
        (4 / 5) / 3
    )
    assert r["coverage"] == 0.75 and r["class_metrics"]["Contradiction"]["recall"] == 0
    q = from_pair_counts(pair_counts(g, o, p))
    assert q["error_introduced"] == 2 and q["error_corrected"] == 1
    assert q["decision_flip_rate"] == pytest.approx(2 / 3)


def test_paired_bootstrap():
    g = np.array([[0, 1], [1, 2], [2, 0]])
    p = {"original": g, "naive": np.array([[0, 2], [1, 0], [2, 0]]), "contextual": g}
    r = evaluate_arrays(g, p, 100, 7)
    assert r == evaluate_arrays(g, p, 100, 7) and r["bootstrap"]["unit"] == "document_id"
    with pytest.raises(ValueError):
        evaluate_arrays(g, {"original": np.array([0, 1])}, 10, 7)


def test_retry_no_truncation(monkeypatch):
    import cdfv2.summaries as module

    monkeypatch.setattr(module, "count", lambda s: len(s.split()))

    class Doc:
        document_id = 1
        original_text = " ".join(["source"] * 600)

    class Backend:
        model = "mock"
        identity = {"digest": "mock"}

        def generate(self, *args):
            return {
                "raw_output": json.dumps({"bullets": [" ".join(["word"] * 300)]}),
                "done_reason": "stop",
                "latency": 0.1,
            }

    r = summarize(Doc(), "naive", Backend())
    assert r["status"] == "SUMMARY_FAILED" and len(r["attempts"]) == 4 and r["summary_words"] == 300


def test_failure_not_imputed():
    class Doc:
        document_id = 1

    class Backend:
        model = "mock"
        identity = {"digest": "mock"}

        def generate(self, *args):
            return {"raw_output": "not json", "thinking_output": "", "done_reason": "stop"}

    r = decide(Doc(), "text", H, Backend(), "naive", "UPSTREAM_SUMMARY_FAILED")
    assert r["predicted_label"] is None and not r["attempts"]
    r = decide(Doc(), "text", H, Backend(), "original")
    assert r["predicted_label"] is None and len(r["attempts"]) == 2


def test_context_guard():
    from src.models.ollama import context_guard, ContextOverflow

    with pytest.raises(ContextOverflow):
        context_guard(32768, 768, 32768)


def test_chunks_no_content_loss():
    from cdfv2.compression import chunks_with_offsets

    class T:
        def encode(self, text, **kwargs):
            return list(text)

        def __call__(self, text, **kwargs):
            return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

    text = "  a long line\nSecond line.\n"
    spans = chunks_with_offsets(text, T(), 7)
    assert "".join(text[a:b] for a, b in spans) == text


def test_configuration_and_partition():
    from cdfv2.common import config

    c = config()
    assert c["batch_size"] == 1 and c["temperature"] == 0 and c["max_words"] == 250
    s = read_json(ROOT / "results/manifests/dataset_seal.json")
    assert len(s["sealed_ids"]) == 100 and len(s["development_ids"]) == 323
    assert not set(s["sealed_ids"]) & set(s["development_ids"])
    assert not set(s["sealed_ids"]) & set(s["excluded_from_sealed_ids"])


def test_reproduction_lock_detects_change(tmp_path, monkeypatch):
    import cdfv2.locking as locking
    from src.common import write_json, sha

    monkeypatch.setattr(locking, "ROOT", tmp_path)
    monkeypatch.setattr(locking, "verify_assets", lambda: None)
    monkeypatch.setattr(locking, "config", lambda: {"endpoint": "http://localhost:11436"})
    protected = tmp_path / "input.txt"
    protected.write_text("unchanged")
    lock = tmp_path / "results/manifests/reproduction_lock.json"
    write_json(
        lock,
        {
            "files_sha256": {"input.txt": sha(protected)},
            "packages": {},
            "endpoint": "http://localhost:11436",
            "methods": ["original"],
        },
    )
    write_json(lock.with_name("reproduction_lock_sha256.json"), {"sha256": sha(lock)})
    assert locking.verify_lock()["methods"] == ["original"]
    protected.write_text("changed")
    with pytest.raises(RuntimeError, match="Protected file changed"):
        locking.verify_lock()


def test_paths_independent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from cdfv2.cli import verify

    assert verify()["published_files_verified"] >= 2
    assert ROOT == Path(__file__).resolve().parents[1]
    for folder in ["cdfv2", "src"]:
        for p in (ROOT / folder).rglob("*.py"):
            code = p.read_text(encoding="utf-8")
            assert "contract_decision_fidelity_v2" not in code
            assert "from cdfv2 import V1" not in code


def test_published_csv_json():
    r = read_json(ROOT / "results/published/sealed.json")
    frame = pd.read_csv(ROOT / "results/published/sealed_comparison.csv")
    assert sum(v["n"] for v in r["methods"].values()) == 6800
    for row in frame.to_dict("records"):
        v = r["methods"][row.pop("method")]
        for k, x in row.items():
            assert x == pytest.approx(
                v["compression_ratio"]["mean"] if k == "compression_ratio" else v[k]
            )


def test_schema_evidence_offsets():
    from src.data.audit import validate_document

    d = {
        "text": "Synthetic clause.",
        "spans": [[0, 17]],
        "annotation_sets": [{"annotations": {"h": {"choice": "Entailment", "spans": [0]}}}],
    }
    validate_document(d, {"h"})
    d["spans"] = [[0, 100]]
    with pytest.raises(ValueError):
        validate_document(d, {"h"})


@pytest.mark.parametrize("page", ["Overview", "Contract Explorer", "Method & reproducibility"])
def test_offline_dashboard(page, monkeypatch):
    import requests
    from streamlit.testing.v1 import AppTest

    def blocked(*args, **kwargs):
        raise AssertionError("Offline dashboard attempted network")

    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception


@pytest.mark.integration
def test_downloaded_official_data():
    if not (ROOT / "data/raw/train.json").exists():
        pytest.skip("Run download/assets/audit for integration checks")
    from cdfv2.inputs import documents
    from cdfv2.common import sha

    seen = set()
    for split, n in [("train", 423), ("dev", 61), ("test", 123)]:
        data = read_json(ROOT / f"data/raw/{split}.json")
        assert len(data["documents"]) == n
        ids = {d["id"] for d in data["documents"]}
        assert not seen & ids
        seen |= ids
    seal = read_json(ROOT / "results/manifests/dataset_seal.json")
    for p, h in seal["files_sha256"].items():
        assert sha(ROOT / p) == h
    assert len(documents("sealed")) == 100
