"""Real one-document, one-hypothesis interface check; not a new benchmark."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdfv2.common import read_json, write_json
from cdfv2.inputs import documents, hypotheses
from cdfv2.runtime import Ollama
from cdfv2.summaries import summarize
from cdfv2.decision import decide


def main():
    out = ROOT / "results/release_smoke/output.json"
    if out.exists():
        raise RuntimeError("Smoke output already exists; preserve first run")
    d = documents("smoke")[0]
    h = hypotheses()[0]
    sums = {}
    results = {}
    b = Ollama("summary")
    try:
        for m in ["naive", "contextual"]:
            sums[m] = summarize(d, m, b)
            write_json(ROOT / f"results/release_smoke/{m}.json", sums[m])
    finally:
        b.unload()
    from cdfv2.compression import Compressor

    c = Compressor()
    try:
        context = c.compress(d, h, 0.30)
    finally:
        c.close()
    write_json(ROOT / "results/release_smoke/context.json", context)
    b = Ollama("decision")
    try:
        for m, text, status in [
            ("original", d.original_text, "SUCCESS"),
            *[
                (
                    m,
                    s["summary"],
                    "SUCCESS" if s["status"] == "SUCCESS" else "UPSTREAM_SUMMARY_FAILED",
                )
                for m, s in sums.items()
            ],
            ("long_30", context["compressed_context"], context["status"]),
        ]:
            results[m] = decide(d, text, h, b, m, status)
            write_json(ROOT / f"results/release_smoke/{m}_prediction.json", results[m])
    finally:
        b.unload()
    record = {
        "purpose": "Release interface validation, not final study regeneration",
        "document_id": d.document_id,
        "hypothesis_id": h.hypothesis_id,
        "summary_statuses": {m: s["status"] for m, s in sums.items()},
        "compression_status": context["status"],
        "judgment_statuses": {m: r["inference_status"] for m, r in results.items()},
        "judgments": 4,
    }
    write_json(out, record)
    print(record)


if __name__ == "__main__":
    main()
