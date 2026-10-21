"""Offline aggregate explorer; contract text is loaded only from local user-prepared files."""

from pathlib import Path
import sys
import html
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import read_json, read_jsonl

st.set_page_config(page_title="Contract Decision Fidelity Explorer", layout="wide")
st.markdown(
    """<style>.block-container {max-width:1400px;padding-top:2rem;}
mark {background:#ffeb9b;color:#162032;} .contract {white-space:pre-wrap;
font-family:Georgia,serif;line-height:1.6;max-height:480px;overflow:auto;
border:1px solid #ddd;padding:1rem;border-radius:8px;}</style>""",
    unsafe_allow_html=True,
)
st.title("Contract Decision Fidelity Explorer")
st.caption(
    "ContractNLI · Original / Naive / Contextualized / LongLLMLingua · NLP research, not legal advice"
)
mode = st.sidebar.radio("View", ["Overview", "Contract Explorer", "Method & reproducibility"])
names = {
    "original": "Original",
    "naive": "Naive",
    "contextual": "Contextualized",
    "long_30": "LongLLMLingua",
}
r = read_json(ROOT / "results/published/sealed.json")
if mode == "Overview":
    st.subheader("Historical sealed evaluation")
    cols = st.columns(4)
    for col, label, value in zip(
        cols,
        ["Contracts", "Hypotheses / contract", "Total decisions", "Valid outputs"],
        ["100", "17", "6,800", "100%"],
    ):
        col.metric(label, value)
    frame = pd.DataFrame(
        [
            {
                "Method": names[m],
                "Accuracy (%)": 100 * v["strict_accuracy"],
                "Macro-F1 (%)": 100 * v["macro_f1"],
                "Decision Flip (%)": 100 * v["decision_flip_rate"],
                "Compression ratio (%)": 100 * v["compression_ratio"]["mean"],
                "Item processing (min)": v["total_latency_seconds"] / 60,
                "Failure rate (%)": 100 * v["output_failure_rate"],
            }
            for m, v in r["methods"].items()
        ]
    )
    st.dataframe(frame, hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        st.bar_chart(
            frame.set_index("Method")[["Accuracy (%)", "Macro-F1 (%)"]],
            color=["#2864a5", "#55a99c"],
            stack=False,
            sort=False,
        )
    with right:
        st.bar_chart(frame.set_index("Method")[["Decision Flip (%)"]], color="#dc8745", sort=False)
    st.warning(
        "Unequal information budgets: LongLLMLingua uses each hypothesis and retains 31.46% of tokens; generic summaries retain 11.42–12.66%. Accuracy remains 6.12 percentage points below Original, with substantially more processing time."
    )
    st.markdown(
        "**LongLLMLingua − Naive:** accuracy +9.12 pp (paired 95% CI +6.12 to +12.29); decision flip −17.24 pp (−20.82 to −13.65). Bootstrap unit: contract; 5,000 draws."
    )
    st.caption(
        "Published aggregates are from the original completed v2 study. Opening this page performs no model calls and does not download data."
    )
    st.download_button(
        "Download comparison CSV",
        (ROOT / "results/published/sealed_comparison.csv").read_bytes(),
        "comparison.csv",
        "text/csv",
    )
elif mode == "Contract Explorer":
    st.subheader("Locally prepared contracts and reproduced predictions")
    path = ROOT / "data/sealed_inputs.jsonl"
    if not path.exists():
        st.info(
            "Contract text is not distributed. Run python -m cdfv2.cli download, assets, and audit first. This view then opens local data; predictions appear after reproduction."
        )
    else:
        docs = read_jsonl(path)
        selected_id = st.selectbox("Contract ID", [d["document_id"] for d in docs])
        d = next(d for d in docs if d["document_id"] == selected_id)
        hyps = read_json(ROOT / "data/hypotheses.json")
        hid = st.selectbox("Hypothesis", [h["hypothesis_id"] for h in hyps])
        h = next(h for h in hyps if h["hypothesis_id"] == hid)
        st.write(h["hypothesis_text"])
        refs = {
            (x["document_id"], x["hypothesis_id"]): x
            for x in read_jsonl(ROOT / "data/evaluation_references.jsonl")
        }
        ref = refs.get((d["document_id"], hid), {})
        gold = ref.get("gold_label")
        spans = sorted((x["start_char"], x["end_char"]) for x in ref.get("gold_evidence", []))
        merged = []
        for a, b in spans:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        pieces = []
        cursor = 0
        text = d["original_text"]
        for a, b in merged:
            pieces.extend(
                [html.escape(text[cursor:a]), "<mark>" + html.escape(text[a:b]) + "</mark>"]
            )
            cursor = b
        pieces.append(html.escape(text[cursor:]))
        st.markdown('<div class="contract">' + "".join(pieces) + "</div>", unsafe_allow_html=True)
        st.caption(
            "Highlighted spans are original ContractNLI evidence offsets; they are not provided to models."
        )
        rows = []
        preds = {}
        for method in names:
            preds[method] = {
                (x["document_id"], x["hypothesis_id"]): x
                for x in read_jsonl(ROOT / f"results/sealed/{method}_predictions.jsonl")
            }
        for hypothesis in hyps:
            key = (d["document_id"], hypothesis["hypothesis_id"])
            g = refs.get(key, {}).get("gold_label")
            row = {"Hypothesis": hypothesis["hypothesis_id"], "Gold": g}
            original = preds["original"].get(key, {}).get("predicted_label")
            for method in names:
                p = preds[method].get(key, {})
                row[names[method]] = p.get("predicted_label") or p.get(
                    "inference_status", "NOT_RUN"
                )
            row["Decision changed"] = (
                any(
                    preds[m].get(key, {}).get("predicted_label") not in [None, original]
                    for m in names
                )
                if original
                else None
            )
            rows.append(row)
        st.dataframe(
            pd.DataFrame(rows).style.apply(
                lambda row: [
                    "background-color: #fff1c2" if row["Decision changed"] else "" for _ in row
                ],
                axis=1,
            ),
            hide_index=True,
        )
        for method in ["naive", "contextual", "long_30"]:
            with st.expander(names[method], expanded=True):
                suffix = "contexts" if method == "long_30" else "summaries"
                records = read_jsonl(ROOT / f"results/sealed/{method}_{suffix}.jsonl")
                item = next(
                    (
                        x
                        for x in records
                        if x["document_id"] == d["document_id"]
                        and (method != "long_30" or x["hypothesis_id"] == hid)
                    ),
                    None,
                )
                if item:
                    st.text(item.get("summary", item.get("compressed_context", "")))
                    st.write({k: item.get(k) for k in ["status", "compression_ratio", "latency"]})
                    p = preds[method].get((d["document_id"], hid), {})
                    st.write(
                        {
                            "prediction": p.get("predicted_label"),
                            "gold": gold,
                            "correct": p.get("predicted_label") == gold
                            if p.get("predicted_label")
                            else None,
                            "status": p.get("inference_status", "NOT_RUN"),
                        }
                    )
                else:
                    st.info("No local reproduction output for this condition yet.")
else:
    st.markdown("""### Frozen design
    - Qwen2.5-7B-Instruct summaries, Qwen3-8B judgments; temperature 0.
    - One hypothesis per decision call for every method.
    - Official LongLLMLingua 0.2.2, Llama 2 7B Chat, 4-bit NF4, requested budget 30%.
    - Gold labels and evidence are evaluator-only. Failures are never relabeled NotMentioned.
    - Sealed set: 100 original Train documents; prior aggregate exposure limits independence.
    - Quote substring validity is a formal check, not semantic factuality.
    """)
    st.json(
        read_json(ROOT / "results/manifests/historical_provenance.json")["original_lock_sha256"]
    )
