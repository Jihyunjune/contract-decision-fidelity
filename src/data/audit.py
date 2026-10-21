"""Official JSON audit; reference export is never imported by model modules."""

from collections import Counter, defaultdict
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.common import (
    ROOT,
    LABELS,
    read_json,
    write_json,
    write_jsonl,
    sha,
    text_hash,
    words,
    table,
)
from src.models.tokenization import count


def describe(xs):
    a = np.asarray(xs)
    return {
        "n": len(xs),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "max": int(a.max()),
        "min": int(a.min()),
    }


def validate_document(d, ids):
    if not isinstance(d["text"], str) or not d["text"].strip():
        raise ValueError("Missing text")
    if len(d["annotation_sets"]) != 1:
        raise ValueError("Unexpected annotation count")
    annotations = d["annotation_sets"][0]["annotations"]
    if set(annotations) != set(ids):
        raise ValueError("Hypothesis mismatch")
    for start, end in d["spans"]:
        if not (
            isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(d["text"])
        ):
            raise ValueError("Invalid character range")
        # Official segmentation includes whitespace-only, usually unreferenced spans.
        # Preserve them and their indices; report separately instead of reindexing.
    for a in annotations.values():
        if a["choice"] not in LABELS:
            raise ValueError("Invalid label")
        if len(a["spans"]) != len(set(a["spans"])):
            raise ValueError("Duplicate span IDs")
        if any(type(i) != int or not 0 <= i < len(d["spans"]) for i in a["spans"]):
            raise ValueError("Invalid evidence index")
        if a["choice"] == "NotMentioned" and a["spans"]:
            raise ValueError("NotMentioned evidence unexpectedly present")
    return annotations


def audit():
    expected = {"train": 423, "dev": 61, "test": 123}
    all_ids = set()
    hashes = defaultdict(list)
    canonical = None
    stats = {}
    refs = []
    inputs = {}
    raw_hashes = {}
    all_lengths = []
    for split, n in expected.items():
        path = ROOT / f"data/raw/{split}.json"
        raw_hashes[split] = sha(path)
        obj = read_json(path)
        if len(obj["documents"]) != n or len(obj["labels"]) != 17:
            raise ValueError("Official counts mismatch")
        if canonical is None:
            canonical = obj["labels"]
        if obj["labels"] != canonical:
            raise ValueError("Hypothesis definitions differ")
        labels = Counter()
        by_hyp = defaultdict(Counter)
        lengths = []
        counts = []
        inputs[split] = []
        span_n = 0
        no_evidence = 0
        whitespace = []
        for d in obj["documents"]:
            if d["id"] in all_ids:
                raise ValueError("Duplicate ID across official splits")
            all_ids.add(d["id"])
            ann = validate_document(d, canonical)
            tok = count(d["text"])
            lengths.append(tok)
            used = {i for a in ann.values() for i in a["spans"]}
            whitespace += [
                {"document_id": d["id"], "span_id": i, "referenced": i in used}
                for i, (a, b) in enumerate(d["spans"])
                if not d["text"][a:b].strip()
            ]
            th = text_hash(d["text"])
            hashes[th].append({"split": split, "id": d["id"]})
            span_n += len(d["spans"])
            inputs[split].append(
                {
                    "document_id": d["id"],
                    "original_text": d["text"],
                    "split": split,
                    "original_token_count": tok,
                    "original_text_sha256": th,
                }
            )
            for hid, a in ann.items():
                labels[a["choice"]] += 1
                by_hyp[hid][a["choice"]] += 1
                counts.append(len(a["spans"]))
                evidence = [
                    {
                        "span_id": i,
                        "start_char": d["spans"][i][0],
                        "end_char": d["spans"][i][1],
                        "text": d["text"][d["spans"][i][0] : d["spans"][i][1]],
                    }
                    for i in a["spans"]
                ]
                no_evidence += a["choice"] != "NotMentioned" and not evidence
                refs.append(
                    {
                        "document_id": d["id"],
                        "hypothesis_id": hid,
                        "hypothesis_text": canonical[hid]["hypothesis"],
                        "gold_label": a["choice"],
                        "gold_evidence_span_ids": a["spans"],
                        "gold_evidence": evidence,
                        "split": split,
                    }
                )
        stats[split] = {
            "documents": n,
            "judgments": n * 17,
            "length_tokens": describe(lengths),
            "labels": dict(labels),
            "by_hypothesis": dict(by_hyp),
            "evidence_count": describe(counts),
            "validated_document_spans": span_n,
            "positive_labels_without_evidence": no_evidence,
            "whitespace_spans": whitespace,
            "whitespace_span_count": len(whitespace),
            "whitespace_referenced_count": sum(w["referenced"] for w in whitespace),
        }
        all_lengths += lengths
    if len(all_ids) != 607:
        raise ValueError("Total mismatch")
    duplicates = [v for v in hashes.values() if len(v) > 1]
    hypotheses = [
        {"hypothesis_id": k, "hypothesis_text": v["hypothesis"]} for k, v in canonical.items()
    ]
    write_json(ROOT / "data/processed/hypotheses.json", hypotheses)
    for split, rows in inputs.items():
        write_jsonl(ROOT / f"data/processed/{split}_inputs.jsonl", rows)
    write_jsonl(ROOT / "data/processed/evaluation_references.jsonl", refs)
    result = {
        "passed": True,
        "total_documents": 607,
        "total_judgments": 10319,
        "raw_sha256": raw_hashes,
        "splits": stats,
        "overall_length_tokens": describe(all_lengths),
        "exact_text_duplicate_groups": duplicates,
        "tokenizer": "Pinned Qwen3-8B tokenizer, no special tokens",
        "official_splits_retained": True,
        "evidence_validation": "Index and original substring/offset validity; not semantic evidence reannotation",
    }
    write_json(ROOT / "results/metrics/data_statistics.json", result)
    files = [
        ROOT / "data/processed/hypotheses.json",
        ROOT / "data/processed/evaluation_references.jsonl",
        *[ROOT / f"data/processed/{s}_inputs.jsonl" for s in expected],
    ]
    write_json(
        ROOT / "results/manifests/data_audit.json",
        {
            "raw_sha256": raw_hashes,
            "processed_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        },
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    for split, rows in inputs.items():
        ax.hist([r["original_token_count"] for r in rows], bins=25, alpha=0.45, label=split)
    ax.set(
        xlabel="Original tokens (Qwen3 tokenizer)",
        ylabel="Documents",
        title="Official ContractNLI document lengths",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "reports/figures/document_length_distribution.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(3)
    for i, split in enumerate(expected):
        ax.bar(
            x + (i - 1) * 0.25,
            [stats[split]["labels"].get(k, 0) for k in LABELS],
            0.25,
            label=split,
        )
    ax.set_xticks(x, LABELS)
    ax.set(ylabel="Document–hypothesis pairs", title="Official label distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "reports/figures/gold_label_distribution.png", dpi=170)
    plt.close(fig)
    report = "# ContractNLI 데이터 감사\n\n공식 JSON 원문·문자 위치를 변경하지 않고 607개 문서, 17개 가설, 10,319개 판단 참조를 확인했다.\n\n"
    report += table(
        ["Split", "Documents", "Pairs", "Mean tokens", "Median", "P90", "Max"],
        [
            [
                s,
                v["documents"],
                v["judgments"],
                round(v["length_tokens"]["mean"], 1),
                v["length_tokens"]["median"],
                v["length_tokens"]["p90"],
                v["length_tokens"]["max"],
            ]
            for s, v in stats.items()
        ],
    )
    report += "\n\n" + table(
        ["Split", *LABELS],
        [[s, *[v["labels"].get(k, 0) for k in LABELS]] for s, v in stats.items()],
    )
    report += (
        "\n\n"
        + table(
            ["Split", "Official whitespace-only spans", "Referenced whitespace spans"],
            [
                [s, v["whitespace_span_count"], v["whitespace_referenced_count"]]
                for s, v in stats.items()
            ],
        )
        + "\n\n공백만 있는 공식 segmentation span도 원본 인덱스 그대로 보존했다. 원문 또는 annotation을 보정하지 않았다."
    )
    report += f"\n\nID 중복/분할 누수: 0. 동일 원문 해시 중복 그룹: {len(duplicates)}. 공식 분할에서 문서를 삭제하거나 이동하지 않았다. 가설별 불균형과 span 수 분포는 data_statistics.json에 저장했다. Gold Evidence는 Entailment/Contradiction에만 적용한다. 미언급은 모순이 아니다. 위치 검증은 의미적 재검증이 아니다.\n\n모델용 입력과 평가 참조를 서로 다른 파일에 저장했다. 공식 원본 SHA-256은 data_audit.json에 기록했다.\n\n출처: https://stanfordnlp.github.io/contract-nli/ (CC BY 4.0).\n"
    (ROOT / "reports/data_audit.md").write_text(report, encoding="utf-8")
    return result
