"""Strict failure handling and document-level paired bootstrap."""

import numpy as np
from src.common import LABELS


def ratio(a, b):
    return float(a / b) if b else None


def confusion(gold, pred):
    matrix = np.zeros((3, 4), dtype=int)
    for g, p in zip(gold, pred):
        matrix[g, p if p >= 0 else 3] += 1
    return matrix


def from_confusion(matrix):
    tp = np.diag(matrix[:, :3]).astype(float)
    fp = matrix[:, :3].sum(0) - tp
    fn = matrix.sum(1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros(3), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(3), where=(tp + fn) > 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(3), where=(2 * tp + fp + fn) > 0)
    total = matrix.sum()
    valid = matrix[:, :3].sum()
    return {
        "strict_accuracy": ratio(tp.sum(), total),
        "valid_accuracy": ratio(tp.sum(), valid),
        "macro_f1": float(f1.mean()),
        "coverage": ratio(valid, total),
        "n": int(total),
        "valid_n": int(valid),
        "failure_n": int(total - valid),
        "class_metrics": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(matrix[i].sum()),
            }
            for i, name in enumerate(LABELS)
        },
        "confusion_matrix": matrix.tolist(),
        "confusion_columns": LABELS + ["FAILED"],
    }


def pair_counts(gold, original, summary):
    valid = (original >= 0) & (summary >= 0)
    flipped = (original != summary) & valid
    oc = original == gold
    sc = summary == gold
    return np.array(
        [
            int(valid.sum()),
            int(flipped.sum()),
            int((oc & sc).sum()),
            int((oc & ~sc).sum()),
            int((~oc & sc).sum()),
            int((~oc & ~sc).sum()),
            len(gold),
        ],
        dtype=int,
    )


def from_pair_counts(a):
    valid, flip, cp, ei, ec, ep, total = map(int, a)
    return {
        "valid_pair_n": valid,
        "flip_n": flip,
        "decision_flip_rate": ratio(flip, valid),
        "pair_failure_n": total - valid,
        "failure_or_flip_rate_all": ratio(total - valid + flip, total),
        "correct_preserved": cp,
        "error_introduced": ei,
        "error_corrected": ec,
        "error_persisted": ep,
        "error_introduced_rate_all": ratio(ei, total),
        "error_introduced_given_original_correct": ratio(ei, cp + ei),
        "error_corrected_rate_all": ratio(ec, total),
        "strict_accuracy_change": ratio(ec - ei, total),
        "category_definition": "All cases, failures are incorrect; flip uses valid pairs only",
    }


def evaluate_arrays(gold, predictions, bootstrap=5000, seed=20261015):
    # Axis 0 is contract ID; all 17 hypotheses remain together in each resample.
    gold = np.asarray(gold, int)
    predictions = {m: np.asarray(p, int) for m, p in predictions.items()}
    if gold.ndim != 2 or any(p.shape != gold.shape for p in predictions.values()):
        raise ValueError("Document/hypothesis shape mismatch")
    cms = {m: np.stack([confusion(g, p) for g, p in zip(gold, a)]) for m, a in predictions.items()}
    pcs = {
        m: np.stack([pair_counts(g, o, p) for g, o, p in zip(gold, predictions["original"], a)])
        for m, a in predictions.items()
        if m != "original"
    }
    metrics = {m: from_confusion(c.sum(0)) for m, c in cms.items()}
    for m, p in pcs.items():
        metrics[m].update(from_pair_counts(p.sum(0)))
    metrics["original"].update(
        decision_flip_rate=0.0, error_introduced_rate_all=0.0, error_corrected_rate_all=0.0
    )
    keys = ["strict_accuracy", "valid_accuracy", "macro_f1", "decision_flip_rate"]
    draws = {m: {k: [] for k in keys} for m in predictions}
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap):
        ix = rng.integers(0, len(gold), len(gold))
        for m in predictions:
            r = from_confusion(cms[m][ix].sum(0))
            r["decision_flip_rate"] = (
                from_pair_counts(pcs[m][ix].sum(0))["decision_flip_rate"] if m in pcs else 0.0
            )
            for k in keys:
                draws[m][k].append(r[k])

    def interval(vals):
        valid = [v for v in vals if v is not None]
        return {
            "interval": np.percentile(valid, [2.5, 97.5]).tolist() if valid else [None, None],
            "defined_replicates": len(valid),
        }

    for m in predictions:
        metrics[m]["ci95"] = {k: interval(v) for k, v in draws[m].items()}
    contrasts = []
    for a, b in [("contextual", "naive"), ("naive", "original"), ("contextual", "original")]:
        if a not in predictions or b not in predictions:
            continue
        for k in keys:
            d = interval(
                [x - y for x, y in zip(draws[a][k], draws[b][k]) if x is not None and y is not None]
            )
            v = metrics[a][k]
            w = metrics[b][k]
            contrasts.append(
                {
                    "contrast": f"{a} minus {b}",
                    "metric": k,
                    "difference": v - w if v is not None and w is not None else None,
                    **d,
                }
            )
    return {
        "methods": metrics,
        "paired_ci95": contrasts,
        "bootstrap": {"unit": "document_id", "replicates": bootstrap, "seed": seed},
    }
