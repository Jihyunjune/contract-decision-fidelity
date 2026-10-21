from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, re

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Entailment", "Contradiction", "NotMentioned"]
METHODS = ["original", "naive", "contextual"]


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temp, path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config():
    cfg = read_json(ROOT / "configs/experiment.json")
    cfg["endpoint"] = os.environ.get("CDF_OLLAMA_URL", cfg["endpoint"])
    return cfg


def words(text):
    return re.findall(r"\b[\w]+(?:['’\-][\w]+)*\b", text)


def immutable(path, obj):
    path = Path(path)
    if path.exists():
        if read_json(path) != obj:
            raise RuntimeError(f"Immutable manifest differs: {path}")
    else:
        write_json(path, obj)


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        + [
            "| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in r) + " |"
            for r in rows
        ]
    )
