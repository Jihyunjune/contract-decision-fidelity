"""Public entry point. Importing or viewing results never starts inference."""

import argparse
import json
from cdfv2.common import ROOT, read_json, read_jsonl, sha


def verify():
    p = read_json(ROOT / "results/manifests/historical_provenance.json")
    for name, h in p["published_files_sha256"].items():
        if sha(ROOT / name) != h:
            raise RuntimeError(f"Published historical result changed: {name}")
    return {
        "published_files_verified": len(p["published_files_sha256"]),
        "historical_lock_sha256": p["original_lock_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["download", "audit", "status", "verify", "lock", "runtime-check"]:
        sub.add_parser(name)
    a = sub.add_parser("assets")
    a.add_argument(
        "--compressor",
        action="store_true",
        help="Download Llama 2 weights; review its license first",
    )
    r = sub.add_parser("run")
    r.add_argument("--split", choices=["smoke", "development", "sealed"], required=True)
    r.add_argument("--stage", choices=["summary", "compress", "judge"], required=True)
    r.add_argument(
        "--method", choices=["original", "naive", "contextual", "long_15", "long_30"], required=True
    )
    e = sub.add_parser("evaluate")
    e.add_argument("--split", choices=["smoke", "development", "sealed"], required=True)
    args = parser.parse_args()
    result = None
    if args.command == "verify":
        result = verify()
    elif args.command == "download":
        from cdfv2.preparation import download_data

        result = download_data()
    elif args.command == "audit":
        from cdfv2.preparation import audit_and_prepare

        result = audit_and_prepare()
    elif args.command == "assets":
        from cdfv2.preparation import assets

        assets(args.compressor)
        result = {"downloaded": True}
    elif args.command == "runtime-check":
        from cdfv2.runtime import Ollama

        result = {}
        for role in ["summary", "decision"]:
            b = Ollama(role)
            result[role] = {
                "model": b.model,
                "digest": b.identity["digest"],
                "version": b.runtime["version"],
            }
    elif args.command == "lock":
        from cdfv2.locking import create_lock

        result = create_lock()
    elif args.command == "run":
        from cdfv2.runner import run_summaries, run_compression, run_decisions

        if args.stage == "summary":
            if args.method not in ["naive", "contextual"]:
                parser.error("Summary stage requires naive or contextual")
            run_summaries(args.split, args.method)
        elif args.stage == "compress":
            if not args.method.startswith("long_"):
                parser.error("Compression stage requires long_15 or long_30")
            run_compression(args.split, int(args.method.split("_")[1]) / 100)
        else:
            run_decisions(args.split, args.method)
    elif args.command == "evaluate":
        from cdfv2.evaluate import evaluate

        methods = ["original", "naive", "contextual", "long_15", "long_30"]
        if args.split == "sealed":
            from cdfv2.locking import verify_lock

            methods = verify_lock()["methods"]
        evaluate(args.split, methods)
    else:
        result = {
            "historical_results": verify(),
            "local_runs": {
                s: {
                    p.name: len(read_jsonl(p))
                    for p in (ROOT / f"results/{s}").glob("*predictions.jsonl")
                }
                for s in ["smoke", "development", "sealed"]
            },
        }
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
