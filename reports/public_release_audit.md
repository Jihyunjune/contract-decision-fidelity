# Public release audit

Status: **READY FOR USER PUBLICATION APPROVAL**. No GitHub repository was created and nothing was pushed. Project-code MIT licensing was explicitly authorized by the rights holder during this release task. This report concerns release packaging, not a new scientific evaluation.

## Preservation and independence

- Original v1: **24,239 files**, original v2: **42,398 files**. Every file was SHA-256 hashed before and after, including virtual environments, model assets and logs. **0 changed, 0 added, 0 removed** in both source directories.
- Public source has no sibling-project Python imports. Necessary team-authored v1 helpers are included locally under `src/`; v2 modules are under `cdfv2/`.
- A fresh virtual environment was installed in the new folder; imports, CLI help, metadata and `pip check` passed. A second verification exported **only staged Git bytes**, without data/models, and ran imports, help, result verification and offline tests there.
- Legacy local wheel URLs were removed from the public environment record and replaced with package versions. No raw machine logs are included.

## Algorithm and result preservation

AST comparisons confirmed equality of 26 top-level function/class definitions across the numerical evaluator, NLI schema, decision, summary, compression, evaluation and runner modules. Formatting and package-root/configuration changes do not change the protected algorithms. The unused older v1 decision runner was excluded; its schema, prompt construction, parser and quote checks were retained.

The complete **6,800 historical predictions** were read from protected source outputs and copied only into an ignored local verification folder. Recalculation produced **exactly equal method metrics and paired 95% intervals** (5,000 document bootstrap draws). This involved **zero new model calls**. Historical aggregate JSON/CSV bytes are preserved and checked against recorded SHA-256 values. Release smoke is separate.

Original experiment lock: `f5624f7c67b97152b0fba2aa1887aa616b681cab7e8537faa2f5de7fadd72fe0`.

## Executed checks

| Check | Actual result |
|---|---|
| Editable package install and `pip check` | Passed in new virtual environment |
| CLI help / status / published result verification | Passed |
| Official archive download | SHA-256 matches frozen official archive |
| Data audit | 607 documents, official 423/61/123 split counts; labels/spans/IDs validated |
| Historical v2 projection | All four input/reference/hypothesis file hashes match original seal |
| Ollama connection | 0.40.2; both pinned Qwen digests match; separate local service/logs |
| Real release smoke | 2 summaries + 1 LongLLMLingua context + 4 judgments, all SUCCESS |
| Full historical evaluation replay | 6,800 existing outputs, metrics and bootstrap exactly equal |
| Pytest with prepared data | **28 passed**, 2 benign SWIG deprecation warnings |
| Clean staged export pytest | **27 passed, 1 skipped** (official data intentionally absent) |
| Ruff formatting and critical lint | Passed |
| Mypy | Passed on 2 schema modules; not a whole-project strict type proof |
| Streamlit | Offline AppTest views passed; actual browser rendered and screenshot inspected |
| Reproduction lock | Created and verified with local data, tokenizers, compressor, library and config hashes |
| Git | New main branch initialized; staged distribution; no remote, no commit, no push |

The real smoke used one previously selected development document/hypothesis. It is an interface check, not a new accuracy benchmark. Compressor weights were copied read-only into the new ignored local model directory and hash-verified; they are not in Git. A separately logged Ollama service used the installed local model cache. The distributed package depends on an HTTP Ollama endpoint, not on the source project's runtime path. Full sealed inference was not repeated.

## Security and distribution inspection

Exactly the staged Git files were exported and scanned with checksum-verified **Gitleaks 8.30.1**, plus a project scanner for private keys, token/password patterns, local absolute paths, forbidden assets and files above 5 MiB. Gitleaks initially flagged **10 verified tokenizer/source SHA-256 values** as generic keys. `.gitleaks.toml` permits only the named file-key/checksum patterns in the named manifests; it does not disable secret rules globally. Final scanning reports **zero findings**.

`.gitignore` probes passed for `.env`, environments, raw dataset, model weights, runtime binaries, wheels, raw predictions and local audit logs. Root-only data/model exclusions deliberately do not exclude `src/data` or `src/models`; clean-export tests verify those modules are shipped. No dataset text, summary text, compressed contract text, raw evidence quotes, model weights, local usernames, local environment paths, remotes or credentials are included. Aggregate graphics and the screenshot contain only public measurements.

`results/manifests/public_files.json` is the exact file inventory with SHA-256 values (excluding its own self-hash). `python scripts/verify_public_files.py` verifies it. The original source and private logs are not part of that manifest. Pattern scanners cannot prove absence of every possible secret or personal datum; their scope and tools are explicitly recorded.

## Licensing

- Project-authored code: **MIT**, explicitly authorized by the rights holder. Collective contributor attribution avoids inventing individual author identities.
- ContractNLI: **CC BY 4.0**, official terms checked. Default distribution policy is download-only; the download retains local LICENSE/TERMS.
- Qwen models: **Apache-2.0**. No weights or tokenizer files bundled.
- LLMLingua 0.2.2: **MIT**. Used as an installed dependency; no upstream source vendored.
- Llama 2: **Llama 2 Community License Agreement + Acceptable Use Policy**, checked in the pinned local asset. These conditions remain independent of the project's MIT license.
- Streamlit and major dependency versions/licenses are listed in `THIRD_PARTY_NOTICES.md` and `docs/dependency_licenses.json`.

## Publication boundary and remaining limits

The next external action requires **user approval to create the GitHub Public repository and push this reviewed distribution**. Git identity settings were not changed; no commit was made so an unreviewed personal commit email is not exposed. Repository owner/name and a public commit identity can be selected at publication time.

Validated OS/GPU: Windows, Python 3.10, RTX 5080. Linux and other GPUs are documented possibilities, not tested platform claims. Live CI will run only after GitHub publication; the included workflow was locally matched by clean-export tests but was not executed by GitHub Actions. Full model regeneration remains dependent on upstream model availability, accepted licenses and hardware/backend behavior. Published metrics show unequal information budgets, higher LongLLMLingua latency and a 6.12 pp accuracy gap to Original; release preparation does not alter those limitations.
