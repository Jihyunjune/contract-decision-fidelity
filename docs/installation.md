# Installation and operational notes

Use a source checkout with an editable installation (`python -m pip install -e .`). Code resolves configuration and data from the checkout with pathlib, not the working directory. Runtime overrides are restricted to a loopback Ollama URL via `CDF_OLLAMA_URL`; remote paid inference is unsupported.

For the offline dashboard, tests and metric verification, install `requirements-viewer.txt`. Full compression needs `requirements.txt` and a compatible CUDA-enabled PyTorch 2.8.0 build. Original validation used CUDA 12.8 wheels, RTX 5080 and Python 3.10. `docs/original_environment.lock.txt` records the historical environment, while `docs/public_environment.lock.txt` records release validation. Neither is a model-weight manifest.

Ollama must expose the pinned model digests, not merely matching tag names. Use `runtime-check` before inference. Default port is 11436 to avoid an unrelated local server. Run one model stage at a time. The software unloads each Ollama model before compression and closes the compressor before judging. No server is started automatically by the library.

The official downloader validates the archive before extracting an explicit allowlist of file names. It does not extract PDFs or follow archive paths. Downloaded LICENSE/TERMS accompany the local dataset. `audit` restores original JSON evidence indices and checks all official counts, split IDs, labels, span ranges and the historical partition hashes. Tokenizers are pinned and downloaded by `assets` first.

Output directories are ignored. Do not force-add `data/`, `models/`, `.env`, logs, summary/context files or raw predictions. New execution uses `results/sealed/`; published historical aggregates remain in `results/published/` and are never overwritten by evaluation.

If model revisions become inaccessible, report the exact asset failure; do not silently substitute another model and call it reproduction. Full regeneration is stochastic at the hardware/backend level despite fixed seeds. A hash-verification pass proves file integrity, not scientific correctness.
