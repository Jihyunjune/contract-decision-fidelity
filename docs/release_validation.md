# Rechecking this distribution

Run these commands from a source checkout. Viewing historical metrics and tests does not require Ollama. The integration data test is skipped until official data is prepared.

```powershell
python -m cdfv2.cli verify
python scripts/verify_public_files.py
python -m pytest -q
python -m pip check
python -m pip install -r requirements-dev.txt
python -m ruff format --check src cdfv2 scripts app tests
python -m ruff check src cdfv2 scripts app tests
python -m mypy
```

Mypy checks the strict input/output schema modules. It is not configured as a repository-wide strict type proof. Ruff checks critical syntax/name errors and formatting; algorithmic equivalence is covered separately by tests and the documented historical evaluation replay.

After `git add .`, inspect exactly the intended staged distribution:

```powershell
python scripts/check_release.py
python scripts/verify_clean_export.py
```

The second command builds a temporary export from Git index bytes, verifies that its imports resolve inside that export, and executes tests without downloaded data or models. It uses installed third-party dependencies, not the source v1/v2 projects. The temporary export is deleted after the check; audit logs are ignored.

Install Gitleaks from its official release and verify its release checksum before use. Scan a clean Git export with the included `.gitleaks.toml`, keeping redaction enabled. During preparation, version 8.30.1 was used. The narrow checksum exceptions and the initial findings are explained in `reports/public_release_audit.md`.

Changing released files invalidates `public_files.json`. Create a new reviewed release inventory rather than pretending modified files have the original hash. Local inference outputs live outside the public inventory. Reproduction locks additionally cover model assets, tokenizers, data projections and the official LLMLingua library; a changed lock or protected file stops sealed execution.

The public reproduction-lock proof was recreated after formatting four previously ignored shared modules; the earlier local verification lock was archived under ignored audit files. No research result, model setting, prompt or metric definition changed. Neither verification lock replaces the original historical experiment lock.
