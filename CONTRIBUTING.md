# Contributing

Thank you for improving the project. Small, focused pull requests are easiest to review.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[all,dev]"
```

## Quality checks

Enable the repository's formatting and lint check before commits:

```bash
make install-hooks
```

The hook checks a temporary copy of the staged files without changing your working tree. Unstaged
formatting fixes cannot hide a broken commit. Installation refuses to replace an existing custom
hook configuration. Fix formatting with `make format`, review and stage the changes, then commit again.

Run the complete CI checks before pushing:

```bash
make check
```

The Makefile uses `.venv/bin/python` when available; override it with `PYTHON=python3.12` if needed.
CI uses the same `make lint` and `make test` targets. Ruff is pinned in `pyproject.toml` so local
formatting agrees with CI. Update its dependency pin and `required-version` together.
Tests require the ML dependencies; a minimal CI-equivalent installation is `pip install -e ".[dev,ml]"`.
CI runs tests on Python 3.11 and 3.12 independently, so both failures remain visible.

Do not commit datasets, fitted vectorizers, neural model files, logs, or generated evaluation
outputs. The `.gitignore` preserves the expected local directories with `.gitkeep` files.

## Pull requests

- Explain the motivation and observable behavior change.
- Add or update tests for code changes.
- Update the README or model card when the interface, methodology, or reported metrics change.
- Never replace historical benchmark values with a new run unless the environment and protocol are
  documented and the machine-readable results are retained.
