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

Run these before opening a pull request:

```bash
ruff check src tests
ruff format --check src tests
pytest
```

Do not commit datasets, fitted vectorizers, neural model files, logs, or generated evaluation
outputs. The `.gitignore` preserves the expected local directories with `.gitkeep` files.

## Pull requests

- Explain the motivation and observable behavior change.
- Add or update tests for code changes.
- Update the README or model card when the interface, methodology, or reported metrics change.
- Never replace historical benchmark values with a new run unless the environment and protocol are
  documented and the machine-readable results are retained.

