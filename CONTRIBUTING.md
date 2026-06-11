# Contributing to ASTRID

Thank you for helping improve ASTRID.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Run the checks before opening a pull request:

```bash
python -m ruff check --select E9,F63,F7,F82 .
python -m pytest
python -m build
```

Launch the application with:

```bash
streamlit run app.py
```

## Pull requests

- Keep changes focused and explain the user-visible behavior.
- Add or update tests for analysis logic, ingestion behavior, and report output.
- Do not commit datasets, audit output, credentials, or generated cache files.
- Treat changes to scoring, thresholds, and policy gates as behavioral changes and document them.

## Reporting bugs

Include the analyzer, input format, Python version, expected behavior, and a minimal reproducible example where possible. Remove confidential or personally identifiable data before attaching samples.
