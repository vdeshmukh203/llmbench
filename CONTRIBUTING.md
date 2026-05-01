# Contributing to llmbench

Thank you for considering a contribution!  All contributions are welcome:
bug reports, feature requests, documentation improvements, and code changes.

## Reporting issues

Open an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce (for bugs) or a use-case description (for features)
- Python version and OS
- Relevant output or error messages

## Development setup

```bash
git clone https://github.com/vdeshmukh203/llmbench.git
cd llmbench
pip install -e ".[yaml]"   # installs llmbench + optional PyYAML
pip install pytest
```

## Running tests

```bash
pytest tests/
```

All tests must pass before submitting a pull request.

## Code style

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Keep the **core** (`llmbench.py`) free of third-party dependencies.  Optional
  dependencies (PyYAML, etc.) must be guarded by a `try/except ImportError`.
- Write docstrings for all public classes, methods, and functions.
- Add or update tests in `tests/test_llmbench.py` for every change in
  behaviour.

## Submitting a pull request

1. Fork the repository and create a feature branch from `main`.
2. Make your changes with clear commit messages.
3. Ensure `pytest tests/` passes.
4. Open a pull request against `main` with a description of *what* changed
   and *why*.

## Adding a new metric

1. Implement the metric as a standalone function in `llmbench.py` with a
   docstring and return type `float`.
2. Add the metric field to `BenchmarkResult`.
3. Score it in `BenchmarkRunner._score`.
4. Include it in `BenchmarkRunner.summarize`.
5. Add tests covering at least: identical inputs (→ 1.0), empty inputs (→ 0.0),
   and a partial-overlap case.

## License

By contributing you agree that your contributions will be licensed under the
[MIT License](LICENSE).
