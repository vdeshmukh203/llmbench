"""
llmbench: Lightweight reproducible LLM benchmarking framework.

This package stub re-exports the public API from the top-level
``llmbench`` module (``llmbench.py``), which is the installed
distribution target declared in ``pyproject.toml``.
"""

__version__ = "0.2.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

# Re-export from the installed top-level module.
# (Works when llmbench.py is on sys.path, e.g. during development.)
try:
    from llmbench import (  # noqa: F401
        BenchmarkRunner,
        BenchmarkResult,
        BenchmarkSpec,
        Task,
        SAMPLE_TASKS,
        exact_match,
        exact_match_normalised,
        rouge_l,
        bleu_1,
        f1_score,
        load_tasks_from_file,
        contains_code,
        main,
    )

    __all__ = [
        "BenchmarkRunner",
        "BenchmarkResult",
        "BenchmarkSpec",
        "Task",
        "SAMPLE_TASKS",
        "exact_match",
        "exact_match_normalised",
        "rouge_l",
        "bleu_1",
        "f1_score",
        "load_tasks_from_file",
        "contains_code",
        "main",
    ]
except ImportError:
    # llmbench.py not yet on path (e.g. bare src/ checkout without install)
    pass
