"""
llmbench: Lightweight reproducible LLM benchmarking framework.

Provides a runner that executes prompt suites against one or more LLM
providers, scores outputs using standard NLP metrics, and emits
reproducible benchmark reports in JSON and CSV formats.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

from .metrics import (
    _tokenise,
    approx_tokens as _approx_tokens,
    bleu_1,
    contains_code,
    exact_match,
    exact_match_normalised,
    f1_score,
    rouge_l,
)
from .runner import BenchmarkResult, BenchmarkRunner
from .spec import BenchmarkSpec, SAMPLE_TASKS, Task

__all__ = [
    "__version__",
    # Core classes
    "BenchmarkRunner",
    "BenchmarkResult",
    "BenchmarkSpec",
    "Task",
    "SAMPLE_TASKS",
    # Metric functions
    "exact_match",
    "exact_match_normalised",
    "rouge_l",
    "bleu_1",
    "f1_score",
    # Helpers (underscore-prefixed for back-compat with standalone script)
    "_approx_tokens",
    "_tokenise",
    "contains_code",
]
