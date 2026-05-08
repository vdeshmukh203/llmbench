"""
llmbench: Lightweight reproducible LLM benchmarking framework.

Provides a declarative benchmark specification format, a provider-agnostic
runner for OpenAI-compatible APIs and local models, configurable text-overlap
metrics, and structured JSONL/CSV output with full provenance.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

from .metrics import (
    _approx_tokens,
    _tokenise,
    bleu_1,
    contains_code,
    exact_match,
    exact_match_normalised,
    f1_score,
    rouge_l,
)
from .runner import BenchmarkResult, BenchmarkRunner
from .spec import SAMPLE_TASKS, BenchmarkSpec, Task

__all__ = [
    # Core classes
    "BenchmarkRunner",
    "BenchmarkResult",
    "BenchmarkSpec",
    "Task",
    "SAMPLE_TASKS",
    # Metrics
    "exact_match",
    "exact_match_normalised",
    "rouge_l",
    "bleu_1",
    "f1_score",
    # Utilities
    "_approx_tokens",
    "_tokenise",
    "contains_code",
]
