"""
llmbench: Lightweight reproducible LLM benchmarking framework.

Provides a declarative task specification format, a provider-agnostic runner
that supports OpenAI-compatible APIs and local callables, configurable text
similarity metrics, and structured JSONL/CSV result export.
"""

__version__ = "0.1.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

from .runner import BenchmarkRunner
from .spec import BenchmarkSpec
from .models import Task, BenchmarkResult, SAMPLE_TASKS
from .metrics import (
    exact_match,
    exact_match_normalised,
    rouge_l,
    bleu_1,
    f1_score,
    _approx_tokens,
    contains_code,
)

__all__ = [
    "BenchmarkRunner",
    "BenchmarkSpec",
    "Task",
    "BenchmarkResult",
    "SAMPLE_TASKS",
    "exact_match",
    "exact_match_normalised",
    "rouge_l",
    "bleu_1",
    "f1_score",
    "_approx_tokens",
    "contains_code",
]
