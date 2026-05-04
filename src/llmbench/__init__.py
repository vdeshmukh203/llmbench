"""
llmbench: Lightweight reproducible LLM benchmarking framework.

Public API
----------
BenchmarkRunner  – execute tasks, score outputs, export results
BenchmarkSpec    – declarative task specification (built-in or JSONL)
BenchmarkResult  – per-task result dataclass
Task             – a single benchmark task

Metrics
-------
rouge_l, bleu_1, f1_score, exact_match, exact_match_normalised, approx_tokens
"""

__version__ = "0.1.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

from .metrics import (
    approx_tokens,
    bleu_1,
    exact_match,
    exact_match_normalised,
    f1_score,
    rouge_l,
)
from .runner import BenchmarkRunner
from .spec import BenchmarkResult, BenchmarkSpec, Task

# Convenience alias kept for scripts that reference the old private name
_approx_tokens = approx_tokens

__all__ = [
    "BenchmarkRunner",
    "BenchmarkSpec",
    "BenchmarkResult",
    "Task",
    "rouge_l",
    "bleu_1",
    "f1_score",
    "exact_match",
    "exact_match_normalised",
    "approx_tokens",
    "_approx_tokens",
]
