"""llmbench: Lightweight reproducible LLM benchmarking framework."""

__version__ = "0.2.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

from .metrics import (
    _tokenise,
    approx_tokens,
    bleu_1,
    contains_code,
    exact_match,
    exact_match_normalised,
    f1_score,
    rouge_l,
)
from .models import BenchmarkResult, Task
from .runner import BenchmarkRunner
from .spec import BenchmarkSpec, SAMPLE_TASKS

# v0.1 compat alias
_approx_tokens = approx_tokens

__all__ = [
    "BenchmarkRunner",
    "BenchmarkSpec",
    "BenchmarkResult",
    "Task",
    "SAMPLE_TASKS",
    "exact_match",
    "exact_match_normalised",
    "rouge_l",
    "bleu_1",
    "f1_score",
    "approx_tokens",
    "contains_code",
    "_tokenise",
    "_approx_tokens",
]
