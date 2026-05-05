"""
llmbench: Lightweight reproducible LLM benchmarking framework.

Exports the full public API so callers can do::

    import llmbench as lb
    runner = lb.BenchmarkRunner()
    results = runner.run_offline(my_model_fn)
    print(lb.rouge_l("Paris", "Paris"))   # 1.0
"""

__version__ = "0.1.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

from .metrics import (
    _tokenise,
    approx_tokens,
    approx_tokens as _approx_tokens,  # backwards-compatible alias
    bleu_1,
    contains_code,
    exact_match,
    exact_match_normalised,
    f1_score,
    rouge_l,
)
from .tasks import BenchmarkResult, SAMPLE_TASKS, Task
from .runner import BenchmarkRunner
from .spec import BenchmarkSpec

__all__ = [
    # classes
    "BenchmarkRunner",
    "BenchmarkSpec",
    "Task",
    "BenchmarkResult",
    # data
    "SAMPLE_TASKS",
    # metrics
    "exact_match",
    "exact_match_normalised",
    "rouge_l",
    "bleu_1",
    "f1_score",
    "approx_tokens",
    "_approx_tokens",
    "_tokenise",
    "contains_code",
]
