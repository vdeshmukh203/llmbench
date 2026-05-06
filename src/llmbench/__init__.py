"""llmbench — lightweight reproducible LLM benchmarking framework.

Public API
----------
:class:`BenchmarkSpec`
    Loads and holds an ordered collection of :class:`Task` objects.
:class:`BenchmarkRunner`
    Runs tasks against a callable model function or an OpenAI-compatible API,
    scores results, and summarises them.
:class:`Task`
    Dataclass for a single prompt / reference pair.
:class:`BenchmarkResult`
    Dataclass for a scored inference result.

Metric functions (all return ``float`` in ``[0, 1]``)
    :func:`exact_match`, :func:`exact_match_normalised`, :func:`rouge_l`,
    :func:`bleu_1`, :func:`f1_score`.

The :data:`SAMPLE_TASKS` list provides ten built-in tasks across QA, coding,
and summarisation categories for quick sanity checks.

Example
-------
>>> from llmbench import BenchmarkRunner
>>> runner = BenchmarkRunner()
>>> results = runner.run_offline(lambda p: "42")
>>> runner.summarize()["overall"]["composite"]
0.007...
"""

__version__ = "0.2.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

from .spec import Task, BenchmarkSpec, SAMPLE_TASKS
from .runner import BenchmarkResult, BenchmarkRunner
from .metrics import (
    exact_match,
    exact_match_normalised,
    rouge_l,
    bleu_1,
    f1_score,
    _approx_tokens,
    contains_code,
)
from .cli import main

__all__ = [
    # Core classes
    "BenchmarkSpec",
    "BenchmarkRunner",
    "Task",
    "BenchmarkResult",
    # Metrics
    "exact_match",
    "exact_match_normalised",
    "rouge_l",
    "bleu_1",
    "f1_score",
    "_approx_tokens",
    "contains_code",
    # Data
    "SAMPLE_TASKS",
    # CLI
    "main",
]
