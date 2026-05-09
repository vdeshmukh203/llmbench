"""
llmbench: Lightweight reproducible LLM benchmarking framework.

Defines a declarative task specification format and provides a runner that
executes prompt suites against one or more LLM providers, records responses
with SHA-256-linked provenance, scores outputs using configurable metrics,
and emits reproducible benchmark reports in JSON, CSV, and Markdown formats.

Quick start::

    from llmbench import BenchmarkRunner, BenchmarkSpec, SAMPLE_TASKS

    runner = BenchmarkRunner(SAMPLE_TASKS)
    results = runner.run_offline(my_model_fn)
    print(runner.summarize())
    runner.export_markdown("report.md")

Metrics are also available as standalone functions::

    from llmbench.metrics import rouge_l, bleu_1, f1_score, exact_match
"""

__version__ = "0.1.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

from ._sample_tasks import SAMPLE_TASKS
from .metrics import (
    bleu_1,
    composite_score,
    exact_match,
    exact_match_normalised,
    f1_score,
    rouge_l,
)
from .runner import BenchmarkResult, BenchmarkRunner
from .spec import BenchmarkSpec, Task

__all__ = [
    # Core classes
    "BenchmarkRunner",
    "BenchmarkResult",
    "BenchmarkSpec",
    "Task",
    # Metrics
    "rouge_l",
    "bleu_1",
    "f1_score",
    "exact_match",
    "exact_match_normalised",
    "composite_score",
    # Data
    "SAMPLE_TASKS",
]
