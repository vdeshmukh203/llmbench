"""
llmbench: Lightweight reproducible LLM benchmarking framework.

Defines a declarative YAML-based benchmark specification format and provides
a runner that executes prompt suites against one or more LLM providers,
records responses with SHA-256-linked provenance, scores outputs using
configurable metrics, and emits reproducible benchmark reports in JSON and
Markdown formats.
"""

__version__ = "0.1.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

from .runner import BenchmarkRunner
from .spec import BenchmarkSpec

__all__ = ["BenchmarkRunner", "BenchmarkSpec"]
