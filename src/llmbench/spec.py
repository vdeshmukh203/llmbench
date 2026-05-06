"""Benchmark task specification and built-in sample tasks.

A :class:`BenchmarkSpec` is an ordered collection of :class:`Task` objects.
Tasks can be loaded from JSON, JSONL, or YAML files (YAML requires PyYAML).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Task:
    """A single benchmark task.

    Parameters
    ----------
    task_id:
        Unique identifier, e.g. ``"qa_01"``.
    category:
        Logical grouping, e.g. ``"qa"``, ``"coding"``, ``"summarization"``.
    prompt:
        The text sent to the model.
    reference:
        The expected / gold-standard answer used for scoring.
    metadata:
        Optional free-form dictionary for dataset provenance, difficulty, etc.
    """

    task_id: str
    category: str
    prompt: str
    reference: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict representation suitable for JSON serialisation."""
        return asdict(self)


#: Ten built-in sample tasks spanning three categories.
SAMPLE_TASKS: List[Task] = [
    Task("qa_01", "qa", "What is the capital of France?", "Paris"),
    Task("qa_02", "qa", "What year did World War II end?", "1945"),
    Task("qa_03", "qa", "What is the chemical symbol for water?", "H2O"),
    Task("qa_04", "qa", "Who wrote Pride and Prejudice?", "Jane Austen"),
    Task("qa_05", "qa", "What is the square root of 144?", "12"),
    Task(
        "code_01",
        "coding",
        "Write a Python function that returns the factorial of n.",
        "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)",
    ),
    Task("code_02", "coding", "Write a Python one-liner to reverse a list called lst.", "lst[::-1]"),
    Task(
        "code_03",
        "coding",
        "Write a Python function to check if a string is a palindrome.",
        "def is_palindrome(s):\n    return s == s[::-1]",
    ),
    Task(
        "summ_01",
        "summarization",
        "Summarize in one sentence: The quick brown fox jumps over the lazy dog. "
        "The dog did not seem to mind.",
        "A fox jumped over a lazy dog.",
    ),
    Task(
        "summ_02",
        "summarization",
        "Summarize: Water boils at 100 degrees Celsius at sea level.",
        "Water boils at 100 degrees Celsius at sea level.",
    ),
]


class BenchmarkSpec:
    """An ordered collection of benchmark tasks with loading helpers.

    Tasks can be provided directly, loaded from a file, or constructed from
    plain Python dicts.

    Parameters
    ----------
    tasks:
        Explicit list of :class:`Task` objects.  Defaults to
        :data:`SAMPLE_TASKS` when *None*.

    Examples
    --------
    >>> spec = BenchmarkSpec()            # built-in sample tasks
    >>> spec = BenchmarkSpec.from_file("my_tasks.jsonl")
    >>> spec = BenchmarkSpec.from_dicts([{"task_id": "t1", "category": "qa",
    ...                                   "prompt": "...", "reference": "..."}])
    """

    def __init__(self, tasks: Optional[List[Task]] = None) -> None:
        self.tasks: List[Task] = tasks if tasks is not None else list(SAMPLE_TASKS)

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: "Path | str") -> "BenchmarkSpec":
        """Load tasks from a JSON, JSONL, or YAML file.

        Parameters
        ----------
        path:
            Path to the task file.  Supported extensions: ``.json``,
            ``.jsonl``, ``.yaml``, ``.yml``.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ImportError
            If a ``.yaml``/``.yml`` file is given but PyYAML is not installed.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark spec file not found: {path}")

        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "PyYAML is required for YAML task specs: pip install pyyaml"
                ) from exc
            with path.open(encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            raw = data if isinstance(data, list) else data.get("tasks", [])
        elif suffix == ".jsonl":
            with path.open(encoding="utf-8") as fh:
                raw = [json.loads(line) for line in fh if line.strip()]
        else:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            raw = data if isinstance(data, list) else data.get("tasks", [])

        return cls([cls._task_from_dict(d) for d in raw])

    @classmethod
    def from_dicts(cls, records: List[Dict[str, Any]]) -> "BenchmarkSpec":
        """Create a spec from a list of plain dicts."""
        return cls([cls._task_from_dict(d) for d in records])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _task_from_dict(d: Dict[str, Any]) -> Task:
        return Task(
            task_id=str(d.get("task_id", "")),
            category=str(d.get("category", "unknown")),
            prompt=str(d.get("prompt", "")),
            reference=str(d.get("reference", d.get("answer", ""))),
            metadata=dict(d.get("metadata", {})),
        )

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def __repr__(self) -> str:  # pragma: no cover
        return f"BenchmarkSpec(n={len(self.tasks)})"
