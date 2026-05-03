"""Benchmark task specification: Task dataclass, SAMPLE_TASKS, and BenchmarkSpec loader."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List


@dataclass
class Task:
    """A single benchmark task consisting of a prompt and its reference answer."""

    task_id: str
    category: str
    prompt: str
    reference: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
    Task(
        "code_02",
        "coding",
        "Write a Python one-liner to reverse a list called lst.",
        "lst[::-1]",
    ),
    Task(
        "code_03",
        "coding",
        "Write a Python function to check if a string is a palindrome.",
        "def is_palindrome(s):\n    return s == s[::-1]",
    ),
    Task(
        "summ_01",
        "summarization",
        (
            "Summarize in one sentence: The quick brown fox jumps over the lazy dog."
            " The dog did not seem to mind."
        ),
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
    """Loads and holds a collection of :class:`Task` objects.

    Tasks can be loaded from a JSONL file (one JSON object per line), a YAML
    file (list of task dicts), or from the built-in :data:`SAMPLE_TASKS`.
    """

    def __init__(self, tasks: List[Task]) -> None:
        if not tasks:
            raise ValueError("BenchmarkSpec must contain at least one task.")
        self.tasks: List[Task] = tasks

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_jsonl(cls, path: Path) -> "BenchmarkSpec":
        """Load tasks from *path* (JSONL, one task object per line)."""
        tasks: List[Task] = []
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {lineno} of {path}: {exc}") from exc
                tasks.append(
                    Task(
                        task_id=obj.get("task_id", f"task_{lineno}"),
                        category=obj.get("category", "unknown"),
                        prompt=obj.get("prompt", ""),
                        reference=obj.get("reference", ""),
                        metadata=obj.get("metadata", {}),
                    )
                )
        if not tasks:
            raise ValueError(f"No tasks found in {path}.")
        return cls(tasks)

    @classmethod
    def from_yaml(cls, path: Path) -> "BenchmarkSpec":
        """Load tasks from a YAML file (requires PyYAML: ``pip install pyyaml``)."""
        try:
            import yaml  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML specs. Install it with: pip install pyyaml"
            ) from exc
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, list):
            raise ValueError(f"YAML benchmark spec in {path} must be a top-level list.")
        tasks = [
            Task(
                task_id=obj.get("task_id", f"task_{i}"),
                category=obj.get("category", "unknown"),
                prompt=obj.get("prompt", ""),
                reference=obj.get("reference", ""),
                metadata=obj.get("metadata", {}),
            )
            for i, obj in enumerate(data, 1)
        ]
        return cls(tasks)

    @classmethod
    def from_sample(cls) -> "BenchmarkSpec":
        """Return a spec backed by the built-in :data:`SAMPLE_TASKS`."""
        return cls(list(SAMPLE_TASKS))

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[Task]:
        return iter(self.tasks)

    def __repr__(self) -> str:  # pragma: no cover
        return f"BenchmarkSpec(n={len(self.tasks)})"
