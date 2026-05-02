"""BenchmarkSpec: load and represent a suite of benchmark tasks from JSON or YAML."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import Task

try:
    import yaml as _yaml  # type: ignore

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


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
    """Declarative benchmark specification loaded from JSON or YAML.

    File format (JSON)::

        {
          "name": "my_suite",
          "version": "1.0",
          "tasks": [
            {"task_id": "q1", "category": "qa", "prompt": "2+2?", "reference": "4"}
          ]
        }

    YAML is also supported when PyYAML is installed (``pip install pyyaml``).
    """

    def __init__(
        self,
        name: str,
        tasks: List[Task],
        version: str = "1.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.tasks = tasks
        self.version = version
        self.metadata: Dict[str, Any] = metadata or {}

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: Path) -> "BenchmarkSpec":
        """Load a BenchmarkSpec from a JSON or YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Task file not found: {path}")
        raw = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            if not _YAML_AVAILABLE:
                raise ImportError(
                    "PyYAML is required to load YAML task files: pip install pyyaml"
                )
            data = _yaml.safe_load(raw)
        else:
            data = json.loads(raw)
        return cls._from_dict(data, source=str(path))

    @classmethod
    def _from_dict(cls, data: Dict[str, Any], source: str = "") -> "BenchmarkSpec":
        name = data.get("name", source or "unnamed")
        version = str(data.get("version", "1.0"))
        metadata = {k: v for k, v in data.items() if k not in ("name", "version", "tasks")}
        tasks: List[Task] = []
        for raw in data.get("tasks", []):
            task_id = raw.get("task_id") or raw.get("id", "")
            if not task_id:
                raise ValueError(f"Task is missing 'task_id': {raw}")
            tasks.append(
                Task(
                    task_id=task_id,
                    category=raw.get("category", "general"),
                    prompt=raw.get("prompt", ""),
                    reference=raw.get("reference", ""),
                    metadata={
                        k: v
                        for k, v in raw.items()
                        if k not in ("task_id", "id", "category", "prompt", "reference")
                    },
                )
            )
        return cls(name=name, tasks=tasks, version=version, metadata=metadata)

    @classmethod
    def builtin(cls) -> "BenchmarkSpec":
        """Return the built-in ten-task sample suite."""
        return cls(name="sample", tasks=list(SAMPLE_TASKS), version="0.1.0")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            **self.metadata,
            "tasks": [t.to_dict() for t in self.tasks],
        }

    def save(self, path: Path) -> None:
        """Persist this spec to a JSON file."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def __len__(self) -> int:
        return len(self.tasks)

    def __repr__(self) -> str:
        return f"BenchmarkSpec(name={self.name!r}, tasks={len(self.tasks)}, version={self.version!r})"
