"""Task specification and BenchmarkSpec loader.

Tasks can be written in JSONL (one JSON object per line, no extra dependencies)
or YAML (requires ``pip install pyyaml``).

JSONL format::

    {"task_id": "qa_01", "category": "qa", "prompt": "What is 2+2?", "reference": "4"}

YAML format (one file, list under optional ``tasks`` key)::

    tasks:
      - task_id: qa_01
        category: qa
        prompt: "What is 2+2?"
        reference: "4"
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml as _yaml  # type: ignore

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


@dataclass
class Task:
    """A single benchmark prompt with an expected reference answer."""

    task_id: str
    category: str
    prompt: str
    reference: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: Dict[str, Any], *, lineno: int = 0) -> "Task":
        for key in ("task_id", "category", "prompt", "reference"):
            if key not in obj:
                loc = f" at line {lineno}" if lineno else ""
                raise ValueError(f"Task{loc} missing required field '{key}'")
        return cls(
            task_id=obj["task_id"],
            category=obj["category"],
            prompt=obj["prompt"],
            reference=obj["reference"],
            metadata=obj.get("metadata", {}),
        )


class BenchmarkSpec:
    """
    Container for a named suite of benchmark :class:`Task` objects.

    Tasks can be loaded from:

    * **JSONL** files (one task per line) via :meth:`from_jsonl` — no extra deps.
    * **YAML** files via :meth:`from_yaml` — requires ``pyyaml``.
    * **Python lists** via :meth:`from_list` — for programmatic use.

    The declarative file format makes benchmark suites version-controllable
    and shareable, satisfying JOSS reproducibility requirements.
    """

    def __init__(self, tasks: List[Task], name: str = "unnamed"):
        if not tasks:
            raise ValueError("BenchmarkSpec requires at least one task")
        self.tasks = tasks
        self.name = name

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_jsonl(cls, path: Path, name: Optional[str] = None) -> "BenchmarkSpec":
        """Load tasks from a JSONL file (one task object per line)."""
        path = Path(path)
        tasks: List[Task] = []
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                tasks.append(Task.from_dict(json.loads(line), lineno=lineno))
        if not tasks:
            raise ValueError(f"{path} contains no tasks")
        return cls(tasks, name=name or path.stem)

    @classmethod
    def from_yaml(cls, path: Path, name: Optional[str] = None) -> "BenchmarkSpec":
        """Load tasks from a YAML file.

        Requires PyYAML: ``pip install pyyaml``
        """
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML support: pip install pyyaml"
            )
        path = Path(path)
        with path.open(encoding="utf-8") as fh:
            data = _yaml.safe_load(fh)
        tasks_data = (
            data.get("tasks", data) if isinstance(data, dict) else data
        )
        if not isinstance(tasks_data, list):
            raise ValueError(
                f"{path} must be a list of tasks or a dict with a 'tasks' key"
            )
        tasks = [Task.from_dict(obj, lineno=i) for i, obj in enumerate(tasks_data, 1)]
        return cls(tasks, name=name or path.stem)

    @classmethod
    def from_list(cls, data: List[Dict[str, Any]], name: str = "inline") -> "BenchmarkSpec":
        """Load tasks from a list of dicts (programmatic use)."""
        tasks = [Task.from_dict(obj, lineno=i) for i, obj in enumerate(data, 1)]
        return cls(tasks, name=name)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_jsonl(self, path: Path) -> None:
        """Write tasks to a JSONL file."""
        with Path(path).open("w", encoding="utf-8") as fh:
            for task in self.tasks:
                fh.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self.tasks)

    def __repr__(self) -> str:
        return f"BenchmarkSpec(name={self.name!r}, n_tasks={len(self.tasks)})"
