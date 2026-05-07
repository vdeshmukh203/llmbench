"""BenchmarkSpec: declarative benchmark specification loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import Task


class BenchmarkSpec:
    """Declarative benchmark specification that holds a named collection of tasks.

    Tasks can be loaded from JSON (object or array) or JSONL files.
    YAML support is available when PyYAML is installed.
    """

    def __init__(
        self,
        tasks: List[Task],
        name: str = "benchmark",
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.tasks = tasks
        self.name = name
        self.description = description
        self.metadata: Dict[str, Any] = metadata or {}

    # ── loaders ───────────────────────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: Path) -> "BenchmarkSpec":
        """Load from a JSON file (``{"tasks": [...]}`` or bare array)."""
        with Path(path).open(encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            tasks = [cls._task_from_dict(d) for d in data]
            return cls(tasks=tasks)
        tasks = [cls._task_from_dict(d) for d in data.get("tasks", [])]
        return cls(
            tasks=tasks,
            name=data.get("name", "benchmark"),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_jsonl(cls, path: Path) -> "BenchmarkSpec":
        """Load from a JSONL file where each line is one task object."""
        tasks: List[Task] = []
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                tasks.append(cls._task_from_dict(json.loads(line)))
        return cls(tasks=tasks)

    @classmethod
    def from_yaml(cls, path: Path) -> "BenchmarkSpec":
        """Load from a YAML file. Requires ``pip install pyyaml``."""
        try:
            import yaml  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML support: pip install pyyaml"
            ) from exc
        with Path(path).open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, list):
            tasks = [cls._task_from_dict(d) for d in data]
            return cls(tasks=tasks)
        tasks = [cls._task_from_dict(d) for d in data.get("tasks", [])]
        return cls(
            tasks=tasks,
            name=data.get("name", "benchmark"),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )

    # ── exporter ──────────────────────────────────────────────────────────────

    def to_json(self, path: Path) -> None:
        """Serialise spec to a JSON file."""
        data = {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "tasks": [t.to_dict() for t in self.tasks],
        }
        with Path(path).open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _task_from_dict(d: Dict[str, Any]) -> Task:
        return Task(
            task_id=d.get("task_id", "unknown"),
            category=d.get("category", "unknown"),
            prompt=d.get("prompt", ""),
            reference=d.get("reference", ""),
            metadata=d.get("metadata", {}),
        )
