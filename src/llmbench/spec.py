"""BenchmarkSpec: declarative JSON benchmark specification."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .tasks import Task


@dataclass
class BenchmarkSpec:
    """Declarative benchmark configuration loadable from a JSON file.

    A spec bundles the task suite, model identifier, and inference
    parameters into a single reproducible artefact.

    Example JSON::

        {
          "name": "my-bench",
          "description": "Quick QA evaluation",
          "model": "gpt-4o-mini",
          "max_tokens": 256,
          "temperature": 0.0,
          "tasks": [
            {"task_id": "q1", "category": "qa", "prompt": "Capital of France?", "reference": "Paris"}
          ]
        }
    """

    name: str
    description: str = ""
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 256
    temperature: float = 0.0
    tasks: List[Task] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkSpec":
        """Build a spec from a plain dict (e.g., parsed JSON)."""
        if "name" not in data:
            raise ValueError("Spec dict must contain a 'name' field.")
        tasks = [
            Task(
                task_id=t["task_id"],
                category=t.get("category", "general"),
                prompt=t["prompt"],
                reference=t["reference"],
                metadata=t.get("metadata", {}),
            )
            for t in data.get("tasks", [])
        ]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            model=data.get("model", "gpt-3.5-turbo"),
            max_tokens=int(data.get("max_tokens", 256)),
            temperature=float(data.get("temperature", 0.0)),
            tasks=tasks,
        )

    @classmethod
    def from_file(cls, path: Path) -> "BenchmarkSpec":
        """Load a spec from a JSON file."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Spec file not found: {path}")
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "tasks": [t.to_dict() for t in self.tasks],
        }

    def save(self, path: Path) -> None:
        """Persist the spec to a JSON file."""
        path = Path(path)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
            fh.write("\n")
