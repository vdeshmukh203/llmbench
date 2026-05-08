"""Task and benchmark specification data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class Task:
    """A single evaluation task: a prompt paired with a reference answer."""

    task_id: str
    category: str
    prompt: str
    reference: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkSpec:
    """Declarative configuration for a benchmark run."""

    model: str = "gpt-3.5-turbo"
    max_tokens: int = 256
    temperature: float = 0.0
    tasks: List[Task] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkSpec":
        tasks = [Task(**t) for t in data.get("tasks", [])]
        return cls(
            model=data.get("model", "gpt-3.5-turbo"),
            max_tokens=int(data.get("max_tokens", 256)),
            temperature=float(data.get("temperature", 0.0)),
            tasks=tasks,
        )

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "tasks": [t.to_dict() for t in self.tasks],
        }


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
        "Summarize in one sentence: The quick brown fox jumps over the lazy dog."
        " The dog did not seem to mind.",
        "A fox jumped over a lazy dog.",
    ),
    Task(
        "summ_02",
        "summarization",
        "Summarize: Water boils at 100 degrees Celsius at sea level.",
        "Water boils at 100 degrees Celsius at sea level.",
    ),
]
