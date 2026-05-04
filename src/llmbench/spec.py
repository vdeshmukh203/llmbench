"""Task and benchmark specification dataclasses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Task:
    """A single benchmark task with a prompt and expected reference answer."""

    task_id: str
    category: str
    prompt: str
    reference: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Per-task evaluation result produced by :class:`BenchmarkRunner`."""

    task_id: str
    category: str
    prompt: str
    reference: str
    prediction: str
    latency_s: float
    exact_match: float
    exact_match_norm: float
    rouge_l: float
    bleu_1: float
    f1: float
    approx_tokens: int
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def composite_score(self) -> float:
        """Weighted composite: 0.40·ROUGE-L + 0.30·F1 + 0.20·EM-norm + 0.10·BLEU-1."""
        return (
            0.40 * self.rouge_l
            + 0.30 * self.f1
            + 0.20 * self.exact_match_norm
            + 0.10 * self.bleu_1
        )


#: Built-in sample tasks covering QA, coding, and summarisation.
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
        "Summarize in one sentence: The quick brown fox jumps over the lazy dog. The dog did not seem to mind.",
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
    """Declarative collection of :class:`Task` objects to evaluate.

    Parameters
    ----------
    tasks:
        Explicit task list.  Defaults to :data:`SAMPLE_TASKS` when *None*.
    """

    SAMPLE_TASKS: List[Task] = SAMPLE_TASKS

    def __init__(self, tasks: Optional[List[Task]] = None) -> None:
        self.tasks: List[Task] = list(tasks) if tasks is not None else list(SAMPLE_TASKS)

    @classmethod
    def from_jsonl(cls, path: "str | Path") -> "BenchmarkSpec":
        """Load tasks from a JSONL file where each line is a JSON task object."""
        tasks: List[Task] = []
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                tasks.append(
                    Task(
                        task_id=obj.get("task_id", "?"),
                        category=obj.get("category", "unknown"),
                        prompt=obj.get("prompt", ""),
                        reference=obj.get("reference", ""),
                        metadata=obj.get("metadata", {}),
                    )
                )
        return cls(tasks)
