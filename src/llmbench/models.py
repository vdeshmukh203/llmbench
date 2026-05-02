"""Data models for benchmark tasks and results."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


@dataclass
class Task:
    """A single benchmark task: a prompt paired with a reference answer."""

    task_id: str
    category: str
    prompt: str
    reference: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def checksum(self) -> str:
        """SHA-256 fingerprint of (task_id, prompt, reference) for provenance tracking."""
        payload = json.dumps(
            {"task_id": self.task_id, "prompt": self.prompt, "reference": self.reference},
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass
class BenchmarkResult:
    """Scored result for one task execution."""

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
    task_checksum: str
    run_timestamp: str
    error: Optional[str] = None

    @property
    def composite_score(self) -> float:
        """Weighted composite: 0.4×ROUGE-L + 0.3×F1 + 0.2×EM_norm + 0.1×BLEU-1."""
        return (
            0.40 * self.rouge_l
            + 0.30 * self.f1
            + 0.20 * self.exact_match_norm
            + 0.10 * self.bleu_1
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["composite_score"] = round(self.composite_score, 4)
        return d

    def result_checksum(self) -> str:
        """SHA-256 fingerprint of this result for tamper detection."""
        payload = json.dumps(
            {
                "task_id": self.task_id,
                "prediction": self.prediction,
                "rouge_l": self.rouge_l,
                "bleu_1": self.bleu_1,
                "f1": self.f1,
                "exact_match": self.exact_match,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
