"""BenchmarkRunner: executes tasks against a callable or OpenAI-compatible API."""

from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .metrics import (
    approx_tokens,
    bleu_1,
    exact_match,
    exact_match_normalised,
    f1_score,
    rouge_l,
)
from .spec import SAMPLE_TASKS, Task


@dataclass
class BenchmarkResult:
    """Scored result for a single benchmark task."""

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

    @property
    def composite_score(self) -> float:
        """Weighted composite: 0.4·ROUGE-L + 0.3·F1 + 0.2·EM-norm + 0.1·BLEU-1."""
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


class BenchmarkRunner:
    """Runs benchmark tasks and accumulates :class:`BenchmarkResult` objects.

    Parameters
    ----------
    tasks:
        Tasks to benchmark. Defaults to :data:`~llmbench.spec.SAMPLE_TASKS`.
    """

    def __init__(self, tasks: Optional[List[Task]] = None) -> None:
        self.tasks: List[Task] = list(tasks) if tasks is not None else list(SAMPLE_TASKS)
        self.results: List[BenchmarkResult] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score(
        self,
        task: Task,
        prediction: str,
        latency: float,
        error: Optional[str],
    ) -> BenchmarkResult:
        return BenchmarkResult(
            task_id=task.task_id,
            category=task.category,
            prompt=task.prompt,
            reference=task.reference,
            prediction=prediction,
            latency_s=round(latency, 4),
            exact_match=exact_match(prediction, task.reference),
            exact_match_norm=exact_match_normalised(prediction, task.reference),
            rouge_l=rouge_l(prediction, task.reference),
            bleu_1=bleu_1(prediction, task.reference),
            f1=f1_score(prediction, task.reference),
            approx_tokens=approx_tokens(prediction),
            error=error,
        )

    # ------------------------------------------------------------------
    # Run methods
    # ------------------------------------------------------------------

    def run_offline(
        self,
        model_fn: Callable[[str], str],
        tasks: Optional[List[Task]] = None,
    ) -> List[BenchmarkResult]:
        """Evaluate *model_fn(prompt) -> str* against each task.

        Results are appended to :attr:`results` and also returned.
        """
        tasks = tasks if tasks is not None else self.tasks
        results: List[BenchmarkResult] = []
        for task in tasks:
            t0 = time.monotonic()
            try:
                prediction = model_fn(task.prompt)
                latency = time.monotonic() - t0
                err: Optional[str] = None
            except Exception as exc:  # noqa: BLE001
                prediction = ""
                latency = time.monotonic() - t0
                err = str(exc)
            results.append(self._score(task, prediction, latency, err))
        self.results.extend(results)
        return results

    def run_openai(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        tasks: Optional[List[Task]] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        timeout: int = 30,
    ) -> List[BenchmarkResult]:
        """Call an OpenAI-compatible chat completions endpoint for each task.

        Results are appended to :attr:`results` and also returned.
        """
        tasks = tasks if tasks is not None else self.tasks
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        results: List[BenchmarkResult] = []
        for task in tasks:
            payload = json.dumps(
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Answer concisely."},
                        {"role": "user", "content": task.prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            ).encode()
            t0 = time.monotonic()
            try:
                req = urllib.request.Request(
                    url, data=payload, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read())
                latency = time.monotonic() - t0
                prediction = data["choices"][0]["message"]["content"].strip()
                err = None
            except urllib.error.HTTPError as exc:
                latency = time.monotonic() - t0
                prediction = ""
                body = exc.read().decode(errors="replace") if exc.fp else ""
                err = f"HTTP {exc.code}: {body}"
            except Exception as exc:  # noqa: BLE001
                latency = time.monotonic() - t0
                prediction = ""
                err = str(exc)
            results.append(self._score(task, prediction, latency, err))
        self.results.extend(results)
        return results

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def summarize(
        self, results: Optional[List[BenchmarkResult]] = None
    ) -> Dict[str, Any]:
        """Return per-category and overall aggregate statistics.

        Returns an empty dict if there are no results.
        """
        results = results if results is not None else self.results
        if not results:
            return {}

        def _avg(vals: List[float]) -> float:
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        cats: Dict[str, List[BenchmarkResult]] = {}
        for r in results:
            cats.setdefault(r.category, []).append(r)

        summary: Dict[str, Any] = {"overall": {}, "by_category": {}}
        for cat, cr in sorted(cats.items()):
            summary["by_category"][cat] = {
                "n": len(cr),
                "exact_match": _avg([r.exact_match for r in cr]),
                "exact_match_norm": _avg([r.exact_match_norm for r in cr]),
                "rouge_l": _avg([r.rouge_l for r in cr]),
                "bleu_1": _avg([r.bleu_1 for r in cr]),
                "f1": _avg([r.f1 for r in cr]),
                "composite": _avg([r.composite_score for r in cr]),
                "avg_latency_s": _avg([r.latency_s for r in cr]),
                "errors": sum(1 for r in cr if r.error),
            }
        summary["overall"] = {
            "n": len(results),
            "exact_match": _avg([r.exact_match for r in results]),
            "exact_match_norm": _avg([r.exact_match_norm for r in results]),
            "rouge_l": _avg([r.rouge_l for r in results]),
            "bleu_1": _avg([r.bleu_1 for r in results]),
            "f1": _avg([r.f1 for r in results]),
            "composite": _avg([r.composite_score for r in results]),
            "avg_latency_s": _avg([r.latency_s for r in results]),
            "errors": sum(1 for r in results if r.error),
        }
        return summary

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(
        self, path: Path, results: Optional[List[BenchmarkResult]] = None
    ) -> None:
        """Export results to *path* as CSV."""
        results = results if results is not None else self.results
        if not results:
            return
        with path.open("w", newline="", encoding="utf-8") as fh:
            fieldnames = list(results[0].to_dict().keys())
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_dict())

    def export_jsonl(
        self, path: Path, results: Optional[List[BenchmarkResult]] = None
    ) -> None:
        """Export results to *path* as JSONL (one JSON object per line)."""
        results = results if results is not None else self.results
        if not results:
            return
        with path.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
