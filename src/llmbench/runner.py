"""BenchmarkRunner and BenchmarkResult.

Each result includes a SHA-256 hash of the (prompt, prediction, reference)
triple so that benchmark numbers can be independently reproduced and verified.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .metrics import (
    bleu_1,
    composite_score,
    exact_match,
    exact_match_normalised,
    f1_score,
    rouge_l,
)
from .spec import BenchmarkSpec, Task


def _approx_tokens(text: str) -> int:
    return len(re.findall(r"[a-zA-Z0-9]+", text))


@dataclass
class BenchmarkResult:
    """Scores and metadata for one benchmark task evaluation."""

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
    sha256: str = ""
    error: Optional[str] = None

    @property
    def composite(self) -> float:
        """Weighted composite: 0.40·ROUGE-L + 0.30·F1 + 0.20·EM_norm + 0.10·BLEU-1."""
        return composite_score(self.rouge_l, self.f1, self.exact_match_norm, self.bleu_1)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["composite_score"] = round(self.composite, 4)
        return d


class BenchmarkRunner:
    """
    Runs a benchmark task suite against an LLM and scores the outputs.

    Supports three inference modes:

    * :meth:`run_offline` — pass any Python callable as the model.
    * :meth:`run_openai` — call an OpenAI-compatible chat completions API.
    * :meth:`run_anthropic` — call the Anthropic Messages API.

    All results are stored in :attr:`results` and contain SHA-256 hashes
    of each (prompt, prediction, reference) triple for reproducibility
    verification.

    Example::

        runner = BenchmarkRunner(SAMPLE_TASKS)
        results = runner.run_offline(my_model)
        print(runner.summarize())
        runner.export_jsonl("results.jsonl")
    """

    def __init__(
        self,
        tasks: Optional[List[Task]] = None,
        spec: Optional[BenchmarkSpec] = None,
    ):
        if spec is not None:
            self.tasks: List[Task] = spec.tasks
        else:
            self.tasks = list(tasks) if tasks else []
        self.results: List[BenchmarkResult] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sha256(prompt: str, prediction: str, reference: str) -> str:
        payload = json.dumps(
            {"prompt": prompt, "prediction": prediction, "reference": reference},
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def _score(
        self, task: Task, prediction: str, latency: float, error: Optional[str]
    ) -> BenchmarkResult:
        sha = "" if error else self._sha256(task.prompt, prediction, task.reference)
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
            approx_tokens=_approx_tokens(prediction),
            sha256=sha,
            error=error,
        )

    # ------------------------------------------------------------------
    # Runners
    # ------------------------------------------------------------------

    def run_offline(
        self,
        model_fn: Callable[[str], str],
        tasks: Optional[List[Task]] = None,
        progress_callback: Optional[Callable[[int, int, "BenchmarkResult"], None]] = None,
    ) -> List[BenchmarkResult]:
        """Evaluate *model_fn(prompt) -> prediction* against each task.

        Parameters
        ----------
        model_fn:
            Any callable that accepts a prompt string and returns a prediction.
        tasks:
            Task subset to evaluate; defaults to ``self.tasks``.
        progress_callback:
            Optional ``fn(done, total, result)`` called after each task.
        """
        tasks = tasks if tasks is not None else self.tasks
        results: List[BenchmarkResult] = []
        for i, task in enumerate(tasks):
            t0 = time.monotonic()
            try:
                pred = model_fn(task.prompt)
                lat = time.monotonic() - t0
                err = None
            except Exception as exc:
                pred = ""
                lat = time.monotonic() - t0
                err = str(exc)
            result = self._score(task, pred, lat, err)
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(tasks), result)
        self.results.extend(results)
        return results

    def run_openai(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        tasks: Optional[List[Task]] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: str = "Answer concisely.",
        progress_callback: Optional[Callable[[int, int, "BenchmarkResult"], None]] = None,
    ) -> List[BenchmarkResult]:
        """Run benchmark against an OpenAI-compatible chat completions API."""
        tasks = tasks if tasks is not None else self.tasks
        results: List[BenchmarkResult] = []
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        for i, task in enumerate(tasks):
            payload = json.dumps(
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
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
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                lat = time.monotonic() - t0
                pred = data["choices"][0]["message"]["content"].strip()
                err = None
            except Exception as exc:
                lat = time.monotonic() - t0
                pred = ""
                err = str(exc)
            result = self._score(task, pred, lat, err)
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(tasks), result)
        self.results.extend(results)
        return results

    def run_anthropic(
        self,
        api_key: str,
        model: str = "claude-3-haiku-20240307",
        tasks: Optional[List[Task]] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        progress_callback: Optional[Callable[[int, int, "BenchmarkResult"], None]] = None,
    ) -> List[BenchmarkResult]:
        """Run benchmark against the Anthropic Messages API."""
        tasks = tasks if tasks is not None else self.tasks
        results: List[BenchmarkResult] = []
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        for i, task in enumerate(tasks):
            payload = json.dumps(
                {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": task.prompt}],
                }
            ).encode()
            t0 = time.monotonic()
            try:
                req = urllib.request.Request(
                    url, data=payload, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                lat = time.monotonic() - t0
                pred = data["content"][0]["text"].strip()
                err = None
            except Exception as exc:
                lat = time.monotonic() - t0
                pred = ""
                err = str(exc)
            result = self._score(task, pred, lat, err)
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(tasks), result)
        self.results.extend(results)
        return results

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def summarize(self, results: Optional[List[BenchmarkResult]] = None) -> Dict[str, Any]:
        """Return per-category and overall aggregate statistics."""
        results = results if results is not None else self.results
        if not results:
            return {}

        def _avg(vals: list) -> float:
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        cats: Dict[str, List[BenchmarkResult]] = {}
        for r in results:
            cats.setdefault(r.category, []).append(r)

        summary: Dict[str, Any] = {"overall": {}, "by_category": {}}
        for cat, cr in cats.items():
            summary["by_category"][cat] = {
                "n": len(cr),
                "exact_match": _avg([r.exact_match for r in cr]),
                "exact_match_norm": _avg([r.exact_match_norm for r in cr]),
                "rouge_l": _avg([r.rouge_l for r in cr]),
                "bleu_1": _avg([r.bleu_1 for r in cr]),
                "f1": _avg([r.f1 for r in cr]),
                "composite": _avg([r.composite for r in cr]),
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
            "composite": _avg([r.composite for r in results]),
            "avg_latency_s": _avg([r.latency_s for r in results]),
            "errors": sum(1 for r in results if r.error),
        }
        return summary

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_jsonl(
        self, path: Path, results: Optional[List[BenchmarkResult]] = None
    ) -> None:
        """Write results to a JSONL file (one JSON object per line)."""
        results = results if results is not None else self.results
        with Path(path).open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    def export_csv(
        self, path: Path, results: Optional[List[BenchmarkResult]] = None
    ) -> None:
        """Write results to a CSV file."""
        results = results if results is not None else self.results
        if not results:
            return
        fieldnames = list(results[0].to_dict().keys())
        with Path(path).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_dict())

    def export_markdown(
        self, path: Path, results: Optional[List[BenchmarkResult]] = None
    ) -> None:
        """Write a Markdown summary report."""
        results = results if results is not None else self.results
        summary = self.summarize(results)
        ov = summary.get("overall", {})
        lines = [
            "# llmbench Report\n",
            "## Overall",
            "| Metric | Score |",
            "|--------|-------|",
            f"| Tasks | {ov.get('n', 0)} |",
            f"| Exact Match | {ov.get('exact_match', 0):.4f} |",
            f"| Exact Match (norm) | {ov.get('exact_match_norm', 0):.4f} |",
            f"| ROUGE-L | {ov.get('rouge_l', 0):.4f} |",
            f"| BLEU-1 | {ov.get('bleu_1', 0):.4f} |",
            f"| F1 | {ov.get('f1', 0):.4f} |",
            f"| Composite | {ov.get('composite', 0):.4f} |",
            f"| Avg Latency (s) | {ov.get('avg_latency_s', 0):.4f} |",
            f"| Errors | {ov.get('errors', 0)} |",
            "",
            "## By Category",
            "| Category | N | ROUGE-L | F1 | Composite |",
            "|----------|---|---------|-----|-----------|",
        ]
        for cat, stats in summary.get("by_category", {}).items():
            lines.append(
                f"| {cat} | {stats['n']} | {stats['rouge_l']:.4f}"
                f" | {stats['f1']:.4f} | {stats['composite']:.4f} |"
            )
        lines += [
            "",
            "## Per-Task Results",
            "| ID | Category | EM | ROUGE-L | F1 | Composite | Latency |",
            "|----|----------|----|---------|-----|-----------|---------|",
        ]
        for r in results:
            lines.append(
                f"| {r.task_id} | {r.category} | {r.exact_match:.2f}"
                f" | {r.rouge_l:.4f} | {r.f1:.4f} | {r.composite:.4f}"
                f" | {r.latency_s:.3f}s |"
            )
        with Path(path).open("w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
