"""BenchmarkRunner: executes tasks and aggregates results."""

from __future__ import annotations

import csv
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .models import Task, BenchmarkResult, SAMPLE_TASKS
from .metrics import (
    exact_match,
    exact_match_normalised,
    rouge_l as _rouge_l,
    bleu_1 as _bleu_1,
    f1_score,
    _approx_tokens,
)


class BenchmarkRunner:
    def __init__(self, tasks: Optional[List[Task]] = None) -> None:
        self.tasks: List[Task] = tasks if tasks is not None else list(SAMPLE_TASKS)
        self.results: List[BenchmarkResult] = []

    # ── internal scoring ──────────────────────────────────────────────────────

    def _score(
        self, task: Task, prediction: str, latency: float, error: Optional[str]
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
            rouge_l=_rouge_l(prediction, task.reference),
            bleu_1=_bleu_1(prediction, task.reference),
            f1=f1_score(prediction, task.reference),
            approx_tokens=_approx_tokens(prediction),
            error=error,
        )

    # ── inference methods ─────────────────────────────────────────────────────

    def run_offline(
        self,
        model_fn: Callable[[str], str],
        tasks: Optional[List[Task]] = None,
    ) -> List[BenchmarkResult]:
        """Run benchmark with a local callable as the model."""
        tasks = tasks if tasks is not None else self.tasks
        results: List[BenchmarkResult] = []
        for task in tasks:
            t0 = time.monotonic()
            try:
                pred = model_fn(task.prompt)
                err: Optional[str] = None
            except Exception as exc:
                pred = ""
                err = str(exc)
            lat = time.monotonic() - t0
            results.append(self._score(task, pred, lat, err))
        self.results.extend(results)
        return results

    def run_openai(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        tasks: Optional[List[Task]] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> List[BenchmarkResult]:
        """Run benchmark against an OpenAI-compatible chat completions endpoint."""
        tasks = tasks if tasks is not None else self.tasks
        results: List[BenchmarkResult] = []
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        for task in tasks:
            payload = json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": "Answer concisely."},
                    {"role": "user", "content": task.prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }).encode()
            t0 = time.monotonic()
            try:
                req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
                pred = data["choices"][0]["message"]["content"].strip()
                err = None
            except Exception as exc:
                pred = ""
                err = str(exc)
            lat = time.monotonic() - t0
            results.append(self._score(task, pred, lat, err))
        self.results.extend(results)
        return results

    # ── aggregation ───────────────────────────────────────────────────────────

    def summarize(self, results: Optional[List[BenchmarkResult]] = None) -> Dict[str, Any]:
        """Return per-category and overall aggregate statistics."""
        results = results if results is not None else self.results
        if not results:
            return {}

        def _avg(vals: List[float]) -> float:
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

    # ── export ────────────────────────────────────────────────────────────────

    def export_csv(self, path: Path, results: Optional[List[BenchmarkResult]] = None) -> None:
        results = results if results is not None else self.results
        if not results:
            return
        fieldnames = list(results[0].to_dict().keys()) + ["composite_score"]
        with Path(path).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = r.to_dict()
                row["composite_score"] = round(r.composite_score, 4)
                writer.writerow(row)

    def export_jsonl(self, path: Path, results: Optional[List[BenchmarkResult]] = None) -> None:
        results = results if results is not None else self.results
        if not results:
            return
        with Path(path).open("w", encoding="utf-8") as fh:
            for r in results:
                row = r.to_dict()
                row["composite_score"] = round(r.composite_score, 4)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
