"""BenchmarkRunner: execute tasks, score outputs, and export results."""

from __future__ import annotations

import csv
import hashlib
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .metrics import exact_match, exact_match_normalised, rouge_l, bleu_1, f1_score, approx_tokens
from .tasks import Task, BenchmarkResult, SAMPLE_TASKS


def _provenance_checksum(data: dict) -> str:
    """SHA-256 of a canonically serialised result dict for tamper detection."""
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


class BenchmarkRunner:
    """Execute benchmark tasks against a model and collect scored results."""

    def __init__(self, tasks: Optional[List[Task]] = None) -> None:
        self.tasks: List[Task] = list(tasks) if tasks is not None else list(SAMPLE_TASKS)
        self.results: List[BenchmarkResult] = []

    # ------------------------------------------------------------------
    # Scoring
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
    # Runners
    # ------------------------------------------------------------------

    def run_offline(
        self,
        model_fn: Callable[[str], str],
        tasks: Optional[List[Task]] = None,
    ) -> List[BenchmarkResult]:
        """Benchmark a callable model function against *tasks*."""
        task_list = tasks if tasks is not None else self.tasks
        results: List[BenchmarkResult] = []
        for task in task_list:
            t0 = time.monotonic()
            try:
                pred = model_fn(task.prompt)
                err: Optional[str] = None
            except Exception as exc:
                pred = ""
                err = str(exc)
            results.append(self._score(task, pred, time.monotonic() - t0, err))
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
        """Benchmark an OpenAI-compatible chat completions endpoint."""
        task_list = tasks if tasks is not None else self.tasks
        results: List[BenchmarkResult] = []
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        for task in task_list:
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
                req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
                pred = data["choices"][0]["message"]["content"].strip()
                err = None
            except Exception as exc:
                pred = ""
                err = str(exc)
            results.append(self._score(task, pred, time.monotonic() - t0, err))
        self.results.extend(results)
        return results

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    def summarize(self, results: Optional[List[BenchmarkResult]] = None) -> Dict[str, Any]:
        """Return per-category and overall aggregate statistics."""
        result_list = results if results is not None else self.results
        if not result_list:
            return {}

        def _avg(vals: List[float]) -> float:
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        cats: Dict[str, List[BenchmarkResult]] = {}
        for r in result_list:
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
            "n": len(result_list),
            "exact_match": _avg([r.exact_match for r in result_list]),
            "exact_match_norm": _avg([r.exact_match_norm for r in result_list]),
            "rouge_l": _avg([r.rouge_l for r in result_list]),
            "bleu_1": _avg([r.bleu_1 for r in result_list]),
            "f1": _avg([r.f1 for r in result_list]),
            "composite": _avg([r.composite_score for r in result_list]),
            "avg_latency_s": _avg([r.latency_s for r in result_list]),
            "errors": sum(1 for r in result_list if r.error),
        }
        return summary

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, path: Path, results: Optional[List[BenchmarkResult]] = None) -> None:
        """Write results to a CSV file."""
        result_list = results if results is not None else self.results
        if not result_list:
            return
        fieldnames = list(result_list[0].to_dict().keys()) + ["composite_score"]
        with Path(path).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in result_list:
                row = r.to_dict()
                row["composite_score"] = round(r.composite_score, 4)
                writer.writerow(row)

    def export_jsonl(self, path: Path, results: Optional[List[BenchmarkResult]] = None) -> None:
        """Write results to a JSONL file with SHA-256 provenance checksums."""
        result_list = results if results is not None else self.results
        if not result_list:
            return
        with Path(path).open("w", encoding="utf-8") as fh:
            for r in result_list:
                row = r.to_dict()
                row["composite_score"] = round(r.composite_score, 4)
                row["sha256"] = _provenance_checksum(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
