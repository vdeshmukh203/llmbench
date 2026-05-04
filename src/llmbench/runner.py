"""BenchmarkRunner: orchestrates task execution and result scoring."""

from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.request
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
from .spec import BenchmarkResult, BenchmarkSpec, Task

ProgressCallback = Callable[[int, int, BenchmarkResult], None]


class BenchmarkRunner:
    """Executes benchmark tasks against a model and scores the outputs.

    Parameters
    ----------
    spec:
        Benchmark specification containing the task list.
        Defaults to the built-in sample tasks when *None*.
    """

    def __init__(self, spec: Optional[BenchmarkSpec] = None) -> None:
        self.spec = spec if spec is not None else BenchmarkSpec()
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
    # Runners
    # ------------------------------------------------------------------

    def run_offline(
        self,
        model_fn: Callable[[str], str],
        tasks: Optional[List[Task]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[BenchmarkResult]:
        """Run tasks through a local callable (no network required).

        Parameters
        ----------
        model_fn:
            A callable that accepts a prompt string and returns the model
            prediction string.  Exceptions are caught and recorded as errors.
        tasks:
            Override task list; defaults to ``self.spec.tasks``.
        progress_callback:
            Optional ``(done, total, result)`` callback invoked after each task.
        """
        task_list = tasks if tasks is not None else self.spec.tasks
        results: List[BenchmarkResult] = []
        for i, task in enumerate(task_list):
            t0 = time.monotonic()
            try:
                pred = model_fn(task.prompt)
                lat = time.monotonic() - t0
                err = None
            except Exception as exc:  # noqa: BLE001
                pred = ""
                lat = time.monotonic() - t0
                err = str(exc)
            result = self._score(task, pred, lat, err)
            results.append(result)
            if progress_callback is not None:
                progress_callback(i + 1, len(task_list), result)
        self.results.extend(results)
        return results

    def run_openai(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        tasks: Optional[List[Task]] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[BenchmarkResult]:
        """Run tasks against an OpenAI-compatible chat completions endpoint.

        Parameters
        ----------
        api_key:
            Bearer token for the API.
        model:
            Model identifier (e.g. ``"gpt-4o"``).
        tasks:
            Override task list; defaults to ``self.spec.tasks``.
        max_tokens:
            Maximum tokens in the model response.
        temperature:
            Sampling temperature (``0.0`` for deterministic output).
        progress_callback:
            Optional ``(done, total, result)`` callback invoked after each task.
        """
        task_list = tasks if tasks is not None else self.spec.tasks
        results: List[BenchmarkResult] = []
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        for i, task in enumerate(task_list):
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
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
                lat = time.monotonic() - t0
                pred = data["choices"][0]["message"]["content"].strip()
                err = None
            except Exception as exc:  # noqa: BLE001
                lat = time.monotonic() - t0
                pred = ""
                err = str(exc)
            result = self._score(task, pred, lat, err)
            results.append(result)
            if progress_callback is not None:
                progress_callback(i + 1, len(task_list), result)
        self.results.extend(results)
        return results

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def summarize(
        self, results: Optional[List[BenchmarkResult]] = None
    ) -> Dict[str, Any]:
        """Return aggregate metrics grouped by category and overall.

        Returns
        -------
        dict
            Keys ``"overall"`` and ``"by_category"``, each containing:
            ``n``, ``exact_match``, ``exact_match_norm``, ``rouge_l``,
            ``bleu_1``, ``f1``, ``composite``, ``avg_latency_s``, ``errors``.
        """
        result_list = results if results is not None else self.results
        if not result_list:
            return {}

        def _avg(vals: List[float]) -> float:
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        cats: Dict[str, List[BenchmarkResult]] = {}
        for r in result_list:
            cats.setdefault(r.category, []).append(r)

        def _cat_stats(cr: List[BenchmarkResult]) -> Dict[str, Any]:
            return {
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

        return {
            "overall": _cat_stats(result_list),
            "by_category": {cat: _cat_stats(cr) for cat, cr in cats.items()},
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(
        self,
        path: "str | Path",
        results: Optional[List[BenchmarkResult]] = None,
    ) -> None:
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

    def export_jsonl(
        self,
        path: "str | Path",
        results: Optional[List[BenchmarkResult]] = None,
    ) -> None:
        """Write results to a JSONL file (one JSON object per line)."""
        result_list = results if results is not None else self.results
        with Path(path).open("w", encoding="utf-8") as fh:
            for r in result_list:
                row = r.to_dict()
                row["composite_score"] = round(r.composite_score, 4)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
