"""BenchmarkRunner: execute tasks against model functions or API endpoints."""
from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
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
from .models import BenchmarkResult, Task
from .spec import BenchmarkSpec


class BenchmarkRunner:
    """Execute a benchmark suite and collect scored results.

    Parameters
    ----------
    spec:
        A :class:`BenchmarkSpec` defining the tasks to run.  When *None* the
        built-in ten-task sample suite is used.
    tasks:
        Convenience alternative to *spec*: pass a list of :class:`Task` objects
        directly.  Ignored when *spec* is given.
    """

    def __init__(
        self,
        spec: Optional[BenchmarkSpec] = None,
        tasks: Optional[List[Task]] = None,
    ) -> None:
        if spec is not None:
            self.spec = spec
        elif tasks is not None:
            self.spec = BenchmarkSpec(name="custom", tasks=tasks)
        else:
            self.spec = BenchmarkSpec.builtin()
        self.results: List[BenchmarkResult] = []

    @property
    def tasks(self) -> List[Task]:
        return self.spec.tasks

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _score(
        self,
        task: Task,
        prediction: str,
        latency: float,
        error: Optional[str],
    ) -> BenchmarkResult:
        ts = datetime.now(tz=timezone.utc).isoformat()
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
            task_checksum=task.checksum(),
            run_timestamp=ts,
            error=error,
        )

    # ------------------------------------------------------------------
    # Run modes
    # ------------------------------------------------------------------

    def run_offline(
        self,
        model_fn: Callable[[str], str],
        tasks: Optional[List[Task]] = None,
    ) -> List[BenchmarkResult]:
        """Run *model_fn* on each task and return scored :class:`BenchmarkResult` objects.

        Parameters
        ----------
        model_fn:
            Callable that accepts a prompt string and returns a prediction string.
        tasks:
            Override the spec's task list for this run only.
        """
        run_tasks = tasks if tasks is not None else self.tasks
        results: List[BenchmarkResult] = []
        for task in run_tasks:
            t0 = time.monotonic()
            try:
                pred = model_fn(task.prompt)
                err = None
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
        model: str = "gpt-4o-mini",
        tasks: Optional[List[Task]] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        base_url: str = "https://api.openai.com/v1",
    ) -> List[BenchmarkResult]:
        """Run tasks against an OpenAI-compatible chat completions endpoint.

        Parameters
        ----------
        api_key:
            API bearer token.
        model:
            Model identifier string (e.g. ``"gpt-4o-mini"``).
        base_url:
            Override for OpenAI-compatible providers (e.g. a local vLLM server).
        """
        run_tasks = tasks if tasks is not None else self.tasks
        results: List[BenchmarkResult] = []
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        for task in run_tasks:
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
            ).encode("utf-8")
            t0 = time.monotonic()
            pred, err = "", None
            try:
                req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                pred = data["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as exc:
                err = f"HTTP {exc.code}: {exc.reason}"
            except urllib.error.URLError as exc:
                err = f"URLError: {exc.reason}"
            except (KeyError, IndexError) as exc:
                err = f"ResponseParseError: {exc}"
            except json.JSONDecodeError as exc:
                err = f"JSONDecodeError: {exc}"
            lat = time.monotonic() - t0
            results.append(self._score(task, pred, lat, err))
        self.results.extend(results)
        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summarize(self, results: Optional[List[BenchmarkResult]] = None) -> Dict[str, Any]:
        """Return overall and per-category aggregate statistics."""
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

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, path: Path, results: Optional[List[BenchmarkResult]] = None) -> None:
        """Write results to *path* in CSV format."""
        results = results if results is not None else self.results
        if not results:
            return
        path = Path(path)
        fieldnames = list(results[0].to_dict().keys())
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                w.writerow(r.to_dict())

    def export_jsonl(self, path: Path, results: Optional[List[BenchmarkResult]] = None) -> None:
        """Write results to *path* in JSONL format (one JSON object per line)."""
        results = results if results is not None else self.results
        path = Path(path)
        with path.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
