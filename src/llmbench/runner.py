"""BenchmarkRunner: orchestrates inference, scoring, and result export.

The runner accepts any callable model function (offline mode) or connects to
an OpenAI-compatible REST API.  All results are stored as
:class:`BenchmarkResult` dataclass instances and can be exported to CSV or
JSONL.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .metrics import (
    exact_match,
    exact_match_normalised,
    rouge_l,
    bleu_1,
    f1_score,
    _approx_tokens,
)
from .spec import Task, BenchmarkSpec, SAMPLE_TASKS


@dataclass
class BenchmarkResult:
    """Scored output for a single benchmark task.

    Attributes
    ----------
    task_id, category, prompt, reference:
        Copied from the originating :class:`~llmbench.spec.Task`.
    prediction:
        The model's raw output string.
    latency_s:
        Wall-clock seconds for the inference call.
    exact_match, exact_match_norm, rouge_l, bleu_1, f1:
        Individual metric scores in [0, 1].
    approx_tokens:
        Approximate token count of *prediction*.
    error:
        Exception message if inference failed; ``None`` otherwise.
    """

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

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of all fields plus composite_score."""
        d = asdict(self)
        d["composite_score"] = round(self.composite_score, 4)
        return d

    @property
    def composite_score(self) -> float:
        """Weighted composite: 0.40·ROUGE-L + 0.30·F1 + 0.20·EM-norm + 0.10·BLEU-1."""
        return (
            0.40 * self.rouge_l
            + 0.30 * self.f1
            + 0.20 * self.exact_match_norm
            + 0.10 * self.bleu_1
        )


class BenchmarkRunner:
    """Runs benchmark tasks and accumulates scored results.

    Parameters
    ----------
    tasks:
        A :class:`~llmbench.spec.BenchmarkSpec`, a plain list of
        :class:`~llmbench.spec.Task` objects, or *None* to use the built-in
        :data:`~llmbench.spec.SAMPLE_TASKS`.

    Examples
    --------
    >>> runner = BenchmarkRunner()
    >>> results = runner.run_offline(lambda p: "42")
    >>> print(runner.summarize()["overall"]["composite"])
    """

    def __init__(self, tasks: Optional[Any] = None) -> None:
        if isinstance(tasks, BenchmarkSpec):
            self.tasks: List[Task] = list(tasks.tasks)
        elif tasks is not None:
            self.tasks = list(tasks)
        else:
            self.tasks = list(SAMPLE_TASKS)
        self.results: List[BenchmarkResult] = []

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_offline(
        self,
        model_fn: Callable[[str], str],
        tasks: Optional[List[Task]] = None,
    ) -> List[BenchmarkResult]:
        """Run *model_fn* on each task and return scored results.

        Parameters
        ----------
        model_fn:
            A callable that accepts a prompt string and returns a prediction
            string.  Exceptions are caught and stored in
            :attr:`BenchmarkResult.error`.
        tasks:
            Override the runner's task list for this call.
        """
        task_list = tasks if tasks is not None else self.tasks
        batch: List[BenchmarkResult] = []
        for task in task_list:
            t0 = time.monotonic()
            try:
                pred = model_fn(task.prompt)
                error: Optional[str] = None
            except Exception as exc:
                pred = ""
                error = str(exc)
            lat = time.monotonic() - t0
            batch.append(self._score(task, pred, lat, error))
        self.results.extend(batch)
        return batch

    def run_openai(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        tasks: Optional[List[Task]] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        timeout: int = 30,
    ) -> List[BenchmarkResult]:
        """Query an OpenAI-compatible chat-completions endpoint.

        Parameters
        ----------
        api_key:
            API key passed in the ``Authorization`` header.
        model:
            Model identifier, e.g. ``"gpt-4o"``.
        tasks:
            Override the runner's task list for this call.
        max_tokens:
            Maximum tokens in the model response.
        temperature:
            Sampling temperature (0.0 = greedy).
        timeout:
            HTTP request timeout in seconds.
        """
        task_list = tasks if tasks is not None else self.tasks
        batch: List[BenchmarkResult] = []
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
            pred = ""
            error: Optional[str] = None
            try:
                req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read())
                pred = data["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as exc:
                error = f"HTTP {exc.code}: {exc.reason}"
            except Exception as exc:
                error = str(exc)
            lat = time.monotonic() - t0
            batch.append(self._score(task, pred, lat, error))
        self.results.extend(batch)
        return batch

    # ------------------------------------------------------------------
    # Scoring & summarisation
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
            approx_tokens=_approx_tokens(prediction),
            error=error,
        )

    def summarize(self, results: Optional[List[BenchmarkResult]] = None) -> Dict[str, Any]:
        """Return per-category and overall aggregate statistics.

        Returns
        -------
        dict
            Keys ``"overall"`` and ``"by_category"``, each containing counts,
            mean metric scores, mean latency, and error count.
        """
        results = results if results is not None else self.results
        if not results:
            return {}

        def _avg(vals: List[float]) -> float:
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        cats: Dict[str, List[BenchmarkResult]] = {}
        for r in results:
            cats.setdefault(r.category, []).append(r)

        def _cat_stats(items: List[BenchmarkResult]) -> Dict[str, Any]:
            return {
                "n": len(items),
                "exact_match": _avg([r.exact_match for r in items]),
                "exact_match_norm": _avg([r.exact_match_norm for r in items]),
                "rouge_l": _avg([r.rouge_l for r in items]),
                "bleu_1": _avg([r.bleu_1 for r in items]),
                "f1": _avg([r.f1 for r in items]),
                "composite": _avg([r.composite_score for r in items]),
                "avg_latency_s": _avg([r.latency_s for r in items]),
                "errors": sum(1 for r in items if r.error),
            }

        return {
            "overall": _cat_stats(results),
            "by_category": {cat: _cat_stats(items) for cat, items in cats.items()},
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, path: "Path | str", results: Optional[List[BenchmarkResult]] = None) -> None:
        """Write results to a CSV file.

        Parameters
        ----------
        path:
            Destination file path (created or overwritten).
        results:
            Results to export; defaults to all accumulated results.
        """
        results = results if results is not None else self.results
        if not results:
            return
        path = Path(path)
        fieldnames = list(results[0].to_dict().keys())
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_dict())

    def export_jsonl(self, path: "Path | str", results: Optional[List[BenchmarkResult]] = None) -> None:
        """Write results to a JSONL file (one JSON object per line).

        Parameters
        ----------
        path:
            Destination file path (created or overwritten).
        results:
            Results to export; defaults to all accumulated results.
        """
        results = results if results is not None else self.results
        if not results:
            return
        path = Path(path)
        with path.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
