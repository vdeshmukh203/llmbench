#!/usr/bin/env python3
"""
llmbench — Lightweight reproducible LLM benchmarking framework.

Evaluates language model outputs with ROUGE-L, BLEU-1, token-level F1,
and exact match.  Core is stdlib-only; PyYAML is optional for YAML task
files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
import urllib.request
import urllib.error
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# Tokenisation helper
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    """Lowercase alphanumeric tokenisation."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def exact_match(prediction: str, reference: str) -> float:
    """Return 1.0 when *prediction* and *reference* are identical strings."""
    return 1.0 if prediction.strip() == reference.strip() else 0.0


def exact_match_normalised(prediction: str, reference: str) -> float:
    """Return 1.0 when predictions match after lowercasing and tokenising."""
    return 1.0 if " ".join(_tokenise(prediction)) == " ".join(_tokenise(reference)) else 0.0


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 based on the longest common subsequence of tokens."""
    pred_tokens = _tokenise(prediction)
    ref_tokens = _tokenise(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (
                dp[i - 1][j - 1] + 1
                if pred_tokens[i - 1] == ref_tokens[j - 1]
                else max(dp[i - 1][j], dp[i][j - 1])
            )
    lcs = dp[m][n]
    prec = lcs / m
    rec = lcs / n
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def bleu_1(prediction: str, reference: str) -> float:
    """Corpus-free BLEU-1: unigram precision with brevity penalty."""
    pred_tokens = _tokenise(prediction)
    ref_tokens = _tokenise(reference)
    if not pred_tokens:
        return 0.0
    ref_counts: Dict[str, int] = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    pred_counts: Dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    clipped = sum(min(c, ref_counts.get(t, 0)) for t, c in pred_counts.items())
    prec = clipped / len(pred_tokens)
    bp = (
        1.0
        if len(pred_tokens) >= len(ref_tokens)
        else math.exp(1 - len(ref_tokens) / len(pred_tokens))
    )
    return bp * prec


def f1_score(prediction: str, reference: str) -> float:
    """Token-level F1 using frequency-weighted intersection (SQuAD style)."""
    pred_tokens = _tokenise(prediction)
    ref_tokens = _tokenise(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts: Dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    ref_counts: Dict[str, int] = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    common = sum(min(pred_counts.get(t, 0), ref_counts.get(t, 0)) for t in ref_counts)
    if common == 0:
        return 0.0
    prec = common / len(pred_tokens)
    rec = common / len(ref_tokens)
    return 2 * prec * rec / (prec + rec)


def _approx_tokens(text: str) -> int:
    """Approximate token count via alphanumeric splitting."""
    return len(_tokenise(text))


def contains_code(text: str) -> bool:
    """Heuristically detect whether *text* contains a code snippet."""
    return bool(re.search(r"```|def |class |import |function |return |\{\}|;", text))


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A single benchmark task with a prompt and reference answer."""

    task_id: str
    category: str
    prompt: str
    reference: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Scores and provenance metadata for a single task evaluation."""

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
    model: str = ""
    timestamp: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def composite_score(self) -> float:
        """Weighted composite: 0.4·ROUGE-L + 0.3·F1 + 0.2·EM-norm + 0.1·BLEU-1."""
        return (
            0.40 * self.rouge_l
            + 0.30 * self.f1
            + 0.20 * self.exact_match_norm
            + 0.10 * self.bleu_1
        )


@dataclass
class BenchmarkSpec:
    """Declarative specification for a benchmark run."""

    tasks: List[Task]
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 256
    temperature: float = 0.0
    seed: Optional[int] = None
    base_url: str = "https://api.openai.com/v1/chat/completions"


# ---------------------------------------------------------------------------
# Built-in sample tasks
# ---------------------------------------------------------------------------

SAMPLE_TASKS: List[Task] = [
    Task("qa_01", "qa", "What is the capital of France?", "Paris"),
    Task("qa_02", "qa", "What year did World War II end?", "1945"),
    Task("qa_03", "qa", "What is the chemical symbol for water?", "H2O"),
    Task("qa_04", "qa", "Who wrote Pride and Prejudice?", "Jane Austen"),
    Task("qa_05", "qa", "What is the square root of 144?", "12"),
    Task(
        "code_01", "coding",
        "Write a Python function that returns the factorial of n.",
        "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)",
    ),
    Task("code_02", "coding", "Write a Python one-liner to reverse a list called lst.", "lst[::-1]"),
    Task(
        "code_03", "coding",
        "Write a Python function to check if a string is a palindrome.",
        "def is_palindrome(s):\n    return s == s[::-1]",
    ),
    Task(
        "summ_01", "summarization",
        "Summarize in one sentence: The quick brown fox jumps over the lazy dog. "
        "The dog did not seem to mind.",
        "A fox jumped over a lazy dog.",
    ),
    Task(
        "summ_02", "summarization",
        "Summarize: Water boils at 100 degrees Celsius at sea level.",
        "Water boils at 100 degrees Celsius at sea level.",
    ),
]


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def load_tasks_from_file(path: "str | Path") -> List[Task]:
    """Load tasks from a JSON array, JSONL, or YAML file.

    Each record must supply ``task_id``, ``category``, ``prompt``, and
    ``reference``.  An optional ``metadata`` dict is also recognised.

    YAML files require PyYAML (``pip install pyyaml``).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    with path.open(encoding="utf-8") as fh:
        if suffix in (".yaml", ".yml"):
            if not _HAS_YAML:
                raise ImportError(
                    "PyYAML is required for YAML task files.  "
                    "Install it with: pip install pyyaml"
                )
            records = _yaml.safe_load(fh)
        elif suffix == ".json":
            records = json.load(fh)
        else:
            # JSONL or extension-less
            records = [json.loads(line) for line in fh if line.strip()]
    return [
        Task(
            task_id=r["task_id"],
            category=r["category"],
            prompt=r["prompt"],
            reference=r["reference"],
            metadata=r.get("metadata", {}),
        )
        for r in records
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Orchestrates benchmark runs, aggregates results, and exports reports."""

    def __init__(self, tasks: Optional[List[Task]] = None) -> None:
        self.tasks: List[Task] = tasks if tasks is not None else list(SAMPLE_TASKS)
        self.results: List[BenchmarkResult] = []
        self._run_metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score(
        self,
        task: Task,
        prediction: str,
        latency: float,
        error: Optional[str],
        model: str = "",
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
            model=model,
            timestamp=datetime.now(timezone.utc).isoformat(),
            error=error,
        )

    # ------------------------------------------------------------------
    # Inference backends
    # ------------------------------------------------------------------

    def run_offline(
        self,
        model_fn: Callable[[str], str],
        tasks: Optional[List[Task]] = None,
    ) -> List[BenchmarkResult]:
        """Evaluate *model_fn(prompt) → prediction* on each task.

        Exceptions raised by *model_fn* are caught and recorded in the
        result's ``error`` field rather than propagating.
        """
        tasks = tasks if tasks is not None else self.tasks
        results: List[BenchmarkResult] = []
        for task in tasks:
            t0 = time.monotonic()
            try:
                pred = model_fn(task.prompt)
                lat = time.monotonic() - t0
                err = None
            except Exception as exc:
                pred = ""
                lat = time.monotonic() - t0
                err = str(exc)
            results.append(self._score(task, pred, lat, err, model="offline"))
        self.results.extend(results)
        return results

    def run_openai(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        tasks: Optional[List[Task]] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        base_url: str = "https://api.openai.com/v1/chat/completions",
    ) -> List[BenchmarkResult]:
        """Evaluate tasks via any OpenAI-compatible chat completions endpoint.

        Parameters
        ----------
        api_key:
            Bearer token for the API.
        model:
            Model identifier understood by the endpoint.
        tasks:
            Tasks to evaluate; defaults to ``self.tasks``.
        max_tokens:
            Upper bound on completion length.
        temperature:
            Sampling temperature (0.0 = greedy).
        seed:
            Integer seed forwarded to the API for deterministic outputs
            (supported by OpenAI ``gpt-*`` models and some open-weight
            servers).
        base_url:
            Full chat completions URL.  Override to target local servers
            such as Ollama or vLLM.
        """
        tasks = tasks if tasks is not None else self.tasks
        run_id = str(uuid.uuid4())
        self._run_metadata = {
            "run_id": run_id,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "base_url": base_url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        results: List[BenchmarkResult] = []
        for task in tasks:
            body: Dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Answer concisely."},
                    {"role": "user", "content": task.prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if seed is not None:
                body["seed"] = seed
            payload = json.dumps(body).encode()
            t0 = time.monotonic()
            try:
                req = urllib.request.Request(
                    base_url, data=payload, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
                lat = time.monotonic() - t0
                pred = data["choices"][0]["message"]["content"].strip()
                err = None
            except Exception as exc:
                lat = time.monotonic() - t0
                pred = ""
                err = str(exc)
            results.append(self._score(task, pred, lat, err, model=model))
        self.results.extend(results)
        return results

    # ------------------------------------------------------------------
    # Reporting
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
        if self._run_metadata:
            summary["run_metadata"] = self._run_metadata
        return summary

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(
        self,
        path: "str | Path",
        results: Optional[List[BenchmarkResult]] = None,
    ) -> None:
        """Write results to *path* as CSV."""
        results = results if results is not None else self.results
        if not results:
            return
        fieldnames = list(results[0].to_dict().keys()) + ["composite_score"]
        with Path(path).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                row = r.to_dict()
                row["composite_score"] = round(r.composite_score, 4)
                w.writerow(row)

    def export_jsonl(
        self,
        path: "str | Path",
        results: Optional[List[BenchmarkResult]] = None,
    ) -> None:
        """Write results to *path* as JSONL.

        A SHA-256 sidecar file ``<path>.sha256`` is written alongside the
        JSONL so consumers can verify the result file has not been modified.
        """
        results = results if results is not None else self.results
        path = Path(path)
        lines: List[str] = []
        with path.open("w", encoding="utf-8") as fh:
            for r in results:
                row = r.to_dict()
                row["composite_score"] = round(r.composite_score, 4)
                line = json.dumps(row, ensure_ascii=False)
                lines.append(line)
                fh.write(line + "\n")
        digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
        path.with_suffix(".sha256").write_text(
            f"{digest}  {path.name}\n", encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="llmbench",
        description="Benchmark LLM outputs against reference answers.",
    )
    sub = p.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run benchmark against an OpenAI-compatible API.")
    run_p.add_argument("--api-key", default=None, help="API bearer token (or set OPENAI_API_KEY).")
    run_p.add_argument("--model", default="gpt-3.5-turbo", help="Model identifier.")
    run_p.add_argument(
        "--base-url",
        default="https://api.openai.com/v1/chat/completions",
        help="Chat completions endpoint URL.",
    )
    run_p.add_argument("--tasks", default=None, help="Path to task file (JSON/JSONL/YAML).")
    run_p.add_argument("--output", "-o", default="results.jsonl", help="JSONL output path.")
    run_p.add_argument("--csv", default=None, help="Also export results as CSV.")
    run_p.add_argument("--max-tokens", type=int, default=256)
    run_p.add_argument("--temperature", type=float, default=0.0)
    run_p.add_argument("--seed", type=int, default=None, help="Seed for deterministic outputs.")

    score_p = sub.add_parser("score", help="Score a JSONL file of predictions.")
    score_p.add_argument("predictions", help="Path to predictions JSONL file.")
    score_p.add_argument("--format", choices=["json", "text"], default="text")

    sub.add_parser("sample", help="Print built-in sample tasks as JSONL.")

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    if args.command == "sample":
        for t in SAMPLE_TASKS:
            print(json.dumps(t.to_dict(), ensure_ascii=False))
        return 0

    if args.command == "score":
        path = Path(args.predictions)
        if not path.is_file():
            print(f"Error: {path} not found", file=sys.stderr)
            return 1
        results: List[BenchmarkResult] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pred = obj.get("prediction", "")
                t = Task(
                    task_id=obj.get("task_id", "?"),
                    category=obj.get("category", "unknown"),
                    prompt=obj.get("prompt", ""),
                    reference=obj.get("reference", ""),
                )
                captured = pred  # capture per-iteration value for closure
                runner = BenchmarkRunner([t])
                results.extend(runner.run_offline(lambda _: captured, [t]))
        scorer = BenchmarkRunner()
        scorer.results = results
        summary = scorer.summarize()
        ov = summary.get("overall", {})
        if args.format == "json":
            print(json.dumps(summary, indent=2))
        else:
            print(
                f"Tasks: {ov.get('n', 0)}  "
                f"EM: {ov.get('exact_match', 0):.4f}  "
                f"ROUGE-L: {ov.get('rouge_l', 0):.4f}  "
                f"F1: {ov.get('f1', 0):.4f}  "
                f"Composite: {ov.get('composite', 0):.4f}"
            )
        return 0

    if args.command == "run":
        import os
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("Error: provide --api-key or set OPENAI_API_KEY", file=sys.stderr)
            return 1
        tasks = load_tasks_from_file(Path(args.tasks)) if args.tasks else None
        runner = BenchmarkRunner(tasks)
        runner.run_openai(
            api_key=api_key,
            model=args.model,
            base_url=args.base_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
        out_path = Path(args.output)
        runner.export_jsonl(out_path)
        if args.csv:
            runner.export_csv(Path(args.csv))
        print(json.dumps(runner.summarize(), indent=2))
        return 0

    # Demo mode — no subcommand
    runner = BenchmarkRunner()
    runner.run_offline(lambda p: p.split("?")[0].strip() if "?" in p else p[:40])
    print(json.dumps(runner.summarize(), indent=2))
    return 0


# Entry-point alias used by pyproject.toml
_cli = main

if __name__ == "__main__":
    sys.exit(main())
