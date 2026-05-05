#!/usr/bin/env python3
"""
llmbench.py — standalone LLM Benchmark Runner (single-file edition)
Evaluates language model outputs with ROUGE-L, BLEU-1, Exact Match, and F1.
Stdlib-only; no external dependencies required.
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
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if prediction.strip() == reference.strip() else 0.0


def exact_match_normalised(prediction: str, reference: str) -> float:
    return 1.0 if " ".join(_tokenise(prediction)) == " ".join(_tokenise(reference)) else 0.0


def rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = _tokenise(prediction)
    ref_tokens = _tokenise(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec = lcs / m
    rec = lcs / n
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def bleu_1(prediction: str, reference: str) -> float:
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
    bp = 1.0 if len(pred_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(pred_tokens))
    return bp * prec


def f1_score(prediction: str, reference: str) -> float:
    pred_set = set(_tokenise(prediction))
    ref_set = set(_tokenise(reference))
    if not pred_set or not ref_set:
        return 0.0
    common = pred_set & ref_set
    if not common:
        return 0.0
    prec = len(common) / len(pred_set)
    rec = len(common) / len(ref_set)
    return 2 * prec * rec / (prec + rec)


def _approx_tokens(text: str) -> int:
    return len(_tokenise(text))


def contains_code(text: str) -> bool:
    return bool(re.search(r"```|def |class |import |function |return |\{\}|;", text))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Task:
    task_id: str
    category: str
    prompt: str
    reference: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkResult:
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
        return (
            0.40 * self.rouge_l
            + 0.30 * self.f1
            + 0.20 * self.exact_match_norm
            + 0.10 * self.bleu_1
        )


# ---------------------------------------------------------------------------
# Sample tasks
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
        "Summarize in one sentence: The quick brown fox jumps over the lazy dog. The dog did not seem to mind.",
        "A fox jumped over a lazy dog.",
    ),
    Task(
        "summ_02", "summarization",
        "Summarize: Water boils at 100 degrees Celsius at sea level.",
        "Water boils at 100 degrees Celsius at sea level.",
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _provenance_checksum(data: dict) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


class BenchmarkRunner:
    def __init__(self, tasks: Optional[List[Task]] = None) -> None:
        self.tasks: List[Task] = list(tasks) if tasks is not None else list(SAMPLE_TASKS)
        self.results: List[BenchmarkResult] = []

    def _score(
        self, task: Task, prediction: str, latency: float, error: Optional[str]
    ) -> BenchmarkResult:
        return BenchmarkResult(
            task_id=task.task_id, category=task.category,
            prompt=task.prompt, reference=task.reference,
            prediction=prediction, latency_s=round(latency, 4),
            exact_match=exact_match(prediction, task.reference),
            exact_match_norm=exact_match_normalised(prediction, task.reference),
            rouge_l=rouge_l(prediction, task.reference),
            bleu_1=bleu_1(prediction, task.reference),
            f1=f1_score(prediction, task.reference),
            approx_tokens=_approx_tokens(prediction),
            error=error,
        )

    def run_offline(
        self,
        model_fn: Callable[[str], str],
        tasks: Optional[List[Task]] = None,
    ) -> List[BenchmarkResult]:
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
        task_list = tasks if tasks is not None else self.tasks
        results: List[BenchmarkResult] = []
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        for task in task_list:
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
            results.append(self._score(task, pred, time.monotonic() - t0, err))
        self.results.extend(results)
        return results

    def summarize(self, results: Optional[List[BenchmarkResult]] = None) -> Dict[str, Any]:
        result_list = results if results is not None else self.results
        if not result_list:
            return {}

        def _avg(vals: list) -> float:
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        cats: Dict[str, list] = {}
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

    def export_csv(self, path: Path, results: Optional[List[BenchmarkResult]] = None) -> None:
        result_list = results if results is not None else self.results
        if not result_list:
            return
        fieldnames = list(result_list[0].to_dict().keys()) + ["composite_score"]
        with Path(path).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in result_list:
                row = r.to_dict()
                row["composite_score"] = round(r.composite_score, 4)
                w.writerow(row)

    def export_jsonl(self, path: Path, results: Optional[List[BenchmarkResult]] = None) -> None:
        result_list = results if results is not None else self.results
        if not result_list:
            return
        with Path(path).open("w", encoding="utf-8") as fh:
            for r in result_list:
                row = r.to_dict()
                row["composite_score"] = round(r.composite_score, 4)
                row["sha256"] = _provenance_checksum(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="llmbench", description="Benchmark LLM outputs against references.")
    sub = p.add_subparsers(dest="command")
    run_p = sub.add_parser("run", help="Run benchmark against an OpenAI-compatible API.")
    run_p.add_argument("--api-key", default=None)
    run_p.add_argument("--model", default="gpt-3.5-turbo")
    run_p.add_argument("--output", "-o", default="results.jsonl")
    run_p.add_argument("--csv", default=None)
    run_p.add_argument("--max-tokens", type=int, default=256)
    run_p.add_argument("--temperature", type=float, default=0.0)
    score_p = sub.add_parser("score", help="Score predictions against references.")
    score_p.add_argument("predictions")
    score_p.add_argument("--format", choices=["json", "text"], default="text")
    sub.add_parser("sample", help="Print built-in sample tasks.")
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
        results = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pred = obj.get("prediction", "")
                ref = obj.get("reference", "")
                task = Task(
                    task_id=obj.get("task_id", "?"),
                    category=obj.get("category", "unknown"),
                    prompt=obj.get("prompt", ""),
                    reference=ref,
                )
                runner = BenchmarkRunner([task])
                results.extend(runner.run_offline(lambda _: pred, [task]))
        reporter = BenchmarkRunner()
        reporter.results = results
        summary = reporter.summarize()
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
        runner = BenchmarkRunner()
        runner.run_openai(
            api_key=api_key, model=args.model,
            max_tokens=args.max_tokens, temperature=args.temperature,
        )
        out_path = Path(args.output)
        runner.export_jsonl(out_path)
        print(f"Results written to {out_path}", file=sys.stderr)
        if args.csv:
            runner.export_csv(Path(args.csv))
        print(json.dumps(runner.summarize(), indent=2))
        return 0

    # Demo mode
    runner = BenchmarkRunner()
    runner.run_offline(lambda p: p.split("?")[0].strip() if "?" in p else p[:40])
    print(json.dumps(runner.summarize(), indent=2))
    return 0


# Backwards-compatible entry-point alias
_cli = main


if __name__ == "__main__":
    sys.exit(main())
