"""Command-line interface for llmbench.

Sub-commands
------------
``sample``
    Print built-in sample tasks as JSONL.
``score``
    Score a JSONL predictions file and print aggregate metrics.
``run``
    Query an OpenAI-compatible API and write results to JSONL.
``gui``
    Launch the optional Tkinter GUI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from .spec import SAMPLE_TASKS, Task, BenchmarkSpec
from .runner import BenchmarkRunner


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="llmbench",
        description="Reproducible LLM benchmarking framework.",
    )
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    # --- sample ---
    sub.add_parser("sample", help="Print built-in sample tasks as JSONL.")

    # --- score ---
    score_p = sub.add_parser("score", help="Score a JSONL predictions file.")
    score_p.add_argument("predictions", help="Path to JSONL file with prediction/reference pairs.")
    score_p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )

    # --- run ---
    run_p = sub.add_parser("run", help="Run benchmark against an OpenAI-compatible API.")
    run_p.add_argument("--api-key", default=None, help="API key (or set OPENAI_API_KEY).")
    run_p.add_argument("--model", default="gpt-3.5-turbo", help="Model identifier.")
    run_p.add_argument("--tasks", default=None, help="Path to task spec file (JSON/JSONL/YAML).")
    run_p.add_argument(
        "--output", "-o", default="results.jsonl", help="Output JSONL file (default: results.jsonl)."
    )
    run_p.add_argument("--csv", default=None, help="Also write a CSV file at this path.")
    run_p.add_argument("--max-tokens", type=int, default=256)
    run_p.add_argument("--temperature", type=float, default=0.0)

    # --- gui ---
    sub.add_parser("gui", help="Launch the Tkinter GUI.")

    return p.parse_args(argv)


def _cmd_sample() -> int:
    for t in SAMPLE_TASKS:
        print(json.dumps(t.to_dict(), ensure_ascii=False))
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    path = Path(args.predictions)
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    results = []
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping malformed line {lineno}: {exc}", file=sys.stderr)
                continue
            task = Task(
                task_id=obj.get("task_id", f"row_{lineno}"),
                category=obj.get("category", "unknown"),
                prompt=obj.get("prompt", ""),
                reference=obj.get("reference", ""),
            )
            pred = obj.get("prediction", "")
            runner = BenchmarkRunner([task])
            results.extend(runner.run_offline(lambda _p, _pred=pred: _pred, [task]))

    if not results:
        print("No results scored.", file=sys.stderr)
        return 1

    aggregate = BenchmarkRunner()
    aggregate.results = results
    summary = aggregate.summarize()
    ov = summary.get("overall", {})

    if args.format == "json":
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"Tasks: {ov.get('n', 0):>4}  "
            f"EM: {ov.get('exact_match', 0):.4f}  "
            f"EM-norm: {ov.get('exact_match_norm', 0):.4f}  "
            f"ROUGE-L: {ov.get('rouge_l', 0):.4f}  "
            f"BLEU-1: {ov.get('bleu_1', 0):.4f}  "
            f"F1: {ov.get('f1', 0):.4f}  "
            f"Composite: {ov.get('composite', 0):.4f}"
        )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: provide --api-key or set OPENAI_API_KEY", file=sys.stderr)
        return 1

    spec = BenchmarkSpec.from_file(args.tasks) if args.tasks else BenchmarkSpec()
    runner = BenchmarkRunner(spec)
    results = runner.run_openai(
        api_key=api_key,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    runner.export_jsonl(args.output, results)
    print(f"Results written to {args.output}", file=sys.stderr)

    if args.csv:
        runner.export_csv(args.csv, results)
        print(f"CSV written to {args.csv}", file=sys.stderr)

    print(json.dumps(runner.summarize(), indent=2))
    return 0


def _cmd_gui() -> int:
    try:
        from .gui import launch
    except ImportError as exc:
        print(f"GUI unavailable: {exc}", file=sys.stderr)
        return 1
    launch()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the ``llmbench`` command-line tool."""
    args = _parse_args(argv)

    if args.command == "sample":
        return _cmd_sample()
    if args.command == "score":
        return _cmd_score(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "gui":
        return _cmd_gui()

    # No sub-command: run demo mode and print summary
    runner = BenchmarkRunner()
    runner.run_offline(lambda p: p.split("?")[0].strip() if "?" in p else p[:40])
    print(json.dumps(runner.summarize(), indent=2))
    return 0
