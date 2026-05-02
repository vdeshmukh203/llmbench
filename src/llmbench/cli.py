"""Command-line interface for llmbench."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .models import Task
from .runner import BenchmarkRunner
from .spec import BenchmarkSpec


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="llmbench",
        description="Benchmark LLM outputs against reference answers.",
    )
    sub = p.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run benchmark against an OpenAI-compatible API.")
    run_p.add_argument("--api-key", default=None, help="API bearer token (or set OPENAI_API_KEY).")
    run_p.add_argument("--model", default="gpt-4o-mini", help="Model identifier.")
    run_p.add_argument("--tasks", default=None, help="Path to a JSON or YAML task file.")
    run_p.add_argument("--output", "-o", default="results.jsonl", help="JSONL output path.")
    run_p.add_argument("--csv", default=None, help="Optional CSV output path.")
    run_p.add_argument(
        "--base-url",
        default="https://api.openai.com/v1",
        help="API base URL for OpenAI-compatible providers.",
    )

    score_p = sub.add_parser("score", help="Score saved predictions against references.")
    score_p.add_argument("predictions", help="JSONL file of predictions.")
    score_p.add_argument("--format", choices=["json", "text"], default="text")

    sub.add_parser("sample", help="Print built-in sample tasks as JSONL.")
    sub.add_parser("gui", help="Launch the graphical interface.")

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    if args.command == "sample":
        spec = BenchmarkSpec.builtin()
        for t in spec.tasks:
            print(json.dumps(t.to_dict(), ensure_ascii=False))
        return 0

    if args.command == "score":
        path = Path(args.predictions)
        if not path.is_file():
            print(f"Error: {path} not found", file=sys.stderr)
            return 1
        tasks: list[Task] = []
        preds: list[str] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                tasks.append(
                    Task(
                        task_id=obj.get("task_id", "?"),
                        category=obj.get("category", "unknown"),
                        prompt=obj.get("prompt", ""),
                        reference=obj.get("reference", ""),
                    )
                )
                preds.append(obj.get("prediction", ""))
        runner = BenchmarkRunner(tasks=tasks)
        # Use a default arg to capture each pred value in the closure correctly.
        for task, pred in zip(tasks, preds):
            runner.run_offline(lambda _, p=pred: p, tasks=[task])
        summary = runner.summarize()
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
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("Error: provide --api-key or set OPENAI_API_KEY", file=sys.stderr)
            return 1
        spec = BenchmarkSpec.from_file(Path(args.tasks)) if args.tasks else BenchmarkSpec.builtin()
        runner = BenchmarkRunner(spec=spec)
        runner.run_openai(api_key=api_key, model=args.model, base_url=args.base_url)
        runner.export_jsonl(Path(args.output))
        if args.csv:
            runner.export_csv(Path(args.csv))
        print(json.dumps(runner.summarize(), indent=2))
        return 0

    if args.command == "gui":
        from .gui import launch_gui

        launch_gui()
        return 0

    # No subcommand: demo run with echo model
    runner = BenchmarkRunner()
    runner.run_offline(lambda p: p.split("?")[0].strip() if "?" in p else p[:40])
    print(json.dumps(runner.summarize(), indent=2))
    return 0
