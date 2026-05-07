"""Command-line interface for llmbench."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .models import SAMPLE_TASKS, Task
from .runner import BenchmarkRunner
from .spec import BenchmarkSpec


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="llmbench",
        description="Benchmark LLM outputs against references.",
    )
    sub = p.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run benchmark against an OpenAI-compatible API.")
    run_p.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY).")
    run_p.add_argument("--model", default="gpt-3.5-turbo", help="Model name.")
    run_p.add_argument("--tasks", default=None, help="Path to tasks file (JSON or JSONL).")
    run_p.add_argument("--output", "-o", default="results.jsonl", help="Output JSONL path.")
    run_p.add_argument("--csv", default=None, help="Also export results as CSV.")
    run_p.add_argument("--max-tokens", type=int, default=256)
    run_p.add_argument("--temperature", type=float, default=0.0)

    score_p = sub.add_parser("score", help="Score predictions against references from a JSONL file.")
    score_p.add_argument("predictions", help="JSONL file with prediction and reference fields.")
    score_p.add_argument("--format", choices=["json", "text"], default="text")

    sub.add_parser("sample", help="Print built-in sample tasks as JSONL.")
    sub.add_parser("gui", help="Launch the graphical user interface.")

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    if args.command == "sample":
        for t in SAMPLE_TASKS:
            print(json.dumps(t.to_dict(), ensure_ascii=False))
        return 0

    if args.command == "gui":
        from .gui import launch_gui
        launch_gui()
        return 0

    if args.command == "score":
        path = Path(args.predictions)
        if not path.is_file():
            print(f"Error: {path} not found", file=sys.stderr)
            return 1
        runner = BenchmarkRunner([])
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pred = obj.get("prediction", "")
                task = Task(
                    task_id=obj.get("task_id", "unknown"),
                    category=obj.get("category", "unknown"),
                    prompt=obj.get("prompt", ""),
                    reference=obj.get("reference", ""),
                )
                # Use default-arg capture to avoid late-binding closure over pred
                runner.run_offline(lambda _p, _pred=pred: _pred, [task])
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
        tasks = None
        if args.tasks:
            tp = Path(args.tasks)
            spec = BenchmarkSpec.from_json(tp) if tp.suffix == ".json" else BenchmarkSpec.from_jsonl(tp)
            tasks = spec.tasks
        runner = BenchmarkRunner(tasks)
        results = runner.run_openai(
            api_key=api_key,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        out_path = Path(args.output)
        runner.export_jsonl(out_path, results)
        if args.csv:
            runner.export_csv(Path(args.csv), results)
        print(json.dumps(runner.summarize(results), indent=2))
        return 0

    # Demo mode (no subcommand given)
    runner = BenchmarkRunner()
    runner.run_offline(lambda p: p.split("?")[0].strip() if "?" in p else p[:40])
    print(json.dumps(runner.summarize(), indent=2))
    return 0
