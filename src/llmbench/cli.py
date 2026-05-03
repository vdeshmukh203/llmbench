"""Command-line interface for llmbench."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from .runner import BenchmarkRunner
from .spec import BenchmarkSpec, Task


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="llmbench",
        description="Benchmark LLM outputs against reference answers.",
    )
    sub = p.add_subparsers(dest="command")

    # ── run ──────────────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Run benchmark against an OpenAI-compatible API.")
    run_p.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help="API key (overrides OPENAI_API_KEY env var).",
    )
    run_p.add_argument("--model", default="gpt-3.5-turbo", metavar="NAME")
    run_p.add_argument(
        "--tasks",
        default=None,
        metavar="FILE",
        help="JSONL task file. Defaults to the built-in sample tasks.",
    )
    run_p.add_argument("--output", "-o", default="results.jsonl", metavar="FILE")
    run_p.add_argument("--csv", default=None, metavar="FILE")
    run_p.add_argument("--max-tokens", type=int, default=256, metavar="N")
    run_p.add_argument("--temperature", type=float, default=0.0, metavar="T")

    # ── score ─────────────────────────────────────────────────────────────
    score_p = sub.add_parser(
        "score",
        help="Score pre-computed predictions stored in a JSONL file.",
    )
    score_p.add_argument(
        "predictions",
        metavar="FILE",
        help="JSONL file with 'prediction' and 'reference' fields per line.",
    )
    score_p.add_argument("--format", choices=["json", "text"], default="text")

    # ── sample ────────────────────────────────────────────────────────────
    sub.add_parser("sample", help="Print built-in sample tasks as JSONL.")

    # ── gui ───────────────────────────────────────────────────────────────
    sub.add_parser("gui", help="Launch the graphical benchmark interface.")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    # ── sample ────────────────────────────────────────────────────────────
    if args.command == "sample":
        spec = BenchmarkSpec.from_sample()
        for task in spec.tasks:
            print(json.dumps(task.to_dict(), ensure_ascii=False))
        return 0

    # ── gui ───────────────────────────────────────────────────────────────
    if args.command == "gui":
        try:
            from .gui import main as gui_main  # noqa: PLC0415
        except ImportError as exc:
            print(f"GUI unavailable: {exc}", file=sys.stderr)
            return 1
        gui_main()
        return 0

    # ── score ─────────────────────────────────────────────────────────────
    if args.command == "score":
        path = Path(args.predictions)
        if not path.is_file():
            print(f"Error: {path} not found.", file=sys.stderr)
            return 1
        runner = BenchmarkRunner([])
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"Warning: skipping invalid JSON on line {lineno}: {exc}", file=sys.stderr)
                    continue
                pred = obj.get("prediction", "")
                task = Task(
                    task_id=obj.get("task_id", f"task_{lineno}"),
                    category=obj.get("category", "unknown"),
                    prompt=obj.get("prompt", ""),
                    reference=obj.get("reference", ""),
                )
                # Default arg captures current pred value (avoids closure pitfall).
                runner.run_offline(lambda _p, _pred=pred: _pred, [task])
        summary = runner.summarize()
        ov = summary.get("overall", {})
        if args.format == "json":
            print(json.dumps(summary, indent=2))
        else:
            print(
                f"Tasks: {ov.get('n', 0)}"
                f"  EM: {ov.get('exact_match', 0):.4f}"
                f"  ROUGE-L: {ov.get('rouge_l', 0):.4f}"
                f"  F1: {ov.get('f1', 0):.4f}"
                f"  Composite: {ov.get('composite', 0):.4f}"
            )
        return 0

    # ── run ───────────────────────────────────────────────────────────────
    if args.command == "run":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("Error: provide --api-key or set OPENAI_API_KEY.", file=sys.stderr)
            return 1
        if args.tasks:
            spec = BenchmarkSpec.from_jsonl(Path(args.tasks))
        else:
            spec = BenchmarkSpec.from_sample()
        runner = BenchmarkRunner(spec.tasks)
        runner.run_openai(
            api_key=api_key,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        runner.export_jsonl(Path(args.output))
        if args.csv:
            runner.export_csv(Path(args.csv))
        print(json.dumps(runner.summarize(), indent=2))
        return 0

    # ── demo (no subcommand) ──────────────────────────────────────────────
    runner = BenchmarkRunner()
    runner.run_offline(
        lambda p: p.split("?")[0].strip() if "?" in p else p[:40]
    )
    print(json.dumps(runner.summarize(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
