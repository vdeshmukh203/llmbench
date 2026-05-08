"""Tests for llmbench package."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Allow running without installation from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import llmbench as lb
from llmbench.metrics import (
    _approx_tokens,
    _tokenise,
    bleu_1,
    contains_code,
    exact_match,
    exact_match_normalised,
    f1_score,
    rouge_l,
)
from llmbench.runner import BenchmarkResult, BenchmarkRunner
from llmbench.spec import SAMPLE_TASKS, BenchmarkSpec, Task


# ── Public API surface ────────────────────────────────────────────────────────

def test_package_exports():
    for name in ["BenchmarkRunner", "BenchmarkResult", "BenchmarkSpec", "Task",
                 "SAMPLE_TASKS", "exact_match", "rouge_l", "bleu_1", "f1_score"]:
        assert hasattr(lb, name), f"lb.{name} missing"


# ── Tokeniser ─────────────────────────────────────────────────────────────────

def test_tokenise_basic():
    assert _tokenise("Hello, World!") == ["hello", "world"]


def test_tokenise_numbers():
    assert "1945" in _tokenise("Year 1945")


def test_tokenise_empty():
    assert _tokenise("") == []


# ── exact_match ───────────────────────────────────────────────────────────────

def test_exact_match_equal():
    assert exact_match("hello", "hello") == 1.0


def test_exact_match_strips_whitespace():
    assert exact_match("  hello  ", "hello") == 1.0


def test_exact_match_different():
    assert exact_match("hello", "world") == 0.0


def test_exact_match_case_sensitive():
    assert exact_match("Paris", "paris") == 0.0


# ── exact_match_normalised ────────────────────────────────────────────────────

def test_exact_match_norm_case_insensitive():
    assert exact_match_normalised("Hello World", "hello world") == 1.0


def test_exact_match_norm_punctuation():
    assert exact_match_normalised("Jane Austen.", "Jane Austen") == 1.0


def test_exact_match_norm_different():
    assert exact_match_normalised("foo bar", "foo baz") == 0.0


# ── rouge_l ───────────────────────────────────────────────────────────────────

def test_rouge_l_perfect():
    assert rouge_l("the cat sat", "the cat sat") == 1.0


def test_rouge_l_empty_pred():
    assert rouge_l("", "reference") == 0.0


def test_rouge_l_empty_ref():
    assert rouge_l("prediction", "") == 0.0


def test_rouge_l_partial():
    score = rouge_l("the cat sat on the mat", "the cat sat")
    assert 0.0 < score < 1.0


def test_rouge_l_no_overlap():
    assert rouge_l("alpha beta", "gamma delta") == 0.0


# ── bleu_1 ────────────────────────────────────────────────────────────────────

def test_bleu_1_perfect():
    assert bleu_1("the cat sat", "the cat sat") == 1.0


def test_bleu_1_empty_pred():
    assert bleu_1("", "reference") == 0.0


def test_bleu_1_brevity_penalty():
    # Short prediction gets penalised.
    score_short = bleu_1("cat", "the cat sat on the mat")
    score_full = bleu_1("the cat sat on the mat", "the cat sat on the mat")
    assert score_short < score_full


def test_bleu_1_partial():
    score = bleu_1("the cat sat on the mat", "the cat sat")
    assert 0.0 < score <= 1.0


# ── f1_score ──────────────────────────────────────────────────────────────────

def test_f1_perfect():
    assert f1_score("the cat sat", "the cat sat") == 1.0


def test_f1_empty_pred():
    assert f1_score("", "reference") == 0.0


def test_f1_empty_ref():
    assert f1_score("prediction", "") == 0.0


def test_f1_no_overlap():
    assert f1_score("hello", "world") == 0.0


def test_f1_partial():
    score = f1_score("the cat sat on the mat", "the cat sat")
    assert 0.0 < score <= 1.0


# ── _approx_tokens ────────────────────────────────────────────────────────────

def test_approx_tokens_count():
    assert _approx_tokens("hello world") == 2


def test_approx_tokens_empty():
    assert _approx_tokens("") == 0


# ── contains_code ─────────────────────────────────────────────────────────────

def test_contains_code_python():
    assert contains_code("def foo():\n    return 1")


def test_contains_code_backticks():
    assert contains_code("```python\nprint('hi')\n```")


def test_contains_code_plain_text():
    assert not contains_code("The answer is forty-two.")


# ── Task ──────────────────────────────────────────────────────────────────────

def test_task_to_dict():
    task = Task("t1", "qa", "prompt?", "answer")
    d = task.to_dict()
    assert d == {
        "task_id": "t1",
        "category": "qa",
        "prompt": "prompt?",
        "reference": "answer",
        "metadata": {},
    }


# ── BenchmarkSpec ─────────────────────────────────────────────────────────────

def test_benchmark_spec_defaults():
    spec = BenchmarkSpec()
    assert spec.model == "gpt-3.5-turbo"
    assert spec.max_tokens == 256
    assert spec.temperature == 0.0


def test_benchmark_spec_round_trip():
    spec = BenchmarkSpec(
        model="gpt-4o",
        tasks=[Task("t1", "qa", "prompt?", "answer")],
    )
    spec2 = BenchmarkSpec.from_dict(spec.to_dict())
    assert spec2.model == "gpt-4o"
    assert len(spec2.tasks) == 1
    assert spec2.tasks[0].task_id == "t1"


def test_benchmark_spec_from_dict_defaults():
    spec = BenchmarkSpec.from_dict({})
    assert spec.model == "gpt-3.5-turbo"
    assert spec.tasks == []


# ── BenchmarkRunner.run_offline ───────────────────────────────────────────────

def test_run_offline_result_count():
    runner = BenchmarkRunner()
    results = runner.run_offline(lambda _: "Paris")
    assert len(results) == len(SAMPLE_TASKS)


def test_run_offline_accumulates():
    runner = BenchmarkRunner()
    runner.run_offline(lambda _: "Paris")
    runner.run_offline(lambda _: "Rome")
    assert len(runner.results) == 2 * len(SAMPLE_TASKS)


def test_run_offline_perfect_score_on_qa01():
    task = Task("qa_01", "qa", "What is the capital of France?", "Paris")
    runner = BenchmarkRunner([task])
    results = runner.run_offline(lambda _: "Paris")
    assert results[0].exact_match == 1.0
    assert results[0].rouge_l == 1.0


def test_run_offline_handles_exception():
    def bad_model(_: str) -> str:
        raise RuntimeError("model crashed")

    runner = BenchmarkRunner()
    results = runner.run_offline(bad_model)
    assert all(r.error is not None for r in results)
    assert all(r.prediction == "" for r in results)


def test_composite_score_bounds():
    runner = BenchmarkRunner()
    results = runner.run_offline(lambda _: "")
    for r in results:
        assert 0.0 <= r.composite_score <= 1.0


def test_composite_score_formula():
    r = BenchmarkResult(
        task_id="x", category="qa", prompt="", reference="",
        prediction="", latency_s=0.0,
        exact_match=0.0, exact_match_norm=1.0,
        rouge_l=1.0, bleu_1=1.0, f1=1.0,
        approx_tokens=1,
    )
    expected = 0.40 * 1.0 + 0.30 * 1.0 + 0.20 * 1.0 + 0.10 * 1.0
    assert abs(r.composite_score - expected) < 1e-9


# ── BenchmarkRunner.summarize ─────────────────────────────────────────────────

def test_summarize_keys():
    runner = BenchmarkRunner()
    runner.run_offline(lambda _: "Paris")
    summary = runner.summarize()
    assert "overall" in summary
    assert "by_category" in summary


def test_summarize_overall_count():
    runner = BenchmarkRunner()
    runner.run_offline(lambda _: "Paris")
    assert runner.summarize()["overall"]["n"] == len(SAMPLE_TASKS)


def test_summarize_empty():
    assert BenchmarkRunner().summarize() == {}


def test_summarize_categories():
    runner = BenchmarkRunner()
    runner.run_offline(lambda _: "Paris")
    cats = runner.summarize()["by_category"]
    assert "qa" in cats and "coding" in cats and "summarization" in cats


# ── Export ────────────────────────────────────────────────────────────────────

def test_export_jsonl(tmp_path: Path):
    runner = BenchmarkRunner()
    runner.run_offline(lambda _: "Paris")
    out = tmp_path / "results.jsonl"
    runner.export_jsonl(out)
    assert out.exists()
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == len(SAMPLE_TASKS)
    obj = json.loads(lines[0])
    assert "composite_score" in obj
    assert "task_id" in obj


def test_export_csv(tmp_path: Path):
    runner = BenchmarkRunner()
    runner.run_offline(lambda _: "Paris")
    out = tmp_path / "results.csv"
    runner.export_csv(out)
    assert out.exists()
    content = out.read_text()
    assert "composite_score" in content
    assert "task_id" in content


def test_export_jsonl_empty(tmp_path: Path):
    runner = BenchmarkRunner()
    out = tmp_path / "empty.jsonl"
    runner.export_jsonl(out)
    assert out.read_text() == ""


def test_export_csv_empty_is_noop(tmp_path: Path):
    runner = BenchmarkRunner()
    out = tmp_path / "empty.csv"
    runner.export_csv(out)
    assert not out.exists()


# ── CLI integration ────────────────────────────────────────────────────────────

def test_cli_sample_returns_zero(capsys):
    from llmbench.cli import main
    rc = main(["sample"])
    assert rc == 0
    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines() if l.strip()]
    assert len(lines) == len(SAMPLE_TASKS)
    obj = json.loads(lines[0])
    assert "task_id" in obj


def test_cli_default_demo_returns_zero(capsys):
    from llmbench.cli import main
    rc = main([])
    assert rc == 0


def test_cli_score_missing_file(tmp_path):
    from llmbench.cli import main
    rc = main(["score", str(tmp_path / "nonexistent.jsonl")])
    assert rc == 1


def test_cli_score_jsonl(tmp_path, capsys):
    from llmbench.cli import main
    preds = tmp_path / "preds.jsonl"
    preds.write_text(
        json.dumps({
            "task_id": "qa_01", "category": "qa",
            "prompt": "What is the capital of France?",
            "reference": "Paris", "prediction": "Paris",
        }) + "\n"
    )
    rc = main(["score", str(preds)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Tasks: 1" in out


def test_cli_run_missing_api_key(capsys):
    from llmbench.cli import main
    rc = main(["run"])
    assert rc == 1
