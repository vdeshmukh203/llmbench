"""Tests for the llmbench package."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import llmbench as lb


# ---------------------------------------------------------------------------
# Metric: exact_match
# ---------------------------------------------------------------------------


def test_exact_match_equal():
    assert lb.exact_match("hello", "hello") == 1.0


def test_exact_match_strips_whitespace():
    assert lb.exact_match("  hello  ", "hello") == 1.0


def test_exact_match_different():
    assert lb.exact_match("hello", "world") == 0.0


def test_exact_match_case_sensitive():
    assert lb.exact_match("Hello", "hello") == 0.0


# ---------------------------------------------------------------------------
# Metric: exact_match_normalised
# ---------------------------------------------------------------------------


def test_exact_match_norm_equal():
    assert lb.exact_match_normalised("Hello World!", "hello world") == 1.0


def test_exact_match_norm_different():
    assert lb.exact_match_normalised("foo", "bar") == 0.0


# ---------------------------------------------------------------------------
# Metric: rouge_l
# ---------------------------------------------------------------------------


def test_rouge_l_identical():
    assert lb.rouge_l("the cat sat", "the cat sat") == 1.0


def test_rouge_l_empty_prediction():
    assert lb.rouge_l("", "hello world") == 0.0


def test_rouge_l_empty_reference():
    assert lb.rouge_l("hello world", "") == 0.0


def test_rouge_l_partial_overlap():
    score = lb.rouge_l("the cat sat on the mat", "the cat sat")
    assert 0.0 < score < 1.0


def test_rouge_l_no_overlap():
    assert lb.rouge_l("foo bar", "baz qux") == 0.0


# ---------------------------------------------------------------------------
# Metric: bleu_1
# ---------------------------------------------------------------------------


def test_bleu_1_identical():
    assert lb.bleu_1("the cat sat", "the cat sat") == 1.0


def test_bleu_1_empty_prediction():
    assert lb.bleu_1("", "hello") == 0.0


def test_bleu_1_no_overlap():
    score = lb.bleu_1("foo bar", "baz qux")
    assert score == 0.0


def test_bleu_1_partial():
    score = lb.bleu_1("the cat", "the cat sat on mat")
    assert 0.0 < score <= 1.0


# ---------------------------------------------------------------------------
# Metric: f1_score
# ---------------------------------------------------------------------------


def test_f1_identical():
    assert lb.f1_score("the cat sat", "the cat sat") == 1.0


def test_f1_empty():
    assert lb.f1_score("", "hello") == 0.0
    assert lb.f1_score("hello", "") == 0.0


def test_f1_no_overlap():
    assert lb.f1_score("foo bar", "baz qux") == 0.0


def test_f1_partial_overlap():
    score = lb.f1_score("the cat", "the cat sat")
    assert 0.0 < score < 1.0


def test_f1_uses_counts_not_sets():
    """Repeated tokens must be counted, not deduplicated."""
    # "a a b" vs "a b b": common=min(2,1)+min(1,2)=1+1=2
    # prec=2/3, rec=2/3, f1=2/3
    score = lb.f1_score("a a b", "a b b")
    assert abs(score - 2 / 3) < 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_approx_tokens_positive():
    n = lb._approx_tokens("hello world")
    assert isinstance(n, int) and n == 2


def test_approx_tokens_empty():
    assert lb._approx_tokens("") == 0


# ---------------------------------------------------------------------------
# BenchmarkRunner — offline
# ---------------------------------------------------------------------------


def test_runner_run_offline_returns_results():
    runner = lb.BenchmarkRunner()
    results = runner.run_offline(lambda p: "Paris")
    assert len(results) == len(lb.SAMPLE_TASKS)
    assert all(r.prediction == "Paris" for r in results)


def test_runner_exact_answer_scores_one():
    task = lb.Task("t1", "qa", "Capital of France?", "Paris")
    runner = lb.BenchmarkRunner([task])
    results = runner.run_offline(lambda _: "Paris")
    assert results[0].exact_match == 1.0
    assert results[0].rouge_l == 1.0


def test_runner_accumulates_results():
    runner = lb.BenchmarkRunner()
    runner.run_offline(lambda p: "x")
    runner.run_offline(lambda p: "y")
    assert len(runner.results) == 2 * len(lb.SAMPLE_TASKS)


def test_runner_error_handling():
    task = lb.Task("t_err", "qa", "prompt", "reference")
    runner = lb.BenchmarkRunner([task])

    def bad_fn(_: str) -> str:
        raise RuntimeError("boom")

    results = runner.run_offline(bad_fn)
    assert results[0].error == "boom"
    assert results[0].prediction == ""


# ---------------------------------------------------------------------------
# BenchmarkResult — composite_score
# ---------------------------------------------------------------------------


def test_composite_score_range():
    task = lb.Task("t1", "qa", "prompt", "Paris")
    runner = lb.BenchmarkRunner([task])
    results = runner.run_offline(lambda _: "Paris")
    cs = results[0].composite_score
    assert 0.0 <= cs <= 1.0


def test_composite_score_in_to_dict():
    task = lb.Task("t1", "qa", "q", "answer")
    runner = lb.BenchmarkRunner([task])
    results = runner.run_offline(lambda _: "answer")
    d = results[0].to_dict()
    assert "composite_score" in d


# ---------------------------------------------------------------------------
# BenchmarkRunner — summarize
# ---------------------------------------------------------------------------


def test_summarize_empty():
    runner = lb.BenchmarkRunner([])
    assert runner.summarize() == {}


def test_summarize_keys():
    runner = lb.BenchmarkRunner()
    runner.run_offline(lambda p: "dummy")
    summary = runner.summarize()
    assert "overall" in summary
    assert "by_category" in summary
    ov = summary["overall"]
    for key in ("n", "exact_match", "rouge_l", "bleu_1", "f1", "composite", "avg_latency_s"):
        assert key in ov, f"missing key: {key}"


# ---------------------------------------------------------------------------
# BenchmarkSpec
# ---------------------------------------------------------------------------


def test_spec_from_sample():
    spec = lb.BenchmarkSpec.from_sample()
    assert len(spec) == len(lb.SAMPLE_TASKS)


def test_spec_from_jsonl(tmp_path: Path):
    jsonl = tmp_path / "tasks.jsonl"
    tasks_data = [
        {"task_id": "x1", "category": "qa", "prompt": "Q?", "reference": "A"},
        {"task_id": "x2", "category": "qa", "prompt": "Q2?", "reference": "A2"},
    ]
    jsonl.write_text("\n".join(json.dumps(t) for t in tasks_data) + "\n", encoding="utf-8")
    spec = lb.BenchmarkSpec.from_jsonl(jsonl)
    assert len(spec) == 2
    assert spec.tasks[0].task_id == "x1"


def test_spec_from_jsonl_empty_raises(tmp_path: Path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No tasks found"):
        lb.BenchmarkSpec.from_jsonl(empty)


def test_spec_from_jsonl_bad_json_raises(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        lb.BenchmarkSpec.from_jsonl(bad)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_jsonl(tmp_path: Path):
    task = lb.Task("t1", "qa", "q", "ans")
    runner = lb.BenchmarkRunner([task])
    runner.run_offline(lambda _: "ans")
    out = tmp_path / "results.jsonl"
    runner.export_jsonl(out)
    lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert "composite_score" in obj
    assert obj["task_id"] == "t1"


def test_export_csv(tmp_path: Path):
    task = lb.Task("t1", "qa", "q", "ans")
    runner = lb.BenchmarkRunner([task])
    runner.run_offline(lambda _: "ans")
    out = tmp_path / "results.csv"
    runner.export_csv(out)
    content = out.read_text(encoding="utf-8")
    assert "composite_score" in content
    assert "t1" in content


def test_export_empty_is_noop(tmp_path: Path):
    runner = lb.BenchmarkRunner([])
    out = tmp_path / "results.jsonl"
    runner.export_jsonl(out)
    assert not out.exists()
