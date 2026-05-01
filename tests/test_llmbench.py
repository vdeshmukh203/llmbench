"""Tests for llmbench — metrics, runner, export, and task loading."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import llmbench as lb


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def test_tokenise_lowercases():
    assert lb._tokenise("Hello WORLD") == ["hello", "world"]


def test_tokenise_ignores_punctuation():
    assert lb._tokenise("hello, world!") == ["hello", "world"]


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------

def test_exact_match_identical():
    assert lb.exact_match("hello", "hello") == 1.0


def test_exact_match_different():
    assert lb.exact_match("hello", "world") == 0.0


def test_exact_match_strips_whitespace():
    assert lb.exact_match("  hello  ", "hello") == 1.0


def test_exact_match_case_sensitive():
    assert lb.exact_match("Hello", "hello") == 0.0


# ---------------------------------------------------------------------------
# exact_match_normalised
# ---------------------------------------------------------------------------

def test_em_norm_case_insensitive():
    assert lb.exact_match_normalised("Hello World", "hello world") == 1.0


def test_em_norm_strips_punctuation():
    assert lb.exact_match_normalised("Hello!", "hello") == 1.0


def test_em_norm_different():
    assert lb.exact_match_normalised("hello", "world") == 0.0


# ---------------------------------------------------------------------------
# rouge_l
# ---------------------------------------------------------------------------

def test_rouge_l_identical():
    assert lb.rouge_l("the cat sat", "the cat sat") == 1.0


def test_rouge_l_empty_prediction():
    assert lb.rouge_l("", "reference") == 0.0


def test_rouge_l_empty_reference():
    assert lb.rouge_l("prediction", "") == 0.0


def test_rouge_l_partial():
    score = lb.rouge_l("the cat", "the cat sat on the mat")
    assert 0.0 < score < 1.0


def test_rouge_l_no_overlap():
    assert lb.rouge_l("foo bar", "baz qux") == 0.0


# ---------------------------------------------------------------------------
# bleu_1
# ---------------------------------------------------------------------------

def test_bleu_1_identical():
    assert lb.bleu_1("the cat sat", "the cat sat") == 1.0


def test_bleu_1_empty_prediction():
    assert lb.bleu_1("", "reference") == 0.0


def test_bleu_1_no_overlap():
    assert lb.bleu_1("foo", "bar") == 0.0


def test_bleu_1_brevity_penalty():
    # Shorter prediction → brevity penalty < 1
    score_short = lb.bleu_1("cat", "the cat sat on the mat")
    score_long  = lb.bleu_1("the cat sat on the mat", "the cat sat on the mat")
    assert score_short < score_long


# ---------------------------------------------------------------------------
# f1_score
# ---------------------------------------------------------------------------

def test_f1_identical():
    assert lb.f1_score("the cat sat", "the cat sat") == 1.0


def test_f1_empty_prediction():
    assert lb.f1_score("", "reference") == 0.0


def test_f1_empty_reference():
    assert lb.f1_score("prediction", "") == 0.0


def test_f1_no_overlap():
    assert lb.f1_score("foo bar", "baz qux") == 0.0


def test_f1_partial():
    score = lb.f1_score("the cat", "the cat sat")
    assert 0.0 < score < 1.0


def test_f1_frequency_weighted():
    # "cat cat" vs "cat": prec = 1/2, rec = 1/1 → F1 = 2*(0.5*1)/(0.5+1) = 2/3
    assert abs(lb.f1_score("cat cat", "cat") - 2 / 3) < 1e-9


def test_f1_symmetric_for_identical():
    assert lb.f1_score("Paris", "Paris") == 1.0


# ---------------------------------------------------------------------------
# contains_code
# ---------------------------------------------------------------------------

def test_contains_code_detects_def():
    assert lb.contains_code("def foo(): pass")


def test_contains_code_detects_backticks():
    assert lb.contains_code("```python\nprint('hi')\n```")


def test_contains_code_plain_text():
    assert not lb.contains_code("The weather today is sunny.")


# ---------------------------------------------------------------------------
# _approx_tokens
# ---------------------------------------------------------------------------

def test_approx_tokens_basic():
    assert lb._approx_tokens("hello world") == 2


def test_approx_tokens_empty():
    assert lb._approx_tokens("") == 0


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

def test_task_to_dict():
    t = lb.Task("t1", "qa", "prompt", "ref")
    d = t.to_dict()
    assert d["task_id"] == "t1"
    assert d["category"] == "qa"


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------

def test_composite_score_perfect():
    r = lb.BenchmarkResult(
        task_id="t", category="qa", prompt="q", reference="a", prediction="a",
        latency_s=0.1, exact_match=1.0, exact_match_norm=1.0,
        rouge_l=1.0, bleu_1=1.0, f1=1.0, approx_tokens=1,
    )
    assert abs(r.composite_score - 1.0) < 1e-9


def test_composite_score_zero():
    r = lb.BenchmarkResult(
        task_id="t", category="qa", prompt="q", reference="a", prediction="",
        latency_s=0.0, exact_match=0.0, exact_match_norm=0.0,
        rouge_l=0.0, bleu_1=0.0, f1=0.0, approx_tokens=0,
    )
    assert r.composite_score == 0.0


def test_composite_score_weights():
    r = lb.BenchmarkResult(
        task_id="t", category="qa", prompt="q", reference="a", prediction="",
        latency_s=0.0, exact_match=0.0, exact_match_norm=0.5,
        rouge_l=0.0, bleu_1=0.0, f1=0.0, approx_tokens=0,
    )
    assert abs(r.composite_score - 0.10) < 1e-9


# ---------------------------------------------------------------------------
# BenchmarkSpec
# ---------------------------------------------------------------------------

def test_benchmark_spec_defaults():
    spec = lb.BenchmarkSpec(tasks=[])
    assert spec.temperature == 0.0
    assert spec.seed is None


# ---------------------------------------------------------------------------
# BenchmarkRunner — run_offline
# ---------------------------------------------------------------------------

def test_run_offline_returns_all_tasks():
    runner = lb.BenchmarkRunner()
    results = runner.run_offline(lambda _: "Paris")
    assert len(results) == len(lb.SAMPLE_TASKS)


def test_run_offline_perfect_answer():
    task = lb.Task("t1", "qa", "Capital?", "Paris")
    runner = lb.BenchmarkRunner([task])
    results = runner.run_offline(lambda _: "Paris")
    assert results[0].exact_match == 1.0
    assert results[0].rouge_l == 1.0


def test_run_offline_captures_errors():
    task = lb.Task("t1", "qa", "q", "a")
    runner = lb.BenchmarkRunner([task])

    def _raises(_):
        raise ValueError("intentional error")

    results = runner.run_offline(_raises)
    assert results[0].error == "intentional error"
    assert results[0].prediction == ""


def test_run_offline_accumulates_results():
    runner = lb.BenchmarkRunner()
    runner.run_offline(lambda _: "x")
    runner.run_offline(lambda _: "y")
    assert len(runner.results) == 2 * len(lb.SAMPLE_TASKS)


def test_run_offline_records_model():
    task = lb.Task("t1", "qa", "q", "a")
    runner = lb.BenchmarkRunner([task])
    results = runner.run_offline(lambda _: "a")
    assert results[0].model == "offline"


def test_run_offline_records_timestamp():
    task = lb.Task("t1", "qa", "q", "a")
    runner = lb.BenchmarkRunner([task])
    results = runner.run_offline(lambda _: "a")
    assert results[0].timestamp  # non-empty ISO string


# ---------------------------------------------------------------------------
# BenchmarkRunner — summarize
# ---------------------------------------------------------------------------

def test_summarize_overall_keys():
    runner = lb.BenchmarkRunner()
    runner.run_offline(lambda _: "Paris")
    summary = runner.summarize()
    for key in ("n", "exact_match", "rouge_l", "bleu_1", "f1", "composite", "avg_latency_s"):
        assert key in summary["overall"]


def test_summarize_by_category():
    runner = lb.BenchmarkRunner()
    runner.run_offline(lambda _: "x")
    summary = runner.summarize()
    assert "qa" in summary["by_category"]
    assert "coding" in summary["by_category"]


def test_summarize_empty_returns_empty_dict():
    runner = lb.BenchmarkRunner()
    assert runner.summarize() == {}


def test_summarize_total_count():
    runner = lb.BenchmarkRunner()
    runner.run_offline(lambda _: "Paris")
    assert runner.summarize()["overall"]["n"] == len(lb.SAMPLE_TASKS)


# ---------------------------------------------------------------------------
# BenchmarkRunner — export_jsonl + SHA-256 sidecar
# ---------------------------------------------------------------------------

def test_export_jsonl_creates_file():
    runner = lb.BenchmarkRunner([lb.SAMPLE_TASKS[0]])
    runner.run_offline(lambda _: "Paris")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "results.jsonl"
        runner.export_jsonl(out)
        assert out.exists()
        lines = out.read_text().splitlines()
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert "composite_score" in obj


def test_export_jsonl_writes_sha256_sidecar():
    runner = lb.BenchmarkRunner([lb.SAMPLE_TASKS[0]])
    runner.run_offline(lambda _: "Paris")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "results.jsonl"
        runner.export_jsonl(out)
        sha_file = out.with_suffix(".sha256")
        assert sha_file.exists()
        content = sha_file.read_text()
        assert "results.jsonl" in content
        assert len(content.split()[0]) == 64  # hex SHA-256


def test_export_jsonl_sha256_is_correct():
    import hashlib
    runner = lb.BenchmarkRunner([lb.SAMPLE_TASKS[0]])
    runner.run_offline(lambda _: "Paris")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "r.jsonl"
        runner.export_jsonl(out)
        content = out.read_text(encoding="utf-8").rstrip("\n")
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        recorded = out.with_suffix(".sha256").read_text().split()[0]
        assert recorded == expected


# ---------------------------------------------------------------------------
# BenchmarkRunner — export_csv
# ---------------------------------------------------------------------------

def test_export_csv_creates_file():
    runner = lb.BenchmarkRunner([lb.SAMPLE_TASKS[0]])
    runner.run_offline(lambda _: "Paris")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "results.csv"
        runner.export_csv(out)
        assert out.exists()
        assert out.stat().st_size > 0


def test_export_csv_has_composite_column():
    runner = lb.BenchmarkRunner([lb.SAMPLE_TASKS[0]])
    runner.run_offline(lambda _: "Paris")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "results.csv"
        runner.export_csv(out)
        header = out.read_text().splitlines()[0]
        assert "composite_score" in header


# ---------------------------------------------------------------------------
# load_tasks_from_file
# ---------------------------------------------------------------------------

def _write_temp(suffix: str, content: str) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
    f.write(content)
    f.close()
    return Path(f.name)


def test_load_tasks_from_jsonl():
    path = _write_temp(".jsonl", json.dumps({
        "task_id": "x1", "category": "qa", "prompt": "Q?", "reference": "A",
    }) + "\n")
    try:
        tasks = lb.load_tasks_from_file(path)
        assert len(tasks) == 1
        assert tasks[0].task_id == "x1"
        assert tasks[0].reference == "A"
    finally:
        path.unlink()


def test_load_tasks_from_json_array():
    path = _write_temp(".json", json.dumps([
        {"task_id": "x1", "category": "qa", "prompt": "Q?", "reference": "A"},
        {"task_id": "x2", "category": "qa", "prompt": "Q2?", "reference": "B"},
    ]))
    try:
        tasks = lb.load_tasks_from_file(path)
        assert len(tasks) == 2
        assert tasks[1].task_id == "x2"
    finally:
        path.unlink()


def test_load_tasks_missing_field_raises():
    path = _write_temp(".jsonl", json.dumps({
        "task_id": "x1", "category": "qa", "prompt": "Q?",
        # 'reference' missing
    }) + "\n")
    try:
        with pytest.raises(KeyError):
            lb.load_tasks_from_file(path)
    finally:
        path.unlink()


# ---------------------------------------------------------------------------
# SAMPLE_TASKS sanity
# ---------------------------------------------------------------------------

def test_sample_tasks_non_empty():
    assert len(lb.SAMPLE_TASKS) > 0


def test_sample_tasks_have_required_fields():
    for t in lb.SAMPLE_TASKS:
        assert t.task_id
        assert t.category
        assert t.prompt
        assert t.reference
