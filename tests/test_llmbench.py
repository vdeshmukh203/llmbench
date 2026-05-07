"""Tests for the llmbench package."""

import json
import sys
from pathlib import Path

# Support both installed package and in-tree development layout
_src = Path(__file__).parent.parent / "src"
if _src.is_dir():
    sys.path.insert(0, str(_src))

import llmbench as lb
from llmbench.models import BenchmarkResult, Task, SAMPLE_TASKS
from llmbench.metrics import (
    _tokenise, exact_match, exact_match_normalised,
    rouge_l, bleu_1, f1_score, _approx_tokens, contains_code,
)
from llmbench.runner import BenchmarkRunner
from llmbench.spec import BenchmarkSpec


# ── import sanity ─────────────────────────────────────────────────────────────

def test_import_top_level():
    assert hasattr(lb, "BenchmarkRunner")
    assert hasattr(lb, "BenchmarkSpec")
    assert hasattr(lb, "Task")
    assert hasattr(lb, "BenchmarkResult")
    assert hasattr(lb, "SAMPLE_TASKS")


def test_version():
    assert lb.__version__ == "0.1.0"


# ── tokeniser ─────────────────────────────────────────────────────────────────

def test_tokenise_basic():
    assert _tokenise("Hello, World!") == ["hello", "world"]


def test_tokenise_empty():
    assert _tokenise("") == []


def test_tokenise_numbers():
    assert _tokenise("H2O CO2") == ["h2o", "co2"]


# ── exact_match ───────────────────────────────────────────────────────────────

def test_exact_match_equal():
    assert exact_match("Paris", "Paris") == 1.0


def test_exact_match_strips_whitespace():
    assert exact_match("  Paris  ", "Paris") == 1.0


def test_exact_match_case_sensitive():
    assert exact_match("paris", "Paris") == 0.0


def test_exact_match_unequal():
    assert exact_match("London", "Paris") == 0.0


# ── exact_match_normalised ────────────────────────────────────────────────────

def test_exact_match_norm_case_insensitive():
    assert exact_match_normalised("Hello World", "hello world") == 1.0


def test_exact_match_norm_punct_ignored():
    assert exact_match_normalised("Hello, World!", "hello world") == 1.0


def test_exact_match_norm_unequal():
    assert exact_match_normalised("foo bar", "foo baz") == 0.0


# ── rouge_l ───────────────────────────────────────────────────────────────────

def test_rouge_l_identical():
    assert rouge_l("the cat sat", "the cat sat") == 1.0


def test_rouge_l_empty_prediction():
    assert rouge_l("", "the cat sat") == 0.0


def test_rouge_l_empty_reference():
    assert rouge_l("the cat sat", "") == 0.0


def test_rouge_l_partial_overlap():
    score = rouge_l("the cat sat on the mat", "the cat sat")
    assert 0.0 < score < 1.0


def test_rouge_l_no_overlap():
    assert rouge_l("foo bar", "baz qux") == 0.0


# ── bleu_1 ────────────────────────────────────────────────────────────────────

def test_bleu_1_identical():
    assert bleu_1("the cat sat", "the cat sat") == 1.0


def test_bleu_1_empty_prediction():
    assert bleu_1("", "the cat sat") == 0.0


def test_bleu_1_no_overlap():
    assert bleu_1("foo bar", "baz qux") == 0.0


def test_bleu_1_brevity_penalty():
    # short prediction against long reference should be penalised
    short_score = bleu_1("cat", "the cat sat on the mat")
    assert short_score < 1.0


# ── f1_score ──────────────────────────────────────────────────────────────────

def test_f1_identical():
    assert f1_score("the cat sat", "the cat sat") == 1.0


def test_f1_partial_overlap():
    score = f1_score("the cat", "the dog")
    assert 0.0 < score < 1.0  # 'the' in common


def test_f1_no_overlap():
    assert f1_score("foo bar", "baz qux") == 0.0


def test_f1_empty_prediction():
    assert f1_score("", "the cat sat") == 0.0


def test_f1_bag_of_words():
    # SQuAD-style: repeated tokens in prediction count against precision
    score_a = f1_score("cat cat cat", "cat")
    score_b = f1_score("cat", "cat")
    assert score_b > score_a


# ── _approx_tokens ────────────────────────────────────────────────────────────

def test_approx_tokens():
    assert _approx_tokens("hello world") == 2


def test_approx_tokens_empty():
    assert _approx_tokens("") == 0


# ── contains_code ─────────────────────────────────────────────────────────────

def test_contains_code_true():
    assert contains_code("def foo(): return 1")
    assert contains_code("```python\nprint('hi')\n```")


def test_contains_code_false():
    assert not contains_code("The capital of France is Paris.")


# ── BenchmarkResult ───────────────────────────────────────────────────────────

def test_composite_score_perfect():
    r = BenchmarkResult(
        task_id="t1", category="qa", prompt="q", reference="a",
        prediction="a", latency_s=0.1,
        exact_match=1.0, exact_match_norm=1.0,
        rouge_l=1.0, bleu_1=1.0, f1=1.0,
        approx_tokens=1, error=None,
    )
    assert abs(r.composite_score - 1.0) < 1e-9


def test_composite_score_zero():
    r = BenchmarkResult(
        task_id="t1", category="qa", prompt="q", reference="a",
        prediction="", latency_s=0.0,
        exact_match=0.0, exact_match_norm=0.0,
        rouge_l=0.0, bleu_1=0.0, f1=0.0,
        approx_tokens=0, error=None,
    )
    assert r.composite_score == 0.0


def test_composite_weights():
    r = BenchmarkResult(
        task_id="t", category="qa", prompt="q", reference="a",
        prediction="x", latency_s=0.0,
        exact_match=0.0, exact_match_norm=0.0,
        rouge_l=1.0, bleu_1=0.0, f1=0.0,
        approx_tokens=1, error=None,
    )
    assert abs(r.composite_score - 0.40) < 1e-9


def test_benchmark_result_to_dict():
    r = BenchmarkResult(
        task_id="t1", category="qa", prompt="q", reference="a",
        prediction="b", latency_s=0.5,
        exact_match=0.0, exact_match_norm=0.0,
        rouge_l=0.0, bleu_1=0.0, f1=0.0,
        approx_tokens=1, error=None,
    )
    d = r.to_dict()
    assert d["task_id"] == "t1"
    assert "composite_score" not in d  # composite is a property, not a field


# ── Task ──────────────────────────────────────────────────────────────────────

def test_task_to_dict():
    t = Task("qa_01", "qa", "prompt", "ref")
    d = t.to_dict()
    assert d["task_id"] == "qa_01"
    assert d["metadata"] == {}


def test_sample_tasks_nonempty():
    assert len(SAMPLE_TASKS) >= 10


# ── BenchmarkRunner ───────────────────────────────────────────────────────────

def test_runner_default_tasks():
    runner = BenchmarkRunner()
    assert len(runner.tasks) == len(SAMPLE_TASKS)


def test_runner_custom_tasks():
    tasks = [Task("t1", "qa", "What?", "Answer")]
    runner = BenchmarkRunner(tasks)
    assert len(runner.tasks) == 1


def test_run_offline_returns_results():
    runner = BenchmarkRunner()
    results = runner.run_offline(lambda _: "test")
    assert len(results) == len(SAMPLE_TASKS)


def test_run_offline_exact_answer():
    tasks = [Task("qa_01", "qa", "What is the capital of France?", "Paris")]
    runner = BenchmarkRunner(tasks)
    results = runner.run_offline(lambda _: "Paris")
    assert results[0].exact_match == 1.0
    assert abs(results[0].composite_score - 1.0) < 1e-9


def test_run_offline_accumulates_results():
    runner = BenchmarkRunner()
    runner.run_offline(lambda _: "a")
    runner.run_offline(lambda _: "b")
    assert len(runner.results) == 2 * len(SAMPLE_TASKS)


def test_run_offline_handles_exception():
    tasks = [Task("t1", "qa", "q?", "a")]
    runner = BenchmarkRunner(tasks)

    def boom(_):
        raise ValueError("model error")

    results = runner.run_offline(boom)
    assert results[0].error == "model error"
    assert results[0].prediction == ""


def test_run_offline_latency_recorded():
    runner = BenchmarkRunner([Task("t1", "qa", "q?", "a")])
    results = runner.run_offline(lambda _: "a")
    assert results[0].latency_s >= 0.0


# ── summarize ─────────────────────────────────────────────────────────────────

def test_summarize_structure():
    runner = BenchmarkRunner()
    runner.run_offline(lambda _: "test")
    summary = runner.summarize()
    assert "overall" in summary
    assert "by_category" in summary
    assert summary["overall"]["n"] == len(SAMPLE_TASKS)


def test_summarize_empty():
    runner = BenchmarkRunner()
    assert runner.summarize() == {}


def test_summarize_categories():
    runner = BenchmarkRunner()
    runner.run_offline(lambda _: "x")
    cats = runner.summarize()["by_category"]
    assert "qa" in cats
    assert "coding" in cats
    assert "summarization" in cats


# ── export ────────────────────────────────────────────────────────────────────

def test_export_jsonl(tmp_path):
    runner = BenchmarkRunner()
    runner.run_offline(lambda _: "test")
    out = tmp_path / "results.jsonl"
    runner.export_jsonl(out)
    assert out.is_file()
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == len(SAMPLE_TASKS)
    row = json.loads(lines[0])
    assert "composite_score" in row
    assert "task_id" in row


def test_export_csv(tmp_path):
    runner = BenchmarkRunner()
    runner.run_offline(lambda _: "test")
    out = tmp_path / "results.csv"
    runner.export_csv(out)
    assert out.is_file()
    import csv
    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(SAMPLE_TASKS)
    assert "composite_score" in rows[0]


def test_export_jsonl_empty(tmp_path):
    runner = BenchmarkRunner()
    out = tmp_path / "empty.jsonl"
    runner.export_jsonl(out)  # no results — should not create file
    assert not out.exists()


# ── BenchmarkSpec ─────────────────────────────────────────────────────────────

def test_spec_from_json_array(tmp_path):
    data = [
        {"task_id": "t1", "category": "qa", "prompt": "q?", "reference": "a", "metadata": {}}
    ]
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps(data))
    spec = BenchmarkSpec.from_json(p)
    assert len(spec.tasks) == 1
    assert spec.tasks[0].task_id == "t1"


def test_spec_from_json_object(tmp_path):
    data = {
        "name": "my-bench",
        "description": "test",
        "metadata": {},
        "tasks": [
            {"task_id": "t1", "category": "qa", "prompt": "q?", "reference": "a", "metadata": {}}
        ],
    }
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(data))
    spec = BenchmarkSpec.from_json(p)
    assert spec.name == "my-bench"
    assert len(spec.tasks) == 1


def test_spec_from_jsonl(tmp_path):
    p = tmp_path / "tasks.jsonl"
    p.write_text(
        json.dumps({"task_id": "t1", "category": "qa", "prompt": "q?", "reference": "a"}) + "\n"
        + json.dumps({"task_id": "t2", "category": "qa", "prompt": "q2?", "reference": "b"}) + "\n"
    )
    spec = BenchmarkSpec.from_jsonl(p)
    assert len(spec.tasks) == 2


def test_spec_to_json_roundtrip(tmp_path):
    spec = BenchmarkSpec(
        tasks=[Task("t1", "qa", "q?", "a")],
        name="roundtrip",
        description="desc",
    )
    out = tmp_path / "spec_out.json"
    spec.to_json(out)
    spec2 = BenchmarkSpec.from_json(out)
    assert spec2.name == "roundtrip"
    assert len(spec2.tasks) == 1
    assert spec2.tasks[0].task_id == "t1"


def test_spec_from_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "tasks.jsonl"
    p.write_text(
        "\n"
        + json.dumps({"task_id": "t1", "category": "qa", "prompt": "q?", "reference": "a"}) + "\n"
        + "\n"
    )
    spec = BenchmarkSpec.from_jsonl(p)
    assert len(spec.tasks) == 1
