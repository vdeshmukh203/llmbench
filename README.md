# llmbench

[![CI](https://github.com/vdeshmukh203/llmbench/actions/workflows/ci.yml/badge.svg)](https://github.com/vdeshmukh203/llmbench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

**llmbench** is a lightweight, stdlib-only Python framework for designing, running, and reporting reproducible benchmarks of large language models (LLMs).  
No third-party dependencies are required for core functionality.

---

## Features

- **Declarative task specification** — define tasks in JSONL; built-in QA, coding, and summarisation examples included.
- **Provider-agnostic** — test any callable (local model, mock, or API wrapper) via `run_offline`, or call an OpenAI-compatible endpoint directly via `run_openai`.
- **Reproducible metrics** — ROUGE-L, unigram BLEU, token-level F1, and exact match (strict + normalised), with a weighted composite score.
- **Structured output** — results exported to JSONL or CSV with full provenance (task ID, prompt, reference, prediction, latency, scores).
- **Graphical interface** — a tkinter GUI for interactive benchmarking, live progress, and one-click export.
- **Zero dependencies** — all metric and IO code uses Python's standard library only.

---

## Installation

```bash
pip install llmbench
```

Or directly from source:

```bash
git clone https://github.com/vdeshmukh203/llmbench.git
cd llmbench
pip install -e .
```

---

## Quick Start

### Python API

```python
from llmbench import BenchmarkRunner, BenchmarkSpec

# Use built-in sample tasks
runner = BenchmarkRunner(BenchmarkSpec())

# Evaluate any callable that maps a prompt → prediction string
results = runner.run_offline(lambda prompt: my_model(prompt))

summary = runner.summarize()
print(summary["overall"])
# {'n': 10, 'rouge_l': 0.82, 'f1': 0.79, 'composite': 0.80, ...}

runner.export_jsonl("results.jsonl")
runner.export_csv("results.csv")
```

### Custom Tasks

```python
from llmbench import BenchmarkSpec, Task

spec = BenchmarkSpec([
    Task("q1", "qa",      "What is the capital of France?",  "Paris"),
    Task("q2", "qa",      "What year did WWII end?",          "1945"),
    Task("s1", "summary", "Summarise: The cat sat on the mat.", "A cat sat."),
])
runner = BenchmarkRunner(spec)
```

Or load from a JSONL file:

```bash
# tasks.jsonl — one task per line
{"task_id":"q1","category":"qa","prompt":"Capital of France?","reference":"Paris"}
```

```python
spec = BenchmarkSpec.from_jsonl("tasks.jsonl")
```

### CLI

```bash
# Print built-in sample tasks
llmbench sample

# Run offline demo (no API key needed)
llmbench

# Score a predictions file
llmbench score predictions.jsonl
llmbench score predictions.jsonl --format json

# Run against OpenAI API
llmbench run --api-key sk-... --model gpt-4o --output results.jsonl --csv results.csv
```

### Graphical Interface

```bash
llmbench gui
# or
llmbench-gui
```

The GUI provides three tabs:

| Tab | Description |
|-----|-------------|
| **Run** | Select mode (Demo / OpenAI API), browse a custom task file, run with live progress log |
| **Results** | Sortable table of per-task metrics |
| **Summary** | Aggregate metrics by category and overall |

Export to JSONL or CSV via the **File** menu or buttons in the Results tab.

---

## Metrics

| Metric | Description | Weight in Composite |
|--------|-------------|---------------------|
| **Exact Match (EM)** | Strict string equality (whitespace-stripped) | — |
| **EM Normalised** | Case-insensitive, punctuation-agnostic EM | 20 % |
| **ROUGE-L** | Longest-common-subsequence F1 over tokens | 40 % |
| **BLEU-1** | Unigram precision with brevity penalty | 10 % |
| **F1** | Set-based token overlap F1 | 30 % |

**Composite score** = `0.40·ROUGE-L + 0.30·F1 + 0.20·EM-norm + 0.10·BLEU-1`

All metrics are in [0, 1]; higher is better.

---

## Output Format

Each JSONL record contains:

```json
{
  "task_id": "qa_01",
  "category": "qa",
  "prompt": "What is the capital of France?",
  "reference": "Paris",
  "prediction": "Paris",
  "latency_s": 0.0012,
  "exact_match": 1.0,
  "exact_match_norm": 1.0,
  "rouge_l": 1.0,
  "bleu_1": 1.0,
  "f1": 1.0,
  "approx_tokens": 1,
  "error": null,
  "composite_score": 1.0
}
```

---

## Standalone Script

`llmbench.py` in the project root is a self-contained single-file version with no package dependencies — suitable for copying into any project without installation.

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Citation

If you use llmbench in published research, please cite:

```bibtex
@article{Deshmukh2026,
  author  = {Deshmukh, Vaibhav},
  title   = {llmbench: A lightweight framework for reproducible benchmarking of large language models},
  journal = {Journal of Open Source Software},
  year    = {2026},
}
```

See also [`CITATION.cff`](CITATION.cff).

---

## License

MIT — see [`LICENSE`](LICENSE).
