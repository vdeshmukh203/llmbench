# llmbench

A lightweight Python framework for reproducible benchmarking of large language models (LLMs).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Features

- **Zero external dependencies** — pure Python stdlib
- **Provider-agnostic** — any OpenAI-compatible chat API, plus local/offline models
- **Built-in metrics** — ROUGE-L, BLEU-1, token-level F1, exact match (raw and normalised)
- **Composite score** — weighted aggregate of all metrics
- **Structured output** — JSONL and CSV with full provenance
- **Graphical interface** — tkinter GUI for interactive benchmarking
- **CLI** — `llmbench run`, `llmbench score`, `llmbench sample`, `llmbench gui`

## Installation

```bash
pip install .
```

## Quick start

### Python API

```python
from llmbench import BenchmarkRunner, Task

tasks = [Task("q1", "qa", "What is the capital of France?", "Paris")]
runner = BenchmarkRunner(tasks)
results = runner.run_offline(lambda prompt: "Paris")  # replace with your model

summary = runner.summarize()
print(summary["overall"])
# {'n': 1, 'exact_match': 1.0, 'rouge_l': 1.0, 'f1': 1.0, 'composite': 1.0, ...}

runner.export_jsonl("results.jsonl")
runner.export_csv("results.csv")
```

### OpenAI-compatible API

```python
from llmbench import BenchmarkRunner

runner = BenchmarkRunner()
runner.run_openai(api_key="sk-...", model="gpt-4o")
print(runner.summarize())
runner.export_jsonl("results.jsonl")
```

### CLI

```bash
# Run offline demo (no API key needed)
python llmbench.py

# Print built-in sample tasks
llmbench sample

# Run against an API
llmbench run --model gpt-4o --output results.jsonl

# Score a predictions file
llmbench score predictions.jsonl
llmbench score predictions.jsonl --format json

# Launch the GUI
llmbench gui
# or
llmbench-gui
```

### Graphical interface

```bash
llmbench gui
```

The GUI provides four tabs:

| Tab | Description |
|-----|-------------|
| **Run API Benchmark** | Configure API key, model, and task file; run with live progress |
| **Score Predictions** | Load a predictions JSONL and compute all metrics |
| **Results** | Sortable table of per-task scores; export to JSONL or CSV |
| **Sample Tasks** | Browse the built-in evaluation tasks |

## Task file format

Tasks are JSONL files with one JSON object per line:

```jsonl
{"task_id": "q1", "category": "qa", "prompt": "What is the capital of France?", "reference": "Paris"}
{"task_id": "q2", "category": "qa", "prompt": "What year did WWII end?", "reference": "1945"}
```

## Metrics

| Metric | Description |
|--------|-------------|
| `exact_match` | 1.0 if prediction equals reference after stripping whitespace |
| `exact_match_norm` | 1.0 if lower-cased, punctuation-stripped tokens match |
| `rouge_l` | ROUGE-L F1 via longest common subsequence |
| `bleu_1` | Unigram BLEU with brevity penalty |
| `f1` | Token-level set F1 (SQuAD-style) |
| `composite` | 0.40 × ROUGE-L + 0.30 × F1 + 0.20 × EM-norm + 0.10 × BLEU-1 |

## Running tests

```bash
pip install pytest
pytest tests/
```

## Citation

If you use llmbench in your research, please cite:

```bibtex
@article{deshmukh2026llmbench,
  title={llmbench: A lightweight framework for reproducible benchmarking of large language models},
  author={Deshmukh, Vaibhav},
  journal={Journal of Open Source Software},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
