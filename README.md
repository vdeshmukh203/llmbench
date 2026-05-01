# llmbench

A lightweight, reproducible LLM benchmarking framework.  No heavy dependencies — the core uses only the Python standard library.

## Features

- **Metrics** — ROUGE-L, BLEU-1, token-level F1 (SQuAD style), exact match (raw and normalised)
- **Composite score** — configurable weighted aggregate for ranking runs
- **Provenance** — SHA-256 checksum sidecar files for tamper detection; run metadata (model, parameters, timestamp, UUID) embedded in all exports
- **Task formats** — built-in sample tasks; load custom tasks from JSON array, JSONL, or YAML
- **Backends** — any OpenAI-compatible chat completions endpoint (OpenAI, Ollama, vLLM, …); offline callable for custom model functions
- **Deterministic runs** — pass `--seed` / `seed=` to pin API outputs where supported
- **GUI** — standalone Tkinter interface, no extra dependencies
- **Export** — CSV and JSONL, each with a SHA-256 sidecar

## Installation

```
pip install llmbench
```

For YAML task files, also install PyYAML:

```
pip install "llmbench[yaml]"
```

## Quick start

### CLI — demo mode (no API key)

```
python llmbench.py
```

### CLI — run against an OpenAI-compatible API

```bash
export OPENAI_API_KEY=sk-...
llmbench run --model gpt-4o --output results.jsonl --csv results.csv
# Target a local server (Ollama, vLLM, …)
llmbench run --base-url http://localhost:11434/v1/chat/completions --model llama3
```

### CLI — score pre-collected predictions

Create a JSONL file where each line has `task_id`, `category`, `prompt`, `reference`, and `prediction`:

```jsonl
{"task_id":"qa_01","category":"qa","prompt":"What is the capital of France?","reference":"Paris","prediction":"Paris"}
```

```
llmbench score preds.jsonl
llmbench score preds.jsonl --format json
```

### CLI — print built-in tasks

```
llmbench sample
```

### Python API

```python
import llmbench as lb

runner = lb.BenchmarkRunner()

# Evaluate any callable
results = runner.run_offline(my_model_fn)
print(runner.summarize())

# OpenAI-compatible API with deterministic seed
results = runner.run_openai(
    api_key="sk-...", model="gpt-4o",
    seed=42, temperature=0.0,
)
runner.export_jsonl("results.jsonl")  # also writes results.sha256
runner.export_csv("results.csv")
```

### Custom tasks

```python
tasks = lb.load_tasks_from_file("my_tasks.yaml")
runner = lb.BenchmarkRunner(tasks)
```

Example YAML task file:

```yaml
- task_id: my_q1
  category: qa
  prompt: What is the boiling point of water?
  reference: 100 degrees Celsius

- task_id: my_code1
  category: coding
  prompt: Write a Python one-liner to square every element of list x.
  reference: "[i**2 for i in x]"
```

### GUI

```
python llmbench_gui.py
# or, if installed:
llmbench-gui
```

The GUI provides a Configuration tab (inference mode, API settings, task file),
a Results tab (sortable per-task scores, colour-coded rows), and a Summary tab
(JSON aggregate statistics).  Results can be exported to CSV or JSONL directly
from the toolbar.

## Metrics

| Metric | Description |
|--------|-------------|
| `exact_match` | 1.0 if prediction equals reference (whitespace-stripped) |
| `exact_match_norm` | 1.0 after lowercasing and alphanumeric tokenising |
| `rouge_l` | ROUGE-L F1 (longest common subsequence of tokens) |
| `bleu_1` | Unigram BLEU with brevity penalty |
| `f1` | Token-level F1 using frequency-weighted intersection (SQuAD style) |
| `composite` | 0.4·ROUGE-L + 0.3·F1 + 0.2·EM-norm + 0.1·BLEU-1 |

## Reproducibility

Each `export_jsonl()` call writes a sidecar `<name>.sha256` file containing the
SHA-256 digest of the JSONL content, so consumers can verify the file has not
been modified.

The `run_metadata` block in every summary records:
- `run_id` — UUID for the run
- `model`, `temperature`, `max_tokens`, `seed` — inference parameters
- `base_url` — endpoint URL
- `timestamp` — ISO 8601 UTC timestamp

## CLI reference

```
llmbench run   [--api-key KEY] [--model MODEL] [--base-url URL]
               [--tasks FILE] [--output FILE] [--csv FILE]
               [--max-tokens N] [--temperature T] [--seed N]

llmbench score PREDICTIONS_JSONL [--format text|json]

llmbench sample
```

## Running tests

```
pip install pytest
pytest tests/
```

## License

MIT — see [LICENSE](LICENSE).
