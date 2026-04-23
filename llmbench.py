"""
llmbench: Lightweight, provider-agnostic benchmarking harness for large language models.

Measures latency, throughput (tokens/sec), output quality (ROUGE-L, exact match),
and cost estimation across OpenAI, Anthropic, and any OpenAI-compatible endpoint.
"""
from __future__ import annotations
import json, time, statistics, datetime, re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class BenchTask:
    """A single benchmark task."""
    task_id: str
    prompt: str
    reference: str = ""
    category: str = "general"
    max_tokens: int = 256
    temperature: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchResult:
    """Result for a single (task, model) pair."""
    task_id: str
    model: str
    provider: str
    prompt: str
    response: str
    reference: str
    latency_s: float
    ttft_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    exact_match: bool
    rouge_l: float
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _lcs_length(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def rouge_l(hypothesis: str, reference: str) -> float:
    """Sentence-level ROUGE-L F1 score."""
    if not hypothesis or not reference:
        return 0.0
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    lcs = _lcs_length(hyp_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(hypothesis: str, reference: str) -> bool:
    """Normalised exact match."""
    def _norm(s: str) -> str:
        return re.sub(r"[^\w\s]", "", s.lower()).strip()
    return _norm(hypothesis) == _norm(reference)


# Cost table: (input_per_1k_usd, output_per_1k_usd) — approximate 2024-Q4
_COST_TABLE: Dict[str, Tuple[float, float]] = {
    "gpt-4o":                    (0.005,   0.015),
    "gpt-4o-mini":               (0.00015, 0.0006),
    "gpt-4-turbo":               (0.01,    0.03),
    "gpt-3.5-turbo":             (0.0005,  0.0015),
    "claude-3-5-sonnet-20241022":(0.003,   0.015),
    "claude-3-haiku-20240307":   (0.00025, 0.00125),
    "claude-3-opus-20240229":    (0.015,   0.075),
    "command-r":                 (0.0005,  0.0015),
    "command-r-plus":            (0.003,   0.015),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    model_key = model.lower()
    for key, (inp, out) in _COST_TABLE.items():
        if key in model_key:
            return (prompt_tokens * inp + completion_tokens * out) / 1000
    return (prompt_tokens * 0.001 + completion_tokens * 0.002) / 1000


class _BaseProvider:
    name: str = "base"

    def complete(self, prompt: str, model: str, max_tokens: int,
                 temperature: float) -> Tuple[str, int, int]:
        raise NotImplementedError


class OpenAIProvider(_BaseProvider):
    """Adapter for OpenAI-compatible APIs."""
    name = "openai"

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def complete(self, prompt, model, max_tokens, temperature):
        import urllib.request
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode()
        url = self.base_url + "/chat/completions"
        req = urllib.request.Request(url, data=payload, headers={
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        })
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        p_tok = usage.get("prompt_tokens", _approx_tokens(prompt))
        c_tok = usage.get("completion_tokens", _approx_tokens(text))
        return text, p_tok, c_tok


class AnthropicProvider(_BaseProvider):
    """Adapter for Anthropic Messages API."""
    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def complete(self, prompt, model, max_tokens, temperature):
        import urllib.request
        payload = json.dumps({
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        text = data["content"][0]["text"]
        usage = data.get("usage", {})
        p_tok = usage.get("input_tokens", _approx_tokens(prompt))
        c_tok = usage.get("output_tokens", _approx_tokens(text))
        return text, p_tok, c_tok


class BenchmarkRunner:
    """Run a suite of tasks against one or more (provider, model) pairs."""

    def __init__(self, tasks: List[BenchTask], output_path: Optional[str] = None) -> None:
        self.tasks = tasks
        self.output_path = Path(output_path) if output_path else None
        self._results: List[BenchResult] = []

    def run_one(self, task: BenchTask, provider: _BaseProvider, model: str) -> BenchResult:
        t0 = time.perf_counter()
        error = ""
        response = ""
        prompt_tok = completion_tok = 0
        try:
            response, prompt_tok, completion_tok = provider.complete(
                task.prompt, model, task.max_tokens, task.temperature
            )
        except Exception as exc:
            error = str(exc)
        latency = time.perf_counter() - t0
        cost = estimate_cost(model, prompt_tok, completion_tok)
        result = BenchResult(
            task_id=task.task_id, model=model, provider=provider.name,
            prompt=task.prompt, response=response, reference=task.reference,
            latency_s=round(latency, 4), ttft_s=0.0,
            prompt_tokens=prompt_tok, completion_tokens=completion_tok,
            total_tokens=prompt_tok + completion_tok,
            cost_usd=round(cost, 8),
            exact_match=exact_match(response, task.reference),
            rouge_l=round(rouge_l(response, task.reference), 4),
            error=error,
        )
        if self.output_path:
            with self.output_path.open("a") as f:
                f.write(json.dumps(result.to_dict()) + "\n")
        return result

    def run(self, providers_models: List[Tuple[_BaseProvider, str]]) -> List[BenchResult]:
        """Run all tasks with all (provider, model) pairs."""
        self._results = []
        for task in self.tasks:
            for provider, model in providers_models:
                self._results.append(self.run_one(task, provider, model))
        return self._results

    def summary(self) -> Dict[str, Any]:
        if not self._results:
            return {}
        by_model: Dict[str, List[BenchResult]] = {}
        for r in self._results:
            by_model.setdefault(r.model, []).append(r)
        out = {}
        for model, results in by_model.items():
            valid = [r for r in results if not r.error]
            out[model] = {
                "n": len(results),
                "errors": len(results) - len(valid),
                "avg_latency_s": round(statistics.mean(r.latency_s for r in valid), 4) if valid else None,
                "p50_latency_s": round(statistics.median(r.latency_s for r in valid), 4) if valid else None,
                "avg_tokens": round(statistics.mean(r.total_tokens for r in valid), 1) if valid else None,
                "total_cost_usd": round(sum(r.cost_usd for r in valid), 6),
                "exact_match_rate": round(sum(r.exact_match for r in valid) / max(len(valid), 1), 4),
                "avg_rouge_l": round(statistics.mean(r.rouge_l for r in valid), 4) if valid else None,
            }
        return out

    def to_markdown(self) -> str:
        summ = self.summary()
        lines = [
            "# LLMBench Results", "",
            "Tasks: " + str(len(self.tasks)) + "  ",
            "Timestamp: " + datetime.datetime.utcnow().isoformat() + "Z", "",
            "| Model | N | Errors | Avg Latency (s) | Avg Tokens | Cost (USD) | EM Rate | ROUGE-L |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for model, s in summ.items():
            lines.append(
                "| " + model + " | " + str(s["n"]) + " | " + str(s["errors"]) + " | " +
                str(s["avg_latency_s"]) + " | " + str(s["avg_tokens"]) + " | " +
                "{:.6f}".format(s["total_cost_usd"]) + " | " +
                "{:.2%}".format(s["exact_match_rate"]) + " | " +
                str(s["avg_rouge_l"]) + " |"
            )
        return "\n".join(lines)


def load_tasks_jsonl(path: str) -> List[BenchTask]:
    """Load benchmark tasks from a JSONL file."""
    tasks = []
    fields = BenchTask.__dataclass_fields__
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                tasks.append(BenchTask(**{k: v for k, v in d.items() if k in fields}))
    return tasks


def _cli() -> None:
    import argparse, os
    parser = argparse.ArgumentParser(
        prog="llmbench",
        description="Benchmark LLMs for latency, quality, and cost.",
    )
    parser.add_argument("tasks_file", help="JSONL file of BenchTask records.")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("-o", "--output", default="bench_results.jsonl")
    args = parser.parse_args()

    api_key = (args.api_key or os.environ.get("OPENAI_API_KEY")
               or os.environ.get("ANTHROPIC_API_KEY", ""))
    provider: _BaseProvider
    if args.provider == "openai":
        provider = OpenAIProvider(api_key=api_key, base_url=args.base_url)
    else:
        provider = AnthropicProvider(api_key=api_key)

    tasks = load_tasks_jsonl(args.tasks_file)
    print("Loaded " + str(len(tasks)) + " tasks. Running with " + args.model + "...")
    runner = BenchmarkRunner(tasks, output_path=args.output)
    runner.run([(provider, args.model)])
    print(runner.to_markdown())


if __name__ == "__main__":
    _cli()
