"""Evaluation metrics for LLM benchmark outputs. Stdlib-only, no external deps."""
from __future__ import annotations

import math
import re
from typing import Dict, List


def _tokenise(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def exact_match(prediction: str, reference: str) -> float:
    """Return 1.0 iff prediction and reference are identical after stripping whitespace."""
    return 1.0 if prediction.strip() == reference.strip() else 0.0


def exact_match_normalised(prediction: str, reference: str) -> float:
    """Return 1.0 iff lowercased, punctuation-stripped token sequences are equal."""
    return 1.0 if " ".join(_tokenise(prediction)) == " ".join(_tokenise(reference)) else 0.0


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 computed from the longest common subsequence of tokens."""
    pred_tokens = _tokenise(prediction)
    ref_tokens = _tokenise(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec = lcs / m
    rec = lcs / n
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def bleu_1(prediction: str, reference: str) -> float:
    """BLEU-1 (unigram precision) with brevity penalty."""
    pred_tokens = _tokenise(prediction)
    ref_tokens = _tokenise(reference)
    if not pred_tokens:
        return 0.0
    ref_counts: Dict[str, int] = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    pred_counts: Dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    clipped = sum(min(c, ref_counts.get(t, 0)) for t, c in pred_counts.items())
    prec = clipped / len(pred_tokens)
    bp = 1.0 if len(pred_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(pred_tokens))
    return bp * prec


def f1_score(prediction: str, reference: str) -> float:
    """Token-set F1 (unique unordered tokens)."""
    pred_set = set(_tokenise(prediction))
    ref_set = set(_tokenise(reference))
    if not pred_set or not ref_set:
        return 0.0
    common = pred_set & ref_set
    if not common:
        return 0.0
    prec = len(common) / len(pred_set)
    rec = len(common) / len(ref_set)
    return 2 * prec * rec / (prec + rec)


def approx_tokens(text: str) -> int:
    """Approximate token count via alphanumeric splitting."""
    return len(_tokenise(text))


def contains_code(text: str) -> bool:
    """Heuristic: True if text appears to contain a code block or statement."""
    return bool(re.search(r"```|def |class |import |function |return |\{\}|;", text))
