"""Evaluation metrics for LLM benchmark scoring."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List


def _tokenise(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def exact_match(prediction: str, reference: str) -> float:
    """Return 1.0 if prediction exactly equals reference (after strip)."""
    return 1.0 if prediction.strip() == reference.strip() else 0.0


def exact_match_normalised(prediction: str, reference: str) -> float:
    """Return 1.0 if lowercased, tokenised prediction matches reference."""
    return 1.0 if " ".join(_tokenise(prediction)) == " ".join(_tokenise(reference)) else 0.0


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 via longest common subsequence (token-level)."""
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
    """BLEU-1: clipped unigram precision with brevity penalty."""
    pred_tokens = _tokenise(prediction)
    ref_tokens = _tokenise(reference)
    if not pred_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    pred_counts = Counter(pred_tokens)
    clipped = sum(min(c, ref_counts.get(t, 0)) for t, c in pred_counts.items())
    prec = clipped / len(pred_tokens)
    bp = (
        1.0
        if len(pred_tokens) >= len(ref_tokens)
        else math.exp(1 - len(ref_tokens) / len(pred_tokens))
    )
    return bp * prec


def f1_score(prediction: str, reference: str) -> float:
    """Token-level F1 score using token counts (SQuAD-style)."""
    pred_tokens = _tokenise(prediction)
    ref_tokens = _tokenise(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    common = sum((pred_counts & ref_counts).values())
    if common == 0:
        return 0.0
    prec = common / len(pred_tokens)
    rec = common / len(ref_tokens)
    return 2 * prec * rec / (prec + rec)


def approx_tokens(text: str) -> int:
    """Approximate token count using alphanumeric tokenisation."""
    return len(_tokenise(text))


def contains_code(text: str) -> bool:
    """Heuristic check: does the text appear to contain code?"""
    return bool(re.search(r"```|def |class |import |function |return |\{\}|;", text))
