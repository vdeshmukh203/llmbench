"""Evaluation metrics for LLM benchmark scoring.

All metrics operate on plain strings and return a float in [0, 1].
No external dependencies are required.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List


def _tokenise(text: str) -> List[str]:
    """Lowercase alphanumeric tokeniser."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def exact_match(prediction: str, reference: str) -> float:
    """Return 1.0 when *prediction* and *reference* are identical after stripping."""
    return 1.0 if prediction.strip() == reference.strip() else 0.0


def exact_match_normalised(prediction: str, reference: str) -> float:
    """Return 1.0 when token sequences match after lowercasing and tokenisation."""
    return 1.0 if " ".join(_tokenise(prediction)) == " ".join(_tokenise(reference)) else 0.0


def rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 via longest common subsequence over tokens.

    References
    ----------
    Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries.
    *ACL Workshop on Text Summarization Branches Out*, pp. 74–81.
    """
    pred_tok = _tokenise(prediction)
    ref_tok = _tokenise(reference)
    if not pred_tok or not ref_tok:
        return 0.0
    m, n = len(pred_tok), len(ref_tok)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tok[i - 1] == ref_tok[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec = lcs / m
    rec = lcs / n
    return 2.0 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def bleu_1(prediction: str, reference: str) -> float:
    """Compute unigram BLEU-1 with brevity penalty.

    References
    ----------
    Papineni et al. (2002). BLEU: a Method for Automatic Evaluation of
    Machine Translation. *ACL 2002*, pp. 311–318.
    """
    pred_tok = _tokenise(prediction)
    ref_tok = _tokenise(reference)
    if not pred_tok:
        return 0.0
    ref_counts: Dict[str, int] = {}
    for t in ref_tok:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    pred_counts: Dict[str, int] = {}
    for t in pred_tok:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    clipped = sum(min(c, ref_counts.get(t, 0)) for t, c in pred_counts.items())
    precision = clipped / len(pred_tok)
    bp = 1.0 if len(pred_tok) >= len(ref_tok) else math.exp(1.0 - len(ref_tok) / len(pred_tok))
    return bp * precision


def f1_score(prediction: str, reference: str) -> float:
    """Compute token-level F1 using bag-of-words set overlap."""
    pred_set = set(_tokenise(prediction))
    ref_set = set(_tokenise(reference))
    if not pred_set or not ref_set:
        return 0.0
    common = pred_set & ref_set
    if not common:
        return 0.0
    prec = len(common) / len(pred_set)
    rec = len(common) / len(ref_set)
    return 2.0 * prec * rec / (prec + rec)


def _approx_tokens(text: str) -> int:
    """Approximate token count via alphanumeric splitting."""
    return len(_tokenise(text))


def contains_code(text: str) -> bool:
    """Heuristically detect whether *text* contains source code."""
    return bool(re.search(r"```|def |class |import |function |return |\{\}|;", text))
