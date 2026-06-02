"""
Evaluation Metrics
==================
Token-level F1 (HotpotQA standard) and exact-match accuracy (MetaQA standard),
plus bootstrap confidence interval for significance testing.
"""

import re
import string
import numpy as np
from collections import Counter


# ---------------------------------------------------------------------------
# Text normalisation (standard HotpotQA / SQuAD preprocessing)
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


# ---------------------------------------------------------------------------
# Token-level F1 (HotpotQA)
# ---------------------------------------------------------------------------

def compute_f1(prediction: str, gold: str) -> float:
    """
    Token-level F1 between prediction and gold answer strings.
    Standard metric for HotpotQA (distractor setting).
    """
    pred_tokens = _normalise(prediction).split()
    gold_tokens = _normalise(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())

    if n_common == 0:
        return 0.0

    precision = n_common / len(pred_tokens)
    recall    = n_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ---------------------------------------------------------------------------
# Exact match (MetaQA)
# ---------------------------------------------------------------------------

def compute_exact_match(prediction: str, gold: str) -> float:
    """
    Exact-match accuracy after normalisation.
    Standard metric for MetaQA-3hop.
    Returns 1.0 if match, 0.0 otherwise.
    """
    return float(_normalise(prediction) == _normalise(gold))


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_confidence_interval(
    scores: list,
    n: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Non-parametric bootstrap confidence interval for mean score.

    Parameters
    ----------
    scores     : per-example scores (F1 or exact-match)
    n          : number of resampling iterations (paper uses 1,000)
    confidence : confidence level (default 0.95)
    seed       : random seed for reproducibility

    Returns
    -------
    (ci_low, ci_high) : lower and upper bounds of the confidence interval
    """
    rng = np.random.default_rng(seed)
    scores_arr = np.array(scores)
    bootstrap_means = np.array([
        rng.choice(scores_arr, size=len(scores_arr), replace=True).mean()
        for _ in range(n)
    ])
    alpha = 1.0 - confidence
    ci_low  = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_high = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    return ci_low, ci_high


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pred = "lactic acidosis"
    gold = "Lactic Acidosis"
    print(f"F1  ('{pred}' vs '{gold}'): {compute_f1(pred, gold):.4f}")  # expect 1.0
    print(f"EM  ('{pred}' vs '{gold}'): {compute_exact_match(pred, gold):.4f}")  # expect 1.0

    pred2 = "kidney failure"
    print(f"F1  ('{pred2}' vs '{gold}'): {compute_f1(pred2, gold):.4f}")  # expect 0.0
    print(f"EM  ('{pred2}' vs '{gold}'): {compute_exact_match(pred2, gold):.4f}")  # expect 0.0

    scores = [0.8, 0.6, 0.9, 0.7, 0.75]
    lo, hi = bootstrap_confidence_interval(scores, n=1000)
    print(f"Bootstrap 95% CI for {scores}: [{lo:.4f}, {hi:.4f}]")
