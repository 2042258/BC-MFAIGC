"""Convert validation scores to trust weights and combine validator weights.

These pure numerical helpers do not perform voting, decide whether an update is
accepted, or select a Krum update. Hard/soft policy decisions remain separate.
"""

import math
from statistics import median


def compute_validation_trust(
    raw_score,
    center=0.0,
    scale=1.0,
    soft_threshold=0.0,
    beta=1.0,
    min_weight=0.05,
):
    """Return a normalized validation score and a bounded trust weight.

    A lower score means that the worker update does not increase the
    validator's loss.  The sigmoid therefore gives such updates larger
    weights, while still allowing updates close to the soft threshold to
    contribute instead of forcing an all-or-nothing decision.
    """
    raw_score = float(raw_score)
    center = float(center)
    scale = max(abs(float(scale)), 1e-12)
    soft_threshold = float(soft_threshold)
    beta = max(float(beta), 0.0)
    min_weight = min(max(float(min_weight), 0.0), 1.0)

    if not math.isfinite(raw_score):
        normalized_score = math.inf
        return normalized_score, min_weight

    normalized_score = (raw_score - center) / scale
    if beta == 0.0:
        sigmoid = 0.5
    else:
        exponent = max(min(beta * (normalized_score - soft_threshold), 60.0), -60.0)
        sigmoid = 1.0 / (1.0 + math.exp(exponent))
    aggregation_weight = min_weight + (1.0 - min_weight) * sigmoid
    return normalized_score, aggregation_weight


def median_trust_weight(values, default=1.0):
    """Combine validator weights robustly using the median."""
    finite_values = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0.0:
            finite_values.append(value)
    if not finite_values:
        return float(default)
    return float(median(finite_values))
