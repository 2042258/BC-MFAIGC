"""Update-aggregation primitives; voting and acceptance policies stay separate."""

from .trust_weighting import compute_validation_trust, median_trust_weight

__all__ = ["compute_validation_trust", "median_trust_weight"]
