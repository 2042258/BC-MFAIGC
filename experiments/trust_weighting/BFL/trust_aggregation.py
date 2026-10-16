"""Legacy function names; trust-weight computation has one shared implementation."""

from bcmfaigc.aggregation.trust_weighting import (
    compute_validation_trust as compute_trust_metrics,
    median_trust_weight as combine_validator_weights,
)

__all__ = ["compute_trust_metrics", "combine_validator_weights"]
