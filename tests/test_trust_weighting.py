"""Trust weights are numerical outputs, not votes or acceptance decisions."""

import inspect
import math
from pathlib import Path
import random
import runpy
import unittest

from bcmfaigc.aggregation import compute_validation_trust, median_trust_weight

ROOT = Path(__file__).resolve().parents[1]


class TrustWeightingTests(unittest.TestCase):
    def test_default_signature_remains_compatible_with_legacy_keywords(self):
        parameters = inspect.signature(compute_validation_trust).parameters
        self.assertEqual(
            list(parameters),
            ["raw_score", "center", "scale", "soft_threshold", "beta", "min_weight"],
        )
        for name, value in {
            "center": 0.0,
            "scale": 1.0,
            "soft_threshold": 0.0,
            "beta": 1.0,
            "min_weight": 0.05,
        }.items():
            self.assertEqual(parameters[name].default, value)
        self.assertEqual(compute_validation_trust(raw_score=0.0), (0.0, 0.525))

    def test_center_scale_and_monotonic_weighting(self):
        normalized, _ = compute_validation_trust(1.8, center=1.2, scale=0.2)
        self.assertAlmostEqual(normalized, 3.0)
        weights = [compute_validation_trust(value)[1] for value in (-10, 0, 10)]
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])
        self.assertTrue(all(0.05 <= value <= 1.0 for value in weights))

    def test_nonfinite_scores_keep_original_rejection_weight(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                self.assertEqual(
                    compute_validation_trust(value, min_weight=0.2), (math.inf, 0.2)
                )

    def test_parameter_conversion_still_precedes_nonfinite_fallback(self):
        with self.assertRaises(ValueError):
            compute_validation_trust(math.nan, center="invalid")
        with self.assertRaises(TypeError):
            compute_validation_trust(None)

    def test_scale_floor_and_negative_scale_behavior(self):
        self.assertEqual(
            compute_validation_trust(2, scale=-3), compute_validation_trust(2, scale=3)
        )
        self.assertEqual(
            compute_validation_trust(2, scale=0),
            compute_validation_trust(2, scale=1e-12),
        )

    def test_beta_and_minimum_weight_clamps_are_unchanged(self):
        self.assertEqual(
            compute_validation_trust(20, beta=-3), compute_validation_trust(20, beta=0)
        )
        self.assertEqual(compute_validation_trust(0, min_weight=-1), (0.0, 0.5))
        self.assertEqual(compute_validation_trust(20, min_weight=2)[1], 1.0)

    def test_extreme_scores_do_not_overflow_the_original_exponent_clip(self):
        for value in (-1e300, 1e300):
            normalized, weight = compute_validation_trust(value)
            self.assertEqual(normalized, value)
            self.assertTrue(math.isfinite(weight))
            self.assertGreaterEqual(weight, 0.05)
            self.assertLessEqual(weight, 1.0)

    def test_median_filters_invalid_values_but_does_not_add_an_upper_cap(self):
        self.assertEqual(
            median_trust_weight([None, "bad", -1, math.nan, math.inf, "0.2", 0.6, 1.0]),
            0.6,
        )
        self.assertEqual(median_trust_weight([2, 4]), 3.0)
        self.assertEqual(median_trust_weight([], default="0.7"), 0.7)
        self.assertEqual(median_trust_weight([math.nan, -1], default=0.9), 0.9)

    def test_generator_inputs_are_consumed_once(self):
        values = (value for value in (0.1, 0.9, 0.3))
        self.assertEqual(median_trust_weight(values), 0.3)
        self.assertEqual(list(values), [])

    def test_computation_does_not_consume_python_rng(self):
        before = random.getstate()
        compute_validation_trust(0.3)
        median_trust_weight([0.2, 0.4])
        self.assertEqual(random.getstate(), before)

    def test_all_legacy_modules_expose_the_shared_functions(self):
        for variant in ("stage1", "stage2", "krum", "soft_stage2"):
            with self.subTest(variant=variant):
                namespace = runpy.run_path(
                    str(ROOT / "experiments" / {"stage1":"comparison", "stage2":"byzantine_ratio", "krum":"krum", "soft_stage2":"trust_weighting"}[variant] / "BFL/trust_aggregation.py")
                )
                self.assertIs(
                    namespace["compute_trust_metrics"], compute_validation_trust
                )
                self.assertIs(
                    namespace["combine_validator_weights"], median_trust_weight
                )


if __name__ == "__main__":
    unittest.main()
