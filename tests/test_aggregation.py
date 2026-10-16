"""Distinct direct/block epoch rules and ordered weighted accumulation."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "Aggregation needs PyTorch")
class AggregationTests(unittest.TestCase):
    def test_weighted_sum_order_integer_buffers_and_input_aliases(self):
        import torch
        from bcmfaigc.aggregation.reduction import sum_weighted_parameters

        a = {"w": torch.tensor([2.0]), "n": torch.tensor(3), "b": torch.tensor(True)}
        b = {"w": torch.tensor([4.0]), "n": torch.tensor(9), "b": torch.tensor(False)}
        result, total = sum_weighted_parameters([(a, 0.3), (b, 0.7)])
        self.assertEqual(total, 1.0)
        self.assertTrue(torch.equal(result["w"], a["w"] * 0.3 + b["w"] * 0.7))
        self.assertEqual(result["n"].item(), 3)
        self.assertTrue(result["b"].item())
        self.assertIsNot(result["w"], a["w"])
        self.assertEqual(a["w"].item(), 2.0)

    def test_direct_first_transaction_and_security_majority(self):
        from bcmfaigc.aggregation.decisions import collect_direct_decisions

        def tx(value, direction, eligible=None):
            item = {
                "worker_device_idx": "w",
                "local_total_accumulated_epochs_this_round": 1,
                "local_updates_params": {"v": value},
                "update_direction": direction,
                "aggregation_weight": 0.5,
            }
            if eligible is not None:
                item["aggregation_eligible"] = eligible
            return item

        first = tx(1, True)
        votes, params, weights = collect_direct_decisions(
            [first, tx(2, False)],
            "hard_filter",
            lambda t: t.get("aggregation_weight", 1.0),
        )
        self.assertIs(params["w"], first["local_updates_params"])
        self.assertEqual(weights["w"], 0.5)
        _, params, _ = collect_direct_decisions(
            [first, tx(2, False), tx(3, False)], "hard_filter", lambda t: 0.5
        )
        self.assertNotIn("w", params)

    def test_direct_soft_eligibility_is_not_security_vote(self):
        from bcmfaigc.aggregation.decisions import collect_direct_decisions

        first = {
            "worker_device_idx": "w",
            "local_total_accumulated_epochs_this_round": 2,
            "local_updates_params": {"v": 1},
            "update_direction": False,
            "aggregation_eligible": True,
            "aggregation_weight": 0.2,
        }
        second = {**first, "local_updates_params": {"v": 2}, "aggregation_weight": 0.8}
        votes, params, weights = collect_direct_decisions(
            [first, second], "trust_weighted", lambda t: t["aggregation_weight"]
        )
        self.assertIs(params["w"], first["local_updates_params"])
        self.assertEqual(weights["w"], 0.5)
        self.assertEqual(votes[("w", 2)]["negative"], 2)

    def test_block_equal_epoch_last_overwrites_and_later_rejection_does_not_rewind(
        self,
    ):
        from bcmfaigc.aggregation.decisions import accumulate_block_decision

        records = {}

        def tx(epoch, value, positive=True):
            return {
                "local_total_accumulated_epochs_this_round": epoch,
                "local_updates_params": {"v": value},
                "local_updates_rewards": 1.0,
                "positive_direction_validators": [{}] if positive else [],
                "negative_direction_validators": [] if positive else [{}],
            }

        for item in [tx(2, 1), tx(2, 2), tx(3, 3, False), tx(1, 4)]:
            accumulate_block_decision(
                records, item, "w", "hard_filter", 0.0, "observer", 0.0, lambda t: 1.0
            )
        # Undiscounted rejection does not enter all_valid_epochs; latest positive stays 2.
        self.assertEqual(records["w"]["finally_used_params"], {"v": 2})
        self.assertEqual(records["w"]["all_valid_epochs"], {1, 2})
        self.assertEqual(records["w"]["negative_epochs"], {3})

    def test_discounted_block_parameters_and_ordered_reward_total(self):
        import torch
        from bcmfaigc.aggregation.decisions import accumulate_block_decision

        params = {"v": torch.tensor([4.0])}
        transaction = {
            "local_total_accumulated_epochs_this_round": 1,
            "local_updates_params": params,
            "local_updates_rewards": 8.0,
            "positive_direction_validators": [],
            "negative_direction_validators": [{}],
            "aggregation_weight": 0.4,
        }
        records = {}
        total, _, _ = accumulate_block_decision(
            records,
            transaction,
            "w",
            "hard_filter",
            0.25,
            "w",
            7.0,
            lambda t: t["aggregation_weight"],
        )
        self.assertEqual(total, 9.0)
        self.assertEqual(records["w"]["finally_used_params"]["v"].item(), 1.0)
        self.assertEqual(records["w"]["finally_used_weight"], 0.1)
        self.assertEqual(params["v"].item(), 4.0)


if __name__ == "__main__":
    unittest.main()
