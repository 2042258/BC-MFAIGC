"""Old/new update-validation contracts, including transaction-visible effects."""

import copy
from contextlib import redirect_stdout
import importlib.util
import io
import itertools
import json
from pathlib import Path
from types import MethodType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_local_validation_support", ROOT / "tests/test_local_training.py"
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


@unittest.skipUnless(
    helpers.HAS_DEPS, "Update validation needs PyTorch and PyCryptodome"
)
class UpdateValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.support_type = helpers.LocalTrainingTests
        cls.support_type.setUpClass()
        legacy = helpers.load_module(
            "_legacy_update_validation",
            ROOT / "tests/fixtures/legacy_update_validation.py",
        ).LegacyUpdateValidation
        cls.references = {
            name: type(
                "OriginalValidation_" + name,
                (legacy, cls.support_type.reference, candidate),
                {},
            )
            for name, candidate in cls.support_type.candidates.items()
        }

    def setUp(self):
        self.support = self.support_type()
        self.support.setUp()

    def tearDown(self):
        self.support.tearDown()

    def node(self, node_type):
        node = self.support.node(node_type)
        node.global_parameters = copy.deepcopy(node.net.state_dict())
        node.global_parameters["scale"].fill_(0.7)
        node.idx = "validator"
        node.computation_power = 2
        node.black_list = set()
        node.check_signature = False
        node.validator_local_accuracy = 1.0  # Historical field stores DDPM loss.
        node.validator_threshold = 0.25
        node.defense_on = True
        node.is_malicious = False
        node.aggregation_method = "hard_filter"
        node.trust_center, node.trust_scale = 0.0, 1.0
        node.trust_soft_threshold, node.trust_beta = 0.0, 1.0
        node.trust_min_weight, node.trust_hard_reject_score = 0.05, 5.0
        node.devices_dict = {
            "worker": SimpleNamespace(return_is_malicious=lambda: False)
        }
        node.modulus, node.public_key = 2**300, 1
        return node

    def canonical(self, value):
        def tensor(item):
            if isinstance(item, self.support.torch.Tensor):
                return {
                    "dtype": str(item.dtype),
                    "shape": list(item.shape),
                    "values": item.tolist(),
                }
            raise TypeError(type(item).__name__)

        return json.dumps(value, sort_keys=True, default=tensor)

    def test_parameter_evaluation_restores_parameters_modes_and_rng(self):
        for variant, candidate in self.support.candidates.items():
            for supplied in (False, True):
                for training in (False, True):
                    outputs = []
                    for node_type in (self.references[variant], candidate):
                        node = self.node(node_type)
                        node.net.train(training)
                        node.net.child.train(not training)
                        original = copy.deepcopy(node.net.state_dict())
                        self.support.seed()
                        loss = node.validate_model_weights(
                            node.global_parameters if supplied else None
                        )
                        self.support.assert_tree(original, node.net.state_dict())
                        outputs.append(
                            (
                                loss,
                                [module.training for module in node.net.modules()],
                                self.support.rng(),
                            )
                        )
                    with self.subTest(
                        variant=variant, supplied=supplied, training=training
                    ):
                        self.assertEqual(outputs[0][:2], outputs[1][:2])
                        self.support.assert_rng(outputs[0][2], outputs[1][2])

    def test_parameter_load_and_evaluator_failures_restore_original_state(self):
        for variant, candidate in self.support.candidates.items():
            for failure in ("wrong_shape", "missing_key", "evaluation"):
                outputs = []
                for node_type in (self.references[variant], candidate):
                    node = self.node(node_type)
                    original = copy.deepcopy(node.net.state_dict())
                    weights = copy.deepcopy(node.global_parameters)
                    if failure == "wrong_shape":
                        weights["scale"] = self.support.torch.ones(2, 2)
                    elif failure == "missing_key":
                        del weights["scale"]
                    else:
                        node.net.fail = True
                    self.support.seed()
                    try:
                        node.validate_model_weights(weights)
                    except (ValueError, RuntimeError) as error:
                        kind = type(error).__name__
                    else:
                        self.fail("Expected characterization failure")
                    self.support.assert_tree(original, node.net.state_dict())
                    outputs.append((kind, node.net.training, self.support.rng()))
                with self.subTest(variant=variant, failure=failure):
                    self.assertEqual(outputs[0][:2], outputs[1][:2])
                    self.support.assert_rng(outputs[0][2], outputs[1][2])

    def test_reference_training_optimizer_argument_timings_and_node_mutations(self):
        for variant, candidate in self.support.candidates.items():
            for optimizer in ("Adam", "SGD", "unsupported"):
                for epochs in (0, 2):
                    outputs = []
                    for node_type in (self.references[variant], candidate):
                        node = self.node(node_type)
                        node.optimizer_name = "SGD" if optimizer == "Adam" else "Adam"
                        original = copy.deepcopy(node.net.state_dict())
                        function = node.validator_update_model_by_one_epoch_and_validate_local_accuracy
                        clock = itertools.count(100)
                        self.support.seed()
                        with (
                            patch.dict(
                                function.__func__.__globals__,
                                {"time": SimpleNamespace(time=lambda: next(clock))},
                            ),
                            redirect_stdout(io.StringIO()) as stdout,
                        ):
                            try:
                                returned = function(optimizer, local_epochs=epochs)
                            except ValueError as error:
                                returned = str(error)
                        self.support.assert_tree(original, node.net.state_dict())
                        outputs.append(
                            (
                                returned,
                                node.validator_local_accuracy,
                                getattr(node, "validator_training_time", None),
                                getattr(node, "validator_evaluation_time", None),
                                stdout.getvalue(),
                                self.support.rng(),
                            )
                        )
                    with self.subTest(
                        variant=variant, optimizer=optimizer, epochs=epochs
                    ):
                        self.assertEqual(outputs[0][:5], outputs[1][:5])
                        self.support.assert_rng(outputs[0][5], outputs[1][5])

    def transaction_case(self, node_type, case):
        from bcmfaigc.blockchain.serialization import stable_digest

        node = self.node(node_type)
        node.computation_power = case.get("power", 2)
        node.black_list = {"worker"} if case.get("blacklisted") else set()
        node.check_signature = case.get("signature", False)
        node.defense_on = case.get("defense", True)
        node.aggregation_method = case.get("aggregation", "hard_filter")
        node.is_malicious = case.get("flip", False)
        node.devices_dict["worker"] = SimpleNamespace(
            return_is_malicious=lambda: case.get("worker_malicious", False)
        )
        trace = []

        def evaluate(self, weights):
            trace.append(("evaluate", copy.deepcopy(weights)))
            if case.get("evaluation_failure"):
                raise ValueError("controlled score failure")
            return case.get("loss", 1.25)

        def sign(self, message):
            trace.append(("sign", copy.deepcopy(message)))
            return stable_digest(message).hex()

        node.validate_model_weights = MethodType(evaluate, node)
        node.sign_msg = MethodType(sign, node)
        transaction = {
            "worker_device_idx": "worker",
            "local_updates_params": copy.deepcopy(node.global_parameters),
        }
        if node.check_signature:
            # Identity exponent tests verification branching, NOT RSA security.
            transaction["worker_rsa_pub_key"] = {"modulus": 2**300, "pub_key": 1}
            transaction["worker_signature"] = int.from_bytes(
                stable_digest(sorted(transaction.items())), "big"
            )
            if case.get("bad_signature"):
                transaction["worker_signature"] += 1
        clock = itertools.count(100)
        self.support.seed()
        with tempfile.TemporaryDirectory(prefix="update-validation-") as tmp:
            root = Path(tmp)
            (root / "comm_1").mkdir()
            function = node.validate_worker_transaction
            with (
                patch.dict(
                    function.__func__.__globals__,
                    {"time": SimpleNamespace(time=lambda: next(clock))},
                ),
                redirect_stdout(io.StringIO()) as stdout,
            ):
                try:
                    returned = function(transaction, 7, str(root), 1, True)
                except ValueError as error:
                    returned = {"exception": str(error)}
            logs = {
                p.relative_to(root).as_posix(): p.read_text()
                for p in sorted(root.rglob("*"))
                if p.is_file()
            }
        return self.canonical(
            (returned, transaction, trace, logs, stdout.getvalue())
        ), self.support.rng()

    def test_transaction_scores_votes_eligibility_logs_rewards_and_signing_match(self):
        cases = [
            {},
            {"loss": 1.250001},
            {"loss": 0.5},
            {"loss": float("nan")},
            {"loss": 6.5, "aggregation": "trust_weighted"},
            {"loss": 1.5, "aggregation": "trust_weighted"},
            {"loss": 1.5, "defense": False},
            {"loss": 1.5, "flip": True},
            {"loss": 0.5, "flip": True, "aggregation": "trust_weighted"},
            {"worker_malicious": True},
            {"worker_malicious": True, "loss": 1.5},
            {"signature": True},
        ]
        for variant, candidate in self.support.candidates.items():
            for case in cases:
                before = self.transaction_case(self.references[variant], case)
                after = self.transaction_case(candidate, case)
                with self.subTest(variant=variant, case=case):
                    self.assertEqual(before[0], after[0])
                    self.support.assert_rng(before[1], after[1])

    def test_transaction_rejection_and_partial_failure_boundaries_match(self):
        cases = [
            {"power": 0},
            {"blacklisted": True},
            {"signature": True, "bad_signature": True},
            {"evaluation_failure": True},
        ]
        for variant, candidate in self.support.candidates.items():
            for case in cases:
                before = self.transaction_case(self.references[variant], case)
                after = self.transaction_case(candidate, case)
                with self.subTest(variant=variant, case=case):
                    self.assertEqual(before[0], after[0])
                    self.support.assert_rng(before[1], after[1])


if __name__ == "__main__":
    unittest.main()
