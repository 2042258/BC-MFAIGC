"""Characterize Device model operations against frozen original methods.

Tests bypass network construction; no RSA keys, dataset downloads or GPU work.
"""

import copy
import importlib.util
import math
from pathlib import Path
import random
import sys
from types import MethodType
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HAS_DEPS = all(importlib.util.find_spec(name) for name in ("torch", "Crypto"))


def load_module(name, path, extra_paths=()):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [*map(str, extra_paths), *sys.path]):
        spec.loader.exec_module(module)
    return module


@unittest.skipUnless(
    HAS_DEPS, "Local-training characterization needs PyTorch and PyCryptodome"
)
class LocalTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        cls.reference = load_module(
            "_legacy_local_training", ROOT / "tests/fixtures/legacy_local_training.py"
        ).LegacyLocalTraining
        cls.candidates = {}
        for variant in ("stage1", "stage2", "krum", "soft_stage2"):
            directory = {'stage1': 'comparison', 'stage2': 'byzantine_ratio', 'krum': 'krum', 'soft_stage2': 'trust_weighting'}[variant]
            project = ROOT / "experiments" / directory
            cls.candidates[variant] = load_module(
                "_device_" + variant,
                project / "BFL/Device.py",
                (project / "BFL", project),
            ).Device

        class TinyDiffusionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.25))
                self.child = torch.nn.Identity()
                self.timesteps = 5
                self.fail = False

            def q_sample(self, images, timesteps, noise):
                if self.fail:
                    raise ValueError("controlled diffusion failure")
                return images + noise * 0.1 + timesteps.float().view(-1, 1, 1, 1) * 0.01

            def forward(self, images, timesteps):
                return (
                    self.child(images) * self.scale
                    + timesteps.float().view(-1, 1, 1, 1) * 0.001
                )

        cls.model_type = TinyDiffusionModel

    def setUp(self):
        torch = self.torch
        self.original_rng = (
            torch.get_rng_state(),
            random.getstate(),
            np.random.get_state(),
        )
        self.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        # Warm one-time optimizer imports before symmetric in-process comparisons.
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.zero_grad()
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()

    def tearDown(self):
        self.torch.set_rng_state(self.original_rng[0])
        random.setstate(self.original_rng[1])
        np.random.set_state(self.original_rng[2])
        self.torch.set_num_threads(self.original_threads)

    def seed(self):
        self.torch.random.default_generator.manual_seed(86)
        random.seed(86)
        np.random.seed(86)

    def rng(self):
        return (
            self.torch.get_rng_state().clone(),
            random.getstate(),
            np.random.get_state(),
        )

    def assert_rng(self, left, right):
        self.assertTrue(self.torch.equal(left[0], right[0]))
        self.assertEqual(left[1], right[1])
        self.assertEqual(left[2][0], right[2][0])
        np.testing.assert_array_equal(left[2][1], right[2][1])
        self.assertEqual(left[2][2:], right[2][2:])

    def assert_tree(self, left, right):
        if isinstance(left, self.torch.Tensor):
            self.assertTrue(self.torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree(a, b)
        else:
            self.assertEqual(left, right)

    def node(self, node_type, optimizer="Adam", parallel=False, empty=False):
        torch = self.torch
        from torch.utils.data import DataLoader, TensorDataset

        node = object.__new__(node_type)
        node.dev = torch.device("cpu")
        node.optimizer_name, node.learning_rate = optimizer, 0.001
        node.validation_seed = 902
        model = self.model_type()
        if parallel:
            # Construct only DataParallel's CPU fallback form; never discover GPUs.
            wrapper = object.__new__(torch.nn.DataParallel)
            torch.nn.Module.__init__(wrapper)
            wrapper.module = model
            wrapper.device_ids = []
            node.net = wrapper
        else:
            node.net = model
        count = 0 if empty else 9
        images = torch.linspace(0.0, 1.0, max(1, count) * 12).reshape(
            max(1, count), 3, 2, 2
        )[:count]
        labels = torch.zeros(count, dtype=torch.long)
        dataset = TensorDataset(images, labels)
        node.train_dl = DataLoader(dataset, batch_size=2, shuffle=not empty)
        node.test_dl = DataLoader(dataset, batch_size=2, shuffle=False)
        return node

    def test_optimizer_options_and_errors(self):
        for variant, node_type in self.candidates.items():
            for kind in ("SGD", "Adam", "unsupported"):
                reference, candidate = (
                    self.node(self.reference, kind),
                    self.node(node_type, kind),
                )
                with self.subTest(variant=variant, optimizer=kind):
                    if kind == "unsupported":
                        for node in (reference, candidate):
                            with self.assertRaisesRegex(
                                ValueError, "Unsupported optimizer: unsupported"
                            ):
                                node._make_optimizer(node.net)
                    else:
                        self.assert_tree(
                            reference._make_optimizer(reference.net).state_dict(),
                            candidate._make_optimizer(candidate.net).state_dict(),
                        )

    def test_loss_and_gradients_preserve_values_and_rng_with_dataparallel(self):
        for variant, node_type in self.candidates.items():
            for parallel in (False, True):
                outputs = []
                for kind in (self.reference, node_type):
                    node = self.node(kind, parallel=parallel)
                    self.seed()
                    loss = node._ddpm_training_loss(
                        node.net, self.torch.ones(2, 3, 2, 2)
                    )
                    loss.backward()
                    outputs.append(
                        (
                            loss.detach(),
                            [
                                parameter.grad.clone()
                                for parameter in node.net.parameters()
                            ],
                            self.rng(),
                        )
                    )
                with self.subTest(variant=variant, parallel=parallel):
                    self.assert_tree(outputs[0][:2], outputs[1][:2])
                    self.assert_rng(outputs[0][2], outputs[1][2])

    def test_full_and_trailing_accumulation_match_original_epoch(self):
        for variant, node_type in self.candidates.items():
            for optimizer_name in ("Adam", "SGD"):
                outputs = []
                for kind in (self.reference, node_type):
                    node = self.node(kind, optimizer_name)
                    optimizer = node._make_optimizer(node.net)
                    self.seed()
                    returned = node._run_local_epoch(
                        node.net, optimizer, accumulation_steps=4
                    )
                    outputs.append(
                        (
                            copy.deepcopy(node.net.state_dict()),
                            copy.deepcopy(optimizer.state_dict()),
                            node.net.training,
                            returned,
                            self.rng(),
                        )
                    )
                with self.subTest(variant=variant, optimizer=optimizer_name):
                    self.assert_tree(outputs[0][:4], outputs[1][:4])
                    self.assert_rng(outputs[0][4], outputs[1][4])

    def test_empty_epoch_clears_gradients_without_creating_optimizer_state(self):
        for variant, node_type in self.candidates.items():
            outputs = []
            for kind in (self.reference, node_type):
                node = self.node(kind, empty=True)
                node.net.train(False)
                node.net.scale.grad = self.torch.ones_like(node.net.scale)
                optimizer = node._make_optimizer(node.net)
                self.seed()
                returned = node._run_local_epoch(node.net, optimizer)
                outputs.append(
                    (
                        copy.deepcopy(node.net.state_dict()),
                        optimizer.state_dict(),
                        node.net.scale.grad,
                        node.net.training,
                        returned,
                        self.rng(),
                    )
                )
            with self.subTest(variant=variant):
                self.assert_tree(outputs[0][:5], outputs[1][:5])
                self.assertEqual(outputs[1][1]["state"], {})
                self.assert_rng(outputs[0][5], outputs[1][5])

    def test_actual_ddpm_training_evaluation_checkpoints_and_rng_match(self):
        from bcmfaigc.models import DDPM
        from torch.utils.data import DataLoader, TensorDataset

        torch = self.torch
        images = torch.linspace(0.0, 1.0, 3 * 3 * 32 * 32).reshape(3, 3, 32, 32)
        dataset = TensorDataset(images, torch.zeros(3, dtype=torch.long))
        outputs = []
        for kind in (self.reference, self.candidates["stage2"]):
            node = self.node(kind)
            torch.random.default_generator.manual_seed(108)
            node.net = DDPM(timesteps=4)
            node.train_dl = DataLoader(dataset, batch_size=2, shuffle=True)
            node.test_dl = DataLoader(dataset, batch_size=2, shuffle=False)
            optimizer = node._make_optimizer(node.net)
            self.seed()
            node._run_local_epoch(node.net, optimizer)
            loss = node._evaluate_ddpm_loss(node.net)
            outputs.append(
                (
                    copy.deepcopy(node.net.state_dict()),
                    copy.deepcopy(optimizer.state_dict()),
                    loss,
                    node.net.training,
                    self.rng(),
                )
            )
        self.assert_tree(outputs[0][:4], outputs[1][:4])
        self.assert_rng(outputs[0][4], outputs[1][4])

    def test_evaluation_restores_mode_and_keeps_empty_loader_nan(self):
        for variant, node_type in self.candidates.items():
            for training, empty in ((True, False), (False, False), (True, True)):
                outputs = []
                for kind in (self.reference, node_type):
                    node = self.node(kind, empty=empty)
                    node.net.train(training)
                    self.seed()
                    result = node._evaluate_ddpm_loss(node.net)
                    outputs.append((result, node.net.training, self.rng()))
                with self.subTest(variant=variant, training=training, empty=empty):
                    if empty:
                        self.assertTrue(
                            math.isnan(outputs[0][0]) and math.isnan(outputs[1][0])
                        )
                    else:
                        self.assertEqual(outputs[0][0], outputs[1][0])
                    self.assertEqual(outputs[0][1], outputs[1][1])
                    self.assertEqual(outputs[1][1], training)
                    self.assert_rng(outputs[0][2], outputs[1][2])

    def test_model_failure_restores_mode_but_preloop_seed_failure_does_not(self):
        for node_type in (self.reference, *self.candidates.values()):
            with self.subTest(node_type=node_type):
                node = self.node(node_type)
                node.net.fail = True
                with self.assertRaisesRegex(ValueError, "controlled diffusion failure"):
                    node._evaluate_ddpm_loss(node.net)
                self.assertTrue(node.net.training)
                node.net.fail = False
                node.validation_seed = None
                with self.assertRaises((TypeError, RuntimeError)):
                    node._evaluate_ddpm_loss(node.net)
                # Preserve the original try/finally boundary, not a silent bug fix.
                self.assertFalse(node.net.training)

    def test_loss_override_is_looked_up_for_each_batch(self):
        for variant, node_type in self.candidates.items():
            outputs = []
            for kind in (self.reference, node_type):
                node = self.node(kind)
                node.calls = []

                def later(self, model, batch):
                    self.calls.append("later")
                    return (model.scale * batch).mean().square()

                def first(self, model, batch):
                    self.calls.append("first")
                    self._ddpm_training_loss = MethodType(later, self)
                    return (model.scale * batch).mean().square()

                node._ddpm_training_loss = MethodType(first, node)
                optimizer = node._make_optimizer(node.net)
                self.seed()
                node._run_local_epoch(node.net, optimizer)
                outputs.append(
                    (node.calls, copy.deepcopy(node.net.state_dict()), self.rng())
                )
            with self.subTest(variant=variant):
                self.assertEqual(
                    outputs[1][0], ["first", "later", "later", "later", "later"]
                )
                self.assert_tree(outputs[0][:2], outputs[1][:2])
                self.assert_rng(outputs[0][2], outputs[1][2])


if __name__ == "__main__":
    unittest.main()
