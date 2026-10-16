"""Dataset compatibility, model aliases and trusted legacy-pickle contracts."""

from contextlib import contextmanager, redirect_stdout
import gzip
import importlib.util
from io import StringIO
import os
from pathlib import Path
import pickle
import runpy
import struct
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

from bcmfaigc.data import Cifar10Dataset
from bcmfaigc.data.compat import DatasetLoad
from bcmfaigc.data.mnist import read_mnist_images, read_mnist_labels

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


@contextmanager
def cifar_fixture():
    previous = Path.cwd()
    state = np.random.get_state()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        data = root / "data/cifar-10-batches-py"
        data.mkdir(parents=True)
        rng = np.random.default_rng(28)
        batches = []
        labels = []
        for number in range(1, 6):
            images = rng.integers(0, 256, (3, 3072), dtype=np.uint8)
            batch_labels = [(number * 3 + offset) % 10 for offset in range(3)]
            (data / f"data_batch_{number}").write_bytes(
                pickle.dumps({b"data": images, b"labels": batch_labels})
            )
            batches.append(images)
            labels.extend(batch_labels)
        test_images = rng.integers(0, 256, (2, 3072), dtype=np.uint8)
        (data / "test_batch").write_bytes(
            pickle.dumps({b"data": test_images, b"labels": [1, 4]})
        )
        bfl = root / "BFL"
        bfl.mkdir()
        os.chdir(bfl)
        try:
            yield np.concatenate(batches), np.array(labels), test_images
        finally:
            os.chdir(previous)
            np.random.set_state(state)


class DatasetTests(unittest.TestCase):
    def test_non_iid_order_normalization_labels_and_sizes(self):
        with cifar_fixture() as (images, labels, test_images):
            dataset = Cifar10Dataset(is_iid=False)
            order = np.argsort(labels)
            expected = np.multiply(
                images.reshape(15, 3, 32, 32).astype(np.float32), 1.0 / 255.0
            )
            np.testing.assert_array_equal(dataset.train_data, expected[order])
            np.testing.assert_array_equal(
                dataset.train_label, np.eye(10)[labels][order]
            )
            expected_test = np.multiply(
                test_images.reshape(2, 3, 32, 32).astype(np.float32), 1.0 / 255.0
            )
            np.testing.assert_array_equal(dataset.test_data, expected_test)
            self.assertEqual(dataset.train_data.dtype, np.dtype("float32"))
            self.assertEqual(dataset.train_label.dtype, np.dtype("float64"))
            self.assertEqual((dataset.train_data_size, dataset.test_data_size), (15, 2))

    def test_iid_shuffle_consumes_the_original_numpy_draws(self):
        with cifar_fixture() as (_, labels, _):
            np.random.seed(81)
            order = np.arange(15)
            np.random.shuffle(order)
            expected_state = np.random.get_state()
            np.random.seed(81)
            dataset = Cifar10Dataset(is_iid=True)
            np.testing.assert_array_equal(
                dataset.train_label, np.eye(10)[labels][order]
            )
            actual_state = np.random.get_state()
            self.assertEqual(actual_state[0], expected_state[0])
            np.testing.assert_array_equal(actual_state[1], expected_state[1])
            self.assertEqual(actual_state[2:], expected_state[2:])

    def test_legacy_constructor_and_reload_keywords(self):
        with cifar_fixture():
            modern = Cifar10Dataset(is_iid=False)
            legacy = DatasetLoad(dataSetName="cifar10", isIID=False)
            np.testing.assert_array_equal(legacy.train_data, modern.train_data)
            legacy.cifar10DataSetConstruct(isIID=False)
            np.testing.assert_array_equal(legacy.train_label, modern.train_label)
            images, labels = legacy.load_cifar10_batch(
                file_path="../data/cifar-10-batches-py/test_batch"
            )
            self.assertEqual(images.shape, (2, 3072))
            self.assertEqual(labels.shape, (2, 10))

    def test_legacy_constructor_still_dispatches_subclass_hook_positionally(self):
        class CustomDataset(DatasetLoad):
            def cifar10DataSetConstruct(self, flag):
                self.observed_flag = flag

        dataset = CustomDataset("cifar10", True)
        self.assertIs(dataset.observed_flag, True)
        self.assertIsNone(dataset.train_data)

    def test_legacy_batch_override_is_used_by_dataset_construction(self):
        class CustomDataset(DatasetLoad):
            calls = 0

            def load_cifar10_batch(self, path):
                self.calls += 1
                return np.zeros((1, 3072), dtype=np.uint8), np.eye(10)[[0]]

        dataset = CustomDataset("cifar10", False)
        self.assertEqual(dataset.calls, 6)
        self.assertEqual(dataset.train_data.shape, (5, 3, 32, 32))

    def test_legacy_unknown_dataset_still_has_empty_fields(self):
        dataset = DatasetLoad("unsupported", False)
        self.assertEqual(dataset.name, "unsupported")
        self.assertIsNone(dataset.train_data)
        self.assertIsNone(dataset.test_label)

    def test_trusted_legacy_dataset_pickle_resolves_the_adapter(self):
        with cifar_fixture():
            dataset = DatasetLoad("cifar10", False)
            legacy_module = ModuleType("DatasetLoad")
            old_type = type("DatasetLoad", (), {})
            old_type.__module__ = "DatasetLoad"
            legacy_module.DatasetLoad = old_type
            old = old_type()
            old.__dict__.update(dataset.__dict__)
            with patch.dict(sys.modules, {"DatasetLoad": legacy_module}):
                payload = pickle.dumps(old)
            current_module = ModuleType("DatasetLoad")
            current_module.DatasetLoad = DatasetLoad
            with patch.dict(sys.modules, {"DatasetLoad": current_module}):
                restored = pickle.loads(payload)
            self.assertIs(type(restored), DatasetLoad)
            np.testing.assert_array_equal(restored.train_data, dataset.train_data)
            restored.cifar10DataSetConstruct(isIID=False)
            np.testing.assert_array_equal(restored.train_data, dataset.train_data)

    def test_mnist_readers_preserve_idx_shapes_and_one_hot_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            images = Path(temporary) / "images.gz"
            labels = Path(temporary) / "labels.gz"
            images.write_bytes(
                gzip.compress(struct.pack(">IIII", 2051, 1, 2, 3) + bytes(range(6)))
            )
            labels.write_bytes(
                gzip.compress(struct.pack(">II", 2049, 3) + bytes([3, 0, 9]))
            )
            with redirect_stdout(StringIO()):
                actual_images = read_mnist_images(images)
                actual_labels = read_mnist_labels(labels)
            np.testing.assert_array_equal(
                actual_images, np.arange(6, dtype=np.uint8).reshape(1, 2, 3, 1)
            )
            np.testing.assert_array_equal(actual_labels, np.eye(10)[[3, 0, 9]])


@unittest.skipUnless(HAS_TORCH, "Model and transform tests require PyTorch")
class ModelTests(unittest.TestCase):
    def test_legacy_model_and_dataset_modules_reexport_shared_implementations(self):
        from bcmfaigc.models.ddpm import DDPM, UNet, TimeEmbedding

        for variant in ("stage1", "stage2", "krum", "soft_stage2"):
            with self.subTest(variant=variant):
                model = runpy.run_path(
                    str(ROOT / "experiments" / {"stage1":"comparison", "stage2":"byzantine_ratio", "krum":"krum", "soft_stage2":"trust_weighting"}[variant] / "models/Models.py")
                )
                dataset = runpy.run_path(
                    str(ROOT / "experiments" / {"stage1":"comparison", "stage2":"byzantine_ratio", "krum":"krum", "soft_stage2":"trust_weighting"}[variant] / "BFL/DatasetLoad.py")
                )
                self.assertIs(model["DDPM"], DDPM)
                self.assertIs(model["UNet"], UNet)
                self.assertIs(model["TimeEmbedding"], TimeEmbedding)
                self.assertIs(dataset["DatasetLoad"], DatasetLoad)

    def test_registered_parameter_keys_and_checkpoint_loading(self):
        import torch
        from bcmfaigc.models import DDPM, DenoisingNetwork

        model = DDPM(timesteps=4)
        self.assertIs(type(model.unet), DenoisingNetwork)
        self.assertTrue(
            any(key.startswith("unet.encoder.") for key in model.state_dict())
        )
        state = {name: value.clone() for name, value in model.state_dict().items()}
        restored = DDPM(timesteps=4)
        restored.load_state_dict(state, strict=True)
        self.assertTrue(
            all(
                torch.equal(value, restored.state_dict()[name])
                for name, value in state.items()
            )
        )

    def test_trusted_old_model_pickle_class_paths_still_resolve(self):
        import torch
        from bcmfaigc.models.ddpm import DDPM, UNet, TimeEmbedding

        model = DDPM(timesteps=4)
        package = ModuleType("models")
        package.__path__ = []
        old_module = ModuleType("models.Models")
        old_module.DDPM, old_module.UNet, old_module.TimeEmbedding = (
            DDPM,
            UNet,
            TimeEmbedding,
        )
        package.Models = old_module
        with patch.dict(sys.modules, {"models": package, "models.Models": old_module}):
            with (
                patch.object(DDPM, "__module__", "models.Models"),
                patch.object(UNet, "__module__", "models.Models"),
                patch.object(TimeEmbedding, "__module__", "models.Models"),
            ):
                payload = pickle.dumps(model)
            restored = pickle.loads(payload)
        self.assertTrue(
            all(
                torch.equal(value, restored.state_dict()[name])
                for name, value in model.state_dict().items()
            )
        )

    def test_gaussian_transform_keeps_values_repr_and_rng(self):
        import torch
        from bcmfaigc.data.transforms import AddGaussianNoise

        state = torch.get_rng_state()
        try:
            tensor = torch.ones(2, 3)
            torch.manual_seed(11)
            expected = tensor + torch.randn(tensor.size()) * 0.3 + 0.2
            expected_state = torch.get_rng_state()
            torch.manual_seed(11)
            transform = AddGaussianNoise(mean=0.2, std=0.3)
            self.assertTrue(torch.equal(transform(tensor), expected))
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_state))
            self.assertEqual(repr(transform), "AddGaussianNoise(mean=0.2, std=0.3)")
        finally:
            torch.set_rng_state(state)


if __name__ == "__main__":
    unittest.main()
