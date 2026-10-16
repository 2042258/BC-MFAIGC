"""FID path/default contracts and numerical checks when SciPy is available.

CLI tests use dependency doubles: they do not load datasets or certify FID.
"""

from contextlib import redirect_stdout
import importlib.util
from io import StringIO
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
NUMERICAL_DEPS = all(importlib.util.find_spec(name) for name in ("numpy", "scipy"))


def fake_module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


def load_fid_cli():
    generated = MagicMock()
    generated.ndim, generated.shape = 4, (4, 3, 32, 32)
    generated.float.return_value = generated
    generated.__add__.return_value = generated
    generated.__truediv__.return_value = generated
    generated.clamp.return_value = generated
    torch = fake_module(
        "torch",
        cuda=SimpleNamespace(is_available=lambda: False),
        device=Mock(return_value=SimpleNamespace(type="cpu")),
        load=Mock(return_value=generated),
    )
    dataset = Mock(side_effect=DatasetReached)
    loader = Mock(return_value=[])
    stubs = {
        "numpy": fake_module("numpy"),
        "torch": torch,
        "torch.utils": fake_module("torch.utils"),
        "torch.utils.data": fake_module("torch.utils.data", DataLoader=loader),
        "pytorch_fid": fake_module("pytorch_fid"),
        "pytorch_fid.inception": fake_module(
            "pytorch_fid.inception", InceptionV3=Mock()
        ),
        "scipy": fake_module("scipy", linalg=Mock()),
        "torchvision": fake_module("torchvision"),
        "torchvision.datasets": fake_module("torchvision.datasets", CIFAR10=dataset),
    }
    with patch.dict(sys.modules, stubs):
        namespace = runpy.run_path(str(ROOT / "bcmfaigc/evaluation/fid.py"))
    return namespace, torch, dataset, loader


class DatasetReached(Exception):
    """Stop before accessing any dataset or running Inception."""


class FidCliTests(unittest.TestCase):
    def test_existing_entrypoints_keep_their_own_dataset_root(self):
        for variant in ("stage1", "stage2", "krum", "soft_stage2"):
            script = ROOT / "experiments" / {"stage1":"comparison", "stage2":"byzantine_ratio", "krum":"krum", "soft_stage2":"trust_weighting"}[variant] / "BFL/evaluate_fid.py"
            entry = Mock()
            stub = fake_module("bcmfaigc.evaluation.fid", main=entry)
            with (
                self.subTest(variant=variant),
                patch.dict(sys.modules, {stub.__name__: stub}),
            ):
                runpy.run_path(str(script), run_name="__main__")
                entry.assert_called_once_with(data_root=script.parents[1] / "data")

    def check_dataset_root(self, arguments, expected, legacy_root=None):
        namespace, torch, dataset, loader = load_fid_cli()
        with self.assertRaises(DatasetReached):
            namespace["main"](
                ["samples.pt", "--output", "fid.json", *arguments],
                data_root=legacy_root,
            )
        self.assertEqual(dataset.call_args.kwargs["root"], str(expected))
        self.assertIs(dataset.call_args.kwargs["train"], False)
        self.assertIs(dataset.call_args.kwargs["download"], False)
        torch.load.assert_called_once_with(
            Path("samples.pt"), map_location="cpu", weights_only=True
        )
        self.assertEqual(
            loader.call_args.kwargs,
            {"batch_size": 64, "shuffle": False, "num_workers": 0},
        )

    def test_module_default_data_root(self):
        self.check_dataset_root([], Path("data"))

    def test_legacy_data_root(self):
        self.check_dataset_root([], Path("experiment_data"), Path("experiment_data"))

    def test_explicit_data_root_overrides_legacy_default(self):
        self.check_dataset_root(
            ["--data-root", "selected_data"],
            Path("selected_data"),
            Path("experiment_data"),
        )

    def test_invalid_batch_size_is_rejected_before_loading(self):
        namespace, torch, _, _ = load_fid_cli()
        with self.assertRaises(ValueError):
            namespace["main"](
                ["samples.pt", "--output", "fid.json", "--batch-size", "0"]
            )
        torch.load.assert_not_called()

    def test_invalid_reference_size_is_rejected_before_loading(self):
        namespace, torch, _, _ = load_fid_cli()
        with self.assertRaises(ValueError):
            namespace["main"](
                ["samples.pt", "--output", "fid.json", "--max-real-samples", "1"]
            )
        torch.load.assert_not_called()

    def test_invalid_sample_shape_is_rejected(self):
        namespace, torch, dataset, _ = load_fid_cli()
        torch.load.return_value.shape = (4, 1, 28, 28)
        with self.assertRaisesRegex(ValueError, "shape"):
            namespace["main"](["samples.pt", "--output", "fid.json"])
        dataset.assert_not_called()

    def test_help_exposes_explicit_dataset_root(self):
        namespace, _, _, _ = load_fid_cli()
        output = StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as error:
            namespace["main"](["--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertIn("--data-root", output.getvalue())


@unittest.skipUnless(NUMERICAL_DEPS, "Numerical FID tests require NumPy and SciPy")
class FrechetDistanceTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from scipy import linalg

        namespace, _, _, _ = load_fid_cli()
        self.distance = namespace["compute_frechet_distance"]
        self.distance.__globals__.update(np=np, linalg=linalg)
        self.np = np

    def test_identical_gaussians_have_zero_distance(self):
        mean = self.np.array([1.0, 2.0])
        covariance = self.np.diag([2.0, 3.0])
        self.assertAlmostEqual(self.distance(mean, covariance, mean, covariance), 0.0)

    def test_diagonal_covariances_and_mean_difference(self):
        mean1, mean2 = self.np.array([0.0, 0.0]), self.np.array([1.0, 2.0])
        covariance1, covariance2 = self.np.diag([1.0, 4.0]), self.np.diag([4.0, 9.0])
        self.assertAlmostEqual(
            self.distance(mean1, covariance1, mean2, covariance2), 7.0
        )


if __name__ == "__main__":
    unittest.main()
