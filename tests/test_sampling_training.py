"""Path, argument and call-order contracts; doubles do not certify training."""

from contextlib import nullcontext, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import random
import runpy
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def module(name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    return result


class Batch:
    def __init__(self, size):
        self.size = size
        self.slices = []

    def cpu(self):
        return self

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        self.slices.append(key)
        return self


def load_sampling():
    torch = module(
        "torch",
        cuda=SimpleNamespace(is_available=lambda: False, manual_seed_all=Mock()),
        load=Mock(),
        manual_seed=Mock(),
        inference_mode=nullcontext,
        save=Mock(),
        cat=Mock(
            side_effect=lambda batches, dim: Batch(sum(len(batch) for batch in batches))
        ),
        device=Mock(return_value=SimpleNamespace(type="cpu")),
    )
    save_image = Mock()
    stubs = {
        "torch": torch,
        "torchvision": module("torchvision"),
        "torchvision.utils": module("torchvision.utils", save_image=save_image),
    }
    with patch.dict(sys.modules, stubs):
        namespace = runpy.run_path(str(ROOT / "bcmfaigc/evaluation/sampling.py"))
    return namespace, torch, save_image


class SamplingTests(unittest.TestCase):
    def test_partial_final_batch_and_cpu_seed(self):
        namespace, torch, _ = load_sampling()
        model = Mock()
        model.sample.side_effect = lambda count, device: Batch(count)
        device = SimpleNamespace(type="cpu")
        samples = namespace["generate_ddpm_samples"](model, 5, 2, 47, device)
        self.assertEqual(len(samples), 5)
        self.assertEqual(
            [call.args[0] for call in model.sample.call_args_list], [2, 2, 1]
        )
        torch.manual_seed.assert_called_once_with(47)
        torch.cuda.manual_seed_all.assert_not_called()

    def test_cuda_seed_is_still_explicit(self):
        namespace, torch, _ = load_sampling()
        model = Mock()
        model.sample.return_value = Batch(1)
        namespace["generate_ddpm_samples"](
            model, 1, 1, 19, SimpleNamespace(type="cuda")
        )
        torch.manual_seed.assert_called_once_with(19)
        torch.cuda.manual_seed_all.assert_called_once_with(19)

    def test_checkpoint_uses_supplied_model_and_strict_state_loading(self):
        namespace, torch, _ = load_sampling()
        torch.load.return_value = {"timesteps": 7, "state_dict": {"test": "weights"}}
        model = Mock()
        model.to.return_value = model
        model_type = Mock(return_value=model)
        device = SimpleNamespace(type="cpu")
        result = namespace["load_ddpm_checkpoint"](
            Path("checkpoint.pt"), model_type, device
        )
        self.assertIs(result, model)
        model_type.assert_called_once_with(timesteps=7)
        model.load_state_dict.assert_called_once_with({"test": "weights"}, strict=True)
        model.eval.assert_called_once_with()
        torch.load.assert_called_once_with(
            Path("checkpoint.pt"), map_location=device, weights_only=True
        )

    def test_checkpoint_without_timesteps_retains_1000_default(self):
        namespace, torch, _ = load_sampling()
        torch.load.return_value = {"state_dict": {}}
        model_type = Mock()
        namespace["load_ddpm_checkpoint"](
            Path("checkpoint.pt"), model_type, SimpleNamespace(type="cpu")
        )
        model_type.assert_called_once_with(timesteps=1000)

    def test_saved_output_names_and_preview_settings(self):
        namespace, torch, save_image = load_sampling()
        samples = Batch(128)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "generated"
            namespace["save_generated_samples"](samples, output)
            torch.save.assert_called_once_with(samples, output / "samples.pt")
            save_image.assert_called_once_with(
                samples,
                output / "preview.png",
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
            )
            self.assertEqual(samples.slices, [slice(None, 64)])

    def test_invalid_counts_fail_before_checkpoint_loading(self):
        for option in ("--num-samples", "--batch-size"):
            namespace, torch, _ = load_sampling()
            with self.subTest(option=option), self.assertRaises(ValueError):
                namespace["main"](
                    ["checkpoint.pt", "--output-dir", "generated", option, "0"],
                    model_type=Mock(),
                )
            torch.load.assert_not_called()

    def test_cli_keeps_generation_defaults_and_call_order(self):
        namespace, torch, _ = load_sampling()
        main = namespace["main"]
        events = Mock()
        events.load.return_value = Mock()
        events.generate.return_value = Batch(1000)
        main.__globals__.update(
            load_ddpm_checkpoint=events.load,
            generate_ddpm_samples=events.generate,
            save_generated_samples=events.save,
        )
        model_type = Mock()
        with redirect_stdout(StringIO()):
            main(["checkpoint.pt", "--output-dir", "generated"], model_type=model_type)
        self.assertEqual(
            [call[0] for call in events.mock_calls], ["load", "generate", "save"]
        )
        self.assertEqual(events.generate.call_args.args[1:4], (1000, 32, 0))
        self.assertEqual(
            events.load.call_args.args[:2], (Path("checkpoint.pt"), model_type)
        )


class DatasetReached(Exception):
    pass


def load_central():
    functional = module("torch.nn.functional")
    nn = module("torch.nn", Module=object, functional=functional)
    torch = module(
        "torch",
        nn=nn,
        cuda=SimpleNamespace(is_available=lambda: False),
        manual_seed=Mock(),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(deterministic=False, benchmark=True)
        ),
        device=Mock(return_value=SimpleNamespace(type="cpu")),
    )
    numpy = module("numpy", random=SimpleNamespace(seed=Mock()))
    stubs = {
        "numpy": numpy,
        "torch": torch,
        "torch.nn": nn,
        "torch.nn.functional": functional,
        "torch.utils": module("torch.utils"),
        "torch.utils.data": module(
            "torch.utils.data", DataLoader=Mock(), TensorDataset=Mock()
        ),
    }
    with patch.dict(sys.modules, stubs):
        namespace = runpy.run_path(str(ROOT / "bcmfaigc/training/central.py"))
    return namespace, torch


class CentralTrainingTests(unittest.TestCase):
    def test_defaults_paths_and_dataset_constructor_are_preserved(self):
        namespace, torch = load_central()
        dataset_type = Mock(side_effect=DatasetReached)
        model_type = Mock()
        previous = Path.cwd()
        state = random.getstate()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                bfl = Path(temporary) / "BFL"
                bfl.mkdir()
                os.chdir(bfl)
                with self.assertRaises(DatasetReached):
                    namespace["main"](
                        ["--seed", "12", "--run-tag", "unit"],
                        dataset_type=dataset_type,
                        model_type=model_type,
                    )
                documents = list(
                    (Path(temporary) / "logs/central").glob("*/args_used.json")
                )
                self.assertEqual(len(documents), 1)
                args = json.loads(documents[0].read_text())
                for key, value in {
                    "epochs": 100,
                    "batchsize": 128,
                    "test_batchsize": 256,
                    "learning_rate": 0.001,
                    "validation_seed": 9001,
                    "noise_probability": 0.0,
                    "noise_scale": 0.0,
                    "checkpoint_frequency": 20,
                    "accumulation_steps": 4,
                    "optimizer_state_reset_each_epoch": True,
                }.items():
                    self.assertEqual(args[key], value, key)
                dataset_type.assert_called_once_with("cifar10", isIID=False)
                model_type.assert_not_called()
                torch.manual_seed.assert_called_once_with(12)
        finally:
            os.chdir(previous)
            random.setstate(state)

    def test_invalid_probability_fails_before_random_seed_or_dataset(self):
        namespace, torch = load_central()
        dataset_type = Mock()
        with self.assertRaises(ValueError):
            namespace["main"](
                ["--seed", "12", "--run-tag", "unit", "--noise-probability", "1.1"],
                dataset_type=dataset_type,
                model_type=Mock(),
            )
        torch.manual_seed.assert_not_called()
        dataset_type.assert_not_called()

    def test_all_transition_entries_supply_original_classes(self):
        model_type = object()
        dataset_type = object()
        entries = [
            ROOT / "experiments" / {"stage1":"comparison", "stage2":"byzantine_ratio", "krum":"krum", "soft_stage2":"trust_weighting"}[variant] / "BFL" / name
            for variant in ("stage1", "stage2", "krum", "soft_stage2")
            for name in ("evaluate_ddpm.py", "train_central_ddpm.py")
        ]
        entries.append(ROOT / "experiments/comparison/train_central_ddpm.py")
        for script in entries:
            central = script.name == "train_central_ddpm.py"
            leaf = (
                "bcmfaigc.training.central"
                if central
                else "bcmfaigc.evaluation.sampling"
            )
            main = Mock()
            stubs = {
                leaf: module(leaf, main=main),
                "models.Models": module("models.Models", DDPM=model_type),
                "DatasetLoad": module("DatasetLoad", DatasetLoad=dataset_type),
            }
            with (
                self.subTest(script=script),
                patch.dict(sys.modules, stubs),
                patch.object(sys, "path", sys.path.copy()),
            ):
                runpy.run_path(str(script), run_name="__main__")
            expected = {"model_type": model_type}
            if central:
                expected["dataset_type"] = dataset_type
            main.assert_called_once_with(**expected)


if __name__ == "__main__":
    unittest.main()
