"""Simulation CLI profiles and persistence contracts; no data or GPU work."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from bcmfaigc.simulation.cli import build_parser, validate_arguments
from bcmfaigc.variants import CLASSIC, EXTENDED, KRUM

ROOT = Path(__file__).resolve().parents[1]


class SimulationConfigurationTests(unittest.TestCase):
    def test_all_original_argument_defaults_aliases_and_choices(self):
        contracts = json.loads(
            (ROOT / "tests/fixtures/simulation_cli_contracts.json").read_text()
        )
        for variant in (CLASSIC, EXTENDED, KRUM):
            parser = build_parser(variant)
            actual = [
                {
                    "dest": a.dest,
                    "options": a.option_strings,
                    "default": a.default,
                    "choices": a.choices,
                    "nargs": a.nargs,
                    "required": a.required,
                    "type": getattr(a.type, "__name__", None),
                }
                for a in parser._actions
            ]
            self.assertEqual(actual, contracts[variant.name], variant.name)

    def test_aggregation_guard_and_krum_alias(self):
        parser = build_parser(KRUM)
        args = vars(
            parser.parse_args(
                [
                    "--aggregation_method",
                    "krum",
                    "--krum_f",
                    "2",
                    "--blockchain_on",
                    "0",
                    "--defense_on",
                    "0",
                ]
            )
        )
        validate_arguments(parser, args, KRUM)
        self.assertEqual(args["multikrum_f"], 2)
        parser.error = lambda message: (_ for _ in ()).throw(ValueError(message))
        args["blockchain_on"] = 1
        with self.assertRaisesRegex(ValueError, "Krum and Multi-Krum"):
            validate_arguments(parser, args, KRUM)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Checkpoint tests need PyTorch")
class SimulationPersistenceTests(unittest.TestCase):
    def test_checkpoint_source_order_schema_and_integer_buffers(self):
        import torch
        from bcmfaigc.simulation.reports import write_global_checkpoint

        def device(idx, online, value):
            return SimpleNamespace(
                return_idx=lambda: idx,
                is_online=lambda: online,
                global_parameters={"w": torch.tensor([value]), "n": torch.tensor(3)},
                global_version=4,
            )

        devices = [device("b", True, 2.0), device("a", False, 1.0)]
        with tempfile.TemporaryDirectory() as tmp:
            write_global_checkpoint(
                {
                    "save_global_checkpoints": 1,
                    "checkpoint_frequency": 5,
                    "model_name": "ddpm",
                    "seed": 8,
                    "validation_seed": 9,
                },
                devices,
                tmp,
                1,
            )
            checkpoint = torch.load(
                Path(tmp) / "checkpoints/global_round_0001.pt", weights_only=False
            )
            self.assertEqual(checkpoint["checkpoint_source_device"], "b")
            self.assertEqual(list(checkpoint["global_versions"]), ["a", "b"])
            self.assertTrue(
                torch.equal(checkpoint["state_dict"]["w"], torch.tensor([2.0]))
            )
            self.assertEqual(checkpoint["state_dict"]["n"].dtype, torch.int64)
            self.assertFalse(checkpoint["global_state_consistent"])
            self.assertEqual(checkpoint["timesteps"], 1000)

    def test_cold_resume_resolves_actual_legacy_model_and_network_references(self):
        import os
        import pickle
        import subprocess
        import sys
        from types import ModuleType
        from unittest.mock import patch
        from bcmfaigc.models.ddpm import DDPM, UNet, TimeEmbedding

        legacy_models = ModuleType("models.Models")
        legacy_parent = ModuleType("models")
        legacy_parent.__path__ = []
        legacy_parent.Models = legacy_models
        legacy_device = ModuleType("Device")
        device_type = type("Device", (), {"__module__": "Device"})
        network_type = type("DevicesInNetwork", (), {"__module__": "Device"})
        legacy_device.Device, legacy_device.DevicesInNetwork = device_type, network_type
        model = DDPM(timesteps=4)
        node = device_type()
        node.net = model
        network = network_type()
        network.devices_set = {"legacy": node}
        classes = (DDPM, UNet, TimeEmbedding)
        original_modules = [cls.__module__ for cls in classes]
        for cls in classes:
            setattr(legacy_models, cls.__name__, cls)
            cls.__module__ = "models.Models"
        try:
            with patch.dict(
                sys.modules,
                {
                    "models": legacy_parent,
                    "models.Models": legacy_models,
                    "Device": legacy_device,
                },
            ):
                payload = pickle.dumps(network)
        finally:
            for cls, module in zip(classes, original_modules):
                cls.__module__ = module
        self.assertIn(b"models.Models", payload)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bfl = root / "BFL"
            snapshot = bfl / "snapshots/legacy"
            logs = root / "logs/blogs/legacy_ddpm"
            snapshot.mkdir(parents=True)
            logs.mkdir(parents=True)
            (snapshot / "snapshot_r_2").write_bytes(payload)
            (logs / "args_used.txt").write_text(
                "--unit_reward 1\n--hard_assign 1,1,1\n--pow_difficulty 0\n"
            )
            entry = bfl / "main.py"
            entry.write_text(
                "from bcmfaigc.simulation.runner import run\nfrom bcmfaigc.variants import EXTENDED\nrun(EXTENDED, entry_file=__file__)\n"
            )
            env = {
                **os.environ,
                "PYTHONPATH": str(ROOT),
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
            result = subprocess.run(
                [
                    sys.executable,
                    str(entry),
                    "--resume_path",
                    "legacy",
                    "--max_num_comm",
                    "2",
                    "--verbose",
                    "0",
                ],
                cwd=bfl,
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("end time:", (logs / "time.txt").read_text())

    def test_snapshot_frequency_and_original_retention_order(self):
        from bcmfaigc.simulation.reports import save_network_snapshot
        import pickle

        args = {"save_network_snapshots": 1, "save_freq": 5, "save_most_recent": 2}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(4):
                (root / ("old_" + str(i))).write_text(str(i))
            save_network_snapshot(args, {"network": "fixture"}, tmp, 1)
            self.assertEqual(
                len(list(root.iterdir())), 3
            )  # Original pruning occurs before save.
            self.assertEqual(
                pickle.loads((root / "snapshot_r_1").read_bytes()),
                {"network": "fixture"},
            )
            save_network_snapshot(args, {}, tmp, 2)
            self.assertEqual(len(list(root.iterdir())), 3)


if __name__ == "__main__":
    unittest.main()
