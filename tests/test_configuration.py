"""Configuration identity and one-run command behavior."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationIdentityTests(unittest.TestCase):
    def test_original_configuration_bytes_and_indexed_run_selection(self):
        hashes = json.loads(
            (ROOT / "tests/fixtures/configuration_hashes.json").read_text()
        )
        for relative, expected in hashes.items():
            self.assertEqual(
                sha256((ROOT / relative).read_bytes()).hexdigest(), expected, relative
            )
        index = json.loads((ROOT / "experiment_index.json").read_text())
        frozen_selections = json.loads(
            (ROOT / "tests/fixtures/experiment_selections.json").read_text()
        )
        self.assertEqual(
            {item["workflow"]: item["run_ids"] for item in index}, frozen_selections
        )
        for item in index:
            document = json.loads((ROOT / item["manifest"]).read_text())
            available = {run["id"] for run in document["runs"]}
            self.assertEqual(len(item["run_ids"]), len(set(item["run_ids"])))
            self.assertTrue(set(item["run_ids"]) <= available)
            self.assertEqual(item["run_count"], len(item["run_ids"]))

    def test_examples_preserve_selected_source_arguments(self):
        for path in (ROOT / "configs/examples").glob("*.json"):
            example = json.loads(path.read_text())
            matrix = json.loads((ROOT / example["source_manifest"]).read_text())
            run = next(
                item for item in matrix["runs"] if item["id"] == example["run_id"]
            )
            self.assertEqual(
                example["arguments"], matrix.get("common", []) + run.get("args", [])
            )


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("Crypto"),
    "Training command imports require prepared dependencies",
)
class SingleRunCommandTests(unittest.TestCase):
    def test_one_matrix_case_preserves_tag_and_override_order(self):
        from bcmfaigc.simulation import command

        config = ROOT / "configs/byzantine_ratio/BFL/ratio_scan_manifest_v1.json"
        with tempfile.TemporaryDirectory() as tmp, patch.object(command, "run") as run:
            previous = Path.cwd()
            command.main(
                [
                    "--config",
                    str(config),
                    "--run-id",
                    "ratio-nm0-fedavg-50000",
                    "--workspace",
                    tmp,
                    "--seed",
                    "88",
                ]
            )
            self.assertEqual(run.call_count, 1)
            variant, arguments = run.call_args.args
            self.assertEqual(variant.name, "extended")
            self.assertEqual(arguments[-2:], ["--seed", "88"])
            self.assertEqual(
                arguments[arguments.index("--run_tag") + 1], "ratio-nm0-fedavg-50000"
            )
            self.assertTrue((Path(tmp) / "BFL/snapshots").is_dir())
            self.assertEqual(Path.cwd(), previous)


if __name__ == "__main__":
    unittest.main()
