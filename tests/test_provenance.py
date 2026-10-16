"""Current-source inventories must not masquerade as historical equality."""

from hashlib import sha256
import json
import os
from pathlib import Path
import random
import tempfile
import unittest

from bcmfaigc.provenance import build_source_manifest, write_source_manifest


class ProvenanceTests(unittest.TestCase):
    def make_experiment(self, root):
        (root / "BFL").mkdir()
        (root / "models").mkdir()
        (root / "BFL/main.py").write_text("value = 1\n")
        (root / "models/Models.py").write_text("value = 2\n")

    def test_records_actual_adapters_and_shared_dependencies_without_rng_draws(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_experiment(root)
            before = random.getstate()
            document = build_source_manifest(root)
            self.assertEqual(random.getstate(), before)
            self.assertEqual(document["historical_reference_status"], "not_evaluated")
            files = document["files"]
            self.assertEqual(
                files["experiment/BFL/main.py"], sha256(b"value = 1\n").hexdigest()
            )
            self.assertIn("shared/data/cifar10.py", files)
            self.assertIn("shared/data/compat.py", files)
            self.assertIn("shared/models/ddpm.py", files)
            self.assertIn("shared/aggregation/trust_weighting.py", files)
            self.assertIn("shared/training/local.py", files)
            self.assertIn("shared/validation.py", files)
            self.assertIn("shared/provenance.py", files)
            self.assertTrue(all(not Path(name).is_absolute() for name in files))
            (root / "models/Models.py").write_text("value = 3\n")
            self.assertNotEqual(
                files["experiment/models/Models.py"],
                build_source_manifest(root)["files"]["experiment/models/Models.py"],
            )

    def test_write_is_additive_and_never_replaces_historical_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_experiment(root)
            historical = root / "code_fingerprint.json"
            historical.write_text('{"historical": "unchanged"}')
            with self.assertRaises(ValueError):
                write_source_manifest(historical, root)
            self.assertEqual(historical.read_text(), '{"historical": "unchanged"}')
            destination = root / "source_manifest.json"
            document = write_source_manifest(destination, root)
            self.assertEqual(json.loads(destination.read_text()), document)
            with self.assertRaises(FileExistsError):
                write_source_manifest(destination, root)

    def test_missing_source_directory_is_not_accepted_as_empty_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                write_source_manifest(root / "source_manifest.json", root)
            self.assertFalse((root / "source_manifest.json").exists())

    @unittest.skipUnless(os.name == "posix", "Source-symlink rejection")
    def test_untracked_source_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_experiment(root)
            (root / "BFL/alias.py").symlink_to(root / "models/Models.py")
            with self.assertRaises(ValueError):
                build_source_manifest(root)


if __name__ == "__main__":
    unittest.main()
