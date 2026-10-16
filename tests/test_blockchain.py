"""Block/chain behavior, canonical payload encoding and legacy paths."""

import ast
import copy
from hashlib import sha256
import importlib.util
from pathlib import Path
import pickle
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "Tensor transaction encoding requires PyTorch")
class BlockchainTests(unittest.TestCase):
    def test_moved_class_and_encoding_bodies_match_frozen_source(self):
        for old, new, names in [
            ("Block.py", "block.py", ["Block"]),
            ("Blockchain.py", "ledger.py", ["Blockchain"]),
            (
                "stable_hash.py",
                "serialization.py",
                ["_canonical_bytes", "stable_digest", "payload_size_bytes"],
            ),
        ]:
            before = ast.parse(
                (ROOT / "experiments/protocol_control/source_snapshot" / old).read_text()
            )
            after = ast.parse((ROOT / "bcmfaigc/blockchain" / new).read_text())
            for name in names:
                a = next(n for n in before.body if getattr(n, "name", None) == name)
                b = next(n for n in after.body if getattr(n, "name", None) == name)
                self.assertEqual(ast.dump(a), ast.dump(b))

    def test_encoding_order_payload_size_and_no_rng(self):
        import torch
        from bcmfaigc.blockchain.serialization import stable_digest, payload_size_bytes

        before = torch.get_rng_state()
        tensor = torch.arange(6).reshape(2, 3)
        self.assertEqual(
            stable_digest({"a": tensor, "b": [1, True]}),
            stable_digest({"b": [1, True], "a": tensor.clone()}),
        )
        self.assertNotEqual(stable_digest(tensor), stable_digest(tensor.float()))
        self.assertEqual(stable_digest("x"), sha256(b"scalar:'x'").digest())
        self.assertEqual(payload_size_bytes(tensor), 48)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_hash_exclusions_and_chain_copy_semantics(self):
        from bcmfaigc.blockchain import Block, Blockchain

        block = Block(1, transactions=[{"worker": 1}], nonce=4)
        original_hash = block.compute_hash()
        block.set_signature(12)
        block.set_mining_rewards(8)
        block.set_pow_proof("proof")
        self.assertEqual(original_hash, block.compute_hash())
        self.assertNotEqual(original_hash, block.compute_hash(True))
        block.nonce_increment()
        self.assertNotEqual(original_hash, block.compute_hash())
        chain = Blockchain()
        self.assertIsNone(chain.return_last_block())
        chain.append_block(block)
        self.assertIsNot(chain.return_last_block(), block)
        self.assertIs(
            chain.return_last_block().return_transactions(), block.return_transactions()
        )
        blocks = [block]
        chain.replace_chain(blocks)
        self.assertIsNot(chain.chain, blocks)
        self.assertIs(chain.chain[0], block)
        self.assertEqual(chain.return_last_block_pow_proof(), block.compute_hash(True))
        block.free_tx()
        block.free_tx()
        self.assertFalse(hasattr(block, "_transactions"))

    def test_original_pickle_symbols_resolve_through_all_adapters(self):
        from bcmfaigc.blockchain import Block, Blockchain

        for variant in ("stage1", "stage2", "krum", "soft_stage2"):
            folder = ROOT / "experiments" / {"stage1":"comparison", "stage2":"byzantine_ratio", "krum":"krum", "soft_stage2":"trust_weighting"}[variant] / "BFL"
            with patch.object(sys, "path", [str(folder), *sys.path]):
                for name, expected in [("Block", Block), ("Blockchain", Blockchain)]:
                    spec = importlib.util.spec_from_file_location(
                        name, folder / (name + ".py")
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.assertIs(getattr(module, name), expected)
                    # Protocol 0 GLOBAL lookup proves legacy module/class resolution.
                    with patch.dict(sys.modules, {name: module}):
                        self.assertIs(
                            pickle.loads(("c" + name + "\n" + name + "\n.").encode()),
                            expected,
                        )


if __name__ == "__main__":
    unittest.main()
