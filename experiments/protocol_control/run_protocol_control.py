"""Matched H1/H3 verification microbenchmark; no model training or network simulation.

Uses an isolated snapshot of the existing prototype's actual verification methods.
Transactions carry small synthetic scalar/vector payloads, not DDPM parameters.
No keys or signed payloads are persisted. Timings are not full BC round times.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "source_snapshot"
sys.path.insert(0, str(SNAPSHOT))
from Crypto.PublicKey import RSA
from Block import Block
from Blockchain import Blockchain
from stable_hash import stable_digest

CONFIGS = ("NOBC-original", "NOBC-matched", "BC-matched")
SEEDS = (47000, 47001, 47002)


def load_verification_class():
    tree = ast.parse((SNAPSHOT / "Device.py").read_text())
    device = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Device")
    names = {"verify_validator_transaction", "verify_block", "check_pow_proof", "sign_msg", "sign_block"}
    methods = [copy.deepcopy(n) for n in device.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods} == names
    module = ast.Module(body=[ast.ClassDef(name="VerificationMethods", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(module)
    # Console output is disabled consistently; no project main/Device imports run.
    ns = {"time": time, "copy": copy, "stable_digest": stable_digest, "print": lambda *a, **k: None}
    exec(compile(module, str(SNAPSHOT / "Device.py"), "exec"), ns)
    return ns["VerificationMethods"]


VerificationMethods = load_verification_class()


class Verifier(VerificationMethods):
    def __init__(self, key, idx):
        self.idx = idx
        self.role = "miner"
        self.computation_power = 1
        self.check_signature = True
        self.black_list = set()
        self.pow_difficulty = 0
        self.modulus, self.private_key = int(key.n), int(key.d)
        self.chain = Blockchain()

    def online_switcher(self):
        return True

    def return_blockchain_object(self):
        return self.chain


def sign_transaction(tx, key):
    tx = copy.deepcopy(tx)
    tx["validator_rsa_pub_key"] = {"modulus": int(key.n), "pub_key": int(key.e)}
    digest = int.from_bytes(stable_digest(sorted(tx.items())), byteorder="big")
    tx["validator_signature"] = pow(digest, int(key.d), int(key.n))
    return tx


def fixtures(seed, scenario, validator_key, rounds, normal_count):
    rng = random.Random(seed)
    prior = None
    batches = []
    for round_number in range(1, rounds + 2):
        normals = []
        for worker in range(normal_count):
            tx = {
                "event_id": f"{scenario}-{seed}-{round_number}-normal-{worker}",
                "worker_device_idx": f"worker_{worker}",
                "validation_done_by": "validator_0",
                "in_round_number": round_number,
                "aggregation_weight": 1.0,
                "update_direction": True,
                "synthetic_update": [rng.uniform(-0.01, 0.01) for _ in range(16)],
            }
            normals.append(sign_transaction(tx, validator_key))
        if round_number >= 2:
            if scenario == "H1":
                attack = copy.deepcopy(normals[0])
                # This matches the old hook: signed aggregation weight 1 -> .25.
                attack["aggregation_weight"] = 0.25
            else:
                # Valid old signature, stale round; do not mutate signed fields.
                attack = copy.deepcopy(prior[0])
            # External event IDs do not alter the signed payload.
            entries = [(tx["event_id"], "normal", tx) for tx in normals]
            entries.append((f"{scenario}-{seed}-{round_number}-attack", "attack", attack))
            rng.shuffle(entries)
            batches.append((round_number, entries))
        prior = normals
    return batches


def transaction_decision(config, verifier, tx, current_round):
    if config == "NOBC-original":
        return True, "unchecked_direct_application"
    # Same predicate and check order as the existing main.py miner path.
    transaction_round = tx.get("in_round_number")
    if transaction_round is not None and int(transaction_round) != current_round:
        return False, "stale_round"
    _, valid = verifier.verify_validator_transaction(tx)
    return bool(valid), "signature_valid" if valid else "invalid_signature"


def commit_block(accepted, round_number, producer, observer, miner_key):
    last = observer.chain.return_last_block()
    previous_hash = last.compute_hash(hash_entire_block=True) if last else None
    block = Block(round_number, previous_block_hash=previous_hash,
                  transactions=accepted, miner_rsa_pub_key={"modulus": int(miner_key.n), "pub_key": int(miner_key.e)},
                  mined_by=producer.idx, mining_rewards=0)
    # Existing zero-difficulty hash/proof convention; no consensus is simulated.
    block.set_pow_proof(block.compute_hash())
    producer.sign_block(block)
    verified, _ = observer.verify_block(block, producer.idx)
    if verified is False:
        return False
    observer.chain.append_block(verified)
    return True


def process(config, batches, miner_key, capture=False):
    verifier = Verifier(miner_key, "observer_0")
    producer = Verifier(miner_key, "producer_0")
    events = []
    elapsed_ns = []
    for round_number, entries in batches:
        accepted = []
        decisions = []
        start = time.perf_counter_ns()
        for event_id, kind, tx in entries:
            applied, reason = transaction_decision(config, verifier, tx, round_number)
            if applied:
                accepted.append(tx)
            if capture:
                decisions.append({"event_id": event_id, "kind": kind, "accepted": applied, "reason": reason})
        block_ok = True
        if config == "BC-matched":
            block_ok = commit_block(accepted, round_number, producer, verifier, miner_key)
        elapsed_ns.append(time.perf_counter_ns() - start)
        if capture:
            for row in decisions:
                row.update(config=config, round=round_number)
                row["outcome"] = "accepted" if row["accepted"] and block_ok else "rejected" if not row["accepted"] else "unprocessed"
                events.append(row)
        if not block_ok:
            raise RuntimeError("Normal block construction/verification failed")
    return events, sum(elapsed_ns) / len(elapsed_ns) / 1e6


def self_test():
    validator_key, miner_key = RSA.generate(1024), RSA.generate(1024)
    for scenario in ("H1", "H3"):
        batches = fixtures(SEEDS[0], scenario, validator_key, 2, 3)
        previous = None
        for config in CONFIGS:
            rows, _ = process(config, batches, miner_key, capture=True)
            normals = [r for r in rows if r["kind"] == "normal"]
            attacks = [r for r in rows if r["kind"] == "attack"]
            assert all(r["outcome"] == "accepted" for r in normals)
            assert all(r["outcome"] == ("accepted" if config == "NOBC-original" else "rejected") for r in attacks)
            if config != "NOBC-original":
                decisions = [(r["event_id"], r["outcome"], r["reason"]) for r in rows]
                if previous is not None:
                    assert decisions == previous
                previous = decisions
    # Independent fixtures validate actual block-hash/signature/tip checks.
    producer, observer = Verifier(miner_key, "producer_0"), Verifier(miner_key, "observer_0")
    tx = fixtures(SEEDS[0], "H1", validator_key, 1, 1)[0][1][0][2]
    assert commit_block([tx], 2, producer, observer, miner_key)
    block = copy.deepcopy(observer.chain.return_last_block())
    block.set_previous_block_hash("tampered")
    assert observer.verify_block(block, producer.idx)[0] is False
    print("Self-tests passed: normal, H1, H3, matched decisions, block integrity.")


def run(args):
    result_dir = ROOT / "results"
    result_dir.mkdir(exist_ok=False)
    events_out, timing_out, summaries = [], [], []
    # Ephemeral keys; the same keys and signed input objects serve all configs.
    validator_key, miner_key = RSA.generate(1024), RSA.generate(1024)
    for scenario in ("H1", "H3"):
        for seed in SEEDS:
            batches = fixtures(seed, scenario, validator_key, args.rounds, args.normal_count)
            paired = {}
            for config in CONFIGS:
                rows, _ = process(config, batches, miner_key, capture=True)
                for row in rows:
                    row.update(seed=seed, scenario=scenario)
                events_out.extend(rows)
                paired[config] = [(r["event_id"], r["outcome"], r["reason"]) for r in rows]
                counts = Counter((r["kind"], r["outcome"]) for r in rows)
                summaries.append({"scenario": scenario, "seed": seed, "config": config,
                                  "attack_events": sum(v for (k, _), v in counts.items() if k == "attack"),
                                  "attack_rejected": counts["attack", "rejected"],
                                  "attack_accepted": counts["attack", "accepted"],
                                  "attack_unprocessed": counts["attack", "unprocessed"],
                                  "normal_events": sum(v for (k, _), v in counts.items() if k == "normal"),
                                  "normal_rejected": counts["normal", "rejected"],
                                  "normal_unprocessed": counts["normal", "unprocessed"]})
            assert paired["NOBC-matched"] == paired["BC-matched"]
            # Warmups and randomized/interleaved configs reduce ordering bias.
            for _ in range(args.warmups):
                for config in CONFIGS:
                    process(config, batches, miner_key)
            order_rng = random.Random(seed + (1 if scenario == "H1" else 2))
            for repetition in range(args.repetitions):
                order = list(CONFIGS)
                order_rng.shuffle(order)
                for config in order:
                    _, ms = process(config, batches, miner_key)
                    timing_out.append({"scenario": scenario, "seed": seed, "config": config,
                                       "technical_repetition": repetition, "mean_ms_per_round": ms})
    assert all(s["attack_unprocessed"] == s["normal_rejected"] == s["normal_unprocessed"] == 0 for s in summaries)
    for name, rows in [("events.jsonl", events_out), ("timings.jsonl", timing_out), ("summary.jsonl", summaries)]:
        (result_dir / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    methods = {}
    for p in SNAPSHOT.glob("*.py"):
        methods[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    metadata = {
        "experiment": "isolated_matched_H1_H3_protocol_microbenchmark",
        "seeds": list(SEEDS), "rounds_per_scenario_seed": args.rounds,
        "normal_transactions_per_round": args.normal_count, "attack_events_per_round": 1,
        "warmups": args.warmups, "technical_repetitions": args.repetitions,
        "rsa_bits": 1024, "keys": "ephemeral; never saved; shared across all configurations",
        "payload": "16 synthetic float values plus validator metadata; not DDPM parameters",
        "timing_includes": "round/freshness checks and validator-signature verification; BC also constructs, signs, verifies and appends one block with accepted transactions using existing prototype methods",
        "timing_excludes": "fixture/signature generation, key generation, logging, training, worker verification, aggregation computation, network propagation, multi-miner voting/consensus and model application",
        "replay_check_scope": "stale-round transactions only; not same-round duplicates",
        "original_NOBC_scope": "post-validator direct application, not a claim that worker signatures are never checked",
        "audit_scope": "external experiment JSONL; not tamper-evident on-chain audit",
        "python": sys.version, "platform": platform.platform(), "source_snapshot_sha256": methods,
    }
    (result_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"summary_rows": len(summaries), "event_rows": len(events_out), "timing_rows": len(timing_out), "results": str(result_dir)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--normal-count", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()
    if min(args.rounds, args.normal_count, args.repetitions) < 1 or args.warmups < 0:
        parser.error("Counts must be positive; warmups must be nonnegative")
    if args.self_test:
        self_test()
    else:
        run(args)
