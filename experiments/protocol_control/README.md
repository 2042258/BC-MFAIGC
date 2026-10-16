# Reviewer 3, Comment 1: Matched H1/H3 Protocol Control

## Scope

This is an isolated protocol-method microbenchmark, not a rerun of the DDPM training pipeline. It addresses whether the controlled H1/H3 rejection paths require a blockchain when the non-blockchain path has the same validator-signature and transaction-round checks.

The original experiment sources were backed up in `source_snapshot/` before implementation. Existing training code, historical results, remote files and manuscript sources were not edited.

## Configurations

- **NOBC-original**: unchecked post-validator direct-application path. This does not imply that the original NOBC never verifies worker signatures.
- **NOBC-matched**: the same stale-round predicate as `main.py` followed by the existing `Device.verify_validator_transaction` method.
- **BC-matched**: identical transaction checks; accepted transactions are additionally placed in one block per round, hashed, signed, verified and appended using existing Block/Blockchain/Device methods.

Existing methods are loaded from a fixed source snapshot without importing or running the training entry point. Block/Blockchain and stable hashing modules are used directly. Device method bodies are extracted with Python AST; verification logic is not replaced. Console prints are disabled consistently. The round predicate is reproduced from the original main.py miner path.

## Workload and Counting

- Scenarios: H1 signed aggregation-weight mutation (1.0 to 0.25 after signing); H3 replay of a valid previous-round transaction.
- Three fixture seeds: 47000, 47001, 47002.
- Nine measured rounds per scenario/seed, 50 normal transactions and one injected attack per round.
- The exact same signed inputs and ephemeral test keys are shared across configurations. Keys and signed payloads are never persisted.
- Update payloads contain 16 synthetic float values, not model tensors.
- Per scenario/configuration: 27 unique injected events and 1,350 normal transactions.
- Outcomes are counted per injected event, not per observer rejection log. Rejected, accepted and unprocessed outcomes are distinguished.

## Results

For both H1 and H3:

| Configuration | Rejected attacks | Accepted attacks | Unprocessed attacks | Normal rejections |
|---|---:|---:|---:|---:|
| NOBC-original | 0/27 | 27/27 | 0/27 | 0/1350 |
| NOBC-matched | 27/27 | 0/27 | 0/27 | 0/1350 |
| BC-matched | 27/27 | 0/27 | 0/27 | 0/1350 |

The matched transaction decisions and reasons were identical for each paired input. For these controlled signed-field tampering and stale-round replay cases, adding a blockchain did not provide additional rejection capability over identical basic checks. This is not a claim of general security equivalence.

## Timing

Two warmups and 20 randomized/interleaved technical repetitions were used per fixture seed. The figure reports the mean of three seed-level medians, with min–max whiskers; technical repetitions are not additional independent seeds.

| Scenario | NOBC-original | NOBC-matched | BC-matched |
|---|---:|---:|---:|
| H1 | 0.00678 ms/round | 5.47594 ms/round | 22.81625 ms/round |
| H3 | 0.00664 ms/round | 5.36486 ms/round | 22.89144 ms/round |

BC timing includes local block construction, hashing, signing, verification and append. Key generation, fixture/validator-signature generation, logging, worker verification, model training, numerical aggregation, model application, communication, multi-miner scheduling and consensus are excluded. These values must not replace or be directly compared with the original full training-round measurements.

## Evidence Boundary

- The replay guard covers stale rounds, not arbitrary same-round duplicate transactions.
- No historical-ledger tampering, Byzantine consensus, Sybil resistance, distributed ledger consistency or tamper-evident logging was evaluated.
- No FID was measured; no model was trained.
- H2-like block mutation is used only as a harness self-test, not a newly reported security experiment.
- Existing prototype RSA/hash methods were reused for fidelity, not presented as production cryptographic validation.

## Reproduction

Use an environment containing PyTorch, PyCryptodome, NumPy and Matplotlib:

```bash
PATH_REQUIRED_9 run_protocol_control.py --self-test
PATH_REQUIRED_9 run_protocol_control.py
PATH_REQUIRED_9 plot_protocol_control.py
```

`run_protocol_control.py` refuses to overwrite an existing `results/` directory. For a new independent run, use a separate experiment directory with the same scripts and source snapshot.

Artifacts: event outcomes, raw timings, summaries, metadata, figure-summary CSV and PDF/PNG/SVG figures. No new figure has been inserted into the manuscript.
