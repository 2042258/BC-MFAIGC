# BC-MFAIGC

Research implementation of federated DDPM training with validation-guided aggregation and simulated blockchain coordination.

## Code layout

| Directory | Responsibility |
|---|---|
| `bcmfaigc/data/` | CIFAR-10 loading/partitioning, MNIST readers and transforms |
| `bcmfaigc/models/` | DDPM and its denoising network; registered checkpoint keys are retained |
| `bcmfaigc/training/` | Local model optimization and centralized reference training |
| `bcmfaigc/validation.py` | Reference-model preparation and model-update loss evaluation |
| `bcmfaigc/aggregation/` | Trust weights, distinct block/direct decisions, weighted sums and Krum selection |
| `bcmfaigc/blockchain/` | Blocks, chain storage, deterministic transaction encoding and payload sizing |
| `bcmfaigc/federation/` | One node/network implementation, small legacy compatibility profiles and trusted snapshot loading |
| `bcmfaigc/simulation/` | Argument parsing, initialization, round scheduling and persistence |
| `configs/` | Original experiment matrices, frozen calibration and single-run examples |
| `experiments/` | Experiment-specific launchers and historical import/checkpoint adapters |
| `analysis/` | Shared collection, calibration, statistics and plotting; unique old plotting work is under `legacy_plots/` |
| `supplemental/` | Retained centralized prototypes, vertical-FL work and early diagnostics |
| `tests/` | Numerical, state/RNG, path, compatibility, policy and persistence contracts |

There is one active node/network implementation and one shared simulation loop, not separate copied engines for each experiment. Compatibility profiles retain real distinctions in attack handling, supported aggregation modes, Krum logging and protocol audit events. The `BFL/` and `models/Models.py` files under experiment directories preserve trusted historical import paths; they are not independent model or training implementations.

## Installation

Use Python 3.12 or newer in a dedicated environment, with an appropriate PyTorch build for your hardware. `environment/requirements-captured.txt` records the original dependency versions; it is not a newly validated portable environment lock.

```bash
python -m pip install -e .
```

In an already prepared environment, `python -m pip install --no-deps -e .` installs only this project. The documented commands use an **editable source checkout**, so experiment/configuration files remain available. No dataset, trained checkpoint, original experiment log, matched reference artifact or Inception weight is bundled.

## Run one experiment

Place CIFAR-10 Python batches in `WORKSPACE/data/cifar-10-batches-py/` (`data_batch_1` through `data_batch_5` and `test_batch`). The source does not download the dataset.

```bash
bcmfaigc-train --config configs/examples/classic_comparison.json --workspace WORKSPACE
```

Examples reproduce the argument list of one selected original run; inspect its run ID before launching. They are not reduced-budget smoke configurations. CLI arguments after the config override earlier occurrences, for example `--seed 17 -max_ncomm 2 -le 1`. Such overrides are exploratory runs, not the paper's matched protocol.

Select exactly one original matrix entry:

```bash
bcmfaigc-train --profile extended \
  --config configs/byzantine_ratio/BFL/ratio_scan_manifest_v1.json \
  --run-id ratio-nm0-fedavg-50000 --workspace WORKSPACE
```

The command never implicitly launches an entire matrix. Supported profiles are `classic`, `extended` and `krum`. Classic preserves the original Gaussian-only workflow; extended adds explicit attack selection and Multi-Krum; Krum adds strict single-selection Krum. Legacy experiment entrypoints still support their original CLI arguments and BFL-relative data/log working-directory behavior.

Output locations are `WORKSPACE/logs/blogs/`, `WORKSPACE/BFL/snapshots/`, and the run's `checkpoints/` directory. Network snapshots are Python pickle files: **only load trusted snapshots**. Historical `models.Models`, `Device`, `DatasetLoad`, `Block` and `Blockchain` class references have explicit compatibility resolution. Resume still follows the original path/naming and provenance limitations described in `docs/SOURCE_PROVENANCE.md`.

## Sampling, FID and analysis

```bash
python experiments/byzantine_ratio/BFL/evaluate_ddpm.py --help
bcmfaigc-fid samples.pt --data-root DATA_ROOT --output fid.json
bcmfaigc-analyze collect_comparison_runs --project-root WORKSPACE -- \
  --manifest configs/comparison/phase1_validation/formal_experiment_manifest_core_v1_released_20260926.json \
  --output runs.jsonl
```

Use `collect_comparison_runs` for the original comparison collector schema. Supply
`--manifest` for formal/new run IDs; without it the collectors retain historical
recognized-ID filtering and can skip unknown tags. Vote-level, worker-round, final aggregation and application statistics remain distinct. Existing reference-calibration and plotting inputs are not supplied; see `docs/PREPARATION.md` and `analysis/README.md`.

Historical Krum/Soft matched-audit scripts retain exact-source and artifact assertions. They are **not general launchers for this reorganized source** and must not be treated as current-release acceptance checks. Use `bcmfaigc-train` for fresh configured runs. The protocol microbenchmark keeps its immutable source snapshot and is a synthetic protocol workload, not full-training timing.

## Verification and limits

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
python tools/check_package.py
```

Tests skip optional dependency-dependent cases when their libraries are unavailable. Real-dependency CPU tests and bounded synthetic fresh-process simulations were also exercised during organization; the detailed accepted scope is in `docs/SOURCE_PROVENANCE.md`. These checks do **not** certify full-data/GPU/FID results, exhaustive resume/offline/fork behavior or paper reproduction.

Fresh runs retain actual source identities rather than substituting historical hashes. Original reference hashes, scientific configurations, selection IDs and strict historical audit assertions have not been relaxed.

## Publication prerequisites

This folder is a publication candidate, not proof of a public release. Original external URLs were removed by user instruction and remain explicit placeholders. Supply the required local data/artifacts, confirm source ownership/licensing, and provide a real public archive/repository address and stable release reference before claiming public-code availability. No license grant or publication URL is invented by this cleanup. The manuscript and response letter have not been edited.
