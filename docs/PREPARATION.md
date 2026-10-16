# Preparation and reproduction boundary

## Environment

Install this editable source checkout in a dedicated environment (`python -m pip install -e .`). Select a PyTorch build appropriate to your hardware. `environment/requirements-captured.txt` preserves the original Python 3.14.4 / CUDA 13.0 environment note and package pins; the captured environment is not a newly validated portable installation lock.

Existing batch runners now default to the active Python interpreter. `BFL_PYTHON` can explicitly override it. Do not copy private environment/authentication files into the publication folder.

## Local inputs

Provide the CIFAR-10 Python batch distribution:

```text
WORKSPACE/data/cifar-10-batches-py/
  data_batch_1 ... data_batch_5
  test_batch
  batches.meta
```

`bcmfaigc-train --workspace WORKSPACE` establishes the original `WORKSPACE/BFL/` working-directory convention. Legacy direct entrypoints still require their experiment's `BFL/` as working directory. FID instead accepts `--data-root WORKSPACE/data`; torchvision loads the test split with `download=False` and may require its complete distribution/integrity metadata. No dataset bytes or checksum is invented.

The source uses the test split for both validation scoring and the FID real-data reference. Reorganization preserves this design; it does not resolve the separate scientific question about evaluation-set overlap.

FID requires compatible pretrained `pytorch-fid` InceptionV3 weights, which are not bundled. The original comparison budget is 5,000 generated samples and the ratio-scan budget is 2,000, with 10,000 real reference images. Do not silently interchange these budgets. Model checkpoints, result logs and matched-control artifacts are also not bundled.

## Single configured run

From the repository root:

```bash
bcmfaigc-train --config configs/examples/classic_comparison.json --workspace WORKSPACE
```

This selects a full original case, not a cheap smoke test. For a deliberately reduced exploratory run, append `-max_ncomm 2 -le 1`; that is not matched paper reproduction. See `configs/README.md` for one-case matrix selection.

## Existing batch runners

After checking local inputs and compute budget, the original core batch runner can be called from `experiments/comparison/`:

```bash
python BFL/run_matrix.py \
  --manifest ../../configs/comparison/phase1_validation/formal_experiment_manifest_core_v1_released_20260926.json \
  --workers 1 --results phase1_validation/reproduced_core.jsonl \
  --timeout-seconds 7200
```

Use exact IDs in `experiment_index.json` and the runner's `--run-ids` option for subsets. Batch execution is not covered merely by testing the single-run command or its help output. Centralized reference matrices use `BFL/run_central_matrix.py`, not the federated command.

## Historical audits and placeholders

The Krum/Soft audit scripts retain argument, role-schedule and historical byte-hash assertions. They now fail early when their source conditions cannot match, rather than spending a full training run before rejecting. Use the new one-run command for fresh reorganized-source training. Supplying artifacts or relaxing old hashes is not part of this reorganization.

The protocol microbenchmark uses its bundled fixed source snapshot, synthetic payloads and separate timing workload. Its timing is not full DDPM-training timing.

Removed addresses and machine paths remain `URL_REQUIRED_N`, `PATH_REQUIRED_N` or `REMOTE_HOST_REQUIRED`. Numeric suffixes identify redacted occurrences, not experiment IDs. Inspect each use; do not replace every placeholder indiscriminately. `AUTHOR_METADATA_REMOVED` is redaction, not a license statement.

## Verified scope

See `SOURCE_PROVENANCE.md` for actual CPU tests and cold-process synthetic runs. These do not establish full-data GPU reproduction, Inception/FID results, exhaustive resume/protocol scenarios or legal/public-release readiness. Keep publication URLs, stable release identifiers and licensing decisions explicit rather than fabricated.
