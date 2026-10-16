# Method comparison and ablation

This directory retains original import/checkpoint adapters and batch launchers. The active implementation is shared under `bcmfaigc/`; analysis is under `analysis/`, configuration under `configs/comparison/`.

From the repository root, one full original comparison case can be selected with:

```bash
bcmfaigc-train --config configs/examples/classic_comparison.json --workspace WORKSPACE
```

CIFAR batch files must be under `WORKSPACE/data/cifar-10-batches-py/`. For batch selection, exact IDs, central reference and calibration distinctions, see `docs/EXPERIMENTS.md` and `docs/PREPARATION.md`. Use the `classic` profile, not the extended or Krum profile, for original comparison cases.

To collect formal core cases:

```bash
python -m analysis.collect_formal_runs \
  --logs-dir WORKSPACE/logs/blogs \
  --manifests configs/comparison/phase1_validation/formal_experiment_manifest_core_v1_released_20260926.json \
  --output WORKSPACE/core_runs.jsonl
```

The shared collectors also accept `--manifest` to obtain formal run tags. Without a manifest, they retain historical recognized-ID filtering and can skip unknown tags.

Unique historical plots moved to `analysis/legacy_plots/`; centralized/vertical-FL prototypes moved to `supplemental/`. The original redacted README is retained at `docs/historical/comparison_original_readme.md` for historical attribution/context; its old path/environment instructions are not current launch instructions.
