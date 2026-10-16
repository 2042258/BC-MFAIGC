# Experiment configurations

Original JSON configuration bytes were moved here without changing scientific parameters, run IDs, reference hashes or matched-artifact assertions. The original internal categories (`BFL`, `phase1_validation`) distinguish matrices that can have similar basenames; they do not define library architecture.

- `comparison/`: original core comparison, ablation, centralized reference and calibration matrices.
- `byzantine_ratio/`: frozen ratio scan and incremental/sign-flip/protocol configurations.
- `krum/`: original six-run matched Krum audit configuration.
- `trust_weighting/`: original twelve-run matched Soft audit configuration.
- `examples/`: one selected run's arguments copied from a source matrix, with its profile, source manifest and run ID recorded.

The threshold, role counts, seeds, sample counts and differing reference/calibration histories are preserved. These are not automatically interchangeable matrices. `experiment_index.json` records exact selected IDs and project-relative manifest paths.

Run exactly one federated case:

```bash
bcmfaigc-train --config configs/examples/single_krum.json --workspace WORKSPACE
bcmfaigc-train --profile extended --config configs/byzantine_ratio/BFL/ratio_scan_manifest_v1.json --run-id ratio-nm0-fedavg-50000 --workspace WORKSPACE
```

`bcmfaigc-train` never automatically launches every case in a matrix. Centralized reference matrices must use centralized-training entrypoints, not the federated command. Historical matched-audit drivers additionally require unbundled raw reference artifacts and exact historical source conditions; reorganized-source numerical checks do not replace those assertions.
