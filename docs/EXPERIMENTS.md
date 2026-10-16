# Experiment map

Manifest paths are project-relative; runner paths are relative to the experiment root. Counts describe selected configurations, not fresh completed executions. Exact selected IDs are in `experiment_index.json`.

| Workflow | Experiment root | Existing runner | Configuration | Selected runs |
|---|---|---|---|---:|
| stage1_core | `experiments/comparison` | `BFL/run_matrix.py` | `configs/comparison/phase1_validation/formal_experiment_manifest_core_v1_released_20260926.json` | 18 |
| stage1_hard_ablation | `experiments/comparison` | `BFL/run_matrix.py` | `configs/comparison/phase1_validation/formal_experiment_manifest_ablation_v1_released_20260927.json` | 12 |
| central_reference | `experiments/comparison` | `BFL/run_central_matrix.py` | `configs/comparison/phase1_validation/formal_experiment_manifest_reference_v1_released_20260927.json` | 1 |
| stage2_ratio_scan | `experiments/byzantine_ratio` | `BFL/run_matrix.py` | `configs/byzantine_ratio/BFL/ratio_scan_manifest_v1.json` | 36 |
| stage2_sign_inversion | `experiments/byzantine_ratio` | `BFL/run_matrix.py` | `configs/byzantine_ratio/BFL/incremental_manifest_v2.json` | 6 |
| stage2_protocol_h0_h3 | `experiments/byzantine_ratio` | `BFL/run_matrix.py` | `configs/byzantine_ratio/BFL/incremental_manifest_v2.json` | 12 |
| krum_supplement | `experiments/krum` | `run_pipeline.py` | `configs/krum/krum_manifest.json` | 6 |
| soft_stage2_supplement | `experiments/trust_weighting` | `run_soft_pipeline.py` | `configs/trust_weighting/soft_manifest.json` | 12 |

For fresh single-case federated runs, prefer `bcmfaigc-train --profile classic|extended|krum --config MANIFEST --run-id ID --workspace WORKSPACE`. Centralized matrices require their centralized runner. Krum/Soft existing pipeline scripts are strict historical matched audits, not current-source launchers.

## Scientific distinctions retained

- Original comparison core/ablation: random role assignment, signature checking disabled, threshold approximately 0.0877 and medium-strength Gaussian noise scale 0.02.
- Byzantine-ratio scan: ten nodes, roles 5/2/3, signatures enabled, malicious counts 0/1/3/5 and threshold 0.013486012299855549. Historical 0/10/30/60-percent matrices are not substitutes.
- The incremental manifest retains its `frozen_incremental_calibration_p99` status. Only selected `inc_signflip*` or `inc_protocol_h*` IDs describe those subsets; the entire incremental file also includes other cases.
- Strict single-selection Krum is a profile of the shared node implementation, not an isolated copied engine. Original Multi-Krum behavior remains distinct.
- Trust-weighting configuration and its exact matched-control assertions remain separate from hard filtering.
- `experiments/protocol_control/` consumes `source_snapshot/` as an intentional immutable runtime dependency. It is a synthetic protocol benchmark, not DDPM round timing.

Unique historical/supplemental directions remain in `analysis/legacy_plots/` and `supplemental/`. Historical shell supervisors still depend on specific completed artifacts and paths; they are not one-command paper reproduction workflows. No results or independent-replicate counts were merged across these categories.
