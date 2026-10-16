# Analysis

These modules analyze existing outputs and do not implement federated training.

```bash
bcmfaigc-analyze collect_runs --project-root WORKSPACE -- --output runs.jsonl
bcmfaigc-analyze analyze_threshold --project-root WORKSPACE -- --runs runs.jsonl --output threshold.json
bcmfaigc-analyze analyze_matrix --project-root WORKSPACE -- --runs runs.jsonl --output timing.json
```

Pass `--manifest MANIFEST.json` after `--` to collect formal/new run IDs using their
configured tags. Without a manifest the collectors retain historical recognized-ID
filtering, so unknown tags can be skipped. For strict formal collection use
`python -m analysis.collect_formal_runs --logs-dir WORKSPACE/logs/blogs --manifests MANIFEST.json --output runs.jsonl`.

Use `collect_comparison_runs` for the original comparison schema; `collect_runs` additionally retains protocol-event and Multi-Krum fields. The two are intentionally distinguished rather than silently changing historical output schemas. Both preserve separate validator-vote, known final worker-round decision, and applied-update denominators.

`build_reports`, `export_latex_tables` and `plot_revision_figures` retain the original report schemas and experiment-specific input filenames. They require the corresponding collected inputs under the selected workspace; no results are invented or bundled. Frozen calibration inputs are under `configs/comparison/phase1_validation/`. The formal released-results tools are `collect_formal_runs.py` and `plot_formal_results.py`; their original dated output names are retained.

`check_fid_control.py` is a separate real-Inception control calculation, not a help-only or lightweight check. It requires local data and pretrained weights; do not run it as part of generic CLI smoke tests.

Unique historical figure/data-processing scripts remain under `legacy_plots/`. They are retained research material, not guaranteed portable standalone commands: inspect their explicit path placeholders and data-column assumptions before use. Their plots and statistical definitions were not arbitrarily merged merely because filenames look similar.

Compatibility calls under experiment directories delegate here and provide their original workspace root. New calls should use explicit `--project-root` through `bcmfaigc-analyze`, or pass `project_root=` to module functions.
