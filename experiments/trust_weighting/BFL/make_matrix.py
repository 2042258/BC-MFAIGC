"""Generate run manifests for the calibration, sweep and formal matrix stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COMMON_10_NODE = [
    "-nd", "10",
    "-ha", "*,*,*",
    "-aio", "1",
    "-pow", "0",
    "-ko", "6",
    "-cs", "0",
    "-B", "128",
    "-mn", "ddpm",
    "-iid", "0",
    "-lr", "0.001",
    "-dtx", "1",
    "-le", "1",
    "-v", "0",
    "--validation_seed", "9001",
    "--test_batchsize", "256",
    "--save_global_checkpoints", "1",
    "--checkpoint_frequency", "20",
]


def build_manifest(stage: str, args) -> dict:
    runs = []
    if stage == "calibration":
        for index in range(args.seeds):
            seed = args.seed_start + index
            runs.append({
                "id": f"calib-{seed}",
                "tag": "calibration",
                "args": [
                    "-max_ncomm", str(args.rounds),
                    "-nm", "0",
                    "-vh", "999",
                    "--seed", str(seed),
                    "--blockchain_on", "1",
                    "--defense_on", "1",
                ],
            })
    elif stage == "sweep":
        for scale in args.noise_scales:
            for index in range(args.seeds):
                seed = args.seed_start + index
                runs.append({
                    "id": f"sweep-s{scale}-{seed}",
                    "tag": f"sweep_sigma_{scale}",
                    "args": [
                        "-max_ncomm", str(args.rounds),
                        "-nm", "1",
                        "-nv", str(scale),
                        "-vh", str(args.threshold),
                        "--seed", str(seed),
                        "--blockchain_on", "1",
                        "--defense_on", "1",
                    ],
                })
    elif stage == "tra":
        for index in range(args.seeds):
            seed = args.seed_start + index
            runs.append({
                "id": f"tra-clean-{seed}",
                "tag": "tra_clean",
                "args": [
                    "--epochs", str(args.epochs),
                    "--batchsize", "128",
                    "--test-batchsize", "256",
                    "--seed", str(seed),
                    "--validation-seed", "9001",
                    "--noise-probability", "0",
                    "--noise-scale", str(args.noise_scale),
                ],
            })
        for malicious_nodes in args.malicious_nodes:
            for index in range(args.seeds):
                seed = args.seed_start + 1000 + malicious_nodes * 100 + index
                runs.append({
                    "id": f"tra-contam-nm{malicious_nodes}-{seed}",
                    "tag": f"tra_contam_nm{malicious_nodes}",
                    "args": [
                        "--epochs", str(args.epochs),
                        "--batchsize", "128",
                        "--test-batchsize", "256",
                        "--seed", str(seed),
                        "--validation-seed", "9001",
                        "--noise-probability", str(malicious_nodes / 10.0),
                        "--noise-scale", str(args.noise_scale),
                    ],
                })
    elif stage in {"uptime", "noniid_validation"}:
        test_shard = stage == "noniid_validation"
        network_args = ["--network_stability", "0.8"] if stage == "uptime" else []
        methods = [
            ("bc1_def1", ["--blockchain_on", "1", "--defense_on", "1"]),
            ("bc0_def1", ["--blockchain_on", "0", "--defense_on", "1"]),
            ("bc0_def0", ["--blockchain_on", "0", "--defense_on", "0"]),
        ] if stage == "uptime" else [
            ("bc1_def1", ["--blockchain_on", "1", "--defense_on", "1"]),
            ("bc0_def1", ["--blockchain_on", "0", "--defense_on", "1"]),
        ]
        malicious_levels = [0, 1, 3] if stage == "uptime" else [0, 1]
        for malicious_nodes in malicious_levels:
            for condition, mode_args in methods:
                for index in range(args.seeds):
                    seed = args.seed_start + malicious_nodes * 1000 + index
                    runs.append({
                        "id": f"{stage}-nm{malicious_nodes}-{condition}-{seed}",
                        "tag": f"{stage}_nm{malicious_nodes}_{condition}",
                        "args": [
                            "-max_ncomm", str(args.rounds),
                            "-nm", str(malicious_nodes),
                            "-nv", str(args.noise_scale),
                            "-vh", str(args.threshold),
                            "--seed", str(seed),
                            *network_args,
                            "--shard_test_data", "1" if test_shard else "0",
                            *mode_args,
                        ],
                    })
    elif stage == "matrix":
        conditions = [
            ("bc1_def1", ["--blockchain_on", "1", "--defense_on", "1"]),
            ("bc0_def1", ["--blockchain_on", "0", "--defense_on", "1"]),
            ("bc0_def0", ["--blockchain_on", "0", "--defense_on", "0"]),
        ]
        if args.include_chain_no_filter:
            conditions.append(("bc1_def0", ["--blockchain_on", "1", "--defense_on", "0"]))
        for malicious_nodes in args.malicious_nodes:
            for condition, mode_args in conditions:
                for index in range(args.seeds):
                    seed = args.seed_start + index + malicious_nodes * 1000
                    runs.append({
                        "id": f"matrix-nm{malicious_nodes}-{condition}-{seed}",
                        "tag": f"nm{malicious_nodes}_{condition}",
                        "args": [
                            "-max_ncomm", str(args.rounds),
                            "-nm", str(malicious_nodes),
                            "-nv", str(args.noise_scale),
                            "-vh", str(args.threshold),
                            "--seed", str(seed),
                            *mode_args,
                        ],
                    })
        clean_conditions = [
            ("bc1_def1", ["--blockchain_on", "1", "--defense_on", "1"]),
            ("bc0_def1", ["--blockchain_on", "0", "--defense_on", "1"]),
            ("bc0_def0", ["--blockchain_on", "0", "--defense_on", "0"]),
        ]
        if args.include_chain_no_filter:
            clean_conditions.append(("bc1_def0", ["--blockchain_on", "1", "--defense_on", "0"]))
        for condition, mode_args in clean_conditions:
            for index in range(args.seeds):
                seed = args.seed_start + 9000 + index
                runs.append({
                    "id": f"matrix-clean-{condition}-{seed}",
                    "tag": f"clean_{condition}",
                    "args": [
                        "-max_ncomm", str(args.rounds),
                        "-nm", "0",
                        "-vh", str(args.threshold),
                        "--seed", str(seed),
                        *mode_args,
                    ],
                })
    else:
        raise SystemExit(f"unknown stage: {stage}")

    return {
        "stage": stage,
        "rounds": args.rounds,
        "threshold": args.threshold,
        "noise_scales": args.noise_scales,
        "noise_scale": args.noise_scale,
        "common": COMMON_10_NODE,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["calibration", "sweep", "matrix", "tra", "uptime", "noniid_validation"])
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=11000)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--noise-scale", type=float, default=0.02)
    parser.add_argument("--noise-scales", type=float, nargs="*", default=[0.005, 0.01, 0.015, 0.02, 0.03])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--malicious-nodes", type=int, nargs="*", default=[1, 3, 6])
    parser.add_argument("--include-chain-no-filter", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(args.stage, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest['runs'])} runs to {args.output}")


if __name__ == "__main__":
    main()
