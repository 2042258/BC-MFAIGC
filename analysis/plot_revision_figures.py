"""Generate the revised paper figures from the collected experiment reports."""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_LABEL = {
    "bc1_def1": "BC-MFAIGC (chain+filter)",
    "bc0_def1": "No-chain + filter",
    "bc0_def0": "Plain FedAvg",
}
METHOD_STYLE = {
    "bc1_def1": {"color": "#2E8B57", "marker": "D"},
    "bc0_def1": {"color": "#8E44AD", "marker": "s"},
    "bc0_def0": {"color": "#D1495B", "marker": "o"},
}
RATIO_LABEL = {0: "0%", 1: "10%", 3: "30%", 6: "60%"}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def series_xy(series: dict, tag: str):
    rounds = sorted((int(round_no) for round_no in series.get(tag, {})))
    means = [series[tag][str(round_no)]["mean"] for round_no in rounds]
    stds = [series[tag][str(round_no)]["std"] for round_no in rounds]
    return (rounds, means, stds)


def main(*, project_root=None) -> None:
    project_root = Path.cwd() if project_root is None else Path(project_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports",
        type=Path,
        default=project_root / "phase1_validation/reports/final_reports.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=project_root / "phase1_validation/figures"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = json.loads(args.reports.read_text(encoding="utf-8"))
    federated = reports["figures"]["per_round_federated"]
    central = reports["figures"]["per_epoch_central"]
    final_federated = reports["figures"]["final_federated"]
    final_central = reports["figures"]["final_central"]
    detection = reports["detection"]
    fid = reports["fid"]
    timing = reports["timing_single_tenant"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for method, style in METHOD_STYLE.items():
        xs, ys, es = ([], [], [])
        for ratio in (0, 1, 3, 6):
            tag = f"clean_{method}" if ratio == 0 else f"nm{ratio}_{method}"
            value = final_federated.get(tag)
            if value is None:
                continue
            xs.append(ratio * 10)
            ys.append(value["mean"])
            es.append(value["std"])
        ax.errorbar(xs, ys, yerr=es, label=METHOD_LABEL[method], capsize=3, **style)
    if final_central:
        tra_mean = sum((v["mean"] for v in final_central.values())) / len(final_central)
        ax.axhline(
            tra_mean,
            linestyle=":",
            color="#2C6E91",
            label="Centralized DDPM (budget-matched)",
        )
    ax.set_xlabel("Malicious nodes (%)")
    ax.set_ylabel("Global ε-prediction loss (round 100)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "figure_10.pdf")
    fig.savefig(args.output_dir / "figure_10.png", dpi=200)
    plt.close(fig)

    def curve_figure(prefix: str, tags_by_ratio: dict, title: str, xlabel: str):
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        colors = plt.cm.viridis([0.1, 0.4, 0.65, 0.9])
        for (ratio, tag), color in zip(sorted(tags_by_ratio.items()), colors):
            rounds, means, stds = series_xy(federated, tag)
            if not rounds:
                continue
            means = list(means)
            stds = list(stds)
            ax.plot(rounds, means, color=color, label=RATIO_LABEL[ratio])
            ax.fill_between(
                rounds,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                color=color,
                alpha=0.15,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Global ε-prediction loss")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, title="Malicious nodes")
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{prefix}.pdf")
        fig.savefig(args.output_dir / f"{prefix}.png", dpi=200)
        plt.close(fig)

    curve_figure(
        "figure_13",
        {
            ratio: f"{('clean' if r == 0 else 'nm' + str(r))}_bc0_def0"
            for r, ratio in [(0, 0), (1, 1), (3, 3), (6, 6)]
        },
        "Plain FedAvg",
        "Communication round",
    )
    curve_figure(
        "figure_14",
        {
            ratio: f"{('clean' if r == 0 else 'nm' + str(r))}_bc1_def1"
            for r, ratio in [(0, 0), (1, 1), (3, 3), (6, 6)]
        },
        "BC-MFAIGC",
        "Communication round",
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = plt.cm.viridis([0.1, 0.4, 0.65, 0.9])
    for (ratio, tag), color in zip(
        [
            (0, "tra_clean"),
            (10, "tra_contam_nm1"),
            (30, "tra_contam_nm3"),
            (60, "tra_contam_nm6"),
        ],
        colors,
    ):
        rounds, means, stds = series_xy(central, tag)
        if not rounds:
            continue
        ax.plot(rounds, means, color=color, label=f"{ratio}% corrupted updates")
    ax.set_xlabel("Centralized training epoch")
    ax.set_ylabel("Validation ε-prediction loss")
    ax.set_title("Centralized DDPM (budget-matched)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "figure_12.pdf")
    fig.savefig(args.output_dir / "figure_12.png", dpi=200)
    plt.close(fig)
    panels = [1, 3, 6]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, ratio in zip(axes, panels):
        for method, style in METHOD_STYLE.items():
            rounds, means, _ = series_xy(federated, f"nm{ratio}_{method}")
            if not rounds:
                continue
            ax.plot(
                rounds,
                means,
                label=METHOD_LABEL[method],
                linewidth=1.2,
                color=style["color"],
                marker=style["marker"],
                markersize=2,
            )
        ax.set_title(f"{ratio * 10}% malicious nodes", fontsize=9)
        ax.set_xlabel("Communication round")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Global ε-prediction loss")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output_dir / "figure_11.pdf")
    fig.savefig(args.output_dir / "figure_11.png", dpi=200)
    plt.close(fig)
    sweep = reports["sigma_selection"]["table"]
    if sweep:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        sigmas = [row["sigma"] for row in sweep]
        tpr = [row["tpr_mean"] for row in sweep]
        err = [row["tpr_std"] or 0 for row in sweep]
        ax.errorbar(sigmas, tpr, yerr=err, marker="o", color="#B03A2E", capsize=3)
        ax.axvline(
            reports["sigma_selection"]["selected"]["sigma"],
            linestyle="--",
            color="grey",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Weight-noise scale σ (log scale)")
        ax.set_ylabel("Malicious-update detection rate (TPR)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.output_dir / "figure_15_detection.pdf")
        fig.savefig(args.output_dir / "figure_15_detection.png", dpi=200)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    labels, values, errors = ([], [], [])
    for method in METHOD_STYLE:
        for ratio in (0, 1, 3, 6):
            tag = f"clean_{method}" if ratio == 0 else f"nm{ratio}_{method}"
            entry = fid.get(tag)
            if entry and entry["mean"] is not None:
                labels.append(
                    f"{METHOD_LABEL[method].split(' (')[0]}\n{RATIO_LABEL[ratio]}"
                )
                values.append(entry["mean"])
                errors.append(entry["std"])
    ax.bar(range(len(values)), values, yerr=errors, capsize=3, color="#4C72B0")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("FID (2000 samples, ↓ better)")
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    budget = (
        read_jsonl(args.output_dir.parent / "reports" / "budget_fid.jsonl")
        if False
        else None
    )
    curve_path = project_root / "phase1_validation/reports/budget_fid.json"
    if curve_path.exists():
        curve = json.loads(curve_path.read_text())
        epochs = sorted((int(k) for k in curve))
        ax.plot(epochs, [curve[str(e)] for e in epochs], marker="o", color="#2E8B57")
        ax.set_xlabel("Centralized training epochs")
        ax.set_ylabel("FID")
        ax.set_title("FID vs training budget (reference)", fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "figure_16_fid.pdf")
    fig.savefig(args.output_dir / "figure_16_fid.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    names = [
        "worker_local_training_seconds",
        "validator_reference_seconds",
        "validation_seconds",
        "modelled_transmission_seconds",
        "block_generation_seconds",
        "round_wall_seconds",
    ]
    labels = [
        "Local training",
        "Validator ref.",
        "Validation",
        "Model transfer (simulated)",
        "Block generation (modelled)",
        "Round wall clock",
    ]
    values, errors = ([], [])
    for name in names:
        entry = timing.get("clean_bc1_def1", {}).get(name) or timing.get(
            "clean_bc0_def1", {}
        ).get(name)
        values.append(entry["mean"] if entry else 0.0)
        errors.append(entry["std"] if entry else 0.0)
    ax.barh(range(len(values)), values, xerr=errors, capsize=3, color="#4C72B0")
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Seconds (log scale)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(args.output_dir / "figure_17_timing.pdf")
    fig.savefig(args.output_dir / "figure_17_timing.png", dpi=200)
    plt.close(fig)
    print(f"figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
