"""Plot only the new isolated matched-protocol experiment results."""
from pathlib import Path
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CONFIGS = ("NOBC-original", "NOBC-matched", "BC-matched")
LABELS = ("Original NOBC", "NOBC + matched checks", "BC + matched checks")
SCENARIOS = ("H1", "H3")
COLORS = ("#8C8C8C", "#2878B5", "#E58B2A")
summary = [json.loads(x) for x in (ROOT / "results/summary.jsonl").read_text().splitlines()]
timings = [json.loads(x) for x in (ROOT / "results/timings.jsonl").read_text().splitlines()]
metadata = json.loads((ROOT / "results/metadata.json").read_text())
rows = []
for scenario in SCENARIOS:
    for config in CONFIGS:
        selected = [s for s in summary if s["scenario"] == scenario and s["config"] == config]
        n = sum(s["attack_events"] for s in selected)
        rejected = sum(s["attack_rejected"] for s in selected)
        accepted = sum(s["attack_accepted"] for s in selected)
        unprocessed = sum(s["attack_unprocessed"] for s in selected)
        assert rejected + accepted + unprocessed == n
        per_seed = [float(np.median([t["mean_ms_per_round"] for t in timings
                                   if t["scenario"] == scenario and t["config"] == config and t["seed"] == seed]))
                    for seed in metadata["seeds"]]
        rows.append({"scenario": scenario, "config": config, "attack_events": n,
                     "rejected": rejected, "accepted": accepted, "unprocessed": unprocessed,
                     "normal_events": sum(s["normal_events"] for s in selected),
                     "normal_rejected": sum(s["normal_rejected"] for s in selected),
                     "mean_of_seed_medians_ms": float(np.mean(per_seed)),
                     "min_seed_median_ms": min(per_seed), "max_seed_median_ms": max(per_seed)})
fig_dir = ROOT / "figures"
fig_dir.mkdir(exist_ok=True)
with (ROOT / "results/figure_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(11, 5.3))
x = np.arange(2)
width = 0.23
for j, (config, label, color) in enumerate(zip(CONFIGS, LABELS, COLORS)):
    picked = [next(r for r in rows if r["scenario"] == scenario and r["config"] == config) for scenario in SCENARIOS]
    positions = x + (j - 1) * width
    values = [r["rejected"] / r["attack_events"] * 100 for r in picked]
    axes[0].bar(positions, values, width, label=label, color=color, edgecolor="white")
    for pos, val, row in zip(positions, values, picked):
        axes[0].text(pos, val + 3, f'{row["rejected"]}/{row["attack_events"]}', ha="center", fontsize=9)
    means = np.array([r["mean_of_seed_medians_ms"] for r in picked])
    lows = means - np.array([r["min_seed_median_ms"] for r in picked])
    highs = np.array([r["max_seed_median_ms"] for r in picked]) - means
    axes[1].bar(positions, means, width, color=color, edgecolor="white", yerr=np.array([lows, highs]), capsize=3)
    for pos, val, high in zip(positions, means, highs):
        axes[1].annotate(f"{val:.2f}", (pos, val + high), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
for ax in axes:
    ax.set_xticks(x, ["H1: signed-field tampering", "H3: stale-round replay"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.22)
axes[0].set_ylim(0, 116)
axes[0].set_ylabel("Unique injected events rejected (%)")
axes[0].set_title("(a) Matched-check security outcomes")
axes[1].set_ylabel("Isolated protocol processing (ms / round)")
axes[1].set_title("(b) Small-payload protocol microbenchmark")
axes[1].set_ylim(0, axes[1].get_ylim()[1] * 1.18)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False)
fig.suptitle("H1/H3 fair protocol control — no model training or network simulation", y=0.99, fontsize=12)
fig.text(0.5, 0.10, "Per scenario/configuration: 27 unique attack events and 1,350 normal transactions; no unprocessed events or normal rejections.", ha="center", fontsize=8.7)
fig.text(0.5, 0.055, "Time bars: mean of three seed-level medians; whiskers: their min–max. Twenty technical repeats per seed; logging excluded.", ha="center", fontsize=8.7)
fig.subplots_adjust(left=0.07, right=0.985, top=0.81, bottom=0.23, wspace=0.30)
for extension in ("pdf", "png", "svg"):
    fig.savefig(fig_dir / f"protocol_fair_control.{extension}", dpi=300)
plt.close(fig)
print(json.dumps(rows, indent=2))
print("Figures:", fig_dir)
