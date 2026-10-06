#!/usr/bin/env python3
"""Plot exact Genie-short novelty across the Zatom2 checkpoint sweeps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


GENIE_SHORT_BASELINE = 98 / 455
LABELS = {
    "joint_force": "Joint + force conditioning",
    "joint_properties_force": "Joint + properties + force",
    "joint_no_force": "Joint, no force conditioning",
    "no_pretraining": "No pretraining",
}
COLORS = {
    "joint_force": "#e52b2b",
    "joint_properties_force": "#16a34a",
    "joint_no_force": "#7c3aed",
    "no_pretraining": "#2f6df6",
}


def run_key(row: dict[str, str]) -> str:
    tag = row["sweep"]
    if "no-pretraining" in tag:
        return "no_pretraining"
    if "properties-force-conditioned" in tag:
        return "joint_properties_force"
    if "without-force-conditioning" in tag:
        return "joint_no_force"
    return "joint_force"


def local_step(row: dict[str, str], key: str) -> int:
    step = int(row["step"])
    return step if key == "no_pretraining" else step - 100000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-summary", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    with args.exact_summary.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            n = int(raw["n_genie_short_confident"])
            rate = float(raw["genie_short_novel_rate_conditional"])
            if n == 0:
                continue
            key = run_key(raw)
            rows.append(
                {
                    "run": key,
                    "label": LABELS[key],
                    "step": int(raw["step"]),
                    "local_step": local_step(raw, key),
                    "n_confident": n,
                    "novel_count": int(raw["genie_short_novel_count"]),
                    "novel_rate": rate,
                }
            )
    if not rows:
        raise SystemExit("No non-empty exact novelty rows found")

    rows.sort(key=lambda row: (str(row["run"]), int(row["local_step"])))
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    data_path = args.out_prefix.with_name(args.out_prefix.name + "_data.csv")
    with data_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) + ["genie_short_baseline"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "genie_short_baseline": GENIE_SHORT_BASELINE})

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for key in LABELS:
        points = [row for row in rows if row["run"] == key]
        if not points:
            continue
        ax.plot(
            [int(row["local_step"]) / 1000 for row in points],
            [100 * float(row["novel_rate"]) for row in points],
            marker="o",
            markersize=4.5,
            linewidth=2,
            color=COLORS[key],
            label=LABELS[key],
        )

    ax.axhline(
        100 * GENIE_SHORT_BASELINE,
        color="#111827",
        linestyle="--",
        linewidth=1.8,
        label="Genie-short paper: 98/455 = 21.5%",
    )
    ax.set_xlabel("Checkpoint step (k; fine-tuning runs shifted by 100k)")
    ax.set_ylabel("Novel and confidently designable structures (%)")
    ax.set_title("Exact Genie-short novelty across Zatom2 SCOPe checkpoint sweeps", fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(args.out_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(args.out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_prefix.with_suffix('.png')}")
    print(f"wrote {args.out_prefix.with_suffix('.pdf')}")
    print(f"wrote {data_path}")


if __name__ == "__main__":
    main()
