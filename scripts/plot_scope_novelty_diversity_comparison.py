#!/usr/bin/env python3
"""Plot comparable SCOPe novelty/diversity checkpoint sweeps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

METRICS = [
    (
        "novelty_mean_best_tm",
        "FoldSeek Nearest-Train TM",
        "mean nearest-train TM (lower is more novel)",
        "lower",
    ),
    (
        "novel_rate_tm_below_0.5",
        "FoldSeek Novel Rate at TM < 0.5",
        "fraction below nearest-train TM 0.5",
        "higher",
    ),
    (
        "tmalign_diversity_auc",
        "TM-align Complete-Linkage Diversity AUC",
        "cluster-fraction AUC (higher is more diverse)",
        "higher",
    ),
    (
        "tmalign_cluster_fraction_at_0.5",
        "TM-align Cluster Fraction at TM 0.5",
        "clusters / backbones at threshold 0.5",
        "higher",
    ),
    (
        "tmalign_n_clusters_at_0.5",
        "TM-align Clusters at TM 0.5",
        "number of clusters at threshold 0.5",
        "higher",
    ),
    (
        "tmalign_mean_pairwise_tm",
        "Mean Pairwise TM",
        "mean pairwise TM (lower is more diverse)",
        "lower",
    ),
    (
        "tmalign_median_pairwise_tm",
        "Median Pairwise TM",
        "median pairwise TM (lower is more diverse)",
        "lower",
    ),
]

MODEL_ORDER = [
    "Zatom-2 from scratch",
    "Zatom-2 joint-FC fine-tune",
    "Zatom-2 joint-properties-FC fine-tune",
    "Zatom-2 joint-no-FC fine-tune",
    "RFD3/EDM from scratch",
]
COLORS = {
    "Zatom-2 from scratch": "#2f6df6",
    "Zatom-2 joint-FC fine-tune": "#e52b2b",
    "Zatom-2 joint-properties-FC fine-tune": "#16a34a",
    "Zatom-2 joint-no-FC fine-tune": "#7c3aed",
    "RFD3/EDM from scratch": "#111827",
}


def read_rows(path: Path, label: str, raw_zero: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("novelty_complete") != "True"
                or row.get("diversity_complete") != "True"
            ):
                continue
            if (
                int(row.get("novelty_n_queries", 0)) != 96
                or int(row.get("diversity_n_queries", 0)) != 96
            ):
                continue
            raw_step = int(row["step"]) if row.get("step", "").strip() else 0
            row["model"] = label
            row["raw_step"] = raw_step
            row["local_step"] = raw_step - raw_zero
            rows.append(row)
    return sorted(rows, key=lambda row: int(row["raw_step"]))


def models_present(rows: list[dict[str, object]]) -> list[str]:
    present = {str(row["model"]) for row in rows}
    return [label for label in MODEL_ORDER if label in present]


def value(row: dict[str, object], key: str) -> float:
    return float(row[key])


def value_text(number: float) -> str:
    if number.is_integer():
        return str(int(number))
    return f"{number:.3f}"


def checkpoint_text(row: dict[str, object]) -> str:
    step = int(row["local_step"])
    return f"{step / 1000:g}k"


def write_combined(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "step",
        "step_name",
        "run_tag",
        "output_root",
        "novelty_complete",
        "diversity_complete",
        *[metric[0] for metric in METRICS],
        "raw_step",
        "local_step",
        "model",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_metric(
    rows: list[dict[str, object]], key: str, title: str, ylabel: str, out: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in models_present(rows):
        points = [row for row in rows if row["model"] == label]
        ax.plot(
            [int(row["local_step"]) / 1000 for row in points],
            [value(row, key) for row in points],
            marker="o",
            linewidth=2,
            markersize=4,
            label=label,
            color=COLORS[label],
        )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("SCOPe local optimizer step (pretrained runs: raw step - 100k)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_grid(rows: list[dict[str, object]], out: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(16, 22))
    axes_flat = list(axes.flat)
    for ax, (key, title, ylabel, _direction) in zip(axes_flat, METRICS):
        for label in models_present(rows):
            points = [row for row in rows if row["model"] == label]
            ax.plot(
                [int(row["local_step"]) / 1000 for row in points],
                [value(row, key) for row in points],
                marker="o",
                linewidth=2,
                markersize=3.5,
                label=label,
                color=COLORS[label],
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("SCOPe local optimizer step (k)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[-1].legend(handles, labels, loc="center", frameon=False, fontsize=11)
    rfd3_points = [row for row in rows if row["model"] == "RFD3/EDM from scratch"]
    has_rfd3 = bool(rfd3_points)
    title = (
        "SCOPe-2k ESMFold2 novelty/diversity: Zatom-2 and RFD3/EDM checkpoint sweeps"
        if has_rfd3
        else "SCOPe-2k ESMFold2 novelty/diversity: four Zatom-2 fine-tuning conditions"
    )
    if len(rfd3_points) == 1:
        title = "SCOPe-2k ESMFold2 novelty/diversity: Zatom-2 sweeps and RFD3/EDM random-init control"
    fig.suptitle(title, fontsize=20, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def best_records(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    labels = models_present(rows)
    for key, title, _ylabel, direction in METRICS:
        best_by_model = {}
        for label in labels:
            candidates = [row for row in rows if row["model"] == label]
            best_by_model[label] = (min if direction == "lower" else max)(
                candidates, key=lambda row: value(row, key)
            )
        best_value = (min if direction == "lower" else max)(
            value(row, key) for row in best_by_model.values()
        )
        winners = [
            label
            for label, row in best_by_model.items()
            if abs(value(row, key) - best_value) < 1e-12
        ]
        winner = winners[0] if len(winners) == 1 else "Tie: " + " / ".join(winners)
        record = {
            "Metric": title,
            "Direction": direction,
            "Winner": winner,
        }
        for label in labels:
            row = best_by_model[label]
            record[f"Best {label} ckpt"] = checkpoint_text(row)
            record[f"Best {label} value"] = value_text(value(row, key))
        records.append(record)
    return records


def write_best(rows: list[dict[str, object]], csv_path: Path, png_path: Path) -> None:
    records = best_records(rows)
    with csv_path.open("w", newline="") as handle:
        fields = list(records[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    labels = models_present(rows)
    table_rows = []
    for record in records:
        table_rows.append(
            [
                record["Metric"],
                record["Direction"],
                record["Winner"],
                *[
                    f"{record[f'Best {label} ckpt']}: {record[f'Best {label} value']}"
                    for label in labels
                ],
            ]
        )
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=["Metric", "Direction", "Winner", *labels],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.7)
    ax.set_title(
        "Best checkpoint from the four-run SCOPe novelty/diversity sweep",
        fontweight="bold",
        pad=18,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--from-scratch", type=Path, required=True)
    parser.add_argument("--joint-fc", type=Path, required=True)
    parser.add_argument("--joint-properties-fc", type=Path, required=True)
    parser.add_argument("--joint-no-fc", type=Path, required=True)
    parser.add_argument("--rfd3", type=Path)
    parser.add_argument(
        "--ckpt0-summary",
        type=Path,
        help="Optional random-init Zatom-2 novelty/diversity summary to prepend to scratch.",
    )
    parser.add_argument(
        "--rfd3-ckpt0-summary",
        type=Path,
        help="Optional random-init RFD3 novelty/diversity summary to prepend to RFD3.",
    )
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    scratch_rows = read_rows(args.from_scratch, "Zatom-2 from scratch", 0)
    if args.ckpt0_summary is not None:
        scratch_rows.extend(read_rows(args.ckpt0_summary, "Zatom-2 from scratch", 0))
        scratch_rows.sort(key=lambda row: int(row["raw_step"]))
    rows.extend(scratch_rows)
    rows.extend(read_rows(args.joint_fc, "Zatom-2 joint-FC fine-tune", 100000))
    rows.extend(
        read_rows(
            args.joint_properties_fc, "Zatom-2 joint-properties-FC fine-tune", 100000
        )
    )
    rows.extend(read_rows(args.joint_no_fc, "Zatom-2 joint-no-FC fine-tune", 100000))
    rfd3_rows = []
    if args.rfd3 is not None:
        rfd3_rows.extend(read_rows(args.rfd3, "RFD3/EDM from scratch", 0))
    if args.rfd3_ckpt0_summary is not None:
        rfd3_rows.extend(read_rows(args.rfd3_ckpt0_summary, "RFD3/EDM from scratch", 0))
    if rfd3_rows:
        rows.extend(sorted(rfd3_rows, key=lambda row: int(row["raw_step"])))
    if not rows:
        raise SystemExit("No complete novelty/diversity rows found")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    write_combined(
        rows,
        args.out_dir.parent.parent / f"novelty_diversity_combined_summary{suffix}.csv",
    )
    (
        args.out_dir.parent.parent
        / f"novelty_diversity_comparison_axis_metadata{suffix}.json"
    ).write_text(
        json.dumps(
            {
                "x_axis": "local_scope_optimizer_step",
                "scratch_transform": "local_step = raw checkpoint step",
                "pretrained_transform": "local_step = raw checkpoint step - 100000",
                "reference": "SCOPe-2k training FoldSeek database for novelty",
                "query_set": "all generated backbones",
                "tm_align_symmetrization": "min directional TM score",
            },
            indent=2,
        )
        + "\n"
    )
    plot_grid(rows, args.out_dir / f"all_novelty_diversity_metrics_grid{suffix}.png")
    for key, title, ylabel, _direction in METRICS:
        plot_metric(rows, key, title, ylabel, args.out_dir / f"{key}{suffix}.png")
    write_best(
        rows,
        args.out_dir / f"best_checkpoint_by_novelty_diversity_metric{suffix}.csv",
        args.out_dir / f"best_checkpoint_by_novelty_diversity_metric{suffix}.png",
    )


if __name__ == "__main__":
    main()
