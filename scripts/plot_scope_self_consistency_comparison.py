#!/usr/bin/env python3
"""Plot comparable SCOPe self-consistency checkpoint sweeps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

METRICS = [
    (
        "designability",
        "Designability Fraction",
        "designable backbones / generated backbones",
        "higher",
    ),
    ("designable_count", "Designable Count", "designable backbones", "higher"),
    ("mean_min_scRMSD", "Mean Best-of-8 scRMSD", "mean min scRMSD (A)", "lower"),
    ("median_min_scRMSD", "Median Best-of-8 scRMSD", "median min scRMSD (A)", "lower"),
    ("mean_max_scTM", "Mean Best-of-8 scTM", "mean max scTM", "higher"),
    ("median_max_scTM", "Median Best-of-8 scTM", "median max scTM", "higher"),
    (
        "all_refolds_mean_scTM",
        "Mean All-Refolds scTM",
        "all-refolds mean scTM",
        "higher",
    ),
    (
        "all_refolds_median_scTM",
        "Median All-Refolds scTM",
        "all-refolds median scTM",
        "higher",
    ),
]

FOCUSED_METRICS = [
    METRICS[0],
    METRICS[2],
    METRICS[4],
]


def read_rows(
    path: Path, label: str, color: str, raw_zero: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("complete") != "1":
                continue
            if (
                int(row.get("n_backbones", 0)) != 96
                or int(row.get("n_refolds", 0)) != 768
            ):
                continue
            raw_step = int(row["step"])
            row["model"] = label
            row["model_short"] = label
            row["color"] = color
            row["raw_step"] = raw_step
            row["local_step"] = raw_step - raw_zero
            rows.append(row)
    return sorted(rows, key=lambda row: int(row["raw_step"]))


def as_float(row: dict[str, object], key: str) -> float:
    return float(row[key])


def write_combined(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "step",
        "step_name",
        "run_tag",
        "output_root",
        "complete",
        "n_backbones",
        "n_refolds",
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
    for label, color in model_order(rows):
        points = [row for row in rows if row["model"] == label]
        ax.plot(
            [int(row["local_step"]) / 1000 for row in points],
            [as_float(row, key) for row in points],
            marker="o",
            linewidth=2,
            markersize=4,
            label=label,
            color=color,
        )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("SCOPe local optimizer step (pretrained runs: raw step - 100k)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def model_order(rows: list[dict[str, object]]) -> list[tuple[str, str]]:
    preferred = [
        "Zatom-2 from scratch",
        "Zatom-2 joint-FC fine-tune",
        "Zatom-2 joint-properties-FC fine-tune",
        "Zatom-2 joint-no-FC fine-tune",
        "RFD3/EDM from scratch",
    ]
    colors = {
        "Zatom-2 from scratch": "#C23B3B",
        "Zatom-2 joint-FC fine-tune": "#17618C",
        "Zatom-2 joint-properties-FC fine-tune": "#3F82A5",
        "Zatom-2 joint-no-FC fine-tune": "#6AA4C5",
        "RFD3/EDM from scratch": "#111827",
    }
    present = {str(row["model"]) for row in rows}
    return [(label, colors[label]) for label in preferred if label in present]


FOCUSED_STYLES = {
    "Zatom-2 from scratch": {
        "color": "#C23B3B",
        "linestyle": (0, (7, 4)),
        "marker": "o",
    },
    "Zatom-2 joint-FC fine-tune": {"color": "#17618C", "linestyle": "-", "marker": "o"},
    "Zatom-2 joint-properties-FC fine-tune": {
        "color": "#3F82A5",
        "linestyle": "-",
        "marker": "^",
    },
    "Zatom-2 joint-no-FC fine-tune": {
        "color": "#6AA4C5",
        "linestyle": "-",
        "marker": "^",
    },
    "RFD3/EDM from scratch": {
        "color": "#111827",
        "linestyle": (0, (3, 2)),
        "marker": "D",
    },
}


def plot_grid(rows: list[dict[str, object]], out: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(16, 22), sharex=False)
    for ax, (key, title, ylabel, _direction) in zip(axes.flat, METRICS):
        for label, color in model_order(rows):
            points = [row for row in rows if row["model"] == label]
            ax.plot(
                [int(row["local_step"]) / 1000 for row in points],
                [as_float(row, key) for row in points],
                marker="o",
                linewidth=2,
                markersize=3.5,
                label=label,
                color=color,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("SCOPe local optimizer step (k)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False, loc="best")
    has_rfd3 = any(row["model"] == "RFD3/EDM from scratch" for row in rows)
    title = (
        "SCOPe-2k ESMFold2 self-consistency: Zatom-2 and RFD3/EDM checkpoint sweeps"
        if has_rfd3
        else "SCOPe-2k ESMFold2 self-consistency: four Zatom-2 fine-tuning conditions"
    )
    fig.suptitle(title, fontsize=20, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_focused_grid(
    rows: list[dict[str, object]],
    novelty_rows: list[dict[str, object]],
    out: Path,
) -> None:
    """Plot the main-paper convergence metrics with novelty in one row."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    panels = [
        FOCUSED_METRICS[0],
        FOCUSED_METRICS[2],
        (
            "novelty_mean_best_tm",
            "Mean Nearest-Train TM",
            "mean nearest-train TM (lower is more novel)",
            "lower",
        ),
    ]
    for ax, (key, metric_title, _ylabel, _direction) in zip(axes, panels):
        source_rows = novelty_rows if key == "novelty_mean_best_tm" else rows
        for label, color in model_order(source_rows):
            points = [row for row in source_rows if row["model"] == label]
            style = FOCUSED_STYLES[label]
            ax.plot(
                [int(row["local_step"]) / 1000 for row in points],
                [as_float(row, key) for row in points],
                marker=style["marker"],
                linewidth=2,
                markersize=4,
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                dash_capstyle="butt",
            )
        ax.set_ylabel(metric_title)
        ax.grid(True, alpha=0.25)
    for ax in axes:
        ax.set_xlabel(r"Optimizer steps ($10^3$)")
    axes[0].legend(frameon=False, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def best_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for key, title, _ylabel, direction in METRICS:
        for label, _color in model_order(rows):
            candidates = [row for row in rows if row["model"] == label]
            best = (min if direction == "lower" else max)(
                candidates, key=lambda row: as_float(row, key)
            )
            output.append(
                {
                    "Metric": title,
                    "Direction": direction,
                    "Model": label,
                    "Checkpoint": f"{int(best['local_step']) / 1000:g}k",
                    "Raw checkpoint step": int(best["raw_step"]),
                    "Value": as_float(best, key),
                }
            )
    return output


def write_best(rows: list[dict[str, object]], csv_path: Path, png_path: Path) -> None:
    best = best_rows(rows)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(best[0]))
        writer.writeheader()
        writer.writerows(best)

    labels = [label for label, _color in model_order(rows)]
    table_rows: list[list[str]] = []
    for _key, title, _ylabel, direction in METRICS:
        line = [title, direction]
        for label in labels:
            item = next(
                item
                for item in best
                if item["Metric"] == title and item["Model"] == label
            )
            line.append(f"{item['Checkpoint']}: {float(item['Value']):.3f}")
        table_rows.append(line)

    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=["Metric", "Direction", *labels],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.7)
    ax.set_title(
        "Best checkpoint from the four-run SCOPe self-consistency sweep",
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
    parser.add_argument(
        "--ckpt0-summary",
        type=Path,
        help="Optional completed step-0 summary to prepend to the from-scratch trajectory.",
    )
    parser.add_argument("--joint-fc", type=Path, required=True)
    parser.add_argument("--joint-properties-fc", type=Path, required=True)
    parser.add_argument("--joint-no-fc", type=Path, required=True)
    parser.add_argument("--rfd3", type=Path)
    parser.add_argument(
        "--rfd3-ckpt0-summary",
        type=Path,
        help="Optional completed random-init RFD3 control summary to prepend to the RFD3 trajectory.",
    )
    parser.add_argument("--novelty-summary", type=Path)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    rows = []
    scratch_rows = read_rows(args.from_scratch, "Zatom-2 from scratch", "#2f6df6", 0)
    if args.ckpt0_summary is not None:
        scratch_rows.extend(
            read_rows(args.ckpt0_summary, "Zatom-2 from scratch", "#2f6df6", 0)
        )
        scratch_rows.sort(key=lambda row: int(row["raw_step"]))
    rows.extend(scratch_rows)
    rows.extend(
        read_rows(args.joint_fc, "Zatom-2 joint-FC fine-tune", "#e52b2b", 100000)
    )
    rows.extend(
        read_rows(
            args.joint_properties_fc,
            "Zatom-2 joint-properties-FC fine-tune",
            "#16a34a",
            100000,
        )
    )
    rows.extend(
        read_rows(args.joint_no_fc, "Zatom-2 joint-no-FC fine-tune", "#7c3aed", 100000)
    )
    if args.rfd3 is not None:
        rfd3_rows = read_rows(args.rfd3, "RFD3/EDM from scratch", "#111827", 0)
        if args.rfd3_ckpt0_summary is not None:
            rfd3_rows.extend(
                read_rows(
                    args.rfd3_ckpt0_summary, "RFD3/EDM from scratch", "#111827", 0
                )
            )
            rfd3_rows.sort(key=lambda row: int(row["raw_step"]))
        rows.extend(rfd3_rows)
    if not rows:
        raise SystemExit("No complete 96-backbone rows found")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    write_combined(rows, args.out_dir.parent / f"combined_summary{suffix}.csv")
    (args.out_dir.parent / f"comparison_axis_metadata{suffix}.json").write_text(
        "{\n"
        '  "x_axis": "local_scope_optimizer_step",\n'
        '  "scratch_transform": "local_step = raw checkpoint step",\n'
        '  "pretrained_transform": "local_step = raw checkpoint step - 100000",\n'
        '  "refolds_filter": "complete rows with n_backbones=96 and n_refolds=768",\n'
        '  "structure_predictor": "ESMFold2",\n'
        '  "n_generations": 96,\n'
        '  "mpnn_sequences_per_backbone": 8\n'
        "}\n"
    )
    plot_grid(rows, args.out_dir / f"all_metrics_grid{suffix}.png")
    if args.novelty_summary is not None:
        novelty_rows = []
        with args.novelty_summary.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if (
                    row.get("novelty_complete") != "True"
                    or row.get("diversity_complete") != "True"
                    or (
                        "novelty_n_queries" in row
                        and int(row.get("novelty_n_queries", 0)) != 96
                    )
                    or (
                        "diversity_n_queries" in row
                        and int(row.get("diversity_n_queries", 0)) != 96
                    )
                ):
                    continue
                row["raw_step"] = int(row["raw_step"])
                row["local_step"] = int(row["local_step"])
                novelty_rows.append(row)
        plot_focused_grid(
            rows,
            sorted(
                novelty_rows, key=lambda row: (str(row["model"]), int(row["raw_step"]))
            ),
            args.out_dir / f"scope_transfer_trajectories{suffix}.png",
        )
    for key, title, ylabel, _direction in METRICS:
        plot_metric(rows, key, title, ylabel, args.out_dir / f"{key}{suffix}.png")
    write_best(
        rows,
        args.out_dir / f"best_checkpoint_by_metric{suffix}.csv",
        args.out_dir / f"best_checkpoint_by_metric{suffix}.png",
    )


if __name__ == "__main__":
    main()
