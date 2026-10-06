#!/usr/bin/env python3
"""Reproduce Figure 2 from audited local checkpoint scores and validation archives.

No inference, encoder execution, interpolation, or network access is performed.
The original four metrics are retained independently of the direction of change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
CONDITIONS = {
    "base_500k": ("Base · 500k/domain", "#69A3C4", "--", "o"),
    "large_500k": ("Large · 500k/domain", "#E8A16A", "--", "^"),
    "base_full": ("Base · full pools", "#17618C", "-", "o"),
    "large_full": ("Large · full pools", "#BD541E", "-", "^"),
}
METRICS = {
    "omol_afd": ("OMol25 generation", "AFD ↓", "omol25_generation"),
    "omat_match": (
        "OMat24 structure",
        "Top-1 match (%) ↑",
        "omat24_structure_prediction",
    ),
    "omol_force_mae": ("OMol25 forces", "MAE (eV/Å) ↓", "omol25_mlip"),
    "omat_force_mae": ("OMat24 forces", "MAE (eV/Å) ↓", "omat24_mlip"),
}
COMPLETE = {
    "measured_wandb",
    "measured_local_complete",
    "recomputed_saved_complete",
    "recomputed_checkpoint_complete",
}
ENCODER = "Ty-Perez/ct-scd-omol25"
MAX_DATASET_EPOCHS = 80


def dataset_epochs(frame: pd.DataFrame) -> pd.Series:
    """Convert zero-based checkpoint indices to nominal dataset-sized epochs.

    One configured epoch uses approximately 5M draws. A completed configured
    epoch therefore represents five epochs on a 1M-example support, or one
    on the full approximately 5M-example support. Use the nominal 5:1 ratio
    for display; the exact configured draw count is retained separately.
    """
    scale = np.where(frame.condition.str.endswith("500k"), 5, 1)
    return (frame.epoch + 1) * scale


def plot_epochs(frame: pd.DataFrame) -> pd.Series:
    """Align generation/structure endpoints visually, retaining their true epochs.

    The shared terminal tick is labelled 80 in the current figure: the
    full-data endpoint is at 76 dataset epochs and the small-data endpoint
    is at 80. Force panels retain their actual dataset-epoch coordinates.
    """
    positions = frame.dataset_epochs.copy()
    for metric in ("omol_afd", "omat_match"):
        displayed = frame.displayed & frame.metric.eq(metric)
        full = frame[displayed & frame.condition.str.endswith("full")]
        if full.empty:
            continue
        terminal = full.dataset_epochs.max()
        small_endpoint = (
            displayed
            & frame.condition.str.endswith("500k")
            & frame.dataset_epochs.eq(MAX_DATASET_EPOCHS)
        )
        positions.loc[small_endpoint] = terminal
    return positions


def add_backfill(frame: pd.DataFrame, root: Path, provenance: dict) -> pd.DataFrame:
    """Load complete early checkpoint scores, retaining step-checkpoint identity."""
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text())
    provenance["inputs"][str(plan_path)] = sha256(plan_path)
    rows = []
    for job in plan["jobs"]:
        path = root / job["condition"] / job["label"] / "scores.json"
        result = json.loads(path.read_text())
        if any(result[key] != value for key, value in job.items()):
            raise ValueError(f"Backfill score identity differs from plan: {path}")
        if (
            result["afd"]["selected_reference_features_sha256"]
            != provenance["selected_reference_features_sha256"]
        ):
            raise ValueError("Backfill AFD uses different reference features")
        digest = sha256(path)
        provenance["inputs"][str(path)] = digest
        for metric, count in (("omol_afd", 1000), ("omat_match", 500)):
            if result["counts"][metric] != count:
                raise ValueError("Incomplete backfill pool")
            value = result["values"][metric]
            if not np.isfinite(value) or value < 0:
                raise ValueError("Invalid backfill value")
            rows.append(
                dict(
                    condition=job["condition"],
                    epoch=job["epoch"],
                    metric=metric,
                    value=value,
                    n_targets=count,
                    dataset_epochs=job["dataset_epochs"],
                    completed_epochs=job["epoch"] + 1
                    if job["checkpoint_kind"] == "epoch"
                    else np.nan,
                    configured_training_draws=(job["epoch"] + 1) * 4_996_604
                    if job["checkpoint_kind"] == "epoch"
                    else np.nan,
                    checkpoint_global_step=job["global_step"],
                    epoch_coordinate_kind=job["epoch_coordinate_kind"],
                    evidence=f"independent {job['checkpoint_kind']}-checkpoint evaluation",
                    source_file=str(path),
                    source_sha256=digest,
                )
            )
    combined = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
    if combined.duplicated(["condition", "epoch", "metric"]).any():
        raise ValueError("Backfill duplicates an existing checkpoint observation")
    combined["displayed"] = combined.dataset_epochs <= MAX_DATASET_EPOCHS
    combined["plot_epoch"] = plot_epochs(combined)
    return combined.sort_values(["metric", "condition", "dataset_epochs"])


def sha256(path: Path) -> str:
    """Hash files incrementally, including compressed validation archives."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_history(history: pd.DataFrame) -> pd.DataFrame:
    """Keep complete observations, including early small-support validation.

    Epoch 15 is supplied by matched subsets. Full-data history retains its
    original starting point; early small-support observations fill the revised
    80-dataset-epoch display range.
    """
    expected = np.where(history.metric == "omol_afd", 1000, 500)
    selected = history[
        history.condition.isin(CONDITIONS)
        & history.metric.isin(METRICS)
        & (
            (history.epoch > 15)
            | (history.condition.str.endswith("500k") & (history.epoch < 15))
        )
        & history.status.isin(COMPLETE)
        & (history.n_samples == expected)
    ].copy()
    afd = selected[selected.metric == "omol_afd"]
    if not (afd.afd_model_name == ENCODER).all():
        raise ValueError("Incompatible AFD encoder in history")
    if selected.duplicated(["condition", "epoch", "metric"]).any():
        raise ValueError("Duplicate historical observations")
    return selected


def check_afd(scoring: dict, reference_digest: str) -> None:
    """Reject encoder or reference changes that would invalidate an AFD curve."""
    if (
        scoring["model_name"] != ENCODER
        or scoring["reference_sampling_seed"] != 0
        or scoring["selected_reference_features_sha256"] != reference_digest
    ):
        raise ValueError("Incompatible AFD encoder/reference protocol")


def load_measurements(root: Path) -> tuple[pd.DataFrame, dict]:
    """Verify archived evidence and assemble observations with explicit provenance."""
    inputs = {}

    def record(path, expected=None):
        path = Path(path)
        if str(path) not in inputs:
            inputs[str(path)] = sha256(path)
        digest = inputs[str(path)]
        if expected is not None and digest != expected:
            raise ValueError(f"Archived source hash mismatch: {path}")
        return digest

    def read(path):
        record(path)
        return json.loads(Path(path).read_text())

    early = root / "zatom2_figure2_epoch15"
    late = root / "zatom2_figure2_epoch39_42"
    archive = root / "zatom2_figure2_20260906"
    early_manifest = read(early / "provenance.json")
    early_protocol = early_manifest["protocol"]
    if early_protocol["afd_model"] != ENCODER:
        raise ValueError("Incompatible epoch-15 AFD encoder")
    for source in early_manifest["results"]:
        record(early / Path(source["path"]).name, source["sha256"])
    reference_digest = read(early / "base_full_omol25_generation.json")[
        "selected_reference_features_sha256"
    ]
    rows = []
    for condition in CONDITIONS:
        for metric, (_, _, task) in METRICS.items():
            path = early / f"{condition}_{task}.json"
            result = read(path)
            if (result["condition"], result["metric"], result["epoch"]) != (
                condition,
                metric,
                15,
            ):
                raise ValueError(f"Unexpected score identity: {path}")
            if metric == "omol_afd":
                check_afd(
                    {
                        "model_name": early_protocol["afd_model"],
                        "reference_sampling_seed": result["reference_seed"],
                        "selected_reference_features_sha256": result[
                            "selected_reference_features_sha256"
                        ],
                    },
                    reference_digest,
                )
                if result["n_targets"] != 1000 or result["n_reference"] != 1000:
                    raise ValueError("Incomplete epoch-15 AFD pool")
            rows.append(
                dict(
                    condition=condition,
                    epoch=15,
                    metric=metric,
                    value=result["value"],
                    n_targets=result["n_targets"],
                    evidence="saved outputs; common epoch-15 targets",
                    source_file=str(path),
                    source_sha256=record(path),
                )
            )

    audit = read(archive / "recomputed/final_analysis_verification.json")
    if not audit["passed"]:
        raise ValueError("Historical analysis did not pass its audit")
    path = archive / "analysis/validation_measurements.csv"
    record(path, audit["analysis_sha256"][path.name])
    history = select_history(pd.read_csv(path))
    for row in history.to_dict("records"):
        sources = [Path(p).resolve() for p in row["source_file"].split("|")]
        # The audit fingerprints rescoring sources and the consolidated table;
        # original scalar histories are retained but have no per-file audit hash.
        digests = [
            record(p, audit["measurement_source_sha256"].get(str(p))) for p in sources
        ]
        if row["metric"] == "omol_afd":
            if len(sources) != 1:
                raise ValueError("AFD must use a single pooled score")
            result = read(sources[0])
            check_afd(result["scoring"], reference_digest)
            metrics = result.get("metrics", result.get("afd_metrics", {}))
            if metrics["afd_ref_sample_size"] != 1000:
                raise ValueError("Incomplete historical AFD reference pool")
        rows.append(
            dict(
                condition=row["condition"],
                epoch=int(row["epoch"]),
                metric=row["metric"],
                value=row["value"] * (100 if row["metric"] == "omat_match" else 1),
                n_targets=int(row["n_samples"]),
                evidence=row["status"],
                source_file="|".join(map(str, sources)),
                source_sha256="|".join(digests),
            )
        )

    late_manifest = read(late / "provenance.json")
    path = late / "figure2_scores.csv"
    artifact = next(
        x for x in late_manifest["artifacts"] if Path(x["path"]).name == path.name
    )
    record(path, artifact["sha256"])
    for row in pd.read_csv(path).to_dict("records"):
        metric, condition = row["metric"], row["condition"]
        if metric not in METRICS:
            continue
        if metric == "omol_afd":
            source = late / f"{condition}_omol25_afd.json"
            result = read(source)
            check_afd(
                {
                    "model_name": result["protocol"]["molecular_afd_encoder"],
                    "reference_sampling_seed": result["reference_seed"],
                    "selected_reference_features_sha256": result[
                        "selected_reference_features_sha256"
                    ],
                },
                reference_digest,
            )
            if result["n_reference"] != 1000 or row["n_targets"] != 1000:
                raise ValueError("Incomplete late AFD pool")
            value = result["value"]
        else:
            source = (
                root
                / "figure2_epoch39_42_16shards"
                / condition
                / "pooled"
                / METRICS[metric][2]
                / "pooled_metrics.json"
            )
            result = read(source)
            if result["n_samples"] != 500 or result["epoch"] != row["epoch"]:
                raise ValueError(f"Incomplete or wrong-epoch checkpoint pool: {source}")
            value = result["value"] * (100 if metric == "omat_match" else 1)
        if not np.isclose(value, row["value"], rtol=1e-12, atol=0):
            raise ValueError(f"Checkpoint score disagrees with source: {source}")
        # Prefer the independently verified checkpoint pool at the same epoch.
        rows = [
            r
            for r in rows
            if (r["condition"], r["epoch"], r["metric"])
            != (condition, row["epoch"], metric)
        ]
        rows.append(
            dict(
                condition=condition,
                epoch=int(row["epoch"]),
                metric=metric,
                value=value,
                n_targets=int(row["n_targets"]),
                evidence="independent checkpoint evaluation",
                source_file=str(source),
                source_sha256=record(source),
            )
        )
    frame = pd.DataFrame(rows).sort_values(["metric", "condition", "epoch"])
    if not np.isfinite(frame.value).all() or (frame.value < 0).any():
        raise ValueError("Nonfinite or negative scores")
    frame["completed_epochs"] = frame.epoch + 1
    frame["configured_training_draws"] = frame.completed_epochs * 4_996_604
    frame["dataset_epochs"] = dataset_epochs(frame)
    frame["displayed"] = frame.dataset_epochs <= MAX_DATASET_EPOCHS
    frame["plot_epoch"] = plot_epochs(frame)
    return frame, {
        "inputs": inputs,
        "afd_encoder": ENCODER,
        "selected_reference_features_sha256": reference_digest,
    }


def plot(frame: pd.DataFrame, output: Path) -> None:
    """Draw four compact panels with raw values and no fitted or smoothed curves."""
    frame = frame[frame.displayed]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#BBBBBB",
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 3.45), sharex="row")
    fig.subplots_adjust(
        left=0.105, right=0.985, bottom=0.205, top=0.93, wspace=0.32, hspace=0.47
    )
    for index, (ax, (metric, (title, ylabel, _))) in enumerate(
        zip(axes.flat, METRICS.items())
    ):
        for condition, (_, color, style, marker) in CONDITIONS.items():
            subset = frame[(frame.metric == metric) & (frame.condition == condition)]
            ax.plot(
                subset.plot_epoch,
                subset.value,
                color=color,
                linestyle=style,
                linewidth=1.45,
                zorder=3,
            )
            # Mark every generation/structure observation, and selected force
            # checkpoints. All intermediate force observations remain on the line.
            dots = (
                subset
                if "force" not in metric
                else subset[
                    subset.epoch.isin([1, 3, 7, 11, 15, 19, 39, 40, 42, 60, 79])
                ]
            )
            ax.plot(
                dots.plot_epoch,
                dots.value,
                linestyle="none",
                marker=marker,
                color=color,
                markersize=3.4,
                markeredgewidth=0.65,
                markerfacecolor="white" if "500k" in condition else color,
                zorder=4,
            )
        ax.set_title(
            f"({chr(97 + index)})  {title}", loc="left", fontweight="bold", pad=7
        )
        ax.set_ylabel(ylabel)
        ax.set_xlim(8, MAX_DATASET_EPOCHS + 2)
        ax.set_xticks([20, 40, 60, 80])
        if index < 2:
            terminal = frame[
                frame.metric.eq(metric) & frame.condition.str.endswith("full")
            ].dataset_epochs.max()
            ax.set_xticks([20, 40, 60, terminal], ["20", "40", "60", "80"])
        ax.grid(axis="y", color="#E7E7E7", linewidth=0.55)
        ax.tick_params(length=2.5, width=0.6)
        ax.margins(y=0.14)
        if metric == "omol_afd":
            ax.set_yticks([0.015, 0.020, 0.025, 0.030])
            upper = max(0.030, 1.07 * frame[frame.metric.eq(metric)].value.max())
            ax.set_ylim(0.014, upper)
            if upper > 0.030:
                from matplotlib.ticker import MaxNLocator

                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        elif metric == "omat_match":
            ax.set_ylim(0, 46)
            ax.set_yticks([0, 20, 40])
        if index >= 2:
            ax.set_xlabel("Dataset epochs")
    handles = [
        Line2D(
            [],
            [],
            label=label,
            color=color,
            linestyle=style,
            marker=marker,
            markersize=4,
            linewidth=1.5,
            markerfacecolor="white" if "500k" in condition else color,
        )
        for condition, (label, color, style, marker) in CONDITIONS.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.005),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handlelength=2.2,
        handletextpad=0.5,
        fontsize=7.5,
    )
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"figure2_scaling.{suffix}", dpi=250)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=REPO / "logs/analysis")
    parser.add_argument(
        "--backfill-root",
        type=Path,
        help="Optional complete step-3000/epoch-7 backfill directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO / "logs/analysis/zatom2_figure2_epochs"
    )
    args = parser.parse_args()
    frame, provenance = load_measurements(args.analysis_root)
    if args.backfill_root is not None:
        frame = add_backfill(frame, args.backfill_root.resolve(), provenance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "figure2_measurements.csv", index=False)
    plot(frame, args.output_dir)
    provenance["script_sha256"] = sha256(Path(__file__))
    provenance["outputs"] = {
        p.name: sha256(p)
        for p in args.output_dir.iterdir()
        if p.name.startswith("figure2_")
    }
    provenance["notes"] = (
        "Dataset epochs = (raw checkpoint epoch + 1) times 5 for 500k/domain, "
        "or times 1 for full pools, using the nominal 5M:1M support ratio. "
        "Display capped at 80 dataset epochs; later observations retained in the "
        "CSV with displayed=False. Raw epoch 15 maps to 80 for small data and "
        "16 for full data. Raw epoch 39 for small data maps to 200 and is omitted. "
        "Earlier complete small-data validation supplies the force curves. "
        "The historical small-data AFD and structure endpoint is at 80 dataset "
        "epochs; optional early backfill observations are described below. "
        "Raw-epoch-15 structure/force denominators are "
        "499/493; other displayed observations use 500. AFD uses 1000+1000. "
        "No interpolation, extrapolation, smoothing, or cross-seed uncertainty. "
        "Equal dataset epochs give full-data runs five times the configured draws "
        "of small-data runs; this is not a matched-draw or matched-FLOP comparison."
        " In panels (a) and (b), the small-data endpoint at 80 dataset epochs is "
        "visually aligned with the last full-data endpoint at 76, with a shared "
        "tick labelled 80. plot_epoch records this display coordinate; dataset_epochs "
        "retains the true epoch. Force panels retain their original positions."
    )
    if args.backfill_root is not None:
        provenance["backfill_notes"] = (
            "Early step-3000 and epoch-7 generation/structure scores are included. "
            "Step 3000 is raw epoch 2, not epoch 0. Its approximate dataset-epoch "
            "coordinate uses saved rank-local progress within epoch 2; it is not "
            "a measured global draw count. Epoch 7 maps to 40 dataset epochs."
        )
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(f"Wrote {len(frame)} verified observations and Figure 2 to {args.output_dir}")


if __name__ == "__main__":
    main()
