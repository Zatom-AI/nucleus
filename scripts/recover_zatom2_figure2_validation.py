"""Recover Figure 2 scalars from existing validation CSVs without running inference.

The output distinguishes complete from partial validation. AFD requires a logged
epoch summary; it cannot be reconstructed from per-example scalar metrics. Force
MAE is the component-weighted ratio of error sums to component counts. Structure
metrics prefer epoch summaries and otherwise use unique per-target top-1 scores.
For these experiments, the per-target and epoch structure evaluators use the same
matcher settings and one candidate/reference per target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

EXPERIMENTS = {
    "base_500k": (
        "paper-figure1-zatom2-omol25-500k-omat24-500k-"
        "joint-properties-force-conditioned"
    ),
    "large_500k": (
        "paper-figure1-zatom2-l-omol25-500k-omat24-500k-"
        "joint-properties-force-conditioned"
    ),
    "base_full": "paper-table3-omol25-omat24-joint-properties-force-conditioned",
    "large_full": (
        "paper-table3-zatom2-l-omol25-omat24-joint-properties-force-conditioned"
    ),
}

PREFIX = "atomistic_property_metrics."
FORCE_SUM = PREFIX + "force_component_absolute_error_sum"
FORCE_COUNT = PREFIX + "num_force_components"
OMOL_PER_TARGET = "structure_prediction_metrics.molecule_best_rmsd_mean_top1"
OMAT_PER_TARGET = "structure_prediction_metrics.material_match_rate_top1"
OMOL_GROUPED = "zevals_epoch.molecule_best_rmsd_mean_top1"
OMAT_GROUPED = "zevals_epoch.material_match_rate_top1_at_1_scale_true"
SELECTED_FIELDS = (
    FORCE_SUM,
    FORCE_COUNT,
    OMOL_PER_TARGET,
    OMAT_PER_TARGET,
    OMOL_GROUPED,
    OMAT_GROUPED,
    "zevals_epoch.num_targets",
    "afd_score",
    "afd_num_samples",
    "afd_ref_sample_size",
)


def finite_float(value: str | None) -> float | None:
    """Parse one finite metric value while preserving missingness."""
    if not value:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def recover(train_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    """Recover complete and partial scalar measurements, retaining provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(dict)
    audit = {"rank_artifacts": [], "duplicate_rows": [], "conflicting_values": []}
    for condition, experiment in EXPERIMENTS.items():
        for job in sorted((train_dir / experiment).iterdir()):
            metrics_dir = job / "val_metrics"
            if not metrics_dir.is_dir():
                continue
            all_epochs = metrics_dir / "validation_output_all_epochs.csv"
            files = [all_epochs] if all_epochs.exists() else []
            rank_files = sorted(
                metrics_dir.glob("validation_output_rank_*_epoch_*.csv")
            )
            ranks_by_epoch = defaultdict(list)
            for path in rank_files:
                match = re.fullmatch(
                    r"validation_output_rank_(\d+)_epoch_(\d+)\.csv", path.name
                )
                if match:
                    ranks_by_epoch[int(match[2])].append(int(match[1]))
            for epoch, ranks in sorted(ranks_by_epoch.items()):
                audit["rank_artifacts"].append(
                    {
                        "condition": condition,
                        "job": str(job),
                        "epoch": epoch,
                        "rank_count": len(ranks),
                        "missing_ranks_of_64": sorted(set(range(64)) - set(ranks)),
                    }
                )
            files.extend(rank_files)
            for path in files:
                duplicate_count = 0
                with path.open(newline="") as handle:
                    for row in csv.DictReader(handle):
                        key = (condition, int(row["epoch"]), row["dataset"])
                        example_id = row["example_id"]
                        values = {
                            field: parsed
                            for field in SELECTED_FIELDS
                            if (parsed := finite_float(row.get(field))) is not None
                        }
                        if example_id in groups[key]:
                            duplicate_count += 1
                            old = groups[key][example_id]
                            for field, value in values.items():
                                if field in old["values"] and not math.isclose(
                                    old["values"][field],
                                    value,
                                    rel_tol=1e-12,
                                    abs_tol=1e-12,
                                ):
                                    audit["conflicting_values"].append(
                                        {
                                            "key": [*key, example_id, field],
                                            "kept": old["values"][field],
                                            "other": value,
                                            "other_source": str(path),
                                        }
                                    )
                                old["values"].setdefault(field, value)
                        else:
                            groups[key][example_id] = {
                                "values": values,
                                "source_file": str(path),
                            }
                if duplicate_count:
                    audit["duplicate_rows"].append(
                        {"source_file": str(path), "duplicate_count": duplicate_count}
                    )

    outputs = []
    coverage = []
    for (condition, epoch, dataset), by_id in sorted(groups.items()):
        rows = [row for name, row in by_id.items() if not name.startswith("__epoch_")]
        summaries = [row for name, row in by_id.items() if name.startswith("__epoch_")]
        domain = "omol" if dataset.startswith("omol25") else "omat"
        expected_count = 1000 if dataset.endswith("_generation") else 500
        coverage.append(
            {
                "condition": condition,
                "epoch": epoch,
                "dataset": dataset,
                "n_unique_samples": len(rows),
                "expected_samples": expected_count,
                "epoch_summary_rows": len(summaries),
            }
        )

        def append_metric(metric, value, n_samples, sources, note):
            complete = n_samples == expected_count
            outputs.append(
                {
                    "condition": condition,
                    "epoch": epoch,
                    "metric": metric,
                    "value": value,
                    "source_file": "|".join(sorted(set(sources))),
                    "n_samples": n_samples,
                    "status": (
                        "measured_local_complete"
                        if complete
                        else "measured_local_partial"
                    ),
                    "notes": f"{note}; {n_samples}/{expected_count} unique samples",
                }
            )

        if dataset.endswith("_generation"):
            for row in summaries:
                values = row["values"]
                if "afd_score" in values:
                    append_metric(
                        f"{domain}_afd",
                        values["afd_score"],
                        int(values["afd_num_samples"]),
                        [row["source_file"]],
                        "Logged epoch AFD summary; reference samples="
                        f"{int(values['afd_ref_sample_size'])}",
                    )
        elif dataset.endswith("_mlip"):
            valid = [
                row
                for row in rows
                if FORCE_SUM in row["values"] and FORCE_COUNT in row["values"]
            ]
            denominator = math.fsum(row["values"][FORCE_COUNT] for row in valid)
            if denominator:
                value = (
                    math.fsum(row["values"][FORCE_SUM] for row in valid) / denominator
                )
                append_metric(
                    f"{domain}_force_mae",
                    value,
                    len(valid),
                    [row["source_file"] for row in valid],
                    "sum(force_component_absolute_error_sum)/sum(num_force_components); "
                    f"components={int(denominator)}; eV/angstrom",
                )
        elif dataset.endswith("_structure_prediction"):
            metric = "omol_rmsd" if domain == "omol" else "omat_match"
            grouped_key = OMOL_GROUPED if domain == "omol" else OMAT_GROUPED
            per_target_key = OMOL_PER_TARGET if domain == "omol" else OMAT_PER_TARGET
            logged = [row for row in summaries if grouped_key in row["values"]]
            if logged:
                for row in logged:
                    append_metric(
                        metric,
                        row["values"][grouped_key],
                        int(row["values"]["zevals_epoch.num_targets"]),
                        [row["source_file"]],
                        f"Logged epoch metric {grouped_key}",
                    )
            else:
                valid = [row for row in rows if per_target_key in row["values"]]
                if valid:
                    append_metric(
                        metric,
                        math.fsum(row["values"][per_target_key] for row in valid)
                        / len(valid),
                        len(valid),
                        [row["source_file"] for row in valid],
                        "Recovered mean of unique per-target top1 metrics; same evaluator "
                        "and matcher settings as grouped metric; epoch summary absent",
                    )

    audit["coverage"] = coverage
    audit["notes"] = [
        "All datasets expect 500 validation targets except generation (1000).",
        "The same target is counted once per condition, epoch, and dataset.",
        "Split job directories for JOB_57908501 are merged by unique example_id.",
        "AFD is retained only from existing epoch summaries, never estimated here.",
        "A missing rank file does not establish the cause of validation interruption.",
    ]
    csv_path = output_dir / "local_metric_recovery.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "condition",
                "epoch",
                "metric",
                "value",
                "source_file",
                "n_samples",
                "status",
                "notes",
            ),
        )
        writer.writeheader()
        writer.writerows(outputs)
    audit_path = output_dir / "local_validation_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    if audit["conflicting_values"]:
        raise ValueError(f"Conflicting duplicate metric values: inspect {audit_path}")
    write_common_target_diagnostic(groups, output_dir)
    return csv_path, audit_path


def write_common_target_diagnostic(groups: dict, output_dir: Path) -> Path:
    """Compare epoch 19 on exactly the targets surviving the base 500k run.

    Missing ranks omit the same target indices from each domain. Comparing the
    partial aggregate to complete runs can reverse small model differences, so
    retain the complete runs' aggregates on this common subset as a diagnostic.
    """
    outputs = []
    for condition in EXPERIMENTS:
        for dataset in (
            "omol25_structure_prediction",
            "omat24_structure_prediction",
            "omol25_mlip",
            "omat24_mlip",
        ):
            baseline = groups.get(("base_500k", 19, dataset), {})
            candidate = groups.get((condition, 19, dataset), {})
            common_ids = {
                example_id
                for example_id in baseline
                if not example_id.startswith("__epoch_")
            }
            if not common_ids or not common_ids.issubset(candidate):
                continue
            rows = [candidate[example_id]["values"] for example_id in common_ids]
            if dataset.endswith("_mlip"):
                valid = [row for row in rows if FORCE_SUM in row and FORCE_COUNT in row]
                denominator = math.fsum(row[FORCE_COUNT] for row in valid)
                if len(valid) != len(common_ids) or not denominator:
                    continue
                value = math.fsum(row[FORCE_SUM] for row in valid) / denominator
            else:
                key = (
                    OMOL_PER_TARGET if dataset.startswith("omol25") else OMAT_PER_TARGET
                )
                valid = [row for row in rows if key in row]
                if len(valid) != len(common_ids):
                    continue
                value = math.fsum(row[key] for row in valid) / len(valid)
            outputs.append(
                {
                    "condition": condition,
                    "dataset": dataset,
                    "epoch": 19,
                    "common_targets": len(common_ids),
                    "value": value,
                }
            )
    path = output_dir / "common_target_epoch19_diagnostic.json"
    path.write_text(
        json.dumps(
            {
                "notes": (
                    "Diagnostic recomputation restricted to the same target IDs "
                    "surviving base500k epoch19. This is not an interpolation and does "
                    "not recover omitted validation targets."
                ),
                "metrics": outputs,
            },
            indent=2,
        )
        + "\n"
    )
    return path


def main() -> None:
    """Expose the bounded artifact audit as a reproducible command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("logs/train"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("logs/analysis/zatom2_figure2_20260906")
    )
    args = parser.parse_args()
    for path in recover(args.train_dir, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
