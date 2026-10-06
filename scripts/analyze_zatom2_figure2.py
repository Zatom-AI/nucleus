#!/usr/bin/env python3
"""Build Figure 2 source tables and plots from the archived W&B histories.

Interpolation is linear in epoch and restricted to the observed range of each
condition and metric. Partial local validations remain partial measurements;
neither missing samples nor entirely absent metrics are synthesized.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

CONDITIONS = {
    "base_500k": ("Base · 500k/domain", 142.4, "#c77723", "--"),
    "large_500k": ("Large · 500k/domain", 276.0, "#c77723", "--"),
    "base_full": ("Base · full pools", 142.4, "#286a9b", "-"),
    "large_full": ("Large · full pools", 276.0, "#286a9b", "-"),
}
METRICS = {
    "omol_afd": ("OMol25 generation", "AFD ↓", "val/omol25_generation/afd_score", 1),
    "omat_afd": ("OMat24 generation", "AFD ↓", "val/omat24_generation/afd_score", 1),
    "omol_rmsd": (
        "OMol25 structure",
        "Mean RMSD (Å) ↓",
        "val/omol25_structure_prediction/zevals_epoch.molecule_best_rmsd_mean_top1",
        1,
    ),
    "omat_match": (
        "OMat24 structure",
        "Top-1 match (%) ↑",
        "val/omat24_structure_prediction/zevals_epoch.material_match_rate_top1_at_1_scale_true",
        100,
    ),
    "omol_force_mae": (
        "OMol25 MLIP",
        "Force MAE (eV/Å) ↓",
        "val/omol25_mlip/atomistic_property_metrics.force_component_mae_eV_per_angstrom",
        1,
    ),
    "omat_force_mae": (
        "OMat24 MLIP",
        "Force MAE (eV/Å) ↓",
        "val/omat24_mlip/atomistic_property_metrics.force_component_mae_eV_per_angstrom",
        1,
    ),
}
TRAIN_METRICS = {
    "total_loss": "train/per_epoch_total_loss",
    "flow_mse": "train/per_epoch_mse_loss_mean",
    "force_loss": "train/per_epoch_atomistic_force_loss",
}
DEFAULT_ROOT = Path("logs/analysis/zatom2_figure2_20260906")
AFD_METRICS = frozenset({"omol_afd", "omat_afd"})
CORRECTED_AFD_PROTOCOL = "species_consistent_afd_v1"
DEFAULT_AFD_MODELS = {
    "omol_afd": "Ty-Perez/ct-scd-omol25",
    "omat_afd": "Ty-Perez/ct-scd-amp20",
}
ORIGINAL_STATUSES = frozenset(
    {
        "measured_wandb",
        "measured_local_complete",
        "measured_wandb_partial",
        "measured_local_partial",
    }
)
RECOMPUTED_STATUSES = frozenset(
    {
        "recomputed_saved_complete",
        "recomputed_saved_partial",
        "recomputed_checkpoint_complete",
    }
)
COMPLETE_STATUSES = frozenset(
    {
        "measured_wandb",
        "measured_local_complete",
        "recomputed_saved_complete",
        "recomputed_checkpoint_complete",
    }
)
MEASUREMENT_COLUMNS = (
    "condition",
    "epoch",
    "metric",
    "value",
    "status",
    "source_file",
    "n_samples",
    "notes",
)


def finite_number(value):
    """Return a finite scalar, leaving absent/non-numeric values missing."""
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return None


def read_history(path: Path):
    """Stream raw history, retaining only non-null scalars used in this analysis.

    Full histories can contain hundreds of explicit null fields per row. The
    source archive remains untouched; discarding these from the analysis view
    avoids loading gigabytes of redundant null-filled dictionaries.
    """
    coordinates = {"_step", "_timestamp", "trainer/global_step", "trainer/epoch"}
    selected = {definition[2] for definition in METRICS.values()}
    selected.update(TRAIN_METRICS.values())
    selected.add("trainer/epoch")
    for _, _, key, _ in METRICS.values():
        prefix = key.rsplit("/", 1)[0]
        selected.update(
            prefix + "/" + suffix
            for suffix in (
                "afd_num_samples",
                "zevals_epoch.num_targets",
                "atomistic_property_metrics.num_energy_systems",
            )
        )
    retained, total_count = [], 0
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            total_count += 1
            values = {
                key: row[key]
                for key in selected
                if finite_number(row.get(key)) is not None
            }
            if values:
                values.update({key: row[key] for key in coordinates if key in row})
                retained.append(values)
    return retained, total_count


def extract_history(root: Path):
    """Associate validation with the most recent logged completed epoch.

    The trainer logs its epoch after training and before validation. Training
    epoch means are emitted before that epoch marker, so they are aligned by
    optimizer step with the epoch marker instead of forward filling.
    """
    validation, training, coverage = [], [], []
    for condition in CONDITIONS:
        path = root / "wandb" / condition / "history.jsonl.gz"
        rows, total_count = read_history(path)
        rows.sort(key=lambda row: row.get("_step", -1))
        epoch_by_step = {
            row["trainer/global_step"]: int(row["trainer/epoch"])
            for row in rows
            if finite_number(row.get("trainer/epoch")) is not None
            and finite_number(row.get("trainer/global_step")) is not None
        }
        validation_counts = {}
        for row in rows:
            for key, value in row.items():
                if key.startswith("val/") and key.endswith(
                    (
                        "/afd_num_samples",
                        "/zevals_epoch.num_targets",
                        "/atomistic_property_metrics.num_energy_systems",
                    )
                ):
                    if finite_number(value) is not None:
                        validation_counts[(row.get("trainer/global_step"), key)] = value
        current_epoch = None
        for row in rows:
            epoch = finite_number(row.get("trainer/epoch"))
            if epoch is not None:
                current_epoch = int(epoch)
            for metric, (_, _, key, _) in METRICS.items():
                value = finite_number(row.get(key))
                if value is not None:
                    if current_epoch is None:
                        raise ValueError(f"{condition}: validation lacks an epoch")
                    step_epoch = epoch_by_step.get(row.get("trainer/global_step"))
                    if step_epoch != current_epoch:
                        raise ValueError(f"{condition}: validation epoch/step mismatch")
                    prefix = key.rsplit("/", 1)[0]
                    if metric.endswith("afd"):
                        count_key = prefix + "/afd_num_samples"
                    elif metric.endswith("force_mae"):
                        count_key = (
                            prefix + "/atomistic_property_metrics.num_energy_systems"
                        )
                    else:
                        count_key = prefix + "/zevals_epoch.num_targets"
                    count = finite_number(
                        validation_counts.get(
                            (row.get("trainer/global_step"), count_key)
                        )
                    )
                    expected = 1000 if metric.endswith("afd") else 500
                    status = "measured_wandb"
                    if count is not None and count < expected:
                        status = "measured_wandb_partial"
                    validation.append(
                        {
                            "condition": condition,
                            "epoch": current_epoch,
                            "metric": metric,
                            "value": value,
                            "status": status,
                            "source_file": str(path),
                            "wandb_step": row.get("_step"),
                            "global_step": row.get("trainer/global_step"),
                            "n_samples": count,
                            "notes": key,
                        }
                    )
            for metric, key in TRAIN_METRICS.items():
                value = finite_number(row.get(key))
                if value is not None:
                    train_epoch = epoch_by_step.get(row.get("trainer/global_step"))
                    training.append(
                        {
                            "condition": condition,
                            "epoch": train_epoch,
                            "metric": metric,
                            "value": value,
                            "global_step": row.get("trainer/global_step"),
                            "wandb_step": row.get("_step"),
                        }
                    )
        keys = {key for row in rows for key in row}
        coverage.append(
            {
                "condition": condition,
                "history_rows": total_count,
                "last_logged_epoch": current_epoch,
                "validation_metric_keys": sorted(
                    k for k in keys if k.startswith("val/")
                ),
            }
        )
    return pd.DataFrame(validation), pd.DataFrame(training), coverage


def read_recomputed_measurements(root: Path, paths: list[Path] | None = None):
    """Read explicit or known recomputation tables without modifying their sources.

    Control/protocol metadata is retained from each source JSON when available.
    Rows remain candidates here; filtering and source precedence happen later.
    """
    explicit = paths is not None
    if paths is None:
        paths = [
            root / "recomputed/recomputed_metrics.csv",
            root / "recomputed/corrected_saved/recomputed_metrics.csv",
            root / "recomputed/structures_saved/recomputed_metrics.csv",
            root / "recomputed/omol25_encoder_update/recomputed_metrics.csv",
            root / "recomputed/checkpoint_base_full_epoch65/pooled/pooled_metrics.csv",
            *[
                root / f"recomputed/checkpoint_{condition}/pooled/pooled_metrics.csv"
                for condition in CONDITIONS
            ],
        ]
    frames, source_metadata = [], {}
    for path in dict.fromkeys(Path(path) for path in paths):
        if not path.exists() and not explicit:
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        required = set(MEASUREMENT_COLUMNS) - {"notes"}
        if missing := required - set(frame.columns):
            raise ValueError(
                f"Recomputation CSV {path} lacks columns: {sorted(missing)}"
            )
        frame["recomputed_csv"] = str(path)
        # Encoder identity comes from scoring provenance, not a stale CSV label.
        frame["afd_model_name"] = pd.Series(None, index=frame.index, dtype=object)
        if "notes" not in frame:
            frame["notes"] = ""
        for index, row in frame.iterrows():
            if not isinstance(row.source_file, str):
                continue
            source = Path(row.source_file)
            if not source.is_absolute() and not source.exists():
                source = path.parent / source
            if source.suffix != ".json" or not source.is_file():
                continue
            if source not in source_metadata:
                source_metadata[source] = json.loads(source.read_text())
            metadata = source_metadata[source]
            frame.loc[index, "afd_model_name"] = (metadata.get("scoring") or {}).get(
                "model_name"
            )
            for name in ("metric_protocol", "supersedes_historical_afd"):
                if name in metadata:
                    frame.loc[index, name] = metadata[name]
            for name in ("control_validation", "species_consistency"):
                proofs = [
                    (container.get(name) or {}).get("passed")
                    for container in (metadata, metadata.get("scoring") or {})
                    if (container.get(name) or {}).get("passed") is not None
                ]
                if proofs:
                    # An explicit failed proof always vetoes conflicting metadata.
                    frame.loc[index, f"{name}_passed"] = all(
                        proof is True for proof in proofs
                    )
        frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=MEASUREMENT_COLUMNS)
    )


def measurement_exclusion_reasons(
    candidates: pd.DataFrame,
    *,
    corrected_afd: bool = False,
    include_partial_afd: bool = False,
    expected_afd_models: dict[str, str] | None = None,
) -> pd.Series:
    """Explicitly reject failed/unknown runs and incompatible AFD protocols."""
    reasons = pd.Series("", index=candidates.index, dtype=object)
    if candidates.empty:
        return reasons

    def reject(mask, reason):
        reasons.loc[mask & reasons.eq("")] = reason

    statuses = candidates["status"]
    reject(~statuses.isin(ORIGINAL_STATUSES | RECOMPUTED_STATUSES), "unapproved_status")
    reject(~candidates.condition.isin(CONDITIONS), "unknown_condition")
    reject(~candidates.metric.isin(METRICS), "unknown_metric")
    values = pd.to_numeric(candidates.value, errors="coerce")
    reject(~np.isfinite(values), "nonfinite_value")
    epochs = pd.to_numeric(candidates.epoch, errors="coerce")
    reject(
        ~np.isfinite(epochs) | epochs.lt(0) | epochs.ne(np.floor(epochs)),
        "invalid_epoch",
    )
    counts = pd.to_numeric(
        candidates.get("n_samples", pd.Series(np.nan, index=candidates.index)),
        errors="coerce",
    )
    expected = candidates.metric.map(
        lambda metric: 1000 if metric in AFD_METRICS else 500
    )
    recomputed = statuses.isin(RECOMPUTED_STATUSES)
    complete = statuses.isin(
        {"recomputed_saved_complete", "recomputed_checkpoint_complete"}
    )
    reject(complete & counts.ne(expected), "incomplete_recomputed_coverage")
    reject(
        statuses.eq("recomputed_saved_partial")
        & (
            ~np.isfinite(counts)
            | counts.le(0)
            | counts.ge(expected)
            | counts.ne(np.floor(counts))
        ),
        "invalid_partial_recomputed_coverage",
    )
    for name in ("control_validation_passed", "species_consistency_passed"):
        if name in candidates:
            failed = (
                candidates[name].astype(str).str.lower().isin({"false", "0", "0.0"})
            )
            reject(recomputed & failed, f"failed_{name}")
    if expected_afd_models is not None:
        selected_models = candidates.metric.map(expected_afd_models)
        gated = candidates.metric.isin(AFD_METRICS) & selected_models.notna()
        actual_models = candidates.get(
            "afd_model_name", pd.Series(None, index=candidates.index, dtype=object)
        )
        known = actual_models.map(lambda value: isinstance(value, str) and bool(value))
        reject(gated & ~known, "unverified_afd_model_name")
        reject(gated & actual_models.ne(selected_models), "incompatible_afd_model_name")
    if corrected_afd:
        afd = candidates.metric.isin(AFD_METRICS)
        reject(afd & statuses.isin(ORIGINAL_STATUSES), "invalid_historical_afd_species")
        protocol = candidates.get(
            "metric_protocol", pd.Series(None, index=candidates.index)
        )
        reject(
            afd & recomputed & protocol.ne(CORRECTED_AFD_PROTOCOL),
            "unverified_corrected_afd_protocol",
        )
        for name in ("control_validation_passed", "species_consistency_passed"):
            verified = candidates.get(
                name, pd.Series(None, index=candidates.index)
            ).map(lambda value: str(value).lower() == "true")
            reject(afd & recomputed & ~verified, f"unverified_{name}")
        if not include_partial_afd:
            reject(
                afd & statuses.eq("recomputed_saved_partial"),
                "partial_afd_requires_separate_matched_n_audit",
            )
    elif "metric_protocol" in candidates:
        reject(
            candidates.metric.isin(AFD_METRICS)
            & recomputed
            & candidates.metric_protocol.eq(CORRECTED_AFD_PROTOCOL),
            "corrected_afd_requires_explicit_policy",
        )
    return reasons


def combine_measurements(
    wandb: pd.DataFrame,
    local: pd.DataFrame,
    recomputed: pd.DataFrame | None = None,
    *,
    corrected_afd: bool = False,
    include_partial_afd: bool = False,
    expected_afd_models: dict[str, str] | None = None,
):
    """Prefer complete observations; among partials prefer greater target coverage.

    Original complete measurements precede complete rescoring of saved samples,
    then independent checkpoint evaluations. Explicit corrected-AFD mode excludes
    the historically invalid species annotations entirely. Partial corrected AFD
    is retained only in a separate audit because its matched reference N differs.
    """
    frames = [
        frame
        for frame in (wandb, local, recomputed)
        if frame is not None and not frame.empty
    ]
    if not frames:
        return pd.DataFrame(columns=MEASUREMENT_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    for column in MEASUREMENT_COLUMNS:
        if column not in combined:
            combined[column] = np.nan
    reasons = measurement_exclusion_reasons(
        combined,
        corrected_afd=corrected_afd,
        include_partial_afd=include_partial_afd,
        expected_afd_models=expected_afd_models,
    )
    combined = combined[reasons.eq("")].copy()
    combined["value"] = pd.to_numeric(combined["value"])
    combined["epoch"] = pd.to_numeric(combined["epoch"]).astype(int)
    combined["n_samples"] = pd.to_numeric(combined["n_samples"], errors="coerce")
    priority = {
        "measured_wandb": 0,
        "measured_local_complete": 1,
        "recomputed_saved_complete": 2,
        "recomputed_checkpoint_complete": 3,
        "measured_wandb_partial": 4,
        "measured_local_partial": 4,
        "recomputed_saved_partial": 4,
    }
    combined["priority"] = combined["status"].map(priority)
    combined["partial_coverage"] = combined.n_samples.where(
        combined.priority.eq(4), 0
    ).fillna(-1)
    combined["partial_source_priority"] = combined.status.map(
        {
            "measured_wandb_partial": 0,
            "measured_local_partial": 1,
            "recomputed_saved_partial": 2,
        }
    ).fillna(0)
    combined["sample_origin"] = combined.status.map(
        {
            **{status: "original_validation" for status in ORIGINAL_STATUSES},
            "recomputed_saved_complete": "rescored_original_samples",
            "recomputed_saved_partial": "rescored_original_samples",
            "recomputed_checkpoint_complete": "independent_checkpoint_resampling",
        }
    )
    # Keep the final W&B observation when a resumed epoch was logged twice.
    combined["sequence"] = np.arange(len(combined))
    return (
        combined.sort_values(
            ["priority", "partial_coverage", "partial_source_priority", "sequence"],
            ascending=[True, False, True, False],
        )
        .drop_duplicates(["condition", "epoch", "metric"])
        .drop(
            columns=[
                "priority",
                "partial_coverage",
                "partial_source_priority",
                "sequence",
            ]
        )
        .sort_values(["condition", "metric", "epoch"])
    )


def shared_afd_model(metric: str, *models) -> str | None:
    """Reject calculations combining known AFD encoders or unverified identities.

    All-absent identities remain valid only for legacy generic-helper callers; the
    CLI rejects every absent identity before selecting measurements.
    """
    if metric not in AFD_METRICS:
        return None
    known = [model for model in models if isinstance(model, str) and model]
    if not known:
        return None
    if len(known) != len(models) or len(set(known)) != 1:
        raise ValueError(f"Cannot combine different or unverified AFD models: {models}")
    return known[0]


def estimate_at_epoch(
    series: pd.DataFrame, epoch: int, *, expected_afd_model: str | None = None
):
    """Use an exact observation or bracketing observations; never extrapolate."""
    if expected_afd_model is not None:
        models = series.get(
            "afd_model_name", pd.Series(None, index=series.index, dtype=object)
        )
        series = series[models.eq(expected_afd_model)]
    series = series.sort_values("epoch")
    exact = series[series["epoch"] == epoch]
    if len(exact):
        row = exact.iloc[-1].to_dict()
        row.update(left_epoch=epoch, right_epoch=epoch, right_weight=0.0)
        return row
    # Prefer complete validation endpoints. Partial endpoints are a separately
    # labelled fallback only when complete observations cannot bracket target.
    complete = series[~series["status"].str.contains("partial")]
    for candidates in (complete, series):
        left = candidates[candidates["epoch"] < epoch]
        right = candidates[candidates["epoch"] > epoch]
        if left.empty or right.empty:
            continue
        lo, hi = left.iloc[-1], right.iloc[0]
        model = shared_afd_model(
            lo.metric, lo.get("afd_model_name"), hi.get("afd_model_name")
        )
        weight = (epoch - lo.epoch) / (hi.epoch - lo.epoch)
        partial = any("partial" in status for status in (lo.status, hi.status))
        return {
            "epoch": epoch,
            "value": (1 - weight) * lo.value + weight * hi.value,
            "status": "interpolated_partial" if partial else "interpolated",
            "left_epoch": lo.epoch,
            "right_epoch": hi.epoch,
            "right_weight": weight,
            "source_file": f"{lo.source_file}|{hi.source_file}",
            "left_status": lo.status,
            "right_status": hi.status,
            "left_sample_origin": lo.get("sample_origin"),
            "right_sample_origin": hi.get("sample_origin"),
            "afd_model_name": model,
            "left_afd_model_name": lo.get("afd_model_name"),
            "right_afd_model_name": hi.get("afd_model_name"),
            "n_samples": None,
            "notes": f"Linear epoch interpolation; endpoints {lo.value}, {hi.value}",
        }
    return {
        "epoch": epoch,
        "value": np.nan,
        "status": "missing",
        "notes": "No exact measurement or same-condition bracketing observations",
        "afd_model_name": expected_afd_model,
    }


def target_table(
    measurements: pd.DataFrame,
    epoch: int,
    *,
    expected_afd_models: dict[str, str] | None = None,
):
    """Preserve a row for every requested condition/panel, including gaps."""
    rows = []
    for condition in CONDITIONS:
        for metric in METRICS:
            subset = measurements[
                (measurements.condition == condition) & (measurements.metric == metric)
            ]
            row = estimate_at_epoch(
                subset,
                epoch,
                expected_afd_model=(expected_afd_models or {}).get(metric),
            )
            row.update(condition=condition, metric=metric)
            rows.append(row)
    return pd.DataFrame(rows)


def afd_imputation_scenarios(target: pd.DataFrame):
    """Explore the unobserved factorial cell under two untested assumptions.

    Neither scenario is a temporal interpolation or an empirical scaling
    result. The three available cells cannot identify a model-by-data
    interaction. These scenarios must remain separate from Figure 2 data.
    """
    rows = []
    for metric in ("omol_afd", "omat_afd"):
        values = target[target.metric == metric].set_index("condition")
        if np.isfinite(values.loc["base_500k", "value"]):
            continue
        source = values.loc[["base_full", "large_full", "large_500k"]]
        if not np.isfinite(source.value).all():
            continue
        model = shared_afd_model(
            metric,
            *source.get("afd_model_name", pd.Series(None, index=source.index)),
        )
        base_full, large_full, large_small = source.value.to_numpy()
        additive = base_full + large_small - large_full
        multiplicative = (
            base_full * large_small / large_full
            if min(base_full, large_full, large_small) > 0
            else np.nan
        )
        for assumption, estimate in (
            ("no interaction in raw AFD", additive),
            ("no interaction in log AFD", multiplicative),
        ):
            rows.append(
                {
                    "condition": "base_500k",
                    "metric": metric,
                    "epoch": int(values.loc["base_500k", "epoch"]),
                    "value": estimate if estimate >= 0 else np.nan,
                    "status": "assumption_based_imputation",
                    "afd_model_name": model,
                    "assumption": assumption,
                    "base_full": base_full,
                    "large_full": large_full,
                    "large_500k": large_small,
                    "source_statuses": "|".join(source.status),
                    "source_left_epochs": "|".join(source.left_epoch.astype(str)),
                    "source_right_epochs": "|".join(source.right_epoch.astype(str)),
                    "notes": "Not an observation or temporal interpolation; interaction unidentifiable",
                }
            )
    return pd.DataFrame(rows)


def interpolation_sensitivity(measurements: pd.DataFrame, target: pd.DataFrame):
    """Expose sensitivity to using nearby partial validation endpoints."""
    rows = []
    for row in target.itertuples():
        if not row.status.startswith("interpolated"):
            continue
        series = measurements[
            (measurements.condition == row.condition)
            & (measurements.metric == row.metric)
        ].sort_values("epoch")
        lo = series[series.epoch < row.epoch].iloc[-1]
        hi = series[series.epoch > row.epoch].iloc[0]
        model = shared_afd_model(
            row.metric,
            getattr(row, "afd_model_name", None),
            lo.get("afd_model_name"),
            hi.get("afd_model_name"),
        )
        weight = (row.epoch - lo.epoch) / (hi.epoch - lo.epoch)
        nearest = (1 - weight) * lo.value + weight * hi.value
        rows.append(
            {
                "condition": row.condition,
                "metric": row.metric,
                "afd_model_name": model,
                "epoch": row.epoch,
                "complete_bracket_value": row.value,
                "complete_left_epoch": row.left_epoch,
                "complete_right_epoch": row.right_epoch,
                "nearest_bracket_value": nearest,
                "nearest_left_epoch": lo.epoch,
                "nearest_right_epoch": hi.epoch,
                "nearest_left_status": lo.status,
                "nearest_right_status": hi.status,
                "nearest_left_n": lo.n_samples,
                "nearest_right_n": hi.n_samples,
                "nearest_minus_complete": nearest - row.value,
                "notes": (
                    "Sensitivity comparison; not a confidence interval"
                    + (
                        "; partial AFD uses different generated/reference N and is "
                        "excluded from primary interpolation"
                        if row.metric in AFD_METRICS
                        and any(
                            "partial" in status for status in (lo.status, hi.status)
                        )
                        else ""
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def matched_target_table(target: pd.DataFrame, diagnostic_path: Path):
    """Restrict original exact samples only while a relevant endpoint is incomplete.

    The old common-target diagnostic belongs to the original validation samples.
    It cannot replace rescored samples or independent checkpoint evaluations.
    Once all relevant exact endpoints are complete, retain their full target sets.
    """
    matched = target.copy()
    if not diagnostic_path.exists():
        return matched
    payload = json.loads(diagnostic_path.read_text())
    metric_by_dataset = {
        "omol25_structure_prediction": "omol_rmsd",
        "omat24_structure_prediction": "omat_match",
        "omol25_mlip": "omol_force_mae",
        "omat24_mlip": "omat_force_mae",
    }
    diagnostic_rows = payload["metrics"]
    eligible_groups = set()
    for dataset, metric in metric_by_dataset.items():
        rows = [row for row in diagnostic_rows if row["dataset"] == dataset]
        for epoch in {row["epoch"] for row in rows}:
            conditions = {row["condition"] for row in rows if row["epoch"] == epoch}
            relevant = target[
                target.condition.isin(conditions)
                & target.metric.eq(metric)
                & target.epoch.eq(epoch)
                & target.status.isin(ORIGINAL_STATUSES)
            ]
            counts = pd.to_numeric(
                relevant.get("n_samples", pd.Series(np.nan, index=relevant.index)),
                errors="coerce",
            )
            incomplete = relevant.status.str.contains("partial") | counts.lt(500)
            if incomplete.any():
                eligible_groups.add((dataset, epoch))
    for row in diagnostic_rows:
        if (row["dataset"], row["epoch"]) not in eligible_groups:
            continue
        mask = (
            (matched.condition == row["condition"])
            & (matched.metric == metric_by_dataset[row["dataset"]])
            & (matched.epoch == row["epoch"])
            & matched.status.isin(ORIGINAL_STATUSES)
        )
        if not mask.any():
            continue
        for column in ("value", "n_samples", "status", "source_file", "notes"):
            matched.loc[mask, f"original_{column}"] = matched.loc[mask, column]
        matched.loc[mask, "value"] = row["value"]
        matched.loc[mask, "n_samples"] = row["common_targets"]
        matched.loc[mask, "status"] = "measured_common_targets"
        matched.loc[mask, "source_file"] = str(diagnostic_path)
        matched.loc[mask, "notes"] = (
            "Original validation restricted to common observed target IDs while "
            "a relevant original endpoint is incomplete; no sample imputation"
        )
    return matched


def plot_imputation_scenarios(scenarios: pd.DataFrame, output: Path):
    if scenarios.empty:
        # A completed checkpoint evaluation can remove the previously missing
        # factorial cell. Do not leave obsolete imputation plots beside it.
        for suffix in ("pdf", "png"):
            output.with_suffix(f".{suffix}").unlink(missing_ok=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.3))
    for metric, ax in zip(("omol_afd", "omat_afd"), axes, strict=True):
        rows = scenarios[scenarios.metric == metric]
        style_axes(ax, METRICS[metric][0], "Imputed base 500k AFD")
        ax.scatter(
            range(len(rows)),
            rows.value,
            marker="D",
            facecolors="none",
            edgecolors="#c77723",
            s=65,
        )
        ax.set_xticks([0, 1], ["Additive\nassumption", "Multiplicative\nassumption"])
        ax.set_xlim(-0.5, 1.5)
        for x, value in enumerate(rows.value):
            ax.annotate(
                f"{value:.4f}",
                (x, value),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        ax.margins(y=0.55)
    fig.suptitle(
        "Missing AFD scenarios · untested model/data interaction assumptions",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.01,
        "Illustrative estimates only; these do not establish scaling trends.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    save_figure(fig, output)


def style_axes(ax, panel, ylabel):
    ax.set_title(panel, loc="left", fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def save_figure(fig, output: Path):
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_target(target: pd.DataFrame, output: Path, epoch: int):
    """Plot the four manuscript panels with explicit estimated markers."""
    panel_metrics = ("omol_afd", "omat_match", "omol_force_mae", "omat_force_mae")
    target = target[target.metric.isin(panel_metrics)]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    for (metric, (title, ylabel, _, scale)), ax, letter in zip(
        ((metric, METRICS[metric]) for metric in panel_metrics),
        axes.flat,
        "abcd",
        strict=True,
    ):
        style_axes(ax, f"({letter}) {title}", ylabel)
        for support in ("500k", "full"):
            conditions = [f"base_{support}", f"large_{support}"]
            subset = target[
                (target.metric == metric) & target.condition.isin(conditions)
            ]
            values = subset.set_index("condition").reindex(conditions)
            color = CONDITIONS[conditions[0]][2]
            xs = [CONDITIONS[c][1] for c in conditions]
            ax.plot(
                xs,
                values.value.to_numpy() * scale,
                color=color,
                linestyle=CONDITIONS[conditions[0]][3],
                alpha=0.7,
            )
            for condition, x in zip(conditions, xs, strict=True):
                row = values.loc[condition]
                if not np.isfinite(row.value):
                    continue
                marker = "D" if row.status.startswith("interpolated") else "o"
                if row.status == "recomputed_checkpoint_complete":
                    marker = "s"
                if (
                    "partial" in row.status or row.status == "measured_common_targets"
                ) and not row.status.startswith("interpolated"):
                    marker = "^"
                filled = row.status in COMPLETE_STATUSES
                ax.plot(
                    x,
                    row.value * scale,
                    marker=marker,
                    color=color,
                    markerfacecolor=color if filled else "white",
                    markersize=6,
                )
            if values.value.isna().any():
                absent = ", ".join(
                    c.split("_")[0]
                    for c in conditions
                    if pd.isna(values.loc[c, "value"])
                )
                ax.text(
                    0.03,
                    0.04 if support == "500k" else 0.17,
                    f"{support}: {absent} unavailable",
                    transform=ax.transAxes,
                    va="bottom",
                    fontsize=7,
                    color=color,
                )
        ax.set_xticks([142.4, 276.0], ["Base\n142.4M", "Large\n276.0M"])
        ax.set_xlim(115, 303)
    handles = [
        Line2D([], [], color="#c77723", ls="--", label="500k per domain"),
        Line2D([], [], color="#286a9b", label="Full pools"),
        Line2D([], [], color="0.25", marker="o", ls="", label="Measured"),
    ]
    if target.status.eq("recomputed_checkpoint_complete").any():
        handles.append(
            Line2D(
                [], [], color="0.25", marker="s", ls="", label="New checkpoint samples"
            )
        )
    if (
        target.status.str.contains("partial")
        | target.status.eq("measured_common_targets")
    ).any():
        handles.append(
            Line2D(
                [],
                [],
                color="0.25",
                marker="^",
                mfc="white",
                ls="",
                label="Partial/common targets",
            )
        )
    if target.status.str.startswith("interpolated").any():
        handles.append(
            Line2D(
                [],
                [],
                color="0.25",
                marker="D",
                mfc="white",
                ls="",
                label="Interpolated",
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        f"Model and data scaling · epoch {epoch} ({epoch + 1} training epochs)",
        fontsize=12,
        y=0.99,
    )
    fig.text(
        0.5,
        0.94,
        "Matched training draws; one training seed per condition",
        ha="center",
        fontsize=9,
        color="0.35",
    )
    interpolated = target[target.status.str.startswith("interpolated")]
    bracket_panels = {}
    for metric, letter in zip(panel_metrics, "abcd", strict=True):
        rows = interpolated[interpolated.metric.eq(metric)]
        if rows.empty:
            continue
        bracket = ", ".join(
            f"{row.condition.replace('_', ' ')} {int(row.left_epoch)}–{int(row.right_epoch)}"
            for row in rows.itertuples()
        )
        bracket_panels.setdefault(bracket, []).append(letter)
    if bracket_panels:
        details = "; ".join(
            f"({','.join(panels)}): {bracket}"
            for bracket, panels in bracket_panels.items()
        )
        fig.text(
            0.5,
            0.90,
            f"Interpolation brackets (epochs): {details}",
            ha="center",
            fontsize=7,
            color="0.35",
        )
    fig.tight_layout(rect=(0, 0.13, 1, 0.925), h_pad=1.6, w_pad=2.0)
    save_figure(fig, output)


def plot_histories(measurements: pd.DataFrame, output: Path, epoch: int):
    fig, axes = plt.subplots(3, 2, figsize=(9, 8))
    for (metric, (title, ylabel, _, scale)), ax, letter in zip(
        METRICS.items(), axes.flat, "abcdef", strict=True
    ):
        style_axes(ax, f"({letter}) {title}", ylabel)
        for condition, (label, _, color, _) in CONDITIONS.items():
            subset = measurements[
                (measurements.condition == condition) & (measurements.metric == metric)
            ].sort_values("epoch")
            complete = subset[~subset.status.str.contains("partial")]
            partial = subset[subset.status.str.contains("partial")]
            linestyle = "--" if condition.startswith("base") else "-"
            ax.plot(
                complete.epoch + 1,
                complete.value * scale,
                color=color,
                ls=linestyle,
                marker="o",
                ms=3,
                label=label,
            )
            resampled = complete[complete.status.eq("recomputed_checkpoint_complete")]
            ax.scatter(
                resampled.epoch + 1,
                resampled.value * scale,
                color=color,
                marker="s",
                s=22,
                zorder=3,
            )
            ax.scatter(
                partial.epoch + 1,
                partial.value * scale,
                color=color,
                marker="x",
                s=17,
                alpha=0.65,
            )
        ax.axvline(epoch + 1, color="0.5", lw=0.8, ls=":")
        ax.set_xlabel("Completed training epochs", fontsize=9)
    fig.legend(
        *axes.flat[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "Validation histories · crosses: partial samples; squares: new checkpoint samples",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    save_figure(fig, output)


def plot_training(training: pd.DataFrame, output: Path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    for metric, ax in zip(TRAIN_METRICS, axes, strict=True):
        style_axes(ax, metric.replace("_", " ").capitalize(), "Epoch mean loss")
        for condition, (label, _, color, _) in CONDITIONS.items():
            subset = training[
                (training.condition == condition) & (training.metric == metric)
            ]
            subset = subset.dropna(subset=["epoch"]).drop_duplicates(
                "epoch", keep="last"
            )
            ax.plot(
                subset.epoch + 1,
                subset.value,
                color=color,
                ls="--" if condition.startswith("base") else "-",
                label=label,
            )
        ax.set_xlabel("Completed training epochs", fontsize=9)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save_figure(fig, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--epoch", type=int, default=19)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to ROOT/analysis; use a separate directory to preserve an earlier analysis.",
    )
    parser.add_argument(
        "--recomputed-csv",
        action="append",
        type=Path,
        help="Recomputation table; repeat for multiple sources. Omit to read known recomputed tables.",
    )
    parser.add_argument(
        "--corrected-afd",
        action="store_true",
        default=True,
        help="Use species_consistent_afd_v1 recomputations (default); historical AFD is always excluded.",
    )
    parser.add_argument(
        "--omol-afd-model",
        default=DEFAULT_AFD_MODELS["omol_afd"],
        help="Required molecular AFD encoder (default: %(default)s); use Ty-Perez/ct-scd-geom10 to reproduce the earlier analysis.",
    )
    args = parser.parse_args()
    expected_afd_models = {**DEFAULT_AFD_MODELS, "omol_afd": args.omol_afd_model}
    output = args.output_dir or args.root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    wandb, training, coverage = extract_history(args.root)
    local_path = args.root / "local_metric_recovery.csv"
    local = pd.read_csv(local_path) if local_path.exists() else pd.DataFrame()
    recomputed = read_recomputed_measurements(args.root, args.recomputed_csv)
    candidates = pd.concat([wandb, local, recomputed], ignore_index=True)
    measurements = combine_measurements(
        wandb,
        local,
        recomputed,
        corrected_afd=args.corrected_afd,
        expected_afd_models=expected_afd_models,
    )
    audit = combine_measurements(
        wandb,
        local,
        recomputed,
        corrected_afd=args.corrected_afd,
        include_partial_afd=True,
        expected_afd_models=expected_afd_models,
    )
    exclusions = measurement_exclusion_reasons(
        candidates,
        corrected_afd=args.corrected_afd,
        expected_afd_models=expected_afd_models,
    )
    excluded = candidates[exclusions.ne("")].copy()
    excluded["exclusion_reason"] = exclusions[exclusions.ne("")]
    target = target_table(
        measurements, args.epoch, expected_afd_models=expected_afd_models
    )
    if args.corrected_afd:
        afd = target.metric.isin(AFD_METRICS)
        target.loc[afd, "metric_protocol"] = CORRECTED_AFD_PROTOCOL
        target.loc[afd, "notes"] = (
            target.loc[afd, "notes"].fillna("")
            + "; historical AFD excluded because output atomic numbers were stale"
        )
    matched = matched_target_table(
        target, args.root / "common_target_epoch19_diagnostic.json"
    )
    scenarios = afd_imputation_scenarios(target)
    sensitivity = interpolation_sensitivity(audit, target)
    wandb.to_csv(output / "wandb_validation.csv", index=False)
    measurements.to_csv(output / "validation_measurements.csv", index=False)
    audit.to_csv(output / "validation_measurements_audit.csv", index=False)
    recomputed.to_csv(output / "recomputation_candidates.csv", index=False)
    excluded.to_csv(output / "excluded_measurement_candidates.csv", index=False)
    training.to_csv(output / "training_epoch_means.csv", index=False)
    target.to_csv(output / f"figure2_epoch{args.epoch}.csv", index=False)
    matched.to_csv(
        output / f"figure2_epoch{args.epoch}_matched_targets.csv", index=False
    )
    scenarios.to_csv(output / "afd_imputation_scenarios.csv", index=False)
    sensitivity.to_csv(output / "interpolation_sensitivity.csv", index=False)
    (output / "history_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    plot_target(target, output / "figure2_scaling", args.epoch)
    plot_target(matched, output / "figure2_scaling_matched_targets", args.epoch)
    plot_histories(audit, output / "validation_trends", args.epoch)
    plot_training(training, output / "training_trends")
    plot_imputation_scenarios(scenarios, output / "afd_imputation_scenarios")
    print(
        target[
            ["condition", "metric", "value", "status", "left_epoch", "right_epoch"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
