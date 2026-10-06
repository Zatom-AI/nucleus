#!/usr/bin/env python3
"""Add reproducible RMS/P95 force metadata to inference JSON fixtures.

The statistics are always computed from the authoritative force vectors; this
script never attempts to infer RMS or a percentile from ``mean_force_norm``.

For structure-prediction fixtures, use the matched MLIP JSON as the source::

    uv run python scripts/augment_force_conditioning_metadata.py paired-json \
      structure_prediction.json energy_force_prediction.json \
      --output structure_prediction.json

For atom-count generation fixtures that retain ASE-LMDB row provenance, use::

    uv run python scripts/augment_force_conditioning_metadata.py aselmdb \
      generation.json /path/to/val.aselmdb --output generation.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

MEAN_ABSOLUTE_TOLERANCE = 1e-12


def force_norm_statistics(
    force_array: Any, *, row_id: str
) -> tuple[float, float, float]:
    """Return mean, RMS, and linearly interpolated P95 force magnitudes."""
    try:
        forces = torch.as_tensor(force_array, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Example {row_id} has a non-numeric force array.") from exc
    if forces.ndim != 2 or forces.shape[-1] != 3 or forces.shape[0] == 0:
        raise ValueError(
            f"Example {row_id} must contain a non-empty [N_atoms, 3] force array."
        )
    if not torch.isfinite(forces).all():
        raise ValueError(f"Example {row_id} contains non-finite forces.")

    force_norms = torch.linalg.norm(forces, dim=-1)
    mean_force_norm = force_norms.mean()
    squared_magnitudes = torch.sum(forces.square(), dim=-1)
    force_rms = torch.sqrt(squared_magnitudes.mean())
    force_p95 = torch.quantile(torch.sqrt(squared_magnitudes), 0.95)
    return (
        float(mean_force_norm.item()),
        float(force_rms.item()),
        float(force_p95.item()),
    )


def _validate_stored_mean(
    metadata: Mapping[str, Any],
    *,
    actual_mean: float,
    row_id: str,
) -> None:
    stored_mean = metadata.get("mean_force_norm")
    if stored_mean is None:
        raise ValueError(f"Example {row_id} is missing mean_force_norm.")
    if not math.isclose(
        float(stored_mean),
        actual_mean,
        rel_tol=0.0,
        abs_tol=MEAN_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError(
            f"Example {row_id} does not match its force source: stored mean "
            f"{stored_mean!r}, recomputed mean {actual_mean!r}."
        )


def _set_force_statistics(
    metadata: dict[str, Any],
    *,
    force_rms: float,
    force_p95: float,
) -> None:
    metadata["force_rms"] = force_rms
    metadata["force_p95"] = force_p95
    metadata["has_force_rms_p95"] = True


def augment_from_paired_json(
    target_entries: dict[str, Any],
    force_entries: Mapping[str, Any],
) -> None:
    """Augment matched structure-prediction entries from an MLIP fixture."""
    if target_entries.keys() != force_entries.keys():
        raise ValueError("Target and force-reference JSON keys do not match exactly.")

    for row_id, specification in target_entries.items():
        reference = force_entries[row_id]
        for identity_key in ("input", "contig"):
            if specification.get(identity_key) != reference.get(identity_key):
                raise ValueError(
                    f"Example {row_id} has mismatched {identity_key!r} values."
                )
        reference_extra = reference.get("extra")
        if not isinstance(reference_extra, Mapping):
            raise ValueError(f"Example {row_id} is missing reference extra metadata.")
        forces = reference_extra.get("forces_gt_eV_per_angstrom")
        mean_force_norm, force_rms, force_p95 = force_norm_statistics(
            forces,
            row_id=row_id,
        )
        _validate_stored_mean(
            specification,
            actual_mean=mean_force_norm,
            row_id=row_id,
        )
        _set_force_statistics(
            specification,
            force_rms=force_rms,
            force_p95=force_p95,
        )


def augment_from_aselmdb(
    target_entries: dict[str, Any],
    *,
    database_path: Path,
) -> None:
    """Augment entries from source rows identified in their ``extra`` metadata."""
    try:
        from atomworks.ml.datasets import ASELMDBDataset
    except ImportError as exc:
        raise ImportError(
            "atomworks is required for the aselmdb mode; install the project "
            "development environment first."
        ) from exc

    database_path = database_path.expanduser().resolve()
    dataset = ASELMDBDataset(
        paths=[database_path],
        name="augment_force_conditioning_metadata",
        return_type="atoms",
        readonly=True,
        readahead=False,
        include_data=True,
    )
    for row_id, specification in target_entries.items():
        metadata = specification.get("extra")
        if not isinstance(metadata, dict):
            raise ValueError(f"Example {row_id} is missing mutable extra metadata.")
        source_shard = metadata.get("source_shard")
        if not source_shard or not database_path.as_posix().endswith(str(source_shard)):
            raise ValueError(
                f"Example {row_id} source shard {source_shard!r} does not match "
                f"{database_path}."
            )
        row_index = int(metadata["source_row_index"])
        if not 0 <= row_index < len(dataset):
            raise IndexError(
                f"Example {row_id} source row {row_index} is out of range."
            )
        atoms = dataset[row_index]
        expected_sid = metadata.get("sid")
        if expected_sid is not None and atoms.info.get("sid") != expected_sid:
            raise ValueError(f"Example {row_id} does not match its source SID.")
        expected_db_id = metadata.get("source_db_id")
        if expected_db_id is not None and atoms.info.get("db_id") != expected_db_id:
            raise ValueError(f"Example {row_id} does not match its source DB ID.")

        mean_force_norm, force_rms, force_p95 = force_norm_statistics(
            atoms.get_forces(),
            row_id=row_id,
        )
        _validate_stored_mean(
            metadata,
            actual_mean=mean_force_norm,
            row_id=row_id,
        )
        _set_force_statistics(
            metadata,
            force_rms=force_rms,
            force_p95=force_p95,
        )


def _read_json(path: Path) -> dict[str, Any]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return entries


def _write_json(path: Path, entries: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    paired_parser = subparsers.add_parser(
        "paired-json",
        help="Read force vectors from a matched energy/force prediction JSON.",
    )
    paired_parser.add_argument("target_json", type=Path)
    paired_parser.add_argument("force_json", type=Path)
    paired_parser.add_argument("--output", type=Path, required=True)

    aselmdb_parser = subparsers.add_parser(
        "aselmdb",
        help="Read force vectors from source ASE-LMDB rows.",
    )
    aselmdb_parser.add_argument("target_json", type=Path)
    aselmdb_parser.add_argument("database", type=Path)
    aselmdb_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the requested fixture augmentation."""
    args = _build_parser().parse_args(argv)
    target_entries = _read_json(args.target_json)
    if args.mode == "paired-json":
        augment_from_paired_json(target_entries, _read_json(args.force_json))
    else:
        augment_from_aselmdb(target_entries, database_path=args.database)
    _write_json(args.output, target_entries)


if __name__ == "__main__":
    main()
