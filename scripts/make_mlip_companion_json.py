#!/usr/bin/env python3
"""Build an MLIP inference manifest from a matched structure manifest.

The source structure manifest supplies the sampled cohort, CIF paths, and
source-record provenance. Energy and force labels are always re-read from the
authoritative ASE-LMDB validation records; force vectors are never inferred
from summary statistics.

Examples::

    uv run python scripts/make_mlip_companion_json.py \
      models/zatom2/docs/examples/omol25_val_structure_prediction_500.json \
      /path/to/omol25/val \
      --domain omol25 \
      --output models/zatom2/docs/examples/omol25_val_energy_force_prediction_500.json

    uv run python scripts/make_mlip_companion_json.py \
      models/zatom2/docs/examples/omat24_val_structure_prediction_500.json \
      /path/to/omat24/val \
      --domain omat24 \
      --output models/zatom2/docs/examples/omat24_val_energy_force_prediction_500.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase.geometry import cell_to_cellpar
from biotite.structure.io.pdbx import CIFFile, get_structure

COORDINATE_ABSOLUTE_TOLERANCE = 1e-5
CELL_PARAMETER_ABSOLUTE_TOLERANCE = 1e-7
FRACTIONAL_COORDINATE_ABSOLUTE_TOLERANCE = 1e-7
FORCE_STATISTIC_ABSOLUTE_TOLERANCE = 1e-12


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"Expected a non-empty JSON object in {path}.")
    if not all(isinstance(value, dict) for value in entries.values()):
        raise ValueError(f"Every manifest entry in {path} must be an object.")
    return entries


def _dataset(paths: Sequence[Path], *, name: str):
    from atomworks.ml.datasets import ASELMDBDataset

    if not paths:
        raise FileNotFoundError(f"No ASE-LMDB paths found for {name}.")
    return ASELMDBDataset(
        paths=[path.resolve() for path in paths],
        name=name,
        return_type="record",
        readonly=True,
        readahead=False,
        include_data=True,
    )


def _mapping_value(mapping: Mapping[str, Any], key: str) -> Any:
    value = mapping.get(key)
    if value is not None:
        return value
    nested = mapping.get("data")
    return nested.get(key) if isinstance(nested, Mapping) else None


def _omol_record_resolver(source_root: Path):
    paths = sorted(
        path
        for path in source_root.rglob("*.aselmdb")
        if not path.name.startswith("._")
    )
    dataset = _dataset(paths, name="omol25_mlip_companion_source")

    def resolve(row_id: str, specification: Mapping[str, Any]) -> dict[str, Any]:
        example_id = specification.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"Example {row_id} has no source example_id.")
        source_index = dataset.id_to_idx(example_id)
        metadata = specification["extra"]
        expected_index = int(metadata["source_row_index"])
        if source_index != expected_index:
            raise ValueError(
                f"Example {row_id} resolves to row {source_index}, not "
                f"source_row_index={expected_index}."
            )
        record = dataset[source_index]
        if record["example_id"] != example_id:
            raise ValueError(f"Example {row_id} does not match its source ID.")
        source_data_id = _mapping_value(record["data"], "data_id")
        if source_data_id != metadata.get("source_data_id"):
            raise ValueError(f"Example {row_id} does not match its source data ID.")
        return record

    return resolve


def _omat_record_resolver(source_root: Path):
    datasets: dict[Path, Any] = {}

    def resolve(row_id: str, specification: Mapping[str, Any]) -> dict[str, Any]:
        metadata = specification["extra"]
        source_shard = metadata.get("source_shard")
        if not isinstance(source_shard, str) or not source_shard:
            raise ValueError(f"Example {row_id} has no source_shard.")
        database_path = (source_root / source_shard).resolve()
        try:
            database_path.relative_to(source_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Example {row_id} source_shard escapes the source root."
            ) from exc
        if database_path not in datasets:
            datasets[database_path] = _dataset(
                [database_path],
                name=f"omat24_mlip_companion_{database_path.parent.name}",
            )
        dataset = datasets[database_path]
        source_index = int(metadata["source_row_index"])
        record = dataset[source_index]

        expected_values = {
            "db_index": metadata.get("source_db_index"),
            "db_id": metadata.get("source_db_id"),
            "example_id": metadata.get("source_example_id"),
        }
        for key, expected in expected_values.items():
            if expected is not None and record[key] != expected:
                raise ValueError(
                    f"Example {row_id} source {key} is {record[key]!r}, "
                    f"not {expected!r}."
                )
        source_sid = _mapping_value(record["data"], "sid")
        if source_sid != metadata.get("sid"):
            raise ValueError(f"Example {row_id} does not match its source SID.")
        source_data_id = _mapping_value(record["data"], "calc_id")
        if source_data_id != metadata.get("source_data_id"):
            raise ValueError(f"Example {row_id} does not match its source data ID.")
        return record

    return resolve


def _finite_scalar(value: Any, *, name: str, row_id: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Example {row_id} {name} must be scalar.")
    scalar = float(array.reshape(-1)[0])
    if not math.isfinite(scalar):
        raise ValueError(f"Example {row_id} {name} must be finite.")
    return scalar


def _force_statistics(forces: np.ndarray) -> tuple[float, float, float]:
    force_norms = np.linalg.norm(forces.astype(np.float64, copy=False), axis=-1)
    return (
        float(force_norms.mean()),
        float(np.sqrt(np.mean(np.square(force_norms)))),
        float(np.quantile(force_norms, 0.95)),
    )


def _validate_force_statistics(
    row_id: str,
    specification: Mapping[str, Any],
    forces: np.ndarray,
) -> None:
    actual_statistics = _force_statistics(forces)
    for key, actual in zip(
        ("mean_force_norm", "force_rms", "force_p95"),
        actual_statistics,
        strict=True,
    ):
        expected = _finite_scalar(specification.get(key), name=key, row_id=row_id)
        if not math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=FORCE_STATISTIC_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError(
                f"Example {row_id} {key}={expected!r} does not match the "
                f"source-force value {actual!r}."
            )


def _validate_cif(
    row_id: str,
    specification: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    examples_dir: Path,
    domain: str,
) -> None:
    input_path = Path(specification["input"])
    cif_path = input_path if input_path.is_absolute() else examples_dir / input_path
    if not cif_path.is_file():
        raise FileNotFoundError(f"Example {row_id} CIF does not exist: {cif_path}")
    cif_file = CIFFile.read(str(cif_path))
    cif_atoms = get_structure(cif_file)[0]
    source_atoms = record["atoms"]
    if len(cif_atoms) != len(source_atoms):
        raise ValueError(f"Example {row_id} CIF and source atom counts differ.")
    metadata = specification["extra"]
    atom_count_key = "num_atoms_with_hydrogens" if domain == "omol25" else "num_atoms"
    if int(metadata[atom_count_key]) != len(source_atoms):
        raise ValueError(
            f"Example {row_id} {atom_count_key} does not match its source atoms."
        )
    hydrogen_count = sum(
        symbol == "H" for symbol in source_atoms.get_chemical_symbols()
    )
    if int(metadata["num_hydrogens"]) != hydrogen_count:
        raise ValueError(
            f"Example {row_id} num_hydrogens does not match its source atoms."
        )
    cif_elements = np.char.upper(np.asarray(cif_atoms.element, dtype=str))
    source_elements = np.char.upper(
        np.asarray(source_atoms.get_chemical_symbols(), dtype=str)
    )
    if not np.array_equal(cif_elements, source_elements):
        raise ValueError(f"Example {row_id} CIF and source atom orders differ.")
    if not np.allclose(
        np.asarray(cif_atoms.coord),
        np.asarray(source_atoms.positions),
        rtol=0.0,
        atol=COORDINATE_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError(f"Example {row_id} CIF and source coordinates differ.")
    if domain == "omat24":
        block = cif_file[next(iter(cif_file))]
        if "cell" not in block:
            raise ValueError(f"Example {row_id} material CIF has no cell category.")
        cell = block["cell"]
        cell_parameter_names = (
            "length_a",
            "length_b",
            "length_c",
            "angle_alpha",
            "angle_beta",
            "angle_gamma",
        )
        cif_cell_parameters = np.asarray(
            [cell[name].as_array(float)[0] for name in cell_parameter_names]
        )
        source_cell_parameters = cell_to_cellpar(np.asarray(source_atoms.cell.array))
        if not np.allclose(
            cif_cell_parameters,
            source_cell_parameters,
            rtol=0.0,
            atol=CELL_PARAMETER_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError(f"Example {row_id} CIF and source cell parameters differ.")

        if "atom_site" not in block:
            raise ValueError(
                f"Example {row_id} material CIF has no atom_site category."
            )
        atom_site = block["atom_site"]
        fractional_names = ("fract_x", "fract_y", "fract_z")
        if not all(name in atom_site for name in fractional_names):
            raise ValueError(
                f"Example {row_id} material CIF has no fractional coordinates."
            )
        cif_fractional = np.stack(
            [atom_site[name].as_array(float) for name in fractional_names],
            axis=-1,
        )
        source_fractional = np.asarray(source_atoms.get_scaled_positions(wrap=True))
        if (
            cif_fractional.shape != source_fractional.shape
            or not np.isfinite(cif_fractional).all()
        ):
            raise ValueError(
                f"Example {row_id} CIF fractional coordinates are malformed."
            )
        # Fractional coordinates are periodic, so values differing by an integer
        # cell translation identify the same source position.
        periodic_difference = cif_fractional - source_fractional
        periodic_difference -= np.round(periodic_difference)
        if not np.allclose(
            periodic_difference,
            0.0,
            rtol=0.0,
            atol=FRACTIONAL_COORDINATE_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError(
                f"Example {row_id} CIF and source fractional coordinates differ."
            )


def build_mlip_companion(
    structure_entries: Mapping[str, dict[str, Any]],
    *,
    source_root: Path,
    examples_dir: Path,
    domain: str,
) -> dict[str, dict[str, Any]]:
    """Return an MLIP task view of one structure-prediction cohort."""
    if domain == "omol25":
        resolve_record = _omol_record_resolver(source_root)
    elif domain == "omat24":
        resolve_record = _omat_record_resolver(source_root)
    else:
        raise ValueError(f"Unsupported domain {domain!r}.")

    companion_entries = {}
    for row_id, specification in structure_entries.items():
        metadata = specification.get("extra")
        if not isinstance(metadata, dict):
            raise ValueError(f"Example {row_id} has no provenance metadata.")
        record = resolve_record(row_id, specification)
        _validate_cif(
            row_id,
            specification,
            record,
            examples_dir=examples_dir,
            domain=domain,
        )

        calculator_results = record.get("calculator_results")
        if not isinstance(calculator_results, Mapping):
            raise ValueError(f"Example {row_id} has no calculator results.")
        energy = _finite_scalar(
            calculator_results.get("energy"),
            name="energy",
            row_id=row_id,
        )
        forces = np.asarray(calculator_results.get("forces"), dtype=np.float64)
        expected_force_shape = (len(record["atoms"]), 3)
        if forces.shape != expected_force_shape or not np.isfinite(forces).all():
            raise ValueError(
                f"Example {row_id} forces must be finite with shape "
                f"{expected_force_shape}, got {forces.shape}."
            )
        _validate_force_statistics(row_id, specification, forces)

        companion_metadata = dict(metadata)
        if domain == "omol25":
            charge_value = _finite_scalar(
                _mapping_value(record["data"], "charge"),
                name="total charge",
                row_id=row_id,
            )
            spin_value = _finite_scalar(
                _mapping_value(record["data"], "spin"),
                name="spin multiplicity",
                row_id=row_id,
            )
            if not charge_value.is_integer() or not spin_value.is_integer():
                raise ValueError(
                    f"Example {row_id} charge and spin must be integer-valued."
                )
            charge = int(charge_value)
            spin = int(spin_value)
            if charge != specification.get("total_charge"):
                raise ValueError(f"Example {row_id} does not match its source charge.")
            if spin < 1 or spin != specification.get("spin_multiplicity"):
                raise ValueError(f"Example {row_id} does not match its source spin.")
            companion_metadata["source_example_id"] = specification["example_id"]
        companion_metadata["atomistic_domain"] = domain
        companion_metadata["atomistic_property_task"] = "mlip"
        if domain == "omol25":
            companion_metadata["total_charge"] = charge
            companion_metadata["spin_multiplicity"] = spin
        companion_metadata["energy_gt_eV"] = energy
        companion_metadata["forces_gt_eV_per_angstrom"] = forces.tolist()

        companion_entries[row_id] = {
            "example_id": specification["example_id"],
            "input": specification["input"],
            "contig": specification["contig"],
            "select_fixed_atoms": True,
            "select_unfixed_sequence": False,
            "extra": companion_metadata,
        }
    return companion_entries


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure_json", type=Path)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--domain", choices=("omol25", "omat24"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Build and write a validated MLIP companion manifest."""
    args = _build_parser().parse_args(argv)
    structure_json = args.structure_json.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output == structure_json:
        raise ValueError(
            "The MLIP companion must not overwrite its structure manifest."
        )
    companion_entries = build_mlip_companion(
        _read_manifest(structure_json),
        source_root=source_root,
        examples_dir=structure_json.parent,
        domain=args.domain,
    )
    output.write_text(
        json.dumps(companion_entries, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
