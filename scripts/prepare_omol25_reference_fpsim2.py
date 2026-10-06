#!/usr/bin/env python3
"""Prepare a FPSim2 reference database from OMol25 SMILES."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from FPSim2.io import create_db_file
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


def _default_valid_smiles_path() -> Path:
    """Return the repo-local OMol25 GEOM train valid SMILES file."""
    repo_root = Path(__file__).resolve().parents[1]
    return (
        repo_root
        / "data"
        / "omol25"
        / "subsets"
        / "train_4M_geom_orca6_train_smiles.txt"
    )


def _default_output_path(input_path: Path, output: Path | None) -> Path:
    """Derive a stable FPSim2 output path when the caller did not supply one."""
    if output is not None:
        return output
    return input_path.with_name(f"{input_path.stem}_fpsim2.h5")


def _default_fp_params() -> dict[str, int]:
    """Return the Morgan fingerprint parameters used for molecule novelty."""
    return {"radius": 2, "fpSize": 2048}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a FPSim2 HDF5 fingerprint database from SMILES."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=_default_valid_smiles_path(),
        help=(
            "Input SMILES file. Defaults to the repo-local "
            "OMol25 GEOM train valid SMILES file."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output FPSim2 .h5 file. Defaults to '<input-stem>_fpsim2.h5'.",
    )
    parser.add_argument(
        "--fp-type",
        default="Morgan",
        help="FPSim2 fingerprint type. Defaults to Morgan.",
    )
    parser.add_argument(
        "--fp-params",
        type=json.loads,
        default=None,
        help=(
            "Fingerprint parameters as JSON. Defaults to "
            f"{json.dumps(_default_fp_params())}."
        ),
    )
    parser.add_argument(
        "--full-sanitization",
        action="store_true",
        help="Use full RDKit sanitization instead of the default partial sanitization.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(
            chunk.count(b"\n") for chunk in iter(lambda: handle.read(65536), b"")
        )


def _iter_smiles_with_ids(path: Path, total_lines: int) -> Iterator[tuple[str, int]]:
    mol_id = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, total=total_lines, desc="Reading SMILES"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            smiles = stripped.split(maxsplit=1)[0]
            mol_id += 1
            yield smiles, mol_id


def _patch_pytables_cpu_info_probe() -> None:
    """Fall back when PyTables CPU detection fails in managed cluster shells."""
    try:
        import tables.leaf as tables_leaf
    except ImportError:
        return

    try:
        tables_leaf.get_cpu_info()
    except Exception as exc:
        LOGGER.warning(
            "PyTables CPU-info probe failed; falling back to default chunk sizing: %s",
            exc,
        )
        tables_leaf.get_cpu_info = lambda: {}


def prepare_fpsim2_database(
    input_path: Path,
    output_path: Path,
    *,
    fp_type: str = "Morgan",
    fp_params: dict[str, Any] | None = None,
    full_sanitization: bool = False,
    overwrite: bool = False,
) -> Path:
    """Create a FPSim2 HDF5 fingerprint database from a SMILES file."""
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input SMILES file does not exist: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            "Pass --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_lines = _count_lines(input_path)
    params = _default_fp_params()
    if fp_params is not None:
        params.update(fp_params)

    LOGGER.info("Writing FPSim2 database to %s", output_path)
    LOGGER.info("Fingerprint type: %s", fp_type)
    LOGGER.info("Fingerprint params: %s", params)
    _patch_pytables_cpu_info_probe()
    create_db_file(
        _iter_smiles_with_ids(input_path, total_lines),
        str(output_path),
        mol_format="smiles",
        fp_type=fp_type,
        fp_params=params,
        sort_by_popcnt=True,
        full_sanitization=full_sanitization,
    )
    LOGGER.info("Finished writing %s", output_path)
    return output_path


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    prepare_fpsim2_database(
        args.input,
        _default_output_path(args.input, args.output),
        fp_type=args.fp_type,
        fp_params=args.fp_params,
        full_sanitization=args.full_sanitization,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
