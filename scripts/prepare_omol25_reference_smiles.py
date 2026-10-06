#!/usr/bin/env python3
"""Prepare a reference SMILES file from OMol25 ASE-LMDB shards.

This script is intended to build the `reference_smiles_path` input used by the
molecule-generation metrics/evaluation pipeline.

It supports:
  - the full OMol25 4M training dataset
  - the GEOM-Drugs partition of the OMol25 4M training dataset via index files

The conversion path writes SMILES only for rows where RDKit can derive a valid
molecule from the atom coordinates. Rows that fail molecule perception are
skipped instead of being represented by disconnected atom lists.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import rootutils
from atomworks.ml.datasets import ASELMDBDataset
from atomworks.ml.datasets.loaders import create_ase_atoms_loader
from dotenv import load_dotenv
from rich.progress import track
from torch.utils.data import Subset

from nucleus.utils.atom_array import (
    _rdkit_smiles_from_atom_array,
    _smiles_from_rdkit_mol,
)


# Setup root dir and environment variables (more info: https://github.com/ashleve/rootutils)
# NOTE: Sets the `PROJECT_ROOT` environment variable to the root directory of the project (where `.project-root` is located)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

load_dotenv(override=True)


LOGGER = logging.getLogger(__name__)
_WORKER_DATASET: Any | None = None


def _default_omol25_data_dir() -> Path:
    """Mirror the repo's OMol25 default path resolution."""
    env_value = os.environ.get("OMOL25_DATA_DIR")
    if env_value:
        return Path(env_value).expanduser()

    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "data" / "omol25" / "raw" / "train_4M"


def _default_geom_indices_path() -> Path:
    """Return the repo-local GEOM-Drugs training subset index file."""
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "data" / "omol25" / "subsets" / "train_4M_geom_orca6_train_indices.txt"


def _omol25_shard_paths(input_dir: Path) -> list[Path]:
    """Collect real OMol25 LMDB shards while ignoring macOS AppleDouble sidecars."""
    return sorted(
        path
        for path in input_dir.rglob("*.aselmdb")
        if not path.name.startswith("._")
    )


def _default_output_path(
    input_dir: Path,
    subset_indices_path: Path | None,
    output: Path | None,
) -> Path:
    """Derive a stable output path when the caller did not supply one."""
    if output is not None:
        return output

    if subset_indices_path is not None:
        stem = subset_indices_path.stem
        if stem.endswith("_indices"):
            stem = stem[: -len("_indices")]
        return subset_indices_path.with_name(f"{stem}_reference_smiles.smi")

    return input_dir.with_name(f"{input_dir.name}_reference_smiles.smi")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a .smi file for molecule-generation evaluation from OMol25 "
            "ASE-LMDB shards."
        )
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=_default_omol25_data_dir(),
        help=(
            "Directory containing OMol25 *.aselmdb shards. Defaults to "
            "OMOL25_DATA_DIR or ./data/omol25/raw/train_4M."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=False,
        help="Output .smi file to write.",
    )
    parser.add_argument(
        "--partition",
        choices=("all", "geom"),
        default="all",
        help=(
            "Which OMol25 partition to export. Use `geom` for the GEOM-Drugs "
            "subset; it defaults to the repo-local GEOM train index file."
        ),
    )
    parser.add_argument(
        "--subset-indices-path",
        type=Path,
        default=None,
        help=(
            "Optional text file with zero-based dataset indices, one per line. "
            "When `--partition geom` is selected, this defaults to "
            "data/omol25/subsets/train_4M_geom_orca6_train_indices.txt."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Number of worker processes to use for SMILES collection. "
            "Set to 0 to disable parallelism."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def _read_indices(indices_path: Path) -> list[int]:
    """Read zero-based dataset indices from a text file.

    Blank lines and full-line or inline `#` comments are ignored.
    """
    indices: list[int] = []
    for raw_line in indices_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line:
            continue
        indices.append(int(line))

    if not indices:
        raise ValueError(f"No indices were found in {indices_path}.")

    return indices


def _subset_ase_lmdb_dataset(dataset: ASELMDBDataset, indices: list[int]) -> Subset:
    """Create a subset view over an ASE-LMDB dataset using explicit indices."""
    if not indices:
        raise ValueError("Expected at least one dataset index.")
    return Subset(dataset, indices)


def _build_omol25_dataset(shard_paths: list[Path], subset_indices: list[int] | None) -> Any:
    """Construct the OMol25 dataset view used by both the main process and workers."""
    dataset: Any = ASELMDBDataset(
        paths=shard_paths,
        name="omol25_reference_smiles",
        return_type="record",
        readonly=True,
        readahead=False,
        include_data=True,
        loader=create_ase_atoms_loader(
            res_name="LIG",
            keep_ase_atoms=True,
            include_chain_info=False,
        ),
    )

    if subset_indices is not None:
        dataset = _subset_ase_lmdb_dataset(dataset, subset_indices)

    return dataset


def _initialize_worker_dataset(shard_paths: list[Path], subset_indices: list[int] | None) -> None:
    """Initialize the dataset once per worker process."""
    global _WORKER_DATASET
    _WORKER_DATASET = _build_omol25_dataset(shard_paths, subset_indices)


def _parallel_chunksize(total_items: int, num_workers: int) -> int:
    """Choose a modest chunk size for process-pool mapping."""
    if total_items <= 0:
        return 1
    return max(1, min(256, total_items // max(1, num_workers * 4) or 1))


def _smiles_from_valid_rdkit_mol(atom_array: Any, *, example_id: str) -> str | None:
    """Return SMILES only when RDKit can derive a molecule from an atom array."""
    mol = _rdkit_smiles_from_atom_array(
        atom_array,
        example_id=example_id,
        return_mol_only=True,
    )
    if mol is None:
        return None
    return _smiles_from_rdkit_mol(mol)


def _collect_smiles_for_index(index: int) -> str | None:
    """Collect the final SMILES string for a dataset row inside a worker process."""
    if _WORKER_DATASET is None:
        raise RuntimeError("Worker dataset has not been initialized.")

    record = _WORKER_DATASET[index]
    example_id = str(record.get("example_id", index))
    atom_array = record.get("atom_array")
    if atom_array is None:
        raise ValueError(
            f"{example_id}: dataset record does not contain an `atom_array` entry."
        )

    return _smiles_from_valid_rdkit_mol(atom_array, example_id=example_id)


def _iter_atom_arrays(dataset: Any):
    """Yield (example_id, atom_array) pairs from a dataset-like object."""
    for idx in range(len(dataset)):
        record = dataset[idx]
        example_id = str(record.get("example_id", idx))
        atom_array = record.get("atom_array")
        if atom_array is None:
            raise ValueError(
                f"{example_id}: dataset record does not contain an `atom_array` entry."
            )
        yield example_id, atom_array


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    input_dir = args.input_dir.expanduser()
    if not input_dir.exists():
        raise FileNotFoundError(f"OMol25 input directory does not exist: {input_dir}")

    shard_paths = _omol25_shard_paths(input_dir)
    if not shard_paths:
        raise FileNotFoundError(
            f"No OMol25 *.aselmdb shards were found under {input_dir}. "
            "Make sure the training split is mounted locally."
        )

    LOGGER.info("Found %d OMol25 shards under %s", len(shard_paths), input_dir)

    subset_indices_path = args.subset_indices_path
    if args.partition == "geom" and subset_indices_path is None:
        subset_indices_path = _default_geom_indices_path()
        LOGGER.info("Using default GEOM subset indices file at %s", subset_indices_path)

    if subset_indices_path is not None:
        subset_indices_path = subset_indices_path.expanduser()
        if not subset_indices_path.exists():
            raise FileNotFoundError(
                f"Subset indices file does not exist: {subset_indices_path}"
            )
        indices = _read_indices(subset_indices_path)
        LOGGER.info(
            "Filtering OMol25 dataset to %d explicit indices from %s",
            len(indices),
            subset_indices_path,
        )
    else:
        indices = None

    dataset = _build_omol25_dataset(shard_paths, indices)
    dataset_length = len(dataset)

    output_path = _default_output_path(input_dir, subset_indices_path, args.output)
    args.output = output_path
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    total = dataset_length
    written = 0
    skipped = 0
    if args.num_workers <= 0:
        LOGGER.info("Collecting SMILES with a single worker")
        with args.output.open("w", encoding="utf-8") as handle:
            for example_id, atom_array in track(
                _iter_atom_arrays(dataset),
                total=dataset_length,
                description="Collecting OMol25 SMILES",
            ):
                smiles = _smiles_from_valid_rdkit_mol(
                    atom_array,
                    example_id=example_id,
                )
                if not smiles:
                    skipped += 1
                    LOGGER.debug(
                        "%s: skipping row because RDKit did not derive a molecule",
                        example_id,
                    )
                    continue

                handle.write(smiles + "\n")
                written += 1
    else:
        worker_count = min(max(1, args.num_workers), dataset_length)
        chunksize = _parallel_chunksize(dataset_length, worker_count)
        LOGGER.info(
            "Collecting SMILES with %d worker processes (chunksize=%d)",
            worker_count,
            chunksize,
        )
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=ctx,
            initializer=_initialize_worker_dataset,
            initargs=(shard_paths, indices),
        ) as executor:
            with args.output.open("w", encoding="utf-8") as handle:
                for smiles in track(
                    executor.map(
                        _collect_smiles_for_index,
                        range(dataset_length),
                        chunksize=chunksize,
                    ),
                    total=dataset_length,
                    description="Collecting OMol25 SMILES",
                ):
                    if not smiles:
                        skipped += 1
                        continue
                    handle.write(smiles + "\n")
                    written += 1

    LOGGER.info(
        "Processed %d examples; wrote %d SMILES to %s; skipped %d invalid rows",
        total,
        written,
        args.output,
        skipped,
    )


if __name__ == "__main__":
    main()
