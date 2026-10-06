#!/usr/bin/env python3
"""Precompute and cache OMat24 material-structure fingerprints.

This script mirrors ``scripts/prepare_omol25_reference_smiles.py``:
it loads the OMat24 ASE-LMDB training dataset, fans out across multiple
worker processes, computes a normalized CrystalNN fingerprint for each valid
material structure, and stores the result as a ``.npy`` cache file that can be
loaded by ``MaterialGenerationMetrics`` at validation time.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import rootutils
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from atomworks.ml.datasets import ASELMDBDataset
from atomworks.ml.datasets.loaders import create_ase_materials_loader
from dotenv import load_dotenv
from rich.progress import track
from torch.utils.data import Subset


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
load_dotenv(override=True)

from nucleus.utils.material_fingerprints import (
    create_material_fingerprint_featurizer,
    record_to_structure,
    structure_fingerprint,
)

LOGGER = logging.getLogger(__name__)
_WORKER_DATASET: Any | None = None
_WORKER_FEATURIZER: Any | None = None


def _default_omat24_data_dir() -> Path:
    """Mirror the repo's OMat24 default path resolution."""
    env_value = os.environ.get("OMAT24_DATA_DIR")
    if env_value:
        return Path(env_value).expanduser()

    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "data" / "omat24" / "raw" / "train"


def _omat24_shard_paths(input_dir: Path) -> list[Path]:
    """Collect real OMat24 LMDB shards while ignoring macOS AppleDouble sidecars."""
    return sorted(
        path
        for path in input_dir.rglob("*.aselmdb")
        if not path.name.startswith("._")
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a .npy fingerprint cache for OMat24 material generation "
            "evaluation from ASE-LMDB shards."
        )
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=_default_omat24_data_dir(),
        help=(
            "Directory containing OMat24 *.aselmdb shards. Defaults to "
            "OMAT24_DATA_DIR or ./data/omat24/raw/train."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output .npy file to write.",
    )
    parser.add_argument(
        "--subset-indices-path",
        type=Path,
        default=None,
        help=(
            "Optional text file with zero-based dataset indices, one per line. "
            "Use this to fingerprint a known OMat24 subset."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Number of worker processes to use. Set to 0 for single-process "
            "execution."
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
    """Read zero-based dataset indices from a text file."""
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


def _build_omat24_dataset(
    shard_paths: list[Path], subset_indices: list[int] | None
) -> Any:
    """Construct the OMat24 dataset view used by both the main process and workers."""
    dataset: Any = ASELMDBDataset(
        paths=shard_paths,
        name="omat24_reference_fingerprints",
        return_type="record",
        readonly=True,
        readahead=False,
        include_data=True,
        loader=create_ase_materials_loader(
            res_name="LIG",
            wrap_fractional_coordinates=True,
            include_atom_array=True,
            keep_ase_atoms=True,
            include_chain_info=True,
        ),
    )

    if subset_indices is not None:
        dataset = _subset_ase_lmdb_dataset(dataset, subset_indices)

    return dataset


def _initialize_worker_dataset(
    shard_paths: list[Path], subset_indices: list[int] | None
) -> None:
    """Initialize the dataset once per worker process."""
    global _WORKER_DATASET, _WORKER_FEATURIZER
    _WORKER_DATASET = _build_omat24_dataset(shard_paths, subset_indices)
    _WORKER_FEATURIZER = create_material_fingerprint_featurizer()


def _parallel_chunksize(total_items: int, num_workers: int) -> int:
    """Choose a modest chunk size for process-pool mapping."""
    if total_items <= 0:
        return 1
    # Keep chunks small enough that progress feedback starts quickly even when
    # individual fingerprint computations are slow.
    return max(1, min(16, total_items // max(1, num_workers * 8) or 1))


def _collect_fingerprint_for_index(index: int) -> np.ndarray | None:
    """Collect a fingerprint for a dataset row inside a worker process."""
    if _WORKER_DATASET is None or _WORKER_FEATURIZER is None:
        raise RuntimeError("Worker dataset has not been initialized.")

    record = _WORKER_DATASET[index]
    example_id = str(record.get("example_id", index))
    atom_array = record.get("atom_array")
    if atom_array is None:
        raise ValueError(
            f"{example_id}: dataset record does not contain an `atom_array` entry."
        )

    structure = record_to_structure(record)
    if structure is None:
        return None
    return structure_fingerprint(structure, _WORKER_FEATURIZER)


def _iter_fingerprints(
    dataset: Any,
    shard_paths: list[Path],
    subset_indices: list[int] | None,
    num_workers: int,
) -> Iterator[np.ndarray | None]:
    """Yield fingerprints for a dataset, optionally in parallel."""
    if len(dataset) == 0:
        return

    if num_workers <= 0:
        featurizer = create_material_fingerprint_featurizer()
        for index in range(len(dataset)):
            record = dataset[index]
            example_id = str(record.get("example_id", index))
            atom_array = record.get("atom_array")
            if atom_array is None:
                raise ValueError(
                    f"{example_id}: dataset record does not contain an `atom_array` entry."
                )

            structure = record_to_structure(record)
            yield structure_fingerprint(structure, featurizer) if structure else None
        return

    worker_count = min(max(1, num_workers), len(dataset))
    chunksize = _parallel_chunksize(len(dataset), worker_count)
    LOGGER.info(
        "Collecting fingerprints with %d worker processes (chunksize=%d)",
        worker_count,
        chunksize,
    )
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=ctx,
        initializer=_initialize_worker_dataset,
        initargs=(shard_paths, subset_indices),
    ) as executor:
        yield from executor.map(
            _collect_fingerprint_for_index,
            range(len(dataset)),
            chunksize=chunksize,
        )


def _collect_valid_fingerprints(
    dataset: Any,
    shard_paths: list[Path],
    subset_indices: list[int] | None,
    num_workers: int,
) -> tuple[list[np.ndarray], int]:
    """Collect valid fingerprints and infer their dimensionality."""
    fingerprints: list[np.ndarray] = []
    fingerprint_dim: int | None = None
    processed = 0

    for fingerprint in track(
        _iter_fingerprints(dataset, shard_paths, subset_indices, num_workers),
        total=len(dataset),
        description="Scanning OMat24 fingerprints",
    ):
        processed += 1
        if fingerprint is None:
            if processed == 1 or processed % 1000 == 0:
                LOGGER.info(
                    "Scanned %d/%d examples (%d valid fingerprints)",
                    processed,
                    len(dataset),
                    len(fingerprints),
                )
            continue

        if fingerprint_dim is None:
            fingerprint_dim = int(fingerprint.shape[0])
        elif fingerprint.shape[0] != fingerprint_dim:
            raise ValueError(
                "Encountered inconsistent fingerprint dimensions while "
                f"processing dataset: expected {fingerprint_dim}, got {fingerprint.shape[0]}"
            )
        fingerprints.append(fingerprint)

        if processed == 1 or processed % 1000 == 0:
            LOGGER.info(
                "Scanned %d/%d examples (%d valid fingerprints)",
                processed,
                len(dataset),
                len(fingerprints),
            )

    if fingerprint_dim is None:
        return [], 0

    return fingerprints, fingerprint_dim


def _write_cache(
    dataset: Any,
    shard_paths: list[Path],
    subset_indices: list[int] | None,
    output_path: Path,
    num_workers: int,
) -> tuple[int, int]:
    """Compute fingerprints and write them to a memory-mapped ``.npy`` file."""
    fingerprints, fingerprint_dim = _collect_valid_fingerprints(
        dataset,
        shard_paths,
        subset_indices,
        num_workers,
    )
    valid_count = len(fingerprints)
    if valid_count == 0 or fingerprint_dim == 0:
        raise ValueError("No valid fingerprints could be computed from the dataset")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(valid_count, fingerprint_dim),
    )

    for write_idx, fingerprint in enumerate(
        track(
            fingerprints,
            total=valid_count,
            description="Writing OMat24 fingerprints",
        )
    ):
        cache[write_idx] = fingerprint.astype(np.float32, copy=False)

    cache.flush()
    return valid_count, fingerprint_dim


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    input_dir = args.input_dir.expanduser()
    if not input_dir.exists():
        raise FileNotFoundError(f"OMat24 input directory does not exist: {input_dir}")

    shard_paths = _omat24_shard_paths(input_dir)
    if not shard_paths:
        raise FileNotFoundError(
            f"No OMat24 *.aselmdb shards were found under {input_dir}. "
            "Make sure the training split is mounted locally."
        )

    LOGGER.info("Found %d OMat24 shards under %s", len(shard_paths), input_dir)

    subset_indices_path = args.subset_indices_path
    if subset_indices_path is not None:
        subset_indices_path = subset_indices_path.expanduser()
        if not subset_indices_path.exists():
            raise FileNotFoundError(
                f"Subset indices file does not exist: {subset_indices_path}"
            )
        indices = _read_indices(subset_indices_path)
        LOGGER.info(
            "Filtering OMat24 dataset to %d explicit indices from %s",
            len(indices),
            subset_indices_path,
        )
    else:
        indices = None

    dataset = _build_omat24_dataset(shard_paths, indices)
    dataset_length = len(dataset)

    output_path = args.output.expanduser()
    if output_path.suffix.lower() != ".npy":
        raise ValueError("Output path must end with .npy")
    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    LOGGER.info("Processing %d examples", dataset_length)
    n_fingerprints, fingerprint_dim = _write_cache(
        dataset,
        shard_paths,
        indices,
        output_path,
        num_workers=args.num_workers,
    )
    LOGGER.info(
        "Wrote %d fingerprints with dimension %d to %s",
        n_fingerprints,
        fingerprint_dim,
        output_path,
    )


if __name__ == "__main__":
    main()
