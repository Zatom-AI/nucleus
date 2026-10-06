#!/usr/bin/env python3
"""Convert Zatom-1 QM9 or MP20 artifacts into AtomWorks ASE-LMDB shards.

The input directory is the dataset root produced by Zatom-1's
``zatom/data/download_data.py``.  The output is a directory of ``*.aselmdb``
files that can be consumed directly by ``atomworks.ml.datasets.ASELMDBDataset``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from ase import Atoms

DatasetName = Literal["qm9", "mp20"]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ASERecord:
    """One ASE database record and its serializable metadata."""

    atoms: Atoms
    key_value_pairs: dict[str, str | int | float | bool]
    data: dict[str, Any]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Zatom-1 QM9 or MP20 dataset root into AtomWorks-compatible "
            "ASE-LMDB shards."
        )
    )
    parser.add_argument("dataset", choices=("qm9", "mp20"))
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help=(
            "Dataset root produced by Zatom-1's download_data.py, containing "
            "data_v3.pt (QM9) or mp20.pt (MP20), either directly or under a "
            "processed/ subdirectory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory in which to write data####.aselmdb shards and manifest.json.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        required=True,
        help=(
            "Number of records to retain. QM9 is deterministically shuffled before "
            "selection; MP20 retains source order."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PyTorch permutation seed used for QM9 (default: 42).",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10_000,
        help="Maximum records per output shard (default: 10000).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity.",
    )
    args = parser.parse_args(argv)

    if args.num_examples <= 0:
        parser.error("--num-examples must be positive")
    if args.shard_size <= 0:
        parser.error("--shard-size must be positive")
    return args


def _as_numpy(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    """Convert a tensor-like value to a CPU NumPy array."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _as_scalar(value: Any) -> Any:
    """Unwrap common one-element containers used by PyG data objects."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"Expected one value, got tensor shape {tuple(value.shape)}."
            )
        return value.detach().cpu().item()
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"Expected one value, got array shape {value.shape}.")
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected one value, got {len(value)} values.")
        return _as_scalar(value[0])
    return value


def _optional_string(value: Any) -> str | None:
    """Return a stable string for optional scalar-like metadata."""
    if value is None:
        return None
    return str(_as_scalar(value))


def qm9_record(example: Any, source_index: int) -> ASERecord:
    """Convert one native PyG QM9 example to an ASE database record."""
    atomic_numbers = _as_numpy(example.z, dtype=np.int64).reshape(-1)
    positions = _as_numpy(example.pos, dtype=np.float64)
    if positions.shape != (len(atomic_numbers), 3):
        raise ValueError(
            f"QM9 source index {source_index} has positions shape {positions.shape}; "
            f"expected ({len(atomic_numbers)}, 3)."
        )

    source_id = _optional_string(getattr(example, "name", None))
    if source_id is None:
        source_id = str(source_index)
    record_data: dict[str, Any] = {}
    smiles = _optional_string(getattr(example, "smiles", None))
    if smiles is not None:
        record_data["smiles"] = smiles

    return ASERecord(
        atoms=Atoms(numbers=atomic_numbers, positions=positions, pbc=False),
        key_value_pairs={
            "dataset": "qm9",
            "source_index": int(source_index),
            "source_id": source_id,
        },
        data=record_data,
    )


def mp20_record(example: Any, source_index: int) -> ASERecord:
    """Convert one Zatom-1 PyG MP20 example to an ASE database record."""
    atomic_numbers = _as_numpy(example.atom_types, dtype=np.int64).reshape(-1)
    cell = _as_numpy(example.cell, dtype=np.float64).reshape(3, 3)
    fractional_coordinates = _as_numpy(example.frac_coords, dtype=np.float64).reshape(
        -1, 3
    )
    if fractional_coordinates.shape != (len(atomic_numbers), 3):
        raise ValueError(
            f"MP20 source index {source_index} has fractional-coordinate shape "
            f"{fractional_coordinates.shape}; expected ({len(atomic_numbers)}, 3)."
        )
    if not np.isfinite(cell).all() or abs(float(np.linalg.det(cell))) <= 1e-8:
        raise ValueError(f"MP20 source index {source_index} has an invalid unit cell.")

    source_id = _optional_string(getattr(example, "id", None))
    if source_id is None:
        source_id = str(source_index)
    space_group = int(_as_scalar(getattr(example, "spacegroup", 1)))

    return ASERecord(
        atoms=Atoms(
            numbers=atomic_numbers,
            scaled_positions=fractional_coordinates,
            cell=cell,
            pbc=True,
        ),
        key_value_pairs={
            "dataset": "mp20",
            "source_index": int(source_index),
            "source_id": source_id,
            "space_group": space_group,
        },
        data={},
    )


def select_indices(
    dataset: DatasetName,
    *,
    dataset_size: int,
    num_examples: int,
    seed: int,
) -> list[int]:
    """Select deterministic source indices for one benchmark training subset."""
    if num_examples > dataset_size:
        raise ValueError(
            f"Requested {num_examples:,} {dataset.upper()} examples, but the source "
            f"contains only {dataset_size:,}."
        )

    if dataset == "mp20":
        return list(range(num_examples))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(dataset_size, generator=generator)[:num_examples].tolist()


def _load_serialized_pyg_dataset(processed_path: Path) -> Any:
    """Load a serialized PyG dataset without importing Zatom-1."""
    from torch_geometric.data import InMemoryDataset

    class SerializedPyGDataset(InMemoryDataset):
        def __init__(self, path: Path) -> None:
            super().__init__(root=None)
            self.load(path)

    return SerializedPyGDataset(processed_path)


def _load_qm9(source_root: Path) -> Any:
    """Load an already-downloaded native PyG QM9 dataset."""
    return _load_serialized_pyg_dataset(_source_artifact_path("qm9", source_root))


def _load_mp20(source_root: Path) -> Any:
    """Load Zatom-1's serialized PyG MP20 dataset without importing Zatom-1."""
    return _load_serialized_pyg_dataset(_source_artifact_path("mp20", source_root))


def _batched(indices: Sequence[int], batch_size: int) -> Iterator[Sequence[int]]:
    """Yield consecutive batches from an index sequence."""
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def _selected_indices_sha256(indices: Sequence[int]) -> str:
    """Hash selected source indices using a stable little-endian representation."""
    encoded = np.asarray(indices, dtype="<i8").tobytes()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    """Stream a file into a SHA-256 digest without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_artifact_path(dataset_name: DatasetName, source_root: Path) -> Path:
    """Return the canonical processed artifact path for one source dataset."""
    filename = "data_v3.pt" if dataset_name == "qm9" else "mp20.pt"
    candidates = (source_root / filename, source_root / "processed" / filename)
    for path in candidates:
        if path.is_file():
            return path
    formatted_candidates = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Expected {formatted_candidates}. Download and extract the official "
        f"Zatom-1 {dataset_name.upper()} processed artifact first."
    )


def _ensure_new_output(output_dir: Path) -> None:
    """Refuse to mix a new conversion with existing output artifacts."""
    existing_shards = (
        sorted(output_dir.glob("*.aselmdb")) if output_dir.exists() else []
    )
    manifest_path = output_dir / "manifest.json"
    partial_paths = sorted(output_dir.glob("*.partial")) if output_dir.exists() else []
    existing_paths = [*existing_shards, *partial_paths]
    if manifest_path.exists():
        existing_paths.append(manifest_path)
    if existing_paths:
        formatted_paths = ", ".join(str(path) for path in existing_paths[:5])
        raise FileExistsError(
            "Output directory already contains conversion artifacts; choose a new "
            f"--output-dir or inspect and remove them first: {formatted_paths}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def write_shards(
    *,
    dataset: Any,
    dataset_name: DatasetName,
    indices: Sequence[int],
    output_dir: Path,
    shard_size: int,
    converter: Callable[[Any, int], ASERecord],
    seed: int,
    source_root: Path,
) -> list[Path]:
    """Write selected examples to new ASE-LMDB shards and a provenance manifest."""
    import ase_db_backends.aselmdb  # noqa: F401
    from ase.db import connect

    _ensure_new_output(output_dir)
    source_artifact_path = _source_artifact_path(dataset_name, source_root)
    source_artifact_sha256 = _file_sha256(source_artifact_path)
    shard_paths: list[Path] = []
    for shard_index, shard_indices in enumerate(_batched(indices, shard_size)):
        shard_path = output_dir / f"data{shard_index:04d}.aselmdb"
        partial_path = output_dir / f".{shard_path.name}.partial"
        LOGGER.info(
            "Writing %s records %d-%d to %s",
            dataset_name.upper(),
            shard_index * shard_size,
            shard_index * shard_size + len(shard_indices) - 1,
            shard_path,
        )
        with connect(
            partial_path,
            type="aselmdb",
            use_lock_file=False,
            serial=True,
        ) as database:
            database.metadata = {
                "dataset": dataset_name,
                "source_root": str(source_root.resolve()),
                "source_artifact": str(source_artifact_path),
                "source_artifact_sha256": source_artifact_sha256,
                "selection": "seeded_torch_permutation"
                if dataset_name == "qm9"
                else "source_order_prefix",
                "selection_seed": seed if dataset_name == "qm9" else None,
                "shard_index": shard_index,
                "num_records": len(shard_indices),
            }
            for source_index in shard_indices:
                record = converter(dataset[source_index], source_index)
                database.write(
                    record.atoms,
                    data=record.data,
                    **record.key_value_pairs,
                )
        partial_path.rename(shard_path)
        shard_paths.append(shard_path)

    manifest = {
        "schema_version": 1,
        "dataset": dataset_name,
        "source_root": str(source_root.resolve()),
        "source_artifact": str(source_artifact_path),
        "source_artifact_sha256": source_artifact_sha256,
        "source_num_examples": len(dataset),
        "num_examples": len(indices),
        "selection": "seeded_torch_permutation"
        if dataset_name == "qm9"
        else "source_order_prefix",
        "selection_seed": seed if dataset_name == "qm9" else None,
        "selected_indices_sha256": _selected_indices_sha256(indices),
        "shard_size": shard_size,
        "shards": [path.name for path in shard_paths],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return shard_paths


def _verify_output(shard_paths: Sequence[Path], expected_size: int) -> None:
    """Read the result through AtomWorks and validate its cardinality/endpoints."""
    from atomworks.ml.datasets import ASELMDBDataset

    with ASELMDBDataset(
        paths=list(shard_paths),
        name="qm9_mp20_conversion_verification",
        return_type="atoms",
        readonly=True,
        readahead=False,
    ) as dataset:
        if len(dataset) != expected_size:
            raise RuntimeError(
                f"Wrote {len(dataset):,} readable records; expected {expected_size:,}."
            )
        if expected_size:
            dataset[0]
            dataset[-1]


def main(argv: Sequence[str] | None = None) -> None:
    """Run the requested conversion."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    source_root = args.source_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if args.dataset == "qm9":
        dataset = _load_qm9(source_root)
        converter = qm9_record
    else:
        dataset = _load_mp20(source_root)
        converter = mp20_record

    indices = select_indices(
        args.dataset,
        dataset_size=len(dataset),
        num_examples=args.num_examples,
        seed=args.seed,
    )
    shard_paths = write_shards(
        dataset=dataset,
        dataset_name=args.dataset,
        indices=indices,
        output_dir=output_dir,
        shard_size=args.shard_size,
        converter=converter,
        seed=args.seed,
        source_root=source_root,
    )
    _verify_output(shard_paths, expected_size=len(indices))
    LOGGER.info(
        "Prepared %d %s records in %d ASE-LMDB shards under %s",
        len(indices),
        args.dataset.upper(),
        len(shard_paths),
        output_dir,
    )


if __name__ == "__main__":
    main()
