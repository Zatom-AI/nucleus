#!/usr/bin/env python3
"""Build an Open Babel bond sidecar LMDB for OMol25 records.

Rows are selected by their ``data_id`` metadata, or all data IDs can be
selected with ``--data-id all``. Selection can optionally be restricted to a
supplied candidate-index file. Each sidecar key is the UTF-8 encoded
``example_id`` from the source record. Each value is a pickled dictionary
containing zero-based, undirected ``[E, 2]`` bond edges, atom-order validation
metadata, and a boolean indicating whether the canonical RDKit SMILES
represents multiple disconnected fragments.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import pickle
import warnings
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import rootutils
from atomworks.ml.datasets import ASELMDBDataset
from atomworks.ml.datasets.loaders import create_ase_atoms_loader
from biotite.structure import AtomArray
from dotenv import load_dotenv
from rdkit import Chem, rdBase

if not os.environ.get("DEBUG"):
    os.environ["DEBUG"] = "false"

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from nucleus.utils.bond_sidecar import (  # noqa: E402
    ATOMIC_NUMBERS_HASH_ALGORITHM,
    BOND_SIDECAR_METADATA_KEY,
    encode_bond_record,
    openbabel,
    openbabel_bond_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT.parent / "OMol25_train_4M" / "train_4M"
DEFAULT_GEOM_INDICES = (
    REPO_ROOT / "data" / "omol25" / "subsets" / "train_4M_geom_orca6_indices.txt"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "omol25" / "sidecars"
DEFAULT_DATA_ID = "geom_orca6"
ALL_DATA_IDS = "all"
KNOWN_EXPECTED_COUNTS = {DEFAULT_DATA_ID: 199_805}
METADATA_KEY = b"__metadata__"
DISCONNECTED_FRAGMENTS_FIELD = "has_disconnected_fragments"
DISCONNECTED_FRAGMENTS_METHOD = "rdkit_canonical_smiles_dot"
ASE_ATOMS_LOADER = create_ase_atoms_loader(
    res_name="LIG",
    keep_ase_atoms=False,
    include_chain_info=False,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BondTask:
    """Minimal ASE-LMDB record payload sent to an Open Babel worker."""

    entry_id: str
    atom_array: AtomArray


@dataclass(frozen=True)
class BondResult:
    """Result returned from an Open Babel worker."""

    entry_id: str
    value: bytes | None
    has_disconnected_fragments: bool | None
    fragment_count: int | None
    error: str | None


@dataclass
class ScanStats:
    """Mutable counters populated while scanning candidate OMol25 rows."""

    scanned_count: int = 0
    matched_count: int = 0


def _default_input_dir() -> Path:
    load_dotenv(REPO_ROOT / ".env", override=False)
    value = os.environ.get("OMOL25_DATA_DIR")
    return Path(value).expanduser() if value else DEFAULT_INPUT_DIR


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_default_input_dir(),
        help=(
            "Directory containing OMol25 *.aselmdb shards. Defaults to "
            "OMOL25_DATA_DIR or ../OMol25_train_4M/train_4M."
        ),
    )
    parser.add_argument(
        "--data-id",
        default=DEFAULT_DATA_ID,
        help=(
            "OMol25 data_id whose rows should be processed. Use `all` to "
            "process every candidate row regardless of its data_id."
        ),
    )
    parser.add_argument(
        "--subset-indices-path",
        type=Path,
        default=None,
        help=(
            "Optional text file of zero-based candidate dataset indices. The "
            "repo-local GEOM index is used automatically for geom_orca6; other "
            "data IDs scan the full dataset when this option is omitted."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=("Output single-file LMDB path. Defaults to a data_id-specific path."),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, min(16, os.cpu_count() or 1)),
        help="Open Babel worker process count. Use 0 for in-process conversion.",
    )
    parser.add_argument(
        "--commit-interval",
        type=int,
        default=2_000,
        help="Maximum number of new records written per LMDB transaction.",
    )
    parser.add_argument(
        "--map-size-gb",
        type=float,
        default=8.0,
        help="LMDB map size in GiB.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help=(
            "Expected selected row count. Defaults to the known count for "
            "geom_orca6 and 0 (disabled) for other data IDs."
        ),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional limit on matching rows, useful for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output instead of resuming it.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="Log progress after this many selected rows; use 0 to disable.",
    )
    parser.add_argument(
        "--scan-progress-every",
        type=int,
        default=100_000,
        help="Log progress after scanning this many candidates; use 0 to disable.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def _omol25_shard_paths(input_dir: Path) -> list[Path]:
    """Collect real OMol25 ASE-LMDB shards, ignoring AppleDouble sidecars."""
    return sorted(
        path for path in input_dir.rglob("*.aselmdb") if not path.name.startswith("._")
    )


def _read_indices(indices_path: Path) -> list[int]:
    """Read zero-based dataset indices, ignoring comments and blank lines."""
    indices: list[int] = []
    for raw_line in indices_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if line:
            indices.append(int(line))

    if not indices:
        raise ValueError(f"No indices were found in {indices_path}.")
    return indices


def _build_omol25_dataset(
    shard_paths: list[Path],
    data_id: str,
) -> ASELMDBDataset:
    """Construct a raw-row view over the OMol25 ASE-LMDB shards."""
    return ASELMDBDataset(
        paths=shard_paths,
        name=f"omol25_{data_id}_bond_sidecar",
        return_type="row",
        readonly=True,
        readahead=False,
        include_data=True,
    )


def _validate_candidate_indices(
    candidate_indices: list[int],
    dataset_length: int,
) -> None:
    """Ensure all explicitly supplied candidate indices address the dataset."""
    invalid_indices = [
        index for index in candidate_indices if index < 0 or index >= dataset_length
    ]
    if invalid_indices:
        raise IndexError(
            f"{len(invalid_indices)} candidate indices are outside dataset length "
            f"{dataset_length}; first invalid index: {invalid_indices[0]}"
        )


def _metadata_value(value: Any) -> Any:
    """Convert scalar array/tensor metadata to a plain Python value."""
    if hasattr(value, "item"):
        return value.item()
    return value


def _row_data_id(row: Any) -> Any:
    """Extract ``data_id`` from an ASE row without converting its atoms."""
    data = row.get("data", {}) if hasattr(row, "get") else {}
    if isinstance(data, dict) and "data_id" in data:
        return _metadata_value(data["data_id"])

    key_value_pairs = getattr(row, "key_value_pairs", {})
    if "data_id" in key_value_pairs:
        return _metadata_value(key_value_pairs["data_id"])

    if hasattr(row, "get"):
        return _metadata_value(row.get("data_id"))
    return None


def _iter_tasks(
    dataset: ASELMDBDataset,
    candidate_indices: Iterable[int],
    *,
    data_id: str,
    max_records: int | None,
    scan_progress_every: int,
    stats: ScanStats,
) -> Iterator[BondTask]:
    for row_index in candidate_indices:
        if max_records is not None and stats.matched_count >= max_records:
            return

        row = dataset[row_index]
        stats.scanned_count += 1
        if data_id != ALL_DATA_IDS and _row_data_id(row) != data_id:
            if scan_progress_every and stats.scanned_count % scan_progress_every == 0:
                LOGGER.info(
                    "Scanned %d candidate rows; matched %d with data_id=%s",
                    stats.scanned_count,
                    stats.matched_count,
                    data_id,
                )
            continue

        entry_id = str(dataset.idx_to_id(row_index))
        loaded_record = ASE_ATOMS_LOADER(row)
        atom_array = loaded_record.get("atom_array")
        if atom_array is None:
            raise ValueError(
                f"{entry_id}: dataset record does not contain an `atom_array`."
            )
        stats.matched_count += 1
        yield BondTask(entry_id=entry_id, atom_array=atom_array)


def _process_task(task: BondTask) -> BondResult:
    try:
        record = openbabel_bond_record(task.atom_array, example_id=task.entry_id)
        has_disconnected_fragments, fragment_count = _disconnected_fragment_assessment(
            task.atom_array,
            record["bond_edges"],
        )
        record[DISCONNECTED_FRAGMENTS_FIELD] = has_disconnected_fragments
        return BondResult(
            entry_id=task.entry_id,
            value=encode_bond_record(record),
            has_disconnected_fragments=has_disconnected_fragments,
            fragment_count=fragment_count,
            error=None,
        )
    except Exception as exc:
        return BondResult(
            entry_id=task.entry_id,
            value=None,
            has_disconnected_fragments=None,
            fragment_count=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def _disconnected_fragment_assessment(
    atom_array: AtomArray,
    bond_edges: Any,
) -> tuple[bool, int]:
    """Assess disconnected fragments based on the RDKit graph.

    Coordinates are omitted because they cannot affect graph connectivity or
    SMILES. Atoms retain the AtomArray order, all stored Open Babel edges are
    represented as unspecified RDKit bonds, and implicit hydrogens are disabled.
    """
    elements = np.asarray(atom_array.element)
    if elements.ndim != 1 or len(elements) == 0 or len(elements) != len(atom_array):
        raise ValueError(
            "AtomArray elements must be a non-empty, one-dimensional atom list."
        )

    edges = np.asarray(bond_edges, dtype=np.int32)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("bond_edges must have shape [E, 2].")

    editable_molecule = Chem.RWMol()
    for element in elements:
        atom = Chem.Atom(str(element).strip().title())
        atom.SetNoImplicit(True)
        editable_molecule.AddAtom(atom)

    seen_edges: set[tuple[int, int]] = set()
    for atom_i_raw, atom_j_raw in edges:
        atom_i = int(atom_i_raw)
        atom_j = int(atom_j_raw)
        edge = tuple(sorted((atom_i, atom_j)))
        if atom_i == atom_j:
            raise ValueError(f"Self-bond found for atom {atom_i}.")
        if edge in seen_edges:
            raise ValueError(f"Duplicate bond found: {edge}.")
        if atom_i < 0 or atom_j < 0:
            raise ValueError(f"Negative atom index in bond {edge}.")
        if atom_i >= len(elements) or atom_j >= len(elements):
            raise ValueError(f"Out-of-range atom index in bond {edge}.")
        seen_edges.add(edge)
        editable_molecule.AddBond(
            atom_i,
            atom_j,
            Chem.BondType.UNSPECIFIED,
        )

    molecule = editable_molecule.GetMol()
    smiles = Chem.MolToSmiles(molecule, canonical=True)
    fragment_count = smiles.count(".") + 1
    if fragment_count != len(Chem.GetMolFrags(molecule)):
        raise ValueError("SMILES and RDKit fragment counts disagree.")
    return "." in smiles, fragment_count


def _iter_results(
    tasks: Iterable[BondTask],
    *,
    num_workers: int,
) -> Iterator[BondResult]:
    if num_workers == 0:
        yield from map(_process_task, tasks)
        return

    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=num_workers) as pool:
        yield from pool.imap(_process_task, tasks, chunksize=32)


def _remove_existing_output(output: Path) -> None:
    for path in (output, Path(f"{output}-lock")):
        if path.exists():
            path.unlink()


def _metadata(
    *,
    input_dir: Path,
    shard_paths: list[Path],
    data_id: str,
    subset_indices_path: Path | None,
    candidate_count: int,
    scanned_count: int,
    selected_count: int,
    written_count: int,
    existing_count: int,
    connected_count: int,
    disconnected_count: int,
    fragment_count_histogram: Counter[int],
    failures: list[dict[str, str]],
    complete: bool,
) -> dict[str, Any]:
    return {
        "dataset": _dataset_name(input_dir),
        "data_id": data_id,
        "key_field": "example_id",
        "key_encoding": "utf-8",
        "value_encoding": "pickle",
        "bond_edges": "int32[E,2], zero-based, undirected, lexicographically sorted",
        "atomic_numbers_hash_algorithm": ATOMIC_NUMBERS_HASH_ALGORITHM,
        DISCONNECTED_FRAGMENTS_FIELD: (
            "bool; true iff canonical RDKit SMILES contains '.', using ordered "
            "source atoms and the stored Open Babel connectivity"
        ),
        "disconnected_fragment_assessment": {
            "field": DISCONNECTED_FRAGMENTS_FIELD,
            "complete": complete,
            "assessed_record_count": connected_count + disconnected_count,
            "connected_count": connected_count,
            "disconnected_count": disconnected_count,
            "fragment_count_histogram": {
                str(fragment_count): count
                for fragment_count, count in sorted(fragment_count_histogram.items())
            },
            "method": DISCONNECTED_FRAGMENTS_METHOD,
            "rdkit_version": rdBase.rdkitVersion,
            "bond_type": "Chem.BondType.UNSPECIFIED",
            "implicit_hydrogens": False,
            "coordinates_affect_assessment": False,
            "assessed_at_utc": datetime.now(UTC).isoformat() if complete else None,
        },
        "method": "openbabel",
        "method_version": str(openbabel.OBReleaseVersion()),
        "input_dir": str(input_dir.resolve()),
        "input_shards": [path.name for path in shard_paths],
        "subset_indices_path": (
            str(subset_indices_path.resolve()) if subset_indices_path else None
        ),
        "candidate_count": candidate_count,
        "scanned_count": scanned_count,
        "selected_count": selected_count,
        "record_count": written_count + existing_count,
        "written_count": written_count,
        "existing_count": existing_count,
        "failure_count": len(failures),
        "failures": failures,
        "complete": complete,
    }


def _write_failure_report(output: Path, failures: list[dict[str, str]]) -> Path:
    report_path = output.with_suffix(f"{output.suffix}.failures.json")
    report_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    return report_path


def _dataset_name(input_dir: Path) -> str:
    """Return the OMol25 split name recorded in sidecar metadata."""
    return f"OMol25_{input_dir.name}"


def _validate_args(
    args: argparse.Namespace,
) -> tuple[
    Path,
    Path | None,
    Path,
    list[Path],
    list[int] | None,
    str,
    int,
]:
    input_dir = args.input_dir.expanduser().resolve()
    data_id = args.data_id.strip()
    if not data_id or data_id in {".", ".."} or Path(data_id).name != data_id:
        raise ValueError("--data-id must be a non-empty filename-safe value.")

    subset_indices_path = args.subset_indices_path
    if (
        subset_indices_path is None
        and data_id == DEFAULT_DATA_ID
        and DEFAULT_GEOM_INDICES.is_file()
    ):
        subset_indices_path = DEFAULT_GEOM_INDICES
    if subset_indices_path is not None:
        subset_indices_path = subset_indices_path.expanduser().resolve()

    output = args.output
    if output is None:
        output = DEFAULT_OUTPUT_DIR / f"train_4M_{data_id}_bonds.lmdb"
    output = output.expanduser().resolve()

    expected_count = args.expected_count
    if expected_count is None:
        expected_count = KNOWN_EXPECTED_COUNTS.get(data_id, 0)

    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.commit_interval <= 0:
        raise ValueError("--commit-interval must be positive.")
    if args.map_size_gb <= 0:
        raise ValueError("--map-size-gb must be positive.")
    if expected_count < 0:
        raise ValueError("--expected-count must be non-negative.")
    if args.max_records is not None and args.max_records <= 0:
        raise ValueError("--max-records must be positive when provided.")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative.")
    if args.scan_progress_every < 0:
        raise ValueError("--scan-progress-every must be non-negative.")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if subset_indices_path is not None and not subset_indices_path.is_file():
        raise FileNotFoundError(
            f"Subset indices file does not exist: {subset_indices_path}"
        )

    shard_paths = _omol25_shard_paths(input_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No *.aselmdb shards found under {input_dir}")
    candidate_indices = (
        _read_indices(subset_indices_path) if subset_indices_path else None
    )
    return (
        input_dir,
        subset_indices_path,
        output,
        shard_paths,
        candidate_indices,
        data_id,
        expected_count,
    )


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if openbabel is None:
        raise ImportError("Open Babel is not installed in this environment.")

    (
        input_dir,
        subset_indices_path,
        output,
        shard_paths,
        explicit_candidate_indices,
        data_id,
        expected_count,
    ) = _validate_args(args)
    dataset = _build_omol25_dataset(shard_paths, data_id)
    if explicit_candidate_indices is not None:
        _validate_candidate_indices(explicit_candidate_indices, len(dataset))
        candidate_indices: Iterable[int] = explicit_candidate_indices
        candidate_source = str(subset_indices_path)
    else:
        candidate_indices = range(len(dataset))
        candidate_source = "all dataset rows"
    candidate_count = len(candidate_indices)
    if data_id == ALL_DATA_IDS and args.expected_count is None:
        expected_count = candidate_count

    output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        _remove_existing_output(output)

    LOGGER.info(
        "Building %s for data_id=%s from %d ASE-LMDB shards; scanning %d "
        "candidates from %s with %d Open Babel workers",
        output,
        data_id,
        len(shard_paths),
        candidate_count,
        candidate_source,
        args.num_workers,
    )
    environment = lmdb.open(
        str(output),
        subdir=False,
        map_size=int(args.map_size_gb * 1024**3),
        create=True,
        readonly=False,
        readahead=False,
        meminit=False,
        map_async=True,
    )

    selected_count = 0
    written_count = 0
    existing_count = 0
    connected_count = 0
    disconnected_count = 0
    fragment_count_histogram: Counter[int] = Counter()
    failures: list[dict[str, str]] = []
    transaction = environment.begin(write=True)
    pending_writes = 0

    scan_stats = ScanStats()
    try:
        tasks = _iter_tasks(
            dataset,
            candidate_indices,
            data_id=data_id,
            max_records=args.max_records,
            scan_progress_every=args.scan_progress_every,
            stats=scan_stats,
        )
        for result in _iter_results(tasks, num_workers=args.num_workers):
            selected_count += 1
            key = result.entry_id.encode("utf-8")
            existing_value = transaction.get(key)

            if result.error is not None:
                failures.append({"entry_id": result.entry_id, "error": result.error})
            else:
                if (
                    result.has_disconnected_fragments is None
                    or result.fragment_count is None
                ):
                    raise AssertionError("Successful result lacks fragment metadata.")
                fragment_count_histogram[result.fragment_count] += 1
                if result.has_disconnected_fragments:
                    disconnected_count += 1
                else:
                    connected_count += 1

                if existing_value is not None:
                    if existing_value != result.value:
                        raise RuntimeError(
                            "LMDB key collision with different bond data: "
                            f"{result.entry_id}"
                        )
                    existing_count += 1
                else:
                    if not transaction.put(key, result.value, overwrite=False):
                        raise RuntimeError(f"Duplicate LMDB key: {result.entry_id}")
                    written_count += 1
                    pending_writes += 1

            if pending_writes >= args.commit_interval:
                transaction.commit()
                transaction = environment.begin(write=True)
                pending_writes = 0

            if args.progress_every and selected_count % args.progress_every == 0:
                LOGGER.info(
                    "Selected %d rows; wrote %d; already present %d; failures %d",
                    selected_count,
                    written_count,
                    existing_count,
                    len(failures),
                )

        limited_run = args.max_records is not None
        expected_matches = expected_count == 0 or selected_count == expected_count
        complete = (
            not limited_run and expected_matches and not failures and selected_count > 0
        )
        assessed_count = connected_count + disconnected_count
        successful_count = written_count + existing_count
        if assessed_count != successful_count:
            raise RuntimeError(
                "Disconnected-fragment assessment count differs from the "
                "successfully processed record count."
            )
        metadata = _metadata(
            input_dir=input_dir,
            shard_paths=shard_paths,
            data_id=data_id,
            subset_indices_path=subset_indices_path,
            candidate_count=candidate_count,
            scanned_count=scan_stats.scanned_count,
            selected_count=selected_count,
            written_count=written_count,
            existing_count=existing_count,
            connected_count=connected_count,
            disconnected_count=disconnected_count,
            fragment_count_histogram=fragment_count_histogram,
            failures=failures,
            complete=complete,
        )
        transaction.put(
            BOND_SIDECAR_METADATA_KEY,
            pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL),
        )
        transaction.commit()
        transaction = None
        environment.sync(True)
    finally:
        if transaction is not None:
            transaction.abort()
        environment.close()
        dataset.close()

    if failures:
        report_path = _write_failure_report(output, failures)
        warnings.warn(
            f"Open Babel failed for {len(failures)} rows; see {report_path}.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        report_path = output.with_suffix(f"{output.suffix}.failures.json")
        if report_path.exists():
            report_path.unlink()
    if selected_count == 0:
        raise RuntimeError(
            f"No OMol25 rows with data_id={data_id!r} were found in the "
            f"{scan_stats.scanned_count} scanned candidates."
        )
    if args.max_records is None and expected_count:
        if selected_count != expected_count:
            raise RuntimeError(
                f"Selected {selected_count} {data_id} rows, expected "
                f"{expected_count}. The sidecar is marked incomplete."
            )

    LOGGER.info(
        "Finished: scanned %d candidates; selected %d rows; wrote %d; already "
        "present %d; output %s",
        scan_stats.scanned_count,
        selected_count,
        written_count,
        existing_count,
        output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
