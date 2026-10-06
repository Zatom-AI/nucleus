#!/usr/bin/env python3
"""Generate OMol25 atom-count contig JSON files for Zatom-2 inference."""

import argparse
import csv
import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "models" / "zatom2" / "docs" / "examples"
DEFAULT_ATOM_COUNT_CACHE_PATH = (
    REPO_ROOT / "data" / "omol25" / "statistics" / "atom_count_cache.csv"
)
DEFAULT_SAMPLE_SIZE = 25
DEFAULT_SEED = 42
DEFAULT_DATA_ID = "geom_orca6"
DEFAULT_OUTPUT_PREFIX = "omol25_geom_atom_count"
DEFAULT_ENTRY_PREFIX = "omol25_geom_atom_count_repr"
ATOM_COUNT_CACHE_VERSION = "1"

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndexedRow:
    row_index: int
    split: str
    source: str


@dataclass(slots=True)
class AtomCountDistribution:
    atom_counts: Counter[int]
    rows_by_atom_count: dict[int, list[IndexedRow]]
    n_rows: int


@dataclass(frozen=True, slots=True)
class CachedAtomCount:
    n_atoms: int
    data_id: str | None


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required to sample atom counts with torch.multinomial. "
            "Run this script in the project environment."
        ) from exc

    return torch


def _read_env_value(env_path: Path, key: str) -> str | None:
    if not env_path.exists():
        return None

    prefix = f"{key}="
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith(prefix):
            continue

        value = line[len(prefix) :].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return os.path.expandvars(os.path.expanduser(value))

    return None


def _default_omol25_data_dir() -> Path:
    env_value = os.environ.get("OMOL25_DATA_DIR")
    if env_value:
        return Path(env_value).expanduser()

    dotenv_value = _read_env_value(REPO_ROOT / ".env", "OMOL25_DATA_DIR")
    if dotenv_value:
        return Path(dotenv_value).expanduser()

    return REPO_ROOT / "data" / "omol25" / "raw" / "train_4M"


def _ensure_import_cache_dirs() -> None:
    cache_root = Path(os.environ.get("TMPDIR", "/tmp"))
    user = os.environ.get("USER", "user")
    if "XDG_CACHE_HOME" not in os.environ:
        xdg_cache = cache_root / f"nucleus-omol25-contig-json-cache-{user}"
        xdg_cache.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(xdg_cache)

    if "MPLCONFIGDIR" not in os.environ:
        mpl_cache = cache_root / f"matplotlib-omol25-contig-json-{user}"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_cache)


def _omol25_shard_paths(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.rglob("*.aselmdb") if not path.name.startswith("._")
    )


def _build_omol25_dataset(shard_paths: list[Path]) -> Any:
    _ensure_import_cache_dirs()

    try:
        from atomworks.ml.datasets import ASELMDBDataset
    except ImportError as exc:
        raise ImportError(
            "atomworks is required to read OMol25 ASE-LMDB shards. "
            "Run this script in the project environment with atomworks installed."
        ) from exc

    return ASELMDBDataset(
        paths=shard_paths,
        name="omol25_atom_count_contig_json",
        return_type="atoms",
        readonly=True,
        readahead=False,
        include_data=True,
    )


def _split_from_index_path(path: Path) -> str:
    stem = path.stem
    for split in ("train", "val", "test"):
        if f"_{split}_" in stem or stem.endswith(f"_{split}"):
            return split
    return stem


def _read_indices(
    paths: list[Path],
    dataset_length: int,
    max_rows: int | None,
) -> list[IndexedRow]:
    rows: list[IndexedRow] = []
    seen: set[int] = set()
    duplicate_count = 0

    for path in paths:
        expanded_path = path.expanduser()
        if not expanded_path.exists():
            raise FileNotFoundError(f"Index file does not exist: {expanded_path}")

        split = _split_from_index_path(expanded_path)
        file_count = 0
        for raw_line in expanded_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", maxsplit=1)[0].strip()
            if not line:
                continue

            row_index = int(line)
            if row_index < 0 or row_index >= dataset_length:
                raise IndexError(
                    f"Index {row_index} from {expanded_path} is outside dataset "
                    f"length {dataset_length}."
                )

            file_count += 1
            if row_index in seen:
                duplicate_count += 1
                continue

            seen.add(row_index)
            rows.append(
                IndexedRow(
                    row_index=row_index,
                    split=split,
                    source=str(expanded_path),
                )
            )
            if max_rows is not None and len(rows) >= max_rows:
                break

        LOGGER.info("Loaded %d candidate rows from %s", file_count, expanded_path)
        if max_rows is not None and len(rows) >= max_rows:
            break

    if not rows:
        raise ValueError("No OMol25 row indices were loaded.")
    if duplicate_count:
        LOGGER.warning("Ignored %d duplicated row indices", duplicate_count)

    return rows


def _all_dataset_rows(
    dataset_length: int,
    max_rows: int | None,
) -> Iterable[IndexedRow]:
    stop = dataset_length if max_rows is None else min(max_rows, dataset_length)
    for row_index in range(stop):
        yield IndexedRow(
            row_index=row_index,
            split="all",
            source="all_dataset_rows",
        )


def _metadata_value(info: dict[str, Any], key: str) -> Any:
    value = info.get(key)
    if hasattr(value, "item"):
        return value.item()
    return value


def _electronic_metadata(atoms: Any, row_index: int) -> tuple[int, int]:
    """Read the integer total charge and spin multiplicity used in training."""
    info = dict(getattr(atoms, "info", {}))
    nested_info = info.get("data")
    if not isinstance(nested_info, dict):
        nested_info = {}
    charge = info.get("charge", nested_info.get("charge"))
    spin = info.get("spin", nested_info.get("spin"))
    try:
        charge_value = float(charge)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Row {row_index} has invalid total charge {charge!r}."
        ) from exc
    try:
        spin_value = float(spin)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Row {row_index} has invalid spin multiplicity {spin!r}."
        ) from exc
    if not math.isfinite(charge_value) or not charge_value.is_integer():
        raise ValueError(f"Row {row_index} has invalid total charge {charge!r}.")
    if not math.isfinite(spin_value) or not spin_value.is_integer() or spin_value < 1:
        raise ValueError(f"Row {row_index} has invalid spin multiplicity {spin!r}.")
    return int(charge_value), int(spin_value)


def _cached_data_id(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _hydrogen_count(atoms: Any) -> int:
    return sum(symbol == "H" for symbol in atoms.get_chemical_symbols())


def _force_array_from_atoms(atoms: Any, row_index: int) -> Any:
    calc = getattr(atoms, "calc", None)
    if calc is not None and "forces" in getattr(calc, "results", {}):
        return calc.results["forces"]

    try:
        return atoms.get_forces()
    except Exception as exc:
        raise ValueError(f"Row {row_index} does not contain forces.") from exc


def _mean_force_norm(atoms: Any, row_index: int) -> float:
    torch = _load_torch()
    forces = torch.as_tensor(
        _force_array_from_atoms(atoms, row_index),
        dtype=torch.float64,
    )
    if forces.ndim != 2 or forces.shape[-1] != 3:
        raise ValueError(
            f"Row {row_index} has invalid forces shape {tuple(forces.shape)}; "
            "expected [N_atoms, 3]."
        )
    if forces.shape[0] == 0:
        raise ValueError(f"Row {row_index} contains zero atoms.")
    if not torch.isfinite(forces).all():
        raise ValueError(f"Row {row_index} contains non-finite forces.")

    return float(torch.linalg.norm(forces, dim=-1).mean().item())


def _validate_data_id_value(
    actual_data_id: Any,
    *,
    row_index: int,
    expected_data_id: str | None,
) -> None:
    if expected_data_id is None:
        return

    if actual_data_id != expected_data_id:
        raise ValueError(
            f"Index {row_index} is not a {expected_data_id} row: "
            f"data_id={actual_data_id!r}"
        )


def _validate_cache_path(cache_path: Path) -> None:
    cache_dir = cache_path.parent
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Atom-count cache directory does not exist: {cache_dir}"
        )
    if not cache_dir.is_dir():
        raise NotADirectoryError(
            f"Atom-count cache parent is not a directory: {cache_dir}"
        )
    if cache_path.exists() and not cache_path.is_file():
        raise ValueError(f"Atom-count cache path is not a file: {cache_path}")


def _parse_cache_metadata(line: str) -> tuple[str, str] | None:
    line = line.removeprefix("#").strip()
    if "=" not in line:
        return None
    key, value = line.split("=", maxsplit=1)
    return key.strip(), value.strip()


def _read_atom_count_cache_metadata(handle: Any) -> tuple[dict[str, str], str | None]:
    metadata: dict[str, str] = {}
    first_data_line: str | None = None

    for raw_line in handle:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            first_data_line = raw_line
            break

        item = _parse_cache_metadata(stripped)
        if item is not None:
            key, value = item
            metadata[key] = value

    return metadata, first_data_line


def _validate_atom_count_cache_metadata(
    metadata: dict[str, str],
    *,
    dataset_length: int,
    cache_path: Path,
) -> None:
    version = metadata.get("version")
    if version != ATOM_COUNT_CACHE_VERSION:
        raise ValueError(
            f"Atom-count cache {cache_path} has version {version!r}; expected "
            f"{ATOM_COUNT_CACHE_VERSION!r}."
        )

    cached_dataset_length = metadata.get("dataset_length")
    if cached_dataset_length != str(dataset_length):
        raise ValueError(
            f"Atom-count cache {cache_path} has dataset_length "
            f"{cached_dataset_length!r}; expected {dataset_length}."
        )


def _read_atom_count_cache(
    cache_path: Path,
    *,
    dataset_length: int,
    required_indices: set[int] | None,
) -> dict[int, CachedAtomCount]:
    if not cache_path.exists():
        LOGGER.info("Atom-count cache does not exist yet: %s", cache_path)
        return {}

    cached_rows: dict[int, CachedAtomCount] = {}
    with cache_path.open("r", encoding="utf-8", newline="") as handle:
        metadata, first_data_line = _read_atom_count_cache_metadata(handle)
        _validate_atom_count_cache_metadata(
            metadata,
            dataset_length=dataset_length,
            cache_path=cache_path,
        )
        if first_data_line is None:
            raise ValueError(f"Atom-count cache has no CSV rows: {cache_path}")

        reader = csv.DictReader(chain([first_data_line], handle))
        expected_fields = {"row_index", "n_atoms", "data_id"}
        if reader.fieldnames is None or set(reader.fieldnames) != expected_fields:
            raise ValueError(
                f"Atom-count cache {cache_path} must have CSV columns "
                "row_index,n_atoms,data_id."
            )

        for row in reader:
            row_index = int(row["row_index"])
            if required_indices is not None and row_index not in required_indices:
                continue
            if row_index in cached_rows:
                raise ValueError(
                    f"Atom-count cache {cache_path} contains duplicate row "
                    f"{row_index}."
                )
            if row_index < 0 or row_index >= dataset_length:
                raise IndexError(
                    f"Atom-count cache row {row_index} is outside dataset "
                    f"length {dataset_length}."
                )

            n_atoms = int(row["n_atoms"])
            if n_atoms <= 0:
                raise ValueError(
                    f"Atom-count cache row {row_index} has invalid n_atoms "
                    f"{n_atoms}."
                )
            data_id = row["data_id"] or None
            cached_rows[row_index] = CachedAtomCount(
                n_atoms=n_atoms,
                data_id=data_id,
            )

    LOGGER.info("Loaded %d atom-count cache rows from %s", len(cached_rows), cache_path)
    return cached_rows


def _write_atom_count_cache_header(
    handle: Any,
    *,
    dataset_length: int,
) -> None:
    handle.write(f"# version={ATOM_COUNT_CACHE_VERSION}\n")
    handle.write(f"# dataset_length={dataset_length}\n")
    writer = csv.writer(handle)
    writer.writerow(["row_index", "n_atoms", "data_id"])


def _write_atom_count_cache_entries(
    cache_path: Path,
    entries: dict[int, CachedAtomCount],
    *,
    dataset_length: int,
    overwrite: bool,
) -> None:
    if not entries and not overwrite:
        return

    cache_exists = cache_path.exists()
    mode = "w" if overwrite or not cache_exists else "a"
    with cache_path.open(mode, encoding="utf-8", newline="") as handle:
        if mode == "w":
            _write_atom_count_cache_header(handle, dataset_length=dataset_length)

        writer = csv.writer(handle)
        for row_index, entry in sorted(entries.items()):
            writer.writerow([row_index, entry.n_atoms, entry.data_id or ""])

    action = "Wrote" if mode == "w" else "Appended"
    LOGGER.info("%s %d atom-count rows to %s", action, len(entries), cache_path)


def collect_atom_count_distribution(
    dataset: Any,
    rows: Iterable[IndexedRow],
    *,
    data_id: str | None,
    progress_every: int,
    total_rows: int | None,
    atom_count_cache: dict[int, CachedAtomCount] | None,
) -> tuple[AtomCountDistribution, dict[int, CachedAtomCount]]:
    atom_counts: Counter[int] = Counter()
    rows_by_atom_count: dict[int, list[IndexedRow]] = defaultdict(list)
    new_cache_entries: dict[int, CachedAtomCount] = {}
    n_rows = 0

    for row_number, row in enumerate(rows, start=1):
        n_rows = row_number
        cached_atom_count = (
            atom_count_cache.get(row.row_index)
            if atom_count_cache is not None
            else None
        )
        if cached_atom_count is not None:
            _validate_data_id_value(
                cached_atom_count.data_id,
                row_index=row.row_index,
                expected_data_id=data_id,
            )
            n_atoms = cached_atom_count.n_atoms
        else:
            atoms = dataset[row.row_index]
            info = dict(getattr(atoms, "info", {}))
            actual_data_id = _metadata_value(info, "data_id")
            _validate_data_id_value(
                actual_data_id,
                row_index=row.row_index,
                expected_data_id=data_id,
            )

            n_atoms = len(atoms)
            new_cache_entries[row.row_index] = CachedAtomCount(
                n_atoms=n_atoms,
                data_id=_cached_data_id(actual_data_id),
            )

        if n_atoms <= 0:
            raise ValueError(f"Row {row.row_index} contains zero atoms.")

        atom_counts[n_atoms] += 1
        rows_by_atom_count[n_atoms].append(row)

        if progress_every and row_number % progress_every == 0:
            total_message = f"/{total_rows}" if total_rows is not None else ""
            LOGGER.info(
                "Collected atom counts for %d%s reference rows",
                row_number,
                total_message,
            )

    if not atom_counts:
        raise ValueError("No atom-count statistics were collected.")

    return (
        AtomCountDistribution(
            atom_counts=atom_counts,
            rows_by_atom_count=dict(rows_by_atom_count),
            n_rows=n_rows,
        ),
        new_cache_entries,
    )


def sample_atom_counts(
    atom_counts: Counter[int],
    *,
    sample_size: int,
    seed: int,
    with_replacement: bool,
) -> tuple[list[int], Any]:
    torch = _load_torch()
    atom_count_values = torch.tensor(sorted(atom_counts), dtype=torch.long)
    weights = torch.tensor(
        [atom_counts[int(n_atoms)] for n_atoms in atom_count_values],
        dtype=torch.float64,
    )
    nonzero_bins = int((weights > 0).sum().item())
    if not with_replacement and sample_size > nonzero_bins:
        raise ValueError(
            f"Cannot sample {sample_size} atom-count bins without replacement "
            f"from {nonzero_bins} populated bins."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)
    sampled_positions = torch.multinomial(
        weights,
        num_samples=sample_size,
        replacement=with_replacement,
        generator=generator,
    )
    sampled_atom_counts = atom_count_values[sampled_positions].tolist()
    return [int(n_atoms) for n_atoms in sampled_atom_counts], generator


def _sample_source_row(
    rows_by_atom_count: dict[int, list[IndexedRow]],
    n_atoms: int,
    generator: Any,
) -> IndexedRow:
    torch = _load_torch()
    candidates = rows_by_atom_count[n_atoms]
    position = int(
        torch.randint(
            low=0,
            high=len(candidates),
            size=(1,),
            generator=generator,
        ).item()
    )
    return candidates[position]


def _sanitize_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _subset_label(indices_paths: list[Path] | None) -> str:
    if not indices_paths:
        return "all"
    if len(indices_paths) == 1:
        return _sanitize_label(indices_paths[0].expanduser().stem)
    stems = "_".join(path.expanduser().stem for path in indices_paths)
    return _sanitize_label(stems)


def _source_distribution(
    *,
    data_id: str | None,
    subset_label: str,
) -> str:
    data_label = _sanitize_label(data_id) if data_id else "all"
    return f"omol25_{data_label}_explicit_h_atom_count_multinomial_over_{subset_label}"


def _default_output_path(args: argparse.Namespace) -> Path:
    subset_label = _subset_label(args.indices_path)
    force_suffix = "_with_forces" if args.save_forces else ""
    output_name = (
        f"{args.output_prefix}_{subset_label}_{args.sample_size}"
        f"_with_hydrogens{force_suffix}.json"
    )
    return args.output_dir / output_name


def _build_entry(
    dataset: Any,
    row: IndexedRow,
    *,
    n_atoms: int,
    sample_index: int,
    sample_size: int,
    seed: int,
    with_replacement: bool,
    save_forces: bool,
    source_distribution: str,
    n_reference_rows: int,
    n_atom_count_bins: int,
    key_width: int,
    entry_prefix: str,
) -> tuple[str, dict[str, Any]]:
    atoms = dataset[row.row_index]
    if len(atoms) != n_atoms:
        raise RuntimeError(
            f"Sampled source row {row.row_index} has {len(atoms)} atoms, "
            f"but atom-count bin is {n_atoms}."
        )

    n_hydrogens = _hydrogen_count(atoms)
    extra: dict[str, Any] = {
        "source_distribution": source_distribution,
        "sample_seed": seed,
        "sample_index": sample_index,
        "sample_size": sample_size,
        "sample_with_replacement": with_replacement,
        "source_split": row.split,
        "source_row_index": row.row_index,
        "source_indices_path": row.source,
        "num_reference_rows": n_reference_rows,
        "num_atom_count_bins": n_atom_count_bins,
        "num_atoms_with_hydrogens": n_atoms,
        "num_hydrogens": n_hydrogens,
        "has_mean_force_norm": False,
    }
    if save_forces:
        extra["mean_force_norm"] = _mean_force_norm(atoms, row.row_index)
        extra["has_mean_force_norm"] = True

    total_charge, spin_multiplicity = _electronic_metadata(atoms, row.row_index)

    key = (
        f"{entry_prefix}_{sample_index:0{key_width}d}"
        f"_atoms{n_atoms}_hydrogens{n_hydrogens}"
    )
    return key, {
        "contig": f"{n_atoms}m",
        "total_charge": total_charge,
        "spin_multiplicity": spin_multiplicity,
        "extra": extra,
    }


def build_entries(
    dataset: Any,
    distribution: AtomCountDistribution,
    *,
    sample_size: int,
    seed: int,
    with_replacement: bool,
    save_forces: bool,
    source_distribution: str,
    entry_prefix: str,
) -> dict[str, dict[str, Any]]:
    sampled_atom_counts, generator = sample_atom_counts(
        distribution.atom_counts,
        sample_size=sample_size,
        seed=seed,
        with_replacement=with_replacement,
    )
    key_width = max(3, len(str(sample_size - 1)))
    entries: dict[str, dict[str, Any]] = {}

    for sample_index, n_atoms in enumerate(sampled_atom_counts):
        row = _sample_source_row(distribution.rows_by_atom_count, n_atoms, generator)
        key, entry = _build_entry(
            dataset,
            row,
            n_atoms=n_atoms,
            sample_index=sample_index,
            sample_size=sample_size,
            seed=seed,
            with_replacement=with_replacement,
            save_forces=save_forces,
            source_distribution=source_distribution,
            n_reference_rows=distribution.n_rows,
            n_atom_count_bins=len(distribution.atom_counts),
            key_width=key_width,
            entry_prefix=entry_prefix,
        )
        entries[key] = entry

    if len(entries) != sample_size:
        raise RuntimeError(
            f"Generated {len(entries)} unique JSON keys for {sample_size} samples."
        )

    return entries


def _write_json(output_path: Path, entries: dict[str, dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(entries, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Zatom-2 molecule-contig JSON by sampling atom counts from "
            "the empirical OMol25 atom-count distribution with torch.multinomial."
        )
    )
    parser.add_argument(
        "--omol25-data-dir",
        "--input-dir",
        dest="omol25_data_dir",
        type=Path,
        default=_default_omol25_data_dir(),
        help=(
            "Directory containing OMol25 *.aselmdb shards. Defaults to "
            "OMOL25_DATA_DIR from the environment/.env, then "
            "data/omol25/raw/train_4M."
        ),
    )
    parser.add_argument(
        "--indices-path",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Optional text file(s) of zero-based OMol25 dataset row indices used "
            "to build the reference atom-count distribution."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. If omitted, a filename is derived under "
            "models/zatom2/docs/examples."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXAMPLES_DIR,
        help="Default directory for derived output JSON files.",
    )
    parser.add_argument(
        "--atom-count-cache-path",
        type=Path,
        default=DEFAULT_ATOM_COUNT_CACHE_PATH,
        help=(
            "CSV cache path for per-row atom counts. The parent directory must "
            "already exist. Existing caches are reused and missing rows are "
            "appended."
        ),
    )
    parser.add_argument(
        "--no-atom-count-cache",
        action="store_true",
        help="Disable reading and writing the per-row atom-count cache.",
    )
    parser.add_argument(
        "--overwrite-atom-count-cache",
        action="store_true",
        help=(
            "Ignore any existing atom-count cache and rewrite it with atom "
            "counts for the requested reference rows."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Filename prefix used when --output is omitted.",
    )
    parser.add_argument(
        "--entry-prefix",
        default=DEFAULT_ENTRY_PREFIX,
        help="Top-level JSON key prefix for generated entries.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of JSON entries to write.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed passed to torch.Generator.",
    )
    parser.add_argument(
        "--without-replacement",
        action="store_true",
        help=(
            "Sample atom-count bins without replacement. By default, atom counts "
            "are sampled with replacement, matching independent multinomial draws."
        ),
    )
    parser.add_argument(
        "--save-forces",
        action="store_true",
        help=(
            "Add mean_force_norm metadata from the sampled source row and set "
            "has_mean_force_norm=true. Without this flag, has_mean_force_norm is "
            "written as false."
        ),
    )
    parser.add_argument(
        "--data-id",
        default=DEFAULT_DATA_ID,
        help=(
            "Expected atoms.info['data_id'] value for reference rows. Pass an "
            "empty string to disable this check."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional debug limit on reference rows processed after subsetting.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Log progress every N reference rows. Use 0 to disable progress logging.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive.")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive when provided.")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative.")
    if args.no_atom_count_cache and args.overwrite_atom_count_cache:
        raise ValueError(
            "--overwrite-atom-count-cache cannot be used with " "--no-atom-count-cache."
        )

    atom_count_cache_path = None
    if not args.no_atom_count_cache:
        atom_count_cache_path = args.atom_count_cache_path.expanduser()
        _validate_cache_path(atom_count_cache_path)

    omol25_data_dir = args.omol25_data_dir.expanduser()
    if not omol25_data_dir.exists():
        raise FileNotFoundError(
            f"OMol25 data directory does not exist: {omol25_data_dir}"
        )

    shard_paths = _omol25_shard_paths(omol25_data_dir)
    if not shard_paths:
        raise FileNotFoundError(
            f"No OMol25 *.aselmdb shards were found in {omol25_data_dir}"
        )

    LOGGER.info("Found %d OMol25 shards under %s", len(shard_paths), omol25_data_dir)
    dataset = _build_omol25_dataset(shard_paths)
    LOGGER.info("Dataset length: %d", len(dataset))

    if args.indices_path:
        rows = _read_indices(args.indices_path, len(dataset), args.max_rows)
        total_rows = len(rows)
        required_cache_indices = {row.row_index for row in rows}
    else:
        rows = _all_dataset_rows(len(dataset), args.max_rows)
        total_rows = (
            len(dataset) if args.max_rows is None else min(args.max_rows, len(dataset))
        )
        required_cache_indices = (
            set(range(total_rows)) if args.max_rows is not None else None
        )
        LOGGER.info("Using %d rows from the full dataset", total_rows)

    atom_count_cache = None
    if atom_count_cache_path is not None:
        if args.overwrite_atom_count_cache:
            atom_count_cache = {}
            LOGGER.info("Ignoring existing atom-count cache for overwrite")
        else:
            atom_count_cache = _read_atom_count_cache(
                atom_count_cache_path,
                dataset_length=len(dataset),
                required_indices=required_cache_indices,
            )

    distribution, new_cache_entries = collect_atom_count_distribution(
        dataset,
        rows,
        data_id=args.data_id or None,
        progress_every=args.progress_every,
        total_rows=total_rows,
        atom_count_cache=atom_count_cache,
    )
    if atom_count_cache_path is not None:
        _write_atom_count_cache_entries(
            atom_count_cache_path,
            new_cache_entries,
            dataset_length=len(dataset),
            overwrite=args.overwrite_atom_count_cache,
        )

    LOGGER.info(
        "Collected %d reference rows across %d atom-count bins",
        distribution.n_rows,
        len(distribution.atom_counts),
    )

    subset_label = _subset_label(args.indices_path)
    entries = build_entries(
        dataset,
        distribution,
        sample_size=args.sample_size,
        seed=args.seed,
        with_replacement=not args.without_replacement,
        save_forces=args.save_forces,
        source_distribution=_source_distribution(
            data_id=args.data_id or None,
            subset_label=subset_label,
        ),
        entry_prefix=args.entry_prefix,
    )

    output_path = (
        args.output.expanduser() if args.output else _default_output_path(args)
    )
    _write_json(output_path, entries)
    LOGGER.info("Wrote %d entries to %s", len(entries), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
