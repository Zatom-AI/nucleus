#!/usr/bin/env python3
"""Filter OMol25 GEOM split indices by mean per-atom force norm."""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBSETS_DIR = REPO_ROOT / "data" / "omol25" / "subsets"
DEFAULT_SPLITS = ("train", "val", "test")
DEFAULT_DATA_ID = "geom_orca6"
FORCE_NORM_THRESHOLD = 1.0

load_dotenv(REPO_ROOT / ".env", override=True)

LOGGER = logging.getLogger(__name__)


def _default_omol25_data_dir() -> Path:
    env_value = os.environ.get("OMOL25_DATA_DIR")
    if env_value:
        return Path(env_value).expanduser()

    return REPO_ROOT / "data" / "omol25" / "raw" / "train_4M"


def _read_indices(indices_path: Path, max_rows: int | None) -> list[int]:
    indices: list[int] = []
    for raw_line in indices_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line:
            continue

        indices.append(int(line))
        if max_rows is not None and len(indices) >= max_rows:
            break

    if not indices:
        raise ValueError(f"No indices were found in {indices_path}.")
    return indices


def _omol25_shard_paths(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*.aselmdb")
        if not path.name.startswith("._")
    )


def _build_omol25_dataset(shard_paths: list[Path]) -> Any:
    try:
        from atomworks.ml.datasets import ASELMDBDataset
    except ImportError as exc:
        raise ImportError(
            "atomworks is required to read OMol25 ASE-LMDB shards. "
            "Run this script in the project environment with atomworks installed."
        ) from exc

    return ASELMDBDataset(
        paths=shard_paths,
        name="omol25_geom_force_norm_filter",
        return_type="atoms",
        readonly=True,
        readahead=False,
        include_data=True,
    )


def _indices_path_for_split(split: str) -> Path:
    return SUBSETS_DIR / f"train_4M_geom_orca6_{split}_indices.txt"


def _output_path_for_split(split: str, output_dir: Path) -> Path:
    return output_dir / f"train_4M_geom_orca6_{split}_low_force_indices.txt"


def _ensure_import_cache_dirs() -> None:
    cache_root = Path(os.environ.get("TMPDIR", "/tmp"))
    user = os.environ.get("USER", "user")
    if "XDG_CACHE_HOME" not in os.environ:
        xdg_cache = cache_root / f"nucleus-omol25-force-filter-cache-{user}"
        xdg_cache.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(xdg_cache)

    if "MPLCONFIGDIR" not in os.environ:
        mpl_cache = cache_root / f"matplotlib-omol25-force-filter-{user}"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_cache)


def _force_array_from_atoms(atoms: Any, row_index: int) -> Any:
    calc = getattr(atoms, "calc", None)
    if calc is not None and "forces" in getattr(calc, "results", {}):
        return calc.results["forces"]

    try:
        return atoms.get_forces()
    except Exception as exc:
        raise ValueError(f"Row {row_index} does not contain forces.") from exc


def _mean_force_norm(atoms: Any, row_index: int) -> float:
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

    force_norms = torch.linalg.norm(forces, dim=-1)
    return float(force_norms.mean().item())


def _metadata_value(info: dict[str, Any], key: str) -> Any:
    value = info.get(key)
    if hasattr(value, "item"):
        return value.item()
    return value


def _filter_indices(
    dataset: Any,
    indices: list[int],
    *,
    split: str,
    data_id: str,
    threshold: float,
    progress_every: int,
) -> list[int]:
    selected_indices: list[int] = []
    for row_number, row_index in enumerate(indices, start=1):
        atoms = dataset[row_index]
        info = dict(getattr(atoms, "info", {}))
        actual_data_id = _metadata_value(info, "data_id")
        if actual_data_id != data_id:
            raise ValueError(
                f"Index {row_index} is not a {data_id} row: "
                f"data_id={actual_data_id!r}"
            )

        if _mean_force_norm(atoms, row_index) < threshold:
            selected_indices.append(row_index)

        if progress_every and row_number % progress_every == 0:
            LOGGER.info(
                "%s: processed %d/%d rows; selected %d",
                split,
                row_number,
                len(indices),
                len(selected_indices),
            )

    return selected_indices


def _write_indices(
    output_path: Path,
    selected_indices: list[int],
    *,
    split: str,
    source_indices_path: Path,
    data_id: str,
    dataset_len: int,
    threshold: float,
    total_count: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# data_id={data_id} dataset_len={dataset_len}\n")
        handle.write(
            f"# split={split} "
            f"source={source_indices_path} "
            "filter=mean_per_atom_force_norm_lt_1 "
            f"threshold={threshold:g} "
            "force_units=eV/Angstrom "
            f"count={len(selected_indices)} "
            f"total_count={total_count}\n"
        )
        for row_index in selected_indices:
            handle.write(f"{row_index}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read OMol25 GEOM train/val/test indices and write each subset "
            "whose mean per-atom force norm is below 1 eV/Angstrom."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_default_omol25_data_dir(),
        help=(
            "Directory containing OMol25 train_4M *.aselmdb shards. Defaults to "
            "OMOL25_DATA_DIR from .env, then data/omol25/raw/train_4M."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=DEFAULT_SPLITS,
        default=list(DEFAULT_SPLITS),
        help="OMol25 GEOM split index files to filter.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SUBSETS_DIR,
        help="Directory where selected index files are written.",
    )
    parser.add_argument(
        "--data-id",
        default=DEFAULT_DATA_ID,
        help="Expected atoms.info['data_id'] value for every indexed row.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional debug limit on rows read from the input index file.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Log progress every N rows. Use 0 to disable progress logging.",
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

    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive when provided.")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative.")

    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser()

    if not input_dir.exists():
        raise FileNotFoundError(f"OMol25 input directory does not exist: {input_dir}")

    shard_paths = _omol25_shard_paths(input_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No OMol25 *.aselmdb shards were found in {input_dir}")

    LOGGER.info("Found %d OMol25 shards under %s", len(shard_paths), input_dir)
    _ensure_import_cache_dirs()
    dataset = _build_omol25_dataset(shard_paths)

    for split in args.splits:
        input_indices_path = _indices_path_for_split(split)
        if not input_indices_path.exists():
            raise FileNotFoundError(
                f"Input indices file does not exist: {input_indices_path}"
            )

        indices = _read_indices(input_indices_path, args.max_rows)
        LOGGER.info(
            "%s: loaded %d input indices from %s",
            split,
            len(indices),
            input_indices_path,
        )

        selected_indices = _filter_indices(
            dataset,
            indices,
            split=split,
            data_id=args.data_id,
            threshold=FORCE_NORM_THRESHOLD,
            progress_every=args.progress_every,
        )
        output_indices_path = _output_path_for_split(split, output_dir)
        _write_indices(
            output_indices_path,
            selected_indices,
            split=split,
            source_indices_path=input_indices_path,
            data_id=args.data_id,
            dataset_len=len(dataset),
            threshold=FORCE_NORM_THRESHOLD,
            total_count=len(indices),
        )

        LOGGER.info(
            "%s: selected %d/%d rows with mean per-atom force norm < %.6g",
            split,
            len(selected_indices),
            len(indices),
            FORCE_NORM_THRESHOLD,
        )
        LOGGER.info("%s: wrote selected indices to %s", split, output_indices_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
