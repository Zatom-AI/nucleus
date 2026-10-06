#!/usr/bin/env python3
"""Fit versioned OMol25/OMat24 energy and force normalization statistics.

The fitter keeps only fixed-size sufficient statistics. It never materializes a
system-by-element design matrix, so its memory use is independent of split size.
Callers may select records with explicit index manifests or opt into every record
under training-only dataset roots with ``--all-records``.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from dotenv import load_dotenv

from nucleus.utils.atomistic_properties import (
    AtomisticPropertyStatistics,
    DomainPropertyStatistics,
)

LOGGER = logging.getLogger(__name__)
MAX_ATOMIC_NUMBER = 127
REPO_ROOT = Path(__file__).resolve().parents[1]

# Match the other dataset-preparation scripts: values exported by the caller take
# precedence, while a repository-local .env remains convenient for local runs.
load_dotenv(REPO_ROOT / ".env", override=False)


def _shard_paths(data_dir: Path) -> list[Path]:
    paths = sorted(
        path for path in data_dir.rglob("*.aselmdb") if not path.name.startswith("._")
    )
    if not paths:
        raise FileNotFoundError(f"No *.aselmdb shards found under {data_dir}.")
    return paths


def _shard_manifest_sha256(data_dir: Path, shard_paths: Sequence[Path]) -> str:
    """Hash stable shard names and sizes using the v1 manifest algorithm.

    The digest deliberately excludes the machine-specific absolute dataset root.
    It is not a full content checksum: hashing every byte of multi-terabyte training
    data would make statistics startup prohibitively expensive.
    """
    digest = hashlib.sha256()
    for shard_path in shard_paths:
        relative_path = shard_path.relative_to(data_dir).as_posix()
        digest.update(f"{relative_path}\t{shard_path.stat().st_size}\n".encode())
    return digest.hexdigest()


def _build_dataset(shard_paths: Sequence[Path], *, domain: str) -> Any:
    try:
        from atomworks.ml.datasets import ASELMDBDataset
    except ImportError as exc:
        raise ImportError(
            "atomworks is required to fit ASE-LMDB property statistics."
        ) from exc

    return ASELMDBDataset(
        paths=list(shard_paths),
        name=f"{domain}_property_statistics",
        return_type="atoms",
        readonly=True,
        readahead=False,
        include_data=True,
    )


def _iter_indices(path: Path, *, max_rows: int | None) -> Iterator[int]:
    yielded = 0
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", maxsplit=1)[0].strip()
            if not line:
                continue
            yield int(line)
            yielded += 1
            if max_rows is not None and yielded >= max_rows:
                return
    if yielded == 0:
        raise ValueError(f"No indices were found in {path}.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_data_dir(domain: str, environ: Mapping[str, str]) -> Path:
    """Resolve a domain training root from its environment variable or repo data."""
    env_key = f"{domain.upper()}_DATA_DIR"
    env_value = environ.get(env_key)
    if env_value:
        return Path(env_value).expanduser()
    if domain == "omol25":
        return REPO_ROOT / "data" / "omol25" / "raw" / "train_4M"
    return REPO_ROOT / "data" / "omat24" / "raw" / "train"


def _calculator_value(atoms: Any, key: str, *, row_index: int) -> Any:
    calculator = getattr(atoms, "calc", None)
    results = getattr(calculator, "results", {}) if calculator is not None else {}
    if key in results:
        return results[key]
    try:
        if key == "energy":
            return atoms.get_potential_energy()
        if key == "forces":
            return atoms.get_forces()
    except Exception as exc:
        raise ValueError(f"Row {row_index} does not contain {key}.") from exc
    raise ValueError(f"Unsupported calculator result {key!r}.")


@dataclass
class StreamingPropertyMoments:
    """Fixed-memory sufficient statistics for elemental-reference regression."""

    max_atomic_number: int = MAX_ATOMIC_NUMBER
    energy_normal_matrix: np.ndarray = field(init=False)
    energy_rhs: np.ndarray = field(init=False)
    per_atom_composition_sum: np.ndarray = field(init=False)
    per_atom_composition_second_moment: np.ndarray = field(init=False)
    per_atom_energy_composition_moment: np.ndarray = field(init=False)
    per_atom_energy_sum: float = 0.0
    per_atom_energy_squared_sum: float = 0.0
    force_squared_sum: float = 0.0
    n_force_components: int = 0
    n_systems: int = 0
    n_atoms: int = 0

    def __post_init__(self) -> None:
        n_elements = self.max_atomic_number + 1
        self.energy_normal_matrix = np.zeros((n_elements, n_elements), dtype=np.float64)
        self.energy_rhs = np.zeros(n_elements, dtype=np.float64)
        self.per_atom_composition_sum = np.zeros(n_elements, dtype=np.float64)
        self.per_atom_composition_second_moment = np.zeros(
            (n_elements, n_elements), dtype=np.float64
        )
        self.per_atom_energy_composition_moment = np.zeros(n_elements, dtype=np.float64)

    def update(
        self,
        atomic_numbers: np.ndarray,
        energy_eV: float,
        forces_eV_per_angstrom: np.ndarray,
    ) -> None:
        """Accumulate one labeled structure."""
        atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
        forces = np.asarray(forces_eV_per_angstrom, dtype=np.float64)
        if atomic_numbers.ndim != 1 or atomic_numbers.size == 0:
            raise ValueError("atomic_numbers must be a non-empty vector.")
        if np.any(atomic_numbers <= 0) or np.any(
            atomic_numbers > self.max_atomic_number
        ):
            raise ValueError(
                f"Atomic numbers must be in [1, {self.max_atomic_number}]."
            )
        if forces.shape != (atomic_numbers.size, 3):
            raise ValueError(
                f"Expected forces shape {(atomic_numbers.size, 3)}, got {forces.shape}."
            )
        if not np.isfinite(energy_eV) or not np.isfinite(forces).all():
            raise ValueError("Energy and forces must be finite.")

        composition = np.bincount(
            atomic_numbers, minlength=self.max_atomic_number + 1
        ).astype(np.float64)
        n_atoms = atomic_numbers.size
        per_atom_composition = composition / n_atoms
        per_atom_energy = float(energy_eV) / n_atoms

        self.energy_normal_matrix += np.outer(composition, composition)
        self.energy_rhs += composition * energy_eV
        self.per_atom_composition_sum += per_atom_composition
        self.per_atom_composition_second_moment += np.outer(
            per_atom_composition, per_atom_composition
        )
        self.per_atom_energy_composition_moment += (
            per_atom_energy * per_atom_composition
        )
        self.per_atom_energy_sum += per_atom_energy
        self.per_atom_energy_squared_sum += per_atom_energy**2
        self.force_squared_sum += float(np.square(forces).sum())
        self.n_force_components += forces.size
        self.n_systems += 1
        self.n_atoms += n_atoms

    def finalize(
        self,
        *,
        name: str,
        domain_id: int,
        source_split_ids: tuple[str, ...],
    ) -> DomainPropertyStatistics:
        """Solve elemental references and analytically recover residual moments."""
        if self.n_systems < 2:
            raise ValueError("At least two labeled systems are required.")
        observed = np.diag(self.energy_normal_matrix) > 0
        observed[0] = False
        observed_indices = np.flatnonzero(observed)
        normal_matrix = self.energy_normal_matrix[np.ix_(observed, observed)]
        rhs = self.energy_rhs[observed]
        references_observed, _, rank, _ = np.linalg.lstsq(
            normal_matrix, rhs, rcond=None
        )
        if rank < observed_indices.size:
            LOGGER.warning(
                "%s elemental-reference normal equations are rank deficient "
                "(%d/%d); storing the minimum-norm solution.",
                name,
                rank,
                observed_indices.size,
            )
        references = np.zeros(self.max_atomic_number + 1, dtype=np.float64)
        references[observed] = references_observed

        residual_sum = self.per_atom_energy_sum - float(
            self.per_atom_composition_sum @ references
        )
        residual_squared_sum = (
            self.per_atom_energy_squared_sum
            - 2 * float(self.per_atom_energy_composition_moment @ references)
            + float(references @ self.per_atom_composition_second_moment @ references)
        )
        residual_mean = residual_sum / self.n_systems
        residual_variance = max(
            0.0,
            residual_squared_sum / self.n_systems - residual_mean**2,
        )
        residual_scale = float(np.sqrt(residual_variance))
        force_rms = float(
            np.sqrt(self.force_squared_sum / max(1, self.n_force_components))
        )
        if residual_scale <= 0 or force_rms <= 0:
            raise ValueError(
                f"Degenerate {name} statistics: residual_scale={residual_scale}, "
                f"force_rms={force_rms}."
            )

        return DomainPropertyStatistics(
            name=name,
            domain_id=domain_id,
            source_split_ids=source_split_ids,
            elemental_references_eV={
                int(z): float(references[z]) for z in observed_indices
            },
            residual_energy_per_atom_mean_eV=residual_mean,
            residual_energy_per_atom_scale_eV=residual_scale,
            force_rms_eV_per_angstrom=force_rms,
            n_systems=self.n_systems,
            n_atoms=self.n_atoms,
        )


def _fit_domain(
    *,
    name: str,
    domain_id: int,
    data_dir: Path,
    indices_path: Path | None,
    all_records: bool,
    split_id: str,
    max_rows: int | None,
    progress_every: int,
) -> DomainPropertyStatistics:
    if all_records == (indices_path is not None):
        raise ValueError(
            "Select exactly one record source: an index manifest or all records."
        )

    shard_paths = _shard_paths(data_dir)
    dataset = _build_dataset(shard_paths, domain=name)
    accumulator = StreamingPropertyMoments()
    try:
        dataset_record_count = len(dataset)
        if dataset_record_count <= 0:
            raise ValueError(f"{name} dataset contains no records under {data_dir}.")

        if all_records:
            selected_record_count = (
                dataset_record_count
                if max_rows is None
                else min(dataset_record_count, max_rows)
            )
            selected_indices = iter(range(selected_record_count))
        else:
            assert indices_path is not None
            selected_indices = _iter_indices(indices_path, max_rows=max_rows)

        for row_number, row_index in enumerate(selected_indices, start=1):
            if row_index < 0 or row_index >= dataset_record_count:
                raise IndexError(
                    f"{name} index {row_index} is outside [0, "
                    f"{dataset_record_count})."
                )
            atoms = dataset[row_index]
            accumulator.update(
                np.asarray(atoms.numbers),
                float(_calculator_value(atoms, "energy", row_index=row_index)),
                np.asarray(_calculator_value(atoms, "forces", row_index=row_index)),
            )
            if progress_every and row_number % progress_every == 0:
                LOGGER.info("%s: processed %d systems", name, row_number)
    finally:
        close_dataset = getattr(dataset, "close", None)
        if close_dataset is not None:
            close_dataset()

    selection = "all_records" if all_records else "index_manifest"
    provenance = [
        split_id,
        f"selection:{selection}",
        f"dataset_record_count:{dataset_record_count}",
        f"selected_record_count:{accumulator.n_systems}",
        f"shard_count:{len(shard_paths)}",
        f"shard_manifest_v1_sha256:{_shard_manifest_sha256(data_dir, shard_paths)}",
    ]
    if indices_path is not None:
        provenance.append(f"indices_sha256:{_sha256(indices_path)}")
    if max_rows is not None:
        provenance.append(f"row_limit:{max_rows}")
    return accumulator.finalize(
        name=name,
        domain_id=domain_id,
        source_split_ids=tuple(provenance),
    )


def _parse_args(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> argparse.Namespace:
    environ = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(description=__doc__)
    for domain in ("omol25", "omat24"):
        env_key = f"{domain.upper()}_DATA_DIR"
        parser.add_argument(
            f"--{domain}-data-dir",
            type=Path,
            default=_default_data_dir(domain, environ),
            help=(
                f"Directory containing {domain.upper()} training *.aselmdb "
                f"shards. Defaults to ${env_key}, then the repository data path."
            ),
        )
        parser.add_argument(
            f"--{domain}-indices",
            type=Path,
            default=None,
            help=(
                f"Optional training-only index manifest for {domain.upper()}. "
                "Omit only when using --all-records."
            ),
        )
        parser.add_argument(
            f"--{domain}-split-id",
            required=True,
            help=f"Stable source training-split identifier for {domain.upper()}.",
        )
    parser.add_argument(
        "--all-records",
        action="store_true",
        help=(
            "Use every record under both dataset roots. This is an explicit "
            "acknowledgement that those roots contain training partitions only."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional debug-only row limit applied independently to each domain.",
    )
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args(argv)
    indices = (args.omol25_indices, args.omat24_indices)
    if args.all_records and any(path is not None for path in indices):
        parser.error(
            "--all-records cannot be combined with --omol25-indices or "
            "--omat24-indices."
        )
    if not args.all_records and any(path is None for path in indices):
        parser.error(
            "Provide both index manifests, or pass --all-records to explicitly "
            "use both training-only dataset roots in full."
        )
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive when provided.")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative.")
    for domain in ("omol25", "omat24"):
        split_id = getattr(args, f"{domain}_split_id")
        if not split_id.strip():
            parser.error(f"--{domain}-split-id must not be blank.")
        setattr(args, f"{domain}_split_id", split_id.strip())
        data_dir = getattr(args, f"{domain}_data_dir").expanduser()
        setattr(args, f"{domain}_data_dir", data_dir)
        if not data_dir.is_dir():
            parser.error(
                f"--{domain}-data-dir is not a directory: {data_dir}. Set "
                f"{domain.upper()}_DATA_DIR or pass the CLI option explicitly."
            )
        indices_path = getattr(args, f"{domain}_indices")
        if indices_path is not None:
            indices_path = indices_path.expanduser()
            setattr(args, f"{domain}_indices", indices_path)
            if not indices_path.is_file():
                parser.error(f"--{domain}-indices is not a file: {indices_path}.")
    args.output = args.output.expanduser()
    return args


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    domains = []
    for domain_id, name in enumerate(("omol25", "omat24")):
        domains.append(
            _fit_domain(
                name=name,
                domain_id=domain_id,
                data_dir=getattr(args, f"{name}_data_dir"),
                indices_path=getattr(args, f"{name}_indices"),
                all_records=args.all_records,
                split_id=getattr(args, f"{name}_split_id"),
                max_rows=args.max_rows,
                progress_every=args.progress_every,
            )
        )

    statistics = AtomisticPropertyStatistics(domains=tuple(domains))
    statistics.save(args.output)
    LOGGER.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
