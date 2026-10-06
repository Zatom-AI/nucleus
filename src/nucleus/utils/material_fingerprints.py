"""Shared helpers for building and loading material fingerprints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from biotite.structure import AtomArray
from pymatgen.core import Lattice, Structure

from nucleus.utils.materials import (
    atom_array_to_pymatgen_structure,
    lattice_vectors_from_lengths_angles,
)

try:
    from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "Material fingerprint helpers require matminer. "
        "Install the material extras used by zevals."
    ) from exc


def create_material_fingerprint_featurizer() -> CrystalNNFingerprint:
    """Create the CrystalNN fingerprint featurizer used for material novelty."""
    return CrystalNNFingerprint.from_preset("ops")


def atom_array_to_structure(atom_array: AtomArray) -> Structure | None:
    """Convert a material AtomArray to a pymatgen Structure.

    Material fractional coordinates and their lattice are the authoritative
    periodic representation. ``AtomArray.coord`` may be zeroed for model
    conditioning or expressed in a different Cartesian lattice orientation,
    so it is used only when fractional annotations are unavailable.
    """
    return atom_array_to_pymatgen_structure(atom_array)


def record_to_structure(record: Any) -> Structure | None:
    """Best-effort conversion from a material dataset record to a Structure."""
    atom_array = record.get("atom_array") if isinstance(record, dict) else None
    species = None
    if atom_array is not None:
        species = np.asarray(getattr(atom_array, "element"), dtype=object)

    if species is None:
        species = np.asarray(record.get("chemical_symbols"), dtype=object)
    if species.size == 0:
        return None

    cartesian = record.get("cartesian_coordinates")
    fractional = record.get("fractional_coordinates")
    lattice_vectors = record.get("lattice_vectors")
    lattice_lengths = record.get("lattice_lengths")
    lattice_angles = record.get("lattice_angles")
    if (
        lattice_vectors is None
        and lattice_lengths is not None
        and lattice_angles is not None
    ):
        lattice_vectors = lattice_vectors_from_lengths_angles(
            lattice_lengths,
            lattice_angles,
        )

    if fractional is not None and lattice_vectors is not None:
        return _structure_from_components(
            species=species,
            lattice_vectors=np.asarray(lattice_vectors, dtype=float),
            coordinates=np.asarray(fractional, dtype=float),
            are_cartesian=False,
        )

    if cartesian is not None and lattice_vectors is not None:
        return _structure_from_components(
            species=species,
            lattice_vectors=np.asarray(lattice_vectors, dtype=float),
            coordinates=np.asarray(cartesian, dtype=float),
            are_cartesian=True,
        )

    return None


def _structure_from_components(
    *,
    species: np.ndarray,
    lattice_vectors: np.ndarray,
    coordinates: np.ndarray,
    are_cartesian: bool,
) -> Structure | None:
    """Create a Structure from explicit species, lattice, and coordinates."""
    if len(coordinates) == 0 or len(species) == 0:
        return None

    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim == 1:
        coords = coords[None, :]
    lattice_vectors = np.asarray(lattice_vectors, dtype=float)
    if (
        coords.shape != (len(species), 3)
        or lattice_vectors.shape != (3, 3)
        or not np.all(np.isfinite(coords))
        or not np.all(np.isfinite(lattice_vectors))
        or abs(float(np.linalg.det(lattice_vectors))) <= 1e-8
    ):
        return None

    try:
        return Structure(
            Lattice(lattice_vectors),
            # Ensure that two-letter symbols are properly capitalized, and that all species are strings.
            species=[str(symbol).title() for symbol in species],
            coords=coords,
            coords_are_cartesian=are_cartesian,
        )
    except Exception:
        return None


def structure_fingerprint(
    structure: Structure,
    featurizer: CrystalNNFingerprint | None = None,
) -> np.ndarray | None:
    """Compute a normalized mean-site fingerprint for a structure."""
    featurizer = featurizer or create_material_fingerprint_featurizer()
    try:
        site_fps = [featurizer.featurize(structure, i) for i in range(len(structure))]
    except Exception:
        return None

    if not site_fps:
        return None

    vec = np.asarray(site_fps, dtype=float).mean(axis=0)
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0 or not np.isfinite(norm):
        return None
    return vec / norm


def compute_structure_fingerprints(
    structures: Sequence[Structure],
    featurizer: CrystalNNFingerprint | None = None,
) -> list[np.ndarray]:
    """Compute normalized fingerprints for a sequence of structures."""
    featurizer = featurizer or create_material_fingerprint_featurizer()
    fingerprints: list[np.ndarray] = []
    for structure in structures:
        fingerprint = structure_fingerprint(structure, featurizer)
        if fingerprint is not None:
            fingerprints.append(fingerprint)
    return fingerprints


def _normalize_fingerprint_sequence(fingerprints: Any) -> list[np.ndarray]:
    """Normalize a serialized fingerprint cache into a list of float arrays."""
    if fingerprints is None:
        return []

    if isinstance(fingerprints, (list, tuple)):
        return [
            np.asarray(fingerprint, dtype=float)
            for fingerprint in fingerprints
            if fingerprint is not None
        ]

    array = np.asarray(fingerprints)
    if array.size == 0:
        return []
    if array.dtype != object:
        if array.ndim == 1:
            return [np.asarray(array, dtype=float)]
        return [np.asarray(row, dtype=float) for row in array]

    normalized: list[np.ndarray] = []
    for fingerprint in array.tolist():
        if fingerprint is not None:
            normalized.append(np.asarray(fingerprint, dtype=float))
    return normalized


def load_reference_fingerprints_from_path(
    path: str | Path | None,
) -> list[np.ndarray] | None:
    """Load precomputed fingerprints from a ``.npy`` or ``.npz`` file."""
    if path is None:
        return None

    fingerprint_path = Path(path)
    if not fingerprint_path.exists():
        raise FileNotFoundError(
            f"Reference fingerprint cache not found: {fingerprint_path}"
        )

    suffix = fingerprint_path.suffix.lower()
    if suffix == ".npy":
        return _normalize_fingerprint_sequence(
            np.load(fingerprint_path, allow_pickle=True)
        )
    if suffix == ".npz":
        with np.load(fingerprint_path, allow_pickle=True) as data:
            if "reference_fingerprints" not in data:
                raise KeyError(
                    "Expected `reference_fingerprints` in fingerprint cache archive."
                )
            return _normalize_fingerprint_sequence(data["reference_fingerprints"])

    raise ValueError(
        f"Unsupported fingerprint cache format: {fingerprint_path.suffix}. "
        "Use `.npy` or `.npz`."
    )


__all__ = [
    "atom_array_to_structure",
    "compute_structure_fingerprints",
    "create_material_fingerprint_featurizer",
    "load_reference_fingerprints_from_path",
    "structure_fingerprint",
]
