"""Utilities for lightweight periodic material support."""

from itertools import product

import numpy as np
import torch
from biotite.structure import AtomArray, AtomArrayStack

MATERIAL_ANNOTATIONS = (
    "is_material",
    "material_frac_coords",
    "material_lattice_lengths",
    "material_lattice_angles",
    "material_lattice_vectors",
)


def normalize_lattice_lengths(lengths, n_atoms: int):
    """Scale lattice lengths by the cube root of the material atom count."""
    n_atoms = max(int(n_atoms), 1)
    return np.asarray(lengths, dtype=float) / float(n_atoms) ** (1.0 / 3.0)


def unnormalize_lattice_lengths(lengths_scaled, n_atoms: int):
    """Undo length normalization used for material lattice targets."""
    n_atoms = max(int(n_atoms), 1)
    return np.asarray(lengths_scaled, dtype=float) * float(n_atoms) ** (1.0 / 3.0)


def lattice_vectors_from_lengths_angles(lengths, angles_degrees):
    """Build row-vector lattice matrix from lengths and angles in degrees."""
    a, b, c = np.asarray(lengths, dtype=float)
    alpha, beta, gamma = np.radians(np.asarray(angles_degrees, dtype=float))

    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)
    if abs(sin_gamma) < 1e-8:
        sin_gamma = 1e-8

    ax = np.array([a, 0.0, 0.0], dtype=float)
    bx = np.array([b * cos_gamma, b * sin_gamma, 0.0], dtype=float)
    cx = c * cos_beta
    cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz_sq = max(c * c - cx * cx - cy * cy, 1e-12)
    cx_vec = np.array([cx, cy, np.sqrt(cz_sq)], dtype=float)
    return np.stack([ax, bx, cx_vec], axis=0)


def lattice_volume(lattice_vectors):
    """Return absolute cell volume for row-vector lattice matrix."""
    lattice_vectors = np.asarray(lattice_vectors, dtype=float)
    return float(
        abs(
            np.dot(lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2]))
        )
    )


def fractional_to_cartesian(frac_coords, lattice_vectors):
    """Convert fractional coordinates to Cartesian coordinates."""
    if torch.is_tensor(frac_coords):
        lattice_vectors = torch.as_tensor(
            lattice_vectors,
            device=frac_coords.device,
            dtype=frac_coords.dtype,
        )
        return frac_coords @ lattice_vectors
    return np.asarray(frac_coords, dtype=float) @ np.asarray(
        lattice_vectors, dtype=float
    )


def cartesian_to_fractional(coords, lattice_vectors):
    """Convert Cartesian coordinates to wrapped fractional coordinates."""
    if torch.is_tensor(coords):
        lattice_vectors = torch.as_tensor(
            lattice_vectors,
            device=coords.device,
            dtype=coords.dtype,
        )
        # torch.linalg.inv does not support low-precision input dtypes.
        inverse_dtype = (
            torch.float32
            if coords.dtype in (torch.float16, torch.bfloat16)
            else coords.dtype
        )
        fractional_coordinates = coords.to(inverse_dtype) @ torch.linalg.inv(
            lattice_vectors.to(inverse_dtype)
        )
        return (fractional_coordinates % 1.0).to(coords.dtype)
    frac = np.asarray(coords, dtype=float) @ np.linalg.inv(
        np.asarray(lattice_vectors, dtype=float)
    )
    return frac % 1.0


def wrap_cartesian_coordinates(coords, lattice_vectors):
    """Wrap Cartesian coordinates into the unit cell defined by the lattice."""
    frac = cartesian_to_fractional(coords, lattice_vectors)
    return fractional_to_cartesian(frac, lattice_vectors)


def _periodic_alignment_rotation_starts(coordinates, target, lattice_vectors):
    """Match a short, noncollinear atom triplet to candidate lattice images."""
    from ase.geometry import find_mic, minkowski_reduce

    squared_distances = np.sum((coordinates[:, None] - coordinates[None]) ** 2, axis=-1)
    squared_distances[squared_distances < 1e-12] = np.inf
    if not np.any(np.isfinite(squared_distances)):
        return []
    anchor, first = np.unravel_index(
        np.argmin(squared_distances), squared_distances.shape
    )
    vectors = coordinates - coordinates[anchor]
    lengths = np.linalg.norm(vectors, axis=-1)
    sin_angle = np.linalg.norm(np.cross(vectors[first], vectors), axis=-1) / np.maximum(
        lengths[first] * lengths, 1e-12
    )
    eligible = sin_angle > 0.25
    if not np.any(eligible):
        eligible = sin_angle > 1e-6
    indices = [first]
    if np.any(eligible):
        indices.append(int(np.argmin(np.where(eligible, lengths, np.inf))))

    reduced_cell, _ = minkowski_reduce(lattice_vectors)
    inverse = np.linalg.inv(reduced_cell)

    def image_vectors(index):
        displacement, distance = find_mic(
            target[index] - target[anchor], lattice_vectors, pbc=True
        )
        # Allow head disagreement, while always including the nearest image.
        radius = max(lengths[index] * 1.2 + 0.5, float(distance) + 1e-6)
        fractional = displacement @ inverse
        bound = radius * np.linalg.norm(inverse, axis=0)
        lower = np.ceil(-fractional - bound).astype(int)
        upper = np.floor(-fractional + bound).astype(int)
        # Collapsed predicted cells can contain enormous numbers of images.
        # Keep the search bounded and fall back to the rotation starts below.
        if np.prod((upper - lower + 1).astype(float)) > 32768:
            return None
        shifts = np.array(
            list(product(*(range(a, b + 1) for a, b in zip(lower, upper, strict=True))))
        )
        images = displacement + shifts @ reduced_cell
        images = images[np.linalg.norm(images, axis=1) <= radius]
        errors = (np.linalg.norm(images, axis=1) - lengths[index]) ** 2
        return images[np.argsort(errors, kind="stable")[:64]]

    images_first = image_vectors(first)
    if images_first is None:
        return []
    if len(indices) == 1:
        targets = np.stack([np.zeros_like(images_first), images_first], axis=1)
    else:
        second = indices[1]
        images_second = image_vectors(second)
        if images_second is None:
            return []
        cost = (np.linalg.norm(images_first, axis=-1)[:, None] - lengths[first]) ** 2
        cost = (
            cost
            + (np.linalg.norm(images_second, axis=-1)[None, :] - lengths[second]) ** 2
        )
        cost = (
            cost
            + (
                np.linalg.norm(images_first[:, None] - images_second[None], axis=-1)
                - np.linalg.norm(vectors[first] - vectors[second])
            )
            ** 2
        )
        a, b = np.unravel_index(
            np.argsort(cost, axis=None, kind="stable")[:64], cost.shape
        )
        targets = np.stack(
            [np.zeros_like(images_first[a]), images_first[a], images_second[b]], axis=1
        )
    source = coordinates[[anchor, *indices]]
    source = source - source.mean(axis=0)
    starts = []
    for points in targets:
        u, _, vt = np.linalg.svd(source.T @ (points - points.mean(axis=0)))
        if np.linalg.det(u @ vt) < 0:
            u[:, -1] *= -1
        starts.append(u @ vt)
    return starts


def fit_periodic_rigid_alignment(coordinates, fractional_coordinates, lattice_vectors):
    """Fit a proper rigid transform from Cartesian predictions to their cell frame.

    Atom correspondence is fixed. Alternating nearest periodic images and Kabsch
    fits avoids treating wrapped fractional coordinates as an unwrapped molecule.
    Matched triplets of periodic images initialize rotations to avoid the local
    minima of rotation-only starts. The returned RMS residual (Angstrom) measures
    agreement between the heads, not structural quality.
    Inconsistent or geometrically degenerate heads cannot determine an exact or
    unique frame. No scaling, reflection, or per-atom deformation is applied.

    Args:
        coordinates: Conditional Cartesian predictions, shape [L, 3].
        fractional_coordinates: Fractional predictions from the same forward pass,
            shape [L, 3], in the basis of ``lattice_vectors``.
        lattice_vectors: Row-vector cell matrix, shape [3, 3].

    Returns:
        Rotation, translation, and periodic RMS residual. Apply the transform as
        ``guided_coordinates @ rotation + translation`` to preserve guidance.
    """
    from ase.geometry import find_mic
    from scipy.spatial.transform import Rotation

    coordinates = np.asarray(coordinates, dtype=float)
    fractional_coordinates = np.asarray(fractional_coordinates, dtype=float)
    lattice_vectors = np.asarray(lattice_vectors, dtype=float)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 3
        or len(coordinates) == 0
        or fractional_coordinates.shape != coordinates.shape
        or lattice_vectors.shape != (3, 3)
        or not all(
            np.all(np.isfinite(value))
            for value in (coordinates, fractional_coordinates, lattice_vectors)
        )
        or np.linalg.det(lattice_vectors) <= 0
    ):
        raise ValueError(
            "Periodic alignment requires finite paired atoms and a positive-volume cell."
        )

    target = (fractional_coordinates % 1.0) @ lattice_vectors
    center = coordinates.mean(axis=0)
    centered = coordinates - center

    def fit(images):
        target_center = images.mean(axis=0)
        u, _, vt = np.linalg.svd(centered.T @ (images - target_center))
        if np.linalg.det(u @ vt) < 0:
            u[:, -1] *= -1
        rotation = u @ vt
        return rotation, target_center - center @ rotation

    initial_rotation, _ = fit(target)
    starts = [
        *_periodic_alignment_rotation_starts(coordinates, target, lattice_vectors),
        np.eye(3),
        initial_rotation,
        *Rotation.random(48, random_state=43).as_matrix(),
    ]
    best_rmsd = np.inf
    for rotation in starts:
        translation = target[0] - coordinates[0] @ rotation
        previous_rmsd = np.inf
        for _ in range(40):
            residual = coordinates @ rotation + translation - target
            minimum_image, _ = find_mic(residual, lattice_vectors, pbc=True)
            rotation, translation = fit(target + residual - minimum_image)
            _, distances = find_mic(
                coordinates @ rotation + translation - target, lattice_vectors, pbc=True
            )
            rmsd = float(np.sqrt(np.mean(distances**2)))
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_rotation, best_translation = rotation, translation
            if abs(previous_rmsd - rmsd) < 1e-9:
                break
            previous_rmsd = rmsd
        if best_rmsd < 1e-7:
            break
    return best_rotation, best_translation, best_rmsd


def get_material_mask(atom_array: AtomArray) -> np.ndarray:
    """Return atom-level material mask, defaulting to all False."""
    if "is_material" not in atom_array.get_annotation_categories():
        return np.zeros(atom_array.array_length(), dtype=bool)
    return np.asarray(atom_array.is_material, dtype=bool)


def has_material(atom_array: AtomArray | AtomArrayStack) -> bool:
    """Return whether an atom array or stack carries material atoms."""
    if isinstance(atom_array, AtomArrayStack):
        atom_array = atom_array[0]
    return bool(np.any(get_material_mask(atom_array)))


def annotate_material_properties(
    atom_array: AtomArray,
    *,
    fractional_coordinates,
    lattice_lengths,
    lattice_angles,
    lattice_vectors=None,
    is_material=None,
) -> AtomArray:
    """Attach periodic material annotations to an atom array."""
    n_atoms = atom_array.array_length()
    if lattice_vectors is None:
        lattice_vectors = lattice_vectors_from_lengths_angles(
            lattice_lengths,
            lattice_angles,
        )

    fractional_coordinates = np.asarray(fractional_coordinates, dtype=float) % 1.0
    lattice_lengths = np.asarray(lattice_lengths, dtype=float)
    lattice_angles = np.asarray(lattice_angles, dtype=float)
    lattice_vectors = np.asarray(lattice_vectors, dtype=float)
    if fractional_coordinates.shape != (n_atoms, 3):
        raise ValueError(
            "Material fractional coordinates must have shape "
            f"({n_atoms}, 3), got {fractional_coordinates.shape}."
        )

    if is_material is None:
        is_material = np.ones(n_atoms, dtype=bool)
    else:
        is_material = np.asarray(is_material, dtype=bool)

    atom_array.set_annotation("is_material", is_material)
    atom_array.set_annotation("material_frac_coords", fractional_coordinates)
    atom_array.set_annotation(
        "material_lattice_lengths",
        np.repeat(lattice_lengths[None, :], n_atoms, axis=0),
    )
    atom_array.set_annotation(
        "material_lattice_angles",
        np.repeat(lattice_angles[None, :], n_atoms, axis=0),
    )
    atom_array.set_annotation(
        "material_lattice_vectors",
        np.repeat(lattice_vectors[None, :, :], n_atoms, axis=0),
    )
    atom_array.box = lattice_vectors.copy()
    return atom_array


def copy_material_annotations(source: AtomArray, target: AtomArray) -> AtomArray:
    """Copy material annotations when reparsing preserves atom order and count."""
    if source.array_length() != target.array_length():
        return target
    for annotation in MATERIAL_ANNOTATIONS:
        if annotation in source.get_annotation_categories():
            target.set_annotation(annotation, source.get_annotation(annotation).copy())
    return target


def get_material_lattice(atom_array: AtomArray):
    """Return material lattice lengths, angles, vectors and mask, or None."""
    mask = get_material_mask(atom_array)
    if not np.any(mask):
        return None
    categories = atom_array.get_annotation_categories()
    if (
        "material_lattice_lengths" not in categories
        or "material_lattice_angles" not in categories
    ):
        return None
    first = np.flatnonzero(mask)[0]
    lengths = np.asarray(atom_array.material_lattice_lengths[first], dtype=float)
    angles = np.asarray(atom_array.material_lattice_angles[first], dtype=float)
    if "material_lattice_vectors" in categories:
        vectors = np.asarray(atom_array.material_lattice_vectors[first], dtype=float)
    else:
        vectors = lattice_vectors_from_lengths_angles(lengths, angles)
    return lengths, angles, vectors, mask


def atom_array_to_pymatgen_structure(atom_array: AtomArray):
    """Convert an annotated periodic AtomArray to a pymatgen Structure.

    Fractional coordinates are authoritative when present. Cartesian
    coordinates are only a compatibility fallback for older AtomArrays.
    """
    from pymatgen.core import Lattice, Structure

    lattice = get_material_lattice(atom_array)
    if lattice is None:
        return None
    _, _, vectors, mask = lattice
    if not np.any(mask):
        return None

    categories = set(atom_array.get_annotation_categories())
    if "material_frac_coords" in categories:
        coordinates = np.asarray(atom_array.material_frac_coords, dtype=float)[mask]
        are_cartesian = False
    else:
        coordinates = np.asarray(atom_array.coord, dtype=float)[mask]
        are_cartesian = True
    species = np.asarray(atom_array.element, dtype=object)[mask]
    vectors = np.asarray(vectors, dtype=float)
    if (
        coordinates.shape != (len(species), 3)
        or vectors.shape != (3, 3)
        or not np.all(np.isfinite(coordinates))
        or not np.all(np.isfinite(vectors))
        or abs(float(np.linalg.det(vectors))) <= 1e-8
    ):
        return None
    try:
        return Structure(
            Lattice(vectors),
            [str(symbol).title() for symbol in species],
            coordinates,
            coords_are_cartesian=are_cartesian,
        )
    except Exception:
        return None


def add_fractional_cif_fields(atom_array: AtomArray) -> AtomArray:
    """Add atom_site fractional-coordinate annotations for CIF export."""
    if "material_frac_coords" not in atom_array.get_annotation_categories():
        return atom_array
    frac = np.asarray(atom_array.material_frac_coords, dtype=float) % 1.0
    atom_array.set_annotation("fract_x", frac[:, 0])
    atom_array.set_annotation("fract_y", frac[:, 1])
    atom_array.set_annotation("fract_z", frac[:, 2])
    return atom_array


def material_cif_extra_categories(atom_array: AtomArray | AtomArrayStack):
    """Build CIF cell categories from material lattice annotations."""
    if isinstance(atom_array, AtomArrayStack):
        atom_array = atom_array[0]

    lattice = get_material_lattice(atom_array)
    if lattice is None:
        return None
    lengths, angles, vectors, _ = lattice
    return {
        "cell": {
            "length_a": float(lengths[0]),
            "length_b": float(lengths[1]),
            "length_c": float(lengths[2]),
            "angle_alpha": float(angles[0]),
            "angle_beta": float(angles[1]),
            "angle_gamma": float(angles[2]),
            "volume": lattice_volume(vectors),
        },
        "space_group": {
            "name_H-M_alt": "P 1",
            "IT_number": 1,
        },
        "symmetry": {
            "space_group_name_H-M": "P 1",
            "Int_Tables_number": 1,
        },
    }
