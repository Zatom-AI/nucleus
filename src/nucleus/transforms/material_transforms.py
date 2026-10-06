"""Material-specific transforms layered onto the RFD3 input pipeline."""

import numpy as np
import torch
from atomworks.ml.transforms._checks import (
    check_contains_keys,
)
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.geometry import (
    apply_batched_rigid,
    get_random_rigid,
    get_random_rots,
    masked_center,
)
from atomworks.ml.utils.token import get_token_starts

try:
    from rfd3na.constants import VIRTUAL_ATOM_ELEMENT_NAME
    from rfd3na.transforms.design_transforms import MotifCenterRandomAugmentation
except ImportError:
    raise ImportError(
        "Could not import RFD3NA design transforms. Please ensure that the `rc-foundry[all]` package is installed and up to date."
    )
from nucleus.utils.materials import (
    annotate_material_properties,
    cartesian_to_fractional,
    get_material_lattice,
    get_material_mask,
    normalize_lattice_lengths,
)


def _synchronize_motif_positions_with_coordinates(data: dict) -> None:
    """Put fixed-coordinate motif features in the augmented Cartesian frame.

    ``motif_pos`` has no diffusion-view axis, while clean coordinates have shape
    ``[D, L, 3]``.  Store the first view as its representative frame.  The views
    differ only by rigid transforms, so this preserves the pair distances used by
    the motif distogram for every view while avoiding a stale canonical-frame
    coordinate feature.

    Non-fixed motif positions are left unchanged because the model masks them out
    of the fixed-coordinate distogram.
    """
    feats = data.get("feats", {})
    motif_positions = feats.get("motif_pos")
    if motif_positions is None:
        return

    coordinates = data["coord_atom_lvl_to_be_noised"]
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError(
            "Motif-position synchronization requires coordinates with shape "
            f"[D, L, 3], got {coordinates.shape}."
        )
    if motif_positions.shape != coordinates.shape[1:]:
        raise ValueError(
            "Motif positions must align with the coordinate atom dimension and "
            f"have shape {coordinates.shape[1:]}, got {motif_positions.shape}."
        )

    fixed_coordinate_mask = feats.get("is_motif_atom_with_fixed_coord")
    if fixed_coordinate_mask is None:
        atom_array = data["atom_array"]
        categories = atom_array.get_annotation_categories()
        if "is_motif_atom_with_fixed_coord" not in categories:
            return
        fixed_coordinate_mask = atom_array.is_motif_atom_with_fixed_coord

    fixed_coordinate_mask = torch.as_tensor(
        fixed_coordinate_mask,
        device=coordinates.device,
        dtype=torch.bool,
    )
    if fixed_coordinate_mask.shape != coordinates.shape[1:2]:
        raise ValueError(
            "The fixed-coordinate mask must align with the coordinate atom "
            f"dimension and have shape {coordinates.shape[1:2]}, got "
            f"{fixed_coordinate_mask.shape}."
        )

    motif_positions = torch.as_tensor(
        motif_positions,
        device=coordinates.device,
        dtype=coordinates.dtype,
    ).clone()
    first_view_coordinates = coordinates[0]
    finite_fixed_mask = fixed_coordinate_mask & torch.isfinite(
        first_view_coordinates
    ).all(dim=-1)
    motif_positions[finite_fixed_mask] = first_view_coordinates[finite_fixed_mask]
    feats["motif_pos"] = motif_positions


def center_periodic_material_coordinates(
    data: dict, lattice_vectors: torch.Tensor | np.ndarray
) -> dict:
    """Zero-center a periodic system and synchronize fractional targets.

    The center is the unweighted centroid of the finite material atoms in each
    diffusion view.  All coordinates in the view are translated by the same
    amount so auxiliary atoms retain their relative positions.  Fractional
    material targets are then recomputed modulo the corresponding (possibly
    rotated) lattice.

    Args:
        data: Transform data containing diffusion-batched Cartesian coordinates.
        lattice_vectors: One lattice ``[3, 3]`` or one per view ``[D, 3, 3]``.

    Returns:
        The input mapping with centered coordinates and synchronized fractional
        targets.
    """
    coordinates = data["coord_atom_lvl_to_be_noised"]
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError(
            "Periodic material centering requires coordinates with shape "
            f"[D, L, 3], got {coordinates.shape}."
        )

    device = coordinates.device
    dtype = coordinates.dtype
    material_mask = torch.as_tensor(
        get_material_mask(data["atom_array"]), device=device, dtype=torch.bool
    )
    if material_mask.shape != (coordinates.shape[1],):
        raise ValueError(
            "Material mask must align with the coordinate atom dimension "
            f"L={coordinates.shape[1]}, got {material_mask.shape}."
        )

    finite_coordinate_mask = torch.isfinite(coordinates).all(dim=-1)
    center_mask = material_mask[None] & finite_coordinate_mask
    center_weights = center_mask.to(dtype=dtype)[..., None]
    valid_counts = center_weights.sum(dim=1, keepdim=True)
    centers = (torch.nan_to_num(coordinates) * center_weights).sum(
        dim=1, keepdim=True
    ) / valid_counts.clamp_min(1)
    has_valid_material_atoms = valid_counts > 0
    centered_coordinates = torch.where(
        has_valid_material_atoms,
        coordinates - centers,
        coordinates,
    )
    data["coord_atom_lvl_to_be_noised"] = centered_coordinates
    _synchronize_motif_positions_with_coordinates(data)

    fractional_targets = data.get("ground_truth", {}).get("fractional_coord_gt_L")
    if fractional_targets is None:
        return data
    fractional_targets = fractional_targets.to(device=device, dtype=dtype)
    if fractional_targets.ndim == 2:
        if fractional_targets.shape != coordinates.shape[1:]:
            raise ValueError(
                "Static fractional targets must have shape [L, 3], got "
                f"{fractional_targets.shape}."
            )
        fractional_targets = fractional_targets.unsqueeze(0).expand(
            coordinates.shape[0], -1, -1
        )
    elif fractional_targets.shape != coordinates.shape:
        raise ValueError(
            "Per-view fractional targets must have shape [D, L, 3], got "
            f"{fractional_targets.shape}."
        )
    fractional_targets = fractional_targets.clone()

    lattice_vectors = torch.as_tensor(lattice_vectors, device=device, dtype=dtype)
    if lattice_vectors.ndim == 2:
        lattice_vectors = lattice_vectors.unsqueeze(0).expand(
            coordinates.shape[0], -1, -1
        )
    if lattice_vectors.shape != (coordinates.shape[0], 3, 3):
        raise ValueError(
            "Material lattice vectors must have shape [3, 3] or [D, 3, 3], got "
            f"{lattice_vectors.shape}."
        )
    safe_coordinates = torch.where(
        center_mask[..., None],
        centered_coordinates,
        torch.zeros_like(centered_coordinates),
    )
    centered_fractional_coordinates = cartesian_to_fractional(
        safe_coordinates, lattice_vectors
    )
    fractional_targets = torch.where(
        center_mask[..., None],
        centered_fractional_coordinates,
        fractional_targets,
    )
    data["ground_truth"]["fractional_coord_gt_L"] = fractional_targets
    return data


class AddMaterialCrystalFeatures(Transform):
    """Attach periodic crystal fields from ASE/CIF loaders to the AtomArray."""

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        if "fractional_coordinates" in data:
            atom_array = annotate_material_properties(
                atom_array,
                fractional_coordinates=data["fractional_coordinates"],
                lattice_lengths=data["lattice_lengths"],
                lattice_angles=data["lattice_angles"],
                lattice_vectors=data.get("lattice_vectors"),
            )

        data["atom_array"] = atom_array
        return data


class AddMaterialPredictionTargets(Transform):
    """Add optional material fractional-coordinate and lattice targets."""

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        n_atoms = atom_array.array_length()
        token_starts = get_token_starts(atom_array)
        n_tokens = len(token_starts)

        material_atom_mask = get_material_mask(atom_array)
        material_token_mask = material_atom_mask[token_starts]

        categories = atom_array.get_annotation_categories()
        fixed_coord = (
            np.asarray(atom_array.is_motif_atom_with_fixed_coord, dtype=bool)
            if "is_motif_atom_with_fixed_coord" in categories
            else np.zeros(n_atoms, dtype=bool)
        )
        coord_generation_mask = material_atom_mask & ~fixed_coord
        lattice_generation_mask = material_token_mask & ~fixed_coord[token_starts]
        coord_valid_mask = coord_generation_mask.copy()
        lattice_valid_mask = lattice_generation_mask.copy()

        frac_gt = np.zeros((n_atoms, 3), dtype=np.float32)
        if "material_frac_coords" in categories:
            frac_gt[material_atom_mask] = np.asarray(
                atom_array.material_frac_coords[material_atom_mask],
                dtype=np.float32,
            )
        else:
            coord_valid_mask[:] = False

        lengths_gt = np.zeros((n_tokens, 3), dtype=np.float32)
        angles_gt = np.zeros((n_tokens, 3), dtype=np.float32)
        lattice = get_material_lattice(atom_array)
        if lattice is not None and np.any(material_token_mask):
            lengths, angles, _, _ = lattice
            n_material_tokens = int(material_token_mask.sum())
            lengths_scaled = normalize_lattice_lengths(lengths, n_material_tokens)
            angles_radians = np.radians(angles)
            lengths_gt[material_token_mask] = lengths_scaled.astype(np.float32)
            angles_gt[material_token_mask] = angles_radians.astype(np.float32)
        else:
            lattice_valid_mask[:] = False

        data.setdefault("ground_truth", {})
        data["ground_truth"]["fractional_coord_gt_L"] = torch.from_numpy(frac_gt)
        data["ground_truth"]["fractional_coord_valid_mask_L"] = torch.from_numpy(
            coord_valid_mask
        )
        data["ground_truth"]["lattice_lengths_scaled_gt_I"] = torch.from_numpy(
            lengths_gt
        )
        data["ground_truth"]["lattice_angles_radians_gt_I"] = torch.from_numpy(
            angles_gt
        )
        data["ground_truth"]["lattice_valid_mask_I"] = torch.from_numpy(
            lattice_valid_mask
        )

        data.setdefault("feats", {})
        data["feats"]["material_atom_mask"] = material_atom_mask
        data["feats"]["material_token_mask"] = material_token_mask
        data["feats"]["material_coord_generation_mask"] = coord_generation_mask
        data["feats"]["material_lattice_generation_mask"] = lattice_generation_mask
        return data


class MaterialAwareMotifCenterRandomAugmentation(MotifCenterRandomAugmentation):
    """Apply RFD3 augmentation to molecules and cell-aware material augmentation.

    RFD3 centers, translates, and rotates finite Cartesian point clouds. Periodic
    crystals use the same zero-centroid gauge, but their fractional targets must
    be recomputed modulo the lattice after centering. A global rotation must rotate
    the unit-cell vectors together with the Cartesian coordinates.

    Centering restores the zero-origin gauge used for unconditional inference.
    Periodic cell-cut augmentation must remain disabled for backbones that do not
    construct minimum-image, lattice-aware geometry.
    """

    def __init__(
        self,
        *args,
        record_rotation_pairs: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.record_rotation_pairs = record_rotation_pairs

    @staticmethod
    def _rotate_force_targets(data: dict, rotations: torch.Tensor) -> None:
        """Rotate vector labels with the same per-view rotations as coordinates."""
        ground_truth = data.get("ground_truth", {})
        forces = ground_truth.get("force_gt_D_L")
        if forces is None:
            return
        if forces.ndim != 3 or forces.shape[-1] != 3:
            raise ValueError(
                "Force targets must have shape [D, L, 3], got "
                f"{tuple(forces.shape)}."
            )
        rotations = rotations.to(device=forces.device, dtype=forces.dtype)
        if rotations.ndim == 2:
            rotations = rotations.unsqueeze(0)
        if rotations.shape != (forces.shape[0], 3, 3):
            raise ValueError(
                "Force-target rotations must have shape [D, 3, 3], got "
                f"{tuple(rotations.shape)} for D={forces.shape[0]}."
            )
        ground_truth["force_gt_D_L"] = torch.einsum("dij,dlj->dli", rotations, forces)

    def _augment_molecule(self, data: dict) -> dict:
        """Reproduce RFD3's molecule augmentation while retaining rotations."""
        coordinates = data["coord_atom_lvl_to_be_noised"]
        mask_atom_lvl = data["ground_truth"]["mask_atom_lvl"]
        mask_atom_lvl = mask_atom_lvl & ~data["feats"]["is_motif_atom_unindexed"]
        valid_atom_mask = mask_atom_lvl.clone()

        fixed_coordinate_mask = torch.tensor(
            data["atom_array"].is_motif_atom_with_fixed_coord, dtype=torch.bool
        )
        if torch.any(fixed_coordinate_mask):
            if self.center_option == "motif":
                center_mask = fixed_coordinate_mask.clone()
            elif self.center_option == "diffuse":
                center_mask = (~fixed_coordinate_mask).clone()
            else:
                center_mask = torch.ones(mask_atom_lvl.shape, dtype=torch.bool)
        else:
            center_mask = torch.ones(mask_atom_lvl.shape, dtype=torch.bool)

        mask_atom_lvl = mask_atom_lvl & center_mask
        if self.record_rotation_pairs and not torch.any(mask_atom_lvl):
            # A clean MLIP task fixes every coordinate, so ``center_option=diffuse``
            # otherwise selects an empty centering set and produces NaNs. TransIP
            # centers the complete molecule before constructing its SO(3) pair.
            mask_atom_lvl = valid_atom_mask
        coordinates = masked_center(
            coordinates, mask_atom_lvl.expand(coordinates.shape[0], -1)
        )

        # Keep the existing RFD3 draw order exactly.  RFD3 currently applies
        # ``self.sigma_perturb`` here for every conditioning mode.
        translation_batch_size = 1 if self.record_rotation_pairs else self.batch_size
        translation = (
            torch.randn((translation_batch_size, 3), device=coordinates.device)[
                :, None, :
            ]
            * self.sigma_perturb
        )
        coordinates = coordinates + translation

        rigid = get_random_rigid(self.batch_size, scale=0)
        if self.batch_size == 1:
            rigid = rigid[0].unsqueeze(0), rigid[1].unsqueeze(0)
        coordinates = apply_batched_rigid(rigid, coordinates)
        data["coord_atom_lvl_to_be_noised"] = coordinates
        _synchronize_motif_positions_with_coordinates(data)
        self._rotate_force_targets(data, rigid[0])
        if self.record_rotation_pairs:
            data.setdefault("feats", {})["augmentation_rotation_D"] = rigid[0]
        return data

    def forward(self, data: dict) -> dict:
        """Rotate coordinates, cells, and vector targets in one sampled frame."""
        material_mask = get_material_mask(data["atom_array"])
        if not np.any(material_mask):
            if "force_gt_D_L" not in data.get("ground_truth", {}):
                # Preserve the exact legacy RFD3 path when property targets are
                # disabled (the default for released and PDB experiments).
                return super().forward(data)
            if data.get("is_inference", False):
                return data
            return self._augment_molecule(data)
        if data.get("is_inference", False):
            return data

        lattice = get_material_lattice(data["atom_array"])
        if lattice is None:
            raise ValueError(
                "Material rotation requires lattice vectors on the atom array."
            )
        _, _, lattice_vectors, _ = lattice

        coordinates = data["coord_atom_lvl_to_be_noised"]
        if coordinates.ndim != 3:
            raise ValueError(
                "Material rotation requires diffusion-batched coordinates with "
                f"shape [D, L, 3], got {coordinates.shape}."
            )
        if coordinates.shape[0] != self.batch_size:
            raise ValueError(
                "Material rotation batch size does not match the configured "
                f"diffusion batch size: {coordinates.shape[0]} != {self.batch_size}."
            )

        rotations = get_random_rots(
            self.batch_size,
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        # RFD3 treats coordinates as column vectors: x' = R x. The row-vector
        # lattice convention therefore transforms each lattice vector identically.
        rotated_coordinates = torch.einsum("dij,dlj->dli", rotations, coordinates)
        lattice_vectors = torch.as_tensor(
            lattice_vectors,
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        rotated_lattice_vectors = torch.einsum(
            "dij,kj->dki", rotations, lattice_vectors
        )

        data["coord_atom_lvl_to_be_noised"] = rotated_coordinates
        self._rotate_force_targets(data, rotations)
        if self.record_rotation_pairs:
            data.setdefault("feats", {})["augmentation_rotation_D"] = rotations
        # The atom array stores one canonical cell, whereas each diffusion duplicate
        # receives its own rotation. Retain the per-duplicate cells transiently for
        # centering and fractional-target synchronization below.
        data["_rotated_material_lattice_vectors_D"] = rotated_lattice_vectors
        return center_periodic_material_coordinates(data, rotated_lattice_vectors)


class RandomPeriodicCellOriginShift(Transform):
    """Randomize the periodic cell cut before rotation, centering, and noising.

    A shared uniform fractional-origin shift is applied to every diffusion view.
    Fractional coordinates are wrapped into the canonical cell and Cartesian
    coordinates are reconstructed from those wrapped representatives. The later
    material-aware augmentation zero-centers and optionally rotates each view.

    Sharing the origin shift preserves the interpretation of multiple views as an
    SO(3) pair, while wrapping changes which periodic images the Euclidean backbone
    sees and therefore provides a genuine periodic-boundary augmentation.

    This transform is an explicit research opt-in. It is valid only when the
    downstream representation constructs lattice-aware minimum-image geometry;
    production Zatom-2 training keeps it disabled for the Euclidean backbone.
    """

    requires_previous_transforms = ["BatchStructuresForDiffusionNoising"]

    def __init__(self, batch_size: int):
        """Initialize the periodic-origin augmentation."""
        self.batch_size = batch_size

    def check_input(self, data: dict):
        """Check that the required batched coordinates and targets are present."""
        check_contains_keys(data, ["ground_truth", "coord_atom_lvl_to_be_noised"])
        check_contains_keys(data, ["feats"])

    def forward(self, data: dict) -> dict:
        """Shift the fractional origin and reconstruct wrapped Cartesian images."""
        if data.get("is_inference", False):
            return data

        atom_array = data["atom_array"]
        lattice = get_material_lattice(atom_array)
        if lattice is None:
            return data
        _, _, canonical_lattice_vectors, material_mask = lattice

        coordinates = data["coord_atom_lvl_to_be_noised"]
        if coordinates.ndim != 3:
            raise ValueError(
                "Periodic-origin augmentation requires diffusion-batched "
                f"coordinates with shape [D, L, 3], got {coordinates.shape}."
            )
        if coordinates.shape[0] != self.batch_size:
            raise ValueError(
                "Periodic-origin augmentation batch size does not match the "
                "configured diffusion batch size: "
                f"{coordinates.shape[0]} != {self.batch_size}."
            )

        device = coordinates.device
        dtype = coordinates.dtype
        lattice_vectors = torch.as_tensor(
            canonical_lattice_vectors, device=device, dtype=dtype
        ).expand(coordinates.shape[0], -1, -1)
        material_mask = torch.as_tensor(material_mask, device=device, dtype=torch.bool)
        finite_coordinate_mask = torch.isfinite(coordinates).all(dim=-1)

        is_virtual = data["feats"].get("is_virtual")
        if is_virtual is None:
            is_virtual = atom_array.element == VIRTUAL_ATOM_ELEMENT_NAME
        is_virtual = torch.as_tensor(is_virtual, device=device, dtype=torch.bool)
        if is_virtual.shape != material_mask.shape:
            raise ValueError(
                "Periodic-origin augmentation requires the virtual-atom mask to "
                f"have shape {material_mask.shape}, got {is_virtual.shape}."
            )

        finite_physical_atom_mask = finite_coordinate_mask.any(dim=0) & ~is_virtual
        finite_nonmaterial_atom_mask = finite_physical_atom_mask & ~material_mask
        if torch.any(finite_nonmaterial_atom_mask):
            atom_indices = torch.nonzero(
                finite_nonmaterial_atom_mask, as_tuple=False
            ).flatten()
            raise ValueError(
                "Periodic-origin augmentation requires every finite physical atom "
                "to be selected by material_mask; found non-material atoms at indices "
                f"{atom_indices.tolist()}."
            )

        fractional_targets = data["ground_truth"]["fractional_coord_gt_L"].to(
            device=device, dtype=dtype
        )
        if fractional_targets.ndim == 2:
            fractional_targets = fractional_targets.unsqueeze(0).expand(
                coordinates.shape[0], -1, -1
            )
        elif fractional_targets.shape != coordinates.shape:
            raise ValueError(
                "Fractional coordinate targets must have shape [L, 3] or "
                f"[D, L, 3], got {fractional_targets.shape}."
            )
        fractional_targets = fractional_targets.clone()
        valid_material_coordinate_mask = material_mask[None] & finite_coordinate_mask

        # Share the shift across views so any subsequent pair differs only by its
        # sampled rotation. Uniform fractional sampling explores every cell cut.
        fractional_origin_shift = torch.rand((1, 1, 3), device=device, dtype=dtype)
        shifted_fractional_coordinates = torch.remainder(
            fractional_targets + fractional_origin_shift, 1.0
        )
        reconstructed_coordinates = shifted_fractional_coordinates @ lattice_vectors

        data["coord_atom_lvl_to_be_noised"] = torch.where(
            valid_material_coordinate_mask[..., None],
            reconstructed_coordinates,
            coordinates,
        )
        _synchronize_motif_positions_with_coordinates(data)
        data["ground_truth"]["fractional_coord_gt_L"] = torch.where(
            valid_material_coordinate_mask[..., None],
            shifted_fractional_coordinates,
            fractional_targets,
        )
        return data


# Backward-compatible import for callers using the original transform name. The
# config flag is retained as well, but now denotes periodic cell-cut augmentation.
RandomlyTranslateMaterialCoordinates = RandomPeriodicCellOriginShift
