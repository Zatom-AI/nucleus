"""Nucleus design transform adapters.

These classes keep nucleus-specific behavior out of the vendored RFD3NA
dependency while reusing the rest of the RFD3NA pipeline unchanged.
"""

import numpy as np
import torch
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.geometry import masked_center

from nucleus.transforms.atomistic_property_transforms import infer_property_task_name
from nucleus.utils.materials import has_material

try:
    from rfd3na.transforms.design_transforms import SubsampleToTypes
except ImportError:
    raise ImportError(
        "Could not import RFD3NA design transforms. Please ensure that the `rc-foundry[all]` package is installed and up to date."
    )


class SubsampleToPolymerOrLigandTypes(SubsampleToTypes):
    """RFD3NA ``SubsampleToTypes`` variant that never drops ligands.

    Kept atoms = ``is_ligand`` (always) plus the polymer types listed in
    ``allowed_types`` (``is_protein`` / ``is_dna`` / ``is_rna``, or ``"ALL"``).
    Since it only removes atoms, examples without a ligand stay ligand-free:

    - ``["is_protein"]`` -> protein design, unchanged
    - ``["is_rna", "is_dna"]`` -> nucleotide design: RNA-only stays
      RNA-only, RNA+ligand keeps both (co-gen)
    - ``["is_protein", "is_rna", "is_dna"]`` -> protein + nucleotide together

    So one config-driven transform covers protein, nucleotide, and
    protein-nucleotide regimes; Zatom-2P keeps its existing protein behavior.
    """

    def forward(self, data):
        """Subsample to ligands plus the allowed polymer types."""
        atom_array = data["atom_array"]

        if self.allowed_types != "ALL":
            # Always retain ligands/molecules/materials; allowed_types gates polymers.
            is_allowed = np.asarray(atom_array.is_ligand, dtype=bool).copy()
            for allowed_type in self.allowed_types:
                is_allowed = np.logical_or(
                    is_allowed,
                    np.asarray(
                        atom_array.get_annotation(allowed_type), dtype=bool
                    ).copy(),
                )
            atom_array = atom_array[is_allowed]

        if atom_array.array_length() == 0:
            raise ValueError(
                "No tokens found in the atom array! Example ID: {}".format(
                    data.get("example_id", "unknown")
                )
            )

        n_polymer_or_ligand = (
            atom_array.is_protein.sum()
            + atom_array.is_dna.sum()
            + atom_array.is_rna.sum()
            + atom_array.is_ligand.sum()
        )
        if n_polymer_or_ligand == 0:
            raise ValueError(
                "No polymer or ligand atoms found in the atom array. "
                "Example ID: {}".format(data.get("example_id", "unknown"))
            )

        data["atom_array"] = atom_array
        return data


def _center_nonperiodic_coordinates(data: dict, *, center_option: str) -> dict:
    """Center non-periodic coordinates with RFD3's motif-aware policy."""
    xyz = data["coord_atom_lvl_to_be_noised"]  # (batch_size, n_atoms, 3)
    mask_atom_lvl = data["ground_truth"]["mask_atom_lvl"]
    mask_atom_lvl = (
        mask_atom_lvl & ~data["feats"]["is_motif_atom_unindexed"]
    )  # Avoid double weighting

    is_motif_atom_with_fixed_coord = torch.tensor(
        data["atom_array"].is_motif_atom_with_fixed_coord, dtype=torch.bool
    )
    if torch.any(is_motif_atom_with_fixed_coord):
        if center_option == "motif":
            center_mask = is_motif_atom_with_fixed_coord.clone()
        elif center_option == "diffuse":
            center_mask = (~is_motif_atom_with_fixed_coord).clone()
        else:
            center_mask = torch.ones(mask_atom_lvl.shape, dtype=torch.bool)
    else:
        center_mask = torch.ones(mask_atom_lvl.shape, dtype=torch.bool)

    centering_mask = mask_atom_lvl & center_mask
    if not torch.any(centering_mask):
        # The requested subset can be empty, e.g. ``center_option="diffuse"``
        # when every MLIP atom has fixed coordinates. Fall back to all valid
        # atoms rather than taking the mean of an empty tensor.
        centering_mask = mask_atom_lvl
    if not torch.any(centering_mask):
        # There is no meaningful center when every atom is invalid or
        # unindexed. Preserve the existing coordinate frame.
        return data
    centering_mask_expanded = centering_mask.expand(xyz.shape[0], -1)

    centered_xyz = masked_center(xyz, centering_mask_expanded)
    data["coord_atom_lvl_to_be_noised"] = centered_xyz

    if infer_property_task_name(data) != "mlip":
        return data

    # Fixed-coordinate and reference-conformer features have no diffusion-view
    # axis. Keep their first-view gauge synchronized with the clean Cartesian
    # input. These features are consumed through pair distances, so translating
    # them does not change their internal geometry.
    feats = data.get("feats", {})
    motif_positions = feats.get("motif_pos")
    if motif_positions is not None:
        motif_positions = torch.as_tensor(
            motif_positions,
            device=centered_xyz.device,
            dtype=centered_xyz.dtype,
        ).clone()
        if motif_positions.shape != centered_xyz.shape[1:]:
            raise ValueError(
                "Motif positions must align with centered coordinates and have "
                f"shape {centered_xyz.shape[1:]}, got {motif_positions.shape}."
            )
        fixed_coordinate_mask = torch.as_tensor(
            feats.get(
                "is_motif_atom_with_fixed_coord",
                data["atom_array"].is_motif_atom_with_fixed_coord,
            ),
            device=centered_xyz.device,
            dtype=torch.bool,
        )
        finite_fixed_mask = fixed_coordinate_mask & torch.isfinite(centered_xyz[0]).all(
            dim=-1
        )
        motif_positions[finite_fixed_mask] = centered_xyz[0, finite_fixed_mask]
        feats["motif_pos"] = motif_positions

    reference_positions = feats.get("ref_pos")
    reference_mask = feats.get("ref_mask")
    if reference_positions is not None and reference_mask is not None:
        reference_positions = torch.as_tensor(
            reference_positions,
            device=centered_xyz.device,
            dtype=centered_xyz.dtype,
        ).clone()
        reference_mask = torch.as_tensor(
            reference_mask,
            device=centered_xyz.device,
            dtype=torch.bool,
        )
        if reference_positions.shape != centered_xyz.shape[1:]:
            raise ValueError(
                "Reference positions must align with centered coordinates and "
                f"have shape {centered_xyz.shape[1:]}, got "
                f"{reference_positions.shape}."
            )
        if reference_mask.shape != centered_xyz.shape[1:2]:
            raise ValueError(
                "Reference mask must align with centered coordinates and have "
                f"shape {centered_xyz.shape[1:2]}, got {reference_mask.shape}."
            )
        finite_reference_mask = reference_mask & torch.isfinite(
            reference_positions
        ).all(dim=-1)
        center_weights = centering_mask_expanded[..., None].to(dtype=xyz.dtype)
        coordinate_center = (torch.nan_to_num(xyz) * center_weights).sum(
            dim=(0, 1)
        ) / center_weights.sum().clamp_min(1)
        reference_positions[finite_reference_mask] -= coordinate_center
        feats["ref_pos"] = reference_positions
    return data


class MotifCentering(Transform):
    """Center training coordinates using the RFD3 motif-aware policy.

    This is the deterministic portion of RFD3's
    ``MotifCenterRandomAugmentation`` transform. It is used when random rigid
    augmentation is disabled so that clean coordinates remain centered.

    Args:
        center_option: Atoms used to calculate the center when fixed motif atoms
            are present. Supported options are ``"motif"``, ``"diffuse"``, and
            ``"all"``.
    """

    requires_previous_transforms = ["BatchStructuresForDiffusionNoising"]

    def __init__(self, center_option: str):
        self.center_option = center_option

    def forward(self, data):
        """Center training coordinates without rotating or translating them."""
        if data["is_inference"]:
            return data  # ori token behaviour set when creating atom array & in sampler
        atom_array = data["atom_array"]
        if hasattr(atom_array, "get_annotation_categories") and has_material(
            atom_array
        ):
            # Keep deterministic material validation in the same zero-centroid
            # periodic gauge used during stochastic training augmentation.
            from nucleus.transforms.material_transforms import (
                center_periodic_material_coordinates,
            )
            from nucleus.utils.materials import get_material_lattice

            lattice = get_material_lattice(atom_array)
            if lattice is None:
                raise ValueError(
                    "Periodic material centering requires lattice vectors on the "
                    "atom array."
                )
            return center_periodic_material_coordinates(data, lattice[2])

        return _center_nonperiodic_coordinates(
            data,
            center_option=self.center_option,
        )


class CanonicalizeMLIPInferenceCoordinates(Transform):
    """Put direct MLIP inference inputs in the deterministic training gauge.

    Generative inference deliberately keeps its original coordinate frame for
    the sampler. Direct all-fixed property inference bypasses that sampler and
    must instead see the same zero-centroid coordinates as MLIP training and
    the standalone evaluator.

    Args:
        center_option: RFD3 centering policy. An all-fixed MLIP example falls
            back to all valid atoms when ``"diffuse"`` selects an empty set.
    """

    requires_previous_transforms = ["BatchStructuresForDiffusionNoising"]

    def __init__(self, center_option: str):
        self.center_option = center_option

    def forward(self, data: dict) -> dict:
        """Center only inference examples routed to direct MLIP prediction."""
        if not data.get("is_inference", False):
            return data
        if infer_property_task_name(data) != "mlip":
            return data

        atom_array = data["atom_array"]
        if hasattr(atom_array, "get_annotation_categories") and has_material(
            atom_array
        ):
            from nucleus.transforms.material_transforms import (
                center_periodic_material_coordinates,
            )
            from nucleus.utils.materials import get_material_lattice

            lattice = get_material_lattice(atom_array)
            if lattice is None:
                raise ValueError(
                    "Periodic MLIP inference centering requires lattice vectors "
                    "on the atom array."
                )
            return center_periodic_material_coordinates(data, lattice[2])

        return _center_nonperiodic_coordinates(
            data,
            center_option=self.center_option,
        )
