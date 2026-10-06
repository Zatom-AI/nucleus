"""Tests for material-specific input transforms."""

import copy

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray
from rfd3na.transforms.design_transforms import MotifCenterRandomAugmentation

from nucleus.transforms.design_transforms import MotifCentering
from nucleus.transforms.material_transforms import (
    MaterialAwareMotifCenterRandomAugmentation,
    RandomPeriodicCellOriginShift,
)
from nucleus.utils.materials import (
    annotate_material_properties,
    cartesian_to_fractional,
)


def test_material_rotation_co_rotates_cell_and_centers_periodic_coordinates(
    monkeypatch,
):
    """A crystal rotation must rotate its Cartesian positions and cell together."""
    lattice_vectors = np.diag([2.0, 4.0, 6.0]).astype(np.float32)
    fractional_coordinates = np.asarray(
        [[0.1, 0.2, 0.3], [0.8, 0.7, 0.6]], dtype=np.float32
    )
    atom_array = AtomArray(2)
    atom_array.coord = fractional_coordinates @ lattice_vectors
    atom_array = annotate_material_properties(
        atom_array,
        fractional_coordinates=fractional_coordinates,
        lattice_lengths=np.asarray([2.0, 4.0, 6.0]),
        lattice_angles=np.asarray([90.0, 90.0, 90.0]),
        lattice_vectors=lattice_vectors,
    )

    rotations = torch.tensor(
        [
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    def mock_random_rots(batch_size, *, device=None, dtype=None):
        assert batch_size == 2
        return rotations.to(device=device, dtype=dtype)

    monkeypatch.setattr(
        "nucleus.transforms.material_transforms.get_random_rots", mock_random_rots
    )

    coordinates = torch.tensor(atom_array.coord).repeat(2, 1, 1)
    original_fractional_targets = torch.tensor(fractional_coordinates)
    data = {
        "atom_array": atom_array,
        "is_inference": False,
        "coord_atom_lvl_to_be_noised": coordinates.clone(),
        "ground_truth": {
            "fractional_coord_gt_L": original_fractional_targets.clone(),
        },
    }

    output = MaterialAwareMotifCenterRandomAugmentation(
        batch_size=2,
        sigma_perturb=2.0,
        center_option="diffuse",
    ).forward(data)

    rotated_coordinates = torch.einsum("dij,dlj->dli", rotations, coordinates)
    expected_lattice_vectors = torch.einsum(
        "dij,kj->dki", rotations, torch.tensor(lattice_vectors)
    )
    expected_coordinates = rotated_coordinates - rotated_coordinates.mean(
        dim=1, keepdim=True
    )
    expected_fractional_targets = cartesian_to_fractional(
        expected_coordinates, expected_lattice_vectors
    )
    torch.testing.assert_close(
        output["coord_atom_lvl_to_be_noised"], expected_coordinates
    )
    torch.testing.assert_close(
        output["coord_atom_lvl_to_be_noised"].mean(dim=1),
        torch.zeros((2, 3)),
        rtol=0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        output["_rotated_material_lattice_vectors_D"], expected_lattice_vectors
    )
    torch.testing.assert_close(
        output["ground_truth"]["fractional_coord_gt_L"],
        expected_fractional_targets,
    )
    torch.testing.assert_close(
        cartesian_to_fractional(expected_coordinates, expected_lattice_vectors),
        expected_fractional_targets,
    )


def test_fixed_mlip_motif_positions_match_augmented_pair_geometry(monkeypatch):
    """Fixed MLIP motif distances must match every rigidly augmented clean view."""
    lattice_vectors = np.diag([4.0, 5.0, 6.0]).astype(np.float32)
    fractional_coordinates = np.asarray(
        [[0.10, 0.20, 0.30], [0.35, 0.75, 0.55], [0.80, 0.40, 0.15]],
        dtype=np.float32,
    )
    atom_array = AtomArray(3)
    atom_array.coord = fractional_coordinates @ lattice_vectors
    atom_array.set_annotation("is_motif_atom_with_fixed_coord", np.ones(3, dtype=bool))
    atom_array = annotate_material_properties(
        atom_array,
        fractional_coordinates=fractional_coordinates,
        lattice_lengths=np.asarray([4.0, 5.0, 6.0]),
        lattice_angles=np.asarray([90.0, 90.0, 90.0]),
        lattice_vectors=lattice_vectors,
    )

    rotations = torch.tensor(
        [
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    monkeypatch.setattr(
        "nucleus.transforms.material_transforms.get_random_rots",
        lambda *args, **kwargs: rotations,
    )
    origin_shift = torch.tensor([[[0.30, 0.40, 0.50]]], dtype=torch.float32)
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *args, **kwargs: origin_shift.to(
            device=kwargs.get("device"), dtype=kwargs.get("dtype")
        ),
    )

    clean_coordinates = torch.tensor(atom_array.coord).repeat(2, 1, 1)
    data = {
        "atom_array": atom_array,
        "is_inference": False,
        "sampled_condition_name": "mlip",
        "coord_atom_lvl_to_be_noised": clean_coordinates,
        "ground_truth": {
            "fractional_coord_gt_L": torch.tensor(fractional_coordinates),
        },
        "feats": {
            "is_motif_atom_with_fixed_coord": torch.ones(3, dtype=torch.bool),
            "motif_pos": torch.tensor(atom_array.coord),
        },
    }

    # Production entrypoints disable this transform for the non-PBC backbone.
    # Exercise the opt-in path here so reintroducing it cannot recreate the stale
    # fixed-coordinate feature that originally motivated this regression test.
    data = RandomPeriodicCellOriginShift(batch_size=2).forward(data)
    output = MaterialAwareMotifCenterRandomAugmentation(
        batch_size=2,
        sigma_perturb=0.0,
        center_option="diffuse",
    ).forward(data)

    coordinates = output["coord_atom_lvl_to_be_noised"]
    motif_positions = output["feats"]["motif_pos"]
    # ``motif_pos`` has no D axis, so it uses view zero as the representative
    # augmented frame. Every view must nevertheless encode identical distances.
    torch.testing.assert_close(motif_positions, coordinates[0])
    motif_pair_distances = torch.cdist(motif_positions, motif_positions)
    for view_coordinates in coordinates:
        torch.testing.assert_close(
            torch.cdist(view_coordinates, view_coordinates),
            motif_pair_distances,
            rtol=1e-5,
            atol=1e-5,
        )


def test_disabled_random_augmentation_centers_materials_and_updates_fractional():
    """Deterministic material preprocessing uses the training periodic gauge."""
    lattice_vectors = np.diag([2.0, 4.0, 6.0])
    fractional_coordinates = np.asarray(
        [[0.1, 0.2, 0.3], [0.8, 0.7, 0.6]], dtype=np.float32
    )
    atom_array = AtomArray(2)
    atom_array.coord = fractional_coordinates @ lattice_vectors
    atom_array = annotate_material_properties(
        atom_array,
        fractional_coordinates=fractional_coordinates,
        lattice_lengths=np.asarray([2.0, 4.0, 6.0]),
        lattice_angles=np.asarray([90.0, 90.0, 90.0]),
        lattice_vectors=lattice_vectors,
    )
    coordinates = torch.tensor(atom_array.coord)[None]
    data = {
        "atom_array": atom_array,
        "is_inference": False,
        "coord_atom_lvl_to_be_noised": coordinates.clone(),
        "ground_truth": {"fractional_coord_gt_L": torch.tensor(fractional_coordinates)},
    }

    output = MotifCentering(center_option="diffuse").forward(data)

    expected_coordinates = coordinates - coordinates.mean(dim=1, keepdim=True)
    expected_fractional = cartesian_to_fractional(
        expected_coordinates, torch.tensor(lattice_vectors)
    )
    torch.testing.assert_close(
        output["coord_atom_lvl_to_be_noised"], expected_coordinates
    )
    torch.testing.assert_close(
        output["ground_truth"]["fractional_coord_gt_L"], expected_fractional
    )


def test_material_aware_rotation_preserves_non_material_rfd3_behavior():
    """Non-periodic examples must still use the original RFD3 augmentation."""
    atom_array = AtomArray(2)
    atom_array.set_annotation(
        "is_motif_atom_with_fixed_coord", np.asarray([False, False])
    )
    data = {
        "atom_array": atom_array,
        "is_inference": False,
        "coord_atom_lvl_to_be_noised": torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]],
                [[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]],
            ]
        ),
        "ground_truth": {"mask_atom_lvl": torch.ones(2, dtype=torch.bool)},
        "feats": {"is_motif_atom_unindexed": torch.zeros(2, dtype=torch.bool)},
        "sampled_condition_name": "unconditional",
    }
    transform_kwargs = {
        "batch_size": 2,
        "sigma_perturb": 2.0,
        "center_option": "diffuse",
    }

    torch.manual_seed(123)
    expected = MotifCenterRandomAugmentation(**transform_kwargs).forward(
        copy.deepcopy(data)
    )
    torch.manual_seed(123)
    output = MaterialAwareMotifCenterRandomAugmentation(**transform_kwargs).forward(
        copy.deepcopy(data)
    )

    torch.testing.assert_close(
        output["coord_atom_lvl_to_be_noised"],
        expected["coord_atom_lvl_to_be_noised"],
    )


def test_random_periodic_origin_shift_wraps_and_reconstructs_shared_cell_cut(
    monkeypatch,
):
    """A shared fractional shift changes periodic images in every view."""
    lattice_vectors = np.diag([2.0, 4.0, 6.0])
    fractional_coordinates = np.asarray(
        [[0.1, 0.2, 0.3], [0.8, 0.7, 0.6]], dtype=np.float32
    )
    atom_array = AtomArray(2)
    atom_array.coord = fractional_coordinates @ lattice_vectors
    atom_array = annotate_material_properties(
        atom_array,
        fractional_coordinates=fractional_coordinates,
        lattice_lengths=np.asarray([2.0, 4.0, 6.0]),
        lattice_angles=np.asarray([90.0, 90.0, 90.0]),
        lattice_vectors=lattice_vectors,
    )

    origin_shift = torch.tensor([[[0.3, 0.4, 0.5]]], dtype=torch.float32)
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *args, **kwargs: origin_shift.to(
            device=kwargs.get("device"), dtype=kwargs.get("dtype")
        ),
    )

    clean_coordinates = torch.tensor(atom_array.coord).repeat(2, 1, 1)
    data = {
        "atom_array": atom_array,
        "is_inference": False,
        "coord_atom_lvl_to_be_noised": clean_coordinates.clone(),
        "ground_truth": {
            "fractional_coord_gt_L": torch.tensor(fractional_coordinates),
            "fractional_coord_valid_mask_L": torch.ones(2, dtype=torch.bool),
        },
        "feats": {"is_motif_atom_with_fixed_coord": torch.zeros(2, dtype=torch.bool)},
    }

    output = RandomPeriodicCellOriginShift(batch_size=2).forward(data)

    expected_fractional = torch.remainder(
        torch.tensor(fractional_coordinates)[None] + origin_shift, 1.0
    ).expand(2, -1, -1)
    expected_coordinates = expected_fractional @ torch.tensor(
        lattice_vectors, dtype=torch.float32
    )
    torch.testing.assert_close(
        output["coord_atom_lvl_to_be_noised"],
        expected_coordinates,
    )
    torch.testing.assert_close(
        output["ground_truth"]["fractional_coord_gt_L"],
        expected_fractional,
    )
    torch.testing.assert_close(expected_fractional[0], expected_fractional[1])
    assert torch.any(
        torch.abs(expected_coordinates - clean_coordinates) > 0.5
    ), "At least one atom should cross a periodic cell boundary."
    np.testing.assert_allclose(
        atom_array.coord, fractional_coordinates @ lattice_vectors
    )


def test_random_periodic_origin_shift_ignores_nonfinite_coordinates(
    monkeypatch,
):
    """Test that the random translation of material coordinates ignores invalid (non-finite) coordinates and does not modify them."""
    lattice_vectors = np.diag([2.0, 4.0, 6.0])
    fractional_coordinates = np.asarray(
        [[0.1, 0.2, 0.3], [0.8, 0.7, 0.6]], dtype=np.float32
    )
    atom_array = AtomArray(2)
    atom_array.coord = fractional_coordinates @ lattice_vectors
    atom_array = annotate_material_properties(
        atom_array,
        fractional_coordinates=fractional_coordinates,
        lattice_lengths=np.asarray([2.0, 4.0, 6.0]),
        lattice_angles=np.asarray([90.0, 90.0, 90.0]),
        lattice_vectors=lattice_vectors,
    )

    monkeypatch.setattr(
        torch,
        "rand",
        lambda *args, **kwargs: torch.tensor([[[0.1, 0.2, 0.3]]]),
    )
    coordinates = torch.tensor(atom_array.coord)[None]
    coordinates[:, 1] = torch.nan
    original_fractional_targets = torch.tensor(fractional_coordinates)
    data = {
        "atom_array": atom_array,
        "is_inference": False,
        "coord_atom_lvl_to_be_noised": coordinates.clone(),
        "ground_truth": {
            "fractional_coord_gt_L": original_fractional_targets.clone(),
            "fractional_coord_valid_mask_L": torch.tensor([True, False]),
        },
        "feats": {"is_motif_atom_with_fixed_coord": torch.zeros(2, dtype=torch.bool)},
    }

    output = RandomPeriodicCellOriginShift(batch_size=1).forward(data)

    expected_coordinates = coordinates.clone()
    shifted_first_fractional = torch.remainder(
        torch.tensor(fractional_coordinates[0]) + torch.tensor([0.1, 0.2, 0.3]),
        1.0,
    )
    expected_coordinates[:, 0] = shifted_first_fractional @ torch.tensor(
        lattice_vectors, dtype=coordinates.dtype
    )
    torch.testing.assert_close(
        output["coord_atom_lvl_to_be_noised"], expected_coordinates, equal_nan=True
    )
    expected_fractional_targets = original_fractional_targets[None].clone()
    expected_fractional_targets[:, 0] = shifted_first_fractional
    torch.testing.assert_close(
        output["ground_truth"]["fractional_coord_gt_L"],
        expected_fractional_targets,
    )
    assert torch.isfinite(output["ground_truth"]["fractional_coord_gt_L"]).all()


def test_random_periodic_origin_shift_rejects_finite_nonmaterial_atoms():
    """Finite physical atoms must all belong to the translated periodic system."""
    lattice_vectors = np.diag([2.0, 4.0, 6.0])
    fractional_coordinates = np.asarray(
        [[0.1, 0.2, 0.3], [0.8, 0.7, 0.6]], dtype=np.float32
    )
    atom_array = AtomArray(2)
    atom_array.coord = fractional_coordinates @ lattice_vectors
    atom_array = annotate_material_properties(
        atom_array,
        fractional_coordinates=fractional_coordinates,
        lattice_lengths=np.asarray([2.0, 4.0, 6.0]),
        lattice_angles=np.asarray([90.0, 90.0, 90.0]),
        lattice_vectors=lattice_vectors,
        is_material=np.asarray([True, False]),
    )
    data = {
        "atom_array": atom_array,
        "is_inference": False,
        "coord_atom_lvl_to_be_noised": torch.tensor(atom_array.coord)[None],
        "ground_truth": {
            "fractional_coord_gt_L": torch.tensor(fractional_coordinates),
            "fractional_coord_valid_mask_L": torch.tensor([True, False]),
        },
        "feats": {
            "is_motif_atom_with_fixed_coord": torch.zeros(2, dtype=torch.bool),
            "is_virtual": torch.zeros(2, dtype=torch.bool),
        },
        "sampled_condition_name": "unconditional",
    }

    with pytest.raises(
        ValueError,
        match=r"every finite physical atom.*non-material atoms at indices \[1\]",
    ):
        RandomPeriodicCellOriginShift(batch_size=1).forward(data)


def test_periodic_origin_shift_then_rotation_retains_zero_centroid(monkeypatch):
    """Cell-cut augmentation precedes rotation and zero-centroid gauge fixing."""
    lattice_vectors = np.diag([2.0, 4.0, 6.0]).astype(np.float32)
    fractional_coordinates = np.asarray(
        [[0.1, 0.2, 0.3], [0.8, 0.7, 0.6]], dtype=np.float32
    )
    atom_array = AtomArray(2)
    atom_array.coord = fractional_coordinates @ lattice_vectors
    atom_array = annotate_material_properties(
        atom_array,
        fractional_coordinates=fractional_coordinates,
        lattice_lengths=np.asarray([2.0, 4.0, 6.0]),
        lattice_angles=np.asarray([90.0, 90.0, 90.0]),
        lattice_vectors=lattice_vectors,
    )

    rotations = torch.tensor(
        [
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    canonical_coordinates = torch.tensor(atom_array.coord).repeat(2, 1, 1)
    origin_shift = torch.tensor([[[0.3, 0.4, 0.5]]])
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *args, **kwargs: origin_shift.to(
            device=kwargs.get("device"), dtype=kwargs.get("dtype")
        ),
    )
    data = {
        "atom_array": atom_array,
        "is_inference": False,
        "coord_atom_lvl_to_be_noised": canonical_coordinates.clone(),
        "ground_truth": {
            "fractional_coord_gt_L": torch.tensor(fractional_coordinates),
            "fractional_coord_valid_mask_L": torch.ones(2, dtype=torch.bool),
        },
        "feats": {"is_motif_atom_with_fixed_coord": torch.zeros(2, dtype=torch.bool)},
    }

    data = RandomPeriodicCellOriginShift(batch_size=2).forward(data)
    with monkeypatch.context() as rotation_patch:
        rotation_patch.setattr(
            "nucleus.transforms.material_transforms.get_random_rots",
            lambda *args, **kwargs: rotations,
        )
        output = MaterialAwareMotifCenterRandomAugmentation(
            batch_size=2,
            sigma_perturb=0.0,
            center_option="all",
        ).forward(data)

    expected_fractional_before_centering = torch.remainder(
        torch.tensor(fractional_coordinates)[None] + origin_shift, 1.0
    ).expand(2, -1, -1)
    wrapped_coordinates = expected_fractional_before_centering @ torch.tensor(
        lattice_vectors
    )
    rotated_coordinates = torch.einsum("dij,dlj->dli", rotations, wrapped_coordinates)
    expected_coordinates = rotated_coordinates - rotated_coordinates.mean(
        dim=1, keepdim=True
    )
    rotated_lattice_vectors = torch.einsum(
        "dij,kj->dki", rotations, torch.tensor(lattice_vectors)
    )
    expected_fractional = cartesian_to_fractional(
        expected_coordinates, rotated_lattice_vectors
    )
    torch.testing.assert_close(
        output["coord_atom_lvl_to_be_noised"], expected_coordinates
    )
    torch.testing.assert_close(
        output["ground_truth"]["fractional_coord_gt_L"], expected_fractional
    )
    torch.testing.assert_close(
        output["coord_atom_lvl_to_be_noised"].mean(dim=1),
        torch.zeros((2, 3)),
        rtol=0,
        atol=1e-6,
    )
