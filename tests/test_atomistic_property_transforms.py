"""Tests for atomistic property extraction, augmentation, and collation."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from atomworks.ml.datasets.loaders import create_ase_atoms_loader

from nucleus.constants import (
    MLIP_PROPERTY_TASK_ID,
    STRUCTURE_PREDICTION_PROPERTY_TASK_ID,
    UNCONDITIONAL_PROPERTY_TASK_ID,
    UNKNOWN_PROPERTY_TASK_ID,
)
from nucleus.transforms.atomistic_property_transforms import (
    OMAT24_DOMAIN_ID,
    OMOL25_DOMAIN_ID,
    AddAtomisticPropertyTargets,
    AddElectronicStateConditioning,
    FinalizeAtomisticPropertyTargets,
)
from nucleus.transforms.material_transforms import (
    MaterialAwareMotifCenterRandomAugmentation,
    RandomPeriodicCellOriginShift,
)
from nucleus.utils.batching import SparseCollate
from nucleus.utils.materials import annotate_material_properties


def _molecule_data(
    *,
    symbols: str = "COH",
    energy: float | None = -7.25,
    forces: np.ndarray | None = None,
    charge: object = 1,
    spin: object = 2,
) -> dict:
    """Build a loader-shaped molecule record with one atom per token."""
    n_atoms = len(Atoms(symbols))
    atoms = Atoms(
        symbols,
        positions=np.arange(n_atoms * 3, dtype=float).reshape(n_atoms, 3) / 10,
    )
    if charge is not None:
        atoms.info["charge"] = charge
    if spin is not None:
        atoms.info["spin"] = spin
    if forces is None and energy is not None:
        forces = np.arange(n_atoms * 3, dtype=float).reshape(n_atoms, 3) / 5
    calculator_kwargs = {}
    if energy is not None:
        calculator_kwargs["energy"] = energy
    if forces is not None:
        calculator_kwargs["forces"] = forces
    if calculator_kwargs:
        atoms.calc = SinglePointCalculator(atoms, **calculator_kwargs)

    loader = create_ase_atoms_loader(
        res_name="LIG", include_chain_info=True, keep_ase_atoms=True
    )
    data = loader(atoms)
    data["atom_array"].set_annotation("atomize", np.ones(n_atoms, dtype=bool))
    return data


def _extract_and_finalize(data: dict, *, diffusion_batch_size: int = 2) -> dict:
    """Apply both stages of the atomistic target transform."""
    data.setdefault("is_inference", False)
    data.setdefault("sampled_condition_name", "mlip")
    data = AddAtomisticPropertyTargets(
        diffusion_batch_size=diffusion_batch_size, domain="omol25"
    )(data)
    return FinalizeAtomisticPropertyTargets(diffusion_batch_size)(data)


def test_extracts_omol_energy_forces_charge_and_spin_in_ase_units():
    """Finite ASE results and OMol metadata populate the exact tensor contract."""
    forces = np.array(
        [[1.0, -2.0, 3.0], [0.5, 0.25, -0.75], [-4.0, 5.0, 6.0]],
        dtype=np.float32,
    )
    data = _extract_and_finalize(_molecule_data(forces=forces))
    targets = data["ground_truth"]

    torch.testing.assert_close(
        targets["energy_gt_B"],
        torch.tensor([-7.25], dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        targets["force_gt_D_L"],
        torch.from_numpy(forces).unsqueeze(0).expand(2, -1, -1),
        rtol=0,
        atol=0,
    )
    assert targets["energy_valid_mask_B"].tolist() == [True]
    assert targets["force_valid_mask_L"].tolist() == [True, True, True]
    assert targets["domain_id_B"].tolist() == [OMOL25_DOMAIN_ID]
    assert targets["source_atom_count_B"].tolist() == [3]
    assert targets["retained_atom_count_B"].tolist() == [3]
    assert targets["total_charge_B"].tolist() == [1.0]
    assert targets["spin_multiplicity_B"].tolist() == [2.0]
    assert targets["charge_valid_mask_B"].tolist() == [True]
    assert targets["spin_valid_mask_B"].tolist() == [True]
    assert targets["property_valid_mask_B"].tolist() == [True]
    assert targets["property_task_id_B"].tolist() == [MLIP_PROPERTY_TASK_ID]
    assert targets["property_task_valid_mask_B"].tolist() == [True]

    feats = data["feats"]
    assert feats["total_charge"].shape == (3, 1)
    assert feats["spin_multiplicity"].shape == (3, 1)
    torch.testing.assert_close(feats["total_charge"], torch.ones(3, 1))
    torch.testing.assert_close(feats["spin_multiplicity"], torch.full((3, 1), 2.0))
    torch.testing.assert_close(feats["has_total_charge"], torch.ones(3, 1))
    torch.testing.assert_close(feats["has_spin_multiplicity"], torch.ones(3, 1))


def test_preserves_large_raw_energy_in_float64_until_reference_subtraction():
    """Large electronic energies retain residual-scale fractional information."""
    energy = -1_000_000_000.125

    targets = _extract_and_finalize(_molecule_data(energy=energy))["ground_truth"]

    assert targets["energy_gt_B"].dtype == torch.float64
    assert targets["energy_gt_B"].item() == energy


def test_extracts_omol_charge_and_spin_from_nested_ase_lmdb_info():
    """OMol ASE-LMDB rows retain electronic metadata under ``info['data']``."""
    data = _molecule_data(charge=None, spin=None)
    data["atoms"].info["data"] = {"charge": -1, "spin": 3}

    targets = _extract_and_finalize(data)["ground_truth"]

    assert targets["total_charge_B"].tolist() == [-1.0]
    assert targets["spin_multiplicity_B"].tolist() == [3.0]
    assert targets["charge_valid_mask_B"].tolist() == [True]
    assert targets["spin_valid_mask_B"].tolist() == [True]
    assert targets["property_valid_mask_B"].tolist() == [True]


@pytest.mark.parametrize(
    ("condition_name", "expected_task_id"),
    [
        ("unconditional", UNCONDITIONAL_PROPERTY_TASK_ID),
        ("structure_prediction", STRUCTURE_PREDICTION_PROPERTY_TASK_ID),
        ("mlip", MLIP_PROPERTY_TASK_ID),
    ],
)
def test_all_omol_training_tasks_expose_charge_and_spin_features(
    condition_name,
    expected_task_id,
):
    """Every OMol training task receives valid electronic-state conditioning."""
    data = _molecule_data(charge=1, spin=2)
    data.update(is_inference=False, sampled_condition_name=condition_name)
    data = _extract_and_finalize(data)

    targets = data["ground_truth"]
    assert targets["property_task_id_B"].item() == expected_task_id
    assert targets["total_charge_B"].item() == 1.0
    assert targets["spin_multiplicity_B"].item() == 2.0
    torch.testing.assert_close(data["feats"]["total_charge"], torch.ones(3, 1))
    torch.testing.assert_close(
        data["feats"]["spin_multiplicity"], torch.full((3, 1), 2.0)
    )
    assert data["feats"]["has_total_charge"].all()
    assert data["feats"]["has_spin_multiplicity"].all()


@pytest.mark.parametrize(
    "condition_name",
    ["unconditional", "structure_prediction", "mlip"],
)
def test_electronic_conditioning_does_not_require_property_targets(condition_name):
    """All training tasks receive charge/spin without property target tensors."""
    data = _molecule_data(charge=-1, spin=3)
    data.update(is_inference=False, sampled_condition_name=condition_name)

    data = AddElectronicStateConditioning(domain="omol25")(data)

    ground_truth = data.get("ground_truth", {})
    assert "energy_gt_B" not in ground_truth
    assert "force_gt_D_L" not in ground_truth
    torch.testing.assert_close(data["feats"]["total_charge"], torch.full((3, 1), -1.0))
    torch.testing.assert_close(
        data["feats"]["spin_multiplicity"], torch.full((3, 1), 3.0)
    )
    assert data["feats"]["has_total_charge"].all()
    assert data["feats"]["has_spin_multiplicity"].all()


@pytest.mark.parametrize(
    "condition_name",
    ["unconditional", "structure_prediction", "mlip"],
)
def test_inference_electronic_conditioning_reads_nested_specification(
    condition_name,
):
    """Inference uses the requested state independently of property targets."""
    data = _molecule_data(charge=None, spin=None, energy=None, forces=None)
    data.update(
        is_inference=True,
        specification={
            "extra": {
                "atomistic_property_task": condition_name,
                "total_charge": -2,
                "spin_multiplicity": 4,
            }
        },
    )

    data = AddElectronicStateConditioning()(data)

    torch.testing.assert_close(data["feats"]["total_charge"], torch.full((3, 1), -2.0))
    torch.testing.assert_close(
        data["feats"]["spin_multiplicity"], torch.full((3, 1), 4.0)
    )
    assert data["feats"]["has_total_charge"].all()
    assert data["feats"]["has_spin_multiplicity"].all()


@pytest.mark.parametrize(
    ("task", "expected_task_id"),
    [
        ("mlip", MLIP_PROPERTY_TASK_ID),
        ("unconditional", UNCONDITIONAL_PROPERTY_TASK_ID),
    ],
)
def test_inference_specification_supplies_property_targets_and_metadata(
    task,
    expected_task_id,
):
    """Inference JSON metadata supplies labels and electronic conditioning."""
    forces = np.arange(9, dtype=np.float32).reshape(3, 3) / 10
    data = _molecule_data(energy=None, forces=None, charge=None, spin=None)
    data.update(
        is_inference=True,
        specification={
            "extra": {
                "atomistic_domain": "omol25",
                "atomistic_property_task": task,
                "total_charge": 1,
                "spin_multiplicity": 2,
                "energy_gt_eV": -3.5,
                "forces_gt_eV_per_angstrom": forces.tolist(),
            }
        },
    )
    data = AddAtomisticPropertyTargets(1)(data)
    data = FinalizeAtomisticPropertyTargets(1)(data)

    targets = data["ground_truth"]
    assert targets["property_task_id_B"].item() == expected_task_id
    assert targets["domain_id_B"].item() == OMOL25_DOMAIN_ID
    assert targets["energy_gt_B"].item() == pytest.approx(-3.5)
    torch.testing.assert_close(
        targets["force_gt_D_L"],
        torch.from_numpy(forces).unsqueeze(0),
    )
    assert targets["total_charge_B"].item() == 1
    assert targets["spin_multiplicity_B"].item() == 2
    torch.testing.assert_close(data["feats"]["total_charge"], torch.ones(3, 1))
    torch.testing.assert_close(
        data["feats"]["spin_multiplicity"], torch.full((3, 1), 2.0)
    )
    assert data["feats"]["has_total_charge"].all()
    assert data["feats"]["has_spin_multiplicity"].all()


def test_unknown_explicit_inference_task_remains_invalid():
    """A misspelled explicit task must not be reinterpreted from masks."""
    data = _molecule_data()
    data.update(
        is_inference=True,
        specification={"extra": {"atomistic_property_task": "structure_prediciton"}},
    )

    targets = AddAtomisticPropertyTargets(1, domain="omol25")(data)["ground_truth"]

    assert targets["property_task_id_B"].item() == UNKNOWN_PROPERTY_TASK_ID
    assert not targets["property_task_valid_mask_B"].item()


def test_missing_labels_use_finite_placeholders_with_independent_masks():
    """Missing calculator labels do not remove or poison the property contract."""
    data = _extract_and_finalize(
        _molecule_data(energy=None, forces=None, charge=0, spin=1)
    )
    targets = data["ground_truth"]

    torch.testing.assert_close(
        targets["energy_gt_B"], torch.zeros(1, dtype=torch.float64)
    )
    torch.testing.assert_close(targets["force_gt_D_L"], torch.zeros(2, 3, 3))
    assert targets["property_valid_mask_B"].tolist() == [True]
    assert targets["energy_present_mask_B"].tolist() == [False]
    assert targets["energy_valid_mask_B"].tolist() == [False]
    assert targets["force_present_mask_B"].tolist() == [False]
    assert not targets["force_valid_mask_L"].any()


def test_nonfinite_labels_are_replaced_and_never_marked_valid():
    """NaN source labels cannot reach either numerical loss target."""
    data = _molecule_data(charge=0, spin=1)
    nonfinite_forces = np.ones((3, 3), dtype=np.float32)
    nonfinite_forces[1, 2] = np.nan
    data["calculator_results"] = {
        "energy": float("nan"),
        "forces": nonfinite_forces,
    }
    data = _extract_and_finalize(data)
    targets = data["ground_truth"]

    assert torch.isfinite(targets["energy_gt_B"]).all()
    assert torch.isfinite(targets["force_gt_D_L"]).all()
    assert not targets["energy_valid_mask_B"].item()
    assert not targets["force_valid_mask_L"].any()


@pytest.mark.parametrize(
    ("charge", "spin", "expected_charge_valid", "expected_spin_valid"),
    [
        (None, 1, False, True),
        (0, None, True, False),
        (0.5, 1, False, True),
        (0, 0, True, False),
        (float("nan"), 1, False, True),
    ],
)
def test_omol_charge_and_spin_are_required_and_integer_validated(
    charge,
    spin,
    expected_charge_valid,
    expected_spin_valid,
):
    """OMol property supervision requires a valid integer charge and multiplicity."""
    data = _extract_and_finalize(_molecule_data(charge=charge, spin=spin))
    targets = data["ground_truth"]
    assert targets["charge_valid_mask_B"].item() is expected_charge_valid
    assert targets["spin_valid_mask_B"].item() is expected_spin_valid
    assert not targets["property_valid_mask_B"].item()
    assert not targets["energy_valid_mask_B"].item()
    assert not targets["force_valid_mask_L"].any()
    expected_charge = torch.zeros(3, 1)
    expected_spin = torch.ones(3, 1) if expected_spin_valid else torch.zeros(3, 1)
    torch.testing.assert_close(data["feats"]["total_charge"], expected_charge)
    torch.testing.assert_close(data["feats"]["spin_multiplicity"], expected_spin)
    assert bool(data["feats"]["has_total_charge"].all()) is expected_charge_valid
    assert bool(data["feats"]["has_spin_multiplicity"].all()) is expected_spin_valid


def test_omat_domain_does_not_require_charge_or_spin():
    """OMat targets remain valid without molecule-only electronic metadata."""
    data = _molecule_data(charge=None, spin=None)
    data = AddAtomisticPropertyTargets(2, domain="omat24")(data)
    data = FinalizeAtomisticPropertyTargets(2)(data)
    targets = data["ground_truth"]

    assert targets["domain_id_B"].tolist() == [OMAT24_DOMAIN_ID]
    assert targets["property_valid_mask_B"].tolist() == [True]
    assert targets["energy_valid_mask_B"].tolist() == [True]
    assert targets["force_valid_mask_L"].all()
    assert not targets["charge_valid_mask_B"].item()
    assert not targets["spin_valid_mask_B"].item()
    torch.testing.assert_close(data["feats"]["total_charge"], torch.zeros(3, 1))
    torch.testing.assert_close(data["feats"]["spin_multiplicity"], torch.zeros(3, 1))
    assert not data["feats"]["has_total_charge"].any()
    assert not data["feats"]["has_spin_multiplicity"].any()


def test_crop_invalidates_energy_and_every_force_target():
    """A partial structure cannot supervise an extensive energy or its forces."""
    data = AddAtomisticPropertyTargets(2, domain="omol25")(_molecule_data())
    data["atom_array"] = data["atom_array"][[0, 2]]
    data = FinalizeAtomisticPropertyTargets(2)(data)
    targets = data["ground_truth"]

    assert targets["source_atom_count_B"].tolist() == [3]
    assert targets["retained_atom_count_B"].tolist() == [2]
    assert targets["property_valid_mask_B"].tolist() == [False]
    assert targets["energy_valid_mask_B"].tolist() == [False]
    assert not targets["force_valid_mask_L"].any()
    torch.testing.assert_close(targets["force_gt_D_L"], torch.zeros(2, 2, 3))


def _rotation_z_90() -> torch.Tensor:
    """Return a +90-degree column-vector rotation around z."""
    return torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def test_molecule_force_targets_use_each_exact_sampled_rotation():
    """Every diffusion view rotates forces by its coordinate-frame rotation."""
    data = _molecule_data(symbols="CO", charge=0, spin=1)
    atom_array = data["atom_array"]
    atom_array.set_annotation("is_motif_atom_with_fixed_coord", np.zeros(2, dtype=bool))
    rotations = torch.stack([torch.eye(3), _rotation_z_90()])
    forces = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]] * 2, dtype=torch.float32)
    coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]] * 2, dtype=torch.float32
    )
    data.update(
        is_inference=False,
        sampled_condition_name="unconditional_design",
        coord_atom_lvl_to_be_noised=coordinates,
        ground_truth={
            "mask_atom_lvl": torch.ones(2, dtype=torch.bool),
            "force_gt_D_L": forces.clone(),
        },
        feats={"is_motif_atom_unindexed": torch.zeros(2, dtype=torch.bool)},
    )
    rigid = (rotations, torch.zeros(2, 3))
    augmentation = MaterialAwareMotifCenterRandomAugmentation(
        batch_size=2, sigma_perturb=0.0, center_option="all"
    )
    with patch(
        "nucleus.transforms.material_transforms.get_random_rigid",
        return_value=rigid,
    ):
        data = augmentation.forward(data)

    expected_forces = torch.einsum("dij,dlj->dli", rotations, forces)
    torch.testing.assert_close(
        data["ground_truth"]["force_gt_D_L"], expected_forces, rtol=0, atol=0
    )


def test_periodic_origin_shift_does_not_change_force_targets():
    """Changing the periodic cell cut preserves forces before cell rotation."""
    data = _molecule_data(symbols="NaCl", charge=None, spin=None)
    atom_array = annotate_material_properties(
        data["atom_array"],
        fractional_coordinates=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        lattice_lengths=np.array([3.0, 3.0, 3.0]),
        lattice_angles=np.array([90.0, 90.0, 90.0]),
        lattice_vectors=np.eye(3) * 3.0,
    )
    rotations = torch.stack([torch.eye(3), _rotation_z_90()])
    forces = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]] * 2, dtype=torch.float32)
    data.update(
        atom_array=atom_array,
        is_inference=False,
        sampled_condition_name="unconditional_design",
        coord_atom_lvl_to_be_noised=torch.tensor(
            [[[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]]] * 2
        ),
        ground_truth={
            "force_gt_D_L": forces.clone(),
            "fractional_coord_gt_L": torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
            "fractional_coord_valid_mask_L": torch.ones(2, dtype=torch.bool),
        },
        feats={"is_motif_atom_with_fixed_coord": torch.zeros(2, dtype=torch.bool)},
    )
    torch.manual_seed(19)
    data = RandomPeriodicCellOriginShift(batch_size=2).forward(data)
    torch.testing.assert_close(
        data["ground_truth"]["force_gt_D_L"], forces, rtol=0, atol=0
    )

    augmentation = MaterialAwareMotifCenterRandomAugmentation(
        batch_size=2, sigma_perturb=0.0, center_option="all"
    )
    with patch(
        "nucleus.transforms.material_transforms.get_random_rots",
        return_value=rotations,
    ):
        data = augmentation.forward(data)
    expected_rotated_forces = torch.einsum("dij,dlj->dli", rotations, forces)
    torch.testing.assert_close(
        data["ground_truth"]["force_gt_D_L"],
        expected_rotated_forces,
        rtol=0,
        atol=0,
    )


def _make_collation_example(symbols: str, energy: float, offset: float) -> dict:
    """Build a complete sparse-collation example with atomistic targets."""
    n_atoms = len(Atoms(symbols))
    forces = np.arange(n_atoms * 3, dtype=np.float32).reshape(n_atoms, 3) + offset
    data = _extract_and_finalize(
        _molecule_data(
            symbols=symbols,
            energy=energy,
            forces=forces,
            charge=0,
            spin=1,
        )
    )
    data["t"] = torch.tensor([0.9, 1.0])
    data["ground_truth"].update(
        coord_atom_lvl=torch.zeros(n_atoms, 3),
        sequence_gt_I=torch.zeros(n_atoms, 32),
        is_original_unindexed_token=torch.zeros(n_atoms, dtype=torch.bool),
    )
    data["feats"].update(
        atom_to_token_map=torch.arange(n_atoms),
    )
    return data


def test_sparse_collate_handles_system_scalars_and_force_diffusion_axis():
    """System values stack over B and forces concatenate over L even when D=L."""
    first = _make_collation_example("CO", -1.0, 0.0)
    second = _make_collation_example("COH", -2.0, 10.0)
    collated = SparseCollate()([first, second])

    targets = collated["ground_truth"]
    assert targets["energy_gt_B"].shape == (2,)
    torch.testing.assert_close(
        targets["energy_gt_B"],
        torch.tensor([-1.0, -2.0], dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    assert targets["domain_id_B"].shape == (2,)
    assert targets["source_atom_count_B"].tolist() == [2, 3]
    assert targets["force_gt_D_L"].shape == (2, 5, 3)
    torch.testing.assert_close(
        targets["force_gt_D_L"][:, :2],
        first["ground_truth"]["force_gt_D_L"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        targets["force_gt_D_L"][:, 2:],
        second["ground_truth"]["force_gt_D_L"],
        rtol=0,
        atol=0,
    )


def test_sparse_collate_stacks_augmentation_rotations_over_systems():
    """Per-view rotation matrices stack over B and D."""
    first = _make_collation_example("CO", -1.0, 0.0)
    second = _make_collation_example("COH", -2.0, 10.0)
    first_rotations = torch.stack((torch.eye(3), -torch.eye(3)))
    second_rotations = torch.stack((torch.eye(3), 2 * torch.eye(3)))
    first["feats"]["augmentation_rotation_D"] = first_rotations
    second["feats"]["augmentation_rotation_D"] = second_rotations

    rotations = SparseCollate()([first, second])["feats"]["augmentation_rotation_D"]

    assert rotations.shape == (2, 2, 3, 3)
    torch.testing.assert_close(rotations[:, 0], first_rotations)
    torch.testing.assert_close(rotations[:, 1], second_rotations)


def test_sparse_collate_handles_mixed_fractional_coordinate_diffusion_axes():
    """Static molecule targets broadcast beside per-view material targets."""
    molecule = _make_collation_example("CO", -1.0, 0.0)
    material = _make_collation_example("COH", -2.0, 10.0)
    molecule_fractional = torch.tensor(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32
    )
    material_fractional = torch.tensor(
        [
            [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]],
            [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
        ],
        dtype=torch.float32,
    )
    molecule["ground_truth"].update(
        fractional_coord_gt_L=molecule_fractional,
        fractional_coord_valid_mask_L=torch.zeros(2, dtype=torch.bool),
    )
    material["ground_truth"].update(
        fractional_coord_gt_L=material_fractional,
        fractional_coord_valid_mask_L=torch.ones(3, dtype=torch.bool),
    )

    targets = SparseCollate()([molecule, material])["ground_truth"]

    assert targets["fractional_coord_gt_L"].shape == (2, 5, 3)
    torch.testing.assert_close(
        targets["fractional_coord_gt_L"][:, :2],
        molecule_fractional.unsqueeze(0).expand(2, -1, -1),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        targets["fractional_coord_gt_L"][:, 2:],
        material_fractional,
        rtol=0,
        atol=0,
    )
    assert targets["fractional_coord_valid_mask_L"].tolist() == [
        False,
        False,
        True,
        True,
        True,
    ]


def test_sparse_collate_keeps_static_fractional_coordinates_compact():
    """All-static fractional targets retain the ordinary [L, 3] contract."""
    first = _make_collation_example("CO", -1.0, 0.0)
    second = _make_collation_example("COH", -2.0, 10.0)
    first_fractional = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    second_fractional = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    first["ground_truth"]["fractional_coord_gt_L"] = first_fractional
    second["ground_truth"]["fractional_coord_gt_L"] = second_fractional

    targets = SparseCollate()([first, second])["ground_truth"]

    assert targets["fractional_coord_gt_L"].shape == (5, 3)
    torch.testing.assert_close(
        targets["fractional_coord_gt_L"],
        torch.cat([first_fractional, second_fractional]),
        rtol=0,
        atol=0,
    )


def test_sparse_collate_rejects_partial_property_contract():
    """Mixed optional-key intersection cannot silently erase property targets."""
    first = _make_collation_example("CO", -1.0, 0.0)
    second = _make_collation_example("COH", -2.0, 10.0)
    del second["ground_truth"]["spin_valid_mask_B"]

    with pytest.raises(KeyError, match="spin_valid_mask_B"):
        SparseCollate()([first, second])
