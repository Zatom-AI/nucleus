"""Tests for the OMat24/OMol25 atomized-ligand fast path."""

import random

import biotite.structure as struc
import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from atomworks.ml.datasets.loaders import (
    create_ase_atoms_loader,
    create_ase_materials_loader,
)
from rfd3na.transforms.conditioning_base import set_default_conditioning_annotations

from nucleus.constants import (
    MLIP_PROPERTY_TASK_ID,
    STRUCTURE_PREDICTION_PROPERTY_TASK_ID,
    UNCONDITIONAL_PROPERTY_TASK_ID,
)
from nucleus.transforms.pipelines import build_atom14_base_pipeline


def _ligand_condition(molecule_conditioning_mode, *, condition_name=None):
    """Returns a dictionary of training conditions for ligand-aware island conditioning."""
    return {
        "_target_": (
            "nucleus.transforms.training_conditions.LigandAwareIslandCondition"
        ),
        "frequency": 1.0,
        "name": condition_name or molecule_conditioning_mode,
        "molecule_conditioning_mode": molecule_conditioning_mode,
        "island_sampling_kwargs": {
            "island_len_min": 0,
            "island_len_max": 0,
            "n_islands_min": 0,
            "n_islands_max": 0,
        },
        "p_diffuse_motif_sidechains": 0.0,
        "p_diffuse_subgraph_atoms": 0.0,
        "subgraph_sampling_kwargs": {
            "residue_p_seed_furthest_from_o": None,
            "residue_n_bond_expectation": None,
            "residue_p_fix_all": None,
            "hetatom_n_bond_expectation": None,
            "hetatom_p_fix_all": None,
        },
        "p_fix_motif_coordinates": 0.0,
        "p_fix_motif_sequence": 0.0,
        "p_unindex_motif_tokens": 0.0,
    }


def _pipeline(
    *,
    optimized,
    is_inference=False,
    molecule_conditioning_mode="unconditional_design",
    condition_name=None,
    enable_atomistic_property_targets=False,
    randomly_augment=False,
    diffusion_batch_size=1,
    mean_force_norm_conditioning_probability=0.0,
    mean_force_norm_conditioning_tasks=None,
    rms_and_p95_conditioning=False,
    record_rotation_pairs=False,
    atomistic_property_domain="omol25",
):
    """Builds a pipeline for atomized-ligand training or inference."""
    disabled_conditioning = {
        name: 0.0
        for name in (
            "calculate_hbonds",
            "calculate_rasa",
            "keep_protein_motif_rasa",
            "hbond_subsample",
            "unindex_leak_global_index",
            "unindex_insert_random_break",
            "unindex_remove_random_break",
            "add_1d_ss_features",
            "featurize_plddt",
            "add_global_is_non_loopy_feature",
            "add_ppi_hotspots",
            "full_binder_crop",
        )
    }
    return build_atom14_base_pipeline(
        is_inference=is_inference,
        return_atom_array=False,
        allowed_types=["is_ligand"],
        crop_size=32,
        crop_center_cutoff_distance=15.0,
        crop_contiguous_probability=0.0,
        crop_spatial_probability=1.0,
        dna_contact_crop_probability=0.0,
        keep_full_binder_in_spatial_crop=False,
        max_binder_length=170,
        max_atoms_in_crop=320,
        b_factor_min=None,
        zero_occ_on_exposure_after_cropping=False,
        sigma_data=16.0,
        diffusion_batch_size=diffusion_batch_size,
        generate_conformers=False,
        generate_conformers_for_non_protein_only=False,
        provide_reference_conformer_when_unmasked=False,
        ground_truth_conformer_policy="IGNORE",
        provide_elements_for_unindexed_components=True,
        use_element_for_atom_names_of_atomized_tokens=True,
        residue_cache_dir=None,
        train_conditions={
            "ligand_condition": _ligand_condition(
                molecule_conditioning_mode,
                condition_name=condition_name,
            )
        },
        meta_conditioning_probabilities=disabled_conditioning,
        n_atoms_per_token=14,
        central_atom="CB",
        sigma_perturb=2.0,
        sigma_perturb_com=1.0,
        association_scheme="dense",
        center_option="diffuse",
        atom_1d_features={},
        token_1d_features=(
            {
                "mean_force_norm": None,
                "has_mean_force_norm": None,
                "force_rms_p95": 2,
                "has_force_rms_p95": 1,
            }
            if rms_and_p95_conditioning
            else {}
        ),
        token_2d_features=None,
        max_ppi_hotspots_frac_to_provide=0.2,
        ppi_hotspot_max_distance=4.5,
        max_ss_frac_to_provide=0.4,
        min_ss_island_len=1,
        max_ss_island_len=10,
        randomly_augment=randomly_augment,
        remove_hydrogens=False,
        noise_process="flow_matching",
        mean_force_norm_conditioning_probability=(
            mean_force_norm_conditioning_probability
        ),
        mean_force_norm_conditioning_tasks=mean_force_norm_conditioning_tasks,
        optimize_atomized_ligand_pipeline=optimized,
        enable_atomistic_property_targets=enable_atomistic_property_targets,
        record_rotation_pairs=record_rotation_pairs,
        atomistic_property_domain=atomistic_property_domain,
    )


def _run_pipeline(
    pipeline,
    *,
    with_properties=False,
    bond_edges=None,
    inference_fixed_sequence=None,
    inference_fixed_coordinates=None,
    material=False,
    coordinate_shift=None,
    wrap_material_fractional_coordinates=True,
):
    """Run a pipeline on a synthetic atomized molecule or material."""
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    if material:
        atoms = Atoms(
            "NaClNa",
            scaled_positions=[
                [0.10, 0.20, 0.30],
                [0.35, 0.75, 0.55],
                [0.80, 0.40, 0.15],
            ],
            cell=np.diag([4.0, 5.0, 6.0]),
            pbc=True,
        )
    else:
        atoms = Atoms(
            "COH",
            positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [-0.5, 0.8, 0.0]],
        )
    if coordinate_shift is not None:
        atoms.translate(np.asarray(coordinate_shift, dtype=float))
    if with_properties:
        atoms.info.update(charge=0, spin=1)
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=-4.5,
            forces=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
        )
    if material:
        loader = create_ase_materials_loader(
            res_name="LIG",
            wrap_fractional_coordinates=wrap_material_fractional_coordinates,
            include_atom_array=True,
            include_chain_info=True,
            keep_ase_atoms=True,
        )
    else:
        loader = create_ase_atoms_loader(
            res_name="LIG",
            include_chain_info=True,
            keep_ase_atoms=True,
        )
    data = loader(atoms)
    data["example_id"] = "synthetic_material" if material else "synthetic_ligand"
    if inference_fixed_sequence is not None:
        if inference_fixed_coordinates is None:
            raise ValueError(
                "inference_fixed_coordinates is required with "
                "inference_fixed_sequence."
            )
        atom_array = set_default_conditioning_annotations(
            data["atom_array"], motif=False
        )
        atom_array.is_motif_atom_with_fixed_seq[:] = inference_fixed_sequence
        atom_array.is_motif_atom_with_fixed_coord[:] = inference_fixed_coordinates
    if bond_edges is not None:
        data["atom_array"].bonds = struc.BondList(
            len(atoms),
            np.asarray(bond_edges, dtype=int),
        )
    return pipeline(data)


def _assert_nested_arrays_equal(expected, actual):
    """Recursively asserts that two nested dictionaries of arrays are equal."""
    assert expected.keys() == actual.keys()
    for key in expected:
        expected_value = expected[key]
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            _assert_nested_arrays_equal(expected_value, actual_value)
        elif torch.is_tensor(expected_value):
            torch.testing.assert_close(
                actual_value,
                expected_value,
                rtol=0,
                atol=0,
                equal_nan=True,
            )
        elif isinstance(expected_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value)


@pytest.mark.parametrize(
    "molecule_conditioning_mode",
    [
        "unconditional_design",
        "structure_fixed_sequence_design",
        "sequence_fixed_structure_design",
    ],
)
def test_atomized_ligand_fast_path_preserves_training_features(
    molecule_conditioning_mode,
):
    """The fast path must retain exact seeded feature and target tensors."""
    expected = _run_pipeline(
        _pipeline(
            optimized=False,
            molecule_conditioning_mode=molecule_conditioning_mode,
        )
    )
    actual = _run_pipeline(
        _pipeline(
            optimized=True,
            molecule_conditioning_mode=molecule_conditioning_mode,
        )
    )

    assert actual.keys() == expected.keys()
    _assert_nested_arrays_equal(expected["feats"], actual["feats"])
    _assert_nested_arrays_equal(expected["ground_truth"], actual["ground_truth"])
    torch.testing.assert_close(actual["t"], expected["t"], rtol=0, atol=0)
    torch.testing.assert_close(actual["noise"], expected["noise"], rtol=0, atol=0)
    torch.testing.assert_close(
        actual["coord_atom_lvl_to_be_noised"],
        expected["coord_atom_lvl_to_be_noised"],
        rtol=0,
        atol=0,
    )


def test_property_targets_are_opt_in_but_electronic_conditioning_is_not():
    """Generative pipelines condition on charge/spin without property tensors."""
    output = _run_pipeline(_pipeline(optimized=True), with_properties=True)
    assert "energy_gt_B" not in output["ground_truth"]
    assert "force_gt_D_L" not in output["ground_truth"]
    torch.testing.assert_close(output["feats"]["total_charge"], torch.zeros(3, 1))
    torch.testing.assert_close(output["feats"]["spin_multiplicity"], torch.ones(3, 1))
    assert output["feats"]["has_total_charge"].all()
    assert output["feats"]["has_spin_multiplicity"].all()


@pytest.mark.parametrize(
    ("fixed_sequence", "fixed_coordinates"),
    [(False, False), (True, False), (True, True)],
    ids=("unconditional", "structure_prediction", "mlip"),
)
def test_inference_all_tasks_receive_electronic_conditioning_without_targets(
    fixed_sequence,
    fixed_coordinates,
):
    """Every inference task receives charge/spin independently of target setup."""
    output = _run_pipeline(
        _pipeline(
            optimized=False,
            is_inference=True,
            enable_atomistic_property_targets=False,
        ),
        with_properties=True,
        inference_fixed_sequence=fixed_sequence,
        inference_fixed_coordinates=fixed_coordinates,
    )

    assert "energy_gt_B" not in output["ground_truth"]
    assert "force_gt_D_L" not in output["ground_truth"]
    torch.testing.assert_close(output["feats"]["total_charge"], torch.zeros(3, 1))
    torch.testing.assert_close(output["feats"]["spin_multiplicity"], torch.ones(3, 1))
    assert output["feats"]["has_total_charge"].all()
    assert output["feats"]["has_spin_multiplicity"].all()


def test_property_pipeline_rotates_each_diffusion_force_view():
    """Finalized source forces survive the full pipeline as per-view vectors."""
    output = _run_pipeline(
        _pipeline(
            optimized=True,
            enable_atomistic_property_targets=True,
            randomly_augment=True,
            diffusion_batch_size=2,
        ),
        with_properties=True,
    )
    targets = output["ground_truth"]

    assert targets["energy_gt_B"].tolist() == [-4.5]
    assert targets["energy_valid_mask_B"].tolist() == [True]
    assert targets["force_gt_D_L"].shape == (2, 3, 3)
    assert targets["force_valid_mask_L"].tolist() == [True, True, True]
    expected_norms = torch.tensor([[1.0, 2.0, 3.0]]).expand(2, -1)
    torch.testing.assert_close(
        torch.linalg.vector_norm(targets["force_gt_D_L"], dim=-1),
        expected_norms,
        rtol=1e-6,
        atol=1e-6,
    )
    assert not torch.equal(targets["force_gt_D_L"][0], targets["force_gt_D_L"][1])
    assert output["feats"]["total_charge"].shape == (3, 1)
    assert output["feats"]["spin_multiplicity"].shape == (3, 1)


def test_mlip_task_provides_clean_structure_and_electronic_metadata():
    """MLIP inputs fix atom types/coordinates and expose charge and spin."""
    output = _run_pipeline(
        _pipeline(
            optimized=True,
            molecule_conditioning_mode="mlip",
            enable_atomistic_property_targets=True,
            diffusion_batch_size=2,
        ),
        with_properties=True,
    )
    feats = output["feats"]
    targets = output["ground_truth"]

    assert output["sampled_condition_name"] == "mlip"
    assert feats["is_motif_atom_with_fixed_seq"].all()
    assert feats["is_motif_atom_with_fixed_coord"].all()
    assert not feats["atom_type_generation_mask"].any()
    torch.testing.assert_close(output["noise"], torch.zeros_like(output["noise"]))
    assert targets["property_task_id_B"].item() == MLIP_PROPERTY_TASK_ID
    assert feats["has_total_charge"].all()
    assert feats["has_spin_multiplicity"].all()
    torch.testing.assert_close(feats["spin_multiplicity"], torch.ones(3, 1))
    assert not feats["ref_pos"].any()
    assert not feats["ref_mask"].any()
    assert not feats["ref_pos_is_ground_truth"].any()


@pytest.mark.parametrize("randomly_augment", [False, True])
@pytest.mark.parametrize("material", [False, True])
def test_mlip_inference_matches_deterministic_evaluation_coordinate_gauge(
    randomly_augment,
    material,
):
    """Direct inference and evaluate-mlip use one translation-stable gauge."""
    domain = "omat24" if material else "omol25"
    evaluation = _run_pipeline(
        _pipeline(
            optimized=False,
            molecule_conditioning_mode="mlip",
            enable_atomistic_property_targets=True,
            randomly_augment=False,
            atomistic_property_domain=domain,
        ),
        with_properties=True,
        material=material,
    )
    inference_pipeline = _pipeline(
        optimized=False,
        is_inference=True,
        enable_atomistic_property_targets=True,
        randomly_augment=randomly_augment,
        atomistic_property_domain=domain,
    )
    inference_kwargs = {
        "with_properties": True,
        "material": material,
        "inference_fixed_sequence": True,
        "inference_fixed_coordinates": True,
    }
    inference = _run_pipeline(inference_pipeline, **inference_kwargs)
    translated_inference = _run_pipeline(
        inference_pipeline,
        coordinate_shift=[7.25, -3.5, 2.75],
        # Preserve the translated Cartesian representative so this exercises
        # gauge fixing rather than the ASE loader's optional cell wrapping.
        wrap_material_fractional_coordinates=not material,
        **inference_kwargs,
    )

    for output in (inference, translated_inference):
        assert output["ground_truth"]["property_task_id_B"].item() == (
            MLIP_PROPERTY_TASK_ID
        )
        torch.testing.assert_close(
            output["coord_atom_lvl_to_be_noised"],
            evaluation["coord_atom_lvl_to_be_noised"],
            rtol=0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            output["feats"]["motif_pos"],
            evaluation["feats"]["motif_pos"],
            rtol=0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            output["feats"]["ref_pos"],
            evaluation["feats"]["ref_pos"],
            rtol=0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            output["coord_atom_lvl_to_be_noised"].mean(dim=1),
            torch.zeros((1, 3)),
            rtol=0,
            atol=1e-6,
        )
        if material:
            torch.testing.assert_close(
                output["ground_truth"]["fractional_coord_gt_L"],
                evaluation["ground_truth"]["fractional_coord_gt_L"],
                rtol=0,
                atol=1e-6,
            )


@pytest.mark.parametrize("randomly_augment", [False, True])
def test_mlip_inference_gauge_does_not_center_generative_inference(
    randomly_augment,
):
    """The inference-only gauge transform must not alter sampler tasks."""
    output = _run_pipeline(
        _pipeline(
            optimized=False,
            is_inference=True,
            enable_atomistic_property_targets=True,
            randomly_augment=randomly_augment,
        ),
        with_properties=True,
        inference_fixed_sequence=False,
        inference_fixed_coordinates=False,
    )

    torch.testing.assert_close(
        output["coord_atom_lvl_to_be_noised"],
        torch.tensor([[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [-0.5, 0.8, 0.0]]]),
    )
    assert output["ground_truth"]["property_task_id_B"].item() == (
        UNCONDITIONAL_PROPERTY_TASK_ID
    )


def test_material_mlip_augmentation_keeps_reference_conformer_absent():
    """Material MLIP coordinates must not leak into the reference conformer."""
    output = _run_pipeline(
        _pipeline(
            optimized=True,
            molecule_conditioning_mode="mlip",
            enable_atomistic_property_targets=True,
            randomly_augment=True,
            diffusion_batch_size=2,
            record_rotation_pairs=True,
            atomistic_property_domain="omat24",
        ),
        with_properties=True,
        material=True,
    )
    feats = output["feats"]
    coordinates = output["coord_atom_lvl_to_be_noised"]
    canonical_coordinates = output["ground_truth"]["coord_atom_lvl"]

    assert output["sampled_condition_name"] == "mlip"
    assert feats["material_atom_mask"].all()
    assert output["ground_truth"]["fractional_coord_gt_L"].shape == coordinates.shape
    assert not feats["ref_pos"].any()
    assert not feats["ref_mask"].any()
    assert not feats["ref_pos_is_ground_truth"].any()

    # ``motif_pos`` represents the supplied fixed structure, whereas ``ref_pos``
    # is an absent, semantically independent reference conformer.
    torch.testing.assert_close(feats["motif_pos"], coordinates[0])
    assert not torch.equal(feats["motif_pos"], canonical_coordinates)
    motif_pair_distances = torch.cdist(feats["motif_pos"], feats["motif_pos"])
    for view_coordinates in coordinates:
        torch.testing.assert_close(
            torch.cdist(view_coordinates, view_coordinates),
            motif_pair_distances,
            rtol=1e-5,
            atol=1e-5,
        )

    # Canonical coordinates remain available as targets/metadata but are not the
    # augmented coordinates supplied to the network.
    torch.testing.assert_close(
        canonical_coordinates,
        torch.tensor(
            [[0.4, 1.0, 1.8], [1.4, 3.75, 3.3], [3.2, 2.0, 0.9]],
            dtype=canonical_coordinates.dtype,
        ),
    )


@pytest.mark.parametrize(
    ("mode", "expected_task_id"),
    [
        ("unconditional_design", UNCONDITIONAL_PROPERTY_TASK_ID),
        ("sequence_fixed_structure_design", STRUCTURE_PREDICTION_PROPERTY_TASK_ID),
    ],
)
def test_generative_property_tasks_expose_electronic_metadata(
    mode,
    expected_task_id,
):
    """Generative OMol tasks receive source charge and spin during training."""
    output = _run_pipeline(
        _pipeline(
            optimized=True,
            molecule_conditioning_mode=mode,
            enable_atomistic_property_targets=True,
        ),
        with_properties=True,
    )
    feats = output["feats"]
    targets = output["ground_truth"]

    assert targets["property_task_id_B"].item() == expected_task_id
    torch.testing.assert_close(feats["total_charge"], torch.zeros(3, 1))
    torch.testing.assert_close(feats["spin_multiplicity"], torch.ones(3, 1))
    assert feats["has_total_charge"].all()
    assert feats["has_spin_multiplicity"].all()


@pytest.mark.parametrize(
    ("mode", "condition_name", "expect_conditioning"),
    [
        ("unconditional_design", "unconditional", True),
        ("sequence_fixed_structure_design", "structure_prediction", True),
        ("mlip", "mlip", False),
    ],
)
def test_mean_force_norm_conditioning_is_scoped_to_configured_tasks(
    mode,
    condition_name,
    expect_conditioning,
):
    """The joint experiment force-conditions both generative training tasks."""
    output = _run_pipeline(
        _pipeline(
            optimized=True,
            molecule_conditioning_mode=mode,
            condition_name=condition_name,
            enable_atomistic_property_targets=True,
            mean_force_norm_conditioning_probability=1.0,
            mean_force_norm_conditioning_tasks=[
                "unconditional",
                "structure_prediction",
            ],
        ),
        with_properties=True,
    )

    assert bool(output["feats"]["has_mean_force_norm"].all()) is expect_conditioning


@pytest.mark.parametrize("optimized", [False, True])
def test_rms_and_p95_conditioning_is_selected_from_model_features(optimized):
    """The standard pipeline follows the same model feature switch as inference."""
    output = _run_pipeline(
        _pipeline(
            optimized=optimized,
            mean_force_norm_conditioning_probability=1.0,
            rms_and_p95_conditioning=True,
        ),
        with_properties=True,
    )

    magnitudes = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    expected = torch.log(
        torch.stack(
            [
                torch.sqrt(torch.mean(magnitudes.square())),
                torch.quantile(magnitudes, 0.95),
            ]
        )
        + 1e-8
    )
    torch.testing.assert_close(
        output["feats"]["force_rms_p95"][0].double(),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )
    assert output["feats"]["has_force_rms_p95"].all()
    assert "mean_force_norm" not in output["feats"]


@pytest.mark.parametrize(
    ("fixed_sequence", "fixed_coordinates", "expected_task_id"),
    [
        (False, False, UNCONDITIONAL_PROPERTY_TASK_ID),
        (True, False, STRUCTURE_PREDICTION_PROPERTY_TASK_ID),
        (True, True, MLIP_PROPERTY_TASK_ID),
    ],
)
def test_inference_property_task_is_inferred_from_conditioning_masks(
    fixed_sequence,
    fixed_coordinates,
    expected_task_id,
):
    """Validation routes tasks even when design specs omit an explicit task."""
    output = _run_pipeline(
        _pipeline(
            optimized=False,
            is_inference=True,
            enable_atomistic_property_targets=True,
        ),
        with_properties=True,
        inference_fixed_sequence=fixed_sequence,
        inference_fixed_coordinates=fixed_coordinates,
    )

    assert output["sampled_condition_name"] is None
    assert output["ground_truth"]["property_task_id_B"].item() == expected_task_id


@pytest.mark.parametrize(
    ("fixed_sequence", "fixed_coordinates", "expect_conditioning"),
    [
        (False, False, False),
        (True, False, True),
        (True, True, False),
    ],
)
def test_inference_mean_force_norm_scope_uses_inferred_task(
    fixed_sequence,
    fixed_coordinates,
    expect_conditioning,
):
    """Inference task scoping does not leak force conditioning across tasks."""
    output = _run_pipeline(
        _pipeline(
            optimized=False,
            is_inference=True,
            enable_atomistic_property_targets=True,
            mean_force_norm_conditioning_probability=1.0,
            mean_force_norm_conditioning_tasks=["structure_prediction"],
        ),
        with_properties=True,
        inference_fixed_sequence=fixed_sequence,
        inference_fixed_coordinates=fixed_coordinates,
    )

    assert bool(output["feats"]["has_mean_force_norm"].all()) is expect_conditioning


def test_property_augmentation_preserves_seeded_coordinate_path():
    """Capturing rotations for forces does not perturb legacy coordinate RNG."""
    legacy = _run_pipeline(
        _pipeline(
            optimized=True,
            randomly_augment=True,
            diffusion_batch_size=2,
        ),
        with_properties=True,
    )
    joint = _run_pipeline(
        _pipeline(
            optimized=True,
            enable_atomistic_property_targets=True,
            randomly_augment=True,
            diffusion_batch_size=2,
        ),
        with_properties=True,
    )

    torch.testing.assert_close(joint["t"], legacy["t"], rtol=0, atol=0)
    torch.testing.assert_close(joint["noise"], legacy["noise"], rtol=0, atol=0)
    torch.testing.assert_close(
        joint["coord_atom_lvl_to_be_noised"],
        legacy["coord_atom_lvl_to_be_noised"],
        rtol=0,
        atol=0,
    )


def test_recorded_mlip_views_are_exact_rotation_pairs():
    """The paired views differ only by the recorded relative SO(3) action."""
    output = _run_pipeline(
        _pipeline(
            optimized=True,
            molecule_conditioning_mode="mlip",
            enable_atomistic_property_targets=True,
            randomly_augment=True,
            diffusion_batch_size=2,
            record_rotation_pairs=True,
        ),
        with_properties=True,
    )

    coordinates = output["coord_atom_lvl_to_be_noised"]
    rotations = output["feats"]["augmentation_rotation_D"]
    relative_rotation = rotations[1] @ rotations[0].T
    expected_second_view = torch.einsum("ij,lj->li", relative_rotation, coordinates[0])

    torch.testing.assert_close(coordinates[1], expected_second_view)


def test_atomized_ligand_fast_path_rejects_polymer_input():
    """The optimized path should fail clearly if its loader contract is violated."""
    pipeline = _pipeline(optimized=True)
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    loader = create_ase_atoms_loader(
        res_name="LIG",
        is_polymer=True,
        include_chain_info=True,
    )
    data = loader(atoms)
    data["example_id"] = "invalid_polymer"

    with pytest.raises(ValueError, match="does not support polymer atoms"):
        pipeline(data)


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize(
    ("condition_name", "molecule_conditioning_mode", "expects_token_bonds"),
    [
        ("structure_prediction", "sequence_fixed_structure_design", True),
        ("unconditional", "unconditional_design", False),
        ("sequence_design", "structure_fixed_sequence_design", False),
    ],
)
def test_token_bonds_are_only_featurized_for_structure_prediction(
    optimized,
    condition_name,
    molecule_conditioning_mode,
    expects_token_bonds,
):
    """Only structure prediction may expose loaded bonds to the model."""
    output = _run_pipeline(
        _pipeline(
            optimized=optimized,
            molecule_conditioning_mode=molecule_conditioning_mode,
            condition_name=condition_name,
        ),
        bond_edges=[[0, 1]],
    )

    assert ("token_bonds" in output["feats"]) is expects_token_bonds
    if expects_token_bonds:
        token_bonds = output["feats"]["token_bonds"]
        assert token_bonds.dtype == torch.bool
        assert token_bonds.sum().item() == 2
        assert token_bonds[0, 1]
        assert token_bonds[1, 0]


@pytest.mark.parametrize(
    ("fixed_sequence", "fixed_coordinates", "expects_token_bonds"),
    [
        (True, False, True),
        (False, True, False),
        (False, False, False),
        (True, True, False),
    ],
)
def test_inference_token_bonds_are_only_featurized_for_structure_prediction(
    fixed_sequence,
    fixed_coordinates,
    expects_token_bonds,
):
    """Inference derives structure prediction from canonical conditioning masks."""
    output = _run_pipeline(
        _pipeline(optimized=False, is_inference=True),
        bond_edges=[[0, 1]],
        inference_fixed_sequence=fixed_sequence,
        inference_fixed_coordinates=fixed_coordinates,
    )

    assert output["sampled_condition_name"] is None
    assert ("token_bonds" in output["feats"]) is expects_token_bonds
    if expects_token_bonds:
        token_bonds = output["feats"]["token_bonds"]
        assert token_bonds.dtype == torch.bool
        assert token_bonds.sum().item() == 2
        assert token_bonds[0, 1]
        assert token_bonds[1, 0]
