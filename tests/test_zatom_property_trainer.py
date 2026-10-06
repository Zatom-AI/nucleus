import numpy as np
import pytest
import torch
from biotite.structure import AtomArray

from nucleus.constants import (
    MLIP_PROPERTY_TASK_ID,
    STRUCTURE_PREDICTION_PROPERTY_TASK_ID,
    UNCONDITIONAL_PROPERTY_TASK_ID,
)
from nucleus.trainers.zatom.zatom import ZatomDesignTrainer
from nucleus.utils.forces import mean_force_norm_target_from_extra_info


@pytest.mark.parametrize("has_atomic_number_annotation", [True, False])
def test_generated_atom_types_reach_afd_with_matching_atomic_numbers(
    has_atomic_number_annotation,
):
    """AFD must see generated species while conditioned atoms stay unchanged."""
    from nucleus.utils.atom_array import _atom_array_to_afd_record

    atoms = AtomArray(3)
    atoms.coord = np.arange(9, dtype=np.float32).reshape(3, 3)
    atoms.element = ["C", "C", "O"]
    atoms.atom_name = ["C1", "C2", "O1"]
    atoms.res_name = ["LIG"] * 3
    atoms.res_id = [1] * 3
    atoms.set_annotation("atomize", np.ones(3, dtype=bool))
    if has_atomic_number_annotation:
        atoms.set_annotation("atomic_number", np.array([6, 6, 8], dtype=np.int8))

    result = ZatomDesignTrainer._apply_atom_type_predictions(
        atom_array=atoms.copy(),
        source_atom_array=atoms,
        atom_type_indices=torch.tensor([[7, 1, 9]]),
        sample_idx=0,
        generation_mask=torch.tensor([True, True, False]),
    )

    assert result.element.tolist() == ["N", "H", "O"]
    assert _atom_array_to_afd_record(result)["z"].tolist() == [7, 1, 8]
    if has_atomic_number_annotation:
        assert result.atomic_number.tolist() == [7, 1, 8]
        assert atoms.atomic_number.tolist() == [6, 6, 8]


def test_loss_extra_info_wires_all_atomistic_property_fields():
    """Test that the loss extra info includes all atomistic property fields."""
    property_targets = {
        "energy_gt_B": torch.tensor([-1.0, -2.0]),
        "energy_valid_mask_B": torch.tensor([True, False]),
        "energy_present_mask_B": torch.tensor([True, False]),
        "force_present_mask_B": torch.tensor([True, False]),
        "force_gt_D_L": torch.zeros((2, 2, 3)),
        "force_valid_mask_L": torch.tensor([True, False]),
        "domain_id_B": torch.tensor([0, 1]),
        "domain_valid_mask_B": torch.tensor([True, True]),
        "source_atom_count_B": torch.tensor([1, 1]),
        "retained_atom_count_B": torch.tensor([1, 1]),
        "total_charge_B": torch.tensor([0.0, 0.0]),
        "spin_multiplicity_B": torch.tensor([1.0, 0.0]),
        "has_total_charge_B": torch.tensor([True, False]),
        "has_spin_multiplicity_B": torch.tensor([True, False]),
        "charge_valid_mask_B": torch.tensor([True, False]),
        "spin_valid_mask_B": torch.tensor([True, False]),
        "property_valid_mask_B": torch.tensor([True, True]),
        "property_task_id_B": torch.tensor([0, 2]),
        "property_task_valid_mask_B": torch.tensor([True, True]),
    }
    example = {
        "coord_atom_lvl_to_be_noised": torch.zeros((2, 2, 3)),
        "ground_truth": {
            "coord_atom_lvl": torch.zeros((2, 3)),
            "mask_atom_lvl": torch.ones(2, dtype=torch.bool),
            "is_original_unindexed_token": torch.zeros(2, dtype=torch.bool),
            "sequence_gt_I": torch.zeros(2, dtype=torch.long),
            "sequence_valid_mask": torch.zeros(2, dtype=torch.bool),
            "atom_type_gt_I": torch.tensor([1, 8]),
            "is_original_unindexed_token_batch_offset": torch.tensor([0, 1]),
            "atom_to_token_map_per_batch_element": torch.tensor([0, 0]),
            **property_targets,
        },
        "feats": {
            "atom_to_token_map": torch.tensor([0, 1]),
            "batch_token_lvl": torch.tensor([0, 1]),
        },
    }
    trainer = object.__new__(ZatomDesignTrainer)

    loss_input = trainer._assemble_loss_extra_info(example)

    for key, expected in property_targets.items():
        assert torch.equal(loss_input[key], expected)


def test_metrics_extra_info_preserves_inference_specification():
    """Metric adapters can read conditioning targets from inference JSON."""
    trainer = object.__new__(ZatomDesignTrainer)
    trainer._assemble_loss_extra_info = lambda example: {}
    specification = {
        "mean_force_norm": [0.375],
        "has_mean_force_norm": [True],
    }
    example = {
        "example_id": ["validation-example"],
        "ground_truth": {},
        "specification": specification,
    }

    extra_info = trainer._assemble_metrics_extra_info(example, network_output={})

    assert extra_info["specification"] is specification
    assert mean_force_norm_target_from_extra_info(extra_info) == pytest.approx(0.375)


def test_atomistic_property_predictions_are_serialized_in_physical_units():
    """Test that the atomistic property predictions are serialized in physical units."""
    network_output = {
        "energy_B": torch.tensor([[-3.5], [-2.0]]),
        "forces_L": torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
            ]
        ),
    }

    metadata = ZatomDesignTrainer._atomistic_property_prediction_metadata(
        network_output,
        sample_index=1,
    )

    assert metadata == {
        "energy_eV": -2.0,
        "forces_eV_per_angstrom": [
            [-1.0, -2.0, -3.0],
            [-4.0, -5.0, -6.0],
        ],
    }


def test_mlip_property_inference_routing_applies_during_validation():
    """Test that the MLIP property inference routing applies during validation."""
    example = {
        "ground_truth": {
            "property_task_id_B": torch.tensor([MLIP_PROPERTY_TASK_ID]),
        }
    }

    assert ZatomDesignTrainer._is_mlip_property_inference(example)
    example["ground_truth"]["property_task_id_B"].zero_()
    assert not ZatomDesignTrainer._is_mlip_property_inference(example)


@pytest.mark.parametrize(
    ("task_id", "task_tag"),
    [
        (UNCONDITIONAL_PROPERTY_TASK_ID, "unconditional"),
        (STRUCTURE_PREDICTION_PROPERTY_TASK_ID, "structure_prediction"),
        (MLIP_PROPERTY_TASK_ID, "mlip"),
    ],
)
def test_atomistic_metric_tags_route_domain_and_task_metrics(task_id, task_tag):
    """Test that the atomistic metric tags route domain and task metrics correctly."""
    example = {
        "ground_truth": {
            "domain_id_B": torch.tensor([0]),
            "property_task_id_B": torch.tensor([task_id]),
        }
    }

    assert ZatomDesignTrainer._atomistic_metrics_tags(example) == {
        "omol25",
        "molecule",
        task_tag,
    }


@pytest.mark.parametrize(
    ("sampled_condition_name", "task_tag"),
    [
        ("unconditional", "unconditional"),
        (["unconditional_design"], "unconditional"),
        (("structure_prediction",), "structure_prediction"),
        ("sequence_fixed_structure_design", "structure_prediction"),
        ("mlip", "mlip"),
    ],
)
def test_atomistic_metric_tags_fall_back_to_sampled_condition_without_targets(
    sampled_condition_name,
    task_tag,
):
    """Generation metrics remain routed when property targets are disabled."""
    atom_array = AtomArray(1)
    atom_array.set_annotation("is_material", [True])
    example = {
        "ground_truth": {},
        "atom_array": atom_array,
        "sampled_condition_name": sampled_condition_name,
    }

    assert ZatomDesignTrainer._atomistic_metrics_tags(example) == {
        "omat24",
        "material",
        task_tag,
    }


@pytest.mark.parametrize(
    ("fixed_sequence", "fixed_coordinates", "task_tag"),
    [
        (False, False, "unconditional"),
        (True, False, "structure_prediction"),
    ],
)
def test_atomistic_metric_tags_infer_validation_task_from_conditioning_masks(
    fixed_sequence,
    fixed_coordinates,
    task_tag,
):
    """Validation routing works without task names or inference metadata."""
    atom_array = AtomArray(1)
    atom_array.set_annotation("is_material", [True])
    atom_array.set_annotation("is_motif_atom_with_fixed_seq", [fixed_sequence])
    atom_array.set_annotation("is_motif_atom_with_fixed_coord", [fixed_coordinates])
    example = {
        "ground_truth": {},
        "atom_array": atom_array,
    }

    assert ZatomDesignTrainer._atomistic_metrics_tags(example) == {
        "omat24",
        "material",
        task_tag,
    }


def test_property_inference_output_uses_one_clean_view_and_public_shapes():
    """Test that the property inference output uses one clean view and public shapes."""
    coordinates = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    property_input = ZatomDesignTrainer._single_view_property_input(
        {
            "X_clean_L": coordinates,
            "X_noisy_L": coordinates + 1,
            "f": {},
        }
    )

    output = ZatomDesignTrainer._format_property_inference_output(
        {
            "energy_B": torch.tensor([-1.25]),
            "normalized_energy_B": torch.tensor([0.5]),
            "forces_L": torch.zeros((4, 3)),
            "normalized_forces_L": torch.ones((4, 3)),
        },
        property_input=property_input,
    )

    torch.testing.assert_close(property_input["X_clean_L"], coordinates[:1])
    torch.testing.assert_close(property_input["X_noisy_L"], coordinates[0:1] + 1)
    assert output["X_L"].shape == (1, 4, 3)
    assert output["energy_B"].shape == (1, 1)
    assert output["forces_L"].shape == (1, 4, 3)
    assert len(output["X_noisy_L_traj"]) == 1
    assert len(output["X_denoised_L_traj"]) == 1
