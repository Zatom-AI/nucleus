import gzip
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from atomworks.ml.utils.token import get_token_starts
from biotite.structure.io.pdbx import CIFFile, get_structure
from nucleus.callbacks.dump_validation_structures import (
    DumpValidationStructuresCallback,
)
from nucleus.inference_input_parsing import (
    accumulate_components_with_molecules,
    create_diffused_material_atoms,
)
from nucleus.metrics.losses import (
    MaterialGeometryLoss,
    SequenceLoss,
    compute_loss_output_gradient_l2_norms,
    pop_loss_output_gradient_diagnostics,
)
from nucleus.trainers.zatom import ZatomDesignTrainer
from nucleus.transforms.material_transforms import AddMaterialPredictionTargets
from nucleus.utils.materials import (
    add_fractional_cif_fields,
    annotate_material_properties,
    fractional_to_cartesian,
    lattice_vectors_from_lengths_angles,
    normalize_lattice_lengths,
    wrap_cartesian_coordinates,
)

from zatom2.run_inference import (
    Zatom2Output,
    _rewrite_inputless_molecule_contig,
)


def _material_loss_inputs(t=0.25):
    """
    Create dummy inputs for testing the MaterialGeometryLoss. The time t can be
    set to test the loss behavior within and outside the specified time window.
    """
    network_input = {
        "t": torch.full((1, 1), t),
        "t_L": torch.full((1, 2), t),
        "t_I": torch.full((1, 1), t),
        "f": {
            "batch_atom_lvl": torch.tensor([0, 0]),
            "diffusion_batch_size": 1,
            "batch_size": 1,
        },
    }
    loss_input = {
        "fractional_coord_gt_L": torch.zeros((2, 3)),
        "fractional_coord_valid_mask_L": torch.ones(2, dtype=torch.bool),
        "lattice_lengths_scaled_gt_B": torch.zeros((1, 3)),
        "lattice_angles_radians_gt_B": torch.zeros((1, 3)),
        "lattice_valid_mask_B": torch.ones(1, dtype=torch.bool),
    }
    return network_input, loss_input


def test_material_geometry_loss_is_zero_for_exact_predictions():
    """
    When the network outputs match the ground truth fractional coordinates and
    lattice parameters, the MaterialGeometryLoss should be zero.
    """
    network_input, loss_input = _material_loss_inputs()
    network_output = {
        "X_L": torch.zeros((1, 2, 3)),
        "fractional_coord_L": torch.zeros((1, 2, 3)),
        "lattice_lengths_B": torch.zeros((1, 1, 3)),
        "lattice_angles_B": torch.zeros((1, 1, 3)),
    }

    loss, loss_dict = MaterialGeometryLoss()(network_input, network_output, loss_input)

    assert loss.item() == 0.0
    assert loss_dict["material_fractional_coord_loss"].item() == 0.0
    assert loss_dict["material_lattice_lengths_loss"].item() == 0.0
    assert loss_dict["material_lattice_angles_loss"].item() == 0.0


def test_material_geometry_loss_respects_time_window():
    """
    When the network outputs are outside the specified time window, the
    MaterialGeometryLoss should be zero.
    """
    network_input, loss_input = _material_loss_inputs(t=0.75)
    network_output = {
        "X_L": torch.zeros((1, 2, 3)),
        "fractional_coord_L": torch.ones((1, 2, 3), requires_grad=True),
        "lattice_lengths_B": torch.ones((1, 1, 3), requires_grad=True),
        "lattice_angles_B": torch.ones((1, 1, 3), requires_grad=True),
    }

    loss, loss_dict = MaterialGeometryLoss(max_t=0.5)(
        network_input,
        network_output,
        loss_input,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert loss_dict["material_fractional_coord_loss"].item() == 0.0
    assert loss_dict["material_lattice_lengths_loss"].item() == 0.0
    assert loss_dict["material_lattice_angles_loss"].item() == 0.0
    assert torch.count_nonzero(network_output["fractional_coord_L"].grad).item() == 0
    assert torch.count_nonzero(network_output["lattice_lengths_B"].grad).item() == 0
    assert torch.count_nonzero(network_output["lattice_angles_B"].grad).item() == 0


def test_material_geometry_loss_averages_only_valid_time_entries():
    """Invalid-time atoms and systems are excluded from geometry loss means."""
    network_input = {
        "t": torch.tensor([[0.25, 0.75], [0.75, 0.25]]),
        "t_L": torch.tensor(
            [
                [0.25, 0.25, 0.75, 0.75],
                [0.75, 0.75, 0.25, 0.25],
            ]
        ),
        "f": {
            "batch_atom_lvl": torch.tensor([0, 0, 1, 1]),
            "diffusion_batch_size": 2,
            "batch_size": 2,
        },
    }
    loss_input = {
        "fractional_coord_gt_L": torch.zeros((4, 3)),
        "fractional_coord_valid_mask_L": torch.ones(4, dtype=torch.bool),
        "lattice_lengths_scaled_gt_B": torch.zeros((2, 3)),
        "lattice_angles_radians_gt_B": torch.zeros((2, 3)),
        "lattice_valid_mask_B": torch.ones(2, dtype=torch.bool),
    }

    fractional_coord = torch.full((2, 4, 3), 0.49)
    fractional_coord[0, 0] = 0.1
    fractional_coord[0, 1] = 0.3
    fractional_coord[1, 2] = 0.2
    fractional_coord[1, 3] = 0.4
    fractional_coord.requires_grad_()

    lattice_lengths = torch.full((2, 2, 3), 10.0)
    lattice_lengths[0, 0] = 0.2
    lattice_lengths[1, 1] = 0.4
    lattice_lengths.requires_grad_()

    lattice_angles = torch.full((2, 2, 3), 10.0)
    lattice_angles[0, 0] = 0.1
    lattice_angles[1, 1] = 0.3
    lattice_angles.requires_grad_()

    network_output = {
        "X_L": torch.zeros((2, 4, 3)),
        "fractional_coord_L": fractional_coord,
        "lattice_lengths_B": lattice_lengths,
        "lattice_angles_B": lattice_angles,
    }

    loss, loss_dict = MaterialGeometryLoss(max_t=0.5)(
        network_input,
        network_output,
        loss_input,
    )
    loss.backward()

    torch.testing.assert_close(
        loss_dict["material_fractional_coord_loss"], torch.tensor(0.075)
    )
    torch.testing.assert_close(
        loss_dict["material_lattice_lengths_loss"], torch.tensor(0.1)
    )
    torch.testing.assert_close(
        loss_dict["material_lattice_angles_loss"], torch.tensor(0.05)
    )
    torch.testing.assert_close(loss, torch.tensor(0.225))

    valid_systems = torch.tensor([[True, False], [False, True]])
    valid_atoms = valid_systems[:, network_input["f"]["batch_atom_lvl"]]
    assert torch.count_nonzero(fractional_coord.grad[valid_atoms]).item() > 0
    assert torch.count_nonzero(fractional_coord.grad[~valid_atoms]).item() == 0
    assert torch.count_nonzero(lattice_lengths.grad[valid_systems]).item() > 0
    assert torch.count_nonzero(lattice_lengths.grad[~valid_systems]).item() == 0
    assert torch.count_nonzero(lattice_angles.grad[valid_systems]).item() > 0
    assert torch.count_nonzero(lattice_angles.grad[~valid_systems]).item() == 0


def test_material_geometry_loss_is_system_mean_and_partition_invariant():
    """Batched values and gradients equal the mean of system-local losses."""
    batch_atom_lvl = torch.tensor([0, 0, 1, 2, 2, 2])
    t = torch.tensor([[0.25, 0.25, 0.25], [0.25, 0.25, 0.75]])
    network_input = {
        "t": t,
        "t_L": t[:, batch_atom_lvl],
        "f": {
            "batch_atom_lvl": batch_atom_lvl,
            "diffusion_batch_size": 2,
            "batch_size": 3,
        },
    }
    loss_input = {
        "fractional_coord_gt_L": torch.zeros((6, 3)),
        "fractional_coord_valid_mask_L": torch.tensor(
            [True, True, False, True, True, True]
        ),
        "lattice_lengths_scaled_gt_B": torch.zeros((3, 3)),
        "lattice_angles_radians_gt_B": torch.zeros((3, 3)),
        "lattice_valid_mask_B": torch.tensor([True, False, True]),
    }

    fractional_coord = torch.zeros((2, 6, 3))
    fractional_coord[0, batch_atom_lvl == 0] = 0.1
    fractional_coord[1, batch_atom_lvl == 0] = 0.3
    fractional_coord[:, batch_atom_lvl == 1] = 0.2
    fractional_coord[0, batch_atom_lvl == 2] = 0.4
    fractional_coord[1, batch_atom_lvl == 2] = 0.49
    fractional_coord.requires_grad_()

    lattice_lengths = torch.zeros((2, 3, 3))
    lattice_lengths[0, 0] = 0.2
    lattice_lengths[1, 0] = 0.4
    lattice_lengths[:, 1] = 0.5
    lattice_lengths[0, 2] = 0.3
    lattice_lengths[1, 2] = 0.8
    lattice_lengths.requires_grad_()

    lattice_angles = torch.zeros((2, 3, 3))
    lattice_angles[0, 0] = 0.1
    lattice_angles[1, 0] = 0.2
    lattice_angles[:, 1] = 0.5
    lattice_angles[0, 2] = 0.4
    lattice_angles[1, 2] = 0.8
    lattice_angles.requires_grad_()

    network_output = {
        "X_L": torch.zeros((2, 6, 3)),
        "fractional_coord_L": fractional_coord,
        "lattice_lengths_B": lattice_lengths,
        "lattice_angles_B": lattice_angles,
    }
    loss_fn = MaterialGeometryLoss(max_t=0.5)
    batched_loss, batched_metrics = loss_fn(network_input, network_output, loss_input)
    batched_loss.backward()

    system_losses = []
    system_metrics = []
    system_fractional_gradients = []
    system_lengths_gradients = []
    system_angles_gradients = []
    for system_index in range(3):
        atom_mask = batch_atom_lvl == system_index
        system_fractional_coord = (
            fractional_coord.detach()[:, atom_mask].clone().requires_grad_()
        )
        system_lattice_lengths = (
            lattice_lengths.detach()[:, system_index : system_index + 1]
            .clone()
            .requires_grad_()
        )
        system_lattice_angles = (
            lattice_angles.detach()[:, system_index : system_index + 1]
            .clone()
            .requires_grad_()
        )
        system_network_input = {
            "t": t[:, system_index : system_index + 1],
            "t_L": t[:, system_index : system_index + 1].expand(-1, atom_mask.sum()),
            "f": {
                "batch_atom_lvl": torch.zeros(atom_mask.sum(), dtype=torch.long),
                "diffusion_batch_size": 2,
                "batch_size": 1,
            },
        }
        system_loss_input = {
            "fractional_coord_gt_L": loss_input["fractional_coord_gt_L"][atom_mask],
            "fractional_coord_valid_mask_L": loss_input[
                "fractional_coord_valid_mask_L"
            ][atom_mask],
            "lattice_lengths_scaled_gt_B": loss_input["lattice_lengths_scaled_gt_B"][
                system_index : system_index + 1
            ],
            "lattice_angles_radians_gt_B": loss_input["lattice_angles_radians_gt_B"][
                system_index : system_index + 1
            ],
            "lattice_valid_mask_B": loss_input["lattice_valid_mask_B"][
                system_index : system_index + 1
            ],
        }
        system_network_output = {
            "X_L": torch.zeros_like(system_fractional_coord),
            "fractional_coord_L": system_fractional_coord,
            "lattice_lengths_B": system_lattice_lengths,
            "lattice_angles_B": system_lattice_angles,
        }
        system_loss, metrics = MaterialGeometryLoss(max_t=0.5)(
            system_network_input,
            system_network_output,
            system_loss_input,
        )
        system_loss.backward()
        system_losses.append(system_loss.detach())
        system_metrics.append(metrics)
        system_fractional_gradients.append(system_fractional_coord.grad)
        system_lengths_gradients.append(system_lattice_lengths.grad)
        system_angles_gradients.append(system_lattice_angles.grad)

    torch.testing.assert_close(
        batched_metrics["material_fractional_coord_loss"], torch.tensor(0.07)
    )
    torch.testing.assert_close(
        batched_metrics["material_lattice_lengths_loss"],
        torch.tensor(0.19 / 3),
    )
    torch.testing.assert_close(
        batched_metrics["material_lattice_angles_loss"],
        torch.tensor(0.185 / 3),
    )
    torch.testing.assert_close(batched_loss, torch.tensor(0.195))
    torch.testing.assert_close(batched_loss, torch.stack(system_losses).mean())
    for metric_name, batched_value in batched_metrics.items():
        torch.testing.assert_close(
            batched_value,
            torch.stack([metrics[metric_name] for metrics in system_metrics]).mean(),
        )

    torch.testing.assert_close(
        fractional_coord.grad,
        torch.cat(system_fractional_gradients, dim=1) / 3,
    )
    torch.testing.assert_close(
        lattice_lengths.grad,
        torch.cat(system_lengths_gradients, dim=1) / 3,
    )
    torch.testing.assert_close(
        lattice_angles.grad,
        torch.cat(system_angles_gradients, dim=1) / 3,
    )


def test_material_geometry_loss_exposes_output_gradient_norms():
    """Material loss branches expose weighted, differentiable diagnostics."""
    network_input, loss_input = _material_loss_inputs()
    network_output = {
        "X_L": torch.zeros((1, 2, 3)),
        "fractional_coord_L": torch.full((1, 2, 3), 0.25, requires_grad=True),
        "lattice_lengths_B": torch.full((1, 1, 3), 0.25, requires_grad=True),
        "lattice_angles_B": torch.full((1, 1, 3), 0.25, requires_grad=True),
    }
    loss_fn = MaterialGeometryLoss(
        fractional_coord_weight=10.0,
        lattice_lengths_weight=1.0,
        lattice_angles_weight=1.0,
    )

    total_loss, _ = loss_fn(network_input, network_output, loss_input)
    diagnostics = pop_loss_output_gradient_diagnostics(loss_fn)
    gradient_norms = compute_loss_output_gradient_l2_norms(diagnostics)

    assert gradient_norms.keys() == {
        "fractional_coordinates",
        "lattice_lengths",
        "lattice_angles",
    }
    assert all(value.item() > 0 for value in gradient_norms.values())

    # Diagnostic autograd calls retain the graph for the real training backward.
    total_loss.backward()
    assert network_output["fractional_coord_L"].grad is not None
    assert network_output["lattice_lengths_B"].grad is not None
    assert network_output["lattice_angles_B"].grad is not None


def test_material_output_gradient_diagnostics_skip_non_material_examples():
    """Loss branches with no material targets do not emit gradient diagnostics."""
    network_input, loss_input = _material_loss_inputs()
    loss_input["fractional_coord_valid_mask_L"] = torch.zeros(2, dtype=torch.bool)
    loss_input["lattice_valid_mask_B"] = torch.zeros(1, dtype=torch.bool)
    network_output = {
        "X_L": torch.zeros((1, 2, 3)),
        "fractional_coord_L": torch.ones((1, 2, 3), requires_grad=True),
        "lattice_lengths_B": torch.ones((1, 1, 3), requires_grad=True),
        "lattice_angles_B": torch.ones((1, 1, 3), requires_grad=True),
    }
    loss_fn = MaterialGeometryLoss()

    loss_fn(network_input, network_output, loss_input)

    assert pop_loss_output_gradient_diagnostics(loss_fn) == {}


def test_atom_type_loss_exposes_output_gradient_norm_for_molecules():
    """Atomized molecule targets expose an atom-type output gradient norm."""
    sequence_logits = torch.zeros((1, 2, 32), requires_grad=True)
    atom_type_logits = torch.zeros((1, 2, 128), requires_grad=True)
    loss_fn = SequenceLoss(sequence_weight=0.1, atom_type_weight=0.1)

    total_loss, _ = loss_fn(
        network_input={
            "t": torch.full((1, 1), 0.25),
            "t_I": torch.full((1, 2), 0.25),
            "f": {
                "diffusion_batch_size": 1,
                "batch_token_lvl": torch.zeros(2, dtype=torch.long),
            },
        },
        network_output={
            "sequence_logits_I": sequence_logits,
            "sequence_indices_I": torch.zeros((1, 2), dtype=torch.long),
            "atom_type_logits_I": atom_type_logits,
            "atom_type_indices_I": torch.zeros((1, 2), dtype=torch.long),
        },
        loss_input={
            "seq_token_lvl": torch.zeros(2, dtype=torch.long),
            "sequence_valid_mask": torch.zeros(2, dtype=torch.bool),
            "atom_type_gt_I": torch.tensor([6, 8]),
            "atom_type_valid_mask": torch.ones(2, dtype=torch.bool),
        },
    )
    diagnostics = pop_loss_output_gradient_diagnostics(loss_fn)
    gradient_norms = compute_loss_output_gradient_l2_norms(diagnostics)

    assert gradient_norms.keys() == {"atom_types"}
    assert gradient_norms["atom_types"].item() > 0

    total_loss.backward()
    assert atom_type_logits.grad is not None


def test_uppercase_m_creates_material_tokens():
    """Uppercase M denotes inputless atomized material generation."""
    atom_array = accumulate_components_with_molecules(
        ["7M"],
        indexed_tokens={},
        unindexed_tokens={},
    )

    assert atom_array.res_id.tolist() == [1] * 7
    assert atom_array.res_name.tolist() == ["LIG"] * 7
    assert atom_array.is_material.tolist() == [True] * 7
    assert len(get_token_starts(atom_array)) == 7


def test_inputless_material_contig_uses_length_path():
    """Inputless material contigs are rewritten before RFD3 validation."""
    spec = _rewrite_inputless_molecule_contig({"contig": "7M"})

    assert spec == {"length": "7M"}


def test_inputless_material_masks_generate_without_ground_truth():
    """Inputless material samples still use predicted frac/lattice outputs."""
    data = AddMaterialPredictionTargets()(
        {"atom_array": create_diffused_material_atoms(3)}
    )

    assert data["feats"]["material_coord_generation_mask"].tolist() == [
        True,
        True,
        True,
    ]
    assert data["feats"]["material_lattice_generation_mask"].tolist() == [
        True,
        True,
        True,
    ]
    assert data["ground_truth"]["fractional_coord_valid_mask_L"].tolist() == [
        False,
        False,
        False,
    ]
    assert data["ground_truth"]["lattice_valid_mask_I"].tolist() == [
        False,
        False,
        False,
    ]


def test_material_output_dump_writes_cell_and_fractional_fields(tmp_path):
    """Material CIF outputs include cell metadata and fractional coordinates."""
    lengths = np.array([5.0, 6.0, 7.0], dtype=float)
    angles = np.array([90.0, 90.0, 120.0], dtype=float)
    frac = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float)
    atom_array = create_diffused_material_atoms(2)
    atom_array.atom_name = np.array(["C1", "O1"])
    atom_array.element = np.array(["C", "O"])
    atom_array.coord = fractional_to_cartesian(
        frac,
        np.array(
            [
                [5.0, 0.0, 0.0],
                [-3.0, 5.19615242, 0.0],
                [0.0, 0.0, 7.0],
            ]
        ),
    )
    atom_array = annotate_material_properties(
        atom_array,
        fractional_coordinates=frac,
        lattice_lengths=lengths,
        lattice_angles=angles,
    )

    output = Zatom2Output(
        example_id="material_model_0",
        atom_array=atom_array,
        metadata={},
    )
    output.dump(tmp_path, verbose=False)

    out_path = tmp_path / "material_model_0.cif.gz"
    assert out_path.exists()
    with gzip.open(out_path, "rt") as handle:
        text = handle.read()
    for tag in (
        "_atom_site.group_PDB",
        "_atom_site.label_atom_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.fract_x",
        "_atom_site.fract_y",
        "_atom_site.fract_z",
        "_cell.length_a",
        "_cell.length_b",
        "_cell.length_c",
        "_cell.angle_alpha",
        "_cell.angle_beta",
        "_cell.angle_gamma",
    ):
        assert tag in text
    assert "\nATOM " in text or "\nHETATM " in text

    with gzip.open(out_path, "rt") as handle:
        cif = CIFFile.read(handle)
    parsed = get_structure(cif)
    np.testing.assert_allclose(parsed.coord[0], atom_array.coord, atol=1e-6)
    serialized_fractional_coordinates = np.column_stack(
        [
            cif.block["atom_site"][field].as_array(float)
            for field in ("fract_x", "fract_y", "fract_z")
        ]
    )
    np.testing.assert_allclose(serialized_fractional_coordinates, frac, atol=1e-8)


def test_validation_dump_writes_material_fractional_fields(tmp_path):
    """Validation CIF dumping preserves available fractional annotations."""
    lengths = np.array([5.0, 6.0, 7.0], dtype=float)
    angles = np.array([90.0, 90.0, 90.0], dtype=float)
    frac = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float)
    atom_array = create_diffused_material_atoms(2)
    atom_array.atom_name = np.array(["C1", "O1"])
    atom_array.element = np.array(["C", "O"])
    atom_array.coord = fractional_to_cartesian(frac, np.diag(lengths))
    atom_array = annotate_material_properties(
        atom_array,
        fractional_coordinates=frac,
        lattice_lengths=lengths,
        lattice_angles=angles,
    )
    atom_array = add_fractional_cif_fields(atom_array)

    callback = DumpValidationStructuresCallback(
        save_dir=tmp_path,
        dump_predictions=True,
        dump_prediction_metadata_json=False,
        one_model_per_file=True,
        dump_every_n=1,
        verbose=False,
    )
    callback.on_validation_batch_end(
        trainer=SimpleNamespace(state={"current_epoch": 5}),
        outputs={
            "network_output": {},
            "predicted_atom_array_stack": [atom_array],
            "prediction_metadata": {},
        },
        batch=[{"example_id": "material_validation"}],
        dataset_name="materials",
    )

    output_path = (
        tmp_path
        / "predictions"
        / "epoch_5"
        / "materials"
        / "material_validation_model_0.cif.gz"
    )
    with gzip.open(output_path, "rt") as handle:
        text = handle.read()
    assert "_atom_site.fract_x" in text
    assert "_atom_site.fract_y" in text
    assert "_atom_site.fract_z" in text
    assert "_cell.length_a" in text


@pytest.mark.parametrize("wrap", [False, True])
def test_material_prediction_writer_can_use_denoised_euclidean_coordinates(wrap):
    """Align the guided endpoint using the paired conditional heads, in a skew cell."""
    from scipy.spatial.transform import Rotation

    lengths = np.array([5.0, 6.0, 7.0], dtype=float)
    angles = np.array([78.0, 103.0, 57.0], dtype=float)
    lattice_vectors = lattice_vectors_from_lengths_angles(lengths, angles)
    conditional_frac = np.array(
        [[-0.3, 0.2, 0.3], [0.4, -0.3, 0.4], [0.2, 0.4, -0.4], [-0.2, -0.3, -0.3]]
    )
    guided_frac = conditional_frac + np.array(
        [
            [0.02, -0.01, 0.03],
            [-0.04, 0.01, -0.02],
            [0.01, 0.02, 0.02],
            [0.02, -0.03, -0.01],
        ]
    )
    rotation = Rotation.from_rotvec([0.4, -0.7, 0.3]).as_matrix()
    shift = np.array([1.1, -0.7, 2.3])
    conditional_coords = conditional_frac @ lattice_vectors @ rotation + shift
    denoised_coords = guided_frac @ lattice_vectors @ rotation + shift

    atom_array = create_diffused_material_atoms(4)
    atom_array.coord = denoised_coords.copy()
    source_atom_array = create_diffused_material_atoms(4)

    network_output = {
        "material_frame_reference_X_L": torch.tensor(conditional_coords[None]),
        "fractional_coord_L": torch.tensor((conditional_frac % 1.0)[None]),
        "lattice_lengths_B": torch.tensor(
            normalize_lattice_lengths(lengths, n_atoms=4)[None, None, :],
        ),
        "lattice_angles_B": torch.tensor(
            np.radians(angles)[None, None, :],
        ),
    }

    atom_array, metadata = ZatomDesignTrainer._apply_material_predictions(
        atom_array=atom_array,
        source_atom_array=source_atom_array,
        network_output=network_output,
        sample_idx=0,
        material_atom_mask=torch.ones(4, dtype=torch.bool),
        material_token_mask=torch.ones(4, dtype=torch.bool),
        material_coord_generation_mask=torch.ones(4, dtype=torch.bool),
        material_lattice_generation_mask=torch.ones(4, dtype=torch.bool),
        material_coordinate_output_source="euclidean",
        wrap_material_euclidean_coordinates=wrap,
    )

    np.testing.assert_allclose(
        atom_array.material_frac_coords, guided_frac % 1, atol=1e-6
    )
    assert metadata["cell_frame_alignment_rmsd_angstrom"] < 1e-7
    if wrap:
        np.testing.assert_allclose(
            atom_array.coord, (guided_frac % 1) @ lattice_vectors, atol=1e-6
        )
    else:
        # Guidance changes the shape; a single proper rigid transform preserves it.
        np.testing.assert_allclose(
            np.linalg.norm(atom_array.coord[:, None] - atom_array.coord[None], axis=-1),
            np.linalg.norm(denoised_coords[:, None] - denoised_coords[None], axis=-1),
            atol=1e-6,
        )


def test_material_prediction_writer_can_wrap_euclidean_coordinates_into_lattice():
    """Euclidean outputs can optionally be wrapped back into the predicted cell."""
    lengths = np.array([5.0, 6.0, 7.0], dtype=float)
    angles = np.array([90.0, 90.0, 90.0], dtype=float)
    lattice_vectors = np.diag(lengths)
    denoised_coords = np.array([[5.4, -0.2, 7.3], [11.1, 6.5, -1.4]], dtype=float)
    expected_coords = wrap_cartesian_coordinates(denoised_coords, lattice_vectors)

    atom_array = create_diffused_material_atoms(2)
    atom_array.coord = denoised_coords.copy()
    source_atom_array = create_diffused_material_atoms(2)
    network_output = {
        "material_frame_reference_X_L": torch.tensor(denoised_coords[None]),
        "fractional_coord_L": torch.tensor(
            (denoised_coords @ np.linalg.inv(lattice_vectors) % 1)[None]
        ),
        "lattice_lengths_B": torch.tensor(
            normalize_lattice_lengths(lengths, n_atoms=2)[None, None, :],
            dtype=torch.float32,
        ),
        "lattice_angles_B": torch.tensor(
            np.radians(angles)[None, None, :],
            dtype=torch.float32,
        ),
    }

    atom_array, metadata = ZatomDesignTrainer._apply_material_predictions(
        atom_array=atom_array,
        source_atom_array=source_atom_array,
        network_output=network_output,
        sample_idx=0,
        material_atom_mask=torch.ones(2, dtype=torch.bool),
        material_token_mask=torch.ones(2, dtype=torch.bool),
        material_coord_generation_mask=torch.ones(2, dtype=torch.bool),
        material_lattice_generation_mask=torch.ones(2, dtype=torch.bool),
        material_coordinate_output_source="euclidean",
        wrap_material_euclidean_coordinates=True,
    )

    np.testing.assert_allclose(atom_array.coord, expected_coords, atol=1e-5)
    assert np.all(atom_array.coord >= -1e-6)
    assert np.all(atom_array.coord <= lengths[None, :] + 1e-6)
    assert metadata["coordinate_output_source"] == "euclidean"


def test_euclidean_material_output_requires_a_paired_frame_reference():
    """Missing conditional heads must not silently restore the old frame bug."""
    with pytest.raises(ValueError, match="paired conditional Cartesian"):
        ZatomDesignTrainer._apply_material_predictions(
            atom_array=create_diffused_material_atoms(2),
            source_atom_array=create_diffused_material_atoms(2),
            network_output={},
            sample_idx=0,
            material_atom_mask=torch.ones(2, dtype=torch.bool),
            material_token_mask=torch.ones(2, dtype=torch.bool),
            material_coord_generation_mask=torch.ones(2, dtype=torch.bool),
            material_lattice_generation_mask=torch.ones(2, dtype=torch.bool),
            material_coordinate_output_source="euclidean",
        )


def test_material_prediction_writer_can_use_predicted_fractional_coordinates():
    """Material CIF writing can opt into the auxiliary fractional-coordinate head."""
    lengths = np.array([5.0, 6.0, 7.0], dtype=float)
    angles = np.array([90.0, 90.0, 90.0], dtype=float)
    lattice_vectors = np.diag(lengths)
    denoised_frac = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float)
    pred_frac = np.array([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], dtype=float)

    atom_array = create_diffused_material_atoms(2)
    atom_array.coord = fractional_to_cartesian(denoised_frac, lattice_vectors)
    source_atom_array = create_diffused_material_atoms(2)
    network_output = {
        "fractional_coord_L": torch.tensor(pred_frac[None, :, :], dtype=torch.float32),
        "lattice_lengths_B": torch.tensor(
            normalize_lattice_lengths(lengths, n_atoms=2)[None, None, :],
            dtype=torch.float32,
        ),
        "lattice_angles_B": torch.tensor(
            np.radians(angles)[None, None, :],
            dtype=torch.float32,
        ),
    }

    atom_array, metadata = ZatomDesignTrainer._apply_material_predictions(
        atom_array=atom_array,
        source_atom_array=source_atom_array,
        network_output=network_output,
        sample_idx=0,
        material_atom_mask=torch.ones(2, dtype=torch.bool),
        material_token_mask=torch.ones(2, dtype=torch.bool),
        material_coord_generation_mask=torch.ones(2, dtype=torch.bool),
        material_lattice_generation_mask=torch.ones(2, dtype=torch.bool),
        material_coordinate_output_source="fractional",
    )

    np.testing.assert_allclose(atom_array.material_frac_coords, pred_frac, atol=1e-6)
    np.testing.assert_allclose(
        atom_array.coord,
        fractional_to_cartesian(pred_frac, lattice_vectors),
        atol=1e-5,
    )
    np.testing.assert_allclose(atom_array.box, lattice_vectors, atol=1e-6)
    assert metadata["coordinate_output_source"] == "fractional"
