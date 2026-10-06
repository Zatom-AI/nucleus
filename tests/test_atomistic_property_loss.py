import pytest
import torch
from torch.nn import functional as F
from zatom2.model.Zatom2_diffusion_module import Zatom2DiffusionModule

from nucleus.constants import (
    MLIP_PROPERTY_TASK_ID,
    STRUCTURE_PREDICTION_PROPERTY_TASK_ID,
    UNCONDITIONAL_PROPERTY_TASK_ID,
)
from nucleus.metrics.losses import (
    PROPERTY_GRADIENT_DIAGNOSTIC_NAME,
    AtomisticPropertyLoss,
    pop_shared_backbone_gradient_diagnostics,
)
from nucleus.model.layers.blocks import LatentEquivarianceHead
from nucleus.utils.torch import get_device_compatible_float64_dtype


def _loss(*, warmup_steps=0, **kwargs):
    references = [[0.0] * 9 for _ in range(2)]
    references[0][1] = 2.0
    references[1][8] = 5.0
    return AtomisticPropertyLoss(
        domain_names=["omol25", "omat24"],
        energy_mean=[1.0, -1.0],
        energy_scale=[2.0, 4.0],
        force_scale=[2.0, 4.0],
        elemental_references=references,
        warmup_steps=warmup_steps,
        **kwargs,
    )


def _inputs(*, t=None, conditioned=False):
    if t is None:
        t = torch.tensor([[1.0, 1.0], [0.95, 0.98]])
    return {
        "t": t,
        "f": {
            "diffusion_batch_size": 2,
            "batch_size": 2,
            "batch_token_lvl": torch.tensor([0, 0, 1]),
            "batch_atom_lvl": torch.tensor([0, 0, 1]),
            "has_mean_force_norm": torch.tensor([[conditioned], [False], [False]]),
        },
    }


def test_property_loss_requires_shared_domain_id_order():
    references = [[0.0] * 9 for _ in range(2)]
    with pytest.raises(ValueError, match="0=omol25"):
        AtomisticPropertyLoss(
            domain_names=["omat24", "omol25"],
            energy_mean=[0.0, 0.0],
            energy_scale=[1.0, 1.0],
            force_scale=[1.0, 1.0],
            elemental_references=references,
        )


def test_property_loss_detects_rms_and_p95_conditioning():
    system_mask = _loss()._system_has_mean_force_conditioning(
        {
            "batch_token_lvl": torch.tensor([0, 0, 1]),
            "has_force_rms_p95": torch.tensor([[True], [False], [False]]),
        },
        B=2,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(system_mask, torch.tensor([True, False]))


def _targets():
    return {
        "energy_gt_B": torch.tensor([14.0, 0.0]),
        "energy_valid_mask_B": torch.tensor([True, True]),
        "force_gt_D_L": torch.tensor(
            [
                [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 4.0, 4.0]],
                [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 4.0, 4.0]],
            ]
        ),
        "force_valid_mask_L": torch.tensor([True, True, True]),
        "domain_id_B": torch.tensor([0, 1]),
        "domain_valid_mask_B": torch.tensor([True, True]),
        "source_atom_count_B": torch.tensor([2, 1]),
        "retained_atom_count_B": torch.tensor([2, 1]),
        "total_charge_B": torch.tensor([0.0, 0.0]),
        "spin_multiplicity_B": torch.tensor([1.0, 0.0]),
        "charge_valid_mask_B": torch.tensor([True, False]),
        "spin_valid_mask_B": torch.tensor([True, False]),
        "property_valid_mask_B": torch.tensor([True, True]),
        "property_task_id_B": torch.tensor(
            [
                UNCONDITIONAL_PROPERTY_TASK_ID,
                UNCONDITIONAL_PROPERTY_TASK_ID,
            ]
        ),
        "property_task_valid_mask_B": torch.tensor([True, True]),
        "atom_type_gt_I": torch.tensor([1, 1, 8]),
        "optimizer_step": 10,
    }


def test_property_loss_normalizes_targets_and_balances_force_systems():
    energy = torch.zeros((2, 2), requires_grad=True)
    forces = torch.zeros((2, 3, 3), requires_grad=True)
    total, metrics = _loss()(
        _inputs(),
        {"normalized_energy_B": energy, "normalized_forces_L": forces},
        _targets(),
    )

    # Standardized energies are [2, -1], hence Smooth-L1 values [1.5, 0.5].
    assert metrics["atomistic_energy_loss"].item() == pytest.approx(1.0)
    # Only t=1 passes sigma<0.25. Equal system weighting gives
    # mean((0.5 / 3 components), 0.5) = 1/3.
    assert metrics["atomistic_force_loss"].item() == pytest.approx(1 / 3)
    assert total.item() == pytest.approx(0.2)
    assert metrics["atomistic_energy_valid_views"].item() == 4
    assert metrics["atomistic_force_valid_views"].item() == 2

    total.backward()
    assert energy.grad is not None
    assert forces.grad is not None


def test_property_loss_packed_batch_matches_mean_of_individual_systems():
    """All property branches and gradients are invariant to system packing."""
    diffusion_batch_size = 3
    batch_size = 3
    network_input = {
        "t": torch.tensor(
            [
                [0.20, 1.00, 1.00],
                [0.40, 0.97, 1.00],
                [0.60, 0.50, 1.00],
            ]
        ),
        "f": {
            "diffusion_batch_size": diffusion_batch_size,
            "batch_size": batch_size,
            "batch_token_lvl": torch.arange(batch_size),
            "batch_atom_lvl": torch.arange(batch_size),
            "has_mean_force_norm": torch.zeros(batch_size, dtype=torch.bool),
        },
    }
    targets = {
        # OMat24 Z=8 has elemental reference 5, mean -1, and scale 4 in
        # ``_loss``. These totals therefore normalize to [0, 1, 2].
        "energy_gt_B": torch.tensor([4.0, 8.0, 12.0]),
        "energy_valid_mask_B": torch.ones(batch_size, dtype=torch.bool),
        "force_gt_D_L": torch.linspace(-0.5, 0.8, 27).reshape(3, 3, 3),
        "force_valid_mask_L": torch.ones(batch_size, dtype=torch.bool),
        "domain_id_B": torch.ones(batch_size, dtype=torch.long),
        "domain_valid_mask_B": torch.ones(batch_size, dtype=torch.bool),
        "source_atom_count_B": torch.ones(batch_size),
        "retained_atom_count_B": torch.ones(batch_size),
        "total_charge_B": torch.zeros(batch_size),
        "spin_multiplicity_B": torch.zeros(batch_size),
        "charge_valid_mask_B": torch.zeros(batch_size, dtype=torch.bool),
        "spin_valid_mask_B": torch.zeros(batch_size, dtype=torch.bool),
        "property_valid_mask_B": torch.ones(batch_size, dtype=torch.bool),
        "property_task_id_B": torch.tensor(
            [
                MLIP_PROPERTY_TASK_ID,
                UNCONDITIONAL_PROPERTY_TASK_ID,
                STRUCTURE_PREDICTION_PROPERTY_TASK_ID,
            ]
        ),
        "property_task_valid_mask_B": torch.ones(batch_size, dtype=torch.bool),
        "atom_type_gt_I": torch.full((batch_size,), 8, dtype=torch.long),
        "optimizer_step": 10,
    }

    packed_energy = torch.linspace(-0.4, 0.7, 9).reshape(3, 3).requires_grad_()
    packed_force = torch.linspace(-0.8, 0.9, 27).reshape(3, 3, 3).requires_grad_()
    packed_latent_prediction = (
        torch.linspace(-0.3, 0.6, 12).reshape(2, 3, 2).requires_grad_()
    )
    packed_latent_target = (
        torch.linspace(0.7, -0.5, 12).reshape(2, 3, 2).requires_grad_()
    )
    loss_kwargs = {
        "energy_weight": 0.7,
        "force_weight": 0.9,
        "latent_equivariance_weight": 0.4,
    }
    packed_total, packed_metrics = _loss(**loss_kwargs)(
        network_input,
        {
            "normalized_energy_B": packed_energy,
            "normalized_forces_L": packed_force,
            "latent_equivariance_prediction_B": packed_latent_prediction,
            "latent_equivariance_target_B": packed_latent_target,
        },
        targets,
    )

    system_vector_keys = (
        "energy_gt_B",
        "energy_valid_mask_B",
        "domain_id_B",
        "domain_valid_mask_B",
        "source_atom_count_B",
        "retained_atom_count_B",
        "total_charge_B",
        "spin_multiplicity_B",
        "charge_valid_mask_B",
        "spin_valid_mask_B",
        "property_valid_mask_B",
        "property_task_id_B",
        "property_task_valid_mask_B",
        "atom_type_gt_I",
    )
    individual_totals = []
    individual_metrics = []
    individual_outputs = []
    for system_index in range(batch_size):
        single_network_input = {
            "t": network_input["t"][:, system_index : system_index + 1].clone(),
            "f": {
                "diffusion_batch_size": diffusion_batch_size,
                "batch_size": 1,
                "batch_token_lvl": torch.zeros(1, dtype=torch.long),
                "batch_atom_lvl": torch.zeros(1, dtype=torch.long),
                "has_mean_force_norm": torch.zeros(1, dtype=torch.bool),
            },
        }
        single_targets = {
            key: targets[key][system_index : system_index + 1].clone()
            for key in system_vector_keys
        }
        single_targets.update(
            {
                "force_gt_D_L": targets["force_gt_D_L"][
                    :, system_index : system_index + 1
                ].clone(),
                "force_valid_mask_L": targets["force_valid_mask_L"][
                    system_index : system_index + 1
                ].clone(),
                "optimizer_step": targets["optimizer_step"],
            }
        )
        single_energy = (
            packed_energy.detach()[:, system_index : system_index + 1]
            .clone()
            .requires_grad_()
        )
        single_force = (
            packed_force.detach()[:, system_index : system_index + 1]
            .clone()
            .requires_grad_()
        )
        single_latent_prediction = (
            packed_latent_prediction.detach()[:, system_index : system_index + 1]
            .clone()
            .requires_grad_()
        )
        single_latent_target = (
            packed_latent_target.detach()[:, system_index : system_index + 1]
            .clone()
            .requires_grad_()
        )
        single_total, single_metrics = _loss(**loss_kwargs)(
            single_network_input,
            {
                "normalized_energy_B": single_energy,
                "normalized_forces_L": single_force,
                "latent_equivariance_prediction_B": single_latent_prediction,
                "latent_equivariance_target_B": single_latent_target,
            },
            single_targets,
        )
        individual_totals.append(single_total)
        individual_metrics.append(single_metrics)
        individual_outputs.append(
            (
                single_energy,
                single_force,
                single_latent_prediction,
                single_latent_target,
            )
        )

    reference_total = torch.stack(individual_totals).mean()
    torch.testing.assert_close(packed_total, reference_total)
    for metric_name in (
        "atomistic_property_loss",
        "atomistic_energy_loss",
        "atomistic_force_loss",
        "atomistic_latent_equivariance_loss",
    ):
        reference_metric = torch.stack(
            [metrics[metric_name] for metrics in individual_metrics]
        ).mean()
        torch.testing.assert_close(packed_metrics[metric_name], reference_metric)

    expected_valid_counts = {
        "atomistic_energy_valid_views": 5,
        "atomistic_force_valid_views": 4,
        "atomistic_latent_equivariance_valid_pairs": 2,
    }
    for metric_name, expected_count in expected_valid_counts.items():
        assert packed_metrics[metric_name].item() == expected_count
        assert packed_metrics[metric_name].item() == sum(
            metrics[metric_name].item() for metrics in individual_metrics
        )

    packed_total.backward()
    reference_total.backward()
    packed_outputs = (
        packed_energy,
        packed_force,
        packed_latent_prediction,
        packed_latent_target,
    )
    for system_index, reference_outputs in enumerate(individual_outputs):
        for packed_output, reference_output in zip(
            packed_outputs, reference_outputs, strict=True
        ):
            torch.testing.assert_close(
                packed_output.grad[:, system_index : system_index + 1],
                reference_output.grad,
            )

    for packed_output in packed_outputs:
        torch.testing.assert_close(
            packed_output.grad[:, 2:3],
            torch.zeros_like(packed_output.grad[:, 2:3]),
        )


def test_property_loss_references_large_total_energies_in_float64():
    """Reference subtraction preserves residuals below the float32 total-energy ULP."""
    reference = -1_000_000_000.125
    references = [[0.0, reference], [0.0, 0.0]]
    loss = AtomisticPropertyLoss(
        domain_names=["omol25", "omat24"],
        energy_mean=[0.0, 0.0],
        energy_scale=[1.0, 1.0],
        force_scale=[1.0, 1.0],
        elemental_references=references,
    )

    normalized, valid = loss._normalized_energy_target(
        loss_input={
            "energy_gt_B": torch.tensor([reference + 0.25], dtype=torch.float64),
            "energy_valid_mask_B": torch.tensor([True]),
            "source_atom_count_B": torch.tensor([1]),
            "atom_type_gt_I": torch.tensor([1]),
        },
        f={"batch_token_lvl": torch.tensor([0])},
        domain_id=torch.tensor([0]),
        base_valid=torch.tensor([True]),
        B=1,
        dtype=torch.float32,
    )

    assert loss.elemental_references.dtype == torch.float64
    assert loss.energy_mean.dtype == torch.float64
    assert normalized.dtype == torch.float32
    torch.testing.assert_close(normalized, torch.tensor([0.25]), rtol=0, atol=0)
    assert valid.tolist() == [True]


def test_float64_dtype_falls_back_only_for_mps():
    """Cancellation-sensitive paths use each backend's highest supported dtype."""
    assert get_device_compatible_float64_dtype("cpu") == torch.float64
    assert get_device_compatible_float64_dtype("cuda") == torch.float64
    assert get_device_compatible_float64_dtype("mps") == torch.float32


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_property_loss_normalizes_energy_on_mps_without_float64():
    """MPS energy normalization uses float32 without moving loss buffers."""
    device = torch.device("mps")
    loss = _loss()
    normalized, valid = loss._normalized_energy_target(
        loss_input={
            "energy_gt_B": torch.tensor([14.0, 0.0], device=device),
            "energy_valid_mask_B": torch.tensor([True, True], device=device),
            "source_atom_count_B": torch.tensor([2, 1], device=device),
            "atom_type_gt_I": torch.tensor([1, 1, 8], device=device),
        },
        f={"batch_token_lvl": torch.tensor([0, 0, 1], device=device)},
        domain_id=torch.tensor([0, 1], device=device),
        base_valid=torch.tensor([True, True], device=device),
        B=2,
        dtype=torch.float32,
    )

    assert normalized.device.type == "mps"
    assert normalized.dtype == torch.float32
    torch.testing.assert_close(normalized.cpu(), torch.tensor([2.0, -1.0]))
    assert valid.cpu().tolist() == [True, True]


def test_property_loss_uses_strict_equivalent_noise_gates():
    network_input = _inputs(
        t=torch.tensor(
            [
                [0.94, 0.95],  # equivalent sigma: 1.021..., 0.842...
                [0.984, 0.985],  # equivalent sigma: 0.260..., 0.243...
            ]
        )
    )
    energy = torch.zeros((2, 2), requires_grad=True)
    forces = torch.zeros((2, 3, 3), requires_grad=True)
    _, metrics = _loss()(
        network_input,
        {"normalized_energy_B": energy, "normalized_forces_L": forces},
        _targets(),
    )

    assert metrics["atomistic_energy_valid_views"].item() == 3
    assert metrics["atomistic_force_valid_views"].item() == 1


def test_mlip_property_loss_uses_every_view_at_the_effective_clean_endpoint():
    """Fixed-coordinate MLIP inputs bypass gates on the unrelated sampled time."""
    targets = _targets()
    targets["property_task_id_B"][:] = MLIP_PROPERTY_TASK_ID
    energy = torch.zeros((2, 2), requires_grad=True)
    forces = torch.zeros((2, 3, 3), requires_grad=True)
    _, metrics = _loss()(
        _inputs(t=torch.full((2, 2), 0.25)),
        {"normalized_energy_B": energy, "normalized_forces_L": forces},
        targets,
    )

    assert metrics["atomistic_energy_valid_views"].item() == 4
    assert metrics["atomistic_force_valid_views"].item() == 4


def test_property_loss_exposes_weighted_shared_gradient_diagnostic():
    targets = _targets()
    targets["property_task_id_B"][:] = MLIP_PROPERTY_TASK_ID
    targets["collect_shared_backbone_gradient_diagnostics"] = True
    energy = torch.zeros((2, 2), requires_grad=True)
    forces = torch.zeros((2, 3, 3), requires_grad=True)
    loss_fn = _loss()
    total, _ = loss_fn(
        _inputs(),
        {"normalized_energy_B": energy, "normalized_forces_L": forces},
        targets,
    )

    diagnostics = pop_shared_backbone_gradient_diagnostics(loss_fn)

    assert diagnostics.keys() == {PROPERTY_GRADIENT_DIAGNOSTIC_NAME}
    assert diagnostics[PROPERTY_GRADIENT_DIAGNOSTIC_NAME] is total
    assert loss_fn.shared_backbone_gradient_diagnostics == {}


def test_structure_prediction_does_not_contribute_property_loss():
    """The type-known generative task remains separate from the MLIP objective."""
    targets = _targets()
    targets["property_task_id_B"][:] = STRUCTURE_PREDICTION_PROPERTY_TASK_ID
    energy = torch.zeros((2, 2), requires_grad=True)
    forces = torch.zeros((2, 3, 3), requires_grad=True)
    total, metrics = _loss()(
        _inputs(),
        {"normalized_energy_B": energy, "normalized_forces_L": forces},
        targets,
    )

    assert total.item() == 0.0
    assert metrics["atomistic_energy_valid_views"].item() == 0
    assert metrics["atomistic_force_valid_views"].item() == 0
    total.backward()
    assert torch.count_nonzero(energy.grad).item() == 0
    assert torch.count_nonzero(forces.grad).item() == 0


def test_unknown_property_task_is_never_supervised():
    """Missing task identity cannot silently fall back to joint supervision."""
    targets = _targets()
    targets["property_task_valid_mask_B"][:] = False
    energy = torch.zeros((2, 2), requires_grad=True)
    forces = torch.zeros((2, 3, 3), requires_grad=True)
    _, metrics = _loss()(
        _inputs(),
        {"normalized_energy_B": energy, "normalized_forces_L": forces},
        targets,
    )

    assert metrics["atomistic_energy_valid_views"].item() == 0
    assert metrics["atomistic_force_valid_views"].item() == 0


def test_property_loss_gates_force_conditioning_and_omol_metadata():
    targets = _targets()
    targets["charge_valid_mask_B"][0] = False
    energy = torch.zeros((2, 2), requires_grad=True)
    forces = torch.zeros((2, 3, 3), requires_grad=True)
    total, metrics = _loss()(
        _inputs(conditioned=True),
        {"normalized_energy_B": energy, "normalized_forces_L": forces},
        targets,
    )

    # OMol is invalid for both independent reasons; OMat remains valid.
    assert metrics["atomistic_energy_valid_views"].item() == 2
    assert metrics["atomistic_force_valid_views"].item() == 1
    assert torch.isfinite(total)


@pytest.mark.parametrize(
    ("metadata_key", "invalid_value"),
    [
        ("total_charge_B", 0.5),
        ("spin_multiplicity_B", 1.5),
    ],
)
def test_property_loss_rejects_noninteger_omol_metadata(
    metadata_key,
    invalid_value,
):
    targets = _targets()
    targets[metadata_key][0] = invalid_value
    energy = torch.zeros((2, 2), requires_grad=True)
    forces = torch.zeros((2, 3, 3), requires_grad=True)
    _, metrics = _loss()(
        _inputs(),
        {"normalized_energy_B": energy, "normalized_forces_L": forces},
        targets,
    )

    # Only the OMat24 system remains eligible for property supervision.
    assert metrics["atomistic_energy_valid_views"].item() == 2
    assert metrics["atomistic_force_valid_views"].item() == 1


def test_property_loss_gates_crops_and_invalid_domains():
    targets = _targets()
    targets["retained_atom_count_B"][0] = 1
    targets["domain_id_B"][1] = -1
    energy = torch.zeros((2, 2), requires_grad=True)
    forces = torch.zeros((2, 3, 3), requires_grad=True)
    total, metrics = _loss()(
        _inputs(),
        {"normalized_energy_B": energy, "normalized_forces_L": forces},
        targets,
    )

    assert total.item() == 0.0
    assert metrics["atomistic_energy_valid_views"].item() == 0
    assert metrics["atomistic_force_valid_views"].item() == 0


def test_energy_and_force_label_presence_are_gated_independently():
    targets = _targets()
    targets["force_valid_mask_L"][:] = False
    energy = torch.zeros((2, 2), requires_grad=True)
    forces = torch.zeros((2, 3, 3), requires_grad=True)
    _, metrics = _loss()(
        _inputs(),
        {"normalized_energy_B": energy, "normalized_forces_L": forces},
        targets,
    )
    assert metrics["atomistic_energy_valid_views"].item() == 4
    assert metrics["atomistic_force_valid_views"].item() == 0

    targets = _targets()
    targets["energy_valid_mask_B"][:] = False
    _, metrics = _loss()(
        _inputs(),
        {"normalized_energy_B": energy, "normalized_forces_L": forces},
        targets,
    )
    assert metrics["atomistic_energy_valid_views"].item() == 0
    assert metrics["atomistic_force_valid_views"].item() == 2


def test_property_loss_missing_labels_return_differentiable_zero_with_stable_keys():
    targets = _targets()
    targets["energy_gt_B"][:] = torch.nan
    targets["energy_valid_mask_B"][:] = False
    targets["force_gt_D_L"][:] = torch.nan
    targets["force_valid_mask_L"][:] = False
    energy = torch.zeros((2, 2), requires_grad=True)
    forces = torch.zeros((2, 3, 3), requires_grad=True)
    total, metrics = _loss()(
        _inputs(),
        {"normalized_energy_B": energy, "normalized_forces_L": forces},
        targets,
    )

    assert total.item() == 0.0
    assert set(metrics) == set(AtomisticPropertyLoss._METRIC_KEYS)
    total.backward()
    assert torch.count_nonzero(energy.grad).item() == 0
    assert torch.count_nonzero(forces.grad).item() == 0


def test_property_loss_warms_weights_by_optimizer_step():
    targets = _targets()
    targets["optimizer_step"] = 5
    energy = torch.zeros((2, 2), requires_grad=True)
    forces = torch.zeros((2, 3, 3), requires_grad=True)
    total, metrics = _loss(warmup_steps=10)(
        _inputs(),
        {"normalized_energy_B": energy, "normalized_forces_L": forces},
        targets,
    )

    assert metrics["atomistic_property_warmup_scale"].item() == pytest.approx(0.5)
    assert total.item() == pytest.approx(0.1)


def test_latent_equivariance_loss_uses_clean_mlip_pairs_from_both_domains():
    targets = _targets()
    targets["property_task_id_B"][:] = MLIP_PROPERTY_TASK_ID
    prediction = torch.zeros((1, 2, 3), requires_grad=True)
    target = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]], requires_grad=True)
    network_output = {
        "normalized_energy_B": torch.zeros((2, 2), requires_grad=True),
        "normalized_forces_L": torch.zeros((2, 3, 3), requires_grad=True),
        "latent_equivariance_prediction_B": prediction,
        "latent_equivariance_target_B": target,
    }

    total, metrics = _loss(
        energy_weight=0.0,
        force_weight=0.0,
        latent_equivariance_weight=0.1,
    )(_inputs(), network_output, targets)

    expected_latent_loss = (1.0 + 4.0 + 9.0 + 4.0 + 9.0 + 16.0) / 6.0
    assert metrics["atomistic_latent_equivariance_loss"].item() == pytest.approx(
        expected_latent_loss
    )
    assert metrics["atomistic_latent_equivariance_valid_pairs"].item() == 2
    assert total.item() == pytest.approx(0.1 * expected_latent_loss)

    total.backward()
    assert prediction.grad is not None
    # Match TransIP's released implementation: the rotated target is not detached.
    assert target.grad is not None
    assert torch.count_nonzero(target.grad[0, 0]).item() == 3
    assert torch.count_nonzero(target.grad[0, 1]).item() == 3


def test_latent_equivariance_head_is_used_when_rank_has_no_eligible_pairs(
    monkeypatch,
):
    """Empty local masks give every latent parameter an explicit zero gradient."""
    head = LatentEquivarianceHead(feature_dim=3)
    embeddings = torch.linspace(-1.0, 1.0, 12).reshape(2, 2, 3).requires_grad_(True)
    rotations = torch.eye(3).expand(2, 2, -1, -1).clone()
    prediction, target = head(embeddings, rotations)
    targets = _targets()
    loss = _loss(
        energy_weight=0.0,
        force_weight=0.0,
        latent_equivariance_weight=1.0,
    )
    # Unconditional examples use the property heads but are deliberately
    # ineligible for the MLIP-only latent-equivariance objective.
    assert torch.all(targets["property_task_id_B"] == UNCONDITIONAL_PROPERTY_TASK_ID)

    def fail_on_tensor_predicate(_tensor):
        raise AssertionError("loss materialized a tensor as a Python predicate")

    # Exercise the empty-mask loss forward while rejecting hidden ``if tensor``
    # control flow. Restore Tensor.__bool__ before backward and assertions.
    with monkeypatch.context() as patch_context:
        patch_context.setattr(torch.Tensor, "__bool__", fail_on_tensor_predicate)
        total, metrics = loss(
            _inputs(),
            {
                "normalized_energy_B": torch.zeros((2, 2), requires_grad=True),
                "normalized_forces_L": torch.zeros((2, 3, 3), requires_grad=True),
                "latent_equivariance_prediction_B": prediction,
                "latent_equivariance_target_B": target,
            },
            targets,
        )

    assert total.item() == 0.0
    assert metrics["atomistic_latent_equivariance_valid_pairs"].item() == 0
    total.backward()
    assert embeddings.grad is not None
    torch.testing.assert_close(embeddings.grad, torch.zeros_like(embeddings))
    for parameter in head.parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))


@pytest.mark.parametrize(
    "invalid_reason",
    [
        "non_mlip_task",
        "invalid_task",
        "invalid_property",
        "cropped",
        "force_conditioned",
        "invalid_domain",
    ],
)
def test_latent_equivariance_loss_excludes_ineligible_systems(invalid_reason):
    """Only complete, unconditioned MLIP systems contribute latent gradients."""
    targets = _targets()
    targets["property_task_id_B"][:] = MLIP_PROPERTY_TASK_ID
    network_input = _inputs()
    if invalid_reason == "non_mlip_task":
        targets["property_task_id_B"][1] = UNCONDITIONAL_PROPERTY_TASK_ID
    elif invalid_reason == "invalid_task":
        targets["property_task_valid_mask_B"][1] = False
    elif invalid_reason == "invalid_property":
        targets["property_valid_mask_B"][1] = False
    elif invalid_reason == "cropped":
        targets["retained_atom_count_B"][1] = 0
    elif invalid_reason == "force_conditioned":
        network_input["f"]["has_mean_force_norm"] = torch.tensor([False, True])
    elif invalid_reason == "invalid_domain":
        targets["domain_valid_mask_B"][1] = False

    prediction = torch.zeros((1, 2, 2), requires_grad=True)
    target = torch.tensor([[[1.0, -1.0], [100.0, -100.0]]], requires_grad=True)
    total, metrics = _loss(
        energy_weight=0.0,
        force_weight=0.0,
        latent_equivariance_weight=1.0,
    )(
        network_input,
        {
            "normalized_energy_B": torch.zeros((2, 2), requires_grad=True),
            "normalized_forces_L": torch.zeros((2, 3, 3), requires_grad=True),
            "latent_equivariance_prediction_B": prediction,
            "latent_equivariance_target_B": target,
        },
        targets,
    )

    assert metrics["atomistic_latent_equivariance_valid_pairs"].item() == 1
    assert metrics["atomistic_latent_equivariance_loss"].item() == pytest.approx(0.5)
    assert total.item() == pytest.approx(0.5)

    total.backward()
    torch.testing.assert_close(prediction.grad[0, 1], torch.zeros(2))
    torch.testing.assert_close(target.grad[0, 1], torch.zeros(2))
    assert torch.count_nonzero(prediction.grad[0, 0]).item() == 2
    assert torch.count_nonzero(target.grad[0, 0]).item() == 2


def test_composed_latent_training_path_matches_independent_reference():
    """Pooling, relative rotations, MLP, masking, and scaling match an oracle."""
    atom_features = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[2.0, 0.0], [4.0, 2.0], [6.0, 4.0]],
            [[-1.0, 1.0], [1.0, 3.0], [3.0, 5.0]],
        ],
        requires_grad=True,
    )
    reference_atom_features = atom_features.detach().clone().requires_grad_(True)
    batch_atom = torch.tensor([0, 0, 1])
    pooled = Zatom2DiffusionModule._mean_pool_features(
        atom_features,
        batch_atom,
        batch_size=2,
    )

    identity = torch.eye(3)
    z_quarter_turn = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    z_negative_quarter_turn = z_quarter_turn.T
    x_half_turn = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    rotations = torch.stack(
        (
            torch.stack((identity, z_quarter_turn)),
            torch.stack((z_quarter_turn, identity)),
            torch.stack((x_half_turn, z_negative_quarter_turn)),
        )
    )
    expected_relative_rotations = torch.stack(
        (
            torch.stack((z_quarter_turn, z_negative_quarter_turn)),
            torch.stack(
                (
                    x_half_turn,
                    torch.diag(torch.tensor([-1.0, -1.0, 1.0])),
                )
            ),
        )
    )

    head = LatentEquivarianceHead(feature_dim=2)
    with torch.no_grad():
        head.transform[0].weight.copy_(
            torch.linspace(-0.4, 0.4, head.transform[0].weight.numel()).reshape_as(
                head.transform[0].weight
            )
        )
        head.transform[0].bias.copy_(torch.tensor([-0.2, -0.1, 0.1, 0.2]))
        head.transform[2].weight.copy_(
            torch.tensor([[0.2, -0.3, 0.4, -0.5], [-0.1, 0.6, -0.2, 0.3]])
        )
        head.transform[2].bias.copy_(torch.tensor([0.05, -0.15]))
    prediction, target = head(pooled, rotations)

    targets = _targets()
    targets["property_task_id_B"][:] = MLIP_PROPERTY_TASK_ID
    targets["optimizer_step"] = 2
    targets["force_gt_D_L"] = targets["force_gt_D_L"][:1].expand(3, -1, -1)
    network_input = _inputs(t=torch.full((3, 2), 0.25))
    network_input["f"]["diffusion_batch_size"] = 3
    loss = _loss(
        energy_weight=0.0,
        force_weight=0.0,
        latent_equivariance_weight=0.3,
        warmup_steps=4,
        weight=2.0,
    )
    total, metrics = loss(
        network_input,
        {
            "normalized_energy_B": torch.zeros((3, 2), requires_grad=True),
            "normalized_forces_L": torch.zeros((3, 3, 3), requires_grad=True),
            "latent_equivariance_prediction_B": prediction,
            "latent_equivariance_target_B": target,
        },
        targets,
    )

    # Independent oracle: explicitly pool each system, use analytically known
    # relative rotations, and apply the two affine maps rather than calling the
    # production head or loss helpers.
    reference_pooled = torch.stack(
        (
            reference_atom_features[:, :2].mean(dim=1),
            reference_atom_features[:, 2],
        ),
        dim=1,
    )
    reference_input = torch.cat(
        (
            reference_pooled[0].expand(2, -1, -1),
            expected_relative_rotations.flatten(start_dim=-2),
        ),
        dim=-1,
    )
    reference_first_weight = (
        head.transform[0].weight.detach().clone().requires_grad_(True)
    )
    reference_first_bias = head.transform[0].bias.detach().clone().requires_grad_(True)
    reference_second_weight = (
        head.transform[2].weight.detach().clone().requires_grad_(True)
    )
    reference_second_bias = head.transform[2].bias.detach().clone().requires_grad_(True)
    reference_hidden = F.gelu(
        reference_input @ reference_first_weight.T + reference_first_bias
    )
    reference_prediction = (
        reference_hidden @ reference_second_weight.T + reference_second_bias
    )
    reference_latent_loss = (
        (reference_prediction - reference_pooled[1:]).square().mean(dim=-1).mean()
    )
    # warmup=2/4, latent coefficient=0.3, and outer weight=2.0.
    reference_total = 0.5 * 0.3 * 2.0 * reference_latent_loss

    torch.testing.assert_close(pooled, reference_pooled.detach())
    torch.testing.assert_close(prediction, reference_prediction.detach())
    torch.testing.assert_close(target, reference_pooled[1:].detach())
    torch.testing.assert_close(
        metrics["atomistic_latent_equivariance_loss"],
        reference_latent_loss.detach(),
    )
    torch.testing.assert_close(total, reference_total)
    assert metrics["atomistic_latent_equivariance_valid_pairs"].item() == 4

    total.backward()
    reference_total.backward()
    torch.testing.assert_close(atom_features.grad, reference_atom_features.grad)
    torch.testing.assert_close(
        head.transform[0].weight.grad, reference_first_weight.grad
    )
    torch.testing.assert_close(head.transform[0].bias.grad, reference_first_bias.grad)
    torch.testing.assert_close(
        head.transform[2].weight.grad,
        reference_second_weight.grad,
    )
    torch.testing.assert_close(head.transform[2].bias.grad, reference_second_bias.grad)


def test_enabled_latent_equivariance_loss_requires_model_pair_outputs():
    with pytest.raises(ValueError, match="requires model outputs"):
        _loss(
            energy_weight=0.0,
            force_weight=0.0,
            latent_equivariance_weight=0.1,
        )(
            _inputs(),
            {
                "normalized_energy_B": torch.zeros((2, 2)),
                "normalized_forces_L": torch.zeros((2, 3, 3)),
            },
            _targets(),
        )


@pytest.mark.parametrize(
    ("warmup_steps", "exception", "message"),
    [
        (-1, ValueError, "warmup_steps must be non-negative"),
        (1.5, TypeError, "warmup_steps must be an integer"),
        (True, TypeError, "warmup_steps must be an integer"),
    ],
)
def test_property_loss_rejects_invalid_warmup_steps(warmup_steps, exception, message):
    with pytest.raises(exception, match=message):
        _loss(warmup_steps=warmup_steps)
