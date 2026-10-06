import pytest
import torch
from nucleus.utils.torch import get_device_compatible_float64_dtype
from torch import nn

from zatom2.model.Zatom2_diffusion_module import (
    AtomisticEnergyHead,
    LatentEquivarianceHead,
    Zatom2DiffusionModule,
)


def test_atomistic_energy_head_selects_domain_and_balances_system_size():
    """Test that the atomistic energy head selects the correct domain and normalizes by the number of atoms in the system."""
    head = AtomisticEnergyHead(c_token=2, n_domains=2, hidden_dim=2)
    head.shared_mlp = nn.Identity()
    with torch.no_grad():
        head.domain_projections[0].weight.copy_(torch.tensor([[1.0, 0.0]]))
        head.domain_projections[1].weight.copy_(torch.tensor([[0.0, 1.0]]))

    Q_I = torch.tensor(
        [
            [[1.0, 10.0], [3.0, 20.0], [5.0, 30.0], [7.0, 40.0]],
            [[2.0, 11.0], [4.0, 21.0], [6.0, 31.0], [8.0, 41.0]],
        ]
    )
    normalized_energy_B, contribution_I = head(
        Q_I,
        batch_token_lvl=torch.tensor([0, 0, 1, 1]),
        domain_id_B=torch.tensor([0, 1]),
        atom_count_B=torch.tensor([2, 4]),
    )

    torch.testing.assert_close(
        contribution_I,
        torch.tensor([[1.0, 3.0, 30.0, 40.0], [2.0, 4.0, 31.0, 41.0]]),
    )
    torch.testing.assert_close(
        normalized_energy_B,
        torch.tensor([[2.0, 17.5], [3.0, 18.0]]),
    )


def test_latent_equivariance_head_uses_relative_rotations_and_all_extra_views():
    """Test that the latent equivariance head uses the relative rotations between the first view and all other views to compute the target, and that it returns all extra views in the prediction."""
    head = LatentEquivarianceHead(feature_dim=2)

    class CaptureInput(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = None

        def forward(self, value):
            self.value = value
            return value[..., :2]

    capture = CaptureInput()
    head.transform = capture
    embeddings = torch.tensor(
        [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]], requires_grad=True
    )
    identity = torch.eye(3)
    quarter_turn = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotations = torch.stack((quarter_turn, identity, quarter_turn.T)).unsqueeze(1)

    prediction, target = head(embeddings, rotations)

    torch.testing.assert_close(prediction, embeddings[0].expand(2, -1, -1))
    torch.testing.assert_close(target, embeddings[1:])
    # Work this out independently of the implementation expression. Relative to
    # the +90-degree base view, identity is -90 degrees and -90 degrees is 180.
    expected_relative = torch.stack(
        (
            quarter_turn.T,
            torch.diag(torch.tensor([-1.0, -1.0, 1.0])),
        )
    ).unsqueeze(1)
    torch.testing.assert_close(
        capture.value[..., 2:], expected_relative.flatten(start_dim=-2)
    )


def _property_decoder(*, device: torch.device | str = "cpu") -> Zatom2DiffusionModule:
    """Prepare a Zatom2DiffusionModule with a property decoder and some fake reference data for testing."""
    module = Zatom2DiffusionModule.__new__(Zatom2DiffusionModule)
    nn.Module.__init__(module)
    module.atomistic_property_prediction_enabled = True
    module.property_domain_names = ("omol25", "omat24")

    energy_dtype = get_device_compatible_float64_dtype(device)
    references = torch.zeros(2, 128, dtype=energy_dtype, device=device)
    references[0, 1] = 1.0
    references[0, 8] = 8.0
    references[1, 6] = 60.0
    reference_valid = torch.zeros(2, 128, dtype=torch.bool, device=device)
    reference_valid[0, 1] = True
    reference_valid[0, 8] = True
    reference_valid[1, 6] = True
    module.register_buffer("property_elemental_references", references)
    module.register_buffer("property_elemental_reference_valid", reference_valid)
    module.register_buffer(
        "property_energy_mean",
        torch.tensor([2.0, 3.0], dtype=energy_dtype, device=device),
    )
    module.register_buffer(
        "property_energy_scale",
        torch.tensor([4.0, 5.0], dtype=energy_dtype, device=device),
    )
    module.register_buffer(
        "property_force_scale", torch.tensor([10.0, 20.0], device=device)
    )

    class AtomTypeHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("valid_out_mask", torch.ones(128, dtype=torch.bool))

    module.atom_type_head = AtomTypeHead()
    module.atom_type_head.to(device)
    return module


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_property_reconstruction_uses_float32_on_mps():
    """MPS reconstructs properties without attempting unsupported float64 ops."""
    device = torch.device("mps")
    module = _property_decoder(device=device)
    outputs = {
        "normalized_energy_B": torch.tensor([[0.5]], device=device),
        "normalized_forces_L": torch.ones(1, 2, 3, device=device),
    }
    f = {
        "domain_id_B": torch.tensor([0], device=device),
        "retained_atom_count_B": torch.tensor([2], device=device),
        "batch_atom_lvl": torch.tensor([0, 0], device=device),
        "ref_element": torch.tensor([1, 8], device=device),
    }

    physical = module.reconstruct_atomistic_properties(outputs, f, atom_types="known")

    assert physical["energy_B"].device.type == "mps"
    assert physical["energy_B"].dtype == torch.float32
    torch.testing.assert_close(
        physical["energy_B"].cpu(), torch.tensor([[17.0]]), rtol=0, atol=0
    )


def test_property_reconstruction_uses_domain_statistics_and_exact_counts():
    """Test that the property reconstruction uses the domain statistics and the exact atom counts to reconstruct physical properties from normalized outputs."""
    module = _property_decoder()
    outputs = {
        "normalized_energy_B": torch.tensor([[0.5, -1.0]]),
        "normalized_forces_L": torch.ones(1, 3, 3),
    }
    f = {
        "domain_id_B": torch.tensor([0, 1]),
        "retained_atom_count_B": torch.tensor([2, 1]),
        "batch_atom_lvl": torch.tensor([0, 0, 1]),
        "ref_element": torch.tensor([1, 8, 6]),
    }

    physical = module.reconstruct_atomistic_properties(outputs, f, atom_types="known")

    torch.testing.assert_close(
        physical["energy_B"], torch.tensor([[17.0, 58.0]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        physical["forces_L"],
        torch.tensor([[[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]]),
    )


def test_property_reconstruction_upcasts_low_precision_outputs():
    """Energy reconstruction uses float64 while forces are restored in float32."""
    module = _property_decoder()
    outputs = {
        "normalized_energy_B": torch.tensor([[0.5]], dtype=torch.bfloat16),
        "normalized_forces_L": torch.ones(1, 2, 3, dtype=torch.bfloat16),
    }
    f = {
        "domain_id_B": torch.tensor([0]),
        "retained_atom_count_B": torch.tensor([2]),
        "batch_atom_lvl": torch.tensor([0, 0]),
        "ref_element": torch.tensor([1, 8]),
    }

    physical = module.reconstruct_atomistic_properties(outputs, f, atom_types="known")

    assert physical["energy_B"].dtype == torch.float64
    assert physical["forces_L"].dtype == torch.float32
    torch.testing.assert_close(
        physical["energy_B"], torch.tensor([[17.0]], dtype=torch.float64)
    )
    torch.testing.assert_close(physical["forces_L"], torch.full((1, 2, 3), 10.0))


def test_property_reconstruction_preserves_small_residual_on_large_reference():
    """Physical energy reconstruction does not quantize away small residuals."""
    module = _property_decoder()
    reference = -1_000_000_000.125
    module.property_elemental_references[0, 8] = reference
    module.property_energy_mean[0] = 0.0
    module.property_energy_scale[0] = 1.0
    outputs = {
        "normalized_energy_B": torch.tensor([[0.25]], dtype=torch.float32),
        "normalized_forces_L": torch.zeros(1, 1, 3),
    }
    f = {
        "domain_id_B": torch.tensor([0]),
        "retained_atom_count_B": torch.tensor([1]),
        "batch_atom_lvl": torch.tensor([0]),
        "ref_element": torch.tensor([8]),
    }

    physical = module.reconstruct_atomistic_properties(outputs, f, atom_types="known")

    expected = torch.tensor([[reference + 0.25]], dtype=torch.float64)
    torch.testing.assert_close(physical["energy_B"], expected, rtol=0, atol=0)


def test_property_reconstruction_can_mask_energy_for_missing_reference():
    """Keep forces while masking only systems with unsupported known elements."""
    module = _property_decoder()
    outputs = {
        "normalized_energy_B": torch.zeros(1, 2),
        "normalized_forces_L": torch.ones(1, 2, 3),
    }
    f = {
        "domain_id_B": torch.tensor([0, 0]),
        "retained_atom_count_B": torch.tensor([1, 1]),
        "batch_atom_lvl": torch.tensor([0, 1]),
        "ref_element": torch.nn.functional.one_hot(
            torch.tensor([1, 6]), num_classes=128
        ).float(),
    }

    with pytest.raises(ValueError, match="available elemental references"):
        module.reconstruct_atomistic_properties(outputs, f, atom_types="known")

    physical = module.reconstruct_atomistic_properties(
        outputs,
        f,
        atom_types="known",
        allow_missing_elemental_references=True,
    )

    torch.testing.assert_close(
        physical["energy_reconstruction_valid_B"], torch.tensor([[True, False]])
    )
    torch.testing.assert_close(
        physical["energy_B"][:, :1], torch.tensor([[3.0]], dtype=torch.float64)
    )
    assert torch.isnan(physical["energy_B"][:, 1:]).all()
    torch.testing.assert_close(physical["forces_L"], torch.full((1, 2, 3), 10.0))


def test_property_reconstruction_uses_soft_masked_atom_counts():
    """Test that the property reconstruction uses soft-masked atom counts when atom_types is 'masked'."""
    module = _property_decoder()
    logits_I = torch.zeros(1, 3, 128)
    outputs = {
        "normalized_energy_B": torch.zeros(1, 2),
        "normalized_forces_L": torch.zeros(1, 3, 3),
        "atom_type_logits_I": logits_I,
    }
    f = {
        "domain_id_B": torch.tensor([0, 1]),
        "retained_atom_count_B": torch.tensor([2, 1]),
        "batch_atom_lvl": torch.tensor([0, 0, 1]),
        "atom_to_token_map": torch.tensor([0, 1, 2]),
    }

    physical = module.reconstruct_atomistic_properties(outputs, f, atom_types="masked")

    # Uniform probabilities over the two referenced OMol elements give 4.5 eV
    # per atom; OMat has only carbon available and therefore gives exactly 60 eV.
    torch.testing.assert_close(
        physical["energy_B"], torch.tensor([[13.0, 63.0]], dtype=torch.float64)
    )


def test_public_outputs_include_properties_but_embeddings_are_opt_in():
    """Test that the public outputs include properties by default, but embeddings are opt-in."""
    recycled = {
        "X_L": torch.zeros(1, 2, 3),
        "sequence_indices_I": torch.zeros(1, 2, dtype=torch.long),
        "sequence_logits_I": torch.zeros(1, 2, 32),
        "atom_type_indices_I": torch.zeros(1, 2, dtype=torch.long),
        "atom_type_logits_I": torch.zeros(1, 2, 128),
        "fractional_coord_L": torch.zeros(1, 2, 3),
        "lattice_lengths_B": torch.zeros(1, 1, 3),
        "lattice_angles_B": torch.zeros(1, 1, 3),
        "normalized_energy_B": torch.zeros(1, 1),
        "normalized_forces_L": torch.zeros(1, 2, 3),
        "Q_sys": torch.zeros(1, 1, 4),
        "Q_I": torch.zeros(1, 2, 4),
        "Q_L": torch.zeros(1, 2, 4),
        "latent_equivariance_prediction_B": torch.zeros(1, 1, 4),
        "latent_equivariance_target_B": torch.zeros(1, 1, 4),
    }

    compact = Zatom2DiffusionModule._public_outputs(recycled)
    with_embeddings = Zatom2DiffusionModule._public_outputs(
        recycled, return_property_embeddings=True
    )

    assert "normalized_energy_B" in compact
    assert "normalized_forces_L" in compact
    assert "Q_sys" not in compact
    assert "latent_equivariance_prediction_B" in compact
    assert "latent_equivariance_target_B" in compact
    assert {"Q_sys", "Q_I", "Q_L"}.issubset(with_embeddings)
