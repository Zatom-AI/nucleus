"""Tests for AllScAIP/UMA-compatible Zatom charge and spin embeddings."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from nucleus.model.layers.blocks import OneDFeatureBottleneckEmbedder
from nucleus.model.layers.charge_spin_embedding import ChgSpinEmbedding
from nucleus.model.layers.encoders import ZatomInitializer


def _embedder() -> OneDFeatureBottleneckEmbedder:
    return OneDFeatureBottleneckEmbedder(
        {
            "restype": 2,
            "total_charge": 1,
            "spin_multiplicity": 1,
        },
        bottleneck_channels=4,
        output_channels=8,
        use_allscaip_charge_spin_embeddings=True,
    )


def test_zatom_uses_allscaip_lookup_modules_for_charge_and_spin():
    """Charge and spin use AllScAIP's trainable random embeddings."""
    embedder = _embedder()
    charge = embedder.embedders["total_charge"]
    spin = embedder.embedders["spin_multiplicity"]

    assert isinstance(charge, ChgSpinEmbedding)
    assert isinstance(spin, ChgSpinEmbedding)
    assert charge.embedding_type == spin.embedding_type == "rand_emb"
    assert charge.embedding_target == "charge"
    assert spin.embedding_target == "spin"
    assert charge.rand_emb.num_embeddings == 201
    assert spin.rand_emb.num_embeddings == 101
    assert charge.rand_emb.embedding_dim == spin.rand_emb.embedding_dim == 8
    assert charge.rand_emb.weight.requires_grad
    assert spin.rand_emb.weight.requires_grad


def test_zatom_initializer_enables_allscaip_charge_and_spin_embeddings():
    """The initializer shared by Zatom-2* opts into the lookup modules."""
    initializer = ZatomInitializer(
        c_s=8,
        c_z=4,
        c_atom=8,
        c_bottleneck=4,
        n_token_head=2,
        n_atom_head=2,
        relative_position_encoding={"r_max": 2, "s_max": 1},
        token_1d_features={"total_charge": 1, "spin_multiplicity": 1},
        atom_1d_features={"ref_charge": 1},
    )

    token_embedders = initializer.token_1d_bottleneck_embedder.embedders
    assert isinstance(token_embedders["total_charge"], ChgSpinEmbedding)
    assert isinstance(token_embedders["spin_multiplicity"], ChgSpinEmbedding)
    assert initializer.token_1d_bottleneck_embedder.row_count_key == "batch_token_lvl"
    assert initializer.atom_1d_embedder.row_count_key == "batch_atom_lvl"


def test_zatom_initializer_can_ablate_allscaip_charge_and_spin_embeddings():
    """Disabling the config switch restores the legacy linear projections."""
    initializer = ZatomInitializer(
        c_s=8,
        c_z=4,
        c_atom=8,
        c_bottleneck=4,
        n_token_head=2,
        n_atom_head=2,
        relative_position_encoding={"r_max": 2, "s_max": 1},
        token_1d_features={"total_charge": 1, "spin_multiplicity": 1},
        atom_1d_features={"ref_charge": 1},
        use_allscaip_charge_spin_embeddings=False,
    )

    token_embedders = initializer.token_1d_bottleneck_embedder.embedders
    assert isinstance(token_embedders["total_charge"], nn.Linear)
    assert isinstance(token_embedders["spin_multiplicity"], nn.Linear)


@pytest.mark.parametrize("model_name", ["zatom2"])
def test_zatom2_base_configs_enable_charge_spin_conditioning(model_name):
    """Every released Zatom-2* base config declares electronic conditioning."""
    repository_root = Path(__file__).parents[1]
    config = OmegaConf.load(
        repository_root
        / "models"
        / model_name
        / "configs"
        / "model"
        / "components"
        / f"{model_name}_net.yaml"
    )

    assert config.token_initializer.use_allscaip_charge_spin_embeddings is True
    assert {
        feature: config.token_initializer.token_1d_features[feature]
        for feature in (
            "total_charge",
            "spin_multiplicity",
            "has_total_charge",
            "has_spin_multiplicity",
        )
    } == {
        "total_charge": 1,
        "spin_multiplicity": 1,
        "has_total_charge": 1,
        "has_spin_multiplicity": 1,
    }
    assert config.token_initializer.force_conditioning_residual_embedding is False


@pytest.mark.parametrize(
    ("force_features", "force_values", "residual_input_channels"),
    [
        (
            {"mean_force_norm": 1, "has_mean_force_norm": 1},
            {
                "mean_force_norm": torch.tensor([[0.1], [0.2]]),
                "has_mean_force_norm": torch.ones(2, 1),
            },
            2,
        ),
        (
            {"force_rms_p95": 2, "has_force_rms_p95": 1},
            {
                "force_rms_p95": torch.tensor([[0.1, 0.5], [0.2, 0.8]]),
                "has_force_rms_p95": torch.ones(2, 1),
            },
            3,
        ),
    ],
)
def test_force_features_can_bypass_shared_bottleneck_through_residual_embedding(
    force_features,
    force_values,
    residual_input_channels,
):
    """Both force schemes use one raw-feature projection outside the bottleneck."""
    features = {"restype": 2, **force_features}
    embedder = OneDFeatureBottleneckEmbedder(
        features,
        bottleneck_channels=4,
        output_channels=8,
        residual_features=list(force_features),
    )

    assert embedder.bottleneck_features == {"restype": 2}
    assert embedder.residual_features == force_features
    assert set(embedder.embedders) == {"restype"}
    assert embedder.bottleneck_mlp[1].in_features == 8
    assert embedder.residual_embedder.weight.shape == (
        8,
        residual_input_channels,
    )

    # Isolate the residual route and verify it is an additive projection of the
    # raw force channels, not an input to the shared bottleneck MLP.
    for parameter in embedder.bottleneck_mlp.parameters():
        parameter.data.zero_()
    embedder.residual_embedder.weight.data.copy_(
        torch.arange(8 * residual_input_channels, dtype=torch.float32).reshape(
            8, residual_input_channels
        )
    )
    inputs = {"restype": torch.eye(2), **force_values}
    residual_inputs = torch.cat(list(force_values.values()), dim=-1)

    torch.testing.assert_close(
        embedder(inputs),
        residual_inputs @ embedder.residual_embedder.weight.T,
    )


def test_force_residual_embedding_disabled_preserves_shared_bottleneck_path():
    features = {
        "restype": 2,
        "mean_force_norm": 1,
        "has_mean_force_norm": 1,
    }
    embedder = OneDFeatureBottleneckEmbedder(
        features,
        bottleneck_channels=4,
        output_channels=8,
    )

    assert embedder.bottleneck_features == features
    assert embedder.residual_features == {}
    assert set(embedder.embedders) == set(features)
    assert embedder.bottleneck_mlp[1].in_features == 3 * 8
    assert embedder.residual_embedder is None


@pytest.mark.parametrize(
    "force_features",
    [
        {"mean_force_norm": 1, "has_mean_force_norm": 1},
        {"force_rms_p95": 2, "has_force_rms_p95": 1},
    ],
)
def test_zatom_initializer_routes_active_force_features_to_residual(force_features):
    initializer = ZatomInitializer(
        c_s=8,
        c_z=4,
        c_atom=8,
        c_bottleneck=4,
        n_token_head=2,
        n_atom_head=2,
        relative_position_encoding={"r_max": 2, "s_max": 1},
        token_1d_features={"restype": 2, **force_features},
        atom_1d_features={"ref_charge": 1},
        force_conditioning_residual_embedding=True,
    )

    embedder = initializer.token_1d_bottleneck_embedder
    assert embedder.residual_features == force_features
    assert set(embedder.bottleneck_features) == {"restype"}


def test_charge_and_spin_indices_match_allscaip_ranges():
    """Charge is offset by 100 while spin is indexed directly."""
    embedder = _embedder()
    charge = embedder.embedders["total_charge"]
    spin = embedder.embedders["spin_multiplicity"]
    charges = torch.tensor([-100, 0, 100])
    spins = torch.tensor([0, 1, 100])

    torch.testing.assert_close(
        charge(charges),
        charge.rand_emb(torch.tensor([0, 100, 200])),
    )
    torch.testing.assert_close(
        spin(spins),
        spin.rand_emb(torch.tensor([0, 1, 100])),
    )


def test_missing_charge_and_spin_default_to_zero():
    """Absent metadata matches explicit UMA defaults before the bottleneck MLP."""
    embedder = _embedder()
    restype = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    missing = {"restype": restype}
    explicit = {
        "restype": restype,
        "total_charge": torch.zeros(2, 1),
        "spin_multiplicity": torch.zeros(2, 1),
    }

    torch.testing.assert_close(embedder(missing), embedder(explicit))


@pytest.mark.parametrize("row_count_key", ["batch_token_lvl", "batch_atom_lvl"])
def test_missing_features_use_the_configured_level_row_count(row_count_key):
    """The empty-feature fallback supports both token and atom embedders."""
    embedder = OneDFeatureBottleneckEmbedder(
        {"total_charge": 1, "spin_multiplicity": 1},
        bottleneck_channels=4,
        output_channels=8,
        use_allscaip_charge_spin_embeddings=True,
        row_count_key=row_count_key,
    )
    batch_indices = torch.tensor([0, 0, 1])
    missing = {row_count_key: batch_indices}
    explicit = {
        row_count_key: batch_indices,
        "total_charge": torch.zeros(3, 1),
        "spin_multiplicity": torch.zeros(3, 1),
    }

    torch.testing.assert_close(embedder(missing), embedder(explicit))
