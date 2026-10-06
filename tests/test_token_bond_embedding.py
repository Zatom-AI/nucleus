"""Regression tests for optional token-bond conditioning."""

import torch

from nucleus.model.layers.encoders import ZatomInitializer


def _initializer() -> ZatomInitializer:
    """Return a ZatomInitializer with a small configuration for testing."""
    return ZatomInitializer(
        c_s=8,
        c_z=4,
        c_atom=8,
        c_bottleneck=4,
        n_token_head=2,
        n_atom_head=2,
        relative_position_encoding={"r_max": 2, "s_max": 1},
        token_1d_features={"restype": 2},
        atom_1d_features={"ref_charge": 1},
    )


def _features() -> dict[str, torch.Tensor]:
    """Return a small set of features for testing."""
    return {
        "restype": torch.eye(2),
        "ref_charge": torch.zeros(2, 1),
        "atom_to_token_map": torch.arange(2),
        "batch_token_lvl": torch.zeros(2, dtype=torch.long),
        "batch_atom_lvl": torch.zeros(2, dtype=torch.long),
        "asym_id": torch.zeros(2, dtype=torch.long),
        "entity_id": torch.zeros(2, dtype=torch.long),
        "residue_index": torch.arange(2),
        "token_index": torch.arange(2),
        "sym_id": torch.zeros(2, dtype=torch.long),
        "unindexing_pair_mask": torch.zeros(2, 2, dtype=torch.bool),
        "ref_space_uid": torch.arange(2),
        "is_ca": torch.ones(2, dtype=torch.bool),
        "is_motif_atom_with_fixed_coord": torch.zeros(2, dtype=torch.bool),
        "is_motif_atom_with_fixed_seq": torch.zeros(2, dtype=torch.bool),
        "motif_pos": torch.zeros(2, 3),
        "ref_pos": torch.zeros(2, 3),
    }


def test_missing_token_bonds_keep_projection_in_autograd_graph():
    """A bond-free rank must contribute a zero rather than an absent gradient."""
    initializer = _initializer()
    features = _features()

    output_without_bonds = initializer(features)
    explicit_zero_features = _features()
    explicit_zero_features["token_bonds"] = torch.zeros(2, 2, dtype=torch.bool)
    output_with_zero_bonds = initializer(explicit_zero_features)

    for key in output_without_bonds:
        torch.testing.assert_close(
            output_without_bonds[key],
            output_with_zero_bonds[key],
        )

    sum(tensor.sum() for tensor in output_without_bonds.values()).backward()

    gradient = initializer.process_token_bonds.weight.grad
    assert gradient is not None
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))
