"""Tests for shared UMA validation adapters."""

import torch

from nucleus.metrics.uma_forces import electronic_conditioning_from_extra_info


def test_electronic_conditioning_reads_trainer_shaped_specification():
    """Inference charge and spin survive the trainer's metadata wrapper."""
    extra_info = {
        "example_id": ["charged-doublet"],
        "specification": {
            "total_charge": -1,
            "spin_multiplicity": 2,
        },
    }

    assert electronic_conditioning_from_extra_info(extra_info) == (-1, 2)


def test_electronic_conditioning_reads_nested_specification_extra():
    """MLIP-style electronic metadata can live below specification.extra."""
    extra_info = {
        "specification": {
            "extra": {
                "total_charge": [-2],
                "spin_multiplicity": [3],
                "has_total_charge": [True],
                "has_spin_multiplicity": [True],
            }
        }
    }

    assert electronic_conditioning_from_extra_info(extra_info) == (-2, 3)


def test_electronic_conditioning_falls_back_from_invalid_runtime_placeholders():
    """Null GT placeholders must not hide the explicit inference request."""
    extra_info = {
        "total_charge_B": torch.tensor([0.0]),
        "spin_multiplicity_B": torch.tensor([0.0]),
        "charge_valid_mask_B": torch.tensor([False]),
        "spin_valid_mask_B": torch.tensor([False]),
        "specification": {
            "total_charge": 1,
            "spin_multiplicity": 2,
        },
    }

    assert electronic_conditioning_from_extra_info(extra_info) == (1, 2)


def test_electronic_conditioning_resolves_charge_and_spin_independently():
    """A runtime charge does not hide a requested spin in a nested mapping."""
    extra_info = {
        "total_charge_B": torch.tensor([-1.0]),
        "charge_valid_mask_B": torch.tensor([True]),
        "specification": {"spin_multiplicity": 4},
    }

    assert electronic_conditioning_from_extra_info(extra_info) == (-1, 4)
