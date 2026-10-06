"""Hydrogen-bond metrics that support padded nucleic-acid outputs."""

from __future__ import annotations

import numpy as np
from rfd3na.constants import VIRTUAL_ATOM_ELEMENT_NAME
from rfd3na.metrics.hbonds_hbplus_metrics import (
    HbondMetrics as RFD3NAHbondMetrics,
)
from rfd3na.metrics.hbonds_hbplus_metrics import (
    get_hbond_metrics as get_rfd3na_hbond_metrics,
)

from nucleus.utils.trainer import (
    cleanup_virtual_atoms_and_assign_atom_name_elements,
)


def _cleanup_for_hbond_metrics(atom_array, association_scheme: str):
    """Return an HBPLUS-ready copy of a possibly padded model output."""
    atom_array = atom_array.copy()
    if np.any(np.asarray(atom_array.element) == VIRTUAL_ATOM_ELEMENT_NAME):
        atom_array = cleanup_virtual_atoms_and_assign_atom_name_elements(
            atom_array,
            association_scheme=association_scheme,
        )
    return atom_array


def get_hbond_metrics(atom_array=None, association_scheme: str = "atom14"):
    """Compute per-structure metrics after removing virtual atom slots."""
    if atom_array is None:
        return get_rfd3na_hbond_metrics(atom_array)
    atom_array = _cleanup_for_hbond_metrics(atom_array, association_scheme)
    return get_rfd3na_hbond_metrics(atom_array)


class HbondMetrics(RFD3NAHbondMetrics):
    """RFD3NA H-bond metrics with modality-aware virtual-atom cleanup.

    RFD3NA's internal fallback assumes every padded residue has a CA or CB.
    Cleaning with the configured association scheme first gives RNA and DNA
    residues their C1'/P/base atom names and removes unused atom slots before
    the upstream HBPLUS metric sees them.
    """

    def __init__(
        self,
        cutoff_HA_dist: float = 3,
        cutoff_DA_distance: float = 3.5,
        association_scheme: str = "atom14",
    ) -> None:
        super().__init__(
            cutoff_HA_dist=cutoff_HA_dist,
            cutoff_DA_distance=cutoff_DA_distance,
        )
        self.association_scheme = association_scheme

    def compute(
        self,
        *,
        ground_truth_atom_array_stack,
        predicted_atom_array_stack,
    ):
        """Compute metrics on cleaned copies of predicted structures."""
        cleaned_predictions = [
            _cleanup_for_hbond_metrics(atom_array, self.association_scheme)
            for atom_array in predicted_atom_array_stack
        ]
        return super().compute(
            ground_truth_atom_array_stack=ground_truth_atom_array_stack,
            predicted_atom_array_stack=cleaned_predictions,
        )
