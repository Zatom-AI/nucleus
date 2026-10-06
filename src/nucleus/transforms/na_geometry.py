"""Failure-safe nucleic-acid secondary-structure conditioning."""

from __future__ import annotations

from typing import Any

import numpy as np
from rfd3na.transforms.na_geom import CalculateNucleicAcidGeomFeats

from nucleus.transforms.hbplus import (
    _hbplus_unavailable_reason,
    patch_rfd3na_hbplus_subprocess_timeout,
)


def _masked_bp_partners(atom_array) -> None:
    """Mark all base-pair relations unknown rather than falsely unpaired."""
    atom_array.set_annotation("bp_partners", np.full(len(atom_array), None))


class SafeCalculateNucleicAcidGeomFeats(CalculateNucleicAcidGeomFeats):
    """Preserve valid NA-SS conditioning and mask unavailable legacy-PDB cases.

    RFD3NA's geometry implementation uses HBPLUS through a temporary PDB file.
    When a valid mmCIF assembly cannot be represented by that path, emit the
    existing all-mask ``bp_partners`` representation rather than fabricating
    unpaired labels or rejecting the complete training example.
    """

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        patch_rfd3na_hbplus_subprocess_timeout()
        atom_array = data["atom_array"]
        if reason := _hbplus_unavailable_reason(atom_array):
            _masked_bp_partners(atom_array)
            log_dict = data.setdefault("log_dict", {})
            log_dict["na_ss_available"] = 0
            log_dict[f"na_ss_unavailable_{reason}"] = 1
            return data

        working_data = dict(data)
        working_data["atom_array"] = atom_array.copy()
        working_data["log_dict"] = dict(data.get("log_dict", {}))
        try:
            super().forward(working_data)
        except Exception:
            _masked_bp_partners(atom_array)
            log_dict = data.setdefault("log_dict", {})
            log_dict["na_ss_available"] = 0
            log_dict["na_ss_unavailable_runtime_error"] = 1
            return data

        atom_array.set_annotation(
            "bp_partners",
            np.asarray(working_data["atom_array"].get_annotation("bp_partners")),
        )
        data["atom_array"] = atom_array
        data["log_dict"] = working_data["log_dict"]
        data["log_dict"]["na_ss_available"] = 1
        return data
