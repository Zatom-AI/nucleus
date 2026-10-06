"""Failure-safe HBPLUS conditioning transforms.

HBPLUS consumes legacy PDB files, whereas AtomWorks parses modern mmCIF
assemblies.  Some valid AtomWorks structures cannot be represented by that
legacy format.  These transforms retain HBPLUS conditioning where it is safe
while allowing the normal feature assembler to zero-fill absent features for
unsupported structures.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from atomworks.ml.transforms.base import Transform
from rfd3na.transforms.hbonds_hbplus import CalculateHbondsPlus


class _BoundedSubprocess:
    """The ``subprocess`` API used by RFD3NA's HBPLUS helpers, with a timeout.

    HBPLUS is called with no timeout and an inherited stdin, so a structure it
    cannot digest stalls one dataloader worker — and therefore one DDP rank —
    until NCCL tears the job down. A bounded call raises instead, which the
    surrounding transforms already degrade gracefully.
    """

    DEVNULL = subprocess.DEVNULL
    TimeoutExpired = subprocess.TimeoutExpired

    def call(self, *args: Any, **kwargs: Any) -> int:
        """Run ``subprocess.call`` with a hard timeout and a closed stdin."""
        kwargs.setdefault("stdin", subprocess.DEVNULL)
        kwargs.setdefault(
            "timeout", float(os.environ.get("NUCLEUS_HBPLUS_TIMEOUT_SECONDS", 120))
        )
        return subprocess.call(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate every other attribute to the real ``subprocess`` module."""
        return getattr(subprocess, name)


def patch_rfd3na_hbplus_subprocess_timeout() -> None:
    """Bound every HBPLUS invocation made by RFD3NA transforms and metrics."""
    from rfd3na.transforms import hbonds_hbplus, na_geom_utils

    for module in (hbonds_hbplus, na_geom_utils):
        if not getattr(module, "_nucleus_hbplus_timeout_patched", False):
            module.subprocess = _BoundedSubprocess()  # type: ignore[assignment]
            module._nucleus_hbplus_timeout_patched = True  # type: ignore[attr-defined]


def _hbplus_unavailable_reason(atom_array) -> str | None:
    """Return a stable reason when an array cannot safely use HBPLUS."""
    if np.any(np.char.str_len(np.asarray(atom_array.res_name, dtype=str)) > 3):
        return "res_name_too_long"

    categories = set(atom_array.get_annotation_categories())
    if "chain_iid" not in categories:
        return "missing_chain_iid"
    if len(np.unique(atom_array.chain_iid)) > 52:
        return "too_many_chain_iids"
    if not np.isfinite(np.asarray(atom_array.coord)).all():
        return "nonfinite_coordinates"

    executable = os.environ.get("HBPLUS_PATH")
    if (
        not executable
        or not Path(executable).is_file()
        or not os.access(executable, os.X_OK)
    ):
        return "executable_unavailable"
    return None


def _mark_hbplus_unavailable(data: dict[str, Any], reason: str) -> dict[str, Any]:
    """Record a numeric unavailable reason without changing model features."""
    log_dict = data.setdefault("log_dict", {})
    log_dict["hbplus_available"] = 0
    log_dict[f"hbplus_unavailable_{reason}"] = 1
    return data


class SafeCalculateHbondsPlus(CalculateHbondsPlus):
    """Calculate HBPLUS features when representable, otherwise leave them absent.

    The base transform mutates its input while serializing temporary PDB files.
    Run it on an AtomArray copy and transfer only the resulting donor/acceptor
    annotations back on success.  On an unsupported or failed invocation, the
    existing feature assembler fills absent donor/acceptor inputs with zeros.
    """

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        patch_rfd3na_hbplus_subprocess_timeout()
        atom_array = data["atom_array"]
        if reason := _hbplus_unavailable_reason(atom_array):
            return _mark_hbplus_unavailable(data, reason)

        working_data = dict(data)
        working_data["atom_array"] = atom_array.copy()
        working_data["log_dict"] = dict(data.get("log_dict", {}))
        try:
            super().forward(working_data)
        except subprocess.TimeoutExpired:
            return _mark_hbplus_unavailable(data, "timeout")
        except Exception:
            return _mark_hbplus_unavailable(data, "runtime_error")

        working_array = working_data["atom_array"]
        for annotation in ("active_donor", "active_acceptor"):
            atom_array.set_annotation(
                annotation, np.asarray(working_array.get_annotation(annotation)).copy()
            )
        data["atom_array"] = atom_array
        data["log_dict"] = working_data["log_dict"]
        data["log_dict"]["hbplus_available"] = 1
        return data


class DisabledCalculateHbondsPlus(Transform):
    """Accept the upstream constructor arguments while intentionally doing nothing."""

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        return data
