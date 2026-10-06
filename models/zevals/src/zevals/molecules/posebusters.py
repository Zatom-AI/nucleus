"""PoseBusters molecule-generation metrics.

This mirrors the Zatom-1 molecule-generation evaluator:

- compute the per-check PoseBusters means
- expose the aggregate `posebusters_rate` as the row-wise AND over all checks

The input is expected to be a sequence of PoseBusters-ready 3D molecule
objects, matching the upstream `PoseBusters.bust(..., None, None)` flow.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from zevals.utils.packages import load_posebusters

POSEBUSTERS_KEYS: tuple[str, ...] = (
    "mol_pred_loaded",
    "sanitization",
    "inchi_convertible",
    "all_atoms_connected",
    "no_radicals",
    "bond_lengths",
    "bond_angles",
    "internal_steric_clash",
    "aromatic_ring_flatness",
    "non-aromatic_ring_non-flatness",
    "double_bond_flatness",
    "internal_energy",
    "posebusters_rate",
)


def _posebusters_mol_config_path() -> Path:
    """Return the editable zevals copy of the PoseBusters mol config."""
    source_tree_config = (
        Path(__file__).resolve().parents[3] / "configs" / "posebusters" / "mol.yml"
    )
    if source_tree_config.exists():
        return source_tree_config

    return Path(str(resources.files("zevals.configs").joinpath("posebusters/mol.yml")))


def _load_posebusters_mol_config() -> dict[str, Any]:
    """Load the zevals PoseBusters mol config as a PoseBusters config dict."""
    return cast(
        dict[str, Any],
        OmegaConf.to_container(
            OmegaConf.load(_posebusters_mol_config_path()),
            resolve=True,
        ),
    )


def compute_posebusters_metrics(mol_list: list[Any]) -> dict[str, float]:
    """Compute PoseBusters pass rate and individual check means.

    Args:
        mol_list: PoseBusters-ready 3D molecule objects. In Zatom-2 these are
            `Chem.Mol` instances, but the underlying PoseBusters backend also
            accepts other supported molecule representations.

    Returns:
        Dict with all PoseBusters columns plus `posebusters_rate`.
    """
    PoseBusters = load_posebusters()

    if not mol_list:
        return {key: 0.0 for key in POSEBUSTERS_KEYS}

    buster = PoseBusters(config=_load_posebusters_mol_config())
    pb_metrics = buster.bust(mol_list, None, None)
    pb_metrics["posebusters_rate"] = pb_metrics.all(axis=1)
    pb_metrics_dict_raw = pb_metrics.mean().to_dict()
    return {key: float(pb_metrics_dict_raw.get(key, 0.0)) for key in POSEBUSTERS_KEYS}
