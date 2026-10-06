"""Zatom-1 Table 2 QM9 metrics, isolated from other generation evaluators.

The upstream evaluator writes pymatgen/OpenBabel PDBs, reads them with RDKit
while retaining hydrogen atoms, and scores valid full molecules with the
released PoseBusters ``mol`` configuration. Its largest-fragment SMILES are
used only for validity and uniqueness. No predicted coordinates are optimized.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from importlib import metadata
from typing import Any

import numpy as np

from zevals.molecules.qm9 import _sample_arrays, _strings_sha256

QM9_ZATOM1_POSEBUSTERS_KEYS = (
    "all_atoms_connected",
    "bond_angles",
    "bond_lengths",
    "aromatic_ring_flatness",
    "double_bond_flatness",
    "internal_energy",
    "internal_steric_clash",
)
QM9_ZATOM1_PACKAGE_VERSIONS = {
    "rdkit": "2025.3.5",
    "openbabel-wheel": "3.1.1.22",
    "posebusters": "0.5.1",
    "pymatgen": "2024.11.13",
}
_UPSTREAM_COMMIT = "0d4676571d9958e1bc072fe57144451474d2e3ac"


def _load_backend() -> tuple[Any, Any, Any, Any, dict[str, str]]:
    """Require the published chemistry stack before counting any predictions."""
    versions = {}
    for package, expected in QM9_ZATOM1_PACKAGE_VERSIONS.items():
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "missing"
        if versions[package] != expected:
            raise RuntimeError(
                f"Zatom-1 Table 2 QM9 scoring requires {package}=={expected}; "
                f"found {versions[package]}. Run this Table 2 command with "
                "uv run --no-sync --with-requirements "
                "models/zatom2/docs/table2_qm9.requirements.txt."
            )
    # Import outside per-sample exception handling: broken dependencies are
    # evaluation errors, not evidence that every prediction is invalid.
    from openbabel import openbabel, pybel  # noqa: F401
    from posebusters import PoseBusters
    from pymatgen.core import Molecule
    from pymatgen.io.babel import BabelMolAdaptor
    from rdkit import Chem

    return Chem, Molecule, BabelMolAdaptor, PoseBusters, versions


def _convert_sample(
    sample: Any, Chem: Any, Molecule: Any, BabelMolAdaptor: Any
) -> tuple[Any, str] | None:
    """Follow upstream PDB bond perception, keeping explicit H and full geometry."""
    try:
        atomic_numbers, positions = _sample_arrays(sample)
        if (
            not len(atomic_numbers)
            or np.any((atomic_numbers < 1) | (atomic_numbers > 118))
            or positions.shape != (len(atomic_numbers), 3)
            or not np.isfinite(positions).all()
        ):
            return None
        molecule = Molecule(species=atomic_numbers, coords=positions)
        # pymatgen 2024.11.13 Molecule.to delegates to this adaptor, but only
        # writes a file. The same Pybel writer can return its PDB text directly.
        pdb_block = BabelMolAdaptor(molecule).pybel_mol.write("pdb")
        # Upstream configs/model/zatom.yaml sets sampling.removeHs=false and
        # passes it to the evaluator, overriding its constructor default.
        mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False)
        if mol is None or not mol.GetNumAtoms():
            return None
        Chem.MolToSmiles(mol, isomericSmiles=True)
        fragments = Chem.rdmolops.GetMolFrags(mol, asMols=True)
        largest_fragment = max(
            fragments, default=mol, key=lambda fragment: fragment.GetNumAtoms()
        )
        smiles = Chem.MolToSmiles(largest_fragment, isomericSmiles=True)
        if not smiles:
            return None
        return mol, smiles
    except Exception:
        # The upstream evaluator marks failed PDB/SMILES conversions invalid.
        # Retaining these entries in the caller preserves the attempted count.
        return None


def compute_qm9_zatom1_generation_metrics(samples: Sequence[Any]) -> dict[str, Any]:
    """Compute the nine QM9 metrics reported in Zatom-1 Table 2.

    Args:
        samples: Atomic-number/Cartesian-position pairs, mappings, or atoms
            objects, in Angstrom. Include ``None`` for unreadable predictions;
            each entry contributes to the validity denominator.

    Returns:
        Rates and percentages, counts, explicit denominators, SMILES digests,
        and the effective chemistry versions and PoseBusters configuration.
        Validity divides by attempted predictions. Uniqueness divides by valid
        predictions. Each PoseBusters mean excludes native missing values,
        matching upstream ``DataFrame.mean()``. Empty denominators produce zero
        and are listed in ``zero_denominator_metrics``.

    Raises:
        RuntimeError: The pinned chemistry stack is absent, or PoseBusters
            does not return one result row per valid molecule with the required
            binary-or-missing check columns.
    """
    Chem, Molecule, BabelMolAdaptor, PoseBusters, versions = _load_backend()
    buster = PoseBusters(config="mol")
    config_payload = json.dumps(
        buster.config, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    config_sha256 = hashlib.sha256(config_payload).hexdigest()

    valid_molecules = []
    valid_smiles = []
    for sample in samples:
        converted = _convert_sample(sample, Chem, Molecule, BabelMolAdaptor)
        if converted is not None:
            molecule, smiles = converted
            valid_molecules.append(molecule)
            valid_smiles.append(smiles)
    num_generated = len(samples)
    num_valid = len(valid_molecules)
    unique_smiles = set(valid_smiles)
    denominators = {
        "valid": num_generated,
        "unique": num_valid,
        **{key: num_valid for key in QM9_ZATOM1_POSEBUSTERS_KEYS},
    }
    numerators = {"valid": num_valid, "unique": len(unique_smiles)}
    numerators.update(dict.fromkeys(QM9_ZATOM1_POSEBUSTERS_KEYS, 0))
    missing_checks = dict.fromkeys(QM9_ZATOM1_POSEBUSTERS_KEYS, 0)
    if valid_molecules:
        # Do not deduplicate or select largest fragments here: connectivity and
        # geometry checks must see each original valid full molecule.
        frame = buster.bust(valid_molecules, None, None)
        missing = set(QM9_ZATOM1_POSEBUSTERS_KEYS).difference(frame.columns)
        if len(frame) != num_valid or missing or not frame.columns.is_unique:
            raise RuntimeError(
                "PoseBusters Table 2 result schema mismatch: "
                f"expected {num_valid} rows, got {len(frame)}; "
                f"missing checks: {sorted(missing)}."
            )
        checks = frame.loc[:, list(QM9_ZATOM1_POSEBUSTERS_KEYS)]
        for key in checks:
            observed = checks[key].dropna()
            if not observed.isin([True, False]).all():
                raise RuntimeError(
                    "PoseBusters Table 2 checks must be binary or missing "
                    f"for every valid molecule; unexpected values in {key}."
                )
            # Zatom-1 uses DataFrame.mean() (skipna=True). A native undefined
            # energy check must not silently count as a pass or a failure.
            denominators[key] = len(observed)
            numerators[key] = int(observed.sum())
            missing_checks[key] = num_valid - len(observed)
    rates = {
        key: numerators[key] / denominator if denominator else 0.0
        for key, denominator in denominators.items()
    }
    return {
        "protocol": {
            "name": "zatom1-table2-qm9",
            "upstream_commit": _UPSTREAM_COMMIT,
            "upstream_source": (
                f"https://github.com/Zatom-AI/zatom/blob/{_UPSTREAM_COMMIT}"
                "/zatom/eval/molecule_generation.py"
            ),
            "coordinate_unit": "angstrom",
            "conversion": "pymatgen.BabelMolAdaptor -> OpenBabel PDB -> RDKit",
            "pdb_remove_hydrogens": False,
            "isomeric_smiles": True,
            "validity_fragment_policy": "largest_fragment",
            "posebusters_fragment_policy": "full_valid_molecule",
            "predicted_geometry_optimized": False,
            "package_versions": versions,
            "posebusters": {
                "config_name": "mol",
                "config": buster.config,
                "config_sha256": config_sha256,
                "missing_check_policy": "exclude_from_per_check_mean",
                # PoseBusters 0.5.1 uses this fixed ETKDG seed internally for
                # the reference energy ensemble, not for the predictions.
                "reference_ensemble_random_seed": 42,
            },
        },
        "counts": {
            "generated": num_generated,
            "valid": num_valid,
            "unique": len(unique_smiles),
            "posebusters_evaluated": num_valid,
            "posebusters_missing_checks": missing_checks,
        },
        "denominators": denominators,
        "zero_denominator_metrics": [
            key for key, denominator in denominators.items() if denominator == 0
        ],
        "rates": rates,
        "percent": {key: 100.0 * value for key, value in rates.items()},
        "digests": {
            "valid_smiles_sha256": _strings_sha256(valid_smiles),
            "unique_smiles_sha256": _strings_sha256(sorted(unique_smiles)),
            "posebusters_config_sha256": config_sha256,
        },
    }
