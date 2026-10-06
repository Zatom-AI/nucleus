"""Sample-directory loaders — turn a folder of generated outputs into the
in-memory `samples` shape each modality's `EvalPipeline.run()` expects.

Each loader has the signature:

    def load_<modality>(
        samples_dir: Path, ground_truth_dir: Path | None = None
    ) -> list[...]

`ground_truth_dir` is accepted by every loader for a uniform call signature
(`ZevalSuite` dispatches through `LOADER_REGISTRY` without caring which
modality it's calling). Molecules and materials opt into reference evaluation
when both directories use their explicit target-grouped SDF/CIF layouts;
otherwise it remains a harmless no-op for backward compatibility. Other
modalities currently ignore it.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from zevals.pipeline.ligands import LigandSample
from zevals.utils.io import (
    glob_cifs,
    glob_sdfs,
    glob_structures,
    load_smiles,
    load_structure_cif,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MoleculeConformerEnsembles:
    """Molecule samples plus target-grouped generated/reference ensembles."""

    molecules: list[Any]
    generated_by_target: dict[str, list[Any]]
    reference_by_target: dict[str, list[Any]]


@dataclass(frozen=True)
class MaterialStructureCandidates:
    """Flat material samples plus target-grouped generated/reference CIFs."""

    structures: list[Any]
    generated_by_target: dict[str, list[Any]]
    reference_by_target: dict[str, list[Any]]


def load_molecules(
    samples_dir: Path, ground_truth_dir: Path | None = None
) -> list[Any] | MoleculeConformerEnsembles:
    """Load generated molecules as RDKit `Chem.Mol` objects from a directory
    of `.smi` and/or `.sdf` files.

    `.sdf` files preserve their 3D conformer — Zatom's native output format
    for a generated 3D molecular structure (real atom identities and
    positions) — needed for the conformer-level checks (`compute_posebusters`,
    `compute_uff_force_norm`). `.smi` files (bare SMILES, no 3D information)
    parse into conformer-less `Chem.Mol` objects: still usable for the
    graph-level checks (validity, diversity, uniqueness, novelty), but they
    don't contribute to the 3D-specific ones, the same graceful degradation
    those checks already apply to any conformer-less molecule.

    Args:
        samples_dir: Directory containing `.smi` and/or `.sdf` files.
        ground_truth_dir: Optional conformer-reference directory. When set,
            both directories must use ``<target_id>/*.sdf`` layout. The
            returned object retains the flat generated list for existing
            unconditional metrics and additionally groups conformers by ID.

    Returns:
        List of RDKit `Chem.Mol` objects, one per generated molecule.
    """
    from zevals.utils.packages import load_rdkit_chem

    Chem = load_rdkit_chem()
    samples_dir = Path(samples_dir)
    mols: list[Any] = []
    for smi_path in sorted(samples_dir.glob("*.smi")):
        for smiles in load_smiles(smi_path):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append(mol)
    for sdf_path in glob_sdfs(samples_dir):
        for mol in Chem.SDMolSupplier(str(sdf_path)):
            if mol is not None:
                mols.append(mol)
    if ground_truth_dir is None:
        return mols

    ground_truth_dir = Path(ground_truth_dir)
    has_generated_ensembles = _has_sdf_ensemble_layout(samples_dir)
    has_reference_ensembles = _has_sdf_ensemble_layout(ground_truth_dir)
    if not has_generated_ensembles and not has_reference_ensembles:
        # ``ground_truth_dir`` was historically a no-op for molecules. Keep
        # that API compatible unless callers opt into the explicit layout.
        return mols
    if has_generated_ensembles != has_reference_ensembles:
        raise ValueError(
            "Generated and reference conformer directories must both use "
            "the <target_id>/*.sdf layout."
        )
    generated_by_target = _load_sdf_ensembles(samples_dir, Chem)
    reference_by_target = _load_sdf_ensembles(ground_truth_dir, Chem)
    return MoleculeConformerEnsembles(
        molecules=[
            molecule
            for conformers in generated_by_target.values()
            for molecule in conformers
        ],
        generated_by_target=generated_by_target,
        reference_by_target=reference_by_target,
    )


def _load_sdf_ensembles(samples_dir: Path, chem: Any) -> dict[str, list[Any]]:
    """Load ``<target_id>/*.sdf`` conformer ensembles without guessing IDs."""
    ensembles: dict[str, list[Any]] = {}
    for target_dir in sorted(path for path in samples_dir.iterdir() if path.is_dir()):
        conformers: list[Any] = []
        for sdf_path in sorted(target_dir.glob("*.sdf")):
            conformers.extend(
                mol for mol in chem.SDMolSupplier(str(sdf_path)) if mol is not None
            )
        ensembles[target_dir.name] = conformers
    if not ensembles:
        raise ValueError(
            f"Expected target subdirectories containing .sdf conformers under {samples_dir}."
        )
    return ensembles


def _has_sdf_ensemble_layout(samples_dir: Path) -> bool:
    """Return whether a directory explicitly contains target-grouped SDFs."""
    return any(
        target_dir.is_dir() and any(target_dir.glob("*.sdf"))
        for target_dir in samples_dir.iterdir()
    )


def load_materials(
    samples_dir: Path, ground_truth_dir: Path | None = None
) -> list[Any] | MaterialStructureCandidates:
    """Load generated crystal structures from a directory of `.cif` files.

    Args:
        samples_dir: Directory containing `.cif` files.
        ground_truth_dir: Optional structure-reference directory. When set,
            both directories must use ``<target_id>/*.cif`` layout. The
            returned object retains the flat generated list and additionally
            groups structures by target ID.

    Returns:
        List of `pymatgen.core.Structure` objects.
    """
    samples_dir = Path(samples_dir)
    structures = [load_structure_cif(path) for path in glob_cifs(samples_dir)]
    if ground_truth_dir is None:
        return structures

    ground_truth_dir = Path(ground_truth_dir)
    has_generated_candidates = _has_cif_candidate_layout(samples_dir)
    has_reference_candidates = _has_cif_candidate_layout(ground_truth_dir)
    if not has_generated_candidates and not has_reference_candidates:
        return structures
    if has_generated_candidates != has_reference_candidates:
        raise ValueError(
            "Generated and reference material directories must both use the "
            "<target_id>/*.cif layout."
        )
    generated_by_target = _load_cif_candidates(samples_dir)
    reference_by_target = _load_cif_candidates(ground_truth_dir)
    return MaterialStructureCandidates(
        structures=[
            structure
            for candidates in generated_by_target.values()
            for structure in candidates
        ],
        generated_by_target=generated_by_target,
        reference_by_target=reference_by_target,
    )


def _load_cif_candidates(samples_dir: Path) -> dict[str, list[Any]]:
    """Load ``<target_id>/*.cif`` candidate sets without inferring target IDs."""
    candidates: dict[str, list[Any]] = {}
    for target_dir in sorted(path for path in samples_dir.iterdir() if path.is_dir()):
        candidates[target_dir.name] = [
            load_structure_cif(path) for path in glob_cifs(target_dir)
        ]
    if not candidates:
        raise ValueError(
            f"Expected target subdirectories containing .cif candidates under {samples_dir}."
        )
    return candidates


def _has_cif_candidate_layout(samples_dir: Path) -> bool:
    """Return whether a directory explicitly contains target-grouped CIFs."""
    return any(
        target_dir.is_dir() and any(glob_cifs(target_dir))
        for target_dir in samples_dir.iterdir()
    )


def load_proteins(
    samples_dir: Path, ground_truth_dir: Path | None = None
) -> list[Path]:
    """Load generated protein backbone structure paths (`.cif` preferred, `.pdb` also accepted).

    Args:
        samples_dir: Directory containing `.cif` and/or `.pdb` files.
        ground_truth_dir: Unused (see module docstring).

    Returns:
        Sorted list of structure file paths — everything downstream (oracles,
        metrics) dispatches on `.suffix` rather than assuming one format.
    """
    del ground_truth_dir
    return glob_structures(Path(samples_dir))


def load_rna(samples_dir: Path, ground_truth_dir: Path | None = None) -> list[Path]:
    """Load generated RNA backbone structure paths. Same shape as `load_proteins`."""
    del ground_truth_dir
    return glob_structures(Path(samples_dir))


def load_ligands(
    samples_dir: Path, ground_truth_dir: Path | None = None
) -> list[LigandSample]:
    """Load generated ligand poses using a one-subdirectory-per-sample convention.

    Expects:
        samples_dir/<id>/ligand.sdf                    (required)
        samples_dir/<id>/protein.pdb                    (optional)
        ground_truth_dir/<id>/reference_ligand.sdf       (optional, matched by <id>)

    Subdirectories missing `ligand.sdf` are skipped with a warning rather
    than aborting the whole load — a partially-populated samples_dir
    shouldn't block evaluating the rest.

    Args:
        samples_dir: Directory of `<id>/` subdirectories, one per sample.
        ground_truth_dir: Directory of `<id>/` subdirectories holding
            reference ligand poses, matched by the same `<id>`.

    Returns:
        List of `LigandSample` objects.
    """
    samples_dir = Path(samples_dir)
    gt_dir = Path(ground_truth_dir) if ground_truth_dir is not None else None

    samples: list[LigandSample] = []
    for subdir in sorted(p for p in samples_dir.iterdir() if p.is_dir()):
        ligand_path = subdir / "ligand.sdf"
        if not ligand_path.is_file():
            warnings.warn(f"Skipping {subdir}: missing ligand.sdf", stacklevel=2)
            continue
        protein_path = subdir / "protein.pdb"
        protein_path = protein_path if protein_path.is_file() else None

        reference_ligand_path = None
        if gt_dir is not None:
            candidate = gt_dir / subdir.name / "reference_ligand.sdf"
            if candidate.is_file():
                reference_ligand_path = candidate

        samples.append(
            LigandSample(
                ligand_path=ligand_path,
                protein_path=protein_path,
                reference_ligand_path=reference_ligand_path,
            )
        )

    return samples


LOADER_REGISTRY: dict[str, Callable[[Path, Path | None], list]] = {
    "molecules": load_molecules,
    "materials": load_materials,
    "proteins": load_proteins,
    "rna": load_rna,
    "ligands": load_ligands,
}
