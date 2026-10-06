"""Load Zatom's GEOM-Drugs release as individual AtomWorks conformers."""

import json
import pickle
import re
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from ase import Atoms
from atomworks.io.parser import parse_atom_array
from atomworks.io.tools.rdkit import atom_array_from_rdkit
from atomworks.ml.datasets import PandasDataset
from atomworks.ml.datasets.loaders.ase import create_ase_atoms_loader
from biotite import structure as struc
from rdkit import Chem


def load_geom_tensors(path: str | Path) -> tuple[dict, dict]:
    """Memory-map one trusted, serialized PyG split from Zatom-AI/geom.

    The release stores one record per conformer, with atom-level tensors packed
    along their first axis and PyG slice offsets. No PyG dataset cache of expanded
    conformer objects is constructed. Requires the ``torch-geometric`` extra to
    deserialize the upstream Data class.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, tuple) or len(payload) != 3:
        raise ValueError(f"{path}: expected Zatom's processed PyG dataset tuple.")
    data, slices, _ = payload
    if not isinstance(data, dict) or not isinstance(slices, dict):
        raise ValueError(f"{path}: invalid GEOM data/slices dictionaries.")
    positions, numbers = data["pos"], data["atom_types"]
    offsets = slices["pos"]
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or numbers.ndim != 1
        or len(numbers) != len(positions)
        or not torch.equal(offsets, slices["atom_types"])
        or offsets.ndim != 1
        or len(offsets) != len(data["id"]) + 1
        or offsets[0] != 0
        or offsets[-1] != len(positions)
        or not (offsets.diff() > 0).all()
    ):
        raise ValueError(f"{path}: inconsistent GEOM conformer slices/atom tensors.")
    return data, slices


class _ProcessedGeomLoader:
    """Read conformer slices lazily; reopen memory maps in spawned workers."""

    def __init__(self, path: Path, remove_hydrogens: bool):
        self.path = path
        self.remove_hydrogens = remove_hydrogens
        self._tensors = None
        self._atomworks_loader = create_ase_atoms_loader(
            res_name="LIG", keep_ase_atoms=False
        )

    @property
    def tensors(self):
        if self._tensors is None:
            self._tensors = load_geom_tensors(self.path)
        return self._tensors

    def __getstate__(self):
        return {**self.__dict__, "_tensors": None}

    def __call__(self, row: pd.Series) -> dict[str, Any]:
        data, slices = self.tensors
        index = int(row["source_index"])
        start, stop = slices["pos"][index : index + 2].tolist()
        numbers = data["atom_types"][start:stop].numpy()
        positions = data["pos"][start:stop].numpy()
        if (
            not np.isfinite(positions).all()
            or not ((numbers >= 1) & (numbers <= 118)).all()
        ):
            raise ValueError(
                f"{row['example_id']}: invalid coordinates/atomic numbers."
            )
        keep = numbers != 1 if self.remove_hydrogens else np.ones(len(numbers), bool)
        if not keep.any():
            raise ValueError(f"{row['example_id']}: hydrogen removal left no atoms.")
        example = self._atomworks_loader(
            Atoms(numbers=numbers[keep], positions=positions[keep], pbc=False)
        )
        atoms = example["atom_array"]
        atoms.set_annotation("source_atom_index", np.flatnonzero(keep))
        example.update(
            example_id=row["example_id"],
            path=self.path,
            assembly_id="1",
            atom_array_stack=struc.stack([atoms]),
            query_pn_unit_iids=np.unique(atoms.pn_unit_iid).tolist(),
            specification={},
            extra_info={
                key: row[key]
                for key in (
                    "source_id",
                    "source_index",
                    "split",
                    "molecule_index",
                    "conformer_index",
                )
            },
        )
        return example


def build_geom_dataset(
    data_dir: str | Path,
    split: str,
    *,
    transform: Callable | None = None,
    remove_hydrogens: bool = False,
    max_samples: int | None = None,
) -> PandasDataset:
    """Load every conformer in an official Zatom processed GEOM split.

    Args:
        data_dir: Extracted Zatom-AI/geom release, containing ``{split}.pt``.
        split: ``train``, ``val``, or ``test``; original split/order is retained.
        transform: Optional Nucleus/AtomWorks transform.
        remove_hydrogens: Remove explicit H; by default all atoms are retained.
        max_samples: Optional global prefix cap, applied before rank sharding.

    Returns:
        A dataset with a separate AtomArray for each stored conformer. All of
        the release's up to five conformers per molecule are retained, including
        repeated molecular identities. IDs are ``geom/{split}/{molecule}/{conformer}``.
        The source supplies atom identities and coordinates, without bond orders
        or formal charges. These are unconditional targets, never conditioning.
        Tensor storage is memory-mapped instead of unpickling raw RDKit molecules
        or expanding all conformers into Python objects at startup.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown GEOM split {split!r}; use train, val, or test.")
    if max_samples is not None and (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, int)
        or max_samples < 1
    ):
        raise ValueError("max_samples must be a positive int or None.")
    path = Path(data_dir).expanduser().resolve() / f"{split}.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing GEOM processed split: {path}. "
            "Run `uv run python scripts/download_geom.py` first."
        )
    loader = _ProcessedGeomLoader(path, remove_hydrogens)
    source_ids = loader.tensors[0]["id"][:max_samples]
    rows = []
    for index, source_id in enumerate(source_ids):
        parts = source_id.split("_")
        if (
            len(parts) != 4
            or parts[:2] != ["geom", split]
            or not all(part.isdecimal() for part in parts[2:])
        ):
            raise ValueError(f"{path}: invalid source conformer ID {source_id!r}.")
        rows.append(
            {
                "example_id": "/".join(parts),
                "source_id": source_id,
                "source_index": index,
                "split": split,
                "molecule_index": int(parts[2]),
                "conformer_index": int(parts[3]),
            }
        )
    return PandasDataset(
        data=pd.DataFrame(rows),
        name=f"geom_{split}",
        loader=loader,
        transform=transform,
    )


def build_geom_inference_dataset(
    data: str | Path,
    *,
    transform: Callable,
    max_samples: int | None = 100,
):
    """Load unconditional atom-count requests for generation-only validation.

    The JSON maps request IDs to ``{"length": "<num_atoms>m"}`` specifications.
    No source molecule or reference coordinates are loaded. The optional prefix
    cap is applied globally before the validation sampler distributes ranks.
    """
    from nucleus.utils.datasets import ZatomContigJsonDataset

    if max_samples is not None and (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, int)
        or max_samples < 1
    ):
        raise ValueError("max_samples must be a positive int or None.")
    entries = json.loads(Path(data).read_text())
    if not isinstance(entries, dict) or not entries:
        raise ValueError("GEOM inference JSON must contain named generation requests.")
    for name, spec in entries.items():
        if (
            not isinstance(spec, dict)
            or set(spec) != {"length"}
            or not isinstance(spec["length"], str)
            or re.fullmatch(r"[1-9][0-9]*m", spec["length"]) is None
        ):
            raise ValueError(
                f"{name}: GEOM unconditional inference requires only a fixed 'Nm' length."
            )
    entries = dict(list(entries.items())[:max_samples])
    return ZatomContigJsonDataset(
        data=entries,
        name="geom_unconditional_generation",
        transform=transform,
        cif_parser_args=None,
        subset_to_keys=None,
        eval_every_n=1,
    )


def load_geom_conformer(
    molecule: Chem.Mol,
    *,
    example_id: str,
    remove_hydrogens: bool = False,
) -> dict[str, Any]:
    """Convert one stored conformer without regenerating coordinates or bonds.

    Args:
        molecule: RDKit molecule with exactly one conformer. The source object
            is never modified, including when hydrogens are removed.
        example_id: Stable identifier for this conformer.
        remove_hydrogens: Remove explicit hydrogens; otherwise keep every atom.

    Returns:
        An AtomWorks pipeline input with source atom indices, molecular graph,
        formal charges, and the original coordinates in Angstroms. Source RDKit
        properties are retained as metadata, not used as conditioning targets.
    """
    if not isinstance(molecule, Chem.Mol) or molecule.GetNumAtoms() == 0:
        raise ValueError(f"{example_id}: expected a nonempty RDKit molecule.")
    if molecule.GetNumConformers() != 1:
        raise ValueError(f"{example_id}: expected exactly one stored conformer.")
    if not np.isfinite(molecule.GetConformer().GetPositions()).all():
        raise ValueError(f"{example_id}: conformer coordinates must be finite.")

    atoms = atom_array_from_rdkit(Chem.Mol(molecule), remove_hydrogens=False)
    if atoms.bonds.get_bond_count() != molecule.GetNumBonds():
        raise ValueError(f"{example_id}: molecule contains unsupported bond types.")
    atoms.res_name[:] = "LIG"
    atoms.res_id[:] = 1
    atoms.chain_id[:] = "A"
    atoms.set_annotation("source_atom_index", np.arange(len(atoms)))
    parsed = parse_atom_array(
        atoms,
        ccd_mirror_path=None,
        add_missing_atoms=False,
        remove_ccds=[],
        remove_waters=False,
        fix_ligands_at_symmetry_centers=False,
        fix_arginines=False,
        fix_formal_charges=False,
        fix_bond_types=False,
        hydrogen_policy="remove" if remove_hydrogens else "keep",
        build_assembly=None,
    )
    atoms = parsed["assemblies"]["1"][0]
    if not len(atoms):
        raise ValueError(f"{example_id}: hydrogen removal left an empty molecule.")
    return {
        "example_id": example_id,
        "assembly_id": "1",
        "atom_array": atoms,
        "atom_array_stack": struc.stack([atoms]),
        "query_pn_unit_iids": np.unique(atoms.pn_unit_iid).tolist(),
        "chain_info": parsed["chain_info"],
        "ligand_info": parsed["ligand_info"],
        "metadata": parsed["metadata"],
        "has_disconnected_fragments": len(Chem.GetMolFrags(molecule)) > 1,
        "specification": {},
        "extra_info": {
            "source_properties": molecule.GetPropsAsDict(
                includePrivate=False, includeComputed=False
            ),
        },
    }


def _load_geom_row(row: pd.Series, *, remove_hydrogens: bool) -> dict[str, Any]:
    example = load_geom_conformer(
        row["molecule"],
        example_id=row["example_id"],
        remove_hydrogens=remove_hydrogens,
    )
    example["path"] = row["path"]
    example["extra_info"].update(
        {
            key: row[key]
            for key in ("smiles", "split", "molecule_index", "conformer_index")
        }
    )
    return example


def build_geom_raw_dataset(
    raw_dir: str | Path,
    split: str,
    *,
    transform: Callable | None = None,
    remove_hydrogens: bool = False,
    max_conformers_per_molecule: int | None = 5,
    max_samples: int | None = None,
) -> PandasDataset:
    """Index one GEOM split as individually sampled conformers.

    Reads ``raw_dir/{split}/{split}_data.pickle`` containing a sequence of
    ``(smiles, [rdkit_mol, ...])`` entries. Each molecule must hold one conformer.
    Only trusted dataset pickles should be used. The selected split is unpickled
    into memory once; AtomWorks parsing and transforms run on access. No files
    are generated, downloaded, or rewritten.

    Args:
        raw_dir: Directory containing the supplied train/val/test folders.
        split: Exactly ``train``, ``val``, or ``test``. Splits and source order
            are preserved, including repeated SMILES entries.
        transform: Optional Nucleus/AtomWorks transform.
        remove_hydrogens: Whether to remove explicit hydrogens during loading.
        max_conformers_per_molecule: Keep the first N conformers in each entry,
            matching the older zatom loader's default of five. ``None`` retains
            all. This limit is not an energy sort or a molecule split operation.
        max_samples: Keep at most this many conformer samples in source order,
            after the per-entry conformer limit. ``None`` retains all selected
            samples. Applied before distributed sampling.

    Returns:
        A ``PandasDataset`` with IDs ``geom/{split}/{entry_index}/{conformer_index}``.
        Sampling is uniform over selected conformers, not over SMILES groups.
        File layout alone does not establish benchmark provenance or disjointness.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown GEOM split {split!r}; use train, val, or test.")
    if max_conformers_per_molecule is not None and (
        isinstance(max_conformers_per_molecule, bool)
        or not isinstance(max_conformers_per_molecule, int)
        or max_conformers_per_molecule < 1
    ):
        raise ValueError("max_conformers_per_molecule must be a positive int or None.")
    if max_samples is not None and (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, int)
        or max_samples < 1
    ):
        raise ValueError("max_samples must be a positive int or None.")
    path = Path(raw_dir).resolve() / split / f"{split}_data.pickle"
    if not path.is_file():
        raise FileNotFoundError(f"Missing GEOM split pickle: {path}")
    with path.open("rb") as handle:
        entries = pickle.load(handle)  # noqa: S301 - supplied, trusted dataset
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ValueError(f"Expected a nonempty sequence of GEOM entries in {path}.")
    rows = []
    for molecule_index, entry in enumerate(entries):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(
                f"{path}: entry {molecule_index} must be (SMILES, conformers)."
            )
        smiles, conformers = entry
        if (
            not isinstance(smiles, str)
            or not smiles
            or not isinstance(conformers, (list, tuple))
            or not conformers
        ):
            raise ValueError(
                f"{path}: entry {molecule_index} has invalid SMILES/conformers."
            )
        for conformer_index, molecule in enumerate(
            conformers[:max_conformers_per_molecule]
        ):
            example_id = f"geom/{split}/{molecule_index}/{conformer_index}"
            if not isinstance(molecule, Chem.Mol):
                raise ValueError(f"{example_id}: expected an RDKit molecule.")
            rows.append(
                {
                    "example_id": example_id,
                    "molecule": molecule,
                    "smiles": smiles,
                    "split": split,
                    "molecule_index": molecule_index,
                    "conformer_index": conformer_index,
                    "path": path,
                }
            )
    return PandasDataset(
        data=pd.DataFrame(rows[:max_samples]),
        name=f"geom_{split}",
        loader=partial(_load_geom_row, remove_hydrogens=remove_hydrogens),
        transform=transform,
    )


class _GeomPipeline:
    """Require complete molecular targets through the shared training pipeline."""

    def __init__(self, pipeline, *, return_atom_array, remove_hydrogens, max_atoms):
        self.pipeline = pipeline
        self.return_atom_array = return_atom_array
        self.remove_hydrogens = remove_hydrogens
        self.max_atoms = max_atoms

    def __call__(self, data: dict) -> dict:
        atoms = data["atom_array"]
        keep = (
            ~np.isin(atoms.element, ["H", "D", "T"])
            if self.remove_hydrogens
            else np.ones(len(atoms), dtype=bool)
        )
        expected = np.sort(atoms.source_atom_index[keep])
        if len(expected) > self.max_atoms:
            raise ValueError(
                f"{data.get('example_id')}: complete GEOM molecule needs "
                f"{len(expected)} atoms/tokens, exceeding capacity {self.max_atoms}; "
                "increase crop_size and max_atoms_in_crop."
            )
        result = self.pipeline(data)
        atoms = result["atom_array"]
        if not np.array_equal(np.sort(atoms.source_atom_index), expected):
            raise ValueError("GEOM preprocessing must retain the complete molecule.")
        condition = result.get("sampled_condition_name")
        if (
            condition not in {"structure_prediction", "unconditional"}
            or atoms.is_motif_atom_with_fixed_coord.any()
            or not np.all(
                atoms.is_motif_atom_with_fixed_seq
                == (condition == "structure_prediction")
            )
        ):
            raise ValueError(
                "GEOM requires structure_prediction or unconditional conditioning "
                "with every coordinate generated."
            )
        # The shared pipeline retains motif coordinates for fixed-sequence
        # motifs too. GEOM has no fixed coordinates, so discard these unused
        # targets from the conditioning features before batching.
        motif_positions = result["feats"]["motif_pos"]
        result["feats"]["motif_pos"] = motif_positions.new_zeros(motif_positions.shape)
        if not self.return_atom_array:
            for key in ("atom_array", "specification", "conditions"):
                result.pop(key, None)
        return result


def build_geom_pipeline(**pipeline_kwargs) -> Callable:
    """Build ligand-only GEOM training/validation with complete molecular targets.

    Uses the shared Nucleus pipeline and its atomized ligand specialization.
    Coordinates are targets only: no reference conformers, force conditioning,
    or property supervision are allowed. Crop budgets are strict whole-molecule
    limits. Accepts ``build_atom14_base_pipeline`` keyword arguments.
    """
    from nucleus.transforms.pipelines import build_atom14_base_pipeline

    if pipeline_kwargs.get("is_inference", False):
        raise ValueError("GEOM training/validation requires is_inference=False.")
    if (
        pipeline_kwargs.get("generate_conformers", False)
        or pipeline_kwargs.get("provide_reference_conformer_when_unmasked", False)
        or pipeline_kwargs.get("ground_truth_conformer_policy", "IGNORE") != "IGNORE"
        or pipeline_kwargs.get("mean_force_norm_conditioning_probability", 0.0) != 0.0
        or pipeline_kwargs.get("enable_atomistic_property_targets", False)
    ):
        raise ValueError(
            "GEOM requires no reference conformers or property conditioning."
        )
    return_atom_array = pipeline_kwargs.get("return_atom_array", False)
    remove_hydrogens = pipeline_kwargs.get("remove_hydrogens", False)
    max_atoms = min(
        pipeline_kwargs["crop_size"],
        pipeline_kwargs.get("max_atoms_in_crop") or pipeline_kwargs["crop_size"],
    )
    pipeline_kwargs.update(
        return_atom_array=True,
        remove_hydrogens=remove_hydrogens,
        optimize_atomized_ligand_pipeline=True,
        mean_force_norm_conditioning_probability=0.0,
        crop_contiguous_probability=0.0,
        crop_spatial_probability=1.0,
        dna_contact_crop_probability=0.0,
        keep_full_binder_in_spatial_crop=False,
    )
    return _GeomPipeline(
        build_atom14_base_pipeline(**pipeline_kwargs),
        return_atom_array=return_atom_array,
        remove_hydrogens=remove_hydrogens,
        max_atoms=max_atoms,
    )
