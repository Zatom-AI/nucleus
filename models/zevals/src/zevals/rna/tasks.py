"""Typed inputs for task-aware RNA model RNA evaluation."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

RNATask = Literal["inverse_design", "structure_prediction", "de_novo"]
_SUPPORTED_TASKS = {"inverse_design", "structure_prediction", "de_novo"}
_CANONICAL_RNA = {"A": "A", "C": "C", "G": "G", "U": "U"}
# Common PDB Chemical Component Dictionary residue names with an unambiguous
# canonical RNA parent. Callers can extend or override this table per sample.
DEFAULT_MODIFIED_NUCLEOTIDE_MAP = {
    "1MA": "A",
    "2MA": "A",
    "6MA": "A",
    "MIA": "A",
    "5MC": "C",
    "OMC": "C",
    "M2G": "G",
    "7MG": "G",
    "OMG": "G",
    "YG": "G",
    "H2U": "U",
    "5MU": "U",
    "PSU": "U",
    "4SU": "U",
    "OMU": "U",
}


@dataclass(frozen=True)
class RNAChainSequence:
    """Canonical parent sequence and residue provenance for one RNA chain."""

    chain_id: str
    sequence: str
    residue_names: tuple[str, ...]
    modified_residue_count: int


@dataclass(frozen=True)
class RNAEvaluationSample:
    """One RNA model output and the context needed to evaluate its task.

    ``structure_path`` is always the RNA model output. For inverse design it
    contains the sequence generated on the supplied backbone. For structure
    prediction it is the prediction made from ``sequence``. For de novo
    generation it is the unconstrained or constraint-conditioned output.
    """

    sample_id: str
    task: RNATask
    structure_path: Path
    sequence: str | list[str] | dict[str, str] | None = None
    reference_structure_path: Path | None = None
    constraints: dict[str, Any] | None = None
    modified_nucleotide_map: dict[str, str] | None = None
    complex_input_path: Path | None = None

    def __post_init__(self) -> None:
        if self.task not in _SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported RNA task {self.task!r}; expected one of "
                f"{sorted(_SUPPORTED_TASKS)}"
            )
        if not self.sample_id:
            raise ValueError("RNA evaluation sample_id must not be empty")
        if Path(self.sample_id).name != self.sample_id:
            raise ValueError("RNA evaluation sample_id must be a single path-safe name")
        if (
            self.task == "structure_prediction"
            and self.reference_structure_path is None
        ):
            raise ValueError(
                "structure_prediction samples require reference_structure_path "
                "to measure target recovery"
            )


def load_rna_task_manifest(path: Path | str) -> list[RNAEvaluationSample]:
    """Load task samples from a JSON manifest.

    Relative structure paths are resolved relative to the manifest. The root
    may be either ``{"samples": [...]}`` or the sample list itself.
    """
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text())
    rows = payload["samples"] if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("RNA task manifest must contain a list of samples")

    samples: list[RNAEvaluationSample] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"RNA task manifest sample {index} must be an object")
        sample_id = str(row.get("sample_id", row.get("id", f"sample_{index:04d}")))
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate RNA task sample_id: {sample_id!r}")
        seen_ids.add(sample_id)

        structure_value = row.get("structure_path", row.get("structure"))
        if not structure_value:
            raise ValueError(f"RNA task sample {sample_id!r} has no structure_path")
        reference_value = row.get(
            "reference_structure_path", row.get("reference_structure")
        )
        samples.append(
            RNAEvaluationSample(
                sample_id=sample_id,
                task=str(row["task"]),
                structure_path=_resolve_manifest_path(
                    manifest_path.parent, structure_value
                ),
                sequence=row.get("sequence"),
                reference_structure_path=(
                    _resolve_manifest_path(manifest_path.parent, reference_value)
                    if reference_value
                    else None
                ),
                constraints=row.get("constraints"),
                modified_nucleotide_map=row.get("modified_nucleotide_map"),
                complex_input_path=(
                    _resolve_manifest_path(
                        manifest_path.parent, row["complex_input_path"]
                    )
                    if row.get("complex_input_path")
                    else None
                ),
            )
        )
    return samples


def extract_rna_chain_sequences(
    path: Path | str,
    modified_nucleotide_map: Mapping[str, str] | None = None,
) -> list[RNAChainSequence]:
    """Extract RNA chains without concatenating them or losing modifications.

    Modified residues are converted only through an explicit canonical-parent
    mapping: the built-in map covers common modifications and a sample-level
    map can extend or override it. Unknown nucleotide-like residues fail
    loudly so sequence-based metrics never operate on a silently altered RNA.
    """
    structure = load_biomolecule_structure(Path(path))
    model = next(structure.get_models())
    parent_map = canonical_parent_map(modified_nucleotide_map)
    chains: list[RNAChainSequence] = []
    unknown_residues: set[str] = set()
    for chain in model:
        chain_sequence: list[str] = []
        residue_names: list[str] = []
        modified_count = 0
        for residue in chain:
            residue_name = residue.get_resname().strip().upper()
            parent = parent_map.get(residue_name)
            if parent is not None and _looks_like_nucleotide(residue):
                chain_sequence.append(parent)
                residue_names.append(residue_name)
                modified_count += int(residue_name not in _CANONICAL_RNA)
            elif _looks_like_nucleotide(residue):
                unknown_residues.add(residue_name)
        if chain_sequence:
            chains.append(
                RNAChainSequence(
                    chain_id=str(chain.id),
                    sequence="".join(chain_sequence),
                    residue_names=tuple(residue_names),
                    modified_residue_count=modified_count,
                )
            )

    if unknown_residues:
        raise ValueError(
            f"{path} contains unsupported or modified nucleotide residues: "
            f"{sorted(unknown_residues)}"
        )
    if not chains:
        raise ValueError(f"{path} contains no supported RNA chains")
    return chains


def extract_single_chain_rna_sequence(
    path: Path | str,
    modified_nucleotide_map: Mapping[str, str] | None = None,
) -> str:
    """Extract one RNA chain, retaining the legacy strict chain-count policy."""
    chains = extract_rna_chain_sequences(path, modified_nucleotide_map)
    if len(chains) != 1:
        raise ValueError(
            f"{path} must contain exactly one RNA chain; found {len(chains)}"
        )
    return chains[0].sequence


def standardize_rna_sequence(sequence: str) -> str:
    """Validate and standardize an explicitly supplied single-chain RNA sequence."""
    standardized = sequence.strip().upper().replace("T", "U")
    invalid = sorted(set(standardized) - set(_CANONICAL_RNA))
    if not standardized or invalid:
        raise ValueError(
            "RNA sequence must be a non-empty canonical A/C/G/U chain; "
            f"invalid symbols: {invalid}"
        )
    return standardized


def standardize_rna_sequences(
    sequence: str | Sequence[str] | Mapping[str, str],
) -> list[RNAChainSequence]:
    """Standardize an explicit one- or multi-chain RNA sequence value."""
    if isinstance(sequence, str):
        records = [("A", sequence)]
    elif isinstance(sequence, Mapping):
        records = [(str(chain_id), value) for chain_id, value in sequence.items()]
    else:
        records = [(str(index), value) for index, value in enumerate(sequence)]
    if not records:
        raise ValueError("RNA sequence collection must not be empty")
    return [
        RNAChainSequence(
            chain_id=chain_id,
            sequence=standardize_rna_sequence(value),
            residue_names=(),
            modified_residue_count=0,
        )
        for chain_id, value in records
    ]


def canonical_parent_map(
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return validated residue-name to canonical RNA-parent mappings."""
    mapping = {**_CANONICAL_RNA, **DEFAULT_MODIFIED_NUCLEOTIDE_MAP}
    for residue_name, parent in dict(overrides or {}).items():
        normalized_parent = str(parent).strip().upper().replace("T", "U")
        if normalized_parent not in _CANONICAL_RNA:
            raise ValueError(
                f"Modified nucleotide {residue_name!r} has invalid RNA parent "
                f"{parent!r}; expected A, C, G, or U"
            )
        mapping[str(residue_name).strip().upper()] = normalized_parent
    return mapping


def rna_parent_base(
    residue: Any, modified_nucleotide_map: Mapping[str, str] | None = None
) -> str | None:
    """Return an RNA residue's canonical parent, or ``None`` if unsupported."""
    if not _looks_like_nucleotide(residue):
        return None
    return canonical_parent_map(modified_nucleotide_map).get(
        residue.get_resname().strip().upper()
    )


def load_biomolecule_structure(path: Path):
    """Load a biomolecular PDB/mmCIF, including gzip-compressed mmCIF."""
    from Bio.PDB import MMCIFParser, PDBParser  # type: ignore[import]

    lower_name = path.name.lower()
    if lower_name.endswith(".cif.gz") or lower_name.endswith(".mmcif.gz"):
        with gzip.open(path, "rt") as handle:
            return MMCIFParser(QUIET=True).get_structure(path.stem, handle)
    if lower_name.endswith((".cif", ".mmcif")):
        return MMCIFParser(QUIET=True).get_structure(path.stem, str(path))
    if lower_name.endswith((".pdb", ".ent")):
        return PDBParser(QUIET=True).get_structure(path.stem, str(path))
    raise ValueError(f"Unsupported biomolecule structure format: {path}")


def _looks_like_nucleotide(residue: Any) -> bool:
    atom_names = {atom.get_name().strip().replace("*", "'") for atom in residue}
    return "C1'" in atom_names and bool({"P", "C4'", "O4'"} & atom_names)


def _resolve_manifest_path(base: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (base / candidate).resolve()
