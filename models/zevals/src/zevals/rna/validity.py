"""RNA generation validity: self-consistency TM-score (scTM).

A generated RNA structure passes validity if, when its backbone is inverse-designed
into sequences (NA-MPNN or gRNAde) and re-predicted (Boltz-2 by default),
the re-predicted structure has scTM > 0.45 relative to the original.

The 0.45 threshold follows RNA-FrameFlow and RFDPoly conventions.
Note: RNA structure prediction is significantly less accurate than protein
prediction — results should be treated with appropriate scepticism.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from zevals.oracles.base import (
    BaseInverseFolder,
    BaseStructurePredictor,
    resolve_binary,
    run_subprocess,
)
from zevals.oracles.inverse_folders._na_mpnn.featurize import (
    to_standard_nucleotide_alphabet,
)
from zevals.proteins.designability import _write_fasta

# Default structural reference atom for RNA scTM — matches
# oracles/inverse_folders/na_mpnn.yaml's reference atom.
_DEFAULT_SC_ATOM = "C1'"
_MOL = "RNA"
logger = logging.getLogger(__name__)
_PROVENANCE_SCHEMA_VERSION = 1


def _usalign_atom(atom: str) -> str:
    """Format a clean atom token (e.g. ``"C1'"``, ``"C4'"``, ``"CA"``) into the
    4-character PDB atom-name field USalign's ``-atom`` flag expects: a leading
    space then the name left-justified in the remaining three columns
    (e.g. ``" C1'"``, ``" CA "``). This reproduces USalign's own documented
    defaults for single-character-element atoms (the C1'/C4'/P/CA/N/C/O cases
    relevant to nucleic-acid and protein backbones)."""
    return " " + f"{atom:<3}"


def compute_validity(
    backbone_pdbs: list[Path],
    structure_predictor: BaseStructurePredictor,
    inverse_folder: BaseInverseFolder,
    output_dir: Path,
    sc_tm_threshold: float = 0.45,
    tm_align_bin: str = "TMalign",
    sc_atom: str = _DEFAULT_SC_ATOM,
) -> dict[str, float]:
    """Fraction of generated RNA structures with scTM > sc_tm_threshold.

    Pipeline per structure:
      1. NA-MPNN / gRNAde → sequences (inverse fold).
      2. Boltz-2 (primary) or RhoFold+ (sensitivity) → re-predicted structure.
      3. TM-score between re-predicted and original backbone.
      4. Valid if max(scTM) > sc_tm_threshold.

    Args:
        backbone_pdbs:       Paths to generated RNA backbone structure files
                             (`.cif` or `.pdb`).
        structure_predictor: RNA forward-folding oracle (Boltz-2 by default).
        inverse_folder:      RNA inverse-folding oracle (NA-MPNN / gRNAde).
        output_dir:          Root directory for oracle I/O.
        sc_tm_threshold:     scTM cutoff (default 0.45, RNA-FrameFlow convention).
        tm_align_bin:        Path to the TMalign binary (part of USalign —
                              `zevals setup --tools usalign`; default assumes
                              it's on PATH).
        sc_atom:             Per-residue reference atom for the scTM alignment
                             (default ``"C1'"``; e.g. ``"C4'"``/``"P"`` for
                             other nucleic-acid conventions).

    Returns:
        {"validity": float in [0, 1], "mean_max_scTM": float}
    """
    aggregate, _, _ = compute_validity_detailed(
        backbone_pdbs,
        structure_predictor,
        inverse_folder,
        output_dir,
        sc_tm_threshold=sc_tm_threshold,
        tm_align_bin=tm_align_bin,
        sc_atom=sc_atom,
    )
    return aggregate


def compute_validity_detailed(
    backbone_pdbs: list[Path],
    structure_predictor: BaseStructurePredictor,
    inverse_folder: BaseInverseFolder,
    output_dir: Path,
    sc_tm_threshold: float = 0.45,
    tm_align_bin: str = "TMalign",
    sc_atom: str = _DEFAULT_SC_ATOM,
) -> tuple[dict[str, float], list[dict[str, Any]], list[str]]:
    """Compute RNA self-consistency with explicit per-backbone oracle status.

    Technical oracle failures are excluded from the validity denominator and
    surfaced through ``evaluation_coverage`` and ``failed_backbones``. They are
    never silently converted into invalid model outputs.
    """
    details: list[dict[str, Any]] = []
    designed_sequences: list[str] = []

    for pdb_path in backbone_pdbs:
        artifact_configuration = _artifact_configuration(
            inverse_folder, structure_predictor, pdb_path
        )
        bb_out = output_dir / pdb_path.stem
        bb_out.mkdir(parents=True, exist_ok=True)
        try:
            provenance_path = bb_out / "provenance.json"
            provenance = _read_matching_provenance(
                provenance_path, artifact_configuration
            )
            fasta_path = bb_out / "designed.fasta"
            standard_sequences = _read_fasta_sequences(fasta_path)
            expected_sequences = getattr(inverse_folder, "num_sequences", None)
            resumed_sequences = (
                provenance.get("sequences_complete") is True
                and bool(standard_sequences)
                and (
                    expected_sequences is None
                    or len(standard_sequences) == expected_sequences
                )
            )
            if not resumed_sequences:
                inv_result = inverse_folder.design(pdb_path, bb_out / "sequences")
                if not inv_result.sequences:
                    raise RuntimeError("Inverse folder returned no RNA sequences")
                standard_sequences = [
                    _standardize_single_rna_chain(sequence)
                    for sequence in inv_result.sequences
                ]
                pending_fasta = fasta_path.with_suffix(".fasta.pending")
                _write_fasta(pending_fasta, standard_sequences)
                pending_fasta.replace(fasta_path)
                provenance = _new_provenance(artifact_configuration)
                provenance["sequences_complete"] = True
                _write_json_atomic(provenance_path, provenance)
            designed_sequences.extend(standard_sequences)

            structure_dir = bb_out / "structures"
            predicted_paths = [
                structure_dir / f"{index:04d}" / "unrelaxed_model.pdb"
                for index in range(len(standard_sequences))
            ]
            resumed_predictions = provenance.get(
                "predictions_complete"
            ) is True and all(path.is_file() for path in predicted_paths)
            if not resumed_predictions:
                pending_structure_dir = bb_out / "structures.pending"
                if pending_structure_dir.exists():
                    shutil.rmtree(pending_structure_dir)
                predictions = structure_predictor.predict(
                    fasta_path, pending_structure_dir
                )
                if not predictions:
                    raise RuntimeError("Structure predictor returned no RNA structures")
                pending_paths = [prediction.pdb_path for prediction in predictions]
                if len(pending_paths) != len(standard_sequences) or not all(
                    path.is_file() for path in pending_paths
                ):
                    raise RuntimeError(
                        "Structure predictor did not produce one RNA structure "
                        "for every designed sequence"
                    )
                if structure_dir.exists():
                    shutil.rmtree(structure_dir)
                pending_structure_dir.replace(structure_dir)
                predicted_paths = [
                    structure_dir / path.relative_to(pending_structure_dir)
                    for path in pending_paths
                ]
                provenance["predictions_complete"] = True
                _write_json_atomic(provenance_path, provenance)

            tms = [
                _tm_score(pdb_path, predicted_path, tm_align_bin, sc_atom)
                for predicted_path in predicted_paths
            ]
            max_sc_tm = max(tms)
            details.append(
                {
                    "structure": str(pdb_path),
                    "status": "complete",
                    "max_scTM": max_sc_tm,
                    "is_designable": max_sc_tm > sc_tm_threshold,
                    "designed_sequences": len(standard_sequences),
                    "predicted_structures": len(predicted_paths),
                    "resumed_sequences": resumed_sequences,
                    "resumed_predictions": resumed_predictions,
                }
            )
        except Exception as exc:
            logger.warning(
                "RNA self-consistency failed for %s", pdb_path, exc_info=True
            )
            details.append(
                {
                    "structure": str(pdb_path),
                    "status": "failed",
                    "error": repr(exc),
                }
            )

    complete = [detail for detail in details if detail["status"] == "complete"]
    sc_tms = [float(detail["max_scTM"]) for detail in complete]
    total = len(backbone_pdbs)
    evaluated = len(complete)
    aggregate = {
        "validity": (
            sum(value > sc_tm_threshold for value in sc_tms) / evaluated
            if evaluated
            else float("nan")
        ),
        "mean_max_scTM": (sum(sc_tms) / evaluated if evaluated else float("nan")),
        "evaluation_coverage": evaluated / total if total else 0.0,
        "evaluated_backbones": float(evaluated),
        "failed_backbones": float(total - evaluated),
    }
    return aggregate, details, designed_sequences


def _artifact_configuration(
    inverse_folder: BaseInverseFolder,
    structure_predictor: BaseStructurePredictor,
    backbone_path: Path,
) -> dict[str, Any]:
    """Return stable input and oracle settings that determine artifacts."""
    return {
        "backbone_sha256": _sha256_file(backbone_path),
        "inverse_folder": _oracle_configuration(
            inverse_folder,
            (
                "num_sequences",
                "temperature",
                "backbone_noise",
                "batch_size",
                "seed",
                "checkpoint_path",
                "expected_checkpoint_sha256",
            ),
        ),
        "structure_predictor": _oracle_configuration(
            structure_predictor,
            (
                "num_samples",
                "num_recycles",
                "device",
                "relax_steps",
                "checkpoint_path",
                "expected_checkpoint_sha256",
                "repo_path",
                "expected_repo_revision",
                "conda_env",
                "singularity_image",
                "entity_type",
                "executable",
                "cache_dir",
                "model",
                "model_name",
                "model_revision",
                "num_loops",
                "sampling_steps",
                "num_sampling_steps",
                "seed",
                "no_kernels",
            ),
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _oracle_configuration(oracle: Any, attributes: tuple[str, ...]) -> dict[str, Any]:
    configuration: dict[str, Any] = {
        "class": f"{oracle.__class__.__module__}.{oracle.__class__.__qualname__}"
    }
    for attribute in attributes:
        if hasattr(oracle, attribute):
            configuration[attribute] = _json_value(getattr(oracle, attribute))
    return configuration


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _new_provenance(configuration: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": _PROVENANCE_SCHEMA_VERSION,
        "configuration": configuration,
        "sequences_complete": False,
        "predictions_complete": False,
    }


def _read_matching_provenance(
    path: Path, configuration: dict[str, Any]
) -> dict[str, Any]:
    if not path.is_file():
        return _new_provenance(configuration)
    try:
        provenance = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return _new_provenance(configuration)
    if (
        provenance.get("schema_version") != _PROVENANCE_SCHEMA_VERSION
        or provenance.get("configuration") != configuration
    ):
        return _new_provenance(configuration)
    return provenance


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    pending_path = path.with_suffix(".json.pending")
    pending_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    pending_path.replace(path)


def _standardize_single_rna_chain(sequence: str) -> str:
    """Translate NA-MPNN output to one standard A/C/G/U RNA chain."""
    chains = sequence.split("/")
    if len(chains) != 1:
        raise ValueError(
            "RNA self-consistency currently requires one generated RNA chain; "
            f"received {len(chains)} chains"
        )
    standard = to_standard_nucleotide_alphabet(chains[0]).upper()
    invalid = sorted(set(standard) - set("ACGU"))
    if invalid:
        raise ValueError(
            f"Forward-fold RNA sequence contains unsupported symbols: {invalid}"
        )
    return standard


def _read_fasta_sequences(path: Path) -> list[str]:
    """Read an existing standardized FASTA for restartable evaluation."""
    if not path.is_file():
        return []
    sequences: list[str] = []
    parts: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if parts:
                sequences.append("".join(parts).upper())
            parts = []
        elif line.strip():
            parts.append(line.strip())
    if parts:
        sequences.append("".join(parts).upper())
    if any(set(sequence) - set("ACGU") for sequence in sequences):
        return []
    return sequences


def _tm_score(
    ref_pdb: Path,
    pred_pdb: Path,
    tm_align_bin: str = "TMalign",
    sc_atom: str = _DEFAULT_SC_ATOM,
) -> float:
    """TM-score between two RNA structures via TMalign, normalized by
    `ref_pdb`'s length — the self-consistency / scTM convention (matches how
    protein designability's Cα RMSD is measured against the original
    backbone, not the refold). Uses the `sc_atom` reference atom (default
    ``"C1'"``, matching NA-MPNN's own structural reference atom).
    """
    result = run_subprocess(
        [
            resolve_binary(
                "TMalign", tm_align_bin if tm_align_bin != "TMalign" else None
            ),
            str(ref_pdb),
            str(pred_pdb),
            "-mol",
            _MOL,
            "-atom",
            _usalign_atom(sc_atom),
            "-outfmt",
            "2",
        ],
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            return float(parts[2])  # TM1, normalized by ref_pdb (chain1)
    return 0.0
