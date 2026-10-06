"""Generated-sequence fold-back consistency through an independent predictor."""

from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any, Protocol

import yaml

from zevals.oracles.base import BaseStructurePredictor, ComplexPrediction
from zevals.proteins.designability import _write_fasta
from zevals.rna.interface import (
    compare_rna_protein_interfaces,
    contains_rna_protein_interface,
)
from zevals.rna.recovery import compare_rna_structures
from zevals.rna.tasks import (
    RNAChainSequence,
    RNAEvaluationSample,
    extract_rna_chain_sequences,
    load_biomolecule_structure,
    standardize_rna_sequences,
)
from zevals.rna.validity import _oracle_configuration, _sha256_file, _tm_score

logger = logging.getLogger(__name__)
_PROVENANCE_SCHEMA_VERSION = 1


class ComplexPredictor(Protocol):
    """Subset of the complete predictor interface required by this metric."""

    def predict_complex(
        self, input_path: Path, output_dir: Path
    ) -> list[ComplexPrediction]: ...


def compute_zatom_sequence_consistency_detailed(
    samples: list[RNAEvaluationSample],
    structure_predictor: BaseStructurePredictor,
    output_dir: Path,
    *,
    complex_predictor: ComplexPredictor | None = None,
    sc_tm_threshold: float = 0.45,
    tm_align_bin: str = "TMalign",
) -> tuple[dict[str, float], list[dict[str, Any]], list[str]]:
    """Refold a generated sequence and compare it with its target backbone.

    Unlike the existing RNA ``validity`` path, this function does not invoke
    NA-MPNN. It evaluates exactly the sequence produced by Zatom2N. In inverse
    design, the target is the supplied reference backbone when present; in de
    novo generation, it is the generated structure itself.
    """
    eligible = [
        sample for sample in samples if sample.task in {"inverse_design", "de_novo"}
    ]
    details: list[dict[str, Any]] = []
    sequences: list[str] = []
    for sample in eligible:
        try:
            chains = (
                standardize_rna_sequences(sample.sequence)
                if sample.sequence is not None
                else extract_rna_chain_sequences(
                    sample.structure_path, sample.modified_nucleotide_map
                )
            )
            sequences.extend(chain.sequence for chain in chains)
            # Inverse design asks whether Zatom2N's sequence recovers the supplied
            # backbone.  De novo consistency is different: even when a benchmark
            # carries a native structure for diagnostics, the generated sequence
            # must fold back to the generated structure rather than that native.
            target_path = (
                sample.reference_structure_path or sample.structure_path
                if sample.task == "inverse_design"
                else sample.structure_path
            )
            sample_output = output_dir / sample.sample_id
            sample_output.mkdir(parents=True, exist_ok=True)
            use_complex_predictor = (
                len(chains) > 1
                or any(chain.modified_residue_count for chain in chains)
                or _contains_protein(target_path)
            )
            predictor = (
                complex_predictor if use_complex_predictor else structure_predictor
            )
            if predictor is None:
                raise ValueError(
                    "Multi-chain, modified-RNA, and RNA--protein consistency "
                    "requires a configured complex_predictor"
                )
            configuration = _artifact_configuration(
                sample,
                target_path,
                [chain.sequence for chain in chains],
                predictor,
            )
            if use_complex_predictor:
                if any(chain.modified_residue_count for chain in chains) and (
                    sample.complex_input_path is None
                ):
                    raise ValueError(
                        "Modified-RNA consistency requires complex_input_path "
                        "with native predictor modification annotations"
                    )
                entities = (
                    None
                    if sample.complex_input_path
                    else _polymer_entities(sample.structure_path, chains)
                )
                prediction_paths, resumed, prediction_metadata = (
                    _predict_complex_or_resume(
                        sample_output,
                        sample.complex_input_path,
                        entities,
                        predictor,
                        configuration,
                        sample.sample_id,
                    )
                )
            else:
                prediction_paths, resumed = _predict_or_resume(
                    sample_output,
                    chains[0].sequence,
                    predictor,
                    configuration,
                )
                prediction_metadata = [{} for _ in prediction_paths]
            scores = [
                _tm_score(target_path, prediction_path, tm_align_bin)
                for prediction_path in prediction_paths
            ]
            best_index = max(range(len(scores)), key=scores.__getitem__)
            best_score = scores[best_index]
            recovery = compare_rna_structures(
                target_path,
                prediction_paths[best_index],
                modified_nucleotide_map=sample.modified_nucleotide_map,
            )
            interface_metrics: list[dict[str, float | int]] = []
            if use_complex_predictor and contains_rna_protein_interface(
                target_path, sample.modified_nucleotide_map
            ):
                interface_metrics = [
                    compare_rna_protein_interfaces(
                        target_path,
                        prediction_path,
                        modified_nucleotide_map=sample.modified_nucleotide_map,
                    )
                    for prediction_path in prediction_paths
                ]
            best_interface_index = (
                max(
                    range(len(interface_metrics)),
                    key=lambda index: float(
                        interface_metrics[index]["interface_contact_f1"]
                    ),
                )
                if interface_metrics
                else None
            )
            details.append(
                {
                    "sample_id": sample.sample_id,
                    "task": sample.task,
                    "structure": str(sample.structure_path),
                    "target_structure": str(target_path),
                    "status": "complete",
                    "sequences": [chain.sequence for chain in chains],
                    "sequence": chains[0].sequence if len(chains) == 1 else None,
                    "sequence_length": sum(len(chain.sequence) for chain in chains),
                    "rna_chain_count": len(chains),
                    "modified_residue_count": sum(
                        chain.modified_residue_count for chain in chains
                    ),
                    "prediction_mode": "joint_complex"
                    if use_complex_predictor
                    else "single_chain",
                    "max_scTM": best_score,
                    "is_consistent": best_score > sc_tm_threshold,
                    "predicted_structures": len(prediction_paths),
                    "resumed_predictions": resumed,
                    "best_refold": str(prediction_paths[best_index]),
                    "best_refold_iptm": prediction_metadata[best_index].get(
                        "iptm", float("nan")
                    ),
                    "best_refold_c1_aligned_rmsd": recovery["c1_aligned_rmsd"],
                    "best_refold_c1_lddt": recovery["c1_lddt"],
                    "best_refold_all_atom_lddt": recovery["all_atom_lddt"],
                    "interface_at_best_scTM_contact_f1": (
                        interface_metrics[best_index]["interface_contact_f1"]
                        if interface_metrics
                        else float("nan")
                    ),
                    "max_refold_interface_contact_f1": (
                        interface_metrics[best_interface_index]["interface_contact_f1"]
                        if best_interface_index is not None
                        else float("nan")
                    ),
                    "best_interface_refold": (
                        str(prediction_paths[best_interface_index])
                        if best_interface_index is not None
                        else None
                    ),
                }
            )
        except Exception as exc:
            logger.warning(
                "Zatom2N sequence consistency failed for %s",
                sample.sample_id,
                exc_info=True,
            )
            details.append(
                {
                    "sample_id": sample.sample_id,
                    "task": sample.task,
                    "structure": str(sample.structure_path),
                    "status": "failed",
                    "error": repr(exc),
                }
            )

    complete = [detail for detail in details if detail["status"] == "complete"]
    sc_tms = [float(detail["max_scTM"]) for detail in complete]
    aggregate = {
        "zatom_sequence_consistency_rate": (
            sum(score > sc_tm_threshold for score in sc_tms) / len(sc_tms)
            if sc_tms
            else float("nan")
        ),
        "zatom_sequence_mean_max_scTM": _mean(sc_tms),
        "zatom_sequence_mean_best_refold_c1_lddt": _mean(
            [float(detail["best_refold_c1_lddt"]) for detail in complete]
        ),
        "zatom_sequence_mean_max_refold_interface_contact_f1": _finite_mean(
            [float(detail["max_refold_interface_contact_f1"]) for detail in complete]
        ),
        "zatom_sequence_interface_consistency_coverage": (
            sum(
                math.isfinite(float(detail["max_refold_interface_contact_f1"]))
                for detail in complete
            )
            / len(complete)
            if complete
            else 0.0
        ),
        "zatom_sequence_consistency_coverage": (
            len(complete) / len(eligible) if eligible else 0.0
        ),
        "zatom_sequence_consistency_evaluated": float(len(complete)),
        "zatom_sequence_consistency_failed": float(len(eligible) - len(complete)),
    }
    return aggregate, details, sequences


def _predict_or_resume(
    sample_output: Path,
    sequence: str,
    structure_predictor: BaseStructurePredictor,
    configuration: dict[str, Any],
) -> tuple[list[Path], bool]:
    provenance_path = sample_output / "provenance.json"
    if provenance_path.is_file():
        try:
            provenance = json.loads(provenance_path.read_text())
        except (OSError, json.JSONDecodeError):
            provenance = {}
        stored_paths = [Path(path) for path in provenance.get("prediction_paths", [])]
        if (
            provenance.get("schema_version") == _PROVENANCE_SCHEMA_VERSION
            and provenance.get("configuration") == configuration
            and stored_paths
            and all(path.is_file() for path in stored_paths)
        ):
            return stored_paths, True

    fasta_path = sample_output / "zatom_sequence.fasta"
    _write_fasta(fasta_path, [sequence])
    prediction_dir = sample_output / "structures"
    pending_dir = sample_output / "structures.pending"
    if pending_dir.exists():
        shutil.rmtree(pending_dir)
    predictions = structure_predictor.predict(fasta_path, pending_dir)
    if not predictions or not all(
        prediction.pdb_path.is_file() for prediction in predictions
    ):
        raise RuntimeError("Structure predictor returned no complete RNA structures")

    pending_paths = [prediction.pdb_path for prediction in predictions]
    relative_paths: list[Path] = []
    for pending_path in pending_paths:
        try:
            relative_paths.append(pending_path.relative_to(pending_dir))
        except ValueError as exc:
            raise RuntimeError(
                "Structure predictor outputs must be written under its output_dir"
            ) from exc
    if prediction_dir.exists():
        shutil.rmtree(prediction_dir)
    pending_dir.replace(prediction_dir)
    prediction_paths = [prediction_dir / path for path in relative_paths]
    provenance = {
        "schema_version": _PROVENANCE_SCHEMA_VERSION,
        "configuration": configuration,
        "prediction_paths": [str(path) for path in prediction_paths],
    }
    pending_provenance = provenance_path.with_suffix(".json.pending")
    pending_provenance.write_text(json.dumps(provenance, indent=2, sort_keys=True))
    pending_provenance.replace(provenance_path)
    return prediction_paths, False


def _artifact_configuration(
    sample: RNAEvaluationSample,
    target_path: Path,
    sequences: list[str],
    structure_predictor: Any,
) -> dict[str, Any]:
    return {
        "structure_sha256": _sha256_file(sample.structure_path),
        "target_sha256": _sha256_file(target_path),
        "sequences": sequences,
        "complex_input_sha256": (
            _sha256_file(sample.complex_input_path)
            if sample.complex_input_path and sample.complex_input_path.is_file()
            else None
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
            ),
        ),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return _mean(finite)


def _predict_complex_or_resume(
    sample_output: Path,
    input_path: Path | None,
    entities: list[dict[str, Any]] | None,
    complex_predictor: ComplexPredictor,
    configuration: dict[str, Any],
    sample_id: str,
) -> tuple[list[Path], bool, list[dict[str, Any]]]:
    provenance_path = sample_output / "complex_provenance.json"
    if provenance_path.is_file():
        try:
            provenance = json.loads(provenance_path.read_text())
        except (OSError, json.JSONDecodeError):
            provenance = {}
        paths = [Path(path) for path in provenance.get("prediction_paths", [])]
        if (
            provenance.get("schema_version") == _PROVENANCE_SCHEMA_VERSION
            and provenance.get("configuration") == configuration
            and paths
            and all(path.is_file() for path in paths)
        ):
            return paths, True, list(provenance.get("prediction_metadata", []))

    prediction_dir = sample_output / "complex_structures"
    pending_dir = sample_output / "complex_structures.pending"
    if pending_dir.exists():
        shutil.rmtree(pending_dir)
    if input_path is not None:
        predictions = complex_predictor.predict_complex(input_path, pending_dir)
    elif entities is not None:
        input_path = sample_output / f"{sample_id}_complex_input.yaml"
        input_path.write_text(
            yaml.safe_dump({"version": 1, "sequences": entities}, sort_keys=False)
        )
        predictions = complex_predictor.predict_complex(input_path, pending_dir)
    else:
        raise ValueError(
            "Complex consistency requires native input or polymer entities"
        )
    if not predictions or not all(
        prediction.structure_path.is_file() for prediction in predictions
    ):
        raise RuntimeError("Complex predictor returned no complete structures")
    relative_paths: list[Path] = []
    for prediction in predictions:
        try:
            relative_paths.append(prediction.structure_path.relative_to(pending_dir))
        except ValueError as exc:
            raise RuntimeError(
                "Complex predictor outputs must be written under its output_dir"
            ) from exc
    if prediction_dir.exists():
        shutil.rmtree(prediction_dir)
    pending_dir.replace(prediction_dir)
    paths = [prediction_dir / path for path in relative_paths]
    metadata = [
        {"plddt": prediction.plddt, "ptm": prediction.ptm, "iptm": prediction.iptm}
        for prediction in predictions
    ]
    provenance = {
        "schema_version": _PROVENANCE_SCHEMA_VERSION,
        "configuration": configuration,
        "prediction_paths": [str(path) for path in paths],
        "prediction_metadata": metadata,
    }
    pending_provenance = provenance_path.with_suffix(".json.pending")
    pending_provenance.write_text(json.dumps(provenance, indent=2, sort_keys=True))
    pending_provenance.replace(provenance_path)
    return paths, False, metadata


def _contains_protein(path: Path) -> bool:
    protein_residues = {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
    structure = load_biomolecule_structure(path)
    model = next(structure.get_models())
    return any(
        residue.get_resname().strip().upper() in protein_residues
        for chain in model
        for residue in chain
    )


def _polymer_entities(
    structure_path: Path, rna_chains: list[RNAChainSequence]
) -> list[dict[str, Any]]:
    protein_residues = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
    entities: list[dict[str, Any]] = [
        {"rna": {"id": chain.chain_id, "sequence": chain.sequence}}
        for chain in rna_chains
    ]
    structure = load_biomolecule_structure(structure_path)
    model = next(structure.get_models())
    for chain in model:
        sequence = "".join(
            protein_residues.get(residue.get_resname().strip().upper(), "")
            for residue in chain
        )
        if sequence:
            entities.append(
                {
                    "protein": {
                        "id": str(chain.id),
                        "sequence": sequence,
                        "msa": "empty",
                    }
                }
            )
    return entities
