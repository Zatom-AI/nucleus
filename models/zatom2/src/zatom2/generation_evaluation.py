"""Sample and evaluate Zatom-2 molecule/material generation checkpoints."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import pickle
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

LOGGER = logging.getLogger(__name__)
DEFAULT_FORCE_AUROC_CUTOFFS = (0.5, 0.75, 1.0)


@dataclass
class GeneratedSample:
    """One generated output and the JSON specification that conditioned it."""

    sample_index: int
    sample_id: str
    target_id: str
    batch_index: int
    model_index: int
    specification: dict[str, Any]
    output: Any

    @property
    def atom_array(self) -> Any:
        """Return the generated atom array."""
        return self.output.atom_array


@dataclass
class _SavedSampleOutput:
    """Minimal inference-output interface reconstructed from saved artifacts."""

    example_id: str
    atom_array: Any
    metadata: dict[str, Any]


def _plain_mapping(value: Any) -> dict[str, Any]:
    """Convert an OmegaConf or ordinary mapping to a detached plain dict."""
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return dict(value) if isinstance(value, Mapping) else {}


def _load_input_specifications(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the inference JSON and apply its optional ``global_args`` block."""
    input_path = Path(path).expanduser().resolve()
    with input_path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Inference JSON must contain an object: {input_path}")

    global_args = data.pop("global_args", {})
    if not isinstance(global_args, dict):
        raise TypeError("Inference JSON global_args must be an object.")
    specifications: dict[str, dict[str, Any]] = {}
    for target_id, value in data.items():
        if not isinstance(value, dict):
            raise TypeError(f"{target_id}: inference specification must be an object.")
        specification = value | global_args
        input_value = specification.get("input")
        if input_value is not None and not Path(input_value).is_absolute():
            specification["input"] = str((input_path.parent / input_value).resolve())
        specifications[str(target_id)] = specification
    return specifications


def _select_input_specifications(
    specifications: Mapping[str, dict[str, Any]],
    *,
    expected_num_samples: int,
    n_batches: int,
    diffusion_batch_size: int,
    json_keys_subset: str | Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Select enough JSON entries for the requested or available output count.

    One JSON specification produces ``n_batches * diffusion_batch_size``
    outputs. Selection happens before constructing the inference engine, so
    ``expected_num_samples`` limits generation rather than merely checking the
    result after all JSON entries have already been sampled. If the selected
    JSON contains fewer entries than requested, all available entries are used.
    """
    if expected_num_samples <= 0:
        raise ValueError("expected_num_samples must be positive.")
    if n_batches <= 0 or diffusion_batch_size <= 0:
        raise ValueError("n_batches and diffusion_batch_size must be positive.")

    if json_keys_subset is None:
        candidate_ids = list(specifications)
    else:
        requested_ids = (
            [key for key in json_keys_subset.split(",") if key]
            if isinstance(json_keys_subset, str)
            else [str(key) for key in json_keys_subset]
        )
        candidate_ids = list(dict.fromkeys(requested_ids))
        missing = [key for key in candidate_ids if key not in specifications]
        if missing:
            raise ValueError(f"json_keys_subset contains unknown keys: {missing}")

    if not candidate_ids:
        raise ValueError("No input specifications are available for generation.")

    samples_per_specification = n_batches * diffusion_batch_size
    available_num_samples = len(candidate_ids) * samples_per_specification
    effective_num_samples = min(expected_num_samples, available_num_samples)
    if effective_num_samples % samples_per_specification:
        raise ValueError(
            "expected_num_samples must be divisible by the number of outputs per "
            f"input specification ({samples_per_specification})."
        )
    if effective_num_samples < expected_num_samples:
        LOGGER.warning(
            "Requested %d samples, but the selected input JSON can produce only "
            "%d; using all available input specifications.",
            expected_num_samples,
            effective_num_samples,
        )

    required_specifications = effective_num_samples // samples_per_specification
    selected_ids = candidate_ids[:required_specifications]
    return {target_id: specifications[target_id] for target_id in selected_ids}


def _canonical_target_prefix(
    *, input_path: str | Path, target_id: str, global_prefix: str | None
) -> str:
    """Mirror RFD3's JSON key prefixing without importing RFD3."""
    if global_prefix is None:
        return f"{Path(input_path).stem}_{target_id}"
    return f"{global_prefix}{target_id}"


def _flatten_outputs(
    outputs: Mapping[str, Sequence[Any]],
    specifications: Mapping[str, dict[str, Any]],
    *,
    input_path: str | Path,
    global_prefix: str | None,
    n_batches: int,
) -> list[GeneratedSample]:
    """Flatten inference outputs in target/batch/model order with exact metadata."""
    expected_groups: dict[str, tuple[str, int]] = {}
    for target_id in specifications:
        prefix = _canonical_target_prefix(
            input_path=input_path,
            target_id=target_id,
            global_prefix=global_prefix,
        )
        for batch_index in range(n_batches):
            expected_groups[f"{prefix}_{batch_index}"] = (target_id, batch_index)

    missing = set(expected_groups) - set(outputs)
    unexpected = set(outputs) - set(expected_groups)
    if missing or unexpected:
        raise ValueError(
            "Inference output groups do not match the requested JSON/batches; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
        )

    samples: list[GeneratedSample] = []
    for generated_group, (target_id, batch_index) in expected_groups.items():
        for model_index, output in enumerate(outputs[generated_group]):
            metadata = _plain_mapping(getattr(output, "metadata", {}))
            specification = _plain_mapping(metadata.get("specification"))
            if not specification:
                specification = dict(specifications[target_id])
            samples.append(
                GeneratedSample(
                    sample_index=len(samples),
                    sample_id=str(output.example_id),
                    target_id=target_id,
                    batch_index=batch_index,
                    model_index=model_index,
                    specification=specification,
                    output=output,
                )
            )
    return samples


def _specification_value(specification: Mapping[str, Any], name: str) -> Any:
    """Read and unbatch a value from the top level or ``extra`` block."""
    if name in specification:
        value = specification[name]
    else:
        extra = specification.get("extra")
        value = extra.get(name) if isinstance(extra, Mapping) else None

    # Inference metadata comes from a collated batch and serializes scalar
    # specification fields as singleton lists, e.g. ``[0.1]`` or ``[-1]``.
    # The source inference JSON contains ordinary scalars, so normalize both
    # representations at this boundary.
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        value = value.detach().cpu().item()
    return value


def _conditioning_force(specification: Mapping[str, Any]) -> float | None:
    """Return an enabled, finite mean-force-norm target in eV/A."""
    has_target = _specification_value(specification, "has_mean_force_norm")
    if has_target is not None and not bool(has_target):
        return None
    value = _specification_value(specification, "mean_force_norm")
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid mean_force_norm conditioning value: {value!r}")
    return value


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based ranks with average ranks for ties."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    """Return Pearson correlation, or None when it is undefined."""
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def _binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Compute tie-aware binary AUROC via the Mann-Whitney statistic."""
    positives = labels.astype(bool)
    n_positive = int(positives.sum())
    n_negative = len(labels) - n_positive
    if n_positive == 0 or n_negative == 0:
        return None
    rank_sum = float(_average_ranks(scores)[positives].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def _normalize_force_cutoffs(cutoffs: Sequence[float]) -> tuple[float, ...]:
    """Return sorted, unique, finite force-bin thresholds."""
    normalized = tuple(sorted({float(cutoff) for cutoff in cutoffs}))
    if not normalized:
        raise ValueError("At least one force AUROC cutoff is required.")
    if any(not math.isfinite(cutoff) or cutoff < 0 for cutoff in normalized):
        raise ValueError(
            "Force AUROC cutoffs must all be finite and non-negative, got "
            f"{list(cutoffs)!r}."
        )
    return normalized


def _force_bin_auroc_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    cutoffs: Sequence[float],
) -> dict[str, dict[str, int | float | None]]:
    """Compute UMA AUROC for high/low ground-truth force bins at each cutoff."""
    results = {}
    for cutoff in _normalize_force_cutoffs(cutoffs):
        high_force = targets >= cutoff
        results[str(float(cutoff))] = {
            "high_force_target_count": int(high_force.sum()),
            "low_force_target_count": int((~high_force).sum()),
            "auroc": _binary_auroc(high_force, predictions),
        }
    return results


def compute_force_consistency_metrics(
    targets: Sequence[float],
    predictions: Sequence[float],
    *,
    cutoff: float = 1.0,
    auroc_cutoffs: Sequence[float] | None = DEFAULT_FORCE_AUROC_CUTOFFS,
) -> dict[str, Any]:
    """Compare conditioned and UMA-predicted mean force norms.

    AUROC treats ``target >= threshold`` as the high-force positive class and
    uses the continuous UMA prediction as its score. The legacy top-level
    ``auroc`` uses ``cutoff``; ``auroc_by_ground_truth_cutoff_eV_per_angstrom``
    reports every threshold in ``auroc_cutoffs``. Passing ``None`` suppresses
    the multi-cutoff calculation. Correlations are undefined (and represented
    by ``None``) for fewer than two points or constant inputs.
    """
    target_array = np.asarray(targets, dtype=np.float64)
    prediction_array = np.asarray(predictions, dtype=np.float64)
    if target_array.shape != prediction_array.shape:
        raise ValueError("Force targets and predictions must have the same shape.")
    if target_array.ndim != 1:
        raise ValueError("Force targets and predictions must be one-dimensional.")
    if not math.isfinite(float(cutoff)) or cutoff < 0:
        raise ValueError(f"cutoff must be finite and non-negative, got {cutoff!r}.")
    finite = np.isfinite(target_array) & np.isfinite(prediction_array)
    target_array = target_array[finite]
    prediction_array = prediction_array[finite]
    labels = target_array >= float(cutoff)
    auroc_by_cutoff = (
        _force_bin_auroc_metrics(
            target_array,
            prediction_array,
            auroc_cutoffs,
        )
        if auroc_cutoffs is not None
        else None
    )

    if len(target_array) == 0:
        metrics = {
            "num_paired_samples": 0,
            "cutoff_eV_per_angstrom": float(cutoff),
            "positive_target_count": 0,
            "negative_target_count": 0,
            "mae_eV_per_angstrom": None,
            "auroc": None,
            "spearman": None,
            "pearson": None,
        }
    else:
        metrics = {
            "num_paired_samples": len(target_array),
            "cutoff_eV_per_angstrom": float(cutoff),
            "positive_target_count": int(labels.sum()),
            "negative_target_count": int((~labels).sum()),
            "mae_eV_per_angstrom": float(
                np.mean(np.abs(prediction_array - target_array))
            ),
            "auroc": _binary_auroc(labels, prediction_array),
            "spearman": _pearson(
                _average_ranks(target_array), _average_ranks(prediction_array)
            ),
            "pearson": _pearson(target_array, prediction_array),
        }
    if auroc_by_cutoff is not None:
        metrics["auroc_by_ground_truth_cutoff_eV_per_angstrom"] = auroc_by_cutoff
    return metrics


def _aggregate_uma_scores(
    scores: Sequence[Mapping[str, int | float]], *, total_samples: int
) -> dict[str, float]:
    """Reduce per-sample UMA scores using the existing zevals definitions."""
    if not scores:
        return {
            "uma_scored_rate": 0.0,
            "uma_force_max_mean": 0.0,
            "uma_force_max_median": 0.0,
            "uma_mean_force_norm_mean": 0.0,
            "uma_force_rms_mean": 0.0,
            "uma_energy_per_atom_mean": 0.0,
        }
    return {
        "uma_scored_rate": len(scores) / total_samples if total_samples else 0.0,
        "uma_force_max_mean": float(np.mean([row["force_max"] for row in scores])),
        "uma_force_max_median": float(np.median([row["force_max"] for row in scores])),
        "uma_mean_force_norm_mean": float(
            np.mean([row["mean_force_norm"] for row in scores])
        ),
        "uma_force_rms_mean": float(np.mean([row["force_rms"] for row in scores])),
        "uma_energy_per_atom_mean": float(
            np.mean([row["energy_per_atom"] for row in scores])
        ),
    }


def _write_scatter_plot(
    path: Path,
    targets: Sequence[float],
    predictions: Sequence[float],
    *,
    cutoff: float,
) -> None:
    """Write the requested conditioned-force versus UMA scatter plot."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    target_array = np.asarray(targets, dtype=float)
    prediction_array = np.asarray(predictions, dtype=float)
    figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    if len(target_array):
        axis.scatter(target_array, prediction_array, alpha=0.55, s=18)
        lower = min(0.0, float(target_array.min()), float(prediction_array.min()))
        upper = max(
            float(cutoff), float(target_array.max()), float(prediction_array.max())
        )
        padding = max(0.05 * (upper - lower), 0.05)
        limits = (lower - padding, upper + padding)
        axis.plot(limits, limits, color="black", linestyle="--", linewidth=1)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
    else:
        axis.text(0.5, 0.5, "No samples scored by UMA", ha="center", va="center")
    axis.axvline(cutoff, color="tab:red", linestyle=":", linewidth=1)
    axis.axhline(cutoff, color="tab:red", linestyle=":", linewidth=1)
    axis.set_xlabel("Conditioned mean force norm (eV/Å)")
    axis.set_ylabel("UMA mean force norm (eV/Å)")
    axis.set_title("Force-conditioning self-consistency")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary_path, dpi=200)
    plt.close(figure)
    temporary_path.replace(path)


def _write_json(path: Path, value: Any) -> None:
    """Atomically write JSON after replacing non-finite scalars with null."""
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w") as handle:
        json.dump(_sanitize_json(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(path)


def _sanitize_json(value: Any) -> Any:
    """Convert scientific scalars and non-finite values to strict JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    return value


def _convert_samples(samples: Sequence[GeneratedSample], domain: str) -> list[Any]:
    """Convert output atom arrays to the native zevals domain representation."""
    if domain == "omol25":
        from nucleus.metrics.molecule_generation import _atom_array_to_mol

        return [_atom_array_to_mol(sample.atom_array) for sample in samples]
    if domain == "omat24":
        from nucleus.utils.atom_array import _is_valid_afd_periodic_cell
        from nucleus.utils.material_fingerprints import atom_array_to_structure

        converted = []
        for sample in samples:
            structure = atom_array_to_structure(sample.atom_array)
            # Apply the existing AFD geometry guard before StructureMatcher or
            # UMA can enumerate millions of images of a nearly coplanar cell.
            # Keep a placeholder so failed samples retain their target/index
            # and remain in match-rate and UMA scored-rate denominators.
            if structure is not None and not _is_valid_afd_periodic_cell(
                structure.lattice.matrix
            ):
                LOGGER.warning(
                    "Sample %s has an unsafe periodic cell (volume %.6g A^3); "
                    "counting it as invalid for structure matching and "
                    "unscorable for UMA. The saved structure is retained.",
                    sample.sample_id,
                    structure.volume,
                )
                structure = None
            converted.append(structure)
        return converted
    raise ValueError(f"Unsupported evaluation domain: {domain!r}")


def _compute_generation_metrics(
    samples: Sequence[GeneratedSample], cfg: DictConfig
) -> tuple[dict[str, float], list[Any]]:
    """Compute validity, uniqueness, diversity, novelty, and PoseBusters."""
    domain = str(cfg.domain)
    converted = _convert_samples(samples, domain)
    if domain == "omol25":
        from nucleus.metrics.molecule_generation import (
            compute_molecule_generation_metrics,
        )

        metrics = compute_molecule_generation_metrics(
            [sample.atom_array for sample in samples],
            reference_fingerprint_path=cfg.reference_fingerprint_path,
            novelty_n_threads=int(cfg.novelty_n_threads),
            use_ground_truth_as_reference=False,
            compute_posebusters=True,
            compute_uff_force_norm=False,
            compute_uma_forces=False,
        )
        return metrics, converted

    from nucleus.metrics.material_generation import compute_material_generation_metrics
    from nucleus.utils.material_fingerprints import (
        load_reference_fingerprints_from_path,
    )

    reference_fingerprints = load_reference_fingerprints_from_path(
        cfg.reference_fingerprints_path
    )
    metrics = compute_material_generation_metrics(
        [sample.atom_array for sample in samples],
        reference_fingerprints=reference_fingerprints,
        use_ground_truth_as_reference=False,
        novelty_use_fingerprint=True,
        compute_uma_forces=False,
    )
    return metrics, converted


def _compute_afd(
    samples: Sequence[GeneratedSample], cfg: DictConfig
) -> dict[str, float]:
    """Compute molecule or periodic-material AFD for all convertible outputs."""
    from nucleus.utils.atom_array import (
        _atom_array_to_afd_record,
        _atom_array_to_material_afd_record,
    )

    domain = str(cfg.domain)
    converter = (
        _atom_array_to_afd_record
        if domain == "omol25"
        else _atom_array_to_material_afd_record
    )
    records = [converter(sample.atom_array) for sample in samples]
    records = [record for record in records if record is not None]
    reference: str | Sequence[Any] = str(cfg.afd.reference)
    if cfg.afd.reference_records_path is not None:
        from zevals.atomistic.afd.data import resolve_records

        reference = resolve_records(str(cfg.afd.reference_records_path))
    if domain == "omol25":
        from zevals.molecules.afd import compute_afd_metrics
    else:
        from zevals.materials.afd import compute_afd_metrics

    metrics = compute_afd_metrics(
        records,
        reference=reference,
        data_modality="molecules" if domain == "omol25" else "materials",
        model_name=cfg.afd.model_name,
        batch_size=int(cfg.afd.batch_size),
        device=cfg.afd.device,
        feature_cache_dir=cfg.afd.feature_cache_dir,
        ref_sample_size=cfg.afd.ref_sample_size,
        match_reference_size=bool(cfg.afd.match_reference_size),
        seed=int(cfg.afd.seed),
        show_progress=bool(cfg.afd.show_progress),
    )
    metrics["afd_input_success_rate"] = len(records) / len(samples) if samples else 0.0
    return metrics


def _conditioned_graph_features_from_reference(
    reference_atom_array: Any,
    *,
    target_id: str,
    partial_t: float | None = None,
) -> dict[str, np.ndarray]:
    """Recreate structure prediction's graph using its coordinate initialization.

    Standard inference zeroes all unfixed coordinates before computing token
    bonds, retaining the full sidecar graph. A non-None ``partial_t`` (including
    zero) instead preserves native coordinates, so its graph uses the original
    AF3 distance cutoff. Both conformers must share the resulting graph for RMSD.
    """
    from nucleus.metrics.structure_prediction import _validate_reference_coordinates

    _validate_reference_coordinates(reference_atom_array.coord, target_id=target_id)
    num_atoms = reference_atom_array.array_length()
    if reference_atom_array.bonds is None:
        raise ValueError(f"{target_id}: conditioned reference has no bond graph.")
    if partial_t is not None:
        from atomworks.ml.transforms.bonds import get_af3_token_bond_features

        atomized_reference = reference_atom_array.copy()
        atomized_reference.set_annotation("atomize", np.ones(num_atoms, dtype=bool))
        atomized_reference.set_annotation("is_polymer", np.zeros(num_atoms, dtype=bool))
        token_bonds = get_af3_token_bond_features(atomized_reference)
    else:
        token_bonds = reference_atom_array.bonds.adjacency_matrix().astype(bool)
        np.fill_diagonal(token_bonds, False)
    if token_bonds.shape != (num_atoms, num_atoms):
        raise ValueError(
            f"{target_id}: conditioned molecular graph has shape "
            f"{token_bonds.shape}, expected {(num_atoms, num_atoms)}."
        )
    return {
        "token_bonds": token_bonds,
        # OMol molecules are fully atomized, so every atom is one token.
        "atom_to_token_map": np.arange(num_atoms, dtype=np.int64),
    }


def _molecule_from_shared_conditioned_graph(
    atom_array: Any,
    features: Mapping[str, Any],
    *,
    reference_atomic_numbers: np.ndarray,
    target_id: str,
    role: str,
) -> Any:
    """Build one conformer with validation's graph and the supplied coordinates."""
    from nucleus.metrics.structure_prediction import _molecule_from_conditioned_graph
    from nucleus.utils.atom_array import _atomic_numbers_from_atom_array

    atomic_numbers = _atomic_numbers_from_atom_array(atom_array, len(atom_array))
    if atomic_numbers is None or not np.array_equal(
        atomic_numbers, reference_atomic_numbers
    ):
        raise ValueError(
            f"{target_id}: {role} atom identities/order do not match the "
            "conditioned reference."
        )
    molecule = _molecule_from_conditioned_graph(atom_array, dict(features))
    if molecule is None:
        raise ValueError(
            f"{target_id}: could not build the {role} conformer from the "
            "conditioned molecular graph."
        )
    return molecule


def _load_conditioned_molecule_structure_prediction_samples(
    samples: Sequence[GeneratedSample],
    *,
    sidecar_loader: Any,
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    """Build references and candidates with validation's shared conditioned graph."""
    from nucleus.inference_input_parsing import inference_load_preserving_hydrogens
    from nucleus.utils.atom_array import _atomic_numbers_from_atom_array

    samples_by_target: dict[str, list[GeneratedSample]] = {}
    for sample in samples:
        samples_by_target.setdefault(sample.target_id, []).append(sample)

    generated: dict[str, list[Any]] = {}
    references: dict[str, list[Any]] = {}
    for target_id, target_samples in samples_by_target.items():
        specification = target_samples[0].specification
        input_path = _specification_value(specification, "input")
        if not input_path:
            raise ValueError(f"{target_id}: structure prediction requires input CIF.")
        source_example_id = _specification_value(specification, "example_id")
        if source_example_id in (None, ""):
            raise ValueError(
                f"{target_id}: bond-sidecar evaluation requires "
                "specification.example_id."
            )

        reference_atom_array = inference_load_preserving_hydrogens(input_path)[
            "atom_array"
        ]
        sidecar_loader.attach_bonds(
            reference_atom_array,
            example_id=str(source_example_id),
        )
        features = _conditioned_graph_features_from_reference(
            reference_atom_array,
            target_id=target_id,
            partial_t=_specification_value(specification, "partial_t"),
        )
        reference_atomic_numbers = _atomic_numbers_from_atom_array(
            reference_atom_array, len(reference_atom_array)
        )
        if reference_atomic_numbers is None:
            raise ValueError(
                f"{target_id}: could not determine reference atomic numbers."
            )

        reference = _molecule_from_shared_conditioned_graph(
            reference_atom_array,
            features,
            reference_atomic_numbers=reference_atomic_numbers,
            target_id=target_id,
            role="reference",
        )
        candidates = [
            _molecule_from_shared_conditioned_graph(
                sample.atom_array,
                features,
                reference_atomic_numbers=reference_atomic_numbers,
                target_id=target_id,
                role=f"candidate-{candidate_index}",
            )
            for candidate_index, sample in enumerate(target_samples)
        ]
        references[target_id] = [reference]
        generated[target_id] = candidates
    return generated, references


def _load_reference_samples(
    samples: Sequence[GeneratedSample],
    domain: str,
) -> dict[str, list[Any]]:
    """Load each structure-prediction target from its absolute input CIF."""
    from nucleus.inference_input_parsing import inference_load_preserving_hydrogens

    specifications = {sample.target_id: sample.specification for sample in samples}
    references: dict[str, list[Any]] = {}
    for target_id, specification in specifications.items():
        input_path = _specification_value(specification, "input")
        if not input_path:
            raise ValueError(f"{target_id}: structure prediction requires input CIF.")
        atom_array = inference_load_preserving_hydrogens(input_path)["atom_array"]
        if domain == "omol25":
            from nucleus.metrics.molecule_generation import _atom_array_to_mol

            reference = _atom_array_to_mol(atom_array)
        else:
            from nucleus.utils.material_fingerprints import atom_array_to_structure

            reference = atom_array_to_structure(atom_array)
        if reference is None:
            raise ValueError(f"{target_id}: could not parse reference structure.")
        references[target_id] = [reference]
    return references


def _compute_structure_prediction_metrics(
    samples: Sequence[GeneratedSample], converted: Sequence[Any], cfg: DictConfig
) -> dict[str, float]:
    """Compute target-grouped best-of-k structure-prediction metrics."""
    domain = str(cfg.domain)
    if domain == "omol25" and cfg.get("bond_sidecar_path") is not None:
        from nucleus.utils.bond_sidecar import BondSidecarASEAtomsLoader

        sidecar_loader = BondSidecarASEAtomsLoader(
            bond_sidecar_path=str(cfg.bond_sidecar_path)
        )
        try:
            generated, references = (
                _load_conditioned_molecule_structure_prediction_samples(
                    samples,
                    sidecar_loader=sidecar_loader,
                )
            )
        finally:
            sidecar_loader.close()
    else:
        generated = {sample.target_id: [] for sample in samples}
        references = _load_reference_samples(samples, domain)
        for sample, candidate in zip(samples, converted, strict=True):
            generated[sample.target_id].append(candidate)

    if domain == "omol25":
        from zevals.molecules.structure_prediction import (
            compute_molecule_structure_prediction_metrics,
        )

        return compute_molecule_structure_prediction_metrics(
            generated, references, top_k=list(cfg.structure_prediction.top_k)
        )

    from zevals.materials.structure_prediction import (
        compute_material_structure_prediction_metrics,
    )

    return compute_material_structure_prediction_metrics(
        generated,
        references,
        ltol=float(cfg.structure_prediction.ltol),
        stol=float(cfg.structure_prediction.stol),
        angle_tol=float(cfg.structure_prediction.angle_tol),
        scale=bool(cfg.structure_prediction.scale),
        top_k=list(cfg.structure_prediction.top_k),
        show_progress=True,
    )


def _score_uma(
    samples: Sequence[GeneratedSample], converted: Sequence[Any], cfg: DictConfig
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Run UMA with each sample's charge/spin and retain aligned force scores."""
    from zevals.atomistic.uma.adapters import sample_to_atoms
    from zevals.atomistic.uma.metrics import score_uma_samples

    atoms_inputs: list[Any | None] = []
    for sample, converted_sample in zip(samples, converted, strict=True):
        atoms = sample_to_atoms(converted_sample)
        if atoms is not None and str(cfg.domain) == "omol25":
            charge = _specification_value(sample.specification, "total_charge")
            spin = _specification_value(sample.specification, "spin_multiplicity")
            if charge is not None:
                atoms.info["charge"] = int(charge)
            if spin is not None:
                atoms.info["spin"] = int(spin)
        atoms_inputs.append(atoms)

    # UMA obtains forces through autograd; inference itself runs under no-grad.
    with torch.enable_grad():
        scores = score_uma_samples(
            atoms_inputs,
            task="omol" if str(cfg.domain) == "omol25" else "omat",
            checkpoint=str(cfg.uma.checkpoint),
            device=cfg.uma.device,
            batch_size=int(cfg.uma.batch_size),
        )
    return _aggregate_uma_scores(scores, total_samples=len(samples)), scores


def _force_rows(
    samples: Sequence[GeneratedSample],
    scores: Sequence[Mapping[str, Any]],
    cutoff: float,
) -> list[dict[str, Any]]:
    """Build one CSV-compatible force row for every generated sample."""
    scores_by_index = {int(score["sample_index"]): score for score in scores}
    rows = []
    for sample in samples:
        target = _conditioning_force(sample.specification)
        score = scores_by_index.get(sample.sample_index)
        prediction = float(score["mean_force_norm"]) if score is not None else None
        rows.append(
            {
                "sample_index": sample.sample_index,
                "sample_id": sample.sample_id,
                "target_id": sample.target_id,
                "batch_index": sample.batch_index,
                "model_index": sample.model_index,
                "conditioning_mean_force_norm_eV_per_angstrom": target,
                "uma_mean_force_norm_eV_per_angstrom": prediction,
                "absolute_error_eV_per_angstrom": (
                    abs(prediction - target)
                    if target is not None and prediction is not None
                    else None
                ),
                "conditioning_force_at_or_above_cutoff": (
                    target >= cutoff if target is not None else None
                ),
                "uma_force_at_or_above_cutoff": (
                    prediction >= cutoff if prediction is not None else None
                ),
                "uma_force_max_eV_per_angstrom": (
                    float(score["force_max"]) if score is not None else None
                ),
                "uma_force_rms_eV_per_angstrom": (
                    float(score["force_rms"]) if score is not None else None
                ),
                "uma_energy_per_atom_eV": (
                    float(score["energy_per_atom"]) if score is not None else None
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write homogeneous records to CSV."""
    if not rows:
        raise ValueError("Cannot write a CSV without rows.")
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _dump_samples(
    samples: Sequence[GeneratedSample], output_dir: Path
) -> list[dict[str, Any]]:
    """Persist every generated CIF/metadata JSON and return a compact manifest."""
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for sample in samples:
        sample.output.dump(samples_dir, verbose=False)
        manifest.append(
            {
                "sample_index": sample.sample_index,
                "sample_id": sample.sample_id,
                "target_id": sample.target_id,
                "batch_index": sample.batch_index,
                "model_index": sample.model_index,
                "conditioning_mean_force_norm_eV_per_angstrom": _conditioning_force(
                    sample.specification
                ),
                "cif_path": str(Path("samples") / f"{sample.sample_id}.cif.gz"),
                "metadata_path": str(Path("samples") / f"{sample.sample_id}.json"),
            }
        )
    return manifest


def _load_saved_samples(
    output_dir: Path,
    *,
    expected_checkpoint: str | Path | None = None,
) -> list[GeneratedSample]:
    """Reconstruct generated samples from a prior evaluation's saved artifacts."""
    from nucleus.inference_input_parsing import inference_load_preserving_hydrogens

    manifest_path = output_dir / "sample_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Metrics-only evaluation requires an existing sample manifest: "
            f"{manifest_path}"
        )
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"Saved sample manifest is empty or invalid: {manifest_path}")

    expected_checkpoint_path = (
        Path(expected_checkpoint).expanduser().resolve()
        if expected_checkpoint is not None
        else None
    )
    samples: list[GeneratedSample] = []
    sample_ids: set[str] = set()
    for expected_index, row in enumerate(manifest):
        if not isinstance(row, Mapping):
            raise TypeError(
                f"Manifest row {expected_index} must be an object: {manifest_path}"
            )
        try:
            sample_index = int(row["sample_index"])
            sample_id = str(row["sample_id"])
            target_id = str(row["target_id"])
            batch_index = int(row["batch_index"])
            model_index = int(row["model_index"])
            cif_path = output_dir / str(row["cif_path"])
            metadata_path = output_dir / str(row["metadata_path"])
        except KeyError as error:
            raise ValueError(
                f"Manifest row {expected_index} is missing {error.args[0]!r}."
            ) from error

        if sample_index != expected_index:
            raise ValueError(
                "Saved sample manifest must contain contiguous, ordered sample_index "
                f"values; row {expected_index} has {sample_index}."
            )
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id in saved manifest: {sample_id}")
        sample_ids.add(sample_id)
        if not cif_path.is_file():
            raise FileNotFoundError(f"Saved sample CIF does not exist: {cif_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Saved sample metadata does not exist: {metadata_path}"
            )

        with metadata_path.open() as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise TypeError(f"Saved sample metadata must be an object: {metadata_path}")
        specification = metadata.get("specification")
        if not isinstance(specification, Mapping):
            raise ValueError(
                "Saved sample metadata has no conditioning specification: "
                f"{metadata_path}"
            )
        saved_checkpoint = metadata.get("ckpt_path")
        if expected_checkpoint_path is not None and saved_checkpoint not in (None, ""):
            saved_checkpoint_path = Path(str(saved_checkpoint)).expanduser().resolve()
            if saved_checkpoint_path != expected_checkpoint_path:
                raise ValueError(
                    f"{sample_id} was generated from {saved_checkpoint_path}, not "
                    f"the requested checkpoint {expected_checkpoint_path}."
                )

        loaded = inference_load_preserving_hydrogens(cif_path)
        if "atom_array" not in loaded:
            raise ValueError(f"Could not load atom_array from saved CIF: {cif_path}")
        output = _SavedSampleOutput(
            example_id=sample_id,
            atom_array=loaded["atom_array"],
            metadata=metadata,
        )
        samples.append(
            GeneratedSample(
                sample_index=sample_index,
                sample_id=sample_id,
                target_id=target_id,
                batch_index=batch_index,
                model_index=model_index,
                specification=dict(specification),
                output=output,
            )
        )
    return samples


def _write_output_shard(
    outputs: Mapping[str, Sequence[Any]], shard_dir: Path, rank: int
) -> None:
    """Atomically persist one rank's inference outputs on shared storage."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"rank-{rank:05d}.pkl"
    temporary_path = shard_path.with_suffix(".pkl.tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(dict(outputs), handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, shard_path)


def _merge_output_shards(shard_dir: Path, world_size: int) -> dict[str, Sequence[Any]]:
    """Load every rank shard, dropping DistributedSampler padding duplicates."""
    merged: dict[str, Sequence[Any]] = {}
    duplicate_groups = 0
    for rank in range(world_size):
        shard_path = shard_dir / f"rank-{rank:05d}.pkl"
        if not shard_path.is_file():
            raise FileNotFoundError(
                f"Missing generation output shard for rank {rank}: {shard_path}"
            )
        with shard_path.open("rb") as handle:
            rank_outputs = pickle.load(handle)
        if not isinstance(rank_outputs, Mapping):
            raise TypeError(f"Invalid generation output shard: {shard_path}")
        for group_name, group_outputs in rank_outputs.items():
            if group_name in merged:
                # torch DistributedSampler pads a non-divisible dataset by
                # repeating entries. Keep the copy produced by the lowest rank.
                duplicate_groups += 1
                continue
            merged[str(group_name)] = group_outputs
    if duplicate_groups:
        LOGGER.info(
            "Discarded %d padded duplicate output groups after distributed inference.",
            duplicate_groups,
        )
    return merged


def run_generation_evaluation(cfg: DictConfig) -> dict[str, Any]:
    """Generate or reload samples, then compute the configured metrics."""
    # Inference quiet mode raises the root level; keep evaluation stage
    # messages visible, including when running directly from saved samples.
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    LOGGER.setLevel(logging.INFO)
    output_dir = Path(cfg.out_dir).expanduser().resolve()
    samples_dir = output_dir / "samples"
    reuse_saved_samples = bool(cfg.evaluation.get("reuse_saved_samples", False))
    n_batches = int(cfg.n_batches)
    expected = int(cfg.evaluation.expected_num_samples)
    shard_dir: Path | None = None
    specifications: dict[str, dict[str, Any]] | None = None

    # Resolve the effective count before either generation or metrics-only
    # reuse. This lets a smaller input JSON cap the configured default while
    # preserving the exact-count behavior when enough inputs are available.
    if cfg.get("inputs") is not None:
        all_specifications = _load_input_specifications(cfg.inputs)
        specifications = _select_input_specifications(
            all_specifications,
            expected_num_samples=expected,
            n_batches=n_batches,
            diffusion_batch_size=int(cfg.diffusion_batch_size),
            json_keys_subset=cfg.get("json_keys_subset"),
        )
        effective_expected = (
            len(specifications) * n_batches * int(cfg.diffusion_batch_size)
        )
        if effective_expected != expected:
            expected = effective_expected
            cfg.evaluation.expected_num_samples = expected

    if reuse_saved_samples:
        if not output_dir.is_dir():
            raise FileNotFoundError(
                f"Metrics-only evaluation output directory does not exist: {output_dir}"
            )
        samples = _load_saved_samples(
            output_dir,
            expected_checkpoint=cfg.ckpt_path,
        )
        if len(samples) != expected:
            raise ValueError(
                f"Expected {expected} saved samples, found {len(samples)} in "
                f"{output_dir / 'sample_manifest.json'}."
            )
        candidates_per_target = n_batches * int(cfg.diffusion_batch_size)
        target_counts: dict[str, int] = {}
        for sample in samples:
            target_counts[sample.target_id] = target_counts.get(sample.target_id, 0) + 1
        invalid_counts = {
            target_id: count
            for target_id, count in target_counts.items()
            if count != candidates_per_target
        }
        if invalid_counts:
            raise ValueError(
                "Saved candidates per target do not match the current evaluation "
                f"configuration ({candidates_per_target}): {invalid_counts}"
            )
        num_targets = len(target_counts)
        OmegaConf.save(
            cfg,
            output_dir / "metrics_resolved_config.yaml",
            resolve=True,
        )
        LOGGER.info(
            "Reusing %d saved samples from %s; checkpoint inference is skipped.",
            len(samples),
            samples_dir,
        )
    else:
        if (
            samples_dir.exists()
            and any(samples_dir.iterdir())
            and not bool(cfg.evaluation.overwrite)
        ):
            raise FileExistsError(
                f"Sample output directory is not empty: {samples_dir}. "
                "Use evaluation.reuse_saved_samples=true to compute metrics from "
                "the existing samples, evaluation.overwrite=true to regenerate, "
                "or choose a new out_dir."
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        if specifications is None:
            raise ValueError("inputs must be configured when generating samples.")
        # RFD3 performs the actual filtering while loading the JSON. Give it the
        # same ordered keys used by this evaluator before the engine is built.
        cfg.json_keys_subset = list(specifications)

        from zatom2.run_inference import build_inference_engine

        engine = build_inference_engine(cfg)
        local_outputs = engine.run(inputs=cfg.inputs, n_batches=n_batches, out_dir=None)
        trainer = engine.trainer
        rank = int(trainer.fabric.global_rank)
        world_size = int(trainer.fabric.world_size)

        if world_size > 1:
            shard_dir = output_dir / ".generation_output_shards"
            _write_output_shard(local_outputs or {}, shard_dir, rank)
            trainer.fabric.barrier()
            if rank != 0:
                # Rank zero performs the non-distributed chemistry/oracle metrics.
                return {}
            outputs = _merge_output_shards(shard_dir, world_size)
        else:
            outputs = local_outputs

        OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)
        if not outputs:
            raise RuntimeError("Inference produced no outputs.")
        samples = _flatten_outputs(
            outputs,
            specifications,
            input_path=cfg.inputs,
            global_prefix=cfg.global_prefix,
            n_batches=n_batches,
        )
        if len(samples) != expected:
            raise ValueError(
                f"Expected {expected} generated samples, got {len(samples)}."
            )

        sample_manifest = _dump_samples(samples, output_dir)
        _write_json(output_dir / "sample_manifest.json", sample_manifest)
        LOGGER.info("Saved %d generated samples to %s", len(samples), samples_dir)
        num_targets = len(specifications)
        candidates_per_target = n_batches * int(cfg.diffusion_batch_size)

    LOGGER.info("Computing generation metrics for %d samples.", len(samples))
    generation_metrics, converted = _compute_generation_metrics(samples, cfg.evaluation)
    metrics: dict[str, Any] = {
        "evaluation": {
            "name": str(cfg.evaluation.name),
            "domain": str(cfg.evaluation.domain),
            "task": str(cfg.evaluation.task),
            "num_samples": len(samples),
            "num_targets": num_targets,
            "candidates_per_target": candidates_per_target,
            "reused_saved_samples": reuse_saved_samples,
        },
        "generation": generation_metrics,
        "applicability": {
            "posebusters": (
                "computed"
                if str(cfg.evaluation.domain) == "omol25"
                else "not applicable to periodic materials"
            ),
        },
    }
    if bool(cfg.evaluation.afd.enabled):
        LOGGER.info("Computing AFD.")
        metrics["afd"] = _compute_afd(samples, cfg.evaluation)
    if bool(cfg.evaluation.structure_prediction.enabled):
        LOGGER.info("Loading references and computing structure-prediction metrics.")
        metrics["structure_prediction"] = _compute_structure_prediction_metrics(
            samples, converted, cfg.evaluation
        )

    if bool(cfg.evaluation.uma.enabled):
        LOGGER.info("Computing UMA force metrics.")
        targets = [_conditioning_force(sample.specification) for sample in samples]
        if bool(cfg.evaluation.force_consistency.require_targets) and any(
            target is None for target in targets
        ):
            missing_count = sum(target is None for target in targets)
            raise ValueError(
                f"{missing_count} generated samples have no enabled mean_force_norm "
                "conditioning target in the inference JSON."
            )
        uma_metrics, uma_scores = _score_uma(samples, converted, cfg.evaluation)
        auroc_cutoff_override = cfg.evaluation.force_consistency.get(
            "auroc_cutoff_override"
        )
        if auroc_cutoff_override is None:
            cutoff = float(cfg.evaluation.force_consistency.cutoff)
            auroc_cutoffs = list(
                cfg.evaluation.force_consistency.get(
                    "auroc_cutoffs", DEFAULT_FORCE_AUROC_CUTOFFS
                )
            )
        else:
            cutoff = float(auroc_cutoff_override)
            auroc_cutoffs = None
        force_rows = _force_rows(samples, uma_scores, cutoff)
        paired_rows = [
            row
            for row in force_rows
            if row["conditioning_mean_force_norm_eV_per_angstrom"] is not None
            and row["uma_mean_force_norm_eV_per_angstrom"] is not None
        ]
        paired_targets = [
            row["conditioning_mean_force_norm_eV_per_angstrom"] for row in paired_rows
        ]
        paired_predictions = [
            row["uma_mean_force_norm_eV_per_angstrom"] for row in paired_rows
        ]
        metrics["uma"] = uma_metrics
        metrics["force_consistency"] = compute_force_consistency_metrics(
            paired_targets,
            paired_predictions,
            cutoff=cutoff,
            auroc_cutoffs=auroc_cutoffs,
        )
        _write_csv(output_dir / "uma_force_consistency.csv", force_rows)
        _write_scatter_plot(
            output_dir / "uma_force_consistency.png",
            paired_targets,
            paired_predictions,
            cutoff=cutoff,
        )

    _write_json(output_dir / "metrics.json", metrics)
    if shard_dir is not None:
        shutil.rmtree(shard_dir, ignore_errors=True)
    LOGGER.info("Evaluation metrics written to %s", output_dir / "metrics.json")
    return metrics


__all__ = [
    "GeneratedSample",
    "compute_force_consistency_metrics",
    "run_generation_evaluation",
]
