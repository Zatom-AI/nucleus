"""Distributed, restartable MLIP checkpoint evaluation for Zatom-2."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from nucleus.utils.batching import SparseCollate
from nucleus.utils.evaluation_manifest import build_mlip_prediction_identity
from nucleus.utils.inference import (
    make_rank_local_inference_model as _make_rank_local_inference_model,
)
from nucleus.utils.torch import as_cpu_float64_tensor
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Sampler

from zatom2.run_inference import build_inference_engine

LOGGER = logging.getLogger(__name__)

_DOMAINS = ("omol25", "omat24")
_OMOL25_PAPER_GROUPS = {
    "biomolecules": {"biomolecules"},
    "electrolytes": {"elytes"},
    "metal_complexes": {"metal_complexes"},
    "neutral_organics": {"ani2x", "orbnet_denali", "geom_orca6", "trans1x", "rgd"},
}


class _PredictionEvaluationError(ValueError):
    """Raised when a rank cannot safely serialize an MLIP prediction."""


class _NonFinitePredictionError(_PredictionEvaluationError):
    """Raised when an evaluable MLIP prediction is not finite."""


_PREDICTION_SCHEMA = pa.schema(
    [
        ("global_index", pa.int64()),
        ("example_id", pa.string()),
        ("data_id", pa.string()),
        ("n_atoms", pa.int32()),
        ("energy_eV", pa.float64()),
        ("forces_eV_per_angstrom", pa.list_(pa.list_(pa.float32(), 3))),
        ("has_reference", pa.bool_()),
        ("energy_absolute_error_eV", pa.float64()),
        ("energy_squared_error_eV2", pa.float64()),
        ("energy_per_atom_absolute_error_eV", pa.float64()),
        ("force_component_absolute_error_sum", pa.float64()),
        ("force_component_squared_error_sum", pa.float64()),
        ("force_vector_error_sum", pa.float64()),
        ("force_vector_squared_error_sum", pa.float64()),
        ("force_norm_absolute_error_sum", pa.float64()),
        ("force_cosine_similarity_sum", pa.float64()),
        ("force_cosine_similarity_count", pa.int64()),
        ("num_force_components", pa.int64()),
        ("num_force_atoms", pa.int64()),
    ]
)


@dataclass
class MetricSums:
    """Additive sufficient statistics for exact distributed aggregation."""

    num_examples: int = 0
    num_energy_systems: int = 0
    energy_absolute_error_sum_eV: float = 0.0
    energy_squared_error_sum_eV2: float = 0.0
    energy_per_atom_absolute_error_sum_eV: float = 0.0
    force_component_absolute_error_sum: float = 0.0
    force_component_squared_error_sum: float = 0.0
    force_vector_error_sum: float = 0.0
    force_vector_squared_error_sum: float = 0.0
    force_norm_absolute_error_sum: float = 0.0
    force_cosine_similarity_sum: float = 0.0
    force_cosine_similarity_count: int = 0
    num_force_components: int = 0
    num_force_atoms: int = 0

    def update(self, row: dict[str, Any]) -> None:
        """Add one serialized prediction row."""
        self.num_examples += 1
        if row["energy_absolute_error_eV"] is not None:
            self.num_energy_systems += 1
            self.energy_absolute_error_sum_eV += row["energy_absolute_error_eV"]
            self.energy_squared_error_sum_eV2 += row["energy_squared_error_eV2"]
            self.energy_per_atom_absolute_error_sum_eV += row[
                "energy_per_atom_absolute_error_eV"
            ]
        if row["force_component_absolute_error_sum"] is not None:
            for key in (
                "force_component_absolute_error_sum",
                "force_component_squared_error_sum",
                "force_vector_error_sum",
                "force_vector_squared_error_sum",
                "force_norm_absolute_error_sum",
                "force_cosine_similarity_sum",
            ):
                setattr(self, key, getattr(self, key) + row[key])
            self.force_cosine_similarity_count += row["force_cosine_similarity_count"]
            self.num_force_components += row["num_force_components"]
            self.num_force_atoms += row["num_force_atoms"]

    def merge(self, other: MetricSums) -> None:
        """Merge another accumulator in place."""
        for key, value in asdict(other).items():
            setattr(self, key, getattr(self, key) + value)

    def finalize(self) -> dict[str, int | float | None]:
        """Return publication metrics in both physical and paper units."""
        energy_count = self.num_energy_systems
        component_count = self.num_force_components
        atom_count = self.num_force_atoms

        def divide(value: float, count: int) -> float | None:
            return value / count if count else None

        energy_mae = divide(self.energy_absolute_error_sum_eV, energy_count)
        energy_rmse = (
            math.sqrt(self.energy_squared_error_sum_eV2 / energy_count)
            if energy_count
            else None
        )
        energy_per_atom_mae = divide(
            self.energy_per_atom_absolute_error_sum_eV, energy_count
        )
        force_component_mae = divide(
            self.force_component_absolute_error_sum, component_count
        )
        force_component_rmse = (
            math.sqrt(self.force_component_squared_error_sum / component_count)
            if component_count
            else None
        )
        force_vector_mae = divide(self.force_vector_error_sum, atom_count)
        force_vector_rmse = (
            math.sqrt(self.force_vector_squared_error_sum / atom_count)
            if atom_count
            else None
        )
        force_norm_mae = divide(self.force_norm_absolute_error_sum, atom_count)
        force_cosine = divide(
            self.force_cosine_similarity_sum,
            self.force_cosine_similarity_count,
        )
        return {
            "num_examples": self.num_examples,
            "num_energy_systems": energy_count,
            "num_force_atoms": atom_count,
            "energy_mae_eV": energy_mae,
            "energy_rmse_eV": energy_rmse,
            "energy_per_atom_mae_eV": energy_per_atom_mae,
            "force_component_mae_eV_per_angstrom": force_component_mae,
            "force_component_rmse_eV_per_angstrom": force_component_rmse,
            "force_vector_mae_eV_per_angstrom": force_vector_mae,
            "force_vector_rmse_eV_per_angstrom": force_vector_rmse,
            "force_norm_mae_eV_per_angstrom": force_norm_mae,
            "force_cosine_similarity": force_cosine,
            "energy_mae_meV": _to_mev(energy_mae),
            "energy_per_atom_mae_meV": _to_mev(energy_per_atom_mae),
            "force_mae_meV_per_angstrom": _to_mev(force_component_mae),
        }


def _to_mev(value: float | None) -> float | None:
    return None if value is None else 1000.0 * value


def resolve_aselmdb_paths(path: str | os.PathLike) -> list[Path]:
    """Resolve non-hidden ASE-LMDB shards deterministically."""
    root = Path(path).expanduser().resolve()
    paths = [root] if root.is_file() else list(root.rglob("*.aselmdb"))
    paths = sorted(path for path in paths if not path.name.startswith("._"))
    if not paths:
        raise FileNotFoundError(f"No *.aselmdb shards found under {root}.")
    return paths


class _IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        example = self.dataset[index]
        example["_evaluation_global_index"] = index
        return example


class _MissingPartSampler(Sampler[int]):
    def __init__(
        self,
        *,
        dataset_size: int,
        rank: int,
        world_size: int,
        part_size: int,
        output_dir: Path,
        resume: bool,
    ):
        self.dataset_size = dataset_size
        self.rank = rank
        self.world_size = world_size
        self.part_size = part_size
        self.output_dir = output_dir
        self.resume = resume

    def __iter__(self) -> Iterator[int]:
        for local_index, global_index in enumerate(
            range(self.rank, self.dataset_size, self.world_size)
        ):
            part_index = local_index // self.part_size
            if (
                self.resume
                and _part_path(self.output_dir, self.rank, part_index).exists()
            ):
                continue
            yield global_index

    def __len__(self) -> int:
        local_size = max(0, (self.dataset_size - 1 - self.rank) // self.world_size + 1)
        if not self.resume:
            return local_size
        completed = 0
        for part_path in self.output_dir.glob(f"rank-{self.rank:05d}/part-*.parquet"):
            completed += pq.ParquetFile(part_path).metadata.num_rows
        return max(0, local_size - completed)


def _identity_collate(examples: list[dict]) -> list[dict]:
    return examples


def _part_path(output_dir: Path, rank: int, part_index: int) -> Path:
    return output_dir / f"rank-{rank:05d}" / f"part-{part_index:06d}.parquet"


def _write_part(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    table = pa.Table.from_pylist(rows, schema=_PREDICTION_SCHEMA)
    pq.write_table(table, temporary_path, compression="zstd")
    temporary_path.replace(path)


def _build_dataset_config(
    train_cfg: DictConfig, domain: str, path: str, max_atoms: int
) -> DictConfig:
    """Build the exact deterministic preprocessing config used for MLIP eval."""
    dataset_cfg = OmegaConf.create(
        OmegaConf.to_container(
            train_cfg.datasets.train[domain].dataset,
            resolve=True,
        )
    )
    dataset_cfg.paths = [str(item) for item in resolve_aselmdb_paths(path)]
    dataset_cfg.name = f"{domain}-validation-mlip"
    loader_cfg = OmegaConf.select(dataset_cfg, "loader", default=None)
    if (
        domain == "omol25"
        and loader_cfg is not None
        and "bond_sidecar_path" in loader_cfg
    ):
        validation_sidecar_path = OmegaConf.select(
            train_cfg,
            ("datasets.val.omol25_structure_prediction.dataset." "bond_sidecar_path"),
            default=None,
        )
        if validation_sidecar_path in (None, ""):
            raise ValueError(
                "The OMol25 training dataset uses a bond sidecar, but the "
                "checkpoint config does not define the validation sidecar at "
                "datasets.val.omol25_structure_prediction.dataset."
                "bond_sidecar_path."
            )
        loader_cfg.bond_sidecar_path = str(validation_sidecar_path)
    transform = dataset_cfg.transform
    transform.is_inference = False
    transform.return_atom_array = True
    transform.diffusion_batch_size = 1
    transform.randomly_augment = False
    transform.remove_hydrogens = False
    transform.crop_size = max_atoms
    transform.max_atoms_in_crop = max_atoms
    if "randomly_translate_fractional_coordinates" in transform:
        transform.randomly_translate_fractional_coordinates = False
    transform.mean_force_norm_conditioning_probability = 0.0
    for condition in transform.train_conditions.values():
        condition.frequency = 0.0
    if "mlip" not in transform.train_conditions:
        raise ValueError("Checkpoint config has no MLIP conditioning task.")
    transform.train_conditions.mlip.frequency = 1.0
    for key in transform.meta_conditioning_probabilities:
        if isinstance(transform.meta_conditioning_probabilities[key], (int, float)):
            transform.meta_conditioning_probabilities[key] = 0.0
    return dataset_cfg


def _build_dataset(train_cfg: DictConfig, domain: str, path: str, max_atoms: int):
    return hydra.utils.instantiate(
        _build_dataset_config(train_cfg, domain, path, max_atoms)
    )


def _prediction_rows(
    *,
    examples: list[dict],
    global_indices: list[int],
    prediction: dict[str, torch.Tensor],
    collated: dict,
) -> list[dict[str, Any]]:
    energies = as_cpu_float64_tensor(prediction["energy_B"], detach=True).reshape(-1)
    energy_reconstruction_valid = (
        torch.as_tensor(
            prediction.get(
                "energy_reconstruction_valid_B",
                torch.ones_like(energies, dtype=torch.bool),
            )
        )
        .bool()
        .cpu()
        .reshape(-1)
    )
    forces = torch.as_tensor(prediction["forces_L"]).detach().float().cpu()
    batch_atom = torch.as_tensor(collated["feats"]["batch_atom_lvl"]).cpu()
    ground_truth = collated["ground_truth"]
    target_energy = as_cpu_float64_tensor(ground_truth["energy_gt_B"])
    energy_valid = torch.as_tensor(ground_truth["energy_valid_mask_B"]).bool().cpu()
    target_forces = torch.as_tensor(ground_truth["force_gt_D_L"])[0].float().cpu()
    force_valid = torch.as_tensor(ground_truth["force_valid_mask_L"]).bool().cpu()
    source_atom_counts = torch.as_tensor(ground_truth["source_atom_count_B"]).cpu()
    if energies.numel() != len(examples):
        raise _PredictionEvaluationError(
            f"Expected {len(examples)} energy predictions, got {energies.numel()}."
        )
    if energy_reconstruction_valid.numel() != len(examples):
        raise _PredictionEvaluationError(
            "Expected one energy reconstruction-valid value per example, got "
            f"{energy_reconstruction_valid.numel()} for {len(examples)} examples."
        )
    expected_force_shape = (batch_atom.numel(), 3)
    if tuple(forces.shape) != expected_force_shape:
        raise _PredictionEvaluationError(
            "Expected one force vector per collated atom; got prediction shape "
            f"{tuple(forces.shape)} for {batch_atom.numel()} atoms."
        )

    rows = []
    for batch_index, (example, global_index) in enumerate(
        zip(examples, global_indices, strict=True)
    ):
        atom_mask = batch_atom == batch_index
        predicted_force = forces[atom_mask]
        n_atoms = int(source_atom_counts[batch_index])
        if predicted_force.shape != (n_atoms, 3):
            raise _PredictionEvaluationError(
                "MLIP validation must preserve every source atom; got prediction "
                f"shape {tuple(predicted_force.shape)} for {n_atoms} source atoms."
            )
        has_energy_reference = bool(energy_valid[batch_index]) and bool(
            energy_reconstruction_valid[batch_index]
        )
        if has_energy_reference and not bool(torch.isfinite(energies[batch_index])):
            raise _NonFinitePredictionError(
                "MLIP evaluation produced a non-finite energy prediction for "
                f"{example['example_id']!r}."
            )
        has_force_reference = bool(torch.all(force_valid[atom_mask]))
        if has_force_reference and not bool(torch.isfinite(predicted_force).all()):
            raise _NonFinitePredictionError(
                "MLIP evaluation produced a non-finite force prediction for "
                f"{example['example_id']!r}."
            )
        row: dict[str, Any] = {
            "global_index": global_index,
            "example_id": str(example["example_id"]),
            "data_id": str(
                example.get("extra_info", {}).get(
                    "data_id",
                    example.get("extra_info", {}).get("calc_id", "unknown"),
                )
            ),
            "n_atoms": n_atoms,
            "energy_eV": (
                float(energies[batch_index]) if has_energy_reference else None
            ),
            "forces_eV_per_angstrom": predicted_force.numpy().tolist(),
            "has_reference": has_energy_reference and has_force_reference,
        }
        row.update(_empty_error_fields())

        if has_energy_reference:
            energy_error = energies[batch_index] - target_energy[batch_index]
            row.update(
                {
                    "energy_absolute_error_eV": float(torch.abs(energy_error)),
                    "energy_squared_error_eV2": float(torch.square(energy_error)),
                    "energy_per_atom_absolute_error_eV": float(
                        torch.abs(energy_error) / max(n_atoms, 1)
                    ),
                }
            )
        elif bool(energy_valid[batch_index]):
            LOGGER.warning(
                "Masking physical-energy evaluation for %s because its composition "
                "has no complete domain-specific elemental reference.",
                example["example_id"],
            )

        if has_force_reference:
            force_error = predicted_force - target_forces[atom_mask]
            vector_error = torch.linalg.vector_norm(force_error, dim=-1)
            predicted_norm = torch.linalg.vector_norm(predicted_force, dim=-1)
            target_norm = torch.linalg.vector_norm(target_forces[atom_mask], dim=-1)
            cosine_denominator = predicted_norm * target_norm
            cosine_valid = cosine_denominator > 0
            cosine_sum = 0.0
            if torch.any(cosine_valid):
                cosine_sum = float(
                    (
                        (predicted_force * target_forces[atom_mask]).sum(dim=-1)[
                            cosine_valid
                        ]
                        / cosine_denominator[cosine_valid]
                    ).sum()
                )
            row.update(
                {
                    "force_component_absolute_error_sum": float(
                        torch.abs(force_error).sum()
                    ),
                    "force_component_squared_error_sum": float(
                        torch.square(force_error).sum()
                    ),
                    "force_vector_error_sum": float(vector_error.sum()),
                    "force_vector_squared_error_sum": float(
                        torch.square(vector_error).sum()
                    ),
                    "force_norm_absolute_error_sum": float(
                        torch.abs(predicted_norm - target_norm).sum()
                    ),
                    "force_cosine_similarity_sum": cosine_sum,
                    "force_cosine_similarity_count": int(cosine_valid.sum()),
                    "num_force_components": int(force_error.numel()),
                    "num_force_atoms": n_atoms,
                }
            )
        rows.append(row)
    return rows


def _empty_error_fields() -> dict[str, Any]:
    return {
        "energy_absolute_error_eV": None,
        "energy_squared_error_eV2": None,
        "energy_per_atom_absolute_error_eV": None,
        "force_component_absolute_error_sum": None,
        "force_component_squared_error_sum": None,
        "force_vector_error_sum": None,
        "force_vector_squared_error_sum": None,
        "force_norm_absolute_error_sum": None,
        "force_cosine_similarity_sum": None,
        "force_cosine_similarity_count": 0,
        "num_force_components": 0,
        "num_force_atoms": 0,
    }


def _synchronize_prediction_error(
    *,
    fabric,
    local_error: _PredictionEvaluationError | None,
    rank: int,
    world_size: int,
) -> None:
    """Propagate a rank-local prediction failure before leaving evaluation."""
    if world_size == 1:
        if local_error is not None:
            raise local_error
        return

    local_failed = torch.tensor(
        [local_error is not None], dtype=torch.int64, device=fabric.device
    )
    gathered_failed = torch.as_tensor(fabric.all_gather(local_failed)).detach().cpu()
    failed_ranks = [
        failed_rank
        for failed_rank, failed in enumerate(gathered_failed.reshape(-1).tolist())
        if failed
    ]
    failure_details = []
    for failed_rank in failed_ranks:
        local_detail = None
        if rank == failed_rank:
            assert local_error is not None
            local_detail = {
                "rank": rank,
                "type": type(local_error).__name__,
                "message": str(local_error),
            }
        failure_details.append(fabric.broadcast(local_detail, src=failed_rank))

    if local_error is not None:
        raise local_error
    if failure_details:
        details = "; ".join(
            f"rank {detail['rank']} ({detail['type']}): {detail['message']}"
            for detail in failure_details
        )
        raise RuntimeError(
            "MLIP evaluation failed on another distributed rank: " + details
        )


def _evaluate_domain(
    *,
    engine,
    train_cfg: DictConfig,
    domain: str,
    dataset_path: str,
    output_dir: Path,
    evaluation_cfg: DictConfig,
) -> int:
    trainer = engine.trainer
    rank = int(trainer.fabric.global_rank)
    world_size = int(trainer.fabric.world_size)
    dataset = _build_dataset(
        train_cfg,
        domain,
        dataset_path,
        int(evaluation_cfg.max_system_atoms),
    )
    dataset_size = len(dataset)
    if evaluation_cfg.max_examples_per_domain is not None:
        dataset_size = min(dataset_size, int(evaluation_cfg.max_examples_per_domain))
    prediction_dir = output_dir / "predictions" / domain
    sampler = _MissingPartSampler(
        dataset_size=dataset_size,
        rank=rank,
        world_size=world_size,
        part_size=int(evaluation_cfg.part_size),
        output_dir=prediction_dir,
        resume=bool(evaluation_cfg.resume),
    )
    loader_kwargs = {
        "dataset": _IndexedDataset(dataset),
        "sampler": sampler,
        "batch_size": int(evaluation_cfg.batch_size),
        "num_workers": int(evaluation_cfg.num_workers),
        "collate_fn": _identity_collate,
        "pin_memory": trainer.fabric.device.type == "cuda",
    }
    if int(evaluation_cfg.num_workers) > 0:
        loader_kwargs.update(
            prefetch_factor=int(evaluation_cfg.prefetch_factor),
            persistent_workers=True,
        )
    loader = DataLoader(**loader_kwargs)

    model = _make_rank_local_inference_model(
        trainer.state["model"],
        strategy=trainer.fabric.strategy,
        world_size=world_size,
    )
    current_part: int | None = None
    buffered_rows: list[dict[str, Any]] = []
    completed = 0
    started = time.monotonic()

    def flush() -> None:
        nonlocal buffered_rows
        if current_part is not None and buffered_rows:
            _write_part(_part_path(prediction_dir, rank, current_part), buffered_rows)
            buffered_rows = []

    local_num_batches = len(loader)
    if world_size == 1:
        max_num_batches = local_num_batches
    else:
        local_batch_count = torch.tensor(
            [local_num_batches], dtype=torch.int64, device=trainer.fabric.device
        )
        gathered_batch_counts = torch.as_tensor(
            trainer.fabric.all_gather(local_batch_count)
        )
        max_num_batches = int(gathered_batch_counts.max().item())

    poll_interval = int(evaluation_cfg.progress_every)
    if poll_interval <= 0:
        raise ValueError("evaluation.progress_every must be positive.")
    loader_iterator = iter(loader)
    local_prediction_error = None
    with torch.inference_mode():
        for chunk_start in range(0, max_num_batches, poll_interval):
            chunk_stop = min(chunk_start + poll_interval, local_num_batches)
            try:
                for batch_index in range(chunk_start, chunk_stop):
                    examples = next(loader_iterator)
                    global_indices = [
                        int(example.pop("_evaluation_global_index"))
                        for example in examples
                    ]
                    collated = SparseCollate()(examples)
                    collated = trainer.fabric.to_device(collated)
                    network_input = trainer._assemble_network_inputs(collated)
                    property_input = trainer._single_view_property_input(network_input)
                    prediction = model.forward(
                        input=property_input,
                        property_prediction_atom_types="known",
                        allow_missing_property_elemental_references=True,
                    )
                    rows = _prediction_rows(
                        examples=examples,
                        global_indices=global_indices,
                        prediction=prediction,
                        collated=collated,
                    )
                    for row in rows:
                        local_index = (row["global_index"] - rank) // world_size
                        row_part = local_index // int(evaluation_cfg.part_size)
                        if current_part is not None and row_part != current_part:
                            flush()
                        current_part = row_part
                        buffered_rows.append(row)
                        completed += 1
                    if batch_index % poll_interval == 0:
                        elapsed = max(time.monotonic() - started, 1e-9)
                        LOGGER.info(
                            "%s rank %d: %d new examples (%.2f examples/s)",
                            domain,
                            rank,
                            completed,
                            completed / elapsed,
                        )
            except _PredictionEvaluationError as error:
                local_prediction_error = error
                # A part file is the resume unit. Do not serialize the incomplete
                # part containing the failed batch; flushed parts remain reusable.
                buffered_rows.clear()

            # Every rank executes the same number of polling rounds, including
            # ranks that have failed or exhausted a shorter resume workload.
            _synchronize_prediction_error(
                fabric=trainer.fabric,
                local_error=local_prediction_error,
                rank=rank,
                world_size=world_size,
            )
    flush()
    trainer.fabric.barrier()
    return dataset_size


def aggregate_prediction_parts(
    prediction_dir: Path,
) -> dict[str, dict[str, int | float | None]]:
    """Aggregate all rank-local Parquet parts without loading force arrays."""
    columns = [
        field.name
        for field in _PREDICTION_SCHEMA
        if field.name != "forces_eV_per_angstrom"
    ]
    sums: dict[str, MetricSums] = defaultdict(MetricSums)
    part_paths = sorted(prediction_dir.glob("rank-*/part-*.parquet"))
    if not part_paths:
        raise RuntimeError(f"No prediction parts found in {prediction_dir}.")
    for part_path in part_paths:
        for batch in pq.ParquetFile(part_path).iter_batches(columns=columns):
            for row in batch.to_pylist():
                sums["all"].update(row)
                sums[row["data_id"]].update(row)
    result = {key: value.finalize() for key, value in sorted(sums.items())}
    if prediction_dir.name == "omol25":
        for group, data_ids in _OMOL25_PAPER_GROUPS.items():
            grouped = MetricSums()
            for data_id in data_ids:
                if data_id in sums:
                    grouped.merge(sums[data_id])
            if grouped.num_examples:
                result[group] = grouped.finalize()
    return result


def _reference_rows() -> list[dict[str, Any]]:
    rows = []

    def add(benchmark, subset, model, energy_per_atom, energy, force, source):
        rows.append(
            {
                "benchmark": benchmark,
                "subset": subset,
                "model": model,
                "energy_per_atom_mae_meV": energy_per_atom,
                "energy_mae_meV": energy,
                "force_mae_meV_per_angstrom": force,
                "source": source,
            }
        )

    zatom_source = "Zatom-1 arXiv:2602.22251v4 Table 13"
    add("omol25_4m_val", "all", "eSEN-md-d.", 1.32, 77.13, 6.78, zatom_source)
    add(
        "omol25_4m_val",
        "all",
        "Zatom-1 MLIP-pretrained*",
        7.13,
        514.0,
        86.99,
        zatom_source,
    )
    add(
        "omol25_4m_val",
        "all",
        "Zatom-1 generatively pretrained*",
        19.46,
        1307.23,
        72.40,
        zatom_source,
    )
    add(
        "omol25_4m_val",
        "all",
        "Zatom-1 jointly pretrained",
        34.45,
        2512.78,
        62.99,
        zatom_source,
    )

    uma_source = "UMA arXiv:2506.23971 Table 16"
    uma_values = {
        "UMA-S": [
            (0.53, 5.69),
            (2.69, 11.65),
            (4.63, 37.85),
            (1.00, 13.15),
            (3.62, 12.02),
        ],
        "UMA-M": [
            (0.44, 3.95),
            (2.28, 10.21),
            (3.60, 28.81),
            (0.68, 7.00),
            (3.21, 9.90),
        ],
        "UMA-L": [
            (0.33, 2.90),
            (1.13, 4.52),
            (3.35, 24.85),
            (0.65, 5.02),
            (2.39, 5.83),
        ],
    }
    subsets = [
        "biomolecules",
        "electrolytes",
        "metal_complexes",
        "neutral_organics",
        "ood_comp",
    ]
    for model, values in uma_values.items():
        for subset, (energy_per_atom, force) in zip(subsets, values, strict=True):
            add(
                "omol25_4m_val", subset, model, energy_per_atom, None, force, uma_source
            )

    allscaip_omol_source = "AllScAIP arXiv:2603.06567v1 Table 7 (OMol25 validation)"
    allscaip_omol_values = {
        "AllScAIP-sm-ft-cons. (OMol25-102M)": [
            (31.94, 3.34),
            (39.70, 6.06),
            (123.69, 25.94),
            (18.77, 8.97),
            (42.48, 6.41),
        ],
        "AllScAIP-md-d. (OMol25-102M)": [
            (22.25, 2.92),
            (33.68, 4.84),
            (105.59, 22.36),
            (18.57, 6.30),
            (35.18, 5.25),
        ],
        "AllScAIP-md-cons. (OMol25-102M)": [
            (35.86, 3.23),
            (37.59, 5.46),
            (107.92, 22.64),
            (15.74, 7.52),
            (40.63, 5.79),
        ],
        "AllScAIP-sm-ft-cons. (OMol25-4M)": [
            (43.82, 4.08),
            (62.09, 7.98),
            (130.94, 32.24),
            (29.35, 12.72),
            (59.93, 8.26),
        ],
        "AllScAIP-md-ft-cons. (OMol25-4M)": [
            (28.27, 3.62),
            (43.85, 7.21),
            (123.35, 30.68),
            (23.11, 10.54),
            (44.40, 7.51),
        ],
    }
    allscaip_omol_subsets = [
        "biomolecules",
        "electrolytes",
        "metal_complexes",
        "neutral_organics",
        "all",
    ]
    for model, values in allscaip_omol_values.items():
        for subset, (energy, force) in zip(allscaip_omol_subsets, values, strict=True):
            add(
                "omol25_4m_val",
                subset,
                model,
                None,
                energy,
                force,
                allscaip_omol_source,
            )

    omat_source = "UMA arXiv:2506.23971v2 Table 15 (OMat24 validation)"
    for model, energy_per_atom, force in (
        ("UMA-S", 11.3, 57.1),
        ("UMA-M", 10.0, 47.3),
        ("UMA-L", 9.7, 43.5),
    ):
        add("omat24_1m_val", "all", model, energy_per_atom, None, force, omat_source)

    allscaip_omat_source = "AllScAIP arXiv:2603.06567v1 Table 8 (OMat24 validation)"
    for model, energy_per_atom, force in (
        ("AllScAIP-sm-d.", 12.7, 60.1),
        ("AllScAIP-sm-ft-cons.", 12.4, 56.0),
        ("AllScAIP-md-d.", 11.4, 56.6),
        ("AllScAIP-md-ft-cons.", 10.7, 54.3),
    ):
        add(
            "omat24_1m_val",
            "all",
            model,
            energy_per_atom,
            None,
            force,
            allscaip_omat_source,
        )
    return rows


def _write_results(
    output_dir: Path,
    metrics: dict[str, dict[str, Any]],
    *,
    checkpoint_path: str,
) -> None:
    rows = []
    zatom2_model = f"Zatom-2 ({Path(checkpoint_path).stem})"
    for domain, domain_metrics in metrics.items():
        benchmark = f"{domain}_{'4m' if domain == 'omol25' else '1m'}_val"
        paper_subsets = (
            ("all", *_OMOL25_PAPER_GROUPS) if domain == "omol25" else ("all",)
        )
        for subset in paper_subsets:
            if subset not in domain_metrics:
                continue
            values = domain_metrics[subset]
            rows.append(
                {
                    "benchmark": benchmark,
                    "subset": subset,
                    "model": zatom2_model,
                    "energy_per_atom_mae_meV": values["energy_per_atom_mae_meV"],
                    "energy_mae_meV": values["energy_mae_meV"],
                    "force_mae_meV_per_angstrom": values["force_mae_meV_per_angstrom"],
                    "source": "this evaluation",
                }
            )
    rows.extend(_reference_rows())
    benchmark_order = {"omol25_4m_val": 0, "omat24_1m_val": 1}
    subset_order = {
        "all": 0,
        "biomolecules": 1,
        "electrolytes": 2,
        "metal_complexes": 3,
        "neutral_organics": 4,
        "ood_comp": 5,
    }
    rows.sort(
        key=lambda row: (
            benchmark_order[row["benchmark"]],
            subset_order[row["subset"]],
            0 if row["model"].startswith("Zatom-2") else 1,
            row["model"],
        )
    )
    fieldnames = list(rows[0])
    with (output_dir / "paper_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# MLIP validation results",
        "",
        "Energy and force values are MAE in meV (or meV/atom) and meV/Å. "
        "Force MAE is component-wise, matching FAIR Chemistry and Table 13 of "
        "the Zatom-1 paper.",
        "",
        "| Benchmark | Subset | Model | E/atom | E | Forces |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:

        def show(value):
            return "—" if value is None else f"{value:.3f}"

        lines.append(
            f"| {row['benchmark']} | {row['subset']} | {row['model']} | "
            f"{show(row['energy_per_atom_mae_meV'])} | "
            f"{show(row['energy_mae_meV'])} | "
            f"{show(row['force_mae_meV_per_angstrom'])} |"
        )
    lines.extend(
        [
            "",
            "Zatom-1 reports MPtrj—not OMat24—material MLIP validation. Those "
            "material numbers are intentionally not placed in the OMat24 comparison.",
            "UMA's OOD-composition result uses the separate OMol25 "
            "`metal_ligand_ood` split. It is retained as a published reference "
            "row, but this validation-only workflow does not report a Zatom-2 "
            "value for it.",
            "AllScAIP's OMol25 Table 7 rows use total-energy MAE because the "
            "paper flags its energy-per-atom metric as affected by a FAIR-Chem "
            "bug. Model suffixes identify whether training used OMol25-4M or "
            "OMol25-102M.",
            "",
            "References: [Zatom-1](https://arxiv.org/html/2602.22251v4), "
            "[UMA](https://arxiv.org/html/2506.23971v2), and "
            "[AllScAIP](https://arxiv.org/html/2603.06567v1).",
        ]
    )
    (output_dir / "paper_results.md").write_text("\n".join(lines) + "\n")


def _write_manifest(
    output_dir: Path,
    *,
    cfg: DictConfig,
    train_cfg: DictConfig,
    rank: int,
    world_size: int,
) -> None:
    if rank != 0:
        return
    checkpoint_path = Path(cfg.ckpt_path).expanduser().resolve()
    checkpoint_stat = checkpoint_path.stat()
    dataset_shards = {}
    for domain in cfg.evaluation.domains:
        paths = resolve_aselmdb_paths(cfg.evaluation.datasets[domain].path)
        dataset_shards[str(domain)] = [
            {
                "path": str(path),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in paths
        ]
    preprocessing_configs = {
        str(domain): _build_dataset_config(
            train_cfg,
            str(domain),
            str(cfg.evaluation.datasets[domain].path),
            int(cfg.evaluation.max_system_atoms),
        )
        for domain in cfg.evaluation.domains
    }
    manifest = {
        "manifest_version": 2,
        "checkpoint": {
            "path": str(checkpoint_path),
            "size": checkpoint_stat.st_size,
            "mtime_ns": checkpoint_stat.st_mtime_ns,
        },
        "world_size": world_size,
        "domains": list(cfg.evaluation.domains),
        "dataset_shards": dataset_shards,
        "result_identity": {
            "part_size": int(cfg.evaluation.part_size),
            "max_system_atoms": int(cfg.evaluation.max_system_atoms),
            "max_examples_per_domain": cfg.evaluation.max_examples_per_domain,
            "use_ema_weights": bool(cfg.get("use_ema_weights", True)),
        },
        "prediction_identity": build_mlip_prediction_identity(
            config=cfg,
            checkpoint_config=train_cfg,
            preprocessing_configs=preprocessing_configs,
            model_package="zatom2",
            prediction_schema=str(_PREDICTION_SCHEMA),
        ),
        "evaluation": OmegaConf.to_container(cfg.evaluation, resolve=True),
    }
    path = output_dir / "run_manifest.json"
    has_prediction_parts = (
        next(output_dir.glob("predictions/*/rank-*/part-*.parquet"), None) is not None
    )
    if path.exists() and bool(cfg.evaluation.resume):
        previous = json.loads(path.read_text())
        for key in (
            "manifest_version",
            "checkpoint",
            "world_size",
            "domains",
            "dataset_shards",
            "result_identity",
            "prediction_identity",
        ):
            if previous.get(key) != manifest[key]:
                if key == "prediction_identity":
                    raise ValueError(
                        "Cannot resume: prediction-affecting configuration, "
                        "preprocessing, code, runtime, or schema differs from "
                        "the existing manifest. Use a new out_dir."
                    )
                raise ValueError(
                    f"Cannot resume: manifest {key}={previous.get(key)!r}, "
                    f"current value={manifest[key]!r}."
                )
    else:
        if has_prediction_parts:
            raise FileExistsError(
                f"Prediction parts already exist under {output_dir}, but no "
                "compatible resume was requested. Use a new out_dir or set "
                "evaluation.resume=true."
            )
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def run_mlip_evaluation(cfg: DictConfig) -> None:
    """Run labeled validation inference across all configured distributed ranks."""
    output_dir = Path(cfg.out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    domains = [str(domain) for domain in cfg.evaluation.domains]
    unknown_domains = set(domains) - set(_DOMAINS)
    if unknown_domains:
        raise ValueError(f"Unknown MLIP evaluation domains: {sorted(unknown_domains)}")

    engine = build_inference_engine(cfg)
    train_cfg = engine.initialize()
    trainer = engine.trainer
    rank = int(trainer.fabric.global_rank)
    world_size = int(trainer.fabric.world_size)
    _write_manifest(
        output_dir,
        cfg=cfg,
        train_cfg=train_cfg,
        rank=rank,
        world_size=world_size,
    )
    trainer.fabric.barrier()

    metrics = {}
    for domain in domains:
        dataset_size = _evaluate_domain(
            engine=engine,
            train_cfg=train_cfg,
            domain=domain,
            dataset_path=str(cfg.evaluation.datasets[domain].path),
            output_dir=output_dir,
            evaluation_cfg=cfg.evaluation,
        )
        if rank == 0:
            domain_metrics = aggregate_prediction_parts(
                output_dir / "predictions" / domain
            )
            if domain_metrics["all"]["num_examples"] != dataset_size:
                raise RuntimeError(
                    f"{domain} wrote {domain_metrics['all']['num_examples']} of "
                    f"{dataset_size} expected predictions."
                )
            metrics[domain] = domain_metrics
            metrics_dir = output_dir / "metrics"
            metrics_dir.mkdir(exist_ok=True)
            (metrics_dir / f"{domain}.json").write_text(
                json.dumps(domain_metrics, indent=2, sort_keys=True) + "\n"
            )
        trainer.fabric.barrier()

    if rank == 0:
        _write_results(output_dir, metrics, checkpoint_path=str(cfg.ckpt_path))
        LOGGER.info("MLIP evaluation complete: %s", output_dir)


__all__ = [
    "MetricSums",
    "aggregate_prediction_parts",
    "resolve_aselmdb_paths",
    "run_mlip_evaluation",
]
