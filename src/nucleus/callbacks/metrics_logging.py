from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from foundry.callbacks.callback import BaseCallback
from foundry.callbacks.metrics_logging import (
    StoreValidationMetricsInDFCallback as FoundryStoreValidationMetricsInDFCallback,
)
from foundry.utils.ddp import RankedLogger
from foundry.utils.logging import print_df_as_table
from omegaconf import ListConfig

from nucleus.utils.atom_array import (
    _atom_array_to_afd_record,
    _atom_array_to_material_afd_record,
)

ranked_logger = RankedLogger(__name__, rank_zero_only=True)

AFD_METRIC_KEYS: tuple[str, ...] = (
    "afd_score",
    "afd_num_samples",
    "afd_ref_sample_size",
    "afd_input_success_rate",
)
ATOM_FRACTION_METRIC_KEYS: tuple[str, ...] = (
    "atom_fraction_o",
    "atom_fraction_n",
    "atom_fraction_c",
)
ATOM_FRACTION_ELEMENT_KEYS: dict[str, str] = {
    "O": "atom_fraction_o",
    "N": "atom_fraction_n",
    "C": "atom_fraction_c",
}
EPOCH_AFD_EXAMPLE_ID = "__epoch_molecule_afd__"
EPOCH_MATERIALS_AFD_EXAMPLE_ID = "__epoch_materials_afd__"
DEFAULT_MATERIALS_AFD_MODEL_NAME = "Ty-Perez/ct-scd-amp20"

_ATOMISTIC_PROPERTY_SUM_SUFFIXES: tuple[str, ...] = (
    "energy_absolute_error_sum_eV",
    "energy_per_atom_absolute_error_sum_eV",
    "force_component_absolute_error_sum",
    "force_component_squared_error_sum",
    "force_vector_error_sum",
    "force_vector_squared_error_sum",
    "force_norm_absolute_error_sum",
    "force_cosine_similarity_sum",
)
_ATOMISTIC_PROPERTY_COUNT_SUFFIXES: tuple[str, ...] = (
    "num_energy_systems",
    "num_force_components",
    "num_force_atoms",
    "force_cosine_similarity_count",
)


def _aggregate_atomistic_property_metric_columns(
    dataset_df: pd.DataFrame,
    default_means: pd.Series,
) -> pd.Series:
    """Replace per-example MLIP means with exact dataset-level reductions."""
    aggregated = default_means.copy()
    prefixes = {
        column.removesuffix(suffix)
        for column in dataset_df.columns
        for suffix in _ATOMISTIC_PROPERTY_SUM_SUFFIXES
        if column.endswith(suffix)
    }

    def _total(column: str) -> float | None:
        if column not in dataset_df:
            return None
        total = pd.to_numeric(dataset_df[column], errors="coerce").sum(min_count=1)
        return None if pd.isna(total) else float(total)

    def _paired_totals(
        prefix: str,
        *,
        count_suffix: str,
        sum_suffixes: tuple[str, ...],
    ) -> tuple[float | None, dict[str, float]]:
        """Reduce sufficient statistics without separating sums from counts."""
        count_column = f"{prefix}{count_suffix}"
        counts = (
            pd.to_numeric(dataset_df[count_column], errors="coerce")
            if count_column in dataset_df
            else pd.Series(np.nan, index=dataset_df.index, dtype=float)
        )
        sums = {
            suffix: (
                pd.to_numeric(dataset_df[f"{prefix}{suffix}"], errors="coerce")
                if f"{prefix}{suffix}" in dataset_df
                else pd.Series(np.nan, index=dataset_df.index, dtype=float)
            )
            for suffix in sum_suffixes
        }
        relevant = counts.notna()
        for values in sums.values():
            relevant |= values.notna()
        if not relevant.any():
            return None, {}

        invalid_count = relevant & (counts.isna() | ~np.isfinite(counts) | counts.lt(0))
        if invalid_count.any():
            raise ValueError(
                "Atomistic sufficient statistics require a finite, nonnegative "
                f"{count_column!r} wherever a paired sum is present."
            )

        positive_count = counts.gt(0)
        zero_count = counts.eq(0)
        totals = {}
        for suffix, values in sums.items():
            column = f"{prefix}{suffix}"
            invalid_positive = positive_count & (values.isna() | ~np.isfinite(values))
            invalid_zero = (
                zero_count & values.notna() & (~np.isfinite(values) | values.ne(0))
            )
            if invalid_positive.any() or invalid_zero.any():
                raise ValueError(
                    "Atomistic sufficient statistics require a finite paired "
                    f"{column!r} for every positive {count_column!r}, and zero "
                    "sum for a zero count."
                )
            totals[suffix] = float(values[positive_count].sum())
        return float(counts[relevant].sum()), totals

    for prefix in prefixes:
        totals = {
            suffix: _total(f"{prefix}{suffix}")
            for suffix in (
                *_ATOMISTIC_PROPERTY_SUM_SUFFIXES,
                *_ATOMISTIC_PROPERTY_COUNT_SUFFIXES,
            )
        }
        for suffix in _ATOMISTIC_PROPERTY_COUNT_SUFFIXES:
            if totals[suffix] is not None:
                aggregated[f"{prefix}{suffix}"] = totals[suffix]

        energy_count, energy_totals = _paired_totals(
            prefix,
            count_suffix="num_energy_systems",
            sum_suffixes=(
                "energy_absolute_error_sum_eV",
                "energy_per_atom_absolute_error_sum_eV",
            ),
        )
        if energy_count:
            aggregated[f"{prefix}energy_mae_eV"] = (
                energy_totals["energy_absolute_error_sum_eV"] / energy_count
            )
            aggregated[f"{prefix}energy_per_atom_mae_eV"] = (
                energy_totals["energy_per_atom_absolute_error_sum_eV"] / energy_count
            )

        component_count, component_totals = _paired_totals(
            prefix,
            count_suffix="num_force_components",
            sum_suffixes=(
                "force_component_absolute_error_sum",
                "force_component_squared_error_sum",
            ),
        )
        if component_count:
            aggregated[f"{prefix}force_component_mae_eV_per_angstrom"] = (
                component_totals["force_component_absolute_error_sum"] / component_count
            )
            aggregated[f"{prefix}force_component_rmse_eV_per_angstrom"] = np.sqrt(
                max(
                    component_totals["force_component_squared_error_sum"]
                    / component_count,
                    0,
                )
            )

        atom_count, atom_totals = _paired_totals(
            prefix,
            count_suffix="num_force_atoms",
            sum_suffixes=(
                "force_vector_error_sum",
                "force_vector_squared_error_sum",
                "force_norm_absolute_error_sum",
            ),
        )
        if atom_count:
            aggregated[f"{prefix}force_vector_mae_eV_per_angstrom"] = (
                atom_totals["force_vector_error_sum"] / atom_count
            )
            aggregated[f"{prefix}force_vector_rmse_eV_per_angstrom"] = np.sqrt(
                max(atom_totals["force_vector_squared_error_sum"] / atom_count, 0)
            )
            aggregated[f"{prefix}force_norm_mae_eV_per_angstrom"] = (
                atom_totals["force_norm_absolute_error_sum"] / atom_count
            )

        cosine_count, cosine_totals = _paired_totals(
            prefix,
            count_suffix="force_cosine_similarity_count",
            sum_suffixes=("force_cosine_similarity_sum",),
        )
        if cosine_count:
            aggregated[f"{prefix}force_cosine_similarity"] = (
                cosine_totals["force_cosine_similarity_sum"] / cosine_count
            )
        elif cosine_count == 0:
            aggregated[f"{prefix}force_cosine_similarity"] = np.nan

    internal_sum_columns = {
        f"{prefix}{suffix}"
        for prefix in prefixes
        for suffix in _ATOMISTIC_PROPERTY_SUM_SUFFIXES
    }
    return aggregated.drop(labels=list(internal_sum_columns), errors="ignore")


def _to_dataframe_value(value: Any) -> Any:
    """Convert metric values to pandas-friendly, hashable values."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()

    if hasattr(value, "numel") and callable(value.numel):
        try:
            if value.numel() == 1:
                return value.item()
        except (RuntimeError, TypeError, ValueError):
            pass

    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            value = value.tolist()
        except (RuntimeError, TypeError, ValueError):
            pass

    if isinstance(value, Mapping):
        return tuple(
            (str(key), _to_dataframe_value(sub_value))
            for key, sub_value in sorted(value.items(), key=lambda item: str(item[0]))
        )

    if isinstance(value, list | tuple | set | ListConfig):
        return tuple(_to_dataframe_value(item) for item in value)

    try:
        hash(value)
    except TypeError:
        return repr(value)

    return value


def _flatten_mapping(
    mapping: Mapping[str, Any],
    parent_key: str = "",
) -> dict[str, Any]:
    """Recursively flatten a nested mapping into a single-level dictionary with dot-separated keys."""
    flattened = {}
    for key, value in mapping.items():
        full_key = f"{parent_key}.{key}" if parent_key else str(key)
        if isinstance(value, Mapping):
            nested = _flatten_mapping(value, full_key)
            if nested:
                flattened.update(nested)
            else:
                flattened[full_key] = value
        else:
            flattened[full_key] = value
    return flattened


def _as_atom_array_list(atom_array_stack: Any) -> list[Any]:
    """Normalize a single AtomArray, stack, or list into a Python list."""
    if atom_array_stack is None:
        return []
    if isinstance(atom_array_stack, list):
        return atom_array_stack
    if isinstance(atom_array_stack, tuple):
        return list(atom_array_stack)

    stack_depth = getattr(atom_array_stack, "stack_depth", None)
    if callable(stack_depth):
        depth = int(stack_depth())
        return [atom_array_stack[i] for i in range(depth)]
    if isinstance(stack_depth, int):
        return [atom_array_stack[i] for i in range(stack_depth)]

    return [atom_array_stack]


def _zero_epoch_afd_metrics(
    *, num_samples: int = 0, total_samples: int = 0
) -> dict[str, float]:
    """Return zero-valued AFD metrics for an epoch-level payload."""
    return {
        "afd_score": 0.0,
        "afd_num_samples": float(num_samples),
        "afd_ref_sample_size": 0.0,
        "afd_input_success_rate": (
            num_samples / total_samples if total_samples else 0.0
        ),
    }


def _zero_atom_counts() -> dict[str, int]:
    """Return zero counts for the atom-fraction metrics."""
    return {key: 0 for key in ATOM_FRACTION_METRIC_KEYS}


def _atom_fraction_metrics(
    atom_counts: Mapping[str, int],
    *,
    total_atoms: int,
) -> dict[str, float]:
    """Convert accumulated O/N/C atom counts into fractions."""
    if total_atoms <= 0:
        return {key: 0.0 for key in ATOM_FRACTION_METRIC_KEYS}
    return {
        key: float(atom_counts.get(key, 0)) / total_atoms
        for key in ATOM_FRACTION_METRIC_KEYS
    }


class _CachedReferenceAFDScorer:
    """AFD scorer backed by precomputed reference features."""

    def __init__(
        self,
        ref_features: torch.Tensor,
        *,
        data_modality: str,
        model_name: str | None,
        batch_size: int,
        device: str | None,
    ) -> None:
        from zevals.atomistic.afd.features import FeatureExtractor, _get_model_name

        if model_name is None:
            model_name = _get_model_name(data_modality)

        self.ref_features = ref_features
        self.ref_ids = None
        self.featurizer = FeatureExtractor(
            model_name=model_name,
            batch_size=batch_size,
            device=device,
        )

    def sample_ref(self, size: int, seed: int | None = None) -> torch.Tensor:
        """Sample reference features for matched-N AFD scoring."""
        if size > len(self.ref_features):
            raise ValueError(
                f"Requested sample size {size} exceeds total number of reference "
                f"features {len(self.ref_features)}."
            )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.ref_features.device).manual_seed(
                seed
            )
        indices = torch.randperm(
            len(self.ref_features),
            generator=generator,
            device=self.ref_features.device,
        )[:size]
        return self.ref_features[indices]

    def embeddings(self, dataset, **kwargs) -> torch.Tensor:
        """Return CT graph-level embeddings for generated atomistic records."""
        return self.featurizer(dataset, **kwargs)

    def __call__(self, dataset, **kwargs) -> float:
        """Compute AFD between cached reference features and generated records."""
        from zevals.atomistic.afd.score import afd_score

        ref_sample_size = kwargs.pop("ref_sample_size", None)
        if ref_sample_size is not None:
            ref_feats = self.sample_ref(ref_sample_size, seed=kwargs.pop("seed", None))
        else:
            ref_feats = self.ref_features

        target_feats = self.embeddings(dataset, **kwargs)
        return afd_score(ref_feats, target_feats)


class StoreValidationMetricsInDFCallback(FoundryStoreValidationMetricsInDFCallback):
    """Store validation metrics while tolerating list-valued metric outputs."""

    def on_validation_batch_end(
        self,
        trainer,
        outputs: dict,
        batch: Any,
        batch_idx: int,
        num_batches: int,
        dataset_name: str | None = None,
        is_distributed_padding: bool = False,
    ):
        """Build a flattened DataFrame from the metrics output and accumulate it."""
        if is_distributed_padding:
            return
        assert "metrics_output" in outputs, "Validation outputs must contain metrics."
        metrics_output = deepcopy(outputs["metrics_output"])

        example_id = _to_dataframe_value(metrics_output.pop("example_id"))
        metrics_as_list_of_dicts = []

        if self.metrics_to_save != "all" and isinstance(
            self.metrics_to_save, list | ListConfig
        ):
            metrics_output = {
                key: value
                for key, value in metrics_output.items()
                if any(key.startswith(prefix) for prefix in self.metrics_to_save)
            }

        def _build_row_from_flattened_dict(
            dict_to_flatten: Mapping[str, Any],
            prefix: str,
            example_id: Any,
        ) -> dict[str, Any]:
            """Flatten a nested dictionary and build a row for the DataFrame."""
            row_data = {"example_id": example_id}
            for sub_key, sub_value in _flatten_mapping(dict_to_flatten).items():
                row_data[f"{prefix}.{sub_key}"] = _to_dataframe_value(sub_value)
            return row_data

        scalar_metrics = {"example_id": example_id}
        for key, value in metrics_output.items():
            if isinstance(value, Mapping):
                metrics_as_list_of_dicts.append(
                    _build_row_from_flattened_dict(value, key, example_id)
                )
            elif isinstance(value, list | ListConfig) and all(
                isinstance(item, Mapping) for item in value
            ):
                for subdict in value:
                    metrics_as_list_of_dicts.append(
                        _build_row_from_flattened_dict(subdict, key, example_id)
                    )
            else:
                assert key not in scalar_metrics, f"Duplicate key: {key}"
                scalar_metrics[key] = _to_dataframe_value(value)

        metrics_as_list_of_dicts.append(scalar_metrics)

        batch_df = pd.DataFrame(metrics_as_list_of_dicts)
        batch_df["epoch"] = trainer.state["current_epoch"]
        batch_df["dataset"] = _to_dataframe_value(dataset_name)

        assert (
            batch_df.duplicated().sum() == 0
        ), "Duplicate rows found in the metrics DataFrame!"

        self.per_gpu_outputs_df = pd.concat(
            [self.per_gpu_outputs_df, batch_df], ignore_index=True
        )

        ranked_logger.info(
            f"Validation Progress: {100 * (batch_idx + 1) / num_batches:.0f}% for {dataset_name}"
        )


class LogMoleculeAFDValidationMetricsCallback(BaseCallback):
    """Compute epoch-level molecule metrics from all validation generations."""

    afd_reference_name = "qm9"
    epoch_afd_example_id = EPOCH_AFD_EXAMPLE_ID
    metric_domain = "molecule"

    def __init__(
        self,
        enabled: bool = False,
        afd_reference_records_path: str | Path | None = None,
        afd_model_name: str | None = None,
        data_modality: str = "molecules",
        afd_batch_size: int = 256,
        afd_device: str | None = None,
        afd_feature_cache_dir: str | Path | None = None,
        afd_ref_sample_size: int | None = None,
        afd_match_reference_size: bool = True,
        seed: int | None = 0,
        show_progress: bool = False,
        compute_afd: bool = True,
        compute_atom_fractions: bool = True,
        dataset_names: Sequence[str] | None = None,
        append_to_validation_results: bool = True,
        log_to_fabric_if_not_appended: bool = True,
    ) -> None:
        self.enabled = enabled
        self.compute_afd = compute_afd
        self.compute_atom_fractions = compute_atom_fractions
        # Size of the reference side of the Fréchet distance. By default
        # (`afd_match_reference_size=True`, `afd_ref_sample_size=None`) the
        # reference is subsampled down to the number of generated samples, so both
        # Gaussians are estimated from the same, possibly small, N. Setting
        # `afd_ref_sample_size` explicitly pins the reference size instead, which
        # reduces variance on the reference side at no extra generation cost --
        # reference features are cached, so a larger reference is embedded once.
        # NOTE: AFD values are only comparable across runs that use the same
        # generated N *and* the same reference N.
        self.afd_ref_sample_size = afd_ref_sample_size
        self.afd_match_reference_size = afd_match_reference_size
        self.afd_reference_records_path = (
            Path(afd_reference_records_path)
            if afd_reference_records_path is not None
            else None
        )
        self.afd_model_name = afd_model_name
        self.data_modality = data_modality
        self.afd_batch_size = afd_batch_size
        self.afd_device = afd_device
        self.afd_feature_cache_dir = (
            Path(afd_feature_cache_dir) if afd_feature_cache_dir is not None else None
        )
        self.seed = seed
        self.show_progress = show_progress
        self.dataset_names = (
            frozenset(str(value) for value in dataset_names)
            if dataset_names is not None
            else None
        )
        self.append_to_validation_results = append_to_validation_results
        self.log_to_fabric_if_not_appended = log_to_fabric_if_not_appended
        self._afd_reference = None
        self._afd_scorer = None
        self._records_by_dataset: dict[str, list[Any]] = {}
        self._total_samples_by_dataset: dict[str, int] = {}
        self._atom_counts_by_dataset: dict[str, dict[str, int]] = {}
        self._total_atoms_by_dataset: dict[str, int] = {}

    def on_validation_epoch_start(self, trainer: Any):
        """Reset epoch-local generated molecule records."""
        del trainer
        self._records_by_dataset = defaultdict(list)
        self._total_samples_by_dataset = defaultdict(int)
        self._atom_counts_by_dataset = defaultdict(_zero_atom_counts)
        self._total_atoms_by_dataset = defaultdict(int)

    def on_validation_batch_end(
        self,
        trainer: Any,
        outputs: dict,
        batch: Any,
        batch_idx: int,
        num_batches: int,
        dataset_name: str | None = None,
        is_distributed_padding: bool = False,
    ):
        """Collect generated atom arrays for enabled epoch-level molecule metrics."""
        del trainer, batch, batch_idx, num_batches
        if (
            is_distributed_padding
            or not self.enabled
            or not (self.compute_afd or self.compute_atom_fractions)
        ):
            return

        dataset = self._dataset_key(dataset_name)
        if self.dataset_names is not None and dataset not in self.dataset_names:
            return

        atom_arrays = _as_atom_array_list(outputs.get("predicted_atom_array_stack"))
        if self.compute_afd:
            self._total_samples_by_dataset[dataset] += len(atom_arrays)

        for atom_array in atom_arrays:
            if self.compute_afd:
                record = self._atom_array_to_afd_record(atom_array)
                if record is not None:
                    self._records_by_dataset[dataset].append(record)

            if self.compute_atom_fractions:
                elements = getattr(atom_array, "element", ())
                if elements is None:
                    elements = ()
                for element in elements:
                    key = ATOM_FRACTION_ELEMENT_KEYS.get(str(element).strip().upper())
                    if key is not None:
                        self._atom_counts_by_dataset[dataset][key] += 1
                self._total_atoms_by_dataset[dataset] += len(elements)

    def on_validation_epoch_end(self, trainer: Any):
        """Gather local payloads across ranks and log one metric row per dataset."""
        if not self.enabled or not (self.compute_afd or self.compute_atom_fractions):
            return

        should_compute, payload = self._gather_payload_to_rank_zero(trainer)
        if not should_compute or not payload:
            return

        dataset_metrics = {}
        for dataset, dataset_payload in payload.items():
            records = dataset_payload["records"]
            total_samples = dataset_payload["total_samples"]
            metrics = self._compute_epoch_metrics(
                records=records,
                total_samples=total_samples,
                atom_counts=dataset_payload.get("atom_counts", {}),
                total_atoms=dataset_payload.get("total_atoms", 0),
            )
            if metrics:
                dataset_metrics[dataset] = metrics
            ranked_logger.info(
                f"Computed epoch-level {self.metric_domain} metrics for "
                f"{dataset}: {metrics}"
            )

        if not dataset_metrics:
            return

        appended = self._append_to_validation_results(trainer, dataset_metrics)
        if not appended and self.log_to_fabric_if_not_appended:
            self._log_to_fabric(trainer, dataset_metrics)

    @staticmethod
    def _dataset_key(dataset_name: str | None) -> str:
        return str(dataset_name) if dataset_name is not None else "validation"

    @staticmethod
    def _atom_array_to_afd_record(atom_array: Any):
        return _atom_array_to_afd_record(atom_array)

    def _local_payload(self) -> dict[str, dict[str, Any]]:
        datasets = (
            set(self._records_by_dataset)
            | set(self._total_samples_by_dataset)
            | set(self._atom_counts_by_dataset)
            | set(self._total_atoms_by_dataset)
        )
        return {
            dataset: {
                "records": list(self._records_by_dataset.get(dataset, [])),
                "total_samples": int(self._total_samples_by_dataset.get(dataset, 0)),
                "atom_counts": dict(
                    self._atom_counts_by_dataset.get(dataset, _zero_atom_counts())
                ),
                "total_atoms": int(self._total_atoms_by_dataset.get(dataset, 0)),
            }
            for dataset in sorted(datasets)
        }

    @staticmethod
    def _merge_payloads(payloads: list[dict[str, dict[str, Any]] | None]):
        merged_records: dict[str, list[Any]] = defaultdict(list)
        merged_totals: dict[str, int] = defaultdict(int)
        merged_atom_counts: dict[str, dict[str, int]] = defaultdict(_zero_atom_counts)
        merged_total_atoms: dict[str, int] = defaultdict(int)

        for payload in payloads:
            if not payload:
                continue
            for dataset, dataset_payload in payload.items():
                merged_records[dataset].extend(dataset_payload.get("records", []))
                merged_totals[dataset] += int(dataset_payload.get("total_samples", 0))
                for key, count in dataset_payload.get("atom_counts", {}).items():
                    merged_atom_counts[dataset][key] += int(count)
                merged_total_atoms[dataset] += int(
                    dataset_payload.get("total_atoms", 0)
                )

        datasets = (
            set(merged_records)
            | set(merged_totals)
            | set(merged_atom_counts)
            | set(merged_total_atoms)
        )
        return {
            dataset: {
                "records": merged_records.get(dataset, []),
                "total_samples": merged_totals.get(dataset, 0),
                "atom_counts": dict(
                    merged_atom_counts.get(dataset, _zero_atom_counts())
                ),
                "total_atoms": merged_total_atoms.get(dataset, 0),
            }
            for dataset in sorted(datasets)
        }

    def _gather_payload_to_rank_zero(
        self,
        trainer: Any,
    ) -> tuple[bool, dict[str, dict[str, Any]]]:
        fabric = getattr(trainer, "fabric", None)
        rank = int(getattr(fabric, "global_rank", 0))
        world_size = int(getattr(fabric, "world_size", 1))
        local_payload = self._local_payload()

        if world_size <= 1:
            return True, local_payload

        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            raise RuntimeError(
                f"Epoch-level {self.metric_domain} validation metrics with distributed "
                "validation require an initialized torch.distributed process "
                "group to aggregate records."
            )

        gathered_payloads = [None] * world_size if rank == 0 else None
        torch.distributed.gather_object(local_payload, gathered_payloads, dst=0)
        if rank != 0:
            return False, {}

        return True, self._merge_payloads(gathered_payloads)

    def _get_afd_reference(self):
        if self.afd_reference_records_path is None:
            return self.afd_reference_name
        if self._afd_reference is None:
            from zevals.atomistic.afd.data import resolve_records

            self._afd_reference = resolve_records(str(self.afd_reference_records_path))
        return self._afd_reference

    def _resolve_afd_model_name(self) -> str:
        if self.afd_model_name is not None:
            return self.afd_model_name

        from zevals.atomistic.afd.features import _get_model_name

        return _get_model_name(self.data_modality)

    def _reference_feature_cache_path(self) -> Path | None:
        if (
            self.afd_feature_cache_dir is None
            or self.afd_reference_records_path is None
        ):
            return None

        safe_model_name = self._resolve_afd_model_name().replace("/", "__")
        return self.afd_feature_cache_dir / f"{safe_model_name}_reference_features.pt"

    def _get_cached_reference_features(self) -> torch.Tensor | None:
        cache_path = self._reference_feature_cache_path()
        if cache_path is None or not cache_path.exists():
            return None

        ranked_logger.info(f"Loading cached AFD reference features from {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    def _get_afd_scorer(self):
        if self._afd_scorer is None:
            cached_ref_features = self._get_cached_reference_features()
            if cached_ref_features is not None:
                self._afd_scorer = _CachedReferenceAFDScorer(
                    cached_ref_features,
                    data_modality=self.data_modality,
                    model_name=self.afd_model_name,
                    batch_size=self.afd_batch_size,
                    device=self.afd_device,
                )
                return self._afd_scorer

            AFDScore = self._get_afd_score_class()

            self._afd_scorer = AFDScore(
                self._get_afd_reference(),
                data_modality=self.data_modality,
                model_name=self.afd_model_name,
                batch_size=self.afd_batch_size,
                device=self.afd_device,
                feature_cache_dir=self.afd_feature_cache_dir,
            )
        return self._afd_scorer

    @staticmethod
    def _get_afd_score_class():
        from zevals.molecules.afd import AFDScore

        return AFDScore

    def _compute_afd_metrics(self, records: list[Any]) -> dict[str, float]:
        from zevals.molecules.afd import compute_afd_metrics

        return compute_afd_metrics(
            records,
            scorer=self._get_afd_scorer(),
            reference=self.afd_reference_name,
            data_modality=self.data_modality,
            model_name=self.afd_model_name,
            ref_sample_size=self.afd_ref_sample_size,
            match_reference_size=self.afd_match_reference_size,
            seed=self.seed,
            show_progress=self.show_progress,
        )

    def _compute_epoch_metrics(
        self,
        *,
        records: list[Any],
        total_samples: int,
        atom_counts: Mapping[str, int],
        total_atoms: int,
    ) -> dict[str, float]:
        metrics = {}

        if self.compute_afd:
            if len(records) < 2:
                metrics.update(
                    _zero_epoch_afd_metrics(
                        num_samples=len(records),
                        total_samples=total_samples,
                    )
                )
            else:
                afd_metrics = self._compute_afd_metrics(records)
                afd_metrics["afd_input_success_rate"] = (
                    len(records) / total_samples if total_samples else 0.0
                )
                metrics.update(
                    {key: float(afd_metrics.get(key, 0.0)) for key in AFD_METRIC_KEYS}
                )

        if self.compute_atom_fractions:
            metrics.update(_atom_fraction_metrics(atom_counts, total_atoms=total_atoms))

        return metrics

    def _append_to_validation_results(
        self,
        trainer: Any,
        dataset_metrics: dict[str, dict[str, float]],
    ) -> bool:
        if not self.append_to_validation_results:
            return False

        fabric = getattr(trainer, "fabric", None)
        if not bool(getattr(fabric, "is_global_zero", True)):
            return False

        validation_results_path = getattr(trainer, "validation_results_path", None)
        if validation_results_path is None:
            return False

        path = Path(validation_results_path)
        epoch = trainer.state["current_epoch"]
        rows = [
            {
                "example_id": self.epoch_afd_example_id,
                "epoch": epoch,
                "dataset": dataset,
                **metrics,
            }
            for dataset, metrics in dataset_metrics.items()
        ]
        epoch_metrics_df = pd.DataFrame(rows)

        if path.exists():
            try:
                existing_df = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                existing_df = pd.DataFrame()
        else:
            existing_df = pd.DataFrame()

        if not existing_df.empty and {
            "epoch",
            "dataset",
            "example_id",
        } <= set(existing_df.columns):
            duplicate_mask = (
                (existing_df["epoch"] == epoch)
                & (existing_df["example_id"] == self.epoch_afd_example_id)
                & (existing_df["dataset"].astype(str).isin(dataset_metrics))
            )
            existing_df = existing_df[~duplicate_mask]

        merged_df = pd.concat(
            [existing_df, epoch_metrics_df],
            ignore_index=True,
            sort=False,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(path, index=False)
        ranked_logger.info(
            f"Appended epoch-level {self.metric_domain} metrics to {path}"
        )
        return True

    def _log_to_fabric(
        self,
        trainer: Any,
        dataset_metrics: dict[str, dict[str, float]],
    ) -> None:
        fabric = getattr(trainer, "fabric", None)
        if fabric is None or not hasattr(fabric, "log_dict"):
            return

        metrics_to_log = {
            f"val/{dataset}/{key}": value
            for dataset, metrics in dataset_metrics.items()
            for key, value in metrics.items()
        }
        if not metrics_to_log:
            return
        fabric.log_dict(metrics_to_log, step=trainer.state.get("global_step"))


class LogMaterialsAFDValidationMetricsCallback(LogMoleculeAFDValidationMetricsCallback):
    """Compute epoch-level periodic-materials AFD over validation generations."""

    afd_reference_name = "mp20"
    epoch_afd_example_id = EPOCH_MATERIALS_AFD_EXAMPLE_ID
    metric_domain = "materials"

    def __init__(
        self,
        enabled: bool = False,
        afd_reference_records_path: str | Path | None = None,
        afd_model_name: str | None = DEFAULT_MATERIALS_AFD_MODEL_NAME,
        afd_batch_size: int = 256,
        afd_device: str | None = None,
        afd_feature_cache_dir: str | Path | None = None,
        seed: int | None = 0,
        show_progress: bool = False,
        dataset_names: Sequence[str] | None = None,
        append_to_validation_results: bool = True,
        log_to_fabric_if_not_appended: bool = True,
    ) -> None:
        super().__init__(
            enabled=enabled,
            afd_reference_records_path=afd_reference_records_path,
            afd_model_name=afd_model_name,
            data_modality="materials",
            afd_batch_size=afd_batch_size,
            afd_device=afd_device,
            afd_feature_cache_dir=afd_feature_cache_dir,
            seed=seed,
            show_progress=show_progress,
            compute_afd=True,
            compute_atom_fractions=False,
            dataset_names=dataset_names,
            append_to_validation_results=append_to_validation_results,
            log_to_fabric_if_not_appended=log_to_fabric_if_not_appended,
        )

    @staticmethod
    def _atom_array_to_afd_record(atom_array: Any):
        return _atom_array_to_material_afd_record(atom_array)

    @staticmethod
    def _get_afd_score_class():
        from zevals.materials.afd import AFDScore

        return AFDScore

    def _compute_afd_metrics(self, records: list[Any]) -> dict[str, float]:
        from zevals.materials.afd import compute_afd_metrics

        return compute_afd_metrics(
            records,
            scorer=self._get_afd_scorer(),
            reference=self.afd_reference_name,
            data_modality="materials",
            model_name=self.afd_model_name,
            ref_sample_size=self.afd_ref_sample_size,
            match_reference_size=self.afd_match_reference_size,
            seed=self.seed,
            show_progress=self.show_progress,
        )


EPOCH_ZEVALS_EXAMPLE_ID = "__epoch_zevals_structure_prediction__"
EPOCH_ZEVALS_PREFIX = "zevals_epoch."


def compute_molecule_structure_prediction_metrics(*args: Any, **kwargs: Any):
    """Lazily load the Zevals molecular structure-prediction metric."""
    from zevals.molecules.structure_prediction import (
        compute_molecule_structure_prediction_metrics as compute_metrics,
    )

    return compute_metrics(*args, **kwargs)


def compute_material_structure_prediction_metrics(*args: Any, **kwargs: Any):
    """Lazily load the Zevals material structure-prediction metric."""
    from zevals.materials.structure_prediction import (
        compute_material_structure_prediction_metrics as compute_metrics,
    )

    return compute_metrics(*args, **kwargs)


def _candidate_budget_label(generated_by_target: Mapping[str, Sequence[Any]]) -> str:
    """Return ``at_K`` for a uniform candidate budget, otherwise variable-K."""
    counts = {len(candidates) for candidates in generated_by_target.values()}
    return f"at_{next(iter(counts))}" if len(counts) == 1 else "at_variable_k"


class LogGroupedZevalsStructurePredictionMetricsCallback(BaseCallback):
    """Compute dataset-level zevals metrics once from all validation targets."""

    def __init__(
        self,
        enabled: bool = False,
        molecule_top_k: Sequence[int] = (1,),
        material_matcher_scales: Sequence[bool] = (True,),
        material_top_k: Sequence[int] = (1,),
        structure_ltol: float = 0.3,
        structure_stol: float = 0.5,
        structure_angle_tol: float = 10.0,
        dataset_names: Sequence[str] | None = None,
        append_to_validation_results: bool = True,
        log_to_fabric: bool = True,
    ) -> None:
        molecule_budgets = tuple(int(value) for value in molecule_top_k)
        if not molecule_budgets or any(value <= 0 for value in molecule_budgets):
            raise ValueError("molecule_top_k must be positive and non-empty.")
        scales = tuple(bool(value) for value in material_matcher_scales)
        if not scales:
            raise ValueError("material_matcher_scales must be non-empty.")
        budgets = tuple(int(value) for value in material_top_k)
        if not budgets or any(value <= 0 for value in budgets):
            raise ValueError("material_top_k must be positive and non-empty.")

        self.enabled = bool(enabled)
        self.molecule_top_k = tuple(sorted(set(molecule_budgets)))
        self.material_matcher_scales = tuple(dict.fromkeys(scales))
        self.material_top_k = tuple(sorted(set(budgets)))
        self.structure_ltol = float(structure_ltol)
        self.structure_stol = float(structure_stol)
        self.structure_angle_tol = float(structure_angle_tol)
        self.append_to_validation_results = bool(append_to_validation_results)
        self.log_to_fabric = bool(log_to_fabric)
        self.dataset_names = (
            frozenset(str(value) for value in dataset_names)
            if dataset_names is not None
            else None
        )
        self._records_by_dataset: dict[str, dict[str, Any]] = {}

    def on_validation_epoch_start(self, trainer: Any) -> None:
        """Reset target-grouped records for the validation epoch."""
        del trainer
        self._records_by_dataset = {}

    def on_validation_batch_end(
        self,
        trainer: Any,
        outputs: dict,
        batch: Any,
        batch_idx: int,
        num_batches: int,
        dataset_name: str | None = None,
        is_distributed_padding: bool = False,
    ) -> None:
        """Convert and collect one target's candidates and references."""
        del trainer, batch, batch_idx, num_batches
        if is_distributed_padding or not self.enabled:
            return

        from nucleus.metrics.structure_prediction import (
            _condition_name,
            _is_material,
            _molecule_from_conditioned_graph,
            _target_id,
            _to_rdkit_molecule,
            _validate_reference_coordinates,
        )
        from nucleus.utils.material_fingerprints import atom_array_to_structure

        extra_info = outputs.get("metrics_extra_info")
        dataset = str(dataset_name or "validation")
        if self.dataset_names is not None and dataset not in self.dataset_names:
            return

        condition_name = _condition_name(extra_info)
        if condition_name is not None and condition_name not in {
            "structure_prediction",
            "sequence_fixed_structure_design",
        }:
            return
        if (
            condition_name is None
            and self.dataset_names is None
            and "structure_prediction" not in dataset
        ):
            return

        predicted_arrays = _as_atom_array_list(
            outputs.get("predicted_atom_array_stack")
        )
        ground_truth_arrays = _as_atom_array_list(
            outputs.get("ground_truth_atom_array_stack")
        )
        if not predicted_arrays or not ground_truth_arrays:
            return

        target_id = _target_id(extra_info)
        reference_array = ground_truth_arrays[0]
        domain = "materials" if _is_material(reference_array) else "molecules"
        records = self._records_by_dataset.setdefault(
            dataset,
            {
                "domain": domain,
                "generated": defaultdict(list),
                "reference": {},
            },
        )
        if records["domain"] != domain:
            raise ValueError(f"{dataset}: mixed molecule/material validation dataset.")
        if target_id in records["reference"]:
            raise ValueError(f"{dataset}: duplicate validation target {target_id!r}.")

        if domain == "materials":
            reference = atom_array_to_structure(reference_array)
            if reference is None:
                raise ValueError(f"{target_id}: invalid material reference structure.")
            candidates = [
                atom_array_to_structure(atom_array) for atom_array in predicted_arrays
            ]
            records["reference"][target_id] = [reference]
            records["generated"][target_id].extend(candidates)
            return

        features = (outputs.get("metric_network_input") or {}).get("f", {})
        _validate_reference_coordinates(
            reference_array.coord,
            target_id=target_id,
        )
        reference = _molecule_from_conditioned_graph(reference_array, features)
        if reference is None:
            reference = _to_rdkit_molecule(
                reference_array,
                example_id=f"{target_id}:reference",
            )
        if reference is None:
            raise ValueError(f"{target_id}: invalid molecular reference conformer.")

        candidates = []
        for index, atom_array in enumerate(predicted_arrays):
            candidate = _molecule_from_conditioned_graph(atom_array, features)
            if candidate is None:
                candidate = _to_rdkit_molecule(
                    atom_array,
                    example_id=f"{target_id}:candidate-{index}",
                )
            candidates.append(candidate)
        records["reference"][target_id] = [reference]
        records["generated"][target_id].extend(candidates)

    def on_validation_epoch_end(self, trainer: Any) -> None:
        """Gather records and log one true dataset-level aggregate per dataset."""
        if not self.enabled:
            return
        should_compute, records_by_dataset = self._gather_to_rank_zero(trainer)
        rank_zero_error: Exception | None = None
        if should_compute:
            try:
                metrics_by_dataset = {
                    dataset: self._compute_dataset_metrics(records)
                    for dataset, records in records_by_dataset.items()
                }
                metrics_by_dataset = {
                    dataset: metrics
                    for dataset, metrics in metrics_by_dataset.items()
                    if metrics
                }
                if metrics_by_dataset:
                    if self.append_to_validation_results:
                        self._append_to_validation_results(trainer, metrics_by_dataset)
                    if self.log_to_fabric:
                        self._log_to_fabric(trainer, metrics_by_dataset)
            except Exception as error:
                rank_zero_error = error

        self._synchronize_rank_zero_exception(
            trainer,
            rank_zero_error,
            operation="computing grouped zevals validation metrics",
        )

    def _compute_dataset_metrics(self, records: dict[str, Any]) -> dict[str, float]:
        generated = dict(records["generated"])
        reference = dict(records["reference"])
        if not generated:
            return {}

        candidate_counts = np.asarray(
            [len(candidates) for candidates in generated.values()], dtype=float
        )
        reference_counts = np.asarray(
            [len(references) for references in reference.values()], dtype=float
        )
        metrics = {
            "num_targets": float(len(generated)),
            "candidate_count_mean": float(candidate_counts.mean()),
            "candidate_count_min": float(candidate_counts.min()),
            "candidate_count_max": float(candidate_counts.max()),
            "reference_count_mean": float(reference_counts.mean()),
            "reference_count_min": float(reference_counts.min()),
            "reference_count_max": float(reference_counts.max()),
        }

        if records["domain"] == "molecules":
            metrics.update(
                compute_molecule_structure_prediction_metrics(
                    generated,
                    reference,
                    top_k=self.molecule_top_k,
                )
            )
            return metrics

        budget_suffix = _candidate_budget_label(generated)
        for scale in self.material_matcher_scales:
            result = compute_material_structure_prediction_metrics(
                generated,
                reference,
                ltol=self.structure_ltol,
                stol=self.structure_stol,
                angle_tol=self.structure_angle_tol,
                scale=scale,
                top_k=self.material_top_k,
            )
            suffix = f"{budget_suffix}_scale_{str(scale).lower()}"
            metrics.update(
                {f"{key}_{suffix}": float(value) for key, value in result.items()}
            )
        return metrics

    def _local_payload(self) -> dict[str, dict[str, Any]]:
        return {
            dataset: {
                "domain": records["domain"],
                "generated": dict(records["generated"]),
                "reference": dict(records["reference"]),
            }
            for dataset, records in self._records_by_dataset.items()
        }

    @staticmethod
    def _merge_payloads(payloads: Sequence[dict[str, dict[str, Any]] | None]):
        """Merge rank payloads without inflating repeated target budgets.

        ``DistributedSampler(drop_last=False)`` repeats leading dataset examples
        when the dataset size is not divisible by the world size. The validation
        loop normally marks and excludes those executions before collection;
        retaining the first complete record is a defensive fallback for payloads
        produced without that provenance.
        """
        merged: dict[str, dict[str, Any]] = {}
        for payload in payloads:
            if not payload:
                continue
            for dataset, records in payload.items():
                destination = merged.setdefault(
                    dataset,
                    {
                        "domain": records["domain"],
                        "generated": defaultdict(list),
                        "reference": {},
                    },
                )
                if destination["domain"] != records["domain"]:
                    raise ValueError(f"{dataset}: mixed molecule/material payloads.")
                generated_by_target = records["generated"]
                references_by_target = records["reference"]
                if set(generated_by_target) != set(references_by_target):
                    raise ValueError(
                        f"{dataset}: generated and reference target IDs do not match."
                    )
                for target_id, target_references in references_by_target.items():
                    if target_id in destination["reference"]:
                        continue
                    destination["reference"][target_id] = target_references
                    destination["generated"][target_id].extend(
                        generated_by_target[target_id]
                    )
        return merged

    @staticmethod
    def _synchronize_rank_zero_exception(
        trainer: Any,
        error: Exception | None,
        *,
        operation: str,
    ) -> None:
        """Propagate a rank-zero callback failure before ranks can diverge."""
        fabric = getattr(trainer, "fabric", None)
        rank = int(getattr(fabric, "global_rank", 0))
        world_size = int(getattr(fabric, "world_size", 1))
        if world_size <= 1:
            if error is not None:
                raise error
            return

        error_payload = [
            (
                {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                if rank == 0 and error is not None
                else None
            )
        ]
        torch.distributed.broadcast_object_list(error_payload, src=0)
        error_info = error_payload[0]
        if error_info is None:
            return
        if error is not None:
            raise error
        raise RuntimeError(
            f"Rank zero failed while {operation}: "
            f"{error_info['type']}: {error_info['message']}"
        )

    def _gather_to_rank_zero(self, trainer: Any):
        fabric = getattr(trainer, "fabric", None)
        rank = int(getattr(fabric, "global_rank", 0))
        world_size = int(getattr(fabric, "world_size", 1))
        local_payload = self._local_payload()
        if world_size <= 1:
            return True, local_payload
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            raise RuntimeError(
                "Grouped zevals validation requires initialized torch.distributed."
            )
        gathered = [None] * world_size if rank == 0 else None
        torch.distributed.gather_object(local_payload, gathered, dst=0)
        merged: dict[str, dict[str, Any]] = {}
        merge_error: Exception | None = None
        if rank == 0:
            try:
                merged = self._merge_payloads(gathered)
            except Exception as error:
                merge_error = error
        self._synchronize_rank_zero_exception(
            trainer,
            merge_error,
            operation="merging grouped zevals validation records",
        )
        return rank == 0, merged

    @staticmethod
    def _prefixed(metrics: Mapping[str, float]) -> dict[str, float]:
        return {f"{EPOCH_ZEVALS_PREFIX}{key}": value for key, value in metrics.items()}

    def _append_to_validation_results(
        self,
        trainer: Any,
        metrics_by_dataset: Mapping[str, Mapping[str, float]],
    ) -> None:
        path_value = getattr(trainer, "validation_results_path", None)
        if path_value is None:
            raise RuntimeError(
                "Grouped zevals callback requires validation_results_path from the "
                "StoreValidationMetricsInDFCallback."
            )
        path = Path(path_value)
        epoch = trainer.state["current_epoch"]
        rows = [
            {
                "example_id": EPOCH_ZEVALS_EXAMPLE_ID,
                "epoch": epoch,
                "dataset": dataset,
                **self._prefixed(metrics),
            }
            for dataset, metrics in metrics_by_dataset.items()
        ]
        try:
            existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
        if not existing.empty and {"epoch", "dataset", "example_id"} <= set(
            existing.columns
        ):
            duplicates = (existing["epoch"] == epoch) & (
                existing["example_id"] == EPOCH_ZEVALS_EXAMPLE_ID
            )
            existing = existing[~duplicates]
        merged = pd.concat(
            [existing, pd.DataFrame(rows)], ignore_index=True, sort=False
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(path, index=False)
        ranked_logger.info(f"Appended grouped zevals metrics to {path}")

    def _log_to_fabric(
        self,
        trainer: Any,
        metrics_by_dataset: Mapping[str, Mapping[str, float]],
    ) -> None:
        fabric = getattr(trainer, "fabric", None)
        if fabric is None or not hasattr(fabric, "log_dict"):
            return
        fabric.log_dict(
            {
                f"val/{dataset}/{key}": value
                for dataset, metrics in metrics_by_dataset.items()
                for key, value in self._prefixed(metrics).items()
            },
            step=trainer.state.get("global_step"),
        )


class LogDesignValidationMetricsCallback(BaseCallback):
    """Log validation metrics averaged across examples for each dataset."""

    def on_validation_epoch_end(self, trainer: Any):
        """Log dataset-level validation means without per-example metric panels."""
        if not trainer.fabric.is_global_zero:
            return

        assert hasattr(
            trainer, "validation_results_path"
        ), "Results path not found! Ensure that StoreValidationMetricsInDFCallback is called first."

        df = pd.read_csv(trainer.validation_results_path)

        df = df[df["epoch"] == df["epoch"].max()].copy()
        df.drop(columns=["epoch"], inplace=True)

        for dataset in df["dataset"].unique():
            dataset_df = df[df["dataset"] == dataset].copy()
            dataset_df.drop(columns=["dataset"], inplace=True)

            print(f"\n+{' ' + str(dataset) + ' ':-^150}+\n")

            remaining_cols = [
                col for col in dataset_df.columns if col not in ["example_id"]
            ]
            remaining_df = dataset_df[remaining_cols].copy()
            remaining_df = remaining_df.dropna(how="all")
            numeric_cols = remaining_df.select_dtypes(include="number").columns

            final_means = remaining_df[numeric_cols].mean()
            final_means = _aggregate_atomistic_property_metric_columns(
                remaining_df, final_means
            )
            non_nan_counts = remaining_df[numeric_cols].count()
            populated_numeric_cols = final_means[final_means.notna()].index

            final_means_df = final_means[populated_numeric_cols].to_frame(name="mean")
            final_means_df["Count"] = non_nan_counts.reindex(populated_numeric_cols)

            print_df_as_table(
                final_means_df.reset_index(),
                f"{dataset} - {trainer.state['current_epoch']} - Design Validation Metrics",
            )

            if trainer.fabric:
                trainer.fabric.log_dict(
                    {
                        f"val/{dataset}/{col}": final_means[col]
                        for col in populated_numeric_cols
                    },
                    step=trainer.state["global_step"],
                )


class LogCompositeValidationMetricsCallback(BaseCallback):
    """Log coverage-aware, task-balanced validation scores.

    Each raw metric is mapped to a dimensionless skill score in ``[0, 1]``
    using fixed bad/good anchors. Metrics are averaged within a task, while
    task scores are combined with a weighted geometric mean so that one strong
    capability cannot completely hide another weak capability.

    The callback is deliberately stateless: it reads the validation CSV after
    the existing epoch-level aggregation callbacks have appended their rows.
    It can therefore be added to a run when that run resumes from an older
    checkpoint without changing checkpoint serialization.

    Args:
        task_metrics: Mapping from task name to metric specifications. Each
            specification requires ``dataset``, ``metric``, ``direction``,
            ``bad``, and ``good`` and may provide ``weight``.
        task_weights: Optional task-level weights. Missing entries default to
            one.
        active_tasks: Optional explicit task list. When omitted, active tasks
            are inferred from positive ``unconditional``,
            ``structure_prediction``, and ``mlip`` training-condition
            frequencies. If inference is unavailable, tasks with metrics in
            the current validation epoch are used.
        enabled: Whether to compute and log the scores.
        epsilon: Positive floor used by the geometric mean.
        log_prefix: Prefix for logged metrics.
    """

    _CONDITION_TO_TASK = {
        "unconditional": "generation",
        "unconditional_design": "generation",
        "structure_prediction": "structure_prediction",
        "mlip": "mlip",
    }
    _MIN_AFD_SAMPLES = 2

    def __init__(
        self,
        *,
        task_metrics: Mapping[str, Sequence[Mapping[str, Any]]],
        task_weights: Mapping[str, float] | None = None,
        active_tasks: Sequence[str] | None = None,
        enabled: bool = True,
        epsilon: float = 1e-6,
        log_prefix: str = "val/composite",
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")
        if not task_metrics:
            raise ValueError("task_metrics must define at least one task.")

        normalized_specs: dict[str, tuple[dict[str, Any], ...]] = {}
        for task_name, specs in task_metrics.items():
            task_name = str(task_name)
            if not specs:
                raise ValueError(f"{task_name}: at least one metric is required.")
            normalized_specs[task_name] = tuple(
                self._validate_metric_spec(task_name, dict(spec)) for spec in specs
            )

        normalized_task_weights = {
            str(task_name): float(weight)
            for task_name, weight in (task_weights or {}).items()
        }
        unknown_weight_tasks = set(normalized_task_weights) - set(normalized_specs)
        if unknown_weight_tasks:
            raise ValueError(
                "task_weights contains unknown tasks: "
                f"{sorted(unknown_weight_tasks)}."
            )
        for task_name in normalized_specs:
            weight = normalized_task_weights.setdefault(task_name, 1.0)
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError(
                    f"{task_name}: task weight must be finite and positive."
                )

        if active_tasks is not None:
            active_tasks = tuple(str(task) for task in active_tasks)
            unknown_active_tasks = set(active_tasks) - set(normalized_specs)
            if unknown_active_tasks:
                raise ValueError(
                    "active_tasks contains unknown tasks: "
                    f"{sorted(unknown_active_tasks)}."
                )

        self.task_metrics = normalized_specs
        self.task_weights = normalized_task_weights
        self.active_tasks = active_tasks
        self.enabled = enabled
        self.epsilon = epsilon
        self.log_prefix = log_prefix.rstrip("/")

    @staticmethod
    def _validate_metric_spec(task_name: str, spec: dict[str, Any]) -> dict[str, Any]:
        required = {"dataset", "metric", "direction", "bad", "good"}
        missing = required - set(spec)
        if missing:
            raise ValueError(f"{task_name}: metric specification is missing {missing}.")

        direction = str(spec["direction"])
        if direction not in {"lower", "higher"}:
            raise ValueError(
                f"{task_name}: direction must be 'lower' or 'higher', got "
                f"{direction!r}."
            )
        bad = float(spec["bad"])
        good = float(spec["good"])
        weight = float(spec.get("weight", 1.0))
        if not all(np.isfinite(value) for value in (bad, good, weight)):
            raise ValueError(f"{task_name}: metric anchors and weight must be finite.")
        if weight <= 0:
            raise ValueError(f"{task_name}: metric weight must be positive.")
        if direction == "lower" and not (bad > good > 0):
            raise ValueError(
                f"{task_name}: lower-is-better anchors require bad > good > 0."
            )
        if direction == "higher" and not good > bad:
            raise ValueError(
                f"{task_name}: higher-is-better anchors require good > bad."
            )

        return {
            "dataset": str(spec["dataset"]),
            "metric": str(spec["metric"]),
            "direction": direction,
            "bad": bad,
            "good": good,
            "weight": weight,
        }

    @staticmethod
    def _metric_skill(value: float, spec: Mapping[str, Any]) -> float | None:
        """Map one finite raw metric to an anchored skill score in ``[0, 1]``."""
        if not np.isfinite(value):
            return None
        bad = float(spec["bad"])
        good = float(spec["good"])
        if spec["direction"] == "lower":
            if value <= 0:
                return 1.0
            skill = np.log(bad / value) / np.log(bad / good)
        else:
            skill = (value - bad) / (good - bad)
        return float(np.clip(skill, 0.0, 1.0))

    @classmethod
    def _metric_is_available(
        cls,
        dataset_metrics: Mapping[str, float],
        spec: Mapping[str, Any],
    ) -> bool:
        """Return whether a reported metric has enough data to be meaningful."""
        if spec["metric"] != "afd_score":
            return True

        num_samples = dataset_metrics.get("afd_num_samples")
        return (
            num_samples is not None
            and np.isfinite(num_samples)
            and num_samples >= cls._MIN_AFD_SAMPLES
        )

    @staticmethod
    def _latest_dataset_metrics(trainer: Any) -> dict[str, dict[str, float]]:
        """Read exact dataset-level metrics for the current validation epoch."""
        path = Path(trainer.validation_results_path)
        try:
            dataframe = pd.read_csv(path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return {}
        if dataframe.empty or not {"epoch", "dataset"} <= set(dataframe.columns):
            return {}

        current_epoch = trainer.state.get("current_epoch")
        epoch_dataframe = dataframe[dataframe["epoch"] == current_epoch].copy()
        if epoch_dataframe.empty:
            return {}

        metrics_by_dataset: dict[str, dict[str, float]] = {}
        for dataset in epoch_dataframe["dataset"].dropna().unique():
            dataset_dataframe = epoch_dataframe[
                epoch_dataframe["dataset"] == dataset
            ].copy()
            dataset_dataframe.drop(
                columns=["epoch", "dataset", "example_id"],
                errors="ignore",
                inplace=True,
            )
            numeric_dataframe = dataset_dataframe.apply(pd.to_numeric, errors="coerce")
            default_means = numeric_dataframe.mean()
            final_means = _aggregate_atomistic_property_metric_columns(
                numeric_dataframe, default_means
            )
            metrics_by_dataset[str(dataset)] = {
                str(metric): float(value)
                for metric, value in final_means.items()
                if pd.notna(value) and np.isfinite(value)
            }
        return metrics_by_dataset

    def _infer_active_tasks(self, trainer: Any) -> tuple[str, ...] | None:
        if self.active_tasks is not None:
            return tuple(self.active_tasks)

        train_config = trainer.state.get("train_cfg")
        try:
            conditions = train_config["datasets"]["global_transform_args"][
                "train_conditions"
            ]
        except (KeyError, TypeError):
            return None

        saw_known_condition = False
        active: set[str] = set()
        for condition_name, task_name in self._CONDITION_TO_TASK.items():
            condition = conditions.get(condition_name)
            if condition is None:
                continue
            saw_known_condition = True
            try:
                frequency = float(condition.get("frequency", 0.0))
            except (TypeError, ValueError):
                continue
            if frequency > 0 and task_name in self.task_metrics:
                active.add(task_name)
        if not saw_known_condition:
            return None
        return tuple(task for task in self.task_metrics if task in active)

    @staticmethod
    def _configured_validation_datasets(trainer: Any) -> set[str] | None:
        """Return configured validation loaders, if available in trainer state."""
        train_config = trainer.state.get("train_cfg")
        try:
            validation_datasets = train_config["datasets"]["val"]
        except (KeyError, TypeError):
            return None
        return {
            str(dataset)
            for dataset, dataset_config in validation_datasets.items()
            if dataset_config is not None
        }

    def on_validation_epoch_end(self, trainer: Any) -> None:
        """Compute scores after all dataset-level validation metrics are available."""
        if not self.enabled or not trainer.fabric.is_global_zero:
            return
        if not hasattr(trainer, "validation_results_path"):
            ranked_logger.warning(
                "Skipping composite validation metrics because "
                "validation_results_path is unavailable."
            )
            return

        dataset_metrics = self._latest_dataset_metrics(trainer)
        if not dataset_metrics:
            return

        task_scores: dict[str, float] = {}
        task_coverages: dict[str, float] = {}
        configured_datasets = self._configured_validation_datasets(trainer)
        for task_name, specs in self.task_metrics.items():
            expected_specs = tuple(
                spec
                for spec in specs
                if configured_datasets is None or spec["dataset"] in configured_datasets
            )
            if not expected_specs:
                task_coverages[task_name] = 0.0
                continue
            weighted_skill = 0.0
            available_weight = 0.0
            total_weight = sum(float(spec["weight"]) for spec in expected_specs)
            for spec in expected_specs:
                current_dataset_metrics = dataset_metrics.get(spec["dataset"], {})
                raw_value = current_dataset_metrics.get(spec["metric"])
                if raw_value is None:
                    continue
                if not self._metric_is_available(current_dataset_metrics, spec):
                    continue
                skill = self._metric_skill(raw_value, spec)
                if skill is None:
                    continue
                metric_weight = float(spec["weight"])
                weighted_skill += metric_weight * skill
                available_weight += metric_weight
            task_coverages[task_name] = available_weight / total_weight
            if available_weight > 0:
                task_scores[task_name] = weighted_skill / available_weight

        inferred_active_tasks = self._infer_active_tasks(trainer)
        active_tasks = (
            inferred_active_tasks
            if inferred_active_tasks is not None
            else tuple(task for task in self.task_metrics if task in task_scores)
        )
        if not active_tasks:
            return

        total_task_weight = sum(self.task_weights[task] for task in active_tasks)
        coverage = (
            sum(
                self.task_weights[task] * task_coverages.get(task, 0.0)
                for task in active_tasks
            )
            / total_task_weight
        )
        available_tasks = [task for task in active_tasks if task in task_scores]

        metrics_to_log: dict[str, float] = {
            f"{self.log_prefix}/coverage": float(coverage),
            f"{self.log_prefix}/active_task_count": float(len(active_tasks)),
            f"{self.log_prefix}/available_task_count": float(len(available_tasks)),
        }
        for task in active_tasks:
            metrics_to_log[f"{self.log_prefix}/task_coverage/{task}"] = float(
                task_coverages.get(task, 0.0)
            )
            if task in task_scores:
                metrics_to_log[f"{self.log_prefix}/task/{task}"] = float(
                    task_scores[task]
                )

        if available_tasks:
            available_weight = sum(self.task_weights[task] for task in available_tasks)
            available_score = np.exp(
                sum(
                    self.task_weights[task]
                    * np.log(max(task_scores[task], self.epsilon))
                    for task in available_tasks
                )
                / available_weight
            )
            metrics_to_log[f"{self.log_prefix}/available"] = float(available_score)
            metrics_to_log[f"{self.log_prefix}/worst_available"] = float(
                min(task_scores[task] for task in available_tasks)
            )
            if np.isclose(coverage, 1.0):
                metrics_to_log[f"{self.log_prefix}/full"] = float(available_score)

        trainer.fabric.log_dict(
            metrics_to_log,
            step=trainer.state.get("global_step"),
        )
