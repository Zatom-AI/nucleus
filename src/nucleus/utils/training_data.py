"""Construct the paper training datasets and deterministic source subsets."""

from pathlib import Path
from typing import Any

def build_train_dataset_and_sampler(cfg: Any, *, n_examples_per_epoch: int | None = None, fabric: Any = None) -> tuple[Any, Any]:
    """The original atomworks path, unchanged in behaviour."""
    from foundry.utils.datasets import recursively_instantiate_datasets_and_samplers

    from nucleus.utils.datasets import (
        subset_dataset_to_example_ids,
        subset_train_dataset_and_sampler_to_data_ids,
        subset_train_dataset_and_sampler_uniformly,
    )

    bundle = recursively_instantiate_datasets_and_samplers(cfg.datasets.train)
    dataset, sampler = bundle["dataset"], bundle["sampler"]

    data_id_subsets: dict[str, tuple[Any, Any]] = {}
    uniform_subsets: dict[str, tuple[int, int, int | None, str | None]] = {}
    for dataset_name, dataset_cfg in cfg.datasets.train.items():
        if dataset_cfg is None or not hasattr(dataset_cfg, "get"):
            continue
        subset_to_data_id = dataset_cfg.get("subset_to_data_id")
        if subset_to_data_id is not None:
            data_id_subsets[str(dataset_name)] = (
                subset_to_data_id,
                dataset_cfg.get("subset_to_data_id_cache_path"),
            )

        uniform_subset_size = dataset_cfg.get("uniform_subset_size")
        if uniform_subset_size is None:
            continue
        uniform_subset_seed = dataset_cfg.get("uniform_subset_seed")
        if uniform_subset_seed is None:
            raise ValueError(
                "datasets.train."
                f"{dataset_name}.uniform_subset_seed is required when "
                "uniform_subset_size is set."
            )
        uniform_subsets[str(dataset_name)] = (
            uniform_subset_size,
            uniform_subset_seed,
            (
                dataset_cfg.uniform_subset_source_size
                if dataset_cfg.get("uniform_subset_source_size") is not None
                else None
            ),
            (
                dataset_cfg.uniform_subset_expected_sha256
                if dataset_cfg.get("uniform_subset_expected_sha256") is not None
                else None
            ),
        )

    if data_id_subsets and uniform_subsets:
        raise ValueError(
            "Only one of datasets.train.<name>.subset_to_data_id or "
            "datasets.train.<name>.uniform_subset_size may be provided."
        )

    if "subset_to_example_ids" in cfg.datasets and (data_id_subsets or uniform_subsets):
        raise ValueError(
            "Only one of datasets.subset_to_example_ids, "
            "datasets.train.<name>.subset_to_data_id, or "
            "datasets.train.<name>.uniform_subset_size may be provided."
        )

    if data_id_subsets:
        cache_paths = [
            Path(cache_path) for _, cache_path in data_id_subsets.values() if cache_path
        ]
        missing = [path for path in cache_paths if not path.exists()]
        rank = getattr(fabric, "global_rank", 0) if fabric is not None else 0
        if missing and fabric is not None and rank != 0:
            fabric.barrier()

        dataset, sampler = subset_train_dataset_and_sampler_to_data_ids(
            dataset, sampler, data_id_subsets
        )

        if missing and fabric is not None and rank == 0:
            fabric.barrier()

    if uniform_subsets:
        dataset, sampler = subset_train_dataset_and_sampler_uniformly(
            dataset,
            sampler,
            uniform_subsets,
        )

    if "subset_to_example_ids" in cfg.datasets:
        # Backdoor for debugging and overfitting: subset to specific example IDs.
        dataset = subset_dataset_to_example_ids(
            dataset, cfg.datasets.subset_to_example_ids
        )
        # The sampler indexes the full dataset and is no longer valid.
        sampler = None

    return dataset, sampler
