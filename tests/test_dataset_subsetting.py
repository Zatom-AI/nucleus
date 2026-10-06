"""Regression tests for deterministic dataset subsetting."""

import hashlib

import numpy as np
import pytest
import torch
from atomworks.ml.datasets import ConcatDatasetWithID
from atomworks.ml.samplers import DistributedMixedSampler, MixedSampler
from torch.utils.data import Dataset, RandomSampler, WeightedRandomSampler

from nucleus.utils.datasets import (
    subset_dataset_and_sampler_uniformly,
    subset_dataset_to_example_ids,
    subset_train_dataset_and_sampler_uniformly,
    uniform_subset_indices,
)


class _IDDataset(Dataset):
    def __init__(self, prefix: str, values: list[str]):
        self.prefix = prefix
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> str:
        return self.values[index]

    def __contains__(self, example_id: str) -> bool:
        return example_id.startswith(f"{self.prefix}:")

    def id_to_idx(self, example_id: str) -> int:
        return int(example_id.split(":", maxsplit=1)[1])

    def idx_to_id(self, index: int) -> str:
        return f"{self.prefix}:{index}"


def test_subset_concatenated_dataset_uses_global_indices_without_loading_rows():
    """An ID in the second child resolves to its concatenated global offset."""
    first = _IDDataset("first", ["a", "b", "c"])
    second = _IDDataset("second", ["d", "e"])
    dataset = ConcatDatasetWithID([first, second])

    subset = subset_dataset_to_example_ids(dataset, ["first:1", "second:0"])

    assert subset.indices == [1, 3]
    assert [subset[index] for index in range(len(subset))] == ["b", "d"]


def test_uniform_subset_indices_are_fixed_unique_and_without_replacement():
    """PCG64 selection is stable and independent of source iteration order."""
    indices = uniform_subset_indices(dataset_length=10, subset_size=4, seed=42)

    assert indices == [0, 4, 5, 6]
    assert len(indices) == len(set(indices)) == 4
    assert indices == sorted(indices)


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("dataset_length", {"dataset_length": 10.5}),
        ("subset_size", {"subset_size": True}),
        ("seed", {"seed": 42.5}),
    ],
)
def test_uniform_subset_rejects_non_integral_inputs(field, kwargs):
    arguments = {"dataset_length": 10, "subset_size": 4, "seed": 42}
    arguments.update(kwargs)

    with pytest.raises(TypeError, match=rf"{field} must be an integer"):
        uniform_subset_indices(**arguments)


@pytest.mark.parametrize(
    ("dataset_length", "expected_sha256"),
    [
        (
            3_986_754,
            "36b7366932ae1dbe3941743373610a063018697a2f2cd16664efbca114bdf01e",
        ),
        (
            1_009_850,
            "73d46f5304147e419a81c6df15f9a0fa628c199a525566637bf2b6fc887a5227",
        ),
    ],
)
def test_figure1_uniform_subset_hashes_are_canonical(
    dataset_length,
    expected_sha256,
):
    indices = uniform_subset_indices(dataset_length, subset_size=500_000, seed=42)
    encoded = np.asarray(indices, dtype="<i8").tobytes()

    assert hashlib.sha256(encoded).hexdigest() == expected_sha256


def test_uniform_subset_rejects_unexpected_source_cardinality():
    dataset = _IDDataset("first", [f"a{i}" for i in range(10)])

    with pytest.raises(ValueError, match="expected 9 examples, found 10"):
        subset_dataset_and_sampler_uniformly(
            dataset,
            RandomSampler(dataset),
            subset_size=4,
            seed=42,
            expected_source_size=9,
        )


def test_uniform_subset_rejects_unexpected_index_hash():
    dataset = _IDDataset("first", [f"a{i}" for i in range(10)])

    with pytest.raises(ValueError, match="index hash mismatch"):
        subset_dataset_and_sampler_uniformly(
            dataset,
            RandomSampler(dataset),
            subset_size=4,
            seed=42,
            expected_source_size=10,
            expected_indices_sha256="0" * 64,
        )


def test_uniform_subsets_preserve_mixed_dataset_domains_and_probabilities():
    first = _IDDataset("first", [f"a{i}" for i in range(10)])
    second = _IDDataset("second", [f"b{i}" for i in range(8)])
    dataset = ConcatDatasetWithID([first, second])
    sampler = MixedSampler(
        datasets_info=[
            {
                "name": "first",
                "dataset": first,
                "sampler": WeightedRandomSampler(
                    torch.ones(len(first)),
                    num_samples=len(first),
                    replacement=True,
                ),
                "probability": 0.5,
            },
            {
                "name": "second",
                "dataset": second,
                "sampler": WeightedRandomSampler(
                    torch.ones(len(second)),
                    num_samples=len(second),
                    replacement=True,
                ),
                "probability": 0.5,
            },
        ]
    )

    subset_dataset, subset_sampler = subset_train_dataset_and_sampler_uniformly(
        dataset,
        sampler,
        {
            "first": (
                4,
                42,
                10,
                "3bc03673ab71dd031825485232fbe8c65dd71e697724506149931d356f61bdef",
            ),
            "second": (
                4,
                42,
                8,
                "f4c1bc7747acac569b363805f1f95a8288283fc0cc86057fe007b45a6a2865dd",
            ),
        },
    )

    assert len(subset_dataset) == 8
    assert isinstance(subset_sampler, MixedSampler)
    assert [info["probability"] for info in subset_sampler.datasets_info] == [
        0.5,
        0.5,
    ]
    assert [len(info["dataset"]) for info in subset_sampler.datasets_info] == [4, 4]
    assert subset_sampler.datasets_info[0]["dataset"].indices == [0, 4, 5, 6]
    assert subset_sampler.datasets_info[1]["dataset"].indices == [0, 3, 4, 6]

    rank_indices = []
    for rank in range(2):
        distributed_sampler = DistributedMixedSampler(
            datasets_info=subset_sampler.datasets_info,
            num_replicas=2,
            rank=rank,
            n_examples_per_epoch=20,
            shuffle=True,
            drop_last=True,
        )
        rank_indices.extend(distributed_sampler)

    assert len(rank_indices) == 20
    assert all(0 <= index < len(subset_dataset) for index in rank_indices)
    assert sum(index < 4 for index in rank_indices) == 10
    assert sum(index >= 4 for index in rank_indices) == 10
    assert len(set(rank_indices)) < len(rank_indices)
