import random

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from nucleus.trainers.fabric import FabricTrainer
from nucleus.utils.datasets import (
    SANITIZED_COORDINATE_NAN_COUNT_KEY,
    TokenBudgetDataLoader,
    _coalesce_worker_tensors,
    _CoalescedWorkerChunk,
    _sanitize_coordinate_nans,
    get_diffusion_expanded_num_tokens,
    pack_token_budget_batches,
)


def _example(example_id: str, num_tokens: int, diffusion_batch_size: int = 2):
    """Create a dummy example with the given number of tokens and diffusion batch size."""
    return {
        "example_id": example_id,
        "t": torch.zeros(diffusion_batch_size),
        "ground_truth": {"sequence_gt_I": torch.zeros(num_tokens, 32)},
    }


def _tensor_leaves(value):
    """Yield tensor leaves from a nested batch tree."""
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _tensor_leaves(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _tensor_leaves(child)


def test_coalesce_worker_tensors_preserves_values_with_one_storage_per_dtype():
    """Worker packing must be exact while consolidating tensor storages."""
    batch = [
        {
            "example_id": "a",
            "feats": {
                "float": torch.arange(6, dtype=torch.float32).view(2, 3),
                "bool": torch.tensor([True, False]),
            },
            "ground_truth": {
                "integer": torch.tensor([2, 3], dtype=torch.int64),
            },
            "metadata": np.asarray(["A", "B"]),
        },
        {
            "example_id": "b",
            "feats": {
                "float": torch.tensor(7.0),
                "bool": torch.tensor([False]),
            },
            "ground_truth": {
                "integer": torch.tensor([], dtype=torch.int64),
            },
            "metadata": np.asarray(["C"]),
        },
    ]

    packed = _coalesce_worker_tensors(batch)
    unpacked = packed.unpack()

    assert isinstance(packed, _CoalescedWorkerChunk)
    assert len(packed.buffers) == 3
    assert [example["example_id"] for example in unpacked] == ["a", "b"]
    for expected, actual in zip(batch, unpacked, strict=True):
        np.testing.assert_array_equal(actual["metadata"], expected["metadata"])
        for feature_group in ("feats", "ground_truth"):
            for feature_name, expected_tensor in expected[feature_group].items():
                torch.testing.assert_close(
                    actual[feature_group][feature_name],
                    expected_tensor,
                    rtol=0,
                    atol=0,
                )

    nonempty_storages = {
        tensor.untyped_storage().data_ptr()
        for tensor in _tensor_leaves(unpacked)
        if tensor.numel() > 0
    }
    assert len(nonempty_storages) == 3


def test_worker_coordinate_nan_sanitation_is_non_mutating_and_counted():
    """Expected unresolved coordinates must be cleaned before GPU transfer."""
    coordinate = torch.tensor([[float("nan"), 1.0, 2.0]])
    noise = torch.tensor([[0.0, float("nan"), 3.0]])
    example = {
        "coord_atom_lvl_to_be_noised": coordinate,
        "noise": noise,
    }

    sanitized = _sanitize_coordinate_nans([example])[0]

    assert sanitized is not example
    assert torch.isnan(example["coord_atom_lvl_to_be_noised"]).sum() == 1
    assert torch.isnan(example["noise"]).sum() == 1
    assert not torch.isnan(sanitized["coord_atom_lvl_to_be_noised"]).any()
    assert not torch.isnan(sanitized["noise"]).any()
    assert sanitized[SANITIZED_COORDINATE_NAN_COUNT_KEY] == 2


def test_token_budget_data_loader_accepts_coalesced_worker_chunks():
    """The token packer must transparently unpack IPC-efficient chunks."""
    dataset = [
        _example("a", 3),
        _example("b", 3),
        _example("c", 2),
        _example("d", 2),
    ]
    loader = TokenBudgetDataLoader(
        dataset,
        batch_size=4,
        worker_chunk_size=2,
        num_workers=2,
        prefetch_factor=1,
        shuffle=False,
        collate_fn=_coalesce_worker_tensors,
        max_num_tokens=10,
    )

    batches = list(loader)

    assert [[example["example_id"] for example in batch] for batch in batches] == [
        ["a", "c"],
        ["b", "d"],
    ]
    assert all(
        torch.is_tensor(example["ground_truth"]["sequence_gt_I"])
        for batch in batches
        for example in batch
    )


def test_get_diffusion_expanded_num_tokens():
    """Test that get_diffusion_expanded_num_tokens correctly computes the number of tokens after diffusion expansion."""
    assert get_diffusion_expanded_num_tokens(_example("a", 7, 4)) == 28


def test_pack_token_budget_batches_respects_every_rank():
    """Test that pack_token_budget_batches correctly packs batches while respecting the token budget for every rank."""
    packed = pack_token_budget_batches(
        token_counts_by_rank=[[6, 6, 4, 4], [4, 4, 6, 6]],
        max_num_tokens=10,
    )

    assert sorted(position for batch in packed for position in batch) == [0, 1, 2, 3]
    assert len(packed) == 2
    for batch in packed:
        assert sum([6, 6, 4, 4][position] for position in batch) <= 10
        assert sum([4, 4, 6, 6][position] for position in batch) <= 10


def test_token_budget_data_loader_packs_without_dropping_examples():
    """Test that TokenBudgetDataLoader correctly packs examples into batches without dropping any examples."""
    dataset = [
        _example("a", 3),
        _example("b", 3),
        _example("c", 2),
        _example("d", 2),
    ]
    loader = TokenBudgetDataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda batch: batch,
        max_num_tokens=10,
    )

    batches = list(loader)

    assert sorted(example["example_id"] for batch in batches for example in batch) == [
        "a",
        "b",
        "c",
        "d",
    ]
    assert len(batches) == 2
    assert all(
        sum(get_diffusion_expanded_num_tokens(example) for example in batch) <= 10
        for batch in batches
    )


def test_token_budget_data_loader_joins_worker_chunks_before_packing():
    """Worker task size must not change the candidate packing window."""
    dataset = [
        _example("a", 3),
        _example("b", 3),
        _example("c", 2),
        _example("d", 2),
    ]
    loader = TokenBudgetDataLoader(
        dataset,
        batch_size=4,
        worker_chunk_size=2,
        shuffle=False,
        collate_fn=lambda batch: batch,
        max_num_tokens=10,
    )

    batches = list(loader)

    assert [[example["example_id"] for example in batch] for batch in batches] == [
        ["a", "c"],
        ["b", "d"],
    ]


def test_token_budget_data_loader_keeps_incomplete_candidate_by_default():
    """An incomplete final candidate is emitted when drop_last is false."""
    dataset = [_example(str(index), 1, diffusion_batch_size=1) for index in range(12)]
    loader = TokenBudgetDataLoader(
        dataset,
        batch_size=8,
        worker_chunk_size=4,
        drop_last=False,
        shuffle=False,
        collate_fn=lambda batch: batch,
        max_num_tokens=100,
    )

    batches = list(loader)

    assert [len(batch) for batch in batches] == [8, 4]
    assert [batch[0]["_token_budget_is_last_batch"] for batch in batches] == [
        False,
        True,
    ]


def test_token_budget_data_loader_drops_incomplete_candidate():
    """drop_last applies to the candidate window, not each worker chunk."""
    dataset = [_example(str(index), 1, diffusion_batch_size=1) for index in range(12)]
    loader = TokenBudgetDataLoader(
        dataset,
        batch_size=8,
        worker_chunk_size=4,
        drop_last=True,
        shuffle=False,
        collate_fn=lambda batch: batch,
        max_num_tokens=100,
    )

    batches = list(loader)

    assert [len(batch) for batch in batches] == [8]
    assert batches[0][0]["_token_budget_is_last_batch"]
    assert [example["example_id"] for example in batches[0]] == [
        str(index) for index in range(8)
    ]


def test_token_budget_data_loader_rejects_misaligned_worker_chunk_size():
    """Candidate windows must contain a fixed integral number of worker chunks."""
    with pytest.raises(ValueError, match="must evenly divide batch_size"):
        TokenBudgetDataLoader(
            [_example("a", 2)],
            batch_size=4,
            worker_chunk_size=3,
            collate_fn=lambda batch: batch,
            max_num_tokens=10,
        )


@pytest.mark.parametrize("snapshot_every_n_steps", [1, 64, None])
def test_token_budget_chunked_worker_loader_resumes_pending_batches(
    snapshot_every_n_steps,
):
    """Chunked multiprocessing must preserve a partially emitted candidate."""
    dataset = [
        _example("a", 3),
        _example("b", 3),
        _example("c", 2),
        _example("d", 2),
        _example("e", 2),
    ]

    def make_loader():
        return TokenBudgetDataLoader(
            dataset,
            batch_size=4,
            worker_chunk_size=2,
            num_workers=2,
            prefetch_factor=1,
            shuffle=False,
            collate_fn=_coalesce_worker_tensors,
            max_num_tokens=10,
            snapshot_every_n_steps=snapshot_every_n_steps,
        )

    loader = make_loader()
    iterator = iter(loader)
    first_batch = next(iterator)
    state = loader.state_dict()
    expected_remaining_batches = list(iterator)

    resumed_loader = make_loader()
    resumed_loader.load_state_dict(state)
    resumed_batches = list(resumed_loader)

    assert [
        [example["example_id"] for example in batch] for batch in resumed_batches
    ] == [
        [example["example_id"] for example in batch]
        for batch in expected_remaining_batches
    ]
    assert sorted(
        example["example_id"]
        for batch in [first_batch, *resumed_batches]
        for example in batch
    ) == ["a", "b", "c", "d", "e"]


def test_token_budget_data_loader_rejects_oversized_example():
    """Test that TokenBudgetDataLoader raises an error when an example exceeds the token budget."""
    loader = TokenBudgetDataLoader(
        [_example("oversized", 6)],
        batch_size=1,
        collate_fn=lambda batch: batch,
        max_num_tokens=10,
    )

    with pytest.raises(ValueError, match="single example exceeds max_num_tokens"):
        list(loader)


def test_token_budget_data_loader_resumes_pending_packed_batches():
    """Test resume within a candidate window without dropping packed outputs."""
    dataset = [
        _example("a", 3),
        _example("b", 3),
        _example("c", 2),
        _example("d", 2),
        _example("e", 2),
    ]
    loader = TokenBudgetDataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda batch: batch,
        max_num_tokens=10,
    )
    iterator = iter(loader)
    first_batch = next(iterator)
    state = loader.state_dict()
    expected_remaining_batches = list(iterator)

    resumed_loader = TokenBudgetDataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda batch: batch,
        max_num_tokens=10,
    )
    resumed_loader.load_state_dict(state)
    resumed_batches = list(resumed_loader)

    assert [
        [example["example_id"] for example in batch] for batch in resumed_batches
    ] == [
        [example["example_id"] for example in batch]
        for batch in expected_remaining_batches
    ]
    assert sorted(
        example["example_id"]
        for batch in [first_batch, *resumed_batches]
        for example in batch
    ) == ["a", "b", "c", "d", "e"]
    assert resumed_batches[-1][0]["_token_budget_is_last_batch"]


class _StochasticTokenBudgetDataset(Dataset):
    """Dataset whose transformed examples consume every CPU RNG stream."""

    token_counts = (3, 3, 2, 2, 2)

    def __len__(self) -> int:
        return len(self.token_counts)

    def __getitem__(self, index: int) -> dict:
        return {
            "example_id": str(index),
            "t": torch.zeros(2),
            "ground_truth": {"sequence_gt_I": torch.zeros(self.token_counts[index])},
            "random_values": (
                random.random(),
                float(np.random.random()),
                float(torch.rand(())),
            ),
        }


def _stochastic_token_budget_loader() -> TokenBudgetDataLoader:
    """Return a loader with multiple packed outputs in its first candidate."""
    return TokenBudgetDataLoader(
        _StochasticTokenBudgetDataset(),
        batch_size=4,
        shuffle=False,
        collate_fn=lambda batch: batch,
        max_num_tokens=10,
        generator=torch.Generator().manual_seed(1234),
    )


def test_token_budget_resume_recreates_stochastic_candidate_exactly():
    """Candidate rewind must preserve stochastic transforms with no workers."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    loader = _stochastic_token_budget_loader()
    iterator = iter(loader)
    next(iterator)
    loader_state = loader.state_dict()
    process_rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    expected_remaining_batches = list(iterator)

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    resumed_loader = _stochastic_token_budget_loader()
    resumed_loader.load_state_dict(loader_state)
    random.setstate(process_rng_state["python"])
    np.random.set_state(process_rng_state["numpy"])
    torch.set_rng_state(process_rng_state["torch"])
    resumed_iterator = iter(resumed_loader)
    first_resumed_batch = next(resumed_iterator)
    second_loader_state = resumed_loader.state_dict()
    second_process_rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    expected_after_second_resume = list(resumed_iterator)
    resumed_batches = [first_resumed_batch, *expected_after_second_resume]

    expected = [
        (example["example_id"], example["random_values"])
        for batch in expected_remaining_batches
        for example in batch
    ]
    actual = [
        (example["example_id"], example["random_values"])
        for batch in resumed_batches
        for example in batch
    ]
    assert actual == expected

    twice_resumed_loader = _stochastic_token_budget_loader()
    twice_resumed_loader.load_state_dict(second_loader_state)
    random.setstate(second_process_rng_state["python"])
    np.random.set_state(second_process_rng_state["numpy"])
    torch.set_rng_state(second_process_rng_state["torch"])
    twice_resumed = [
        (example["example_id"], example["random_values"])
        for batch in twice_resumed_loader
        for example in batch
    ]
    expected_twice_resumed = [
        (example["example_id"], example["random_values"])
        for batch in expected_after_second_resume
        for example in batch
    ]
    assert twice_resumed == expected_twice_resumed


class _VariableLengthTrainer(FabricTrainer):
    """A subclass of FabricTrainer that records training calls for testing purposes."""

    def training_step(self, batch, batch_idx: int, is_accumulating: bool) -> None:
        """Record the training step call for testing purposes."""
        self.training_calls.append((batch, batch_idx, is_accumulating))
        self._current_train_return = {}

    def validation_step(self, batch, batch_idx: int, val_loader_name=None) -> dict:
        """Not implemented for this test subclass."""
        raise NotImplementedError

    def step_optimizer(self):
        """Increment the optimizer step count for testing purposes."""
        self.optimizer_steps += 1

    def step_scheduler(self, level, current_value):
        """Not implemented for this test subclass."""
        pass


class _RecordingFabric:
    """A mock Fabric class that records calls for testing purposes."""

    def call(self, hook_name: str, **kwargs) -> None:
        pass


class _OverestimatingLoader:
    """Mimic a dynamic loader whose safe length bound exceeds its output."""

    max_num_tokens = 10

    def __len__(self):
        return 10

    def __iter__(self):
        yield [{"name": "batch-0", "_token_budget_is_last_batch": False}]
        yield [{"name": "batch-1", "_token_budget_is_last_batch": False}]
        yield [{"name": "batch-2", "_token_budget_is_last_batch": True}]


def test_train_loop_handles_variable_batch_count_and_steps_partial_accumulation():
    """Test that the train loop correctly handles variable batch counts and partial accumulation."""
    trainer = object.__new__(_VariableLengthTrainer)
    trainer.fabric = _RecordingFabric()
    trainer.state = {
        "model": type("Model", (), {"training": True})(),
        "optimizer": object(),
        "global_step": 0,
    }
    trainer.should_stop = False
    trainer.grad_accum_steps = 2
    trainer.checkpoint_every_n_steps = None
    trainer.training_calls = []
    trainer.optimizer_steps = 0

    trainer.train_loop(train_loader=_OverestimatingLoader())

    assert trainer.training_calls == [
        ([{"name": "batch-0", "_token_budget_is_last_batch": False}], 0, True),
        ([{"name": "batch-1", "_token_budget_is_last_batch": False}], 1, False),
        ([{"name": "batch-2", "_token_budget_is_last_batch": True}], 2, False),
    ]
    assert trainer.optimizer_steps == 2
    assert trainer.state["global_step"] == 2
