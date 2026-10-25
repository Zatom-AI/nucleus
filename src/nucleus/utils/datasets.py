import hashlib
import importlib
import os
import random
import signal
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property, partial
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
from atomworks.ml.datasets import ConcatDatasetWithID
from atomworks.ml.samplers import DistributedMixedSampler, MixedSampler
from foundry.utils.datasets import wrap_dataset_and_sampler_with_fallbacks
from foundry.utils.ddp import RankedLogger
from omegaconf import DictConfig
from rfd3.inference.input_parsing import DesignInputSpecification
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
    Subset,
    WeightedRandomSampler,
)
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader import Stateful, StatefulDataLoader

from nucleus.inference_input_parsing import (
    _coerce_zatom_design_specification_for,
    install_molecule_contig_support,
    install_molecule_contig_support_for,
    rewrite_inputless_generated_contig,
)
from nucleus.utils.bond_sidecar import (
    BondSidecarASEAtomsLoader,
    perceive_ligand_bonds_from_native_coordinates,
)

ranked_logger = RankedLogger(__name__, rank_zero_only=True)
# A stalled example is rank-local, so its report must not be rank-zero only.
all_ranks_logger = RankedLogger(__name__, rank_zero_only=False)

_TOKEN_BUDGET_LAST_BATCH_KEY = "_token_budget_is_last_batch"
SANITIZED_COORDINATE_NAN_COUNT_KEY = "_sanitized_coordinate_nan_count"




def _rewrite_inputless_ligand_contig(spec):
    """Backward-compatible alias for generated-only contig rewriting."""
    return rewrite_inputless_generated_contig(spec)


def _coerce_ligand_design_specification(spec):
    """Validate ligand contig specs with the Zatom schema when available."""
    try:
        from nucleus.inference_input_parsing import ZatomDesignInputSpecification
    except ModuleNotFoundError as exc:
        if exc.name not in {"nucleus", "nucleus.inference_input_parsing"}:
            raise
        specification_cls = DesignInputSpecification
    else:
        specification_cls = ZatomDesignInputSpecification

    return specification_cls.safe_init(**spec)


class ZatomContigJsonDataset:
    """Contig JSON dataset with Nucleus typed generated-contig support.

    When ``bond_sidecar_path`` is configured, a specification's ``example_id``
    selects the corresponding sidecar LMDB record, falling back to the outer
    JSON key when omitted. Without a sidecar, bonds for fully resolved
    non-material ligand inputs are perceived from preserved native coordinates
    using the same Open Babel logic that constructs the sidecar.
    """

    def __init__(
        self,
        *args,
        parser_backend: str = "rfd3",
        bond_sidecar_path: str | Path | None = None,
        **kwargs,
    ):
        if parser_backend == "rfd3":
            install_molecule_contig_support()
        else:
            install_molecule_contig_support_for(parser_backend)

        datasets_module = importlib.import_module(
            f"{parser_backend}.inference.datasets"
        )
        input_parsing_module = importlib.import_module(
            f"{parser_backend}.inference.input_parsing"
        )
        self._specification_cls = input_parsing_module.DesignInputSpecification
        self._ensure_input_is_abspath = input_parsing_module.ensure_input_is_abspath
        self._dataset = datasets_module.ContigJsonDataset(*args, **kwargs)
        self._bond_sidecar_loader = (
            BondSidecarASEAtomsLoader(bond_sidecar_path=bond_sidecar_path)
            if bond_sidecar_path is not None
            else None
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying dataset."""
        return getattr(self._dataset, name)

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self._dataset)

    def __contains__(self, example_id: str) -> bool:
        """Check if an example ID is in the dataset."""
        return example_id in self._dataset

    def __getitem__(self, idx: int) -> Any:
        """Return the parsed pipeline input for the example at the given index."""
        example_id = self._dataset.idx_to_id(idx)
        spec = self._dataset.data[example_id]

        if not isinstance(spec, self._specification_cls):
            spec = self._ensure_input_is_abspath(spec, self._dataset.json_path)
            spec = _rewrite_inputless_ligand_contig(spec)
            spec["cif_parser_args"] = self._dataset.cif_parser_args
            spec = _coerce_zatom_design_specification_for(spec, self._specification_cls)

        source_example_id = getattr(spec, "example_id", None) or example_id
        data = spec.to_pipeline_input(example_id=example_id)
        native_coordinates = getattr(spec, "_nucleus_native_coordinates", None)
        if self._bond_sidecar_loader is not None:
            data["has_disconnected_fragments"] = self._bond_sidecar_loader.attach_bonds(
                data["atom_array"],
                example_id=str(source_example_id),
            )
        elif "has_disconnected_fragments" not in data:
            # The rfd3 backend already perceives bonds in to_pipeline_input.
            has_disconnected_fragments = perceive_ligand_bonds_from_native_coordinates(
                data["atom_array"],
                native_coordinates,
                example_id=str(source_example_id),
            )
            if has_disconnected_fragments is not None:
                data["has_disconnected_fragments"] = has_disconnected_fragments
        data = self._dataset.transform(data)

        if native_coordinates is not None and "ground_truth" in data:
            native_coordinates = torch.from_numpy(native_coordinates)
            valid_coordinates = torch.isfinite(native_coordinates).all(dim=-1)
            data["ground_truth"]["coord_atom_lvl"] = torch.nan_to_num(
                native_coordinates
            )
            data["ground_truth"]["mask_atom_lvl"] &= valid_coordinates
        return data


class LigandContigJsonDataset(ZatomContigJsonDataset):
    """Backward-compatible name for :class:`ZatomContigJsonDataset`."""


def _sanitize_coordinate_nans(batch: list[Any]) -> list[Any]:
    """Replace expected coordinate NaNs before worker-to-trainer transfer.

    Some cropped structures contain unresolved coordinates whose losses are
    masked. Sanitizing them in loader workers avoids synchronous GPU NaN checks
    and per-batch exception logging in the training process.
    """
    sanitized_batch = []
    for value in batch:
        if not isinstance(value, dict):
            sanitized_batch.append(value)
            continue

        example = value
        nan_count = 0
        for key in ("coord_atom_lvl_to_be_noised", "noise"):
            tensor = example.get(key)
            if not torch.is_tensor(tensor) or not (
                tensor.is_floating_point() or tensor.is_complex()
            ):
                continue
            nan_mask = torch.isnan(tensor)
            current_nan_count = int(nan_mask.sum())
            nan_count += current_nan_count
            if current_nan_count:
                if example is value:
                    example = dict(value)
                example[key] = torch.nan_to_num(tensor)

        if example is value:
            example = dict(value)
        example[SANITIZED_COORDINATE_NAN_COUNT_KEY] = nan_count
        sanitized_batch.append(example)
    return sanitized_batch


def _identity_collate(batch, *, sanitize_coordinate_nans: bool = False):
    """Return the batch unchanged, optionally sanitizing coordinate NaNs."""
    return _sanitize_coordinate_nans(batch) if sanitize_coordinate_nans else batch


@dataclass(frozen=True)
class _CoalescedTensorView:
    """Metadata needed to reconstruct one tensor from a worker buffer."""

    buffer_index: int
    offset: int
    numel: int
    shape: tuple[int, ...]


@dataclass
class _CoalescedWorkerChunk:
    """IPC-efficient worker output with tensor leaves grouped by dtype."""

    encoded_batch: list[Any]
    buffers: tuple[torch.Tensor, ...]

    def unpack(self) -> list[Any]:
        """Reconstruct the original batch tree using views into shared buffers."""

        def decode(value: Any) -> Any:
            if isinstance(value, _CoalescedTensorView):
                flat_buffer = self.buffers[value.buffer_index]
                return flat_buffer.narrow(0, value.offset, value.numel).view(
                    value.shape
                )
            if isinstance(value, dict):
                return {key: decode(child) for key, child in value.items()}
            if isinstance(value, list):
                return [decode(child) for child in value]
            if isinstance(value, tuple):
                return tuple(decode(child) for child in value)
            return value

        return decode(self.encoded_batch)


def _coalesce_worker_tensors(
    batch: list[Any], *, sanitize_coordinate_nans: bool = False
) -> _CoalescedWorkerChunk:
    """Pack all CPU tensor leaves into one contiguous buffer per dtype.

    PyTorch multiprocessing otherwise transfers every tensor storage in every
    example separately. Atomized-ligand examples contain many small tensors,
    so coalescing them before worker IPC substantially reduces shared-memory
    mappings while preserving the list-of-dictionaries interface after unpack.
    """
    if sanitize_coordinate_nans:
        batch = _sanitize_coordinate_nans(batch)

    dtype_to_buffer_index: dict[torch.dtype, int] = {}
    flat_tensors_by_buffer: list[list[torch.Tensor]] = []
    next_offsets: list[int] = []

    def encode(value: Any) -> Any:
        if torch.is_tensor(value):
            if value.device.type != "cpu":
                raise ValueError(
                    "Worker tensor coalescing only supports CPU tensors, got "
                    f"{value.device}."
                )
            if value.layout != torch.strided:
                raise ValueError(
                    "Worker tensor coalescing only supports strided tensors, got "
                    f"{value.layout}."
                )

            buffer_index = dtype_to_buffer_index.get(value.dtype)
            if buffer_index is None:
                buffer_index = len(flat_tensors_by_buffer)
                dtype_to_buffer_index[value.dtype] = buffer_index
                flat_tensors_by_buffer.append([])
                next_offsets.append(0)

            flat_tensor = value.detach().contiguous().reshape(-1)
            offset = next_offsets[buffer_index]
            flat_tensors_by_buffer[buffer_index].append(flat_tensor)
            next_offsets[buffer_index] += flat_tensor.numel()
            return _CoalescedTensorView(
                buffer_index=buffer_index,
                offset=offset,
                numel=flat_tensor.numel(),
                shape=tuple(value.shape),
            )
        if isinstance(value, dict):
            return {key: encode(child) for key, child in value.items()}
        if isinstance(value, list):
            return [encode(child) for child in value]
        if isinstance(value, tuple):
            return tuple(encode(child) for child in value)
        return value

    encoded_batch = encode(batch)
    buffers = tuple(torch.cat(flat_tensors) for flat_tensors in flat_tensors_by_buffer)
    return _CoalescedWorkerChunk(encoded_batch=encoded_batch, buffers=buffers)


def get_diffusion_expanded_num_tokens(example: dict[str, Any]) -> int:
    """Return the number of tokens represented across diffusion samples.

    The dominant token-axis activations have a leading diffusion dimension, so
    the batching cost is ``diffusion_batch_size * num_tokens`` rather than only
    the structural token count.
    """
    try:
        diffusion_batch_size = int(example["t"].shape[0])
        num_tokens = int(example["ground_truth"]["sequence_gt_I"].shape[0])
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(
            "Token-budgeted batching requires each example to contain `t` and "
            "`ground_truth.sequence_gt_I` tensors."
        ) from exc

    if diffusion_batch_size <= 0 or num_tokens <= 0:
        raise ValueError(
            "Token-budgeted batching requires positive diffusion and token "
            f"dimensions, got {diffusion_batch_size=} and {num_tokens=}."
        )
    return diffusion_batch_size * num_tokens


def pack_token_budget_batches(
    token_counts_by_rank: list[list[int]], max_num_tokens: int
) -> list[list[int]]:
    """Pack aligned per-rank examples without exceeding a token budget.

    Each position denotes one sampler output on every distributed rank. The
    same positions must be assigned to each batch on all ranks to keep DDP
    collective calls aligned. First-fit decreasing provides deterministic,
    reasonably dense packing while checking the budget independently on every
    rank.

    Args:
        token_counts_by_rank: Expanded token counts shaped ``[world_size, B]``.
        max_num_tokens: Maximum expanded tokens in a batch on any rank.

    Returns:
        Lists of example positions, one list per packed batch.
    """
    if max_num_tokens <= 0:
        raise ValueError(f"max_num_tokens must be positive, got {max_num_tokens}.")
    if not token_counts_by_rank:
        return []

    batch_size = len(token_counts_by_rank[0])
    if any(len(counts) != batch_size for counts in token_counts_by_rank):
        raise ValueError("All distributed ranks must provide the same batch size.")
    if any(count <= 0 for counts in token_counts_by_rank for count in counts):
        raise ValueError("All expanded token counts must be positive.")

    largest_count = max(max(counts) for counts in token_counts_by_rank)
    if largest_count > max_num_tokens:
        raise ValueError(
            "A single example exceeds max_num_tokens: "
            f"expanded_num_tokens={largest_count}, max_num_tokens={max_num_tokens}. "
            "Reduce datasets.diffusion_batch_size_train or increase "
            "datasets.max_num_tokens."
        )

    world_size = len(token_counts_by_rank)
    positions = sorted(
        range(batch_size),
        key=lambda position: max(
            token_counts_by_rank[rank][position] for rank in range(world_size)
        ),
        reverse=True,
    )
    packed_positions: list[list[int]] = []
    packed_token_counts: list[list[int]] = []

    for position in positions:
        for positions_in_batch, counts_in_batch in zip(
            packed_positions, packed_token_counts
        ):
            if all(
                counts_in_batch[rank] + token_counts_by_rank[rank][position]
                <= max_num_tokens
                for rank in range(world_size)
            ):
                positions_in_batch.append(position)
                for rank in range(world_size):
                    counts_in_batch[rank] += token_counts_by_rank[rank][position]
                break
        else:
            packed_positions.append([position])
            packed_token_counts.append(
                [token_counts_by_rank[rank][position] for rank in range(world_size)]
            )

    return packed_positions


def _get_sampler_generators(sampler: Sampler) -> list[torch.Generator]:
    """Return dedicated generators reachable through a sampler tree."""
    generators: list[torch.Generator] = []
    visited = set()

    def collect_generators(current_sampler: Any) -> None:
        if id(current_sampler) in visited:
            return
        visited.add(id(current_sampler))

        generator = getattr(current_sampler, "generator", None)
        if isinstance(generator, torch.Generator):
            generators.append(generator)

        for attribute in ("sampler", "fallback_sampler"):
            child_sampler = getattr(current_sampler, attribute, None)
            if child_sampler is not None:
                collect_generators(child_sampler)
        for child_sampler in getattr(current_sampler, "samplers", ()):
            collect_generators(child_sampler)

    collect_generators(sampler)
    return generators


def _capture_process_rng_state() -> dict[str, Any]:
    """Capture process RNG state used by samplers and main-process transforms."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    for backend_name in ("cuda", "mps", "xpu"):
        backend = getattr(torch, backend_name, None)
        if backend is None or not backend.is_available():
            continue
        get_rng_state_all = getattr(backend, "get_rng_state_all", None)
        state[backend_name] = (
            get_rng_state_all()
            if get_rng_state_all is not None
            else backend.get_rng_state()
        )
    return state


def _restore_process_rng_state(state: dict[str, Any]) -> None:
    """Restore process RNG state captured by :func:`_capture_process_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    for backend_name in ("cuda", "mps", "xpu"):
        if backend_name not in state:
            continue
        backend = getattr(torch, backend_name, None)
        if backend is None or not backend.is_available():
            ranked_logger.warning(
                f"Checkpoint contains {backend_name.upper()} RNG state, but that backend is unavailable; accelerator RNG state was not restored."
            )
            continue
        set_rng_state_all = getattr(backend, "set_rng_state_all", None)
        if set_rng_state_all is not None:
            set_rng_state_all(state[backend_name])
        else:
            backend.set_rng_state(state[backend_name])


def _capture_sampler_rng_state(sampler: Sampler) -> dict[str, Any]:
    """Capture RNG state used while materializing a sampler epoch."""
    return {
        "process": _capture_process_rng_state(),
        "generators": [
            generator.get_state() for generator in _get_sampler_generators(sampler)
        ],
    }


def _restore_sampler_rng_state(sampler: Sampler, state: dict[str, Any]) -> None:
    """Restore RNG state used while materializing a sampler epoch."""
    _restore_process_rng_state(state["process"])

    generators = _get_sampler_generators(sampler)
    generator_states = state["generators"]
    if len(generators) != len(generator_states):
        raise ValueError(
            "Sampler generator structure changed while restoring dataloader state: "
            f"checkpoint has {len(generator_states)} generators, current sampler has "
            f"{len(generators)}."
        )
    for generator, generator_state in zip(generators, generator_states, strict=True):
        generator.set_state(generator_state)


class _StatefulSamplerIterator(Iterator):
    """
    Sampler iterator that can regenerate and seek within one sampled epoch.

    Args:
        sampler (Sampler): The underlying sampler to wrap. Must be finite and
            deterministic given its RNG state.
    """

    def __init__(self, sampler: Sampler) -> None:
        self.sampler = sampler
        self.iterator: Iterator | None = None
        self.initial_rng_state: dict[str, Any] | None = None
        self.num_yielded = 0

    def __iter__(self) -> "_StatefulSamplerIterator":
        """Return self as an iterator."""
        return self

    def _initialize_iterator(self) -> None:
        """Initialize the sampler iterator, restoring RNG state and cursor if needed."""
        if self.iterator is not None:
            return

        if self.initial_rng_state is None:
            self.initial_rng_state = _capture_sampler_rng_state(self.sampler)
            self.iterator = iter(self.sampler)
            return

        current_rng_state = _capture_sampler_rng_state(self.sampler)
        try:
            _restore_sampler_rng_state(self.sampler, self.initial_rng_state)
            self.iterator = iter(self.sampler)
            for _ in range(self.num_yielded):
                next(self.iterator)
            restored_generator_states = [
                generator.get_state()
                for generator in _get_sampler_generators(self.sampler)
            ]
        finally:
            _restore_sampler_rng_state(self.sampler, current_rng_state)
        generators = _get_sampler_generators(self.sampler)
        if len(generators) != len(restored_generator_states):
            raise ValueError(
                "Sampler generator structure changed while restoring its iterator."
            )
        for generator, generator_state in zip(
            generators, restored_generator_states, strict=True
        ):
            generator.set_state(generator_state)

    def __next__(self) -> Any:
        """Return the next index from the sampler iterator, initializing it if needed."""
        self._initialize_iterator()
        assert self.iterator is not None
        index = next(self.iterator)
        self.num_yielded += 1
        return index

    def state_dict(self) -> dict[str, Any]:
        """Return enough state to regenerate this epoch and seek to its cursor."""
        self._initialize_iterator()
        return {
            "initial_rng_state": self.initial_rng_state,
            "num_yielded": self.num_yielded,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the sampled epoch and cursor lazily on the next access."""
        self.iterator = None
        self.initial_rng_state = state_dict["initial_rng_state"]
        self.num_yielded = int(state_dict["num_yielded"])


class StatefulSampler(Sampler):
    """Adapter that makes an arbitrary finite sampler iterator checkpointable.

    Checkpoints retain RNG state and a cursor. Restoring replays only the sampler
    iterator—not dataset loading or transforms—to reach that cursor.

    Args:
        sampler (Sampler): The underlying sampler to wrap. Must be finite and
            deterministic given its RNG state.
    """

    def __init__(self, sampler: Sampler) -> None:
        self.sampler = sampler

    def __iter__(self) -> _StatefulSamplerIterator:
        """Return a checkpointable iterator over the sampler."""
        return _StatefulSamplerIterator(self.sampler)

    def __len__(self) -> int:
        """Return the number of samples in the sampler."""
        return len(self.sampler)

    def set_epoch(self, epoch: int) -> None:
        """Forward the distributed epoch to the wrapped sampler."""
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


class ExampleLoadTimeoutError(RuntimeError):
    """Raised when loading a single example exceeds its wall-clock budget."""


class TimeBoundedDataset(Dataset):
    """Dataset adapter that bounds the wall-clock cost of one example.

    A single pathological structure otherwise blocks its dataloader worker, and
    therefore its DDP rank, until the remaining ranks time out inside a
    collective. Raising instead lets the surrounding ``FallbackDatasetWrapper``
    retry a different example and records the offending ID.

    The deadline uses ``SIGALRM``, delivered to the process' main thread, where
    dataloader workers load examples. It repeats after expiring, because
    library code that catches broad exceptions would otherwise absorb the
    single deadline and resume waiting.

    Args:
        dataset (Dataset): The dataset to wrap.
        timeout_seconds (float): Per-example wall-clock budget in seconds.
        retry_seconds (float): Interval at which an absorbed deadline repeats.
    """

    def __init__(
        self,
        dataset: Dataset,
        timeout_seconds: float,
        retry_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive; got {timeout_seconds}."
            )
        self.dataset = dataset
        self.timeout_seconds = float(timeout_seconds)
        self.retry_seconds = float(retry_seconds)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying dataset."""
        try:
            dataset = object.__getattribute__(self, "dataset")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        return getattr(dataset, name)

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.dataset)

    def _describe(self, index: Any) -> str:
        """Return a human-readable identifier for an index, if available."""
        idx_to_id = getattr(self.dataset, "idx_to_id", None)
        if idx_to_id is None:
            return f"index {index}"
        try:
            return f"index {index} ({idx_to_id(index)})"
        except Exception:  # Identification must never mask the original error.
            return f"index {index}"

    def __getitem__(self, index: Any) -> Any:
        """Return the example at ``index`` within the configured time budget."""

        def _on_timeout(signum: int, frame: Any) -> None:
            del signum, frame
            raise ExampleLoadTimeoutError(
                f"Loading {self._describe(index)} exceeded "
                f"{self.timeout_seconds:g}s; falling back to another example."
            )

        try:
            previous_handler = signal.signal(signal.SIGALRM, _on_timeout)
        except ValueError:  # not this process' main thread: cannot enforce
            return self.dataset[index]

        previous_timer = signal.setitimer(
            signal.ITIMER_REAL, self.timeout_seconds, self.retry_seconds
        )
        try:
            return self.dataset[index]
        except ExampleLoadTimeoutError as error:
            all_ranks_logger.error(str(error))
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
            signal.signal(signal.SIGALRM, previous_handler)


class _WorkerRNGStateDataset(Dataset):
    """
    Dataset adapter that includes dataloader-worker RNG in loader snapshots.

    Args:
        dataset (Dataset): The underlying dataset to wrap. Must be finite and
            deterministic given its RNG state.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying dataset."""
        try:
            dataset = object.__getattribute__(self, "dataset")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        return getattr(dataset, name)

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, index: Any) -> Any:
        """Return the example at the given index."""
        return self.dataset[index]

    def state_dict(self) -> dict[str, Any]:
        """Return underlying dataset state and worker-local RNG state."""
        state: dict[str, Any] = {}
        if isinstance(self.dataset, Stateful):
            state["dataset"] = self.dataset.state_dict()

        if torch.utils.data.get_worker_info() is not None:
            state["worker_rng"] = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore underlying dataset state and worker-local RNG state."""
        if "dataset" in state_dict:
            if not isinstance(self.dataset, Stateful):
                raise ValueError(
                    "Checkpoint contains dataset state, but the current dataset is not stateful."
                )
            self.dataset.load_state_dict(state_dict["dataset"])

        worker_rng_state = state_dict.get("worker_rng")
        if worker_rng_state is not None:
            random.setstate(worker_rng_state["python"])
            np.random.set_state(worker_rng_state["numpy"])
            torch.set_rng_state(worker_rng_state["torch"])


class TokenBudgetDataLoader(StatefulDataLoader):
    """DataLoader that packs transformed examples up to an expanded-token cap.

    The ordinary DataLoader first loads a candidate window whose size is set by
    ``batch_size``. This iterator then packs that window using actual transformed
    token counts. Candidate windows remain bounded, so worker prefetching and
    fallback behavior are unchanged.

    Args:
        max_num_tokens (int): Maximum number of expanded tokens in a packed batch.
            Must be positive.
        worker_chunk_size (int | None): Number of examples transformed in one
            worker task. When smaller than ``batch_size``, multiple worker
            results are joined before token-budget packing. This keeps the
            candidate packing window unchanged while reducing the shared-memory
            footprint of each prefetched worker result.
    """

    def __init__(
        self,
        *args,
        max_num_tokens: int,
        worker_chunk_size: int | None = None,
        **kwargs,
    ):
        if max_num_tokens <= 0:
            raise ValueError(f"max_num_tokens must be positive, got {max_num_tokens}.")
        candidate_batch_size = kwargs.get("batch_size")
        if candidate_batch_size is None:
            raise ValueError("TokenBudgetDataLoader requires an explicit batch_size.")
        if worker_chunk_size is None:
            worker_chunk_size = candidate_batch_size
        if worker_chunk_size <= 0:
            raise ValueError(
                f"worker_chunk_size must be positive, got {worker_chunk_size}."
            )
        if candidate_batch_size % worker_chunk_size != 0:
            raise ValueError(
                "worker_chunk_size must evenly divide batch_size, got "
                f"{worker_chunk_size=} and {candidate_batch_size=}."
            )

        self.max_num_tokens = int(max_num_tokens)
        self.candidate_batch_size = int(candidate_batch_size)
        self.worker_chunk_size = int(worker_chunk_size)
        self._worker_chunks_per_candidate = (
            self.candidate_batch_size // self.worker_chunk_size
        )
        self._candidate_start_state: dict[str, Any] | None = None
        self._candidate_start_rng_state: dict[str, Any] | None = None
        self._candidate_batch_idx = 0
        self._next_candidate_batch_idx = 0
        self._resume_candidate_batch_idx = 0
        self._next_packed_batch_idx = 0
        self._resume_packed_batch_idx = 0
        self._resume_candidate_rng_state: dict[str, Any] | None = None
        kwargs["batch_size"] = self.worker_chunk_size
        super().__init__(*args, **kwargs)

    def __len__(self) -> int:
        """Return a safe upper bound on the number of emitted batches."""
        if self.sampler is not None:
            return len(self.sampler)
        return super().__len__()

    def _gather_token_counts(self, local_token_counts: list[int]) -> list[list[int]]:
        """Gather per-rank token counts across distributed ranks for packing."""
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return [local_token_counts]

        gathered_token_counts: list[list[int] | None] = [
            None for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather_object(gathered_token_counts, local_token_counts)
        return [counts for counts in gathered_token_counts if counts is not None]

    def __iter__(self):
        """Construct the base iterator eagerly, then return a packing iterator."""
        num_worker_batches = super().__len__()
        if self.drop_last:
            num_candidate_batches = (
                num_worker_batches // self._worker_chunks_per_candidate
            )
        else:
            num_candidate_batches = (
                num_worker_batches + self._worker_chunks_per_candidate - 1
            ) // self._worker_chunks_per_candidate
        candidate_iter = super().__iter__()
        first_candidate_batch_idx = self._resume_candidate_batch_idx
        self._resume_candidate_batch_idx = 0
        return self._iter_packed_batches(
            candidate_iter,
            first_candidate_batch_idx=first_candidate_batch_idx,
            num_candidate_batches=num_candidate_batches,
        )

    def _next_candidate_batch(self, candidate_iter: Iterator) -> list[dict[str, Any]]:
        """Join worker-sized chunks into one token-packing candidate window."""
        candidate_batch = []
        for _ in range(self._worker_chunks_per_candidate):
            try:
                worker_chunk = next(candidate_iter)
            except StopIteration:
                break
            if isinstance(worker_chunk, _CoalescedWorkerChunk):
                worker_chunk = worker_chunk.unpack()
            if not isinstance(worker_chunk, list):
                raise ValueError(
                    "TokenBudgetDataLoader requires collate_fn to return a list of "
                    "per-example dicts or a coalesced worker chunk."
                )
            candidate_batch.extend(worker_chunk)

        if not candidate_batch:
            raise StopIteration
        return candidate_batch

    def _iter_packed_batches(
        self,
        candidate_iter: Iterator,
        *,
        first_candidate_batch_idx: int,
        num_candidate_batches: int,
    ) -> Iterator[list[dict[str, Any]]]:
        """Yield packed batches that satisfy the expanded-token budget."""
        for candidate_batch_idx in range(
            first_candidate_batch_idx, num_candidate_batches
        ):
            self._candidate_batch_idx = candidate_batch_idx
            self._candidate_start_state = super().state_dict()
            self._candidate_start_rng_state = (
                self._resume_candidate_rng_state
                if self._resume_candidate_rng_state is not None
                else _capture_process_rng_state()
            )
            try:
                if self._resume_candidate_rng_state is None:
                    candidate_batch = self._next_candidate_batch(candidate_iter)
                else:
                    # A partially emitted candidate must be fetched again to
                    # reconstruct its transformed examples. Recreate it from
                    # its original RNG state without advancing model-side RNG
                    # beyond the checkpoint state.
                    checkpoint_rng_state = _capture_process_rng_state()
                    try:
                        _restore_process_rng_state(self._resume_candidate_rng_state)
                        candidate_batch = self._next_candidate_batch(candidate_iter)
                    finally:
                        _restore_process_rng_state(checkpoint_rng_state)
                    self._resume_candidate_rng_state = None
            except StopIteration:
                self._candidate_start_state = None
                self._candidate_start_rng_state = None
                self._resume_candidate_rng_state = None
                break
            local_token_counts = [
                get_diffusion_expanded_num_tokens(example)
                for example in candidate_batch
            ]
            token_counts_by_rank = self._gather_token_counts(local_token_counts)
            packed_positions = pack_token_budget_batches(
                token_counts_by_rank, self.max_num_tokens
            )
            resume_packed_batch_idx = self._resume_packed_batch_idx
            self._resume_packed_batch_idx = 0
            for packed_batch_idx, positions in enumerate(packed_positions):
                if packed_batch_idx < resume_packed_batch_idx:
                    continue
                is_last_batch = (
                    candidate_batch_idx + 1 == num_candidate_batches
                    and packed_batch_idx + 1 == len(packed_positions)
                )
                packed_batch = []
                for position in positions:
                    example = dict(candidate_batch[position])
                    example[_TOKEN_BUDGET_LAST_BATCH_KEY] = is_last_batch
                    packed_batch.append(example)
                self._next_packed_batch_idx = packed_batch_idx + 1
                yield packed_batch
            self._candidate_start_state = None
            self._candidate_start_rng_state = None
            self._next_candidate_batch_idx = candidate_batch_idx + 1
            self._next_packed_batch_idx = 0
        self._next_candidate_batch_idx = 0

    def state_dict(self) -> dict[str, Any]:
        """Return loader state without losing a partially emitted candidate batch."""
        if self._candidate_start_state is None:
            return {
                "token_budget_version": 2,
                "base_state": super().state_dict(),
                "resume_current_candidate": False,
                "candidate_batch_idx": self._next_candidate_batch_idx,
                "packed_batch_idx": 0,
                "candidate_rng_state": None,
            }
        return {
            "token_budget_version": 2,
            "base_state": self._candidate_start_state,
            "resume_current_candidate": True,
            "candidate_batch_idx": self._candidate_batch_idx,
            "packed_batch_idx": self._next_packed_batch_idx,
            "candidate_rng_state": self._candidate_start_rng_state,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the base candidate and packed-output cursor."""
        if state_dict.get("token_budget_version") != 2:
            raise ValueError("Unsupported TokenBudgetDataLoader checkpoint state.")
        super().load_state_dict(state_dict["base_state"])
        self._candidate_start_state = None
        self._candidate_start_rng_state = None
        self._candidate_batch_idx = 0
        self._next_candidate_batch_idx = 0
        self._resume_candidate_batch_idx = int(state_dict["candidate_batch_idx"])
        self._next_packed_batch_idx = 0
        self._resume_packed_batch_idx = (
            int(state_dict["packed_batch_idx"])
            if state_dict["resume_current_candidate"]
            else 0
        )
        self._resume_candidate_rng_state = (
            state_dict["candidate_rng_state"]
            if state_dict["resume_current_candidate"]
            else None
        )


class SubsetWithID(Subset):
    """Subset that preserves example ID conversion when the parent supports it."""

    @cached_property
    def _parent_idx_to_subset_idx(self) -> dict[int, int]:
        """Map parent dataset indices to local subset indices."""
        return {
            int(parent_idx): subset_idx
            for subset_idx, parent_idx in enumerate(self.indices)
        }

    def __contains__(self, example_id: str) -> bool:
        """Check whether an example ID is in the subset."""
        try:
            self.id_to_idx(example_id)
        except (AttributeError, KeyError, ValueError):
            return False
        return True

    def id_to_idx(self, example_id: str | list[str]) -> int | list[int]:
        """Convert parent example ID(s) to local subset index/indices."""
        if isinstance(example_id, list):
            return [self.id_to_idx(id_) for id_ in example_id]

        parent_idx = self.dataset.id_to_idx(example_id)
        try:
            return self._parent_idx_to_subset_idx[int(parent_idx)]
        except KeyError as exc:
            raise KeyError(f"Example ID {example_id!r} is not in this subset.") from exc

    def idx_to_id(self, idx: int | list[int]) -> str | list[str]:
        """Convert local subset index/indices to parent example ID(s)."""
        if isinstance(idx, list):
            return [self.idx_to_id(i) for i in idx]

        return self.dataset.idx_to_id(self.indices[idx])


def _read_data_id_index_cache(
    cache_path: Path,
    data_id: str,
    dataset_len: int,
) -> list[int]:
    """Read cached indices for the given data_id and dataset length, validating the cache header."""
    with cache_path.open() as handle:
        header = handle.readline().strip()
        expected_header = f"# data_id={data_id} dataset_len={dataset_len}"
        if header != expected_header:
            raise ValueError(
                f"Data-id subset cache {cache_path} does not match this dataset. "
                f"Expected header {expected_header!r}, found {header!r}."
            )
        return [
            int(line)
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]


def _write_data_id_index_cache(
    cache_path: Path,
    data_id: str,
    dataset_len: int,
    indices: list[int],
) -> None:
    """Write cached indices for the given data_id and dataset length, with a header for validation."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp")
    with tmp_path.open("w") as handle:
        handle.write(f"# data_id={data_id} dataset_len={dataset_len}\n")
        for index in indices:
            handle.write(f"{index}\n")
    tmp_path.replace(cache_path)


def _is_ase_lmdb_like_dataset(dataset: Dataset) -> bool:
    """Check if the dataset has the internal structure of an ASE-LMDB-like dataset, which allows for efficient data_id filtering."""
    return hasattr(dataset, "_lengths") and hasattr(dataset, "_get_row_by_local_index")


def _row_matches_data_id(row, data_id: str) -> bool:
    """Check if the given row from an ASE-LMDB-like dataset matches the requested data_id."""
    row_data = row.get("data", {}) if hasattr(row, "get") else {}
    return isinstance(row_data, dict) and row_data.get("data_id") == data_id


def _ase_lmdb_indices_for_data_id(dataset: Dataset, data_id: str) -> list[int]:
    """Scan through the shards of an ASE-LMDB-like dataset to find global indices of rows matching the requested data_id."""
    indices = []
    offset = 0
    for shard_idx, shard_len in enumerate(dataset._lengths):
        for local_idx in range(shard_len):
            row = dataset._get_row_by_local_index(shard_idx, local_idx)
            if _row_matches_data_id(row, data_id):
                indices.append(offset + local_idx)
        offset += shard_len
    return indices


def _indices_for_data_id(dataset: Dataset, data_id: str) -> list[int]:
    """Recursively find global indices of examples with the given data_id in the dataset, supporting nested Subset and ConcatDataset structures, as well as efficient scanning for ASE-LMDB-like datasets."""
    if _is_ase_lmdb_like_dataset(dataset):
        return _ase_lmdb_indices_for_data_id(dataset, data_id)

    if isinstance(dataset, Subset):
        parent_indices = set(_indices_for_data_id(dataset.dataset, data_id))
        return [
            subset_idx
            for subset_idx, parent_idx in enumerate(dataset.indices)
            if parent_idx in parent_indices
        ]

    if isinstance(dataset, ConcatDataset):
        indices = []
        offset = 0
        found_supported_dataset = False
        for sub_dataset in dataset.datasets:
            try:
                sub_indices = _indices_for_data_id(sub_dataset, data_id)
            except TypeError:
                offset += len(sub_dataset)
                continue

            found_supported_dataset = True
            indices.extend(offset + index for index in sub_indices)
            offset += len(sub_dataset)

        if not found_supported_dataset:
            raise TypeError(
                "No ASE-LMDB-like dataset was found while filtering by data_id."
            )
        return indices

    raise TypeError(
        f"Dataset of type {type(dataset).__name__} does not support data_id filtering."
    )


def subset_dataset_to_data_id(
    dataset: Dataset,
    data_id: str,
    cache_path: str | Path | None = None,
) -> Subset:
    """Subset an ASE-LMDB-backed dataset to rows with the requested ``data_id``.

    If ``cache_path`` is provided, matching integer indices are cached and reused.
    The cache is keyed by the requested data ID and the top-level dataset length.
    """
    indices = _indices_for_data_id_with_cache(dataset, data_id, cache_path)

    ranked_logger.info(f"Subsetting dataset to {len(indices)} examples.")
    return SubsetWithID(dataset, indices)


def _indices_for_data_id_with_cache(
    dataset: Dataset,
    data_id: str,
    cache_path: str | Path | None = None,
) -> list[int]:
    """Find indices matching ``data_id``, optionally using the on-disk cache."""
    if not data_id:
        raise ValueError("Expected a non-empty data_id for dataset subsetting.")

    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.exists():
        indices = _read_data_id_index_cache(cache_path, data_id, len(dataset))
        ranked_logger.info(
            f"Loaded {len(indices)} examples with data_id={data_id!r} from {cache_path}."
        )
    else:
        ranked_logger.info(f"Scanning dataset for examples with data_id={data_id!r}...")
        indices = _indices_for_data_id(dataset, data_id)
        if cache_path is not None:
            _write_data_id_index_cache(cache_path, data_id, len(dataset), indices)
            ranked_logger.info(
                f"Cached {len(indices)} examples with data_id={data_id!r} to {cache_path}."
            )

    if not indices:
        raise ValueError(f"No examples found with data_id={data_id!r}.")

    return indices


def _subset_sampler_to_indices(
    sampler: Sampler,
    indices: list[int],
) -> WeightedRandomSampler:
    """Create a child sampler aligned with a subset of its dataset."""
    if hasattr(sampler, "weights"):
        weights = torch.as_tensor(sampler.weights)
        index_tensor = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=weights.device,
        )
        subset_weights = weights[index_tensor]
    else:
        ranked_logger.warning(
            f"Sampler of type {type(sampler).__name__} does not expose weights; "
            "using uniform weights for the requested subset."
        )
        subset_weights = torch.ones(len(indices))

    return WeightedRandomSampler(
        weights=subset_weights,
        num_samples=len(indices),
        replacement=getattr(sampler, "replacement", True),
        generator=getattr(sampler, "generator", None),
    )


def uniform_subset_indices(
    dataset_length: int,
    subset_size: int,
    seed: int,
) -> list[int]:
    """Select a deterministic uniform subset without replacement.

    NumPy's explicit PCG64 generator keeps the selection independent of global
    NumPy, Python, and PyTorch RNG state. Sorting does not change subset
    membership and gives every rank the same stable child-dataset ordering.

    Args:
        dataset_length: Number of examples in the source dataset.
        subset_size: Number of unique examples to retain.
        seed: Non-negative PCG64 seed.

    Returns:
        Sorted source-dataset indices for the selected subset.
    """
    for name, value in (
        ("dataset_length", dataset_length),
        ("subset_size", subset_size),
        ("seed", seed),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer, got {value!r}.")
    dataset_length = int(dataset_length)
    subset_size = int(subset_size)
    seed = int(seed)

    if dataset_length <= 0:
        raise ValueError(f"dataset_length must be positive, got {dataset_length}.")
    if subset_size <= 0 or subset_size > dataset_length:
        raise ValueError(
            "subset_size must be positive and no greater than dataset_length; "
            f"got subset_size={subset_size}, dataset_length={dataset_length}."
        )
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}.")

    generator = np.random.Generator(np.random.PCG64(seed))
    indices = generator.choice(dataset_length, size=subset_size, replace=False)
    return np.sort(indices).tolist()


def _index_subset_sha256(indices: list[int]) -> str:
    """Hash indices using a stable little-endian signed-int64 encoding."""
    encoded = np.asarray(indices, dtype="<i8").tobytes()
    return hashlib.sha256(encoded).hexdigest()


def subset_dataset_and_sampler_uniformly(
    dataset: Dataset,
    sampler: Sampler,
    subset_size: int,
    seed: int,
    expected_source_size: int | None = None,
    expected_indices_sha256: str | None = None,
) -> tuple[SubsetWithID, WeightedRandomSampler]:
    """Uniformly subset one dataset and keep its sampler weights aligned."""
    if expected_source_size is not None and (
        isinstance(expected_source_size, bool)
        or not isinstance(expected_source_size, Integral)
    ):
        raise TypeError(
            "expected_source_size must be an integer or None, got "
            f"{expected_source_size!r}."
        )
    if expected_indices_sha256 is not None and not isinstance(
        expected_indices_sha256, str
    ):
        raise TypeError(
            "expected_indices_sha256 must be a string or None, got "
            f"{expected_indices_sha256!r}."
        )
    if expected_source_size is not None and len(dataset) != expected_source_size:
        raise ValueError(
            "Uniform-subset source cardinality mismatch: expected "
            f"{expected_source_size:,} examples, found {len(dataset):,}."
        )
    indices = uniform_subset_indices(len(dataset), subset_size, seed)
    indices_sha256 = _index_subset_sha256(indices)
    if (
        expected_indices_sha256 is not None
        and indices_sha256 != expected_indices_sha256
    ):
        raise ValueError(
            "Uniform-subset index hash mismatch: expected "
            f"{expected_indices_sha256}, computed {indices_sha256}."
        )
    subset_dataset = SubsetWithID(dataset, indices)
    subset_sampler = _subset_sampler_to_indices(sampler, indices)
    ranked_logger.info(
        "Uniformly subsetting dataset from "
        f"{len(dataset):,} to {len(indices):,} examples without replacement "
        f"(algorithm=PCG64, seed={seed}, "
        f"indices_sha256={indices_sha256})."
    )
    return subset_dataset, subset_sampler


def subset_dataset_and_sampler_to_data_id(
    dataset: Dataset,
    sampler: Sampler,
    data_id: str,
    cache_path: str | Path | None = None,
) -> tuple[SubsetWithID, WeightedRandomSampler]:
    """Subset a dataset by ``data_id`` and keep its sampler weights aligned."""
    indices = _indices_for_data_id_with_cache(dataset, data_id, cache_path)
    subset_dataset = SubsetWithID(dataset, indices)
    subset_sampler = _subset_sampler_to_indices(sampler, indices)
    ranked_logger.info(f"Subsetting dataset to {len(indices)} examples.")
    return subset_dataset, subset_sampler


def subset_dataset_to_example_ids(
    dataset: Dataset,
    example_ids: list[str],
) -> Subset:
    """Subset an ID-aware dataset without loading rows during index lookup.

    Calling AtomWorks' row-and-index helper on a concatenated dataset currently
    resolves the correct global index but then indexes the selected child with
    that global offset.  Debug subsetting only needs the indices, so using the
    dataset's public ``id_to_idx()`` contract directly avoids that mismatch and
    preserves the caller-provided example order.
    """
    if not hasattr(dataset, "id_to_idx"):
        raise TypeError(
            "Example-ID subsetting requires a dataset with an id_to_idx() method."
        )

    indices = [int(dataset.id_to_idx(str(example_id))) for example_id in example_ids]
    invalid_indices = [index for index in indices if index < 0 or index >= len(dataset)]
    if invalid_indices:
        raise IndexError(
            f"Resolved out-of-bounds example indices {invalid_indices} for a "
            f"dataset of length {len(dataset)}."
        )
    return Subset(dataset, indices)


def subset_train_dataset_and_sampler_to_data_ids(
    dataset: Dataset,
    sampler: Sampler,
    data_id_subsets: dict[str | None, tuple[str, str | Path | None]],
) -> tuple[Dataset, Sampler]:
    """Apply data_id subsets while preserving mixed train datasets."""
    if not data_id_subsets:
        return dataset, sampler

    if not isinstance(sampler, MixedSampler):
        if len(data_id_subsets) != 1:
            raise ValueError("Only one data_id subset can be applied to one sampler.")
        data_id, cache_path = next(iter(data_id_subsets.values()))
        return subset_dataset_and_sampler_to_data_id(
            dataset,
            sampler,
            data_id,
            cache_path=cache_path,
        )

    datasets_info = [dict(info) for info in sampler.datasets_info]

    active_names = {info.get("name") for info in datasets_info}
    unknown_names = set(data_id_subsets) - active_names
    if unknown_names:
        raise ValueError(
            "Configured data_id subsets for inactive or unknown train datasets: "
            f"{sorted(str(name) for name in unknown_names)}."
        )

    for info in datasets_info:
        subset = data_id_subsets.get(info.get("name"))
        if subset is None:
            continue
        data_id, cache_path = subset

        info["dataset"], info["sampler"] = subset_dataset_and_sampler_to_data_id(
            info["dataset"],
            info["sampler"],
            data_id,
            cache_path=cache_path,
        )

    return (
        ConcatDatasetWithID(datasets=[info["dataset"] for info in datasets_info]),
        MixedSampler(datasets_info=datasets_info, shuffle=sampler.shuffle),
    )


def subset_train_dataset_and_sampler_uniformly(
    dataset: Dataset,
    sampler: Sampler,
    uniform_subsets: dict[
        str | None,
        tuple[int, int, int | None, str | None],
    ],
) -> tuple[Dataset, Sampler]:
    """Apply deterministic uniform subsets to a freshly constructed sampler.

    This helper is part of dataset construction and must run before a
    ``MixedSampler`` receives an epoch length or iterator/checkpoint state.
    """
    if not uniform_subsets:
        return dataset, sampler

    if not isinstance(sampler, MixedSampler):
        if len(uniform_subsets) != 1:
            raise ValueError("Only one uniform subset can be applied to one sampler.")
        (
            subset_size,
            seed,
            expected_source_size,
            expected_indices_sha256,
        ) = next(iter(uniform_subsets.values()))
        return subset_dataset_and_sampler_uniformly(
            dataset,
            sampler,
            subset_size,
            seed,
            expected_source_size,
            expected_indices_sha256,
        )

    if sampler.epoch != 0 or hasattr(sampler, "n_examples_per_epoch"):
        raise RuntimeError(
            "Uniform subsetting must run before MixedSampler epoch or length state "
            "is assigned."
        )

    datasets_info = [dict(info) for info in sampler.datasets_info]
    active_names = {info.get("name") for info in datasets_info}
    unknown_names = set(uniform_subsets) - active_names
    if unknown_names:
        raise ValueError(
            "Configured uniform subsets for inactive or unknown train datasets: "
            f"{sorted(str(name) for name in unknown_names)}."
        )

    for info in datasets_info:
        subset = uniform_subsets.get(info.get("name"))
        if subset is None:
            continue
        subset_size, seed, expected_source_size, expected_indices_sha256 = subset
        info["dataset"], info["sampler"] = subset_dataset_and_sampler_uniformly(
            info["dataset"],
            info["sampler"],
            subset_size,
            seed,
            expected_source_size,
            expected_indices_sha256,
        )

    return (
        ConcatDatasetWithID(datasets=[info["dataset"] for info in datasets_info]),
        MixedSampler(datasets_info=datasets_info, shuffle=sampler.shuffle),
    )


def assemble_distributed_loader(
    dataset: Dataset,
    sampler: Sampler | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    n_examples_per_epoch: int | None = None,
    loader_cfg: DictConfig | dict | None = None,
    max_num_tokens: int | None = None,
    stateful: bool = False,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Assembles a distributed DataLoader for training or validation.

    Performs the following steps:
        (1) If not already a distributed sampler, wraps the sampler with a DistributedSampler or DistributedMixedSampler
        (2) Wraps the dataset and sampler with a fallback mechanism, if needed
        (3) Assembles the final DataLoader

    Args:
        dataset (Dataset): The dataset to be used for training or validation.
        sampler (Sampler): The sampler to be used for training or validation. May already be distributed.
        rank (int): The rank of the current process in distributed training.
        world_size (int): The total number of processes participating in the distributed training.
        n_examples_per_epoch (int): The number of examples to sample per epoch, across all GPUs.
            For example, if we have 8 GPUs, with 2 gradient accumulation steps and 10 optimizer
            steps per epoch, we would sample 160 examples per epoch (8 * 2 * 10).
        loader_cfg (DictConfig or dict, optional): Additional configuration parameters for the
            DataLoader, such as `batch_size` and `num_workers`. Defaults to an empty dictionary.
        max_num_tokens: If provided, dynamically pack transformed examples so
            ``diffusion_batch_size * num_tokens`` does not exceed this value on
            any distributed rank. The configured DataLoader ``batch_size`` is
            used as the maximum candidate packing window.
        stateful: Whether to preserve sampler, worker, and loader state for
            exact mid-epoch checkpoint resumption. Defaults to ``False`` to
            avoid worker-snapshot overhead in epoch-only training.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        drop_last (bool, optional): Whether to drop the last incomplete batch if the dataset size
            is not divisible by the number of GPUs. Defaults to False.

    Returns:
        DataLoader: A PyTorch DataLoader configured for distributed training, with datasets
            concatenated and sampled according to their defined probabilities.
    """
    if not loader_cfg:
        loader_cfg = {}

    if isinstance(sampler, MixedSampler):
        # (If given a MixedSampler, we must convert to a DistributedMixedSampler)
        assert (
            rank is not None
            and world_size is not None
            and n_examples_per_epoch is not None
        ), "Rank, world_size, and n_examples_per_epoch must be provided for MixedSampler"
        sampler = DistributedMixedSampler(
            datasets_info=sampler.datasets_info,
            num_replicas=world_size,
            rank=rank,
            n_examples_per_epoch=n_examples_per_epoch,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    elif isinstance(sampler, (RandomSampler, SequentialSampler)):
        # (If given a RandomSampler or SequentialSampler, we must convert to a DistributedSampler)
        assert (
            rank is not None and world_size is not None
        ), "Rank and world_size must be provided for RandomSampler or SequentialSampler"
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    elif sampler is None and isinstance(dataset, Subset):
        # We are subsetting the dataset to a specific set of example IDs
        ranked_logger.info(f"Subsetting dataset to {len(dataset)} examples!")
    else:
        # (We assume we are already given a DistributedSampler or DistributedMixedSampler)
        assert (
            rank is None and world_size is None
        ), "Rank and world_size will have no effect on the provided sampler and should be None"
        assert isinstance(
            sampler, (DistributedSampler, DistributedMixedSampler)
        ), "Invalid sampler type for distributed training."

    # ... bound single-example cost before fallbacks, so an over-budget example
    # is retried like any other loader failure
    example_timeout_seconds = loader_cfg.get("example_timeout_seconds", None)
    if example_timeout_seconds:
        ranked_logger.info(
            f"Bounding example loading to {float(example_timeout_seconds):g}s."
        )
        dataset = TimeBoundedDataset(dataset, float(example_timeout_seconds))

    # ... wrap the composed dataset and sampler with a fallback mechanism, if needed
    if (
        "n_fallback_retries" in loader_cfg
        and loader_cfg.get("n_fallback_retries", 0) > 0
        and sampler is not None
    ):
        ranked_logger.info(
            f"Wrapping train dataset and sampler with {loader_cfg.get('n_fallback_retries', 0)} fallbacks..."
        )
        dataset, sampler = wrap_dataset_and_sampler_with_fallbacks(
            dataset_to_be_wrapped=dataset,
            sampler_to_be_wrapped=sampler,
            dataset_to_fallback_to=dataset,
            sampler_to_fallback_to=sampler,
            n_fallback_retries=loader_cfg.get("n_fallback_retries", 0),
        )

    if stateful:
        # Preserve stochastic transform state independently in each worker.
        dataset = _WorkerRNGStateDataset(dataset)

        # AtomWorks' mixed/fallback samplers are finite but do not expose
        # iterator state. Regenerate only their index stream when restoring.
        if sampler is not None and not isinstance(sampler, DistributedSampler):
            sampler = StatefulSampler(sampler)

    # ... assemble the final loader
    if max_num_tokens is not None:
        loader_cls = TokenBudgetDataLoader
    elif stateful:
        loader_cls = StatefulDataLoader
    else:
        loader_cls = DataLoader
    token_budget_kwargs = (
        {"max_num_tokens": int(max_num_tokens)} if max_num_tokens is not None else {}
    )
    if max_num_tokens is not None and not stateful:
        token_budget_kwargs["snapshot_every_n_steps"] = None
    dataloader_params = dict(loader_cfg.get("dataloader_params", {}))
    if dataloader_params.get("timeout") and not dataloader_params.get("num_workers"):
        # PyTorch rejects a worker timeout for in-process loading; the
        # per-example budget above still applies in that configuration.
        ranked_logger.info(
            "Dropping the dataloader worker timeout because num_workers=0."
        )
        dataloader_params.pop("timeout")
    coalesce_worker_tensors = bool(
        dataloader_params.pop("coalesce_worker_tensors", False)
    )
    sanitize_coordinate_nans = bool(
        dataloader_params.pop("sanitize_coordinate_nans", False)
    )
    if coalesce_worker_tensors and max_num_tokens is None:
        raise ValueError(
            "coalesce_worker_tensors is currently supported only for "
            "token-budgeted training loaders."
        )
    loader = loader_cls(
        dataset=dataset,
        sampler=sampler,
        collate_fn=partial(
            _coalesce_worker_tensors if coalesce_worker_tensors else _identity_collate,
            sanitize_coordinate_nans=sanitize_coordinate_nans,
        ),
        **token_budget_kwargs,
        **dataloader_params,
    )

    return loader


class _AtomicPickleWriter:
    """The pandas API used by the AtomWorks parser, writing caches atomically.

    The parser pickles a cache entry directly onto its final path, so a reader
    on another rank can pick up a half-written file. Publishing through a
    rename keeps concurrent writers and readers safe.
    """

    def __init__(self, pandas: Any) -> None:
        self._pandas = pandas

    def to_pickle(self, obj: Any, path: Any, *args: Any, **kwargs: Any) -> None:
        """Write the pickle to a sibling temporary file, then rename it."""
        path = Path(path)
        # Keep the final suffixes: pandas infers compression from them.
        temporary = path.with_name(f".tmp-{os.getpid()}-{path.name}")
        try:
            self._pandas.to_pickle(obj, temporary, *args, **kwargs)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def __getattr__(self, name: str) -> Any:
        """Delegate every other attribute to the real pandas module."""
        return getattr(self._pandas, name)


def patch_atomworks_cif_cache_atomic_writes() -> None:
    """Make AtomWorks' CIF cache safe to write from concurrent processes."""
    from atomworks.io import parser

    if not getattr(parser, "_nucleus_atomic_cache_patched", False):
        parser.pd = _AtomicPickleWriter(parser.pd)  # type: ignore[assignment]
        parser._nucleus_atomic_cache_patched = True  # type: ignore[attr-defined]


def patch_atomworks_pandas_dataset_query_engine() -> None:
    """Work around a pandas query engine regression on string-heavy filters.

    The PDB training configs use `DataFrame.query()` expressions with
    `.str.contains(...)`. On the current pandas/numexpr combination, the
    default engine can fail with an `UndefinedVariableError` on a temporary
    `Series_*` symbol. For these dataset filters, the Python engine is correct
    and robust, so we force it for atomworks' `PandasDataset`.
    """
    from atomworks.ml.datasets.pandas_dataset import PandasDataset

    if getattr(PandasDataset, "_nucleus_query_engine_patched", False):
        return

    def _apply_query(self, query: str) -> None:
        original_num_rows = len(self.data)
        self.data = self.data.query(query, engine="python")
        filtered_num_rows = len(self.data)
        self._validate_filter_impact(query, original_num_rows, filtered_num_rows)

    PandasDataset._apply_query = _apply_query  # type: ignore[method-assign]
    PandasDataset._nucleus_query_engine_patched = True  # type: ignore[attr-defined]
