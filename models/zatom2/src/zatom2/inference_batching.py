"""Batch independent generation inputs without adding diffusion replicas."""

import random

import numpy as np
import torch
from nucleus.constants import MLIP_PROPERTY_TASK_ID
from nucleus.inference_input_parsing import install_molecule_contig_support
from rfd3.inference.datasets import ContigJsonDataset
from torch.utils.data import DataLoader


def _identity_collate(examples):
    return examples


def _initialize_inference_worker(_worker_id):
    # Spawned workers need the same input-specification extensions as the parent.
    install_molecule_contig_support()
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_inference_loader(
    specs,
    *,
    pipeline,
    rank: int,
    world_size: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    seed: int | None,
):
    """Prefetch rank-local inputs without DistributedSampler padding duplicates."""
    dataset = ContigJsonDataset(
        data=specs,
        transform=pipeline,
        name="inference-dataset",
        cif_parser_args=None,
        subset_to_keys=None,
        eval_every_n=1,
    )
    # Keep DataLoader iterator/worker seed draws separate from the sampler RNG.
    generator = torch.Generator()
    generator.manual_seed((torch.initial_seed() if seed is None else seed) + rank)
    kwargs = {}
    if num_workers:
        kwargs.update(
            multiprocessing_context="spawn",
            prefetch_factor=prefetch_factor,
            worker_init_fn=_initialize_inference_worker,
        )
    return DataLoader(
        dataset,
        sampler=range(rank, len(dataset), world_size),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_identity_collate,
        generator=generator,
        **kwargs,
    )


def batching_fallback_reason(examples) -> str | None:
    """Identify inputs whose sampler behavior currently requires a single example."""
    feature_keys = set(examples[0]["feats"]) - {"token_bonds"}
    property_modes = set()
    for example in examples:
        features = example["feats"]
        task_ids = example.get("ground_truth", {}).get("property_task_id_B")
        if (
            task_ids is not None
            and (torch.as_tensor(task_ids) == MLIP_PROPERTY_TASK_ID).any()
        ):
            return "clean-endpoint property inference"
        if set(features) - {"token_bonds"} != feature_keys:
            return "different optional conditioning features"
        if "partial_t" in features:
            return "partial diffusion schedules"
        unindexed = features.get("is_motif_atom_unindexed")
        if unindexed is not None and torch.as_tensor(unindexed).any():
            return "unindexed motif atoms"
        # Generated-property reconstruction makes batch-wide decisions about
        # type and force conditioning. Keep those decisions unchanged per input.
        type_mask = features.get("atom_type_generation_mask")
        type_mode = None
        if type_mask is not None:
            type_mask = torch.as_tensor(type_mask, dtype=torch.bool)
            type_mode = (bool(type_mask.any()), bool(type_mask.all()))
        force_mode = tuple(
            bool(torch.as_tensor(features[key]).any())
            for key in ("has_mean_force_norm", "has_force_rms_p95")
            if key in features
        )
        property_modes.add((type_mode, force_mode))
    if len(property_modes) > 1:
        return "different property reconstruction modes"
    return None


# Explicit output layouts avoid confusing the diffusion, atom, token and system
# axes when two happen to have the same length. These are the fields consumed by
# the existing per-example structure/metadata writer, all with leading [D, ...].
_ATOM_OUTPUTS = {
    "X_L",
    "fractional_coord_L",
    "material_frame_reference_X_L",
    "normalized_forces_L",
    "forces_L",
}
_TOKEN_OUTPUTS = {
    "sequence_logits_I",
    "sequence_indices_I",
    "atom_type_logits_I",
    "atom_type_indices_I",
}
_SYSTEM_OUTPUTS = {
    "lattice_lengths_B",
    "lattice_angles_B",
    "normalized_energy_B",
    "energy_B",
}


def split_prediction_outputs(network_output, examples):
    """Move final predictions to CPU once and split packed [D, L/I/B, ...] axes.

    Trajectories are intentionally excluded here; trajectory export uses the
    existing single-example path. Original specifications stay with each input.
    """
    outputs = {
        key: value.detach().cpu()
        for key, value in network_output.items()
        if key in _ATOM_OUTPUTS | _TOKEN_OUTPUTS | _SYSTEM_OUTPUTS and value is not None
    }
    num_atoms = sum(
        example["ground_truth"]["coord_atom_lvl"].shape[0] for example in examples
    )
    num_tokens = sum(
        example["ground_truth"]["sequence_gt_I"].shape[0] for example in examples
    )
    diffusion_size = examples[0]["t"].shape[0]
    for key, value in outputs.items():
        expected_size = (
            num_atoms
            if key in _ATOM_OUTPUTS
            else num_tokens
            if key in _TOKEN_OUTPUTS
            else len(examples)
        )
        if value.ndim < 2 or value.shape[:2] != (diffusion_size, expected_size):
            raise ValueError(
                f"Packed prediction {key!r} must start with shape "
                f"[{diffusion_size}, {expected_size}], got {tuple(value.shape)}."
            )
    atom_offset = token_offset = 0
    for index, example in enumerate(examples):
        num_atoms = example["ground_truth"]["coord_atom_lvl"].shape[0]
        num_tokens = example["ground_truth"]["sequence_gt_I"].shape[0]
        sliced = {}
        for key, value in outputs.items():
            if key in _ATOM_OUTPUTS:
                selected = slice(atom_offset, atom_offset + num_atoms)
            elif key in _TOKEN_OUTPUTS:
                selected = slice(token_offset, token_offset + num_tokens)
            else:
                selected = slice(index, index + 1)
            sliced[key] = value[:, selected]
        yield sliced
        atom_offset += num_atoms
        token_offset += num_tokens
