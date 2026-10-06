"""Small, logger-agnostic summaries for tensor-scale diagnostics."""

from __future__ import annotations

from typing import TypedDict

import torch


class RMSAccumulator(TypedDict):
    """Sufficient statistics for an exact RMS reduction."""

    sum_squares: torch.Tensor
    count: torch.Tensor


class TensorScaleSummary(TypedDict):
    """Scale statistics that can be combined across batches and ranks."""

    rms: RMSAccumulator
    absmax: torch.Tensor


class ModelDiagnostics(TypedDict):
    """Logger-neutral diagnostics emitted by a model forward pass."""

    rms: dict[str, RMSAccumulator]
    max: dict[str, torch.Tensor]


def summarize_tensor_scale(value: torch.Tensor) -> TensorScaleSummary:
    """Return detached float32 statistics sufficient for RMS and absolute max.

    Keeping the sum of squares and element count separate lets the logging layer
    compute an exact global RMS when distributed ranks contain different numbers
    of atoms or tokens.
    """
    value = value.detach().float()
    return {
        "rms": {
            "sum_squares": value.square().sum(),
            "count": torch.tensor(value.numel(), device=value.device),
        },
        "absmax": value.abs().amax(),
    }


def summarize_logit_scale(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return detached magnitude diagnostics for classifier logits."""
    logits = logits.detach().float()
    span = logits.amax(dim=-1) - logits.amin(dim=-1)
    return {
        "absmax": logits.abs().amax(),
        "span_max": span.amax(),
    }


def summarize_zatom_activation_scales(
    *,
    token_trunk: torch.Tensor,
    atom_trunk: torch.Tensor,
    token_condition: torch.Tensor,
    sequence_head_input: torch.Tensor,
    atom_type_head_input: torch.Tensor,
    lattice_head_input: torch.Tensor,
    sequence_logits: torch.Tensor,
    atom_type_logits: torch.Tensor,
    lattice_lengths: torch.Tensor,
    lattice_angles: torch.Tensor,
    energy_head_input: torch.Tensor | None = None,
    normalized_energy: torch.Tensor | None = None,
    energy_token_contributions: torch.Tensor | None = None,
) -> ModelDiagnostics:
    """Summarize Zatom representations and every output path derived from Q_I."""
    activations = {
        "token_head_trunk_q": token_trunk,
        "token_head_atom_q": atom_trunk,
        "token_head_condition_s": token_condition,
        "sequence_head_input": sequence_head_input,
        "atom_type_head_input": atom_type_head_input,
        "lattice_head_input": lattice_head_input,
        "lattice_head_lengths": lattice_lengths,
        "lattice_head_angles": lattice_angles,
    }
    optional_activations = {
        "energy_head_input": energy_head_input,
        "energy_head_normalized_energy": normalized_energy,
        "energy_head_token_contribution": energy_token_contributions,
    }
    activations.update(
        {
            name: value
            for name, value in optional_activations.items()
            if value is not None
        }
    )
    rms: dict[str, RMSAccumulator] = {}
    maximum: dict[str, torch.Tensor] = {}
    summaries_by_tensor: dict[int, TensorScaleSummary] = {}
    for name, value in activations.items():
        tensor_id = id(value)
        if tensor_id not in summaries_by_tensor:
            summaries_by_tensor[tensor_id] = summarize_tensor_scale(value)
        summary = summaries_by_tensor[tensor_id]
        rms[f"{name}_rms"] = summary["rms"]
        maximum[f"{name}_absmax"] = summary["absmax"]

    for name, logits in {
        "sequence_head_logit": sequence_logits,
        "atom_type_head_logit": atom_type_logits,
    }.items():
        summary = summarize_logit_scale(logits)
        maximum[f"{name}_absmax"] = summary["absmax"]
        maximum[f"{name}_span_max"] = summary["span_max"]

    return {"rms": rms, "max": maximum}
