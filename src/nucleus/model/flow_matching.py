"""Core linear flow-matching operations used by training and sampling.

The convention follows Emyx: ``t=0`` is the Gaussian prior and ``t=1`` is
clean data.
"""

from __future__ import annotations

import torch

FLOW_MATCHING = "flow_matching"


def _append_dims(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Append singleton dimensions until ``value`` broadcasts over ``target``.

    Args:
        value: A tensor to be reshaped.
        target: A tensor whose shape is the target shape for broadcasting.

    Returns:
        A tensor with the same data as ``value`` but with additional singleton dimensions
        appended to the end until it can broadcast with ``target``.
    """
    while value.ndim < target.ndim:
        value = value.unsqueeze(-1)
    return value


def sample_time(
    batch_size: int,
    *,
    beta_alpha: float = 1.9,
    uniform_probability: float = 0.02,
    min_time: float = 1e-3,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample the Emyx mixture of ``Beta(alpha, 1)`` and uniform times.

    Args:
        batch_size: The number of times to sample.
        beta_alpha: The alpha parameter of the Beta distribution. Must be positive.
        uniform_probability: The probability of sampling from the uniform distribution.
            Must be in [0, 1].
        min_time: The minimum time to sample. Must be in [0, 0.5).
        device: The device on which to create the tensor.
        dtype: The data type of the tensor.

    Returns:
        A tensor of shape ``(batch_size,)`` containing the sampled times.
    """
    if beta_alpha <= 0:
        raise ValueError(f"beta_alpha must be positive, got {beta_alpha}.")
    if not 0 <= uniform_probability <= 1:
        raise ValueError(
            "uniform_probability must be in [0, 1], got " f"{uniform_probability}."
        )
    if not 0 <= min_time < 0.5:
        raise ValueError(f"min_time must be in [0, 0.5), got {min_time}.")

    # NOTE: If U ~ Uniform(0, 1), U**(1/alpha) ~ Beta(alpha, 1).
    times = torch.rand(batch_size, device=device, dtype=dtype).pow(1 / beta_alpha)
    if uniform_probability > 0:
        use_uniform = torch.rand(batch_size, device=device) < uniform_probability
        uniform_times = torch.rand(batch_size, device=device, dtype=dtype)
        times = torch.where(use_uniform, uniform_times, times)
    return times.clamp(min_time, 1 - min_time)


def interpolate(
    clean: torch.Tensor, prior: torch.Tensor, time: torch.Tensor
) -> torch.Tensor:
    """
    Return ``x_t = (1 - t) * prior + t * clean``.

    Args:
        clean: The clean data tensor.
        prior: The prior data tensor.
        time: A tensor of shape ``(batch_size,)`` containing the flow times.

    Returns:
        A tensor of the same shape as ``clean`` and ``prior`` containing the interpolated values.
    """
    time = _append_dims(time.to(device=clean.device, dtype=clean.dtype), clean)
    return (1 - time) * prior + time * clean


def endpoint_from_velocity(
    state: torch.Tensor, velocity: torch.Tensor, time: torch.Tensor
) -> torch.Tensor:
    """
    Recover the clean endpoint as ``x_t + (1 - t) * v``.

    Args:
        state: The current state tensor.
        velocity: The velocity tensor.
        time: A tensor of shape ``(batch_size,)`` containing the flow times.

    Returns:
        A tensor of the same shape as ``state`` and ``velocity`` containing the recovered clean endpoint.
    """
    time = _append_dims(time.to(device=state.device, dtype=state.dtype), state)
    return state + (1 - time) * velocity


def edm_noise_to_flow_time(
    noise_level: torch.Tensor, sigma_data: float
) -> torch.Tensor:
    """
    Map an absolute EDM noise standard deviation to normalized flow time.

    Args:
        noise_level: A tensor of shape ``(batch_size,)`` containing the absolute EDM noise standard deviations.
        sigma_data: The standard deviation of the clean data.

    Returns:
        A tensor of shape ``(batch_size,)`` containing the normalized flow times.
    """
    return sigma_data / (sigma_data + noise_level)


def flow_time_to_edm_noise(flow_time: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """
    Map flow time to the equivalent absolute EDM noise standard deviation.

    Args:
        flow_time: A tensor of shape ``(batch_size,)`` containing the normalized flow times.
        sigma_data: The standard deviation of the clean data.

    Returns:
        A tensor of shape ``(batch_size,)`` containing the absolute EDM noise standard deviations.
    """
    return sigma_data * (1 - flow_time) / flow_time.clamp_min(1e-12)


def edm_state_to_flow_state(
    edm_state: torch.Tensor,
    flow_time: torch.Tensor,
    *,
    fixed_atom_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Map an EDM state to the equivalent flow state, preserving fixed atoms.

    Args:
        edm_state: A tensor of shape ``(batch_size, num_atoms, 3)`` containing the EDM states.
        flow_time: A tensor of shape ``(batch_size,)`` containing the normalized flow times.
        fixed_atom_mask: An optional boolean tensor of shape ``(num_atoms,)`` indicating which atoms are fixed.

    Returns:
        A tensor of shape ``(batch_size, num_atoms, 3)`` containing the equivalent flow states, with fixed atoms preserved.
    """
    time = _append_dims(
        flow_time.to(device=edm_state.device, dtype=edm_state.dtype), edm_state
    )
    flow_state = time * edm_state
    if fixed_atom_mask is not None:
        fixed_atom_mask = fixed_atom_mask.to(device=edm_state.device, dtype=torch.bool)
        flow_state = torch.where(fixed_atom_mask[None, :, None], edm_state, flow_state)
    return flow_state
