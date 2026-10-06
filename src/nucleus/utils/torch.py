from collections.abc import Mapping

import torch


def as_cpu_float64_tensor(value, *, detach: bool = False) -> torch.Tensor:
    """Move a value to CPU before converting it to float64.

    MPS cannot execute ``Tensor.double()``. Performing the device transfer first
    retains float64 evaluation arithmetic without attempting an unsupported cast
    on the source device.
    """
    tensor = torch.as_tensor(value)
    if detach:
        tensor = tensor.detach()
    return tensor.cpu().double()


def cast_float64_tensors_to_float32(value):
    """Recursively cast float64 tensor leaves to float32.

    This is intended for inputs being transferred to backends such as MPS that
    cannot represent float64. Integer and lower-precision floating tensors are
    returned unchanged.
    """
    if isinstance(value, torch.Tensor) and value.dtype == torch.float64:
        return value.float()
    if isinstance(value, Mapping):
        return type(value)(
            (key, cast_float64_tensors_to_float32(child))
            for key, child in value.items()
        )
    if isinstance(value, tuple):
        return tuple(cast_float64_tensors_to_float32(child) for child in value)
    if isinstance(value, list):
        return [cast_float64_tensors_to_float32(child) for child in value]
    return value


def get_device_compatible_float64_dtype(
    device: torch.device | str | None = None,
) -> torch.dtype:
    """Return float64 unless the target device cannot represent it.

    MPS has no float64 support. When ``device`` is omitted, use PyTorch's
    current default device so this also works inside Fabric's module-init
    context, where parameters and buffers are allocated directly on-device.

    Args:
        device: Target device, or the current default device when omitted.

    Returns:
        ``torch.float32`` for MPS and ``torch.float64`` otherwise.
    """
    target_device = (
        torch.get_default_device() if device is None else torch.device(device)
    )
    return torch.float32 if target_device.type == "mps" else torch.float64


def expand_t_to_level(t: torch.Tensor, f: dict, *, level_len: int, batch_key: str):
    """
    Expand per-example diffusion times to atom or token level.

    Args:
        t: Tensor of diffusion times
        f: Dictionary containing batch information
        level_len: Length of the level to expand to
        batch_key: Key in the batch dictionary to use for expansion

    Returns:
        Expanded tensor of diffusion times
    """
    if t.ndim == 1:
        return t[:, None].expand(-1, level_len)
    if t.ndim == 2:
        if t.shape[1] == level_len:
            return t
        if t.shape[1] == 1:
            return t.expand(-1, level_len)
        if batch_key in f:
            batch = f[batch_key].to(device=t.device, dtype=torch.long)
            return t.gather(1, batch.unsqueeze(0).expand(t.shape[0], -1))
    raise ValueError(
        f"Cannot expand timestep tensor with shape {tuple(t.shape)} to length "
        f"{level_len} using {batch_key}."
    )
