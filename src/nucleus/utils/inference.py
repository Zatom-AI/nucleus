import numpy as np
import torch


def make_rank_local_inference_model(model, *, strategy, world_size: int):
    """Remove distributed forward collectives while retaining precision and EMA.

    Independent inference shards can contain different numbers of batches. A
    DDP forward can otherwise wait for ranks that have already finished. Fabric's
    local wrapper preserves its input/output conversions and autocast context.
    """
    if world_size == 1:
        return model

    from lightning.fabric.wrappers import _FabricModule

    if not isinstance(model, _FabricModule):
        raise RuntimeError(
            "Distributed inference expected Fabric to wrap the model, got "
            f"{type(model).__name__}."
        )
    local_model = model.module
    local_wrapper = _FabricModule(
        forward_module=local_model,
        strategy=strategy,
        original_module=local_model,
    )
    # A new nn.Module wrapper starts in training mode even for an eval model.
    # Copy its mode without recursively changing existing submodule modes.
    local_wrapper.training = model.training
    return local_wrapper


def set_sequential_atom_ids(structure):
    """Assign CIF-safe integer atom IDs while preserving integer dtype when possible."""
    dtype = np.uint32
    if "atom_id" in structure.get_annotation_categories():
        current_dtype = structure.atom_id.dtype
        if np.issubdtype(current_dtype, np.integer):
            dtype = current_dtype
        else:
            structure.del_annotation("atom_id")
    structure.set_annotation(
        "atom_id",
        np.arange(structure.array_length(), dtype=dtype),
    )
    return structure


def find_ema_wrapper(model):
    """Return the EMA wrapper, unwrapping Fabric modules when needed."""
    current = model
    for _ in range(5):
        if (
            hasattr(current, "model")
            and hasattr(current, "shadow")
            and callable(getattr(current, "update", None))
        ):
            return current
        next_module = getattr(current, "module", None)
        if next_module is None or next_module is current:
            break
        current = next_module
    return None


def copy_live_weights_to_ema_shadow(model) -> bool:
    """Make eval-mode EMA inference use the live model weights."""
    ema_model = find_ema_wrapper(model)
    if ema_model is None:
        return False
    with torch.no_grad():
        ema_model.shadow.load_state_dict(ema_model.model.state_dict())
    return True
