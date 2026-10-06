import torch

try:
    from torch.nn.attention.flex_attention import (
        create_block_mask,
    )
except ImportError:
    create_block_mask = None


@torch.no_grad()
def maybe_create_flex_block_mask(
    attn_mask: torch.Tensor,
    q_len: int,
    device: torch.device,
    use_flex_attention: bool,
    flex_block_mask=None,
    flex_block_mask_ready: bool = False,
):
    """Create a flex block mask once, or reuse a caller-supplied one."""

    if flex_block_mask is not None or flex_block_mask_ready or not use_flex_attention:
        return flex_block_mask, flex_block_mask_ready

    try:
        flex_block_mask = create_flex_attention_block_mask(
            attn_mask,
            q_len,
            q_len,
            device,
        )
    except RuntimeError:
        flex_block_mask = None

    return flex_block_mask, True


def create_flex_attention_block_mask(
    A_mask: torch.Tensor,
    q_len: int,
    kv_len: int,
    device: torch.device,
    block_size: int = 64,
    compile: bool = False,
):
    """Create a FlexAttention block mask only when it is actually required."""

    if create_block_mask is None:
        raise RuntimeError(
            "torch.nn.attention.flex_attention is not available in this environment."
        )

    A_mask = A_mask.to(dtype=torch.bool, device=device)
    if torch.all(A_mask):
        return None
    invalid_queries = ~torch.any(A_mask, dim=-1)

    if A_mask.ndim == 2:

        def mask_mod(_b, _h, q_idx, kv_idx):
            allowed = A_mask[q_idx, kv_idx]
            # Keep empty rows safe by allowing a synthetic self edge.
            return allowed | (invalid_queries[q_idx] & (q_idx == kv_idx))

        return create_block_mask(
            # TODO: Maybe move BlockMask creation outside of model forward pass to avoid recompilation.
            mask_mod,
            None,
            None,
            q_len,
            kv_len,
            device=device,
            BLOCK_SIZE=block_size,
            _compile=compile,
        )

    if A_mask.ndim == 3:

        def mask_mod(b, _h, q_idx, kv_idx):
            allowed = A_mask[b, q_idx, kv_idx]
            return allowed | (invalid_queries[b, q_idx] & (q_idx == kv_idx))

        return create_block_mask(
            mask_mod,
            A_mask.shape[0],
            None,
            q_len,
            kv_len,
            device=device,
            BLOCK_SIZE=block_size,
            _compile=compile,
        )

    raise ValueError("A_mask must be [L, L] or [B, L, L] for the flex attention path.")
