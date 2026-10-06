import torch


@torch.no_grad()
def create_attention_indices(
    f, n_attn_keys, n_attn_seq_neighbours, X_L=None, tok_idx=None
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    """
    Entry-point function for creating attention indices for sequence & structure-local attention.

    Args:
        f: input features of the model
        n_attn_keys: number of (atom) attention keys
        n_attn_seq_neighbours: number of neighbouring sequence tokens (residues) to attend to
        X_L: optional input tensor for atom positions | if None, choose random padding atoms
        tok_idx: optional input tensor for atom-to-token mapping | if None, use from f

    Returns:
        attn_indices: [B, L, k] where k = n_attn_keys and indices[b, i, j] = atom index for atom i to j-th attn query
        index_mask: [B, L, k] indicating which sparse slots contain real keys
    """

    tok_idx = f["atom_to_token_map"] if tok_idx is None else tok_idx
    device = X_L.device if X_L is not None else tok_idx.device
    L = len(tok_idx)
    I = len(f["restype"])

    if X_L is None:
        X_L = torch.randn(
            (1, L, 3), device=device, dtype=torch.float
        )  # [L, 3] - random
    D_LL = torch.cdist(X_L, X_L, p=2)  # [B, L, L] - pairwise atom distances

    # Create attention indices using neighbour distances
    base_mask = ~f["unindexing_pair_mask"][
        tok_idx[None, :], tok_idx[:, None]
    ]  # [n_atoms, n_atoms]
    k_actual = min(n_attn_keys, L)

    # For symmetric structures, ensure inter-chain interactions are included
    chain_ids = f["asym_id"][tok_idx] if "asym_id" in f else None
    if (
        chain_ids is not None and len(torch.unique(chain_ids)) > 3
    ):  # Multi-chain structure
        # Reserve 25% of attention keys for inter-chain interactions
        # while retaining one intra-chain slot so queries with no inter-chain
        # partners still have a valid key (normally their self key).
        k_inter_chain = min(max(k_actual - 1, 0), max(32, k_actual // 4))
        k_intra_chain = k_actual - k_inter_chain

        if k_inter_chain > 0:
            attn_indices, index_mask = get_sparse_attention_indices_with_inter_chain(
                tok_idx,
                D_LL,
                n_seq_neighbours=n_attn_seq_neighbours,
                k_intra=k_intra_chain,
                k_inter=k_inter_chain,
                num_res=I,
                chain_id=chain_ids,
                base_mask=base_mask,
            )
        else:
            attn_indices, index_mask = get_sparse_attention_indices(
                tok_idx,
                D_LL,
                n_seq_neighbours=n_attn_seq_neighbours,
                k_max=k_actual,
                num_res=I,
                chain_id=chain_ids,
                base_mask=base_mask,
            )
    else:
        # Regular attention for single chain or small structures
        attn_indices, index_mask = get_sparse_attention_indices(
            tok_idx,
            D_LL,
            n_seq_neighbours=n_attn_seq_neighbours,
            k_max=k_actual,
            num_res=I,
            chain_id=chain_ids,
            base_mask=base_mask,
        )  # [B, L, k] | indices[b, i, j] = atom index for atom i to j-th attn query

    return attn_indices, index_mask


@torch.no_grad()
def get_sparse_attention_indices_with_inter_chain(
    tok_idx, D_LL, n_seq_neighbours, k_intra, k_inter, num_res, chain_id, base_mask
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    """
    Create attention indices that guarantee inter-chain interactions for clash avoidance.

    Args:
        tok_idx: atom to token mapping
        D_LL: pairwise distances [B, L, L]
        n_seq_neighbours: number of sequence neighbors
        k_intra: number of intra-chain attention keys
        k_inter: number of inter-chain attention keys
        num_res: total number of unique tokens
        chain_id: chain IDs for each atom
        base_mask: base mask for valid pairs

    Returns:
        attn_indices: [B, L, k_total] where k_total = k_intra + k_inter
        index_mask: [B, L, k_total] indicating which sparse slots contain real keys
    """
    # 1. Create disjoint masks for valid intra- and inter-chain pairs.
    # unsqueeze allows us to broadcast the 1D or 2D chain_id into an [L, L] or [B, L, L] matrix
    same_chain_mask = chain_id.unsqueeze(-1) == chain_id.unsqueeze(-2)
    valid_intra_mask = same_chain_mask & base_mask
    valid_inter_mask = (~same_chain_mask) & base_mask

    # 2. Select enough intra-chain candidates to backfill any unused inter slots.
    k_total = k_intra + k_inter
    all_intra_indices, all_intra_mask = get_sparse_attention_indices(
        tok_idx,
        D_LL,
        n_seq_neighbours,
        k_total,
        num_res,
        chain_id,
        valid_intra_mask,
    )  # [B, L, k_total]
    intra_order = torch.argsort(
        all_intra_mask.to(dtype=torch.int8), dim=-1, descending=True, stable=True
    )
    all_intra_indices = all_intra_indices.gather(-1, intra_order)
    all_intra_mask = all_intra_mask.gather(-1, intra_order)
    intra_indices = all_intra_indices[..., :k_intra]
    intra_mask = all_intra_mask[..., :k_intra]
    replacement_indices = all_intra_indices[..., k_intra:]
    replacement_mask = all_intra_mask[..., k_intra:]

    # 3. Select enough inter-chain candidates to backfill unused intra slots.
    masked_D_LL = D_LL.masked_fill(~valid_inter_mask, float("inf"))
    all_inter_distances, all_inter_indices = torch.topk(
        masked_D_LL, k_total, dim=-1, largest=False
    )
    all_inter_mask = torch.isfinite(all_inter_distances)
    inter_order = torch.argsort(
        all_inter_mask.to(dtype=torch.int8), dim=-1, descending=True, stable=True
    )
    all_inter_indices = all_inter_indices.gather(-1, inter_order)
    all_inter_mask = all_inter_mask.gather(-1, inter_order)
    inter_indices = all_inter_indices[..., :k_inter]
    inter_mask = all_inter_mask[..., :k_inter]
    inter_replacement_indices = all_inter_indices[..., k_inter:]
    inter_replacement_mask = all_inter_mask[..., k_inter:]

    # 4. Backfill unused capacity in both directions while keeping the two
    # candidate pools disjoint, so real indices remain unique.
    intra_replacement_positions = (
        (~intra_mask).long().cumsum(dim=-1).sub(1).clamp_min(0)
    )
    inter_replacement_indices = inter_replacement_indices.gather(
        -1, intra_replacement_positions
    )
    inter_replacement_mask = inter_replacement_mask.gather(
        -1, intra_replacement_positions
    )
    intra_indices = torch.where(intra_mask, intra_indices, inter_replacement_indices)
    intra_mask = intra_mask | inter_replacement_mask

    replacement_positions = (~inter_mask).long().cumsum(dim=-1).sub(1).clamp_min(0)
    replacement_indices = replacement_indices.gather(-1, replacement_positions)
    replacement_mask = replacement_mask.gather(-1, replacement_positions)
    inter_indices = torch.where(inter_mask, inter_indices, replacement_indices)
    inter_mask = inter_mask | replacement_mask

    # 5. Combine indices and carry validity explicitly. Values in masked slots
    # remain safe, in-range gather indices but must never participate in attention.
    combined_indices = torch.cat(
        [intra_indices, inter_indices], dim=-1
    )  # [B, L, k_total]
    combined_mask = torch.cat([intra_mask, inter_mask], dim=-1)

    return combined_indices.detach(), combined_mask.detach()


@torch.no_grad()
def build_index_mask(
    tok_idx: torch.Tensor,
    n_sequence_neighbours: int,
    k_max: int,
    num_tok: int,
    chain_id: torch.Tensor | None = None,
    base_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Build a mask that includes entire tokens from neighboring positions within a
    tokenized sequence, never partially including a token. Limits range to k_max,
    which is interpreted at the token level.

    Args:
        tok_idx: (L,) tensor of token indices.
        n_sequence_neighbours: number of tokens to include on either side.
        k_max: max total number of tokens (across both directions).
        num_tok: total number of unique tokens.
        chain_id: (L,) chain identifiers for each position (optional).
        base_mask: (L, L) optional pre-mask to AND with.

    Returns:
        mask: (L, L) boolean tensor where True indicates positions to attend to.
    """
    device = tok_idx.device
    L = tok_idx.shape[0]
    I = num_tok
    k_max = min(k_max, L)
    n_atoms_per_token = torch.zeros(I, device=device).float()
    n_atoms_per_token.scatter_add_(0, tok_idx.long(), torch.ones_like(tok_idx).float())

    # Create index masks for tokens and atoms
    token_indices = torch.arange(I, device=device)
    token_diff = (token_indices[:, None] - token_indices[None, :]).abs()
    atom_indices = torch.arange(L, device=device)
    atom_diff = (atom_indices[:, None] - atom_indices[None, :]).abs()

    # Build token-token mask: [I, I]
    token_mask = token_diff <= n_sequence_neighbours

    # Expand token_mask to full [L, L] mask using broadcast
    # token_to_idx maps each position to a token index [L]
    token_i = tok_idx[:, None]  # (L, 1)
    token_j = tok_idx[None, :]  # (1, L)
    mask = token_mask[token_i, token_j]  # (L, L)
    mask = mask & (atom_diff <= (k_max // 2))

    # Exclude tokens which are partially filled (L, I)
    n_query_per_token = torch.zeros((L, I), device=device).float()
    n_query_per_token.scatter_add_(
        1, tok_idx.long()[None, :].expand(L, -1).contiguous(), mask.float()
    )

    # Find mask for the atoms for which the number of keys
    # match the number of atoms in the token (L, I)
    fully_included = n_query_per_token == n_atoms_per_token[None, :]

    # Contract to (L, L) and count the number of atoms within tokens that
    # fully include other tokens
    n_atoms_fully_included = torch.zeros((I, I), device=device)
    n_atoms_fully_included.index_add_(0, tok_idx.long(), fully_included.float())
    full_token_mask = n_atoms_fully_included == n_atoms_per_token[:, None]

    # Map this back to (L, L) — include token j in row i only if all its atoms are included
    full_token_mask = full_token_mask[token_i, token_j]  # (L, L)
    mask &= full_token_mask

    if chain_id is not None:
        same_chain = chain_id.unsqueeze(-1) == chain_id.unsqueeze(-2)
        mask = mask & same_chain

    if base_mask is not None:
        mask = mask & base_mask

    return mask


def extend_index_mask_with_neighbours(
    mask: torch.Tensor,
    D_LL: torch.Tensor,
    k: int,
    base_mask: torch.Tensor | None = None,
) -> tuple[torch.LongTensor, torch.Tensor]:
    """
    Extend a boolean mask of pre-selected neighbors to ensure exactly k neighbors per query token.

    NOTE: Indices of the mask are placed first along k dimension. E.g.,
           indices[i, :] = [1, 2, 3, nan, nan] (from pre-built mask)
        -> indices[i, :] = [1, 2, 3, 0, 5]  # where 0, 5 are additional k NN (here k=5)

    NOTE: E.g., if k_neigh = 14 * (2*n_seq_neigh + 1) (from above), then for tokens in the middle there will
        be exactly no D_LL-local neighbours, but for tokens at the edges there will be an increasingly
        large number of neighbours.

    Args:
        mask       : (L, L) bool                      # pre-selected neighbours (True = keep)
        D_LL       : (B, L, L) float32/float64        # pairwise distances (lower = closer)
        k          : int                              # desired neighbours per query token
        base_mask  : (L, L)                           # optional additional mask to AND with (e.g. for chain breaks)

    Returns:
        neigh_idx  : (B, L, k_neigh) long             # exactly k_neigh indices per row
        neigh_mask : (B, L, k_neigh) bool             # True for real neighbours, False for padding
    """
    if D_LL.ndim == 2:
        D_LL = D_LL.unsqueeze(0)
    B, L, _ = D_LL.shape
    k = min(k, L)
    assert mask.shape == (L, L) and D_LL.shape == (B, L, L)
    device = D_LL.device
    inf = torch.tensor(float("inf"), dtype=D_LL.dtype, device=device)

    # 1. Selection of sequence neighbours
    # MPS does not handle non-contiguous inputs to torch.where correctly,
    # so use .repeat() (allocates) there; .expand() (zero-copy view) elsewhere.
    if device.type == "mps":
        all_idx_row = torch.arange(L, device=device).unsqueeze(0).repeat(L, 1)
    else:
        all_idx_row = torch.arange(L, device=device).unsqueeze(0).expand(L, L)
    index_sentinel = torch.full((L, L), L, device=device, dtype=torch.long)
    indices = torch.where(mask.contiguous(), all_idx_row, index_sentinel)
    indices = indices.sort(dim=1)[0][:, :k]  # (L, k)
    index_mask = indices != L
    indices = indices.clamp(max=max(L - 1, 0))

    # 2. Find k-nn excluding forced indices
    D_LL = torch.where(mask.contiguous(), inf, D_LL)
    if base_mask is not None:
        # NOTE: `base_mask` prevents different examples in the batch from attending to each other.
        D_LL = D_LL.masked_fill(~base_mask, inf)
    filler_distances, filler_idx = torch.topk(D_LL, k, dim=-1, largest=False)

    # ... Reverse last axis s.t. best matched indices are last
    filler_idx = filler_idx.flip(dims=[-1])
    filler_mask = torch.isfinite(filler_distances).flip(dims=[-1])

    # 3. Fill indices
    to_fill = ~index_mask
    to_fill = to_fill.expand_as(filler_idx).contiguous()
    indices = indices.expand_as(filler_idx).contiguous()
    index_mask = index_mask.expand_as(filler_idx).contiguous()
    indices = torch.where(to_fill, filler_idx, indices)
    index_mask = torch.where(to_fill, filler_mask, index_mask)

    return indices.long(), index_mask  # (B, L, k)


def get_sparse_attention_indices(
    res_idx, D_LL, n_seq_neighbours, k_max, num_res, chain_id=None, base_mask=None
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    """
    Get attention indices for each query token, ensuring that we include entire tokens
    from neighboring positions within a tokenized sequence, and then filling the rest
    with nearest neighbors based on D_LL. Limits range to k_max, which is interpreted
    at the token level.

    Args:
        res_idx: (L,) tensor of token indices.
        D_LL: (B, L, L) pairwise distance tensor.
        n_seq_neighbours: number of tokens to include on either side.
        k_max: max total number of tokens (across both directions).
        num_res: total number of unique tokens.
        chain_id: (L,) chain identifiers for each position (optional).
        base_mask: (L, L) optional pre-mask to AND with.

    Returns:
        indices: (B, L, k_max) tensor of attention indices for each query token, where indices[b, i, j] = atom index for atom i to j-th attn query.
        index_mask: (B, L, k_max) tensor indicating which slots contain real keys.
    """
    mask = build_index_mask(
        res_idx,
        n_seq_neighbours,
        k_max,
        num_res,
        chain_id=chain_id,
        base_mask=base_mask,
    )
    indices, index_mask = extend_index_mask_with_neighbours(
        mask, D_LL, k_max, base_mask=base_mask
    )

    # Sort and assert no duplicate real keys (padded self-indices are allowed).
    indices, sort_order = torch.sort(indices, dim=-1)
    index_mask = index_mask.gather(-1, sort_order)
    duplicate_real_keys = (
        index_mask[..., 1:]
        & index_mask[..., :-1]
        & (indices[..., 1:] == indices[..., :-1])
    )
    if (
        indices.device.type != "mps"
        # and not torch.compiler.is_compiling()
        and duplicate_real_keys.any()
    ):
        raise AssertionError("Tensor has duplicate elements along the last dimension.")

    assert (
        indices.shape[-1] == k_max
    ), f"Expected k_max={k_max} indices, got {indices.shape[-1]} instead."
    # Detach to avoid gradients flowing through graph construction.
    return indices.detach(), index_mask.detach()
