import torch


def build_valid_mask(tok_idx: torch.Tensor, n_atoms_per_tok_max: int) -> torch.Tensor:
    """
    Build a boolean mask of valid atoms for each token.

    Args:
        tok_idx: (n_atoms,) - Non-negative integer array
        n_atoms_per_tok_max: Pad/truncate up to this size

    Returns:
        valid_mask: (n_tokens, A) - True where an atom exists
    """
    A = n_atoms_per_tok_max
    _, counts = torch.unique(tok_idx, return_counts=True)

    # Build [n_tokens, A] mask; broadcasting keeps it vectorised
    atom_idx_grid = torch.arange(A, device=tok_idx.device)[None, :]  # (1, A)
    valid_mask = atom_idx_grid < counts[:, None]  # (n_tok, A)

    return valid_mask
