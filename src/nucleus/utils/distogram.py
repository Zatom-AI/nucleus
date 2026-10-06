import torch
import torch.nn.functional as F


def bucketize_scaled_distogram_original(
    R_L, min_dist=1, max_dist=30, sigma_data=16, n_bins=65
):
    """
    Bucketizes pairwise distances into bins based on edm scaling

    min dist and max dist given as angstroms
    Will use bin ranges based on scaled angstrom distances

    R_L: B, N, 3
    D_LL: B, N, N
    D_LL_binned: B, N, N, n_bins
    """
    D_LL = R_L.unsqueeze(-2) - R_L.unsqueeze(-3)  # [B, N, N, 3]
    D_LL = torch.linalg.norm(D_LL, dim=-1)  # [B, N, N]

    # normalize
    min_dist, max_dist = min_dist / sigma_data, max_dist / sigma_data

    bins = torch.linspace(min_dist, max_dist, n_bins - 1, device=D_LL.device)
    bin_idxs = torch.bucketize(D_LL, bins)
    return F.one_hot(bin_idxs, num_classes=len(bins) + 1).float()


def bucketize_scaled_distogram(
    R_L,
    min_dist=1,
    max_dist=30,
    sigma_data=16,
    n_bins=65,
):
    """
    Bucketize pairwise distances into a one-hot distogram representation.

    This is written without Python-side shape extraction so it stays friendly to
    tracing / compilation paths used by FlexAttention.
    """
    D_LL = R_L.unsqueeze(-2) - R_L.unsqueeze(-3)  # [B, N, N, 3]
    D_LL = torch.linalg.norm(D_LL, dim=-1)  # [B, N, N]

    min_dist = min_dist / sigma_data
    max_dist = max_dist / sigma_data
    bins = torch.linspace(
        min_dist,
        max_dist,
        n_bins - 1,
        device=D_LL.device,
    )

    # Equivalent to torch.bucketize(..., bins) but keeps the implementation
    # tracing-friendly for symbolic shapes.
    bin_idxs = (D_LL.unsqueeze(-1) > bins).sum(dim=-1).to(dtype=torch.long)
    return (bin_idxs[..., None] == torch.arange(n_bins, device=D_LL.device)).float()
