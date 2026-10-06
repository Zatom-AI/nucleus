import torch

from nucleus.utils.distogram import (
    bucketize_scaled_distogram,
    bucketize_scaled_distogram_original,
)


def test_bucketize_scaled_distogram_matches_original():
    """Test that the new bucketize_scaled_distogram implementation matches the original one."""
    torch.manual_seed(0)
    r_l = torch.rand(2, 4, 3) * 30

    distogram = bucketize_scaled_distogram(r_l)
    distogram_original = bucketize_scaled_distogram_original(r_l)

    assert distogram.shape == (2, 4, 4, 65)
    assert torch.equal(
        distogram, distogram_original
    ), "The new implementation does not match the original."
