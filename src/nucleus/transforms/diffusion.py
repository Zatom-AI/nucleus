"""Flow-matching transforms compatible with the RFD3 data contract."""

from typing import Any, ClassVar

import torch
from atomworks.ml.transforms._checks import check_contains_keys
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms.diffusion.batch_structures import (
    BatchStructuresForDiffusionNoising,
)

from nucleus.model.flow_matching import interpolate, sample_time
from nucleus.utils.materials import has_material

PPI_PERTURB_COM_SCALE = 1.5


class SampleFlowMatchingNoise(Transform):
    """Sample a linear flow state and expose it as additive ``noise``.

    The existing trainer constructs its input as ``clean + noise``. Storing
    ``x_t - clean`` therefore keeps the surrounding RFD3 batching and trainer
    code unchanged.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [
        BatchStructuresForDiffusionNoising
    ]

    def __init__(
        self,
        *,
        sigma_data: float,
        diffusion_batch_size: int,
        beta_alpha: float = 1.9,
        uniform_probability: float = 0.02,
        min_time: float = 1e-3,
        sigma_perturb_com: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sigma_data = sigma_data
        self.diffusion_batch_size = diffusion_batch_size
        self.beta_alpha = beta_alpha
        self.uniform_probability = uniform_probability
        self.min_time = min_time
        self.sigma_perturb_com = sigma_perturb_com

    def check_input(self, data: dict[str, Any]) -> None:
        """
        Check that the input data contains the required keys and that the coordinates are batched correctly.

        Args:
            data: A dictionary containing the input data.

        Raises:
            ValueError: If the coordinates are not batched correctly.
        """
        check_contains_keys(data, ["coord_atom_lvl_to_be_noised", "feats"])
        coordinates = data["coord_atom_lvl_to_be_noised"]
        if coordinates.shape[0] != self.diffusion_batch_size:
            raise ValueError(
                "Coordinates must be batched before flow noising: expected "
                f"{self.diffusion_batch_size}, got {coordinates.shape[0]}."
            )

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Apply the diffusion transform to the input data.

        Args:
            data: A dictionary containing the input data.

        Returns:
            A dictionary containing the transformed data.
        """
        clean = data["coord_atom_lvl_to_be_noised"]
        time = sample_time(
            self.diffusion_batch_size,
            beta_alpha=self.beta_alpha,
            uniform_probability=self.uniform_probability,
            min_time=self.min_time,
            device=clean.device,
            dtype=clean.dtype,
        )
        fixed_atom_mask = data["feats"]["is_motif_atom_with_fixed_coord"].to(
            device=clean.device, dtype=torch.bool
        )
        valid_atom_mask = data.get("ground_truth", {}).get("mask_atom_lvl")
        if valid_atom_mask is None:
            valid_atom_mask = torch.ones_like(fixed_atom_mask)
        valid_atom_mask = valid_atom_mask.to(device=clean.device, dtype=torch.bool)
        is_unindexed_motif = data["feats"].get("is_motif_atom_unindexed")
        if is_unindexed_motif is not None:
            valid_atom_mask &= ~is_unindexed_motif.to(
                device=clean.device, dtype=torch.bool
            )

        def masked_center(mask: torch.Tensor) -> torch.Tensor:
            """
            Compute the center of the masked atoms.

            Args:
                mask: A boolean tensor of shape (B, L) indicating which atoms to include.

            Returns:
                A tensor of shape (B, 1, 3) representing the center of the masked atoms.
            """
            weights = mask.to(clean.dtype)[None, :, None]
            return (torch.nan_to_num(clean) * weights).sum(
                dim=1, keepdim=True
            ) / weights.sum(dim=1, keepdim=True).clamp_min(1)

        target_center = masked_center(valid_atom_mask)
        motif_center_mask = fixed_atom_mask & valid_atom_mask
        if torch.any(motif_center_mask):
            motif_center = masked_center(motif_center_mask)
            use_motif_center = (
                torch.rand(self.diffusion_batch_size, 1, 1, device=clean.device) < 0.5
            )
            prior_center = torch.where(use_motif_center, motif_center, target_center)
        else:
            prior_center = target_center

        # Jitter the prior rather than the interpolated state so the path remains
        # linear and the displacement naturally decays as (1 - t).
        atom_array = data.get("atom_array")
        is_periodic_material = atom_array is not None and has_material(atom_array)
        if (
            self.sigma_perturb_com > 0
            and not is_periodic_material
            and (not data.get("is_inference", False) or not fixed_atom_mask.any())
        ):
            sigma_perturb_com = self.sigma_perturb_com
            if data.get("sampled_condition_name") == "ppi":
                sigma_perturb_com *= PPI_PERTURB_COM_SCALE
            prior_center = prior_center + sigma_perturb_com * torch.randn(
                (self.diffusion_batch_size, 1, 3),
                device=clean.device,
                dtype=clean.dtype,
            )

        prior = prior_center + self.sigma_data * torch.randn_like(clean)
        prior[:, fixed_atom_mask] = clean[:, fixed_atom_mask]

        state = interpolate(clean, prior, time)
        data["t"] = time
        data["noise"] = state - clean
        return data
