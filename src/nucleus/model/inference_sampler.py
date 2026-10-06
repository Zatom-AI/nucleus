from typing import Any

import torch

from nucleus.model.flow_matching import (
    FLOW_MATCHING,
    edm_noise_to_flow_time,
    edm_state_to_flow_state,
)

try:
    from rfd3na.model.inference_sampler import ConditionalDiffusionSampler
except ImportError:
    raise ImportError(
        "Could not import RFD3NA inference sampler. Please ensure that the `rc-foundry[all]` package is installed and up to date."
    )


class FlowMatchingEDMAdapter:
    """Present a flow velocity model as a denoiser to the existing EDM sampler."""

    def __init__(self, flow_module):
        self.flow_module = flow_module
        self.sigma_data = flow_module.sigma_data

    def __call__(
        self,
        *,
        X_noisy_L: torch.Tensor,
        t: torch.Tensor,
        f: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Convert the denoiser interface to a flow velocity interface, and call the flow module.

        Args:
            X_noisy_L: Noisy atomic coordinates of shape (L, 3).
            t: Noise level of shape (1,).
            f: Feature dictionary containing additional information.

        Returns:
            A dictionary containing the flow velocity and any additional outputs from the flow module.
        """
        flow_time = edm_noise_to_flow_time(t, self.sigma_data)
        flow_state = edm_state_to_flow_state(
            X_noisy_L,
            flow_time,
            fixed_atom_mask=f["is_motif_atom_with_fixed_coord"],
        )
        return self.flow_module(
            X_noisy_L=flow_state,
            t=flow_time,
            f=f,
            **kwargs,
        )


class ZatomConditionalDiffusionSampler(ConditionalDiffusionSampler):
    """RFD3NA sampler that preserves auxiliary outputs from the final step."""

    def __init__(self, kind="default", *, n_recycle: int | None = None, **kwargs):
        """Keep the denoiser pass override even when the parent drops it.

        Args:
            kind: Registered sampler implementation.
            n_recycle: Total denoiser passes; one disables recycling and None
                keeps the checkpoint's default.
            **kwargs: Arguments forwarded to the underlying sampler.
        """
        if n_recycle is not None and (
            isinstance(n_recycle, bool)
            or not isinstance(n_recycle, int)
            or n_recycle < 1
        ):
            raise ValueError("n_recycle must be a positive integer or None.")
        self.n_recycle = n_recycle
        super().__init__(kind=kind, **kwargs)

    def sample_diffusion_like_af3(self, **kwargs: Any) -> dict[str, Any]:
        diffusion_module = kwargs["diffusion_module"]
        f = kwargs["f"]
        auxiliary_outputs = {}

        sampling_module = (
            FlowMatchingEDMAdapter(diffusion_module)
            if getattr(diffusion_module, "f_pred", None) == FLOW_MATCHING
            else diffusion_module
        )

        def record_auxiliary_outputs(*args, **module_kwargs):
            # RFD3NA does not forward this option to either denoiser branch.
            # Apply it here so both conditional and reference CFG calls agree.
            n_recycle = getattr(self, "n_recycle", None)
            if n_recycle is not None:
                module_kwargs["n_recycle"] = n_recycle
            outputs = sampling_module(*args, **module_kwargs)
            if module_kwargs.get("f") is f and isinstance(outputs, dict):
                for key in (
                    "atom_type_logits_I",
                    "atom_type_indices_I",
                    "normalized_energy_B",
                    "normalized_forces_L",
                    "fractional_coord_L",
                    "lattice_lengths_B",
                    "lattice_angles_B",
                ):
                    if key in outputs:
                        auxiliary_outputs[key] = outputs[key]
                if "fractional_coord_L" in outputs and "X_L" in outputs:
                    # Pair the cell-frame reference with the conditional heads,
                    # not the reference CFG branch or the guided endpoint.
                    auxiliary_outputs["material_frame_reference_X_L"] = outputs["X_L"]
            return outputs

        sample = super().sample_diffusion_like_af3(
            **{**kwargs, "diffusion_module": record_auxiliary_outputs}
        )
        if self.sampler.allow_realignment and torch.any(
            f["is_motif_atom_with_fixed_coord"]
        ):
            # The parent transforms/reinserts the motif after the last forward
            # pass. That endpoint no longer shares the reference's frame.
            auxiliary_outputs.pop("material_frame_reference_X_L", None)
        for key, value in auxiliary_outputs.items():
            sample.setdefault(key, value)
        return sample
