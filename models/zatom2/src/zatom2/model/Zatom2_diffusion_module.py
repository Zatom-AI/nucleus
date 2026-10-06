from __future__ import annotations

import functools
from collections.abc import Mapping
from contextlib import ExitStack
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
from foundry.model.layers.blocks import FourierEmbedding
from nucleus.metrics.tensor_diagnostics import summarize_zatom_activation_scales
from nucleus.model.flow_matching import FLOW_MATCHING, endpoint_from_velocity
from nucleus.model.layers.attention import prepare_token_attention_inputs
from nucleus.model.layers.backbone import TransformerBackbone
from nucleus.model.layers.block_utils import create_attention_indices
from nucleus.model.layers.blocks import (
    AdaLNCoordinatesHead,
    AdaLNFeatureProcessor,
    AtomisticEnergyHead,
    Downcast,
    LatentEquivarianceHead,
    LinearAtomTypesHead,
    LinearEmbedWithPool,
    LinearLatticeHead,
)
from nucleus.model.self_embedding_guidance import SelfEmbeddingConditioner
from nucleus.utils.distogram import bucketize_scaled_distogram
from nucleus.utils.torch import (
    expand_t_to_level,
    get_device_compatible_float64_dtype,
)
from rfd3.model.layers.layer_utils import RMSNorm, linearNoBias
from rfd3.model.RFD3_diffusion_module import LinearSequenceHead

if TYPE_CHECKING:
    from nucleus.utils.atomistic_properties import AtomisticPropertyStatistics


class Zatom2DiffusionModule(nn.Module):
    """Zatom-2 diffusion module with a shared atom/token backbone.

    UNet-like architecture for diffusion with optional recycling,
    supporting molecules, materials, and proteins and dataloader
    batch_size > 1.

    Args:
        c_atom: Atom-level feature dimension
        c_s: Sequence-level feature dimension
        c_t_embed: Time embedding dimension
        n_token_head: Number of attention heads for token-level attention
        sigma_data: Sigma for scaling positions in and out of the model
        f_pred: Coordinate objective ("edm", "flow_matching", "noise_pred", or
            "unconditioned")
        n_token_attn_seq_neighbours: Number of sequential neighbors to attend to in token local attention
        n_token_attn_keys: Number of keys to attend to in token local attention
        n_atom_attn_seq_neighbours: Number of sequential neighbors to attend to in atom local attention
        n_atom_attn_keys: Number of keys to attend to in atom local attention
        n_recycle: Number of recycling iterations during inference
        atom_encoder: Config for the LocalAtomTransformer encoder
        token_trunk: Config for the TokenTrunk in the diffusion block
        atom_decoder: Config for the CompactStreamingDecoder
        downcast: Config for the Downcast layers
        upcast: Config for the Upcast layers
        use_local_token_attention: Whether to use local attention for tokens (default: False)
        self_embedding_guidance_enabled: Whether to enable SCD conditioning support
        self_embedding_guidance_probability: Training probability of using SCD
        self_embedding_atom_condition_probability: Probability of pooling clean atom
            features instead of clean token features for the SCD condition
        self_embedding_condition_dropout: Probability of dropping the SCD condition
        self_embedding_condition_clip_value: Clamp value for the pooled SCD condition
        activation_scale_diagnostics: Whether to emit activation-scale diagnostics
        token_head_architecture: Processing applied before trunk-derived sequence,
            atom-type, lattice, and energy prediction heads. ``linear`` retains
            legacy behavior; ``adaln_zero`` applies head-local AdaLN-Zero.
    """

    def __init__(
        self,
        *,
        c_atom: int,
        c_s: int,
        c_t_embed: int,
        n_token_head: int,
        sigma_data: float,
        f_pred: Literal["edm", "flow_matching", "unconditioned", "noise_pred"],
        n_token_attn_seq_neighbours: int,
        n_token_attn_keys: int,
        n_atom_attn_seq_neighbours: int,
        n_atom_attn_keys: int,
        n_recycle: int | None,
        atom_encoder: dict,
        token_trunk: dict,
        atom_decoder: dict,
        downcast: dict,
        upcast: dict,
        use_local_token_attention: bool = False,
        self_embedding_guidance_enabled: bool = False,
        self_embedding_guidance_probability: float = 0.25,
        self_embedding_atom_condition_probability: float = 0.5,
        self_embedding_condition_dropout: float = 0.2,
        self_embedding_condition_clip_value: float = 100.0,
        atomistic_property_head: Mapping | None = None,
        activation_scale_diagnostics: bool = False,
        token_head_architecture: Literal["linear", "adaln_zero"] = "linear",
        **_,
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.c_atom = c_atom
        self.c_s = c_s
        self.f_pred = f_pred
        self.n_token_attn_seq_neighbours = n_token_attn_seq_neighbours
        self.n_token_attn_keys = n_token_attn_keys
        self.n_atom_attn_seq_neighbours = n_atom_attn_seq_neighbours
        self.n_atom_attn_keys = n_atom_attn_keys
        self.n_recycle = n_recycle
        self.use_local_token_attention = use_local_token_attention
        self.self_embedding_guidance_enabled = self_embedding_guidance_enabled
        self.self_embedding_guidance_probability = self_embedding_guidance_probability
        self.activation_scale_diagnostics = activation_scale_diagnostics
        self.token_head_architecture = token_head_architecture

        # Geometry processing
        self.process_r_token = LinearEmbedWithPool(c_s)
        self.process_r_atom = linearNoBias(3, c_atom)
        self.process_s = RMSNorm(c_s)
        self.process_c = RMSNorm(c_atom)

        # Time processing
        self.fourier_embedding = nn.ModuleList(
            [FourierEmbedding(c_t_embed), FourierEmbedding(c_t_embed)]
        )
        self.process_n = nn.ModuleList(
            [
                nn.Sequential(RMSNorm(c_t_embed), linearNoBias(c_t_embed, c_atom)),
                nn.Sequential(RMSNorm(c_t_embed), linearNoBias(c_t_embed, c_s)),
            ]
        )

        # Self-embedding guidance
        self.self_embedding_conditioner = (
            SelfEmbeddingConditioner(
                c_token=c_s,
                c_s=c_s,
                c_atom=c_atom,
                atom_probability=self_embedding_atom_condition_probability,
                condition_dropout=self_embedding_condition_dropout,
                clip_value=self_embedding_condition_clip_value,
            )
            if self.self_embedding_guidance_enabled
            else None
        )

        # Recyling guidance
        self.n_distogram_bins = 65
        self.bucketize_fn = functools.partial(
            bucketize_scaled_distogram,
            min_dist=1,
            max_dist=30,
            sigma_data=1,  # NOTE: Must be 1 for distogram processing, since the distogram is computed from scaled positions
            n_bins=self.n_distogram_bins,
        )
        self.process_distogram_rec = linearNoBias(self.n_distogram_bins, n_token_head)

        # UNet-like architecture for processing across tokens and atoms
        self.backbone = TransformerBackbone(
            c_atom=c_atom,
            c_s=c_s,
            atom_encoder=atom_encoder,
            atom_decoder=atom_decoder,
            token_trunk=token_trunk,
            downcast=downcast,
            upcast=upcast,
        )

        # Output processing
        self.euclidean_coord_head = AdaLNCoordinatesHead(c_atom=c_atom)
        self.sequence_head = LinearSequenceHead(c_token=c_s)
        self.atom_type_head = LinearAtomTypesHead(c_token=c_s)
        self.sequence_head_processor = self._make_prediction_head_processor(c_s)
        self.atom_type_head_processor = self._make_prediction_head_processor(c_s)
        self.fractional_coord_head = AdaLNCoordinatesHead(c_atom=c_atom)
        self.lattice_init_token = nn.Parameter(torch.zeros(1, 1, c_s))
        self.downcast_lattice = Downcast(c_atom=c_s, c_token=c_s, c_s=None, **downcast)
        self.lattice_head_processor = self._make_prediction_head_processor(c_s)
        self.lattice_head = LinearLatticeHead(c_token=c_s)
        if self.token_head_architecture == "adaln_zero":
            self._zero_initialize_output_heads()

        # Optional property heads. Keeping these disabled unless an
        # explicit config is supplied preserves released checkpoint structure.
        property_config = dict(atomistic_property_head or {})
        self.atomistic_property_prediction_enabled = bool(
            property_config.get("enabled", bool(property_config))
        )
        self.latent_equivariance_enabled = False
        if self.atomistic_property_prediction_enabled:
            statistics = self._load_property_statistics(property_config)
            self._property_statistics = statistics
            if statistics.domain_names != ("omol25", "omat24"):
                raise ValueError(
                    "Zatom-2 property heads require domain IDs 0=omol25 and "
                    f"1=omat24; got {statistics.domain_names}."
                )
            # Fabric constructs modules under a default-device context. MPS
            # cannot allocate float64 tensors, so retain float64 statistics on
            # supported devices and use the highest supported precision on MPS.
            statistics_tensors = statistics.tensors(
                max_atomic_number=127,
                dtype=get_device_compatible_float64_dtype(),
            )
            self.property_domain_names = statistics.domain_names
            self.register_buffer(
                "property_elemental_references",
                statistics_tensors["elemental_references"],
            )
            self.register_buffer(
                "property_elemental_reference_valid",
                statistics_tensors["elemental_reference_valid"],
            )
            self.register_buffer(
                "property_energy_mean",
                statistics_tensors["energy_mean"],
            )
            self.register_buffer(
                "property_energy_scale",
                statistics_tensors["energy_scale"],
            )
            self.register_buffer(
                "property_force_scale",
                statistics_tensors["force_scale"].float(),
            )
            self.energy_head_processor = self._make_prediction_head_processor(c_s)
            self.energy_head = AtomisticEnergyHead(
                c_token=c_s,
                n_domains=len(self.property_domain_names),
                hidden_dim=property_config.get("energy_hidden_dim"),
            )
            # Force prediction consumes atom features rather than Q_I;
            # AdaLNCoordinatesHead already applies atom-level AdaLN preprocessing.
            self.force_head = AdaLNCoordinatesHead(c_atom=c_atom)
            latent_equivariance_config = dict(
                property_config.get("latent_equivariance") or {}
            )
            self.latent_equivariance_enabled = bool(
                latent_equivariance_config.get(
                    "enabled", bool(latent_equivariance_config)
                )
            )
            if self.latent_equivariance_enabled:
                self.latent_equivariance_head_processor = (
                    self._make_prediction_head_processor(c_atom)
                )
                self.latent_equivariance_head = LatentEquivarianceHead(
                    feature_dim=c_atom,
                    hidden_multiplier=latent_equivariance_config.get(
                        "hidden_multiplier", 2.0
                    ),
                )

    def _make_prediction_head_processor(self, c_token: int) -> nn.Module | None:
        """Build the configured processor for one trunk-derived prediction head."""
        if self.token_head_architecture == "linear":
            return None
        if self.token_head_architecture == "adaln_zero":
            return AdaLNFeatureProcessor(c_features=c_token)
        raise ValueError(
            f"Unknown token_head_architecture: {self.token_head_architecture!r}"
        )

    @staticmethod
    def _process_prediction_head_features(
        features: torch.Tensor,
        conditioning: torch.Tensor,
        processor: nn.Module | None,
    ) -> torch.Tensor:
        """Apply head-local AdaLN while leaving shared trunk outputs unchanged."""
        return features if processor is None else processor(features, conditioning)

    def _zero_initialize_output_heads(self) -> None:
        """Zero-initialize projections following head-local AdaLN processors."""
        projections = (
            self.sequence_head.linear,
            self.atom_type_head.linear,
            self.lattice_head.linear_lengths,
            self.lattice_head.linear_angles,
        )
        for projection in projections:
            nn.init.zeros_(projection.weight)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

    @staticmethod
    def _mean_pool_features(
        features: torch.Tensor,
        batch_level: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Mean-pool token- or atom-level features per input structure."""
        batch_indices = batch_level.to(features.device)
        scatter_indices = batch_indices[None, :, None].expand(
            features.shape[0], -1, features.shape[-1]
        )
        pooled = features.new_zeros(features.shape[0], batch_size, features.shape[-1])
        pooled.scatter_add_(1, scatter_indices, features)
        counts = torch.bincount(batch_indices, minlength=batch_size).clamp_min(1)
        return pooled / counts.to(features.dtype)[None, :, None]

    @staticmethod
    def _load_property_statistics(
        property_config: Mapping,
    ) -> AtomisticPropertyStatistics:
        """Load property normalization values from a versioned artifact or config."""
        from nucleus.utils.atomistic_properties import AtomisticPropertyStatistics

        if property_config.get("statistics_path") is not None:
            return AtomisticPropertyStatistics.load(property_config["statistics_path"])

        raw_statistics = property_config.get("statistics")
        if isinstance(raw_statistics, AtomisticPropertyStatistics):
            return raw_statistics
        if raw_statistics is not None:
            return AtomisticPropertyStatistics.from_dict(raw_statistics)

        required_inline_keys = (
            "domain_names",
            "energy_mean",
            "energy_scale",
            "force_scale",
            "elemental_references",
        )
        missing = [key for key in required_inline_keys if key not in property_config]
        if missing:
            raise ValueError(
                "Enabled atomistic property heads require `statistics_path`, "
                "`statistics`, or complete inline statistics; missing "
                f"{missing}."
            )
        return AtomisticPropertyStatistics.from_values(
            domain_names=property_config["domain_names"],
            energy_mean=property_config["energy_mean"],
            energy_scale=property_config["energy_scale"],
            force_scale=property_config["force_scale"],
            elemental_references=property_config["elemental_references"],
        )

    def _property_domain_ids(
        self,
        f: dict,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Resolve explicit domain IDs or infer molecule/material IDs."""
        if "domain_id_B" in f:
            domain_id_B = torch.as_tensor(f["domain_id_B"], device=device).long()
            domain_id_B = domain_id_B.reshape(-1)
            if domain_id_B.numel() != batch_size:
                raise ValueError(
                    "domain_id_B must contain one domain ID per system; got "
                    f"{domain_id_B.numel()} values for batch size {batch_size}."
                )
            return domain_id_B

        if "is_material_token" not in f:
            return torch.full((batch_size,), -1, device=device, dtype=torch.long)

        batch_token_lvl = f["batch_token_lvl"].to(device=device, dtype=torch.long)
        is_material_I = f["is_material_token"].to(device=device, dtype=torch.bool)
        is_material_I = is_material_I.reshape(batch_token_lvl.numel(), -1).any(dim=-1)
        domain_id_B = torch.zeros(batch_size, device=device, dtype=torch.long)
        domain_id_B.index_add_(0, batch_token_lvl, is_material_I.long())
        return domain_id_B.clamp_max(1)

    @staticmethod
    def _property_atom_counts(
        f: dict,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return retained atom counts, with a sparse-batch fallback."""
        if "retained_atom_count_B" in f:
            atom_count_B = torch.as_tensor(
                f["retained_atom_count_B"], device=device, dtype=dtype
            ).reshape(-1)
            if atom_count_B.numel() != batch_size:
                raise ValueError(
                    "retained_atom_count_B must contain one value per system."
                )
            return atom_count_B
        return torch.bincount(
            f["batch_atom_lvl"].to(device=device, dtype=torch.long),
            minlength=batch_size,
        ).to(dtype=dtype)

    def reconstruct_atomistic_properties(
        self,
        outputs: dict,
        f: dict,
        *,
        atom_types: Literal["known", "masked"] = "known",
        allow_missing_elemental_references: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Convert normalized head outputs to eV and eV/angstrom.

        Elemental-reference energies are summed using exact input atom types in
        known mode and expected (soft) predicted counts in masked mode.

        When ``allow_missing_elemental_references`` is true, systems containing
        an unsupported known element retain physical force predictions but
        receive a NaN physical energy and a false reconstruction-valid mask.
        """
        if not self.atomistic_property_prediction_enabled:
            raise RuntimeError("Atomistic property heads are not enabled.")
        if atom_types not in ("known", "masked"):
            raise ValueError("atom_types must be either 'known' or 'masked'.")

        normalized_energy_B = outputs["normalized_energy_B"]
        normalized_forces_L = outputs["normalized_forces_L"]
        device = normalized_energy_B.device
        # Elemental-reference reconstruction is cancellation-sensitive for large
        # OMol25 total energies. Use float64 for this small post-processing path
        # where supported, with float32 as the MPS fallback; network activations
        # and force reconstruction retain their usual precision.
        energy_dtype = get_device_compatible_float64_dtype(device)
        normalized_energy_for_reconstruction_B = normalized_energy_B.to(
            dtype=energy_dtype
        )
        batch_size = normalized_energy_B.shape[-1]
        domain_id_B = self._property_domain_ids(f, batch_size=batch_size, device=device)
        n_domains = len(self.property_domain_names)
        valid_domain_B = (domain_id_B >= 0) & (domain_id_B < n_domains)
        if not valid_domain_B.all():
            invalid = (~valid_domain_B).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"Cannot reconstruct properties for domains {invalid}.")
        domain_id_B = domain_id_B.clamp(min=0, max=n_domains - 1)
        atom_count_B = self._property_atom_counts(
            f,
            batch_size=batch_size,
            device=device,
            dtype=energy_dtype,
        )

        batch_atom_lvl = f["batch_atom_lvl"].to(device=device, dtype=torch.long)
        domain_id_L = domain_id_B[batch_atom_lvl]
        elemental_reference_L = self.property_elemental_references[domain_id_L].to(
            dtype=energy_dtype
        )
        elemental_reference_valid_L = self.property_elemental_reference_valid[
            domain_id_L
        ]

        if atom_types == "known":
            if "ref_element" not in f:
                raise ValueError("Known atom-type inference requires f['ref_element'].")
            ref_element = f["ref_element"].to(device=device)
            if ref_element.numel() == batch_atom_lvl.numel():
                atomic_number_L = ref_element.reshape(-1).long()
                valid_atomic_number_L = (atomic_number_L >= 0) & (
                    atomic_number_L < elemental_reference_L.shape[-1]
                )
                safe_atomic_number_L = atomic_number_L.clamp(
                    min=0, max=elemental_reference_L.shape[-1] - 1
                )
                reference_is_available_L = elemental_reference_valid_L.gather(
                    -1, safe_atomic_number_L.unsqueeze(-1)
                ).squeeze(-1)
                energy_reference_valid_L = (
                    valid_atomic_number_L & reference_is_available_L
                )
                if (
                    not allow_missing_elemental_references
                    and not energy_reference_valid_L.all()
                ):
                    raise ValueError(
                        "Known atom types include an atomic number without an "
                        "elemental reference for its domain."
                    )
                elemental_energy_L = elemental_reference_L.gather(
                    -1, safe_atomic_number_L.unsqueeze(-1)
                ).squeeze(-1)
                elemental_energy_L = elemental_energy_L.masked_fill(
                    ~energy_reference_valid_L, 0.0
                )
                elemental_energy_L = elemental_energy_L.unsqueeze(0).expand(
                    normalized_energy_B.shape[0], -1
                )
            elif ref_element.ndim >= 2 and ref_element.shape[-1] == 128:
                element_probability_L = ref_element.reshape(-1, 128).to(
                    dtype=energy_dtype
                )
                probability_sum_L = element_probability_L.sum(dim=-1)
                invalid_probability_L = (
                    element_probability_L
                    * (~elemental_reference_valid_L).to(dtype=energy_dtype)
                ).sum(dim=-1)
                probabilities_are_normalized = torch.allclose(
                    probability_sum_L,
                    torch.ones_like(probability_sum_L),
                    atol=1e-6,
                    rtol=0.0,
                )
                if not probabilities_are_normalized:
                    raise ValueError("Known ref_element vectors must be normalized.")
                energy_reference_valid_L = invalid_probability_L <= 1e-6
                if (
                    not allow_missing_elemental_references
                    and not energy_reference_valid_L.all()
                ):
                    raise ValueError(
                        "Known ref_element vectors must have available elemental "
                        "references."
                    )
                elemental_energy_L = (
                    element_probability_L * elemental_reference_L
                ).sum(dim=-1)
                elemental_energy_L = elemental_energy_L.masked_fill(
                    ~energy_reference_valid_L, 0.0
                )
                elemental_energy_L = elemental_energy_L.unsqueeze(0).expand(
                    normalized_energy_B.shape[0], -1
                )
            else:
                raise ValueError(
                    "ref_element must contain integer atomic numbers [L] or "
                    "128-channel element vectors [L, 128]."
                )
        else:
            atom_type_logits_I = outputs["atom_type_logits_I"]
            if atom_type_logits_I.dtype in (torch.float16, torch.bfloat16):
                atom_type_logits_I = atom_type_logits_I.float()
            atom_to_token_map = f["atom_to_token_map"].to(
                device=device, dtype=torch.long
            )
            atom_type_logits_L = atom_type_logits_I[:, atom_to_token_map, :]
            valid_element_L = (
                elemental_reference_valid_L
                & self.atom_type_head.valid_out_mask.to(device=device).unsqueeze(0)
            )
            if not valid_element_L.any(dim=-1).all():
                raise ValueError(
                    "At least one system has no atom types with elemental references."
                )
            masked_logits_L = atom_type_logits_L.masked_fill(
                ~valid_element_L.unsqueeze(0),
                torch.finfo(atom_type_logits_L.dtype).min,
            )
            element_probability_L = torch.softmax(masked_logits_L, dim=-1).to(
                dtype=energy_dtype
            )
            elemental_energy_L = (
                element_probability_L * elemental_reference_L.unsqueeze(0)
            ).sum(dim=-1)
            energy_reference_valid_L = torch.ones_like(batch_atom_lvl, dtype=torch.bool)

        invalid_reference_count_B = torch.zeros(
            batch_size, dtype=torch.long, device=device
        ).scatter_add(
            0,
            batch_atom_lvl,
            (~energy_reference_valid_L).to(dtype=torch.long),
        )
        energy_reconstruction_valid_B = invalid_reference_count_B == 0

        elemental_energy_B = normalized_energy_for_reconstruction_B.new_zeros(
            (normalized_energy_B.shape[0], batch_size)
        )
        elemental_energy_B.index_add_(1, batch_atom_lvl, elemental_energy_L)

        energy_mean_B = self.property_energy_mean[domain_id_B].to(dtype=energy_dtype)
        energy_scale_B = self.property_energy_scale[domain_id_B].to(dtype=energy_dtype)
        residual_energy_per_atom_B = (
            normalized_energy_for_reconstruction_B * energy_scale_B.unsqueeze(0)
            + energy_mean_B.unsqueeze(0)
        )
        energy_B = (
            residual_energy_per_atom_B * atom_count_B.unsqueeze(0) + elemental_energy_B
        )
        energy_B = energy_B.masked_fill(
            ~energy_reconstruction_valid_B.unsqueeze(0), float("nan")
        )

        force_dtype = (
            torch.float32
            if normalized_forces_L.dtype in (torch.float16, torch.bfloat16)
            else normalized_forces_L.dtype
        )
        force_scale_L = self.property_force_scale[domain_id_L].to(dtype=force_dtype)
        forces_L = normalized_forces_L.to(dtype=force_dtype) * force_scale_L.view(
            1, -1, 1
        )
        return {
            "energy_B": energy_B,
            "forces_L": forces_L,
            "energy_reconstruction_valid_B": energy_reconstruction_valid_B.unsqueeze(
                0
            ).expand(normalized_energy_B.shape[0], -1),
            "residual_energy_per_atom_B": residual_energy_per_atom_B,
            "elemental_reference_energy_B": elemental_energy_B,
        }

    def scale_positions_in(
        self, X_noisy_L: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Scale positions into the model based on the noise level and prediction type.

        Args:
            X_noisy_L: Noisy input positions [B, L, 3]
            t: Noise level tensor [B] or [B, L]

        Returns:
            R_noisy_L: Scaled positions for input to the model [B, L, 3]
        """
        if t.ndim == 1:
            t = t[..., None, None]  # [B, (n_atoms), (3)]
        elif t.ndim == 2:
            t = t[..., None]  # [B, n_atoms, (3)]

        if self.f_pred == "edm":
            return X_noisy_L / torch.sqrt(t**2 + self.sigma_data**2)
        if self.f_pred == FLOW_MATCHING:
            return X_noisy_L / self.sigma_data
        if self.f_pred == "unconditioned":
            return torch.zeros_like(X_noisy_L)
        if self.f_pred == "noise_pred":
            return X_noisy_L
        raise ValueError(f"{self.f_pred=} unrecognized")

    def scale_positions_out(
        self, R_update_L: torch.Tensor, X_noisy_L: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Scale positions out of the model based on the noise level and prediction type.

        Args:
            R_update_L: Position updates predicted by the model [B, L, 3]
            X_noisy_L: Original noisy input positions [B, L, 3]
            t: Noise level tensor [B] or [B, L]

        Returns:
            X_out_L: Denoised output positions [B, L, 3]
        """
        if t.ndim == 1:
            t = t[..., None, None]
        elif t.ndim == 2:
            t = t[..., None]  # [B, n_atoms, (3)]

        if self.f_pred == "edm":
            return (self.sigma_data**2 / (self.sigma_data**2 + t**2)) * X_noisy_L + (
                self.sigma_data * t / (self.sigma_data**2 + t**2) ** 0.5
            ) * R_update_L
        if self.f_pred == FLOW_MATCHING:
            velocity_L = self.sigma_data * R_update_L
            return endpoint_from_velocity(X_noisy_L, velocity_L, t)
        if self.f_pred == "unconditioned":
            return R_update_L
        if self.f_pred == "noise_pred":
            return X_noisy_L + R_update_L
        raise ValueError(f"{self.f_pred=} unrecognized")

    def process_time_(self, t_L: torch.Tensor, i: int) -> torch.Tensor:
        """
        Process time features for either atom-level (i=0) or sequence-level (i=1) inputs.

        Args:
            t_L: Time tensor for atoms or tokens [B, n_atoms] or [B, n_tokens]
            i: Index to select which time processor to use (0 for atoms, 1 for tokens)

        Returns:
            C_L: Time features to be added to the model inputs [B, n_atoms, c_atom] or [B, n_tokens, c_s]
        """
        time_embedding_input = (
            t_L
            if self.f_pred == FLOW_MATCHING
            else 1 / 4 * torch.log(torch.clamp(t_L, min=1e-20) / self.sigma_data)
        )
        C_L = self.process_n[i](self.fourier_embedding[i](time_embedding_input))
        if self.f_pred == FLOW_MATCHING:
            return C_L
        # Mask out zero-time features
        return C_L * (t_L > 0).float()[..., None]  # [B, L, C_atom]

    def _condition_time_for_fixed_atoms(
        self, time: torch.Tensor, fixed_mask: torch.Tensor
    ) -> torch.Tensor:
        """Use the clean endpoint for fixed atoms under either time convention."""
        fixed_mask = fixed_mask.to(device=time.device, dtype=torch.bool).unsqueeze(0)
        if getattr(self, "f_pred", "edm") == FLOW_MATCHING:
            return torch.where(fixed_mask, torch.ones_like(time), time)
        return time * (~fixed_mask).to(time.dtype)

    def forward(
        self,
        X_noisy_L: torch.Tensor,
        t: torch.Tensor,
        f: dict,
        # Features from initialization
        S_I: torch.Tensor,
        Z_II: torch.Tensor,
        C_L: torch.Tensor,
        P_LL: torch.Tensor,
        n_recycle: int | None = None,
        initializer_outputs: dict | None = None,
        # Self-embedding parameters
        use_self_embedding_guidance: bool = False,
        X_self_embedding_L: torch.Tensor = None,
        f_self_embedding: dict = None,
        self_embedding_initializer_outputs: dict | None = None,
        return_property_embeddings: bool = False,
        # Miscellaneous parameters
        **kwargs,
    ) -> dict:
        """
        Diffusion forward pass with optional recycling.
        Computes denoised positions given encoded features and noisy coordinates.

        Args:
            X_noisy_L: Noisy input positions [B, L, 3]
            t: Noise level tensor [B] or [B, L]
            f: Feature dictionary containing necessary features and masks for attention and pooling
            S_I: Processed single token features [n_tokens, c_s]
            Z_II: Processed pairwise token features [n_tokens, n_tokens, n_token_head]
            C_L: Processed single atom features [L, c_atom]
            P_LL: Processed pairwise atom features [L, L, n_atom_head]
            n_recycle: Number of recycling iterations to perform (if None, uses self.n_recycle)
            initializer_outputs: Optional outputs from the initializer to be used in chunked mode
            use_self_embedding_guidance: Whether to use self-embedding guidance during the forward pass
            X_self_embedding_L: Clean input positions for self-embedding guidance [B, L, 3]
            f_self_embedding: Feature dictionary for self-embedding guidance (can be the same as f)
            self_embedding_initializer_outputs: Outputs from the initializer for self-embedding guidance
            return_property_embeddings: Whether to return property embeddings for property prediction

        Returns:
            outputs: Dictionary containing:
                "X_L": Denoised output positions [B, L, 3]
                "V_L" (optional): Denoised output velocities [B, L, 3] (only if f_pred is "flow_matching")
                "sequence_indices_I": Predicted sequence indices [B, n_tokens]
                "sequence_logits_I": Predicted sequence logits [B, n_tokens, c_s=c_token]
                "atom_type_indices_I": Predicted atom type indices [B, n_tokens]
                "atom_type_logits_I": Predicted atom type logits [B, n_tokens, c_s=c_token]
                "fractional_coord_L": Predicted fractional coordinates [B, L, 3]
                "lattice_lengths_B": Predicted lattice lengths [B, n_examples, 3]
                "lattice_angles_B": Predicted lattice angles [B, n_examples, 3]
        """
        # Prepare time inputs for atoms and tokens
        if kwargs.get("t_L") is None:
            kwargs["t_L"] = expand_t_to_level(
                t,
                f,
                level_len=X_noisy_L.shape[1],
                batch_key="batch_atom_lvl",
            )
        if kwargs.get("t_I") is None:
            kwargs["t_I"] = expand_t_to_level(
                t,
                f,
                level_len=S_I.shape[0],
                batch_key="batch_token_lvl",
            )

        t_L = self._condition_time_for_fixed_atoms(
            kwargs["t_L"], f["is_motif_atom_with_fixed_coord"]
        )
        t_I = self._condition_time_for_fixed_atoms(
            kwargs["t_I"], f["is_motif_token_with_fully_fixed_coord"]
        )

        # ... Maybe perform self-embedding forward pass
        if use_self_embedding_guidance:
            return self.forward_with_self_embedding_guidance(
                X_noisy_L=X_noisy_L,
                X_self_embedding_L=X_self_embedding_L,
                t_L_raw=kwargs["t_L"],
                t_I_raw=kwargs["t_I"],
                f=f,
                f_self_embedding=f_self_embedding
                if f_self_embedding is not None
                else f,
                S_I=S_I,
                Z_II=Z_II,
                C_L=C_L,
                P_LL=P_LL,
                self_embedding_initializer_outputs=(
                    self_embedding_initializer_outputs or {}
                ),
                initializer_outputs=initializer_outputs,
                return_property_embeddings=return_property_embeddings,
            )

        # ... Geometry embedding: fuse scaled positions with token and atom features
        R_L_uniform = self.scale_positions_in(X_noisy_L, kwargs["t_L"])
        R_noisy_L = self.scale_positions_in(X_noisy_L, t_L)

        R_noisy_I_embed = self.process_r_token(
            # TODO: Here, consider using the Fourier projection in Algorithm 17 of the Emyx paper.
            R_noisy_L,
            tok_idx=f["atom_to_token_map"],
            num_tok=S_I.shape[0],
        )
        R_noisy_L_embed = self.process_r_atom(R_noisy_L)

        S_I = self.process_s(S_I + R_noisy_I_embed)
        C_L = self.process_c(C_L + R_noisy_L_embed)

        Q_I = S_I.clone()  # (B, I, c_s), primary single token features
        Q_L = C_L.clone()  # (B, L, c_atom), primary single atom features

        # ... Time embedding: fuse time features with (processed) token and atom features
        S_I_list = [S_I, self.process_time_(t_I, i=1)]
        C_L_list = [C_L, self.process_time_(t_L, i=0)]
        S_I = sum(S_I_list) / len(S_I_list)
        C_L = sum(C_L_list) / len(C_L_list)

        # ... Graph construction: create attention indices for local attention
        G_I, f = prepare_token_attention_inputs(
            f=f,
            X_I=X_noisy_L[..., f["is_ca"], :],
            use_local_attention=self.use_local_token_attention,
            n_attn_keys=self.n_token_attn_keys,
            n_attn_seq_neighbours=self.n_token_attn_seq_neighbours,
        )
        G_L, G_L_mask = create_attention_indices(
            X_L=X_noisy_L,
            f=f,
            n_attn_keys=self.n_atom_attn_keys,
            n_attn_seq_neighbours=self.n_atom_attn_seq_neighbours,
        )

        # ... Reuse validity mask returned by graph construction
        f["atom_pair_mask"] = G_L_mask

        # ... Run forward with recycling
        recycled_features = self.forward_with_recycle(
            n_recycle,
            X_noisy_L=X_noisy_L,
            R_L_uniform=R_L_uniform,
            t_L=t_L,
            f=f,
            Q_I=Q_I,
            Z_II=Z_II,
            S_I=S_I,
            Q_L=Q_L,
            P_LL=P_LL,
            C_L=C_L,
            G_I=G_I,
            G_L=G_L,
            initializer_outputs=initializer_outputs,
        )

        # ... Collect outputs
        outputs = {
            "X_L": recycled_features["X_L"],  # [B, L, 3] denoised positions
            "sequence_indices_I": recycled_features["sequence_indices_I"],
            "sequence_logits_I": recycled_features["sequence_logits_I"],
            "atom_type_indices_I": recycled_features["atom_type_indices_I"],
            "atom_type_logits_I": recycled_features["atom_type_logits_I"],
            "fractional_coord_L": recycled_features["fractional_coord_L"],
            "lattice_lengths_B": recycled_features["lattice_lengths_B"],
            "lattice_angles_B": recycled_features["lattice_angles_B"],
        }
        if "V_L" in recycled_features:
            outputs["V_L"] = recycled_features["V_L"]
        for key in (
            "normalized_energy_B",
            "normalized_forces_L",
            "energy_token_contributions_I",
            "property_domain_id_B",
            "property_atom_count_B",
            "latent_equivariance_prediction_B",
            "latent_equivariance_target_B",
        ):
            if key in recycled_features:
                outputs[key] = recycled_features[key]
        if return_property_embeddings:
            for key in ("Q_sys", "Q_I", "Q_L"):
                if key in recycled_features:
                    outputs[key] = recycled_features[key]
        if "model_diagnostics" in recycled_features:
            outputs["model_diagnostics"] = recycled_features["model_diagnostics"]
        return outputs

    def _prepare_recycle_inputs(
        self,
        *,
        X_L: torch.Tensor,
        t_L_raw: torch.Tensor,
        t_I_raw: torch.Tensor,
        f: dict,
        S_I: torch.Tensor,
        Z_II: torch.Tensor,
        C_L: torch.Tensor,
        P_LL: torch.Tensor,
        initializer_outputs: dict | None = None,
    ) -> dict:
        """
        Prepare inputs for the recycling process, including scaling positions and pooling features.

        Args:
            X_L: Noisy input positions [B, L, 3]
            t_L_raw: Raw noise level tensor for atoms [B, L]
            t_I_raw: Raw noise level tensor for tokens [B, n_tokens]
            f: Feature dictionary containing necessary features and masks for attention and pooling
            S_I: Processed single token features [n_tokens, c_s]
            Z_II: Processed pairwise token features [n_tokens, n_tokens, n_token_head]
            C_L: Processed single atom features [L, c_atom]
            P_LL: Processed pairwise atom features [L, L, n_atom_head]
            initializer_outputs: Optional outputs from the initializer to be used in chunked mode

        Returns:
            recycle_inputs: Dictionary containing prepared inputs for the recycling process, including:
                "X_noisy_L": Noisy input positions [B, L, 3]
                "R_L_uniform": Scaled positions for input to the model [B, L, 3]
                "t_L": Noise level tensor for atoms [B, L]
                "f": Feature dictionary
                "Q_I": Processed (geometry-aware) single token features [B, n_tokens, c_s=c_token]
                "Z_II": Processed pairwise token features [B, n_tokens, n_tokens, n_token_head]
                "S_I": Processed (geometry and time-aware) single token features for conditioning [B, n_tokens, c_s]
                "Q_L": Processed (geometry-aware) single atom features [B, L, c_atom]
                "P_LL": Processed pairwise atom features [B, L, L, n_atom_head]
                "C_L": Processed (geometry and time-aware) single atom features for conditioning [B, L, c_atom]
                "G_I": Attention indices for tokens [B, n_tokens, n_token_attn_keys]
                "G_L": Attention indices for atoms [B, L, n_atom_attn_keys]
                "initializer_outputs": Optional outputs from the initializer
        """
        # Prepare time inputs for atoms and tokens
        t_L = self._condition_time_for_fixed_atoms(
            t_L_raw, f["is_motif_atom_with_fixed_coord"]
        )
        t_I = self._condition_time_for_fixed_atoms(
            t_I_raw, f["is_motif_token_with_fully_fixed_coord"]
        )

        # ... Geometry embedding: fuse scaled positions with token and atom features
        R_L_uniform = self.scale_positions_in(X_L, t_L_raw)
        R_L = self.scale_positions_in(X_L, t_L)

        R_I_embed = self.process_r_token(
            # TODO: Here, consider using the Fourier projection in Algorithm 17 of the Emyx paper.
            R_L,
            tok_idx=f["atom_to_token_map"],
            num_tok=S_I.shape[0],
        )
        R_L_embed = self.process_r_atom(R_L)

        S_I = self.process_s(S_I + R_I_embed)
        C_L = self.process_c(C_L + R_L_embed)

        Q_I = S_I.clone()  # (B, I, c_s), primary single token features
        Q_L = C_L.clone()  # (B, L, c_atom), primary single atom features

        # ... Time embedding: fuse time features with (processed) token and atom features
        S_I_list = [S_I, self.process_time_(t_I, i=1)]
        C_L_list = [C_L, self.process_time_(t_L, i=0)]
        S_I = sum(S_I_list) / len(S_I_list)
        C_L = sum(C_L_list) / len(C_L_list)

        # ... Graph construction: create attention indices for local attention
        G_I, f = prepare_token_attention_inputs(
            f=f,
            X_I=X_L[..., f["is_ca"], :],
            use_local_attention=self.use_local_token_attention,
            n_attn_keys=self.n_token_attn_keys,
            n_attn_seq_neighbours=self.n_token_attn_seq_neighbours,
        )
        G_L, G_L_mask = create_attention_indices(
            X_L=X_L,
            f=f,
            n_attn_keys=self.n_atom_attn_keys,
            n_attn_seq_neighbours=self.n_atom_attn_seq_neighbours,
        )

        # ... Reuse validity mask returned by graph construction
        f["atom_pair_mask"] = G_L_mask

        return {
            "X_noisy_L": X_L,
            "R_L_uniform": R_L_uniform,
            "t_L": t_L,
            "f": f,
            "Q_I": Q_I,
            "Z_II": Z_II,
            "S_I": S_I,
            "Q_L": Q_L,
            "P_LL": P_LL,
            "C_L": C_L,
            "G_I": G_I,
            "G_L": G_L,
            "initializer_outputs": initializer_outputs,
        }

    @staticmethod
    def _public_outputs(
        recycled_features: dict,
        *,
        return_property_embeddings: bool = False,
    ) -> dict:
        """
        Extract public outputs from the recycled features dictionary.

        Args:
            recycled_features: Dictionary containing outputs from the recycling process

        Returns:
            public_outputs: Dictionary containing only the public outputs:
                "X_L": Denoised output positions [B, L, 3]
                "V_L" (optional): Denoised output velocities [B, L, 3] (only if f_pred is "flow_matching")
                "sequence_indices_I": Predicted sequence indices [B, n_tokens]
                "sequence_logits_I": Predicted sequence logits [B, n_tokens, c_s=c_token]
                "atom_type_indices_I": Predicted atom type indices [B, n_tokens]
                "atom_type_logits_I": Predicted atom type logits [B, n_tokens, c_s=c_token]
                "fractional_coord_L": Predicted fractional coordinates [B, L, 3]
                "lattice_lengths_B": Predicted lattice lengths [B, n_examples, 3]
                "lattice_angles_B": Predicted lattice angles [B, n_examples, 3]
            return_property_embeddings: Whether to include property embeddings in the outputs
        """
        outputs = {
            "X_L": recycled_features["X_L"],
            "sequence_indices_I": recycled_features["sequence_indices_I"],
            "sequence_logits_I": recycled_features["sequence_logits_I"],
            "atom_type_indices_I": recycled_features["atom_type_indices_I"],
            "atom_type_logits_I": recycled_features["atom_type_logits_I"],
            "fractional_coord_L": recycled_features["fractional_coord_L"],
            "lattice_lengths_B": recycled_features["lattice_lengths_B"],
            "lattice_angles_B": recycled_features["lattice_angles_B"],
        }
        if "V_L" in recycled_features:
            outputs["V_L"] = recycled_features["V_L"]
        for key in (
            "normalized_energy_B",
            "normalized_forces_L",
            "energy_token_contributions_I",
            "property_domain_id_B",
            "property_atom_count_B",
            "latent_equivariance_prediction_B",
            "latent_equivariance_target_B",
        ):
            if key in recycled_features:
                outputs[key] = recycled_features[key]
        if return_property_embeddings:
            for key in ("Q_sys", "Q_I", "Q_L"):
                if key in recycled_features:
                    outputs[key] = recycled_features[key]
        if "model_diagnostics" in recycled_features:
            outputs["model_diagnostics"] = recycled_features["model_diagnostics"]
        return outputs

    def forward_with_self_embedding_guidance(
        self,
        *,
        X_noisy_L: torch.Tensor,
        X_self_embedding_L: torch.Tensor,
        t_L_raw: torch.Tensor,
        t_I_raw: torch.Tensor,
        f: dict,
        f_self_embedding: dict,
        S_I: torch.Tensor,
        Z_II: torch.Tensor,
        C_L: torch.Tensor,
        P_LL: torch.Tensor,
        self_embedding_initializer_outputs: dict,
        initializer_outputs: dict | None = None,
        return_property_embeddings: bool = False,
    ) -> dict:
        """
        Forward pass with self-embedding guidance. This method performs a clean forward pass to compute the self-embedding condition, and then uses that condition in the noisy forward pass.

        Args:
            X_noisy_L: Noisy input positions [B, L, 3]
            X_self_embedding_L: Clean input positions for self-embedding guidance [B, L, 3]
            t_L_raw: Raw noise level tensor for atoms [B, L]
            t_I_raw: Raw noise level tensor for tokens [B, n_tokens]
            f: Feature dictionary containing necessary features and masks for attention and pooling
            f_self_embedding: Feature dictionary for self-embedding guidance (can be the same as f)
            S_I: Processed single token features [n_tokens, c_s]
            Z_II: Processed pairwise token features [n_tokens, n_tokens, n_token_head]
            C_L: Processed single atom features [L, c_atom]
            P_LL: Processed pairwise atom features [L, L, n_atom_head]
            self_embedding_initializer_outputs: Outputs from the initializer for self-embedding guidance
            initializer_outputs: Optional outputs from the initializer to be used in chunked mode
            return_property_embeddings: Whether to return property embeddings for property prediction

        Returns:
            outputs: Dictionary containing:
                "X_L": Denoised output positions [B, L, 3]
                "V_L" (optional): Denoised output velocities [B, L, 3] (only if f_pred is "flow_matching")
                "sequence_indices_I": Predicted sequence indices [B, n_tokens]
                "sequence_logits_I": Predicted sequence logits [B, n_tokens, c_s=c_token]
                "atom_type_indices_I": Predicted atom type indices [B, n_tokens]
                "atom_type_logits_I": Predicted atom type logits [B, n_tokens, c_s=c_token]
                "fractional_coord_L": Predicted fractional coordinates [B, L, 3]
                "lattice_lengths_B": Predicted lattice lengths [B, n_examples, 3]
                "lattice_angles_B": Predicted lattice angles [B, n_examples, 3]
        """
        if self.self_embedding_conditioner is None:
            raise ValueError(
                "Self-embedding guidance was requested, but the diffusion module "
                "was not initialized with self_embedding_guidance_enabled=True."
            )
        if X_self_embedding_L is None:
            raise ValueError("Self-embedding guidance requires X_self_embedding_L.")

        clean_outputs = {
            "S_I": S_I,
            "Z_II": Z_II,
            "C_L": C_L,
            "P_LL": P_LL,
        } | self_embedding_initializer_outputs

        clean_time_L = (
            torch.ones_like(t_L_raw)
            if self.f_pred == FLOW_MATCHING
            else torch.zeros_like(t_L_raw)
        )
        clean_time_I = (
            torch.ones_like(t_I_raw)
            if self.f_pred == FLOW_MATCHING
            else torch.zeros_like(t_I_raw)
        )
        clean_pass_inputs = self._prepare_recycle_inputs(
            X_L=X_self_embedding_L,
            t_L_raw=clean_time_L,
            t_I_raw=clean_time_I,
            f=f_self_embedding,
            S_I=clean_outputs["S_I"],
            Z_II=clean_outputs["Z_II"],
            C_L=clean_outputs["C_L"],
            P_LL=clean_outputs["P_LL"],
            initializer_outputs=clean_outputs,
        )
        clean_features = self.process_(
            X_L_rec=None,
            collect_model_diagnostics=False,
            **clean_pass_inputs,
        )
        condition_B = self.self_embedding_conditioner.embed_random_level(
            token_embeddings=clean_features["self_embedding_token_features"],
            atom_embeddings=clean_features["self_embedding_atom_features"],
            batch_token_lvl=f_self_embedding["batch_token_lvl"],
            batch_atom_lvl=f_self_embedding["batch_atom_lvl"],
            batch_size=f_self_embedding["batch_size"],
        )

        noisy_pass_inputs = self._prepare_recycle_inputs(
            X_L=X_noisy_L,
            t_L_raw=t_L_raw,
            t_I_raw=t_I_raw,
            f=f,
            S_I=S_I,
            Z_II=Z_II,
            C_L=C_L,
            P_LL=P_LL,
            initializer_outputs=initializer_outputs,
        )
        recycled_features = self.process_(
            X_L_rec=None,
            self_embedding_condition_B=condition_B,
            collect_model_diagnostics=(
                self.training and self.activation_scale_diagnostics
            ),
            **noisy_pass_inputs,
        )
        return self._public_outputs(
            recycled_features,
            return_property_embeddings=return_property_embeddings,
        )

    def forward_with_recycle(
        self,
        n_recycle: int | None,
        **kwargs,
    ) -> dict:
        """
        Perform forward pass with optional recycling. During training, n_recycle must be provided as an argument. During inference, n_recycle can be set as an attribute of the model and will default to that if not provided as an argument.

        Args:
            n_recycle: Number of recycling iterations to perform (if None, uses self.n_recycle)
            **kwargs: Keyword arguments containing all necessary inputs for the forward pass

        Returns:
            recycled_features: Dictionary containing the outputs from the final recycling iteration, including:
                "X_L": Denoised output positions [B, L, 3]
                "V_L" (optional): Denoised output velocities [B, L, 3] (only if f_pred is "flow_matching")
                "sequence_indices_I": Predicted sequence indices [B, n_tokens]
                "sequence_logits_I": Predicted sequence logits [B, n_tokens, c_s=c_token]
                "atom_type_indices_I": Predicted atom type indices [B, n_tokens]
                "atom_type_logits_I": Predicted atom type logits [B, n_tokens, c_s=c_token]
                "fractional_coord_L": Predicted fractional coordinates [B, L, 3]
                "lattice_lengths_B": Predicted lattice lengths [B, n_examples, 3]
                "lattice_angles_B": Predicted lattice angles [B, n_examples, 3]
        """
        if not self.training:
            n_recycle = n_recycle if n_recycle is not None else self.n_recycle
        else:
            assert n_recycle is not None

        recycled_features = {}
        for i in range(n_recycle):
            with ExitStack() as stack:
                last = not (i < n_recycle - 1)
                if not last:
                    stack.enter_context(torch.no_grad())

                # Clear the autocast cache if gradients are enabled (workaround for autocast bug)
                # See: https://github.com/pytorch/pytorch/issues/65766
                if torch.is_grad_enabled():
                    torch.clear_autocast_cache()
                # Run forward
                recycled_features = self.process_(
                    X_L_rec=recycled_features.get("X_L"),
                    collect_model_diagnostics=(
                        last and self.training and self.activation_scale_diagnostics
                    ),
                    **kwargs,
                )

        return recycled_features

    def process_(
        self,
        X_L_rec: torch.Tensor | None,
        *,
        X_noisy_L: torch.Tensor,
        t_L: torch.Tensor,
        f: dict,
        S_I: torch.Tensor,
        Z_II: torch.Tensor,
        Q_I: torch.Tensor,
        C_L: torch.Tensor,
        P_LL: torch.Tensor,
        Q_L: torch.Tensor,
        G_I: torch.Tensor | None,
        G_L: torch.Tensor | None,
        self_embedding_condition_B: torch.Tensor | None = None,
        collect_model_diagnostics: bool = False,
        **_,
    ) -> dict:
        """
        Process one iteration of the diffusion forward pass, including encoder, trunk, and decoder steps.

        Args:
            X_L_rec: Denoised positions from the previous iteration [B, L, 3] (or None for the first iteration). If available, only used to compute distogram features.
            X_noisy_L: Original noisy input positions [B, L, 3]
            t_L: Noise level tensor for atoms [B, L]
            f: Feature dictionary containing necessary features and masks for attention and pooling
            Q_I: Processed (geometry-aware) single token features [B, n_tokens, c_s=c_token]
            Z_II: Processed pairwise token features [n_tokens, n_tokens, n_token_head]
            S_I: Processed (geometry and time-aware) single token features for conditioning [B, n_tokens, c_s]
            Q_L: Processed (geometry-aware) single atom features [B, L, c_atom]
            P_LL: Processed pairwise atom features [L, L, n_atom_head]
            C_L: Processed (geometry and time-aware) single atom features for conditioning [B, L, c_atom]
            G_I: Optional attention indices for tokens [B, n_tokens, n_token_attn_keys]
            G_L: Optional attention indices for atoms [B, L, n_atom_attn_keys]
            self_embedding_condition_B: Optional self-embedding condition tensor [B, c_s=c_token] to be used in the forward pass

        Returns:
            outputs: Dictionary containing:
                "X_L": Denoised output positions [B, L, 3]
                "V_L" (optional): Denoised output velocities [B, L, 3] (only if f_pred is "flow_matching")
                "sequence_indices_I": Predicted sequence indices [B, n_tokens]
                "sequence_logits_I": Predicted sequence logits [B, n_tokens, c_s=c_token]
                "atom_type_indices_I": Predicted atom type indices [B, n_tokens]
                "atom_type_logits_I": Predicted atom type logits [B, n_tokens, c_s=c_token]
                "fractional_coord_L": Predicted fractional coordinates [B, L, 3]
                "lattice_lengths_B": Predicted lattice lengths [B, n_examples, 3]
                "lattice_angles_B": Predicted lattice angles [B, n_examples, 3]
                "self_embedding_token_features": Token-level features for self-embedding guidance [B, n_tokens, c_s=c_token]
                "self_embedding_atom_features": Atom-level features for self-embedding guidance [B, L, c_atom]
        """
        D = f["diffusion_batch_size"]
        B = f["batch_size"]
        I = S_I.shape[1]  # I = number of tokens; L = number of atoms

        # ... Self-embedding guidance
        if self.self_embedding_conditioner is not None:
            Q_I, S_I = self.self_embedding_conditioner.condition_token_inputs(
                A_I=Q_I,  # NOTE: A_I=Q_I is geometry-aware
                S_I=S_I,  # NOTE: S_I is both geometry and time-aware for sake of conditioning
                condition_B=self_embedding_condition_B,
                batch_token_lvl=f["batch_token_lvl"],
            )

        # ... Recycling guidance
        is_rec_iter = X_L_rec is not None
        if is_rec_iter:
            D_II_rec = self.bucketize_fn(X_L_rec[..., f["is_ca"], :])
        else:
            D_II_rec = torch.zeros(D, I, I, self.n_distogram_bins, device=Z_II.device)

        Z_II = Z_II + self.process_distogram_rec(D_II_rec) * float(is_rec_iter)
        P_LL = P_LL.unsqueeze(0).expand(D, -1, -1, -1)  # [D, L, L, n_atom_head]

        # ... Diffusion transformer
        backbone_outputs = self.backbone(
            Q_I=Q_I,
            Z_II=Z_II,
            S_I=S_I,
            Q_L=Q_L,
            P_LL=P_LL,
            C_L=C_L,
            G_I=G_I,
            G_L=G_L,
            f=f,
        )

        # ... Processing outputs for modalities
        Q_I = backbone_outputs["Q_I"]
        Q_L = backbone_outputs["Q_L"]

        self_embedding_token_features = Q_I
        self_embedding_atom_features = Q_L

        R_update_L = self.euclidean_coord_head(Q_L, C_L)
        X_out_L = self.scale_positions_out(R_update_L, X_noisy_L, t_L)

        velocity_L = None
        if self.f_pred == FLOW_MATCHING:
            velocity_L = R_update_L
            velocity_L = (
                velocity_L
                * (~f["is_motif_atom_with_fixed_coord"]).to(velocity_L.dtype)[
                    None, :, None
                ]
            )

        A_sequence = self._process_prediction_head_features(
            Q_I, S_I, self.sequence_head_processor
        )
        A_atom_type = self._process_prediction_head_features(
            Q_I, S_I, self.atom_type_head_processor
        )
        sequence_logits_I, sequence_indices_I = self.sequence_head(A_I=A_sequence)
        atom_type_logits_I, atom_type_indices_I = self.atom_type_head(A_I=A_atom_type)
        fractional_coord_L = self.fractional_coord_head(Q_L, C_L)

        Q_sys_init = self.lattice_init_token.type(Q_I.dtype).repeat(D, B, 1)
        Q_sys = self.downcast_lattice(
            Q_L=Q_I,
            A_I=Q_sys_init,
            S_I=None,
            tok_idx=f["batch_token_lvl"],
            n_atoms_per_tok_max=max(f["batch_num_tokens"]),
        )
        S_sys = self._mean_pool_features(S_I, f["batch_token_lvl"], B)
        A_lattice = self._process_prediction_head_features(
            Q_sys, S_sys, self.lattice_head_processor
        )
        lattice_lengths_B, lattice_angles_B = self.lattice_head(A_I=A_lattice)

        outputs = {
            "X_L": X_out_L,
            "sequence_logits_I": sequence_logits_I,
            "sequence_indices_I": sequence_indices_I,
            "atom_type_logits_I": atom_type_logits_I,
            "atom_type_indices_I": atom_type_indices_I,
            "fractional_coord_L": fractional_coord_L,
            "lattice_lengths_B": lattice_lengths_B,
            "lattice_angles_B": lattice_angles_B,
            "self_embedding_token_features": self_embedding_token_features,
            "self_embedding_atom_features": self_embedding_atom_features,
        }
        if self.atomistic_property_prediction_enabled:
            domain_id_B = self._property_domain_ids(
                f,
                batch_size=B,
                device=Q_I.device,
            )
            atom_count_B = self._property_atom_counts(
                f,
                batch_size=B,
                device=Q_I.device,
                dtype=Q_I.dtype,
            )
            A_energy = self._process_prediction_head_features(
                Q_I, S_I, self.energy_head_processor
            )
            normalized_energy_B, energy_token_contributions_I = self.energy_head(
                A_energy,
                batch_token_lvl=f["batch_token_lvl"],
                domain_id_B=domain_id_B,
                atom_count_B=atom_count_B,
            )
            outputs.update(
                {
                    "normalized_energy_B": normalized_energy_B,
                    "normalized_forces_L": self.force_head(Q_L, C_L),
                    "energy_token_contributions_I": (energy_token_contributions_I),
                    "property_domain_id_B": domain_id_B,
                    "property_atom_count_B": atom_count_B,
                    "Q_sys": Q_sys,
                    "Q_I": Q_I,
                    "Q_L": Q_L,
                }
            )
            if self.latent_equivariance_enabled and self.training:
                if "augmentation_rotation_D" not in f:
                    raise ValueError(
                        "Enabled latent equivariance requires recorded augmentation "
                        "rotations; set datasets.record_rotation_pairs=True."
                    )
                # Use the atom representation consumed by the force head so the
                # auxiliary objective regularizes its upstream backbone features.
                A_latent_equivariance = self._process_prediction_head_features(
                    Q_L,
                    C_L,
                    self.latent_equivariance_head_processor,
                )
                pooled_Q_B = self._mean_pool_features(
                    A_latent_equivariance, f["batch_atom_lvl"], B
                )
                latent_prediction_B, latent_target_B = self.latent_equivariance_head(
                    pooled_Q_B,
                    f["augmentation_rotation_D"],
                )
                outputs.update(
                    {
                        "latent_equivariance_prediction_B": latent_prediction_B,
                        "latent_equivariance_target_B": latent_target_B,
                    }
                )

        if collect_model_diagnostics:
            outputs["model_diagnostics"] = summarize_zatom_activation_scales(
                token_trunk=Q_I,
                atom_trunk=Q_L,
                token_condition=S_I,
                sequence_head_input=A_sequence,
                atom_type_head_input=A_atom_type,
                lattice_head_input=A_lattice,
                sequence_logits=sequence_logits_I,
                atom_type_logits=atom_type_logits_I,
                lattice_lengths=lattice_lengths_B,
                lattice_angles=lattice_angles_B,
                energy_head_input=(
                    A_energy if self.atomistic_property_prediction_enabled else None
                ),
                normalized_energy=outputs.get("normalized_energy_B"),
                energy_token_contributions=outputs.get("energy_token_contributions_I"),
            )
        if velocity_L is not None:
            outputs["V_L"] = velocity_L
        return outputs
