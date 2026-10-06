import functools
import os
from contextlib import ExitStack

import torch
import torch.nn as nn
from foundry.model.layers.blocks import FourierEmbedding
from nucleus.model.flow_matching import FLOW_MATCHING, endpoint_from_velocity
from nucleus.model.layers.attention import (
    indices_to_mask_with_padding,
    infer_attention_mask_from_features,
)
from nucleus.model.layers.block_utils import create_attention_indices
from nucleus.model.layers.blocks import (
    CompactStreamingDecoder,
    Downcast,
    LinearAtomTypesHead,
    LinearEmbedWithPool,
    LinearFractionalCoordinatesHead,
    LinearLatticeHead,
    LocalAtomTransformer,
    LocalTokenTransformer,
)
from nucleus.model.layers.encoders import DiffusionTokenEncoder
from nucleus.model.self_embedding_guidance import SelfEmbeddingConditioner
from nucleus.utils.attention import maybe_create_flex_block_mask
from nucleus.utils.distogram import bucketize_scaled_distogram
from nucleus.utils.torch import expand_t_to_level
from rfd3.model.layers.layer_utils import RMSNorm, linearNoBias
from rfd3.model.RFD3_diffusion_module import LinearSequenceHead


class RFD3DiffusionModule(torch.nn.Module):
    """
    UNet-like architecture for diffusion with recycling,
    supporting molecules, materials, and proteins and
    dataloader batch_size > 1.

    Args:
        c_atom: Atom-level feature dimension
        c_atompair: Atom-pair feature dimension
        c_token: Token-level feature dimension
        c_s: Sequence-level feature dimension
        c_z: Token-pair feature dimension
        c_t_embed: Time embedding dimension
        sigma_data: Sigma for scaling positions in and out of the model
        f_pred: Coordinate objective ("edm", "flow_matching", "noise_pred", or
            "unconditioned")
        n_attn_seq_neighbours: Number of sequential neighbors to attend to in local attention
        n_attn_keys: Number of keys to attend to in local attention
        n_recycle: Number of recycling iterations during inference
        atom_attention_encoder: Config for the LocalAtomTransformer encoder
        diffusion_token_encoder: Config for the DiffusionTokenEncoder
        diffusion_transformer: Config for the LocalTokenTransformer in the diffusion block
        atom_attention_decoder: Config for the CompactStreamingDecoder
        downcast: Config for the Downcast layers
        use_local_token_attention: Whether to use local attention in the diffusion transformer
        attn_backend: Attention backend to use ("manual", "megafold", or "flex")
        self_embedding_guidance_enabled: Whether to enable SCD conditioning support
        self_embedding_guidance_probability: Training probability of using SCD
        self_embedding_atom_condition_probability: Probability of pooling clean atom
            features instead of clean token features for the SCD condition
        self_embedding_condition_dropout: Probability of dropping the SCD condition
        self_embedding_condition_clip_value: Clamp value for the pooled SCD condition
    """

    def __init__(
        self,
        *,
        c_atom,
        c_atompair,
        c_token,
        c_s,
        c_z,
        c_t_embed,
        sigma_data,
        f_pred,
        n_attn_seq_neighbours,
        n_attn_keys,
        n_recycle,
        atom_attention_encoder,
        diffusion_token_encoder,
        diffusion_transformer,
        atom_attention_decoder,
        downcast,
        use_local_token_attention=True,
        attn_backend="manual",
        self_embedding_guidance_enabled=False,
        self_embedding_guidance_probability=0.25,
        self_embedding_atom_condition_probability=0.5,
        self_embedding_condition_dropout=0.2,
        self_embedding_condition_clip_value=100.0,
        **_,
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s = c_s
        self.c_z = c_z
        self.f_pred = f_pred
        self.n_attn_seq_neighbours = n_attn_seq_neighbours
        self.n_attn_keys = n_attn_keys
        self.use_local_token_attention = use_local_token_attention
        self.attn_backend = attn_backend
        self.self_embedding_guidance_enabled = self_embedding_guidance_enabled
        self.self_embedding_guidance_probability = self_embedding_guidance_probability

        # Auxiliary
        self.process_r = linearNoBias(3, c_atom)
        self.to_r_update = nn.Sequential(RMSNorm((c_atom,)), linearNoBias(c_atom, 3))
        if self.f_pred == FLOW_MATCHING:
            nn.init.zeros_(self.to_r_update[-1].weight)
        self.sequence_head = LinearSequenceHead(c_token=c_token)

        self.n_recycle = n_recycle
        self.n_bins = 65
        self.bucketize_fn = functools.partial(
            bucketize_scaled_distogram,
            min_dist=1,
            max_dist=30,
            sigma_data=1,  # NOTE: Must be 1 for distogram processing, since the distogram is computed from scaled positions
            n_bins=self.n_bins,
        )

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
        self.downcast_c = Downcast(c_atom=c_atom, c_token=c_s, c_s=None, **downcast)
        self.downcast_q = Downcast(c_atom=c_atom, c_token=c_token, c_s=c_s, **downcast)
        self.process_a = LinearEmbedWithPool(c_token)
        self.process_c = nn.Sequential(RMSNorm(c_atom), linearNoBias(c_atom, c_atom))

        # UNet-like architecture for processing across tokens and atoms
        self.encoder = LocalAtomTransformer(
            c_atom=c_atom, c_s=c_atom, c_atompair=c_atompair, **atom_attention_encoder
        )

        self.diffusion_token_encoder = DiffusionTokenEncoder(
            c_s=c_s,
            c_token=c_token,
            c_z=c_z,
            c_atompair=c_atompair,
            **diffusion_token_encoder,
        )

        self.diffusion_transformer = LocalTokenTransformer(
            c_token=c_token,
            c_tokenpair=c_z,
            c_s=c_s,
            **diffusion_transformer,
        )

        self.decoder = CompactStreamingDecoder(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            c_s=c_s,
            c_tokenpair=c_z,
            **atom_attention_decoder,
        )
        self.atom_type_head = LinearAtomTypesHead(c_token=c_token)
        self.fractional_coord_head = LinearFractionalCoordinatesHead(c_atom=c_atom)
        self.lattice_head = LinearLatticeHead(c_token=c_token)
        self.lattice_init_token = torch.nn.Parameter(torch.zeros(1, 1, c_token))
        self.downcast_lattice = Downcast(
            c_atom=c_token, c_token=c_token, c_s=None, **downcast
        )
        self.self_embedding_conditioner = (
            SelfEmbeddingConditioner(
                c_token=c_token,
                c_s=c_s,
                c_atom=c_atom,
                atom_probability=self_embedding_atom_condition_probability,
                condition_dropout=self_embedding_condition_dropout,
                clip_value=self_embedding_condition_clip_value,
            )
            if self.self_embedding_guidance_enabled
            else None
        )

        self.use_flex_attention = self.attn_backend == "flex"

    def scale_positions_in(self, X_noisy_L, t) -> torch.Tensor:
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
            R_noisy_L = X_noisy_L / torch.sqrt(t**2 + self.sigma_data**2)
        elif self.f_pred == FLOW_MATCHING:
            R_noisy_L = X_noisy_L / self.sigma_data
        elif self.f_pred == "unconditioned":
            R_noisy_L = torch.zeros_like(X_noisy_L)
        elif self.f_pred == "noise_pred":
            R_noisy_L = X_noisy_L
        else:
            raise Exception(f"{self.f_pred=} unrecognized")
        return R_noisy_L

    def scale_positions_out(self, R_update_L, X_noisy_L, t) -> torch.Tensor:
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
            X_out_L = (self.sigma_data**2 / (self.sigma_data**2 + t**2)) * X_noisy_L + (
                self.sigma_data * t / (self.sigma_data**2 + t**2) ** 0.5
            ) * R_update_L
        elif self.f_pred == FLOW_MATCHING:
            X_out_L = endpoint_from_velocity(X_noisy_L, self.sigma_data * R_update_L, t)
        elif self.f_pred == "unconditioned":
            X_out_L = R_update_L
        elif self.f_pred == "noise_pred":
            X_out_L = X_noisy_L + R_update_L
        else:
            raise Exception(f"{self.f_pred=} unrecognized")
        return X_out_L

    def process_time_(self, t_L, i) -> torch.Tensor:
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
        # Mask out zero-time features;
        C_L = C_L * (t_L > 0).float()[..., None]  # [B, L, C_atom]
        return C_L

    def _condition_time_for_fixed_atoms(self, time, fixed_mask):
        """Use the clean endpoint for fixed atoms under either time convention."""
        fixed_mask = fixed_mask.to(device=time.device, dtype=torch.bool).unsqueeze(0)
        if getattr(self, "f_pred", "edm") == FLOW_MATCHING:
            return torch.where(fixed_mask, torch.ones_like(time), time)
        return time * (~fixed_mask).to(time.dtype)

    def forward(
        self,
        X_noisy_L,
        t,
        f,
        # Features from initialization
        Q_L_init,
        C_L,
        P_LL,
        S_I,
        Z_II,
        n_recycle=None,
        # Chunked memory optimization parameters
        chunked_pairwise_embedder=None,
        initializer_outputs=None,
        token_flex_block_mask=None,
        token_flex_block_mask_ready=False,
        use_self_embedding_guidance=False,
        X_self_embedding_L=None,
        f_self_embedding=None,
        self_embedding_initializer_outputs=None,
        return_self_embedding_features=False,
        **kwargs,
    ) -> dict:
        """
        Diffusion forward pass with recycling.
        Computes denoised positions given encoded features and noisy coordinates.

        Args:
            X_noisy_L: Noisy input positions [B, L, 3]
            t: Noise level tensor [B] or [B, L]
            f: Feature dictionary containing necessary features and masks for attention and pooling
            Q_L_init: Initial atom-level features from the encoder [L, c_atom]
            C_L: Atom-level features for time embedding [L, c_atom]
            P_LL: Atom-pair features [L, L, c_atompair]
            S_I: Sequence-level features [n_tokens, c_s]
            Z_II: Token-pair features [n_tokens, n_tokens, c_z]
            n_recycle: Number of recycling iterations to perform (if None, uses self.n_recycle)
            chunked_pairwise_embedder: Optional embedder for computing pairwise features in a memory-efficient chunked manner
            initializer_outputs: Optional outputs from the initializer to be used in chunked mode
            token_flex_block_mask: Optional precomputed flexible block mask for token-level attention
            token_flex_block_mask_ready: Whether the token_flex_block_mask is ready to be used
            use_self_embedding_guidance: Whether to use self-embedding guidance during this forward pass
            X_self_embedding_L: Clean atom-level positions for self-embedding guidance [B, L, 3]
            f_self_embedding: Feature dictionary for self-embedding guidance (if different from f)
            self_embedding_initializer_outputs: Optional outputs from the initializer for self-embedding guidance
            return_self_embedding_features: Whether to return self-embedding features in the output

        Returns:
            outputs: Dictionary containing:
                "X_L": Denoised output positions [B, L, 3]
                "V_L" (optional): Denoised velocity vectors [B, L, 3] if present in recycled_features
                "sequence_indices_I": Predicted sequence indices [B, n_tokens]
                "sequence_logits_I": Predicted sequence logits [B, n_tokens, c_token]
        """
        # ... Expand t tensor to atom level
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
                Q_L_init=Q_L_init,
                C_L_init=C_L,
                P_LL=P_LL,
                S_I_init=S_I,
                Z_II_init=Z_II,
                chunked_pairwise_embedder=chunked_pairwise_embedder,
                self_embedding_initializer_outputs=(
                    self_embedding_initializer_outputs or {}
                ),
                initializer_outputs=initializer_outputs,
                token_flex_block_mask=token_flex_block_mask,
                token_flex_block_mask_ready=token_flex_block_mask_ready,
                return_self_embedding_features=return_self_embedding_features,
            )

        recycle_inputs = self._prepare_recycle_inputs(
            X_L=X_noisy_L,
            t_L_raw=kwargs["t_L"],
            t_I_raw=kwargs["t_I"],
            f=f,
            Q_L_init=Q_L_init,
            C_L_init=C_L,
            P_LL=P_LL,
            S_I_init=S_I,
            Z_II=Z_II,
            chunked_pairwise_embedder=chunked_pairwise_embedder,
            initializer_outputs=initializer_outputs,
        )
        flex_block_mask = recycle_inputs.pop("flex_block_mask")
        flex_block_mask_ready = recycle_inputs.pop("flex_block_mask_ready")

        # ... Run forward with recycling
        recycled_features = self.forward_with_recycle(
            flex_block_mask=flex_block_mask,
            flex_block_mask_ready=flex_block_mask_ready,
            token_flex_block_mask=token_flex_block_mask,
            token_flex_block_mask_ready=token_flex_block_mask_ready,
            n_recycle=n_recycle,
            **recycle_inputs,
        )

        # ... Collect outputs
        return self._public_outputs(
            recycled_features,
            return_self_embedding_features=return_self_embedding_features,
        )

    def _prepare_recycle_inputs(
        self,
        *,
        X_L,
        t_L_raw,
        t_I_raw,
        f,
        Q_L_init,
        C_L_init,
        P_LL,
        S_I_init,
        Z_II,
        chunked_pairwise_embedder=None,
        initializer_outputs=None,
    ) -> dict:
        """
        Prepare inputs for the recycling process, including attention indices and scaled positions.

        Args:
            X_L: Noisy input positions [B, L, 3]
            t_L_raw: Raw noise level tensor for atoms [B, L]
            t_I_raw: Raw noise level tensor for tokens [B, n_tokens]
            f: Feature dictionary containing necessary features and masks for attention and pooling
            Q_L_init: Initial atom-level features from the encoder [L, c_atom]
            C_L_init: Atom-level features for time embedding [L, c_atom]
            P_LL: Atom-pair features [L, L, c_atompair]
            S_I_init: Sequence-level features [n_tokens, c_s]
            Z_II: Token-pair features [n_tokens, n_tokens, c_z]
            chunked_pairwise_embedder: Optional embedder for computing pairwise features in a memory-efficient chunked manner
            initializer_outputs: Optional outputs from the initializer to be used in chunked mode

        Returns:
            recycle_inputs: Dictionary containing prepared inputs for recycling, including:
                "X_noisy_L": Noisy input positions [B, L, 3]
                "R_L_uniform": Scaled noisy positions for input to the model [B, L, 3]
                "t_L": Noise level tensor for atoms [B, L]
                "f": Feature dictionary with attention indices
                "Q_L": Atom-level features from the encoder [B, L, c_atom]
                "C_L": Atom-level features for time embedding [B, L, c_atom]
                "P_LL": Atom-pair features [B, L, L, c_atompair]
                "A_I": Sequence-level features after pooling [B, n_tokens, c_token]
                "S_I": Sequence-level features for time embedding [B, n_tokens, c_s]
                "Z_II": Token-pair features [B, n_tokens, n_tokens, c_z]
                "chunked_pairwise_embedder": Optional embedder for computing pairwise features in a memory-efficient chunked manner
                "initializer_outputs": Optional outputs from the initializer to be used in chunked mode
                "flex_block_mask": Optional precomputed flexible block mask for attention
                "flex_block_mask_ready": Whether the flex_block_mask is ready to be used
        """
        # ... Collect inputs
        tok_idx = f["atom_to_token_map"]
        f["attn_indices"], f["attn_index_mask"] = create_attention_indices(
            X_L=X_L,
            f=f,
            n_attn_keys=self.n_attn_keys,
            n_attn_seq_neighbours=self.n_attn_seq_neighbours,
        )

        t_L = self._condition_time_for_fixed_atoms(
            t_L_raw, f["is_motif_atom_with_fixed_coord"]
        )
        t_I = self._condition_time_for_fixed_atoms(
            t_I_raw, f["is_motif_token_with_fully_fixed_coord"]
        )

        # ... Create scaled positions
        R_L_uniform = self.scale_positions_in(X_L, t_L_raw)
        R_noisy_L = self.scale_positions_in(X_L, t_L)

        # ... Pool initial representation to sequence level
        A_I = self.process_a(R_noisy_L, tok_idx=tok_idx, num_tok=S_I_init.shape[0])
        S_I = self.downcast_c(C_L_init, S_I_init, tok_idx=tok_idx)

        # ... Add batch-wise features to inputs
        Q_L = Q_L_init.unsqueeze(0) + self.process_r(R_noisy_L)
        C_L = C_L_init.unsqueeze(0) + self.process_time_(t_L, i=0)
        S_I = S_I.unsqueeze(0) + self.process_time_(t_I, i=1)
        C_L = C_L + self.process_c(C_L)

        flex_block_mask = None
        flex_block_mask_ready = False
        if self.use_flex_attention:
            attn_mask = infer_attention_mask_from_features(
                f,
                f["attn_indices"],
                tok_idx=tok_idx,
                index_mask=f["attn_index_mask"],
            )
            flex_block_mask, flex_block_mask_ready = maybe_create_flex_block_mask(
                attn_mask=indices_to_mask_with_padding(f["attn_indices"], attn_mask),
                q_len=Q_L.shape[1],
                device=Q_L.device,
                use_flex_attention=self.use_flex_attention,
            )

        # ... Run Local-Atom Self Attention and Pool
        if chunked_pairwise_embedder is not None:
            # Chunked mode: pass chunked embedder and feature dict
            Q_L = self.encoder(
                Q_L,
                C_L,
                P_LL=None,
                indices=f["attn_indices"],
                index_mask=f["attn_index_mask"],
                f=f,  # Pass feature dict for chunked computation
                tok_idx=tok_idx,
                chunked_pairwise_embedder=chunked_pairwise_embedder,
                initializer_outputs=initializer_outputs,
                flex_block_mask=flex_block_mask,
                flex_block_mask_ready=flex_block_mask_ready,
            )
        else:
            # Standard mode: use full P_LL
            Q_L = self.encoder(
                Q_L,
                C_L,
                P_LL,
                indices=f["attn_indices"],
                index_mask=f["attn_index_mask"],
                f=f,
                tok_idx=tok_idx,
                flex_block_mask=flex_block_mask,
                flex_block_mask_ready=flex_block_mask_ready,
            )
        A_I = self.downcast_q(Q_L, A_I=A_I, S_I=S_I, tok_idx=tok_idx)

        # ... Run forward with recycling
        return {
            "X_noisy_L": X_L,
            "R_L_uniform": R_L_uniform,
            "t_L": t_L,
            "f": f,
            "Q_L": Q_L,
            "C_L": C_L,
            "P_LL": P_LL,
            "A_I": A_I,
            "S_I": S_I,
            "Z_II": Z_II,
            "chunked_pairwise_embedder": chunked_pairwise_embedder,
            "initializer_outputs": initializer_outputs,
            "flex_block_mask": flex_block_mask,
            "flex_block_mask_ready": flex_block_mask_ready,
        }

    @staticmethod
    def _public_outputs(
        recycled_features: dict,
        *,
        return_self_embedding_features: bool = False,
    ) -> dict:
        """
        Prepare the public outputs from the recycled features, including denoised positions, sequence indices, and logits.

        Args:
            recycled_features: Dictionary containing the outputs from the final recycling iteration
            return_self_embedding_features: Whether to include self-embedding features in the output

        Returns:
            outputs: Dictionary containing:
                "X_L": Denoised output positions [B, L, 3]
                "V_L" (optional): Denoised velocity vectors [B, L, 3] if present in recycled_features
                "sequence_indices_I": Predicted sequence indices [B, n_tokens]
                "sequence_logits_I": Predicted sequence logits [B, n_tokens, c_token]
                "atom_type_indices_I": Predicted atom type indices [B, n_tokens]
                "atom_type_logits_I": Predicted atom type logits [B, n_tokens, c_atom_type]
                "fractional_coord_L": Predicted fractional coordinates [B, L, 3]
                "lattice_lengths_B": Predicted lattice lengths [B, 3]
                "lattice_angles_B": Predicted lattice angles [B, 3]
                "self_embedding_token_features" (optional): Self-embedding token features [B, n_tokens, c_token] if return_self_embedding_features is True
        """
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
        if return_self_embedding_features:
            outputs["self_embedding_token_features"] = recycled_features[
                "self_embedding_token_features"
            ]
        return outputs

    def forward_with_self_embedding_guidance(
        self,
        *,
        X_noisy_L,
        X_self_embedding_L,
        t_L_raw,
        t_I_raw,
        f,
        f_self_embedding,
        Q_L_init,
        C_L_init,
        P_LL,
        S_I_init,
        Z_II_init,
        chunked_pairwise_embedder=None,
        self_embedding_initializer_outputs=None,
        initializer_outputs=None,
        token_flex_block_mask=None,
        token_flex_block_mask_ready=False,
        return_self_embedding_features=False,
    ) -> dict:
        """
        Forward pass with self-embedding guidance. This method computes the self-embedding condition from clean features and applies it to the noisy input during the diffusion process.

        Args:
            X_noisy_L: Noisy input positions [B, L, 3]
            X_self_embedding_L: Clean atom-level positions for self-embedding guidance [B, L, 3]
            t_L_raw: Raw noise level tensor for atoms [B, L]
            t_I_raw: Raw noise level tensor for tokens [B, n_tokens]
            f: Feature dictionary containing necessary features and masks for attention and pooling
            f_self_embedding: Feature dictionary for self-embedding guidance (if different from f)
            Q_L_init: Initial atom-level features from the encoder [L, c_atom]
            C_L_init: Atom-level features for time embedding [L, c_atom]
            P_LL: Atom-pair features [L, L, c_atompair]
            S_I_init: Sequence-level features [n_tokens, c_s]
            Z_II_init: Token-pair features [n_tokens, n_tokens, c_z]
            chunked_pairwise_embedder: Optional embedder for computing pairwise features in a memory-efficient chunked manner
            self_embedding_initializer_outputs: Optional outputs from the initializer for self-embedding guidance
            initializer_outputs: Optional outputs from the initializer to be used in chunked mode
            token_flex_block_mask: Optional precomputed flexible block mask for token-level attention
            token_flex_block_mask_ready: Whether the token_flex_block_mask is ready to be used
            return_self_embedding_features: Whether to return self-embedding features in the output

        Returns:
            outputs: Dictionary containing:
                "X_L": Denoised output positions [B, L, 3]
                "V_L" (optional): Denoised velocity vectors [B, L, 3] if present in recycled_features
                "sequence_indices_I": Predicted sequence indices [B, n_tokens]
                "sequence_logits_I": Predicted sequence logits [B, n_tokens, c_token]
                "atom_type_indices_I": Predicted atom type indices [B, n_tokens]
                "atom_type_logits_I": Predicted atom type logits [B, n_tokens, c_atom_type]
                "fractional_coord_L": Predicted fractional coordinates [B, L, 3]
                "lattice_lengths_B": Predicted lattice lengths [B, 3]
                "lattice_angles_B": Predicted lattice angles [B, 3]
                "self_embedding_token_features" (optional): Self-embedding token features [B, n_tokens, c_token] if return_self_embedding_features is True
        """
        if self.self_embedding_conditioner is None:
            raise ValueError(
                "Self-embedding guidance was requested, but the diffusion module "
                "was not initialized with self_embedding_guidance_enabled=True."
            )
        if X_self_embedding_L is None:
            raise ValueError("Self-embedding guidance requires X_self_embedding_L.")

        clean_outputs = {
            "Q_L_init": Q_L_init,
            "C_L": C_L_init,
            "P_LL": P_LL,
            "S_I": S_I_init,
            "Z_II": Z_II_init,
            "chunked_pairwise_embedder": chunked_pairwise_embedder,
            "token_flex_block_mask": token_flex_block_mask,
            "token_flex_block_mask_ready": token_flex_block_mask_ready,
        } | (self_embedding_initializer_outputs or {})

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
            Q_L_init=clean_outputs["Q_L_init"],
            C_L_init=clean_outputs["C_L"],
            P_LL=clean_outputs["P_LL"],
            S_I_init=clean_outputs["S_I"],
            Z_II=clean_outputs["Z_II"],
            chunked_pairwise_embedder=clean_outputs.get("chunked_pairwise_embedder"),
            initializer_outputs=clean_outputs,
        )
        clean_flex_block_mask = clean_pass_inputs.pop("flex_block_mask")
        clean_flex_block_mask_ready = clean_pass_inputs.pop("flex_block_mask_ready")
        clean_features = self.process_(
            D_II_self=None,
            X_L_self=None,
            flex_block_mask=clean_flex_block_mask,
            flex_block_mask_ready=clean_flex_block_mask_ready,
            token_flex_block_mask=clean_outputs.get("token_flex_block_mask"),
            token_flex_block_mask_ready=clean_outputs.get(
                "token_flex_block_mask_ready", False
            ),
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
            Q_L_init=Q_L_init,
            C_L_init=C_L_init,
            P_LL=P_LL,
            S_I_init=S_I_init,
            Z_II=Z_II_init,
            chunked_pairwise_embedder=chunked_pairwise_embedder,
            initializer_outputs=initializer_outputs,
        )
        noisy_flex_block_mask = noisy_pass_inputs.pop("flex_block_mask")
        noisy_flex_block_mask_ready = noisy_pass_inputs.pop("flex_block_mask_ready")
        recycled_features = self.process_(
            D_II_self=None,
            X_L_self=None,
            flex_block_mask=noisy_flex_block_mask,
            flex_block_mask_ready=noisy_flex_block_mask_ready,
            token_flex_block_mask=token_flex_block_mask,
            token_flex_block_mask_ready=token_flex_block_mask_ready,
            self_embedding_condition_B=condition_B,
            **noisy_pass_inputs,
        )
        return self._public_outputs(
            recycled_features,
            return_self_embedding_features=return_self_embedding_features,
        )

    def forward_with_recycle(
        self,
        n_recycle,
        flex_block_mask=None,
        flex_block_mask_ready=False,
        token_flex_block_mask=None,
        token_flex_block_mask_ready=False,
        **kwargs,
    ) -> dict:
        """
        Perform forward pass with recycling. During training, n_recycle must be provided as an argument. During inference, n_recycle can be set as an attribute of the model and will default to that if not provided as an argument.

        Args:
            n_recycle: Number of recycling iterations to perform (if None, uses self.n_recycle)
            flex_block_mask: Optional precomputed flexible block mask for attention (only valid for the first iteration)
            flex_block_mask_ready: Whether the flex_block_mask is ready to be used (only True for the first iteration)
            token_flex_block_mask: Optional precomputed flexible block mask for token-level attention
            token_flex_block_mask_ready: Whether the token_flex_block_mask is ready to be used
            **kwargs: Keyword arguments containing all necessary inputs for the forward pass

        Returns:
            recycled_features: Dictionary containing the outputs from the final recycling iteration, including:
                "X_L": Denoised output positions [B, L, 3]
                "V_L" (optional): Denoised velocity vectors [B, L, 3] if present in recycled_features
                "D_II_self": Self-distance features [B, L, L, c_atompair]
                "sequence_indices_I": Predicted sequence indices [B, n_tokens]
                "sequence_logits_I": Predicted sequence logits [B, n_tokens, c_token]
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
                    D_II_self=recycled_features.get("D_II_self"),
                    X_L_self=recycled_features.get("X_L"),
                    flex_block_mask=flex_block_mask,
                    flex_block_mask_ready=flex_block_mask_ready,
                    token_flex_block_mask=token_flex_block_mask,
                    token_flex_block_mask_ready=token_flex_block_mask_ready,
                    **kwargs,
                )

        return recycled_features

    def process_(
        self,
        D_II_self,
        X_L_self,
        flex_block_mask=None,
        flex_block_mask_ready=False,
        token_flex_block_mask=None,
        token_flex_block_mask_ready=False,
        *,
        R_L_uniform,
        X_noisy_L,
        t_L,
        f,
        Q_L,
        C_L,
        P_LL,
        A_I,
        S_I,
        Z_II,
        chunked_pairwise_embedder=None,
        initializer_outputs=None,
        self_embedding_condition_B=None,
        **_,
    ) -> dict:
        """
        Process one iteration of the diffusion forward pass, including embedding, transformer, and decoding steps.

        Args:
            D_II_self: Self-distance features from the previous iteration [B, L, L, c_atompair] (or None for the first iteration)
            X_L_self: Denoised positions from the previous iteration [B, L, 3] (or None for the first iteration)
            flex_block_mask: Optional precomputed flexible block mask for attention (only valid for the first iteration)
            flex_block_mask_ready: Whether the flex_block_mask is ready to be used (only True for the first iteration)
            token_flex_block_mask: Optional precomputed flexible block mask for token-level attention
            token_flex_block_mask_ready: Whether the token_flex_block_mask is ready to be used
            R_L_uniform: Scaled noisy positions for input to the model [B, L, 3]
            X_noisy_L: Original noisy input positions [B, L, 3]
            t_L: Noise level tensor for atoms [B, L]
            f: Feature dictionary containing necessary features and masks for attention and pooling
            Q_L: Atom-level features from the encoder [B, L, c_atom]
            C_L: Atom-level features for time embedding [B, L, c_atom]
            P_LL: Atom-pair features [B, L, L, c_atompair]
            A_I: Sequence-level features after pooling [B, n_tokens, c_token]
            S_I: Sequence-level features for time embedding [B, n_tokens, c_s]
            Z_II: Token-pair features [B, n_tokens, n_tokens, c_z]
            chunked_pairwise_embedder: Optional embedder for computing pairwise features in a memory-efficient chunked manner
            initializer_outputs: Optional outputs from the initializer to be used in chunked mode
            self_embedding_condition_B: Optional self-embedding condition tensor [B, c_token] for self-embedding guidance

        Returns:
            outputs: Dictionary containing:
                "X_L": Denoised output positions [B, L, 3]
                "V_L" (optional): Denoised velocity vectors [B, L, 3] if present in recycled_features
                "D_II_self": Self-distance features [B, L, L, c_atompair]
                "sequence_indices_I": Predicted sequence indices [B, n_tokens]
                "sequence_logits_I": Predicted sequence logits [B, n_tokens, c_token]
        """
        if self.self_embedding_conditioner is not None:
            A_I, S_I = self.self_embedding_conditioner.condition_token_inputs(
                A_I=A_I,
                S_I=S_I,
                condition_B=self_embedding_condition_B,
                batch_token_lvl=f["batch_token_lvl"],
            )

        # ... Embed token level features with atom level encodings
        S_I, Z_II = self.diffusion_token_encoder(
            f=f,
            R_L=R_L_uniform,
            D_II_self=D_II_self,
            S_init_I=S_I,
            Z_init_II=Z_II,
            C_L=C_L,
            P_LL=P_LL,
            token_flex_block_mask=token_flex_block_mask,
            token_flex_block_mask_ready=token_flex_block_mask_ready,
        )

        # ... Diffusion transformer
        A_I = self.diffusion_transformer(
            A_I,
            S_I,
            Z_II,
            f=f,
            X_L=(
                X_noisy_L[..., f["is_ca"], :]
                if X_L_self is None
                else X_L_self[..., f["is_ca"], :]
            ),
            full=not (os.environ.get("RFD3_LOW_MEMORY_MODE", None) == "1"),
            use_local_attention=self.use_local_token_attention,
            # In the first recyling iteration, we can reuse the atom-level block mask.
            # Once coordinates are updated, the original block mask is no longer valid,
            # so we recompute it in the Diffusion Transformer by passing None here.
            flex_block_mask=flex_block_mask if X_L_self is None else None,
            flex_block_mask_ready=flex_block_mask_ready if X_L_self is None else False,
        )
        self_embedding_token_features = A_I
        # ... Decoder readout
        # Check if using chunked P_LL mode

        if chunked_pairwise_embedder is not None:
            # Chunked mode: pass embedder and no P_LL
            raise NotImplementedError(
                "Chunked P_LL mode is not yet implemented with sparse input batch support."
            )
            A_I, Q_L, o = self.decoder(
                A_I,
                S_I,
                Q_L,
                C_L,
                P_LL=None,  # Not used in chunked mode
                tok_idx=f["atom_to_token_map"],
                indices=f["attn_indices"],
                index_mask=f["attn_index_mask"],
                f=f,  # Pass f for chunked computation
                chunked_pairwise_embedder=chunked_pairwise_embedder,
                initializer_outputs=initializer_outputs,
                flex_block_mask=flex_block_mask,
                flex_block_mask_ready=flex_block_mask_ready,
            )
        else:
            # Original mode: use full P_LL
            A_I, Q_L, o = self.decoder(
                A_I,
                S_I,
                Q_L,
                C_L,
                P_LL=P_LL,
                tok_idx=f["atom_to_token_map"],
                indices=f["attn_indices"],
                index_mask=f["attn_index_mask"],
                f=f,
                flex_block_mask=flex_block_mask,
                flex_block_mask_ready=flex_block_mask_ready,
            )
        self_embedding_atom_features = Q_L

        # ... Process outputs to positions update
        R_update_L = self.to_r_update(Q_L)
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

        sequence_logits_I, sequence_indices_I = self.sequence_head(A_I=A_I)
        atom_type_logits_I, atom_type_indices_I = self.atom_type_head(A_I=A_I)
        fractional_coord_L = self.fractional_coord_head(Q_L=Q_L)

        D = f["diffusion_batch_size"]
        B = f["batch_size"]
        A_sys_init = self.lattice_init_token.type(A_I.dtype).repeat(D, B, 1)
        A_sys = self.downcast_lattice(
            Q_L=A_I,
            A_I=A_sys_init,
            S_I=None,
            tok_idx=f["batch_token_lvl"],
            n_atoms_per_tok_max=max(f["batch_num_tokens"]),
        )
        lattice_lengths_B, lattice_angles_B = self.lattice_head(A_I=A_sys)

        D_II_self = self.bucketize_fn(X_out_L[..., f["is_ca"], :].detach())

        # ... Mask out distances between different examples in the batch
        D_II_self = D_II_self.masked_fill(
            f["unindexing_pair_mask"][None, ..., None], 0.0
        )

        outputs = {
            "X_L": X_out_L,
            "D_II_self": D_II_self,
            "sequence_logits_I": sequence_logits_I,
            "sequence_indices_I": sequence_indices_I,
            "atom_type_logits_I": atom_type_logits_I,
            "atom_type_indices_I": atom_type_indices_I,
            "fractional_coord_L": fractional_coord_L,
            "lattice_lengths_B": lattice_lengths_B,
            "lattice_angles_B": lattice_angles_B,
            "self_embedding_token_features": self_embedding_token_features,
            "self_embedding_atom_features": self_embedding_atom_features,
        } | o
        if velocity_L is not None:
            outputs["V_L"] = velocity_L
        return outputs
