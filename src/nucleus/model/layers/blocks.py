import torch
import torch.nn as nn
import torch.nn.functional as F
from atomworks.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER
from foundry import DISABLE_CHECKPOINTING
from foundry.common import exists
from foundry.utils.torch import scatter_mean
from torch.nn.functional import one_hot

from nucleus.model.layers.attention import (
    LocalAttentionPairBias,
    ResidualGatedCrossAttention,
    indices_to_mask_with_padding,
    infer_attention_mask_from_features,
)
from nucleus.model.layers.block_utils import (
    create_attention_indices,
)
from nucleus.model.layers.charge_spin_embedding import ChgSpinEmbedding
from nucleus.model.layers.layer_utils import Transition
from nucleus.utils.attention import (
    maybe_create_flex_block_mask,
)
from nucleus.utils.block import build_valid_mask

try:
    from rfd3.model.layers.attention import (
        GatedCrossAttention,
    )
    from rfd3.model.layers.block_utils import (
        group_atoms,
        ungroup_atoms,
    )
    from rfd3.model.layers.blocks import (
        ConditionedTransitionBlock,
    )
    from rfd3.model.layers.blocks import (
        Downcast as RFD3Downcast,
    )
    from rfd3.model.layers.blocks import (
        LinearEmbedWithPool as RFD3LinearEmbedWithPool,
    )
    from rfd3.model.layers.blocks import (
        Upcast as RFD3Upcast,
    )
    from rfd3.model.layers.layer_utils import (
        RMSNorm,
        linearNoBias,
    )
except ImportError:
    raise ImportError(
        "Could not import RFD3 modules. Please ensure that the `rc-foundry[all]` package is installed and up to date."
    )


class RelativePositionEncodingWithIndexRemoval(nn.Module):
    """
    Usual RPE but utilizes `is_motif_atom_3d_unindexed` to ensure within-chain position
    is spoofed and that inter-batch interactions are properly masked.

    Args:
        r_max: maximum relative residue position (in either direction) to encode. Distances above this will be clipped to the same value, and distances between different chains will be given a separate encoding.
        s_max: maximum relative sym_id position (in either direction) to encode. Distances above this will be clipped to the same value, and distances between different chains will be given a separate encoding.
        c_z: number of channels for the output encoding.
    """

    def __init__(self, r_max, s_max, c_z):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z

        self.num_tok_pos_bins = (
            2 * self.r_max + 2
        ) + 1  # original af3 + 1 for unknown index
        self.linear = linearNoBias(
            2 * self.num_tok_pos_bins + (2 * self.s_max + 2) + 1, c_z
        )

    def forward(self, f) -> torch.Tensor:
        """
        Compute relative position encodings with index removal.

        Args:
            f: dictionary containing necessary features, including:
                - "asym_id": [L] chain identifiers for each atom
                - "entity_id": [L] entity identifiers for each atom
                - "residue_index": [L] residue indices for each atom
                - "token_index": [L] token indices for each atom
                - "sym_id": [L] symmetry identifiers for each atom
                - "unindexing_pair_mask": [L, L] boolean mask indicating which atom pairs should have their positional encodings removed (e.g., due to unindexed motifs or being in different examples)
                - "batch_token_lvl": [L] batch indices for each token, used to identify pairs of atoms that belong to different examples

        Returns:
            Tensor of shape [L, L, c_z] containing the relative position encodings for each pair of atoms, with appropriate handling for unindexed motifs and inter-example pairs.
        """
        b_samechain_II = f["asym_id"].unsqueeze(-1) == f["asym_id"].unsqueeze(-2)
        b_same_entity_II = f["entity_id"].unsqueeze(-1) == f["entity_id"].unsqueeze(-2)
        d_residue_II = torch.where(
            b_samechain_II,
            torch.clip(
                f["residue_index"].unsqueeze(-1)
                - f["residue_index"].unsqueeze(-2)
                + self.r_max,
                0,
                2 * self.r_max,
            ),
            2 * self.r_max + 1,
        )
        b_sameresidue_II = f["residue_index"].unsqueeze(-1) == f[
            "residue_index"
        ].unsqueeze(-2)
        tok_distance = (
            f["token_index"].unsqueeze(-1) - f["token_index"].unsqueeze(-2) + self.r_max
        )
        d_token_II = torch.where(
            b_samechain_II * b_sameresidue_II,
            torch.clip(
                tok_distance,
                0,
                2 * self.r_max,
            ),
            2 * self.r_max + 1,
        )

        # Chain distances are kept
        d_chain_II = torch.where(
            # NOTE: Implementing bugfix from the Protenix Technical report, where we use `same_entity` instead of `not same_chain` (as in the AF-3 pseudocode)
            # Reference: https://github.com/bytedance/Protenix/blob/main/Protenix_Technical_Report.pdf
            b_same_entity_II,
            torch.clip(
                f["sym_id"].unsqueeze(-1) - f["sym_id"].unsqueeze(-2) + self.s_max,
                0,
                2 * self.s_max,
            ),
            2 * self.s_max + 1,
        )

        #########################################################
        # Cancel out distances from unidexed motifs or between different examples
        unindexing_pair_mask = f[
            "unindexing_pair_mask"
        ]  # [L, L] representing the parts which shouldnt' talk to one another
        different_example_mask = f["batch_token_lvl"].unsqueeze(-1) != f[
            "batch_token_lvl"
        ].unsqueeze(-2)  # [L, L] representing the parts which are in different examples

        # Special position cases
        d_token_II[unindexing_pair_mask] = (
            # NOTE: `unindexing_pair_mask` already accounts for different examples.
            self.num_tok_pos_bins - 1
        )
        d_residue_II[unindexing_pair_mask] = self.num_tok_pos_bins - 1
        A_relpos_II = one_hot(d_residue_II.long(), self.num_tok_pos_bins)
        A_reltoken_II = one_hot(d_token_II, self.num_tok_pos_bins)

        # Special example-boundary case
        b_same_entity_II = b_same_entity_II & (~different_example_mask)

        d_chain_II[different_example_mask] = 2 * self.s_max + 1
        A_relchain_II = one_hot(d_chain_II.long(), 2 * self.s_max + 2)
        #########################################################

        return self.linear(
            torch.cat(
                [
                    A_relpos_II,
                    A_reltoken_II,
                    b_same_entity_II.unsqueeze(-1),
                    A_relchain_II,
                ],
                dim=-1,
            ).to(torch.float)
        )


class LinearAtomTypesHead(nn.Module):
    """
    Linear head for predicting atom types from molecule token features.
    This is used in the diffusion module to predict the atom types of
    the denoised molecule structure at later denoising steps.

    Args:
        c_token: number of channels in the token features
    """

    def __init__(self, c_token):
        super().__init__()
        n_tok_all = 128
        disallowed_idxs = [ELEMENT_NAME_TO_ATOMIC_NUMBER[k] for k in ["X"]]
        mask = torch.ones(n_tok_all, dtype=torch.bool)
        mask[disallowed_idxs] = False
        self.register_buffer("valid_out_mask", mask)
        self.linear = nn.Linear(c_token, n_tok_all)

    def forward(self, A_I, **_):
        """
        Compute the forward pass of the linear atom types head.

        Args:
            A_I: [B, L, C_token] token features for each token,
                where B is the batch size, L is the number of tokens,
                and C_token is the number of channels in the token features.

        Returns:
            logits: [B, L, n_tok_all] output logits for each token, where
            n_tok_all is the total number of possible token types (including
            invalid ones that will be masked out).
        """
        logits = self.linear(A_I)
        indices = self.decode(logits)
        return logits, indices

    def decode(self, logits):
        """
        Decode the output logits to obtain the predicted atom type indices,
        applying a mask to ensure that only valid atom types are predicted.

        Args:
            logits: Tensor of shape [D, L, n_tok_all] containing the output logits
                from the linear layer, where D is the batch size, L is the number
                of tokens, and n_tok_all is the number of atom-type classes.

        Returns:
            Tensor of shape [D, L] containing the predicted atom type indices, where each
            index corresponds to a valid atom type as determined by the `valid_out_mask`.
        """
        # logits: [D, L, 128]
        # indices: [D, L] in [0,128-1]
        probs = F.softmax(logits, dim=-1)
        probs = probs * self.valid_out_mask[None, None, :].to(probs.device)
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        indices = probs.argmax(dim=-1)
        return indices


class AdaLNFeatureProcessor(nn.Module):
    """Normalize predicted features and modulate them with static conditioning.

    The modulation projection is zero-initialized, so this module initially
    behaves exactly like an RMSNorm. This is the feature-processing portion of
    an adaLN-Zero output head and can be shared by continuous and discrete
    prediction heads.

    Args:
        c_features: Number of channels in the predicted features.
        c_conditioning: Number of channels in the conditioning features. Defaults
            to ``c_features``.
    """

    def __init__(self, c_features: int, c_conditioning: int | None = None):
        super().__init__()
        c_conditioning = c_features if c_conditioning is None else c_conditioning

        self.process_c = nn.Sequential(
            nn.SiLU(), linearNoBias(c_conditioning, c_features * 2)
        )
        self.process_q = RMSNorm(c_features)
        AdaLNFeatureProcessor.reset_parameters(self)

    def reset_parameters(self) -> None:
        """Zero-initialize the conditioning modulation."""
        nn.init.zeros_(self.process_c[-1].weight)

    def forward(self, Q: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm followed by conditioning-dependent shift and scale."""
        beta, gamma = self.process_c(C).chunk(2, dim=-1)
        return self.process_q(Q) * (1 + gamma) + beta


class AdaLNCoordinatesHead(AdaLNFeatureProcessor):
    """Adaptive layer-normalized head for Euclidean or fractional coordinates."""

    def __init__(self, c_atom):
        super().__init__(c_features=c_atom, c_conditioning=c_atom)
        self.process_r = linearNoBias(c_atom, 3)  # Predicts the final coordinates
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Zero-initialize both modulation and coordinate projection."""
        super().reset_parameters()
        nn.init.zeros_(self.process_r.weight)

    def forward(self, Q_L, C_L):
        """
        Compute the forward pass of the AdaLN coordinates head.

        Args:
            Q_L: [B, L, C_atom] features for each atom
            C_L: [B, L, C_atom] conditioning features for each atom

        Returns:
            Tensor of shape [B, L, 3] containing the predicted coordinates for each atom.
        """
        return self.process_r(super().forward(Q_L, C_L))


class LatentEquivarianceHead(nn.Module):
    """Predict rotated system embeddings from a base view and relative rotation.

    This is the minimal TransIP transformation network: final atom embeddings
    are mean-pooled per system, and a two-layer MLP maps ``[h, R]`` to the
    embedding of another SO(3)-augmented view. A flattened rotation matrix avoids
    the quaternion double cover while retaining the complete group action.
    """

    def __init__(self, feature_dim: int, hidden_multiplier: float = 2.0):
        super().__init__()
        if hidden_multiplier <= 0:
            raise ValueError("hidden_multiplier must be positive.")
        hidden_dim = int(hidden_multiplier * feature_dim)
        self.transform = nn.Sequential(
            nn.Linear(feature_dim + 9, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(
        self,
        system_embeddings: torch.Tensor,
        rotations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return transformed base embeddings and observed rotated embeddings."""
        if system_embeddings.ndim != 3:
            raise ValueError(
                "system_embeddings must have shape [D, B, C], got "
                f"{tuple(system_embeddings.shape)}."
            )
        diffusion_batch_size, batch_size, feature_dim = system_embeddings.shape
        if diffusion_batch_size < 2:
            raise ValueError("Latent equivariance requires diffusion_batch_size >= 2.")
        expected_rotation_shape = (diffusion_batch_size, batch_size, 3, 3)
        if rotations.shape != expected_rotation_shape:
            raise ValueError(
                "rotations must have shape "
                f"{expected_rotation_shape}, got {tuple(rotations.shape)}."
            )

        rotations = rotations.to(
            device=system_embeddings.device,
            dtype=system_embeddings.dtype,
        )
        relative_rotations = rotations[1:] @ rotations[0].transpose(-1, -2).unsqueeze(0)
        base_embeddings = system_embeddings[0].expand(
            diffusion_batch_size - 1, batch_size, feature_dim
        )
        transformation_input = torch.cat(
            (base_embeddings, relative_rotations.flatten(start_dim=-2)), dim=-1
        )
        return self.transform(transformation_input), system_embeddings[1:]


class LinearFractionalCoordinatesHead(nn.Module):
    """Linear head for material fractional coordinates in the unit cell."""

    def __init__(self, c_atom):
        super().__init__()
        self.linear = nn.Linear(c_atom, 3)

    def forward(self, Q_L):
        return torch.sigmoid(self.linear(Q_L))


class LinearLatticeHead(nn.Module):
    """Linear token head for normalized lattice lengths and radian angles."""

    def __init__(self, c_token):
        super().__init__()
        self.linear_lengths = nn.Linear(c_token, 3)
        self.linear_angles = nn.Linear(c_token, 3)

    def forward(self, A_I):
        lengths = F.softplus(self.linear_lengths(A_I))
        angles = torch.pi * torch.sigmoid(self.linear_angles(A_I))
        return lengths, angles


class AtomisticEnergyHead(nn.Module):
    """Predict standardized residual energy per atom from final token features.

    A shared two-layer MLP produces token features, while the final scalar
    projection is domain specific. Token contributions are summed within each
    system and divided by its atom count, matching the normalization used for
    atomistic energy targets.
    """

    def __init__(
        self,
        *,
        c_token: int,
        n_domains: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or c_token
        self.n_domains = n_domains
        self.shared_mlp = nn.Sequential(
            RMSNorm(c_token),
            linearNoBias(c_token, hidden_dim),
            nn.SiLU(),
            linearNoBias(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.domain_projections = nn.ModuleList(
            [linearNoBias(hidden_dim, 1) for _ in range(n_domains)]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Start from the composition-only normalization baseline."""
        for projection in self.domain_projections:
            nn.init.zeros_(projection.weight)

    def forward(
        self,
        Q_I: torch.Tensor,
        *,
        batch_token_lvl: torch.Tensor,
        domain_id_B: torch.Tensor,
        atom_count_B: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return system predictions and selected per-token contributions.

        Args:
            Q_I: Final token features ``[D, I, c_token]``.
            batch_token_lvl: System index for each token ``[I]``.
            domain_id_B: Electronic-structure domain per system ``[B]``.
            atom_count_B: Number of retained atoms per system ``[B]``.

        Returns:
            A tuple containing standardized residual energy per atom ``[D, B]``
            and selected scalar token contributions ``[D, I]``.
        """
        if Q_I.ndim != 3:
            raise ValueError(f"Q_I must have shape [D, I, C]; got {Q_I.shape}.")

        device = Q_I.device
        batch_token_lvl = batch_token_lvl.to(device=device, dtype=torch.long)
        domain_id_B = domain_id_B.to(device=device, dtype=torch.long).reshape(-1)
        atom_count_B = atom_count_B.to(device=device, dtype=Q_I.dtype).reshape(-1)
        batch_size = domain_id_B.numel()
        if atom_count_B.numel() != batch_size:
            raise ValueError("atom_count_B and domain_id_B must have equal length.")
        if batch_token_lvl.numel() != Q_I.shape[1]:
            raise ValueError("batch_token_lvl must contain one index per token.")

        hidden_I = self.shared_mlp(Q_I)
        all_contributions_I = torch.stack(
            [
                projection(hidden_I).squeeze(-1)
                for projection in self.domain_projections
            ],
            dim=-1,
        )
        valid_domain_B = (domain_id_B >= 0) & (domain_id_B < self.n_domains)
        safe_domain_B = domain_id_B.clamp(min=0, max=self.n_domains - 1)
        safe_domain_I = safe_domain_B[batch_token_lvl]
        token_contributions_I = all_contributions_I.gather(
            -1,
            safe_domain_I.view(1, -1, 1).expand(Q_I.shape[0], -1, 1),
        ).squeeze(-1)
        token_contributions_I = token_contributions_I * valid_domain_B[
            batch_token_lvl
        ].to(dtype=Q_I.dtype).unsqueeze(0)

        summed_energy_B = Q_I.new_zeros((Q_I.shape[0], batch_size))
        summed_energy_B.index_add_(1, batch_token_lvl, token_contributions_I)
        normalized_energy_B = summed_energy_B / atom_count_B.clamp_min(1).unsqueeze(0)
        return normalized_energy_B, token_contributions_I


######################################################################################
##########################     Local Atom Transformer       ##########################
######################################################################################


class LocalTokenTransformer(nn.Module):
    """
    Transformer operating over the sequence of local tokens, where each token attends to a local neighborhood of other tokens determined by `create_attention_indices`. This is used as part of the streaming decoder, where we want to allow for some communication between the local windows.

    Args:
        c_token: number of channels in the token features
        c_tokenpair: number of channels in the pairwise token features
        c_s: number of channels in the global structure features (if using)
        n_block: number of transformer blocks to apply
        diffusion_transformer_block: dictionary of hyperparameters for the transformer blocks (e.g., number of attention heads, dropout rate, etc.)
        n_registers: number of registers to use for sparse attention (if using a sparse attention implementation that requires this)
        n_local_tokens: number of local tokens on either side to attend to (i.e., a token will attend to 2*n_local_tokens + 1 tokens total, including itself)
        n_keys: number of keys to use in the attention mechanism (if using a sparse attention implementation that requires this)
        attn_backend: attention backend to use for the attention pair bias in the transformer blocks ("manual", "megafold", or "flex")
    """

    def __init__(
        self,
        c_token,
        c_tokenpair,
        c_s,
        n_block,
        diffusion_transformer_block,
        n_registers=None,
        n_local_tokens=8,
        n_keys=32,
        attn_backend="manual",
    ):
        super().__init__()
        self.n_local_tokens = n_local_tokens
        self.n_keys = n_keys
        self.blocks = nn.ModuleList(
            [
                StructureLocalAtomTransformerBlock(
                    c_atom=c_token,
                    c_s=c_s,
                    c_atompair=c_tokenpair,
                    **diffusion_transformer_block,
                )
                for _ in range(n_block)
            ]
        )
        self.use_flex_attention = attn_backend == "flex"
        self.use_checkpointing = not DISABLE_CHECKPOINTING

    def forward(
        self,
        A_I,
        S_I,
        Z_II,
        f,
        X_L,
        full=False,
        use_local_attention=True,
        flex_block_mask=None,
        flex_block_mask_ready=False,
    ) -> torch.Tensor:
        """
        Compute the forward pass of the local token transformer.

        Args:
            A_I: [B, L, C_token] atom features for each token
            S_I: [B, L, C_s] global structure features for each token (if using)
            Z_II: [B, L, L, C_tokenpair] pairwise token features for each pair of tokens
            f: dictionary containing necessary features for computing attention indices, including:
                - "token_index": [L] token indices for each atom, used to determine local neighborhoods for attention
            X_L: [B, L, C_token] input token features, used to determine attention indices
            full: boolean flag indicating whether to use the full attention (i.e., attend to all tokens) or the local attention (attend only to local neighbors). If True, the attention will be computed over all tokens regardless of the local neighborhood. This can be used during training to allow for better gradient flow, while during inference we can set this to False to save computation by only attending to local neighbors.
            use_local_attention: whether to build sparse/kNN token attention indices. If False, use dense token-token indices and avoid `create_attention_indices`.
            flex_block_mask: optional precomputed FlexAttention block_mask (as returned by `create_flex_attention_block_mask` / `torch.nn.attention.flex_attention.create_block_mask`). When provided, it will be forwarded to the flex attention kernel.
            flex_block_mask_ready: set True once the caller has either computed `flex_block_mask` (or determined no block mask is needed / available), so downstream layers do not attempt to build it again.

        Returns:
            A_I: [B, L, C_token] updated atom features for each token after applying the local token transformer blocks.
        """
        tok_idx = torch.arange(A_I.shape[1], device=A_I.device)
        if use_local_attention:
            indices, index_mask = create_attention_indices(
                X_L=X_L,
                f=f,
                tok_idx=tok_idx,
                n_attn_keys=self.n_keys,
                n_attn_seq_neighbours=self.n_local_tokens,
            )
        else:
            indices = tok_idx.view(1, 1, -1).expand(A_I.shape[0], A_I.shape[1], -1)
            index_mask = None
            full = True

        if (
            self.use_flex_attention
            and flex_block_mask is None
            and not flex_block_mask_ready
        ):
            attn_mask = infer_attention_mask_from_features(
                f, indices, tok_idx=tok_idx, index_mask=index_mask
            )
            flex_block_mask, flex_block_mask_ready = maybe_create_flex_block_mask(
                attn_mask=indices_to_mask_with_padding(indices, attn_mask),
                q_len=A_I.shape[1],
                device=A_I.device,
                use_flex_attention=self.use_flex_attention,
                flex_block_mask=flex_block_mask,
                flex_block_mask_ready=flex_block_mask_ready,
            )

        for i, block in enumerate(self.blocks):
            # Set checkpointing
            block.attention_pair_bias.use_checkpointing = self.use_checkpointing
            # A_I: [B, L, C_token]
            # S_I: [B, L, C_s]
            # Z_II: [B, L, L, C_tokenpair]
            A_I = block(
                A_I,
                S_I,
                Z_II,
                indices=indices,
                index_mask=index_mask,
                f=f,
                tok_idx=tok_idx,
                full=full,  # (self.training and torch.is_grad_enabled()),  # Does not accelerate inference, but memory *does* scale better
                flex_block_mask=flex_block_mask,
                flex_block_mask_ready=flex_block_mask_ready,
            )

        return A_I


class LocalAtomTransformer(nn.Module):
    """
    Transformer operating over the sequence of atoms, where each atom attends to all other atoms in the structure. This is used in the encoder, where we want to allow for global communication between all atoms.

    Args:
        c_atom: number of channels in the atom features
        c_s: number of channels in the global structure features
        c_atompair: number of channels in the pairwise atom features
        atom_transformer_block: dictionary containing the configuration for the atom transformer block
        n_blocks: number of transformer blocks to apply
        attn_backend: attention backend to use for the attention pair bias in the transformer blocks ("manual", "megafold", or "flex")

    Returns:
        Q_L: [B, L, C_atom] updated atom features for each atom after applying the local atom transformer blocks.
    """

    def __init__(
        self, c_atom, c_s, c_atompair, atom_transformer_block, n_blocks, attn_backend
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                StructureLocalAtomTransformerBlock(
                    c_atom=c_atom,
                    c_s=c_s,
                    c_atompair=c_atompair,
                    **atom_transformer_block,
                )
                for _ in range(n_blocks)
            ]
        )
        self.use_flex_attention = attn_backend == "flex"

    def forward(
        self,
        Q_L,
        C_L,
        P_LL,
        indices=None,
        index_mask=None,
        f=None,
        tok_idx=None,
        flex_block_mask=None,
        flex_block_mask_ready=False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the forward pass of the local atom transformer.

        Args:
            Q_L: [B, L, C_atom] atom features for each atom
            C_L: [B, L, C_s] global structure features for each atom
            P_LL: [B, L, L, C_atompair] pairwise atom features for each pair of atoms
            indices: precomputed attention indices for the local atom transformer blocks, used to determine which atoms attend to which other atoms based on their token indices and the local neighborhood size
            index_mask: optional boolean mask marking real slots in indices
            f: dictionary containing necessary features for computing attention indices and pairwise biases, including:
                - "residue_index": [L] residue indices for each atom, used for computing attention biases based on relative residue positions
                - "asym_id": [L] chain identifiers for each atom, used for computing attention biases based on whether atoms are in the same chain or not
                - "entity_id": [L] entity identifiers for each atom, used for computing attention biases based on whether atoms are in the same entity or not
                - "sym_id": [L] symmetry identifiers for each atom, used for computing attention biases based on relative symmetry positions
                - "unindexing_pair_mask": [L, L] boolean mask indicating which atom pairs should have their attention biases removed (e.g., due to unindexed motifs or being in different examples)
                - "batch_token_lvl": [L] batch indices for each token, used to identify pairs of atoms that belong to different examples
            tok_idx: [L] token indices for each atom, used for computing attention indices and for the upcast and downcast modules to know which token each atom belongs to
            flex_block_mask: optional precomputed FlexAttention block_mask (as returned by `create_flex_attention_block_mask` / `torch.nn.attention.flex_attention.create_block_mask`). When provided, it will be forwarded to the flex attention kernel.
            flex_block_mask_ready: set True once the caller has either computed `flex_block_mask` (or determined no block mask is needed / available), so downstream layers do not attempt to build it again.
            **kwargs: additional keyword arguments that may be needed for the transformer blocks (e.g., features for computing attention indices, etc.)

        Returns:
            Q_L: [B, L, C_atom] updated atom features for each atom after applying the local atom transformer blocks.
        """
        if (
            indices is not None
            and f is not None
            and self.use_flex_attention
            and not flex_block_mask_ready
        ):
            attn_mask = infer_attention_mask_from_features(
                f, indices, tok_idx=tok_idx, index_mask=index_mask
            )
            flex_block_mask, flex_block_mask_ready = maybe_create_flex_block_mask(
                attn_mask=indices_to_mask_with_padding(indices, attn_mask),
                q_len=Q_L.shape[1],
                device=Q_L.device,
                use_flex_attention=self.use_flex_attention,
                flex_block_mask=flex_block_mask,
                flex_block_mask_ready=flex_block_mask_ready,
            )

        for block in self.blocks:
            Q_L = block(
                Q_L,
                C_L,
                P_LL,
                f=f,
                indices=indices,
                index_mask=index_mask,
                flex_block_mask=flex_block_mask,
                flex_block_mask_ready=flex_block_mask_ready,
                **kwargs,
            )
        return Q_L


class StructureLocalAtomTransformerBlock(nn.Module):
    """
    Single block of the local atom transformer, consisting of a local attention with pair bias followed by a transition block. The attention allows each atom to attend to all other atoms in the structure, with biases derived from the pairwise atom features. The transition block allows for further mixing of information across channels.

    Args:
        c_atom: number of channels in the atom features
        c_s: number of channels in the global structure features
        c_atompair: number of channels in the pairwise atom features
        dropout: dropout rate to apply after the attention and transition blocks
        no_residual_connection_between_attention_and_transition: whether to disable the residual connection between the attention and transition blocks
        **transformer_block: additional keyword arguments for the transformer block (e.g., number of attention heads, etc.)
    """

    def __init__(
        self,
        *,
        c_atom,
        c_s,
        c_atompair,
        dropout,
        no_residual_connection_between_attention_and_transition,
        **transformer_block,
    ):
        super().__init__()
        assert not no_residual_connection_between_attention_and_transition
        self.c_s = c_s
        self.dropout = nn.Dropout(dropout)
        self.attention_pair_bias = LocalAttentionPairBias(
            c_a=c_atom, c_s=c_s, c_pair=c_atompair, **transformer_block
        )
        if exists(c_s):
            self.transition_block = ConditionedTransitionBlock(c_token=c_atom, c_s=c_s)
        else:
            self.transition_block = Transition(c=c_atom, n=4)

    def forward(
        self,
        Q_L,  # [..., I, C_token]
        C_L,  # [..., I, C_s]
        P_LL,  # [..., I, I, C_tokenpair]
        f=None,
        chunked_pairwise_embedder=None,
        initializer_outputs=None,
        flex_block_mask=None,
        flex_block_mask_ready=False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the forward pass of the local atom transformer block.

        Args:
            Q_L: [B, L, C_atom] atom features for each atom
            C_L: [B, L, C_s] global structure features for each atom (if using)
            P_LL: [B, L, L, C_atompair] pairwise atom features for each pair of atoms
            f: dictionary containing necessary features for computing attention indices and pairwise biases, including:
                - "residue_index": [L] residue indices for each atom, used for computing attention biases based on relative residue positions
                - "asym_id": [L] chain identifiers for each atom, used for computing attention biases based on whether atoms are in the same chain or not
                - "entity_id": [L] entity identifiers for each atom, used for computing attention biases based on whether atoms are in the same entity or not
                - "sym_id": [L] symmetry identifiers for each atom, used for computing attention biases based on relative symmetry positions
                - "unindexing_pair_mask": [L, L] boolean mask indicating which atom pairs should have their attention biases removed (e.g., due to unindexed motifs or being in different examples)
                - "batch_token_lvl": [L] batch indices for each token, used to identify pairs of atoms that belong to different examples
            chunked_pairwise_embedder: optional module for computing pairwise embeddings in a memory-efficient way by chunking (if the full pairwise features cannot fit in memory at once)
            initializer_outputs: optional outputs from the initializer, which may contain precomputed features that can be reused for efficiency (e.g., precomputed pairwise embeddings, etc.)
            flex_block_mask: optional precomputed FlexAttention block_mask (as returned by `create_flex_attention_block_mask` / `torch.nn.attention.flex_attention.create_block_mask`). When provided, it will be forwarded to the flex attention kernel.
            flex_block_mask_ready: set True once the caller has either computed `flex_block_mask` (or determined no block mask is needed / available), so downstream layers do not attempt to build it again.
            **kwargs: additional keyword arguments that may be needed for the attention pair bias (e.g., features for computing attention indices, etc.)

        Returns:
            Q_L: [B, L, C_atom] updated atom features for each atom after applying the attention and transition blocks.
        """
        Q_L = Q_L + self.dropout(
            self.attention_pair_bias(
                Q_L,
                C_L,
                P_LL,
                f=f,
                chunked_pairwise_embedder=chunked_pairwise_embedder,
                initializer_outputs=initializer_outputs,
                flex_block_mask=flex_block_mask,
                flex_block_mask_ready=flex_block_mask_ready,
                **kwargs,
            )
        )
        if exists(C_L):
            Q_L = Q_L + self.transition_block(Q_L, C_L)
        else:
            Q_L = Q_L + self.transition_block(Q_L)
        return Q_L


class CompactStreamingDecoder(nn.Module):
    """
    Streaming decoder architecture that alternates between upcasting to the full atom space, applying a local atom transformer, and then downcasting back to the token space. This allows for efficient decoding while still enabling communication between tokens through the local atom transformer blocks.

    Args:
        c_atom: number of channels in the atom features
        c_atompair: number of channels in the pairwise atom features
        c_token: number of channels in the token features
        c_s: number of channels in the global structure features (if using)
        c_tokenpair: number of channels in the pairwise token features
        atom_transformer_block: dictionary containing the configuration for the atom transformer block used in the local atom transformer
        upcast: dictionary containing the configuration for the upcast module (e.g., number of layers, etc.)
        downcast: dictionary containing the configuration for the downcast module (e.g., number of layers, etc.)
        n_blocks: number of alternating blocks of upcasting, local atom transformer, and downcasting to apply
        diffusion_transformer_block: dictionary containing the configuration for the transformer blocks used in the local atom transformer (e.g., number of attention heads, etc.)
        attn_backend: attention backend to use for the attention pair bias in the transformer blocks ("manual", "megafold", or "flex")
    """

    def __init__(
        self,
        c_atom,
        c_atompair,
        c_token,
        c_s,
        c_tokenpair,
        atom_transformer_block,
        upcast,
        downcast,
        n_blocks,
        diffusion_transformer_block=False,
        attn_backend="manual",
    ):
        super().__init__()
        self.n_blocks = n_blocks

        self.upcast = nn.ModuleList(
            [Upcast(c_atom=c_atom, c_token=c_token, **upcast) for _ in range(n_blocks)]
        )
        self.atom_transformer = nn.ModuleList(
            [
                StructureLocalAtomTransformerBlock(
                    c_atom=c_atom,
                    c_s=c_atom,
                    c_atompair=c_atompair,
                    **atom_transformer_block,
                )
                for _ in range(n_blocks)
            ]
        )
        self.downcast = Downcast(c_atom=c_atom, c_token=c_token, c_s=c_s, **downcast)
        self.use_flex_attention = attn_backend == "flex"

    def forward(
        self,
        A_I,
        S_I,
        Q_L,
        C_L,
        P_LL,
        tok_idx,
        indices,
        index_mask=None,
        f=None,
        chunked_pairwise_embedder=None,
        initializer_outputs=None,
        flex_block_mask=None,
        flex_block_mask_ready=False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Compute the forward pass of the compact streaming decoder.

        Args:
            A_I: [B, L, C_token] atom features for each token
            S_I: [B, L, C_s] global structure features for each token (if using)
            Q_L: [B, L, C_atom] atom features for each atom, used as input to the local atom transformer blocks
            C_L: [B, L, C_s] global structure features for each atom, used as input to the local atom transformer blocks (if using)
            P_LL: [B, L, L, C_atompair] pairwise atom features for each pair of atoms, used as input to the local atom transformer blocks
            tok_idx: [L] token indices for each atom, used for computing attention indices and for the upcast and downcast modules to know which token each atom belongs to
            indices: precomputed attention indices for the local atom transformer blocks, used to determine which atoms attend to which other atoms based on their token indices and the local neighborhood size
            index_mask: optional boolean mask marking real slots in indices
            f: dictionary containing necessary features for computing attention indices and pairwise biases, including:
                - "residue_index": [L] residue indices for each atom, used for computing attention biases based on relative residue positions
                - "asym_id": [L] chain identifiers for each atom, used for computing attention biases based on whether atoms are in the same chain or not
                - "entity_id": [L] entity identifiers for each atom, used for computing attention biases based on whether atoms are in the same entity or not
                - "sym_id": [L] symmetry identifiers for each atom, used for computing attention biases based on relative symmetry positions
                - "unindexing_pair_mask": [L, L] boolean mask indicating which atom pairs should have their attention biases removed (e.g., due to unindexed motifs or being in different examples)
                - "batch_token_lvl": [L] batch indices for each token, used to identify pairs of atoms that belong to different examples
            chunked_pairwise_embedder: optional module for computing pairwise embeddings in a memory-efficient way by chunking (if the full pairwise features cannot fit in memory at once)
            initializer_outputs: optional outputs from the initializer, which may contain precomputed features that can be reused for efficiency (e.g., precomputed pairwise embeddings, etc.)
            flex_block_mask: optional precomputed FlexAttention block_mask (as returned by `create_flex_attention_block_mask` / `torch.nn.attention.flex_attention.create_block_mask`). When provided, it will be forwarded to the flex attention kernel.
            flex_block_mask_ready: set True once the caller has either computed `flex_block_mask` (or determined no block mask is needed / available), so downstream layers do not attempt to build it again.

        Returns:
            A_I: [B, L, C_token] updated atom features for each token after applying the compact streaming decoder blocks.
            Q_L: [B, L, C_atom] updated atom features for each atom after applying the local atom transformer blocks.
            o: dictionary containing any additional outputs or intermediate features that may be useful for loss computation, debugging, etc.
        """
        if f is not None and self.use_flex_attention and not flex_block_mask_ready:
            attn_mask = infer_attention_mask_from_features(
                f, indices, tok_idx=tok_idx, index_mask=index_mask
            )
            flex_block_mask, flex_block_mask_ready = maybe_create_flex_block_mask(
                attn_mask=indices_to_mask_with_padding(indices, attn_mask),
                q_len=Q_L.shape[1],
                device=Q_L.device,
                use_flex_attention=self.use_flex_attention,
                flex_block_mask=flex_block_mask,
                flex_block_mask_ready=flex_block_mask_ready,
            )

        for i in range(self.n_blocks):
            Q_L = self.upcast[i](Q_L, A_I, tok_idx=tok_idx)
            Q_L = self.atom_transformer[i](
                Q_L,
                C_L,
                P_LL,
                indices=indices,
                index_mask=index_mask,
                f=f,
                tok_idx=tok_idx,
                chunked_pairwise_embedder=chunked_pairwise_embedder,
                initializer_outputs=initializer_outputs,
                flex_block_mask=flex_block_mask,
                flex_block_mask_ready=flex_block_mask_ready,
            )

        # Downcast to sequence
        A_I = self.downcast(Q_L.detach(), A_I.detach(), S_I.detach(), tok_idx=tok_idx)

        o = {}
        return A_I, Q_L, o


class Upcast(RFD3Upcast):
    """Upcast modules for when atoms are already reshaped from N_atoms -> (N_tokens, 1/14/22/23)"""

    def __init__(
        self,
        c_token,
        c_atom,
        method="broadcast",
        cross_attention_block=None,
        n_split=6,
        n_atoms_per_tok_max=23,
    ):
        method_ = method
        super().__init__(
            c_token=c_token,
            c_atom=c_atom,
            method=method.replace("residual_", ""),
            cross_attention_block=cross_attention_block,
            n_split=n_split,
        )
        self.n_atoms_per_tok_max = n_atoms_per_tok_max

        if method_ == "broadcast":
            self.project = nn.Sequential(
                RMSNorm((c_token,)), linearNoBias(c_token, c_atom)
            )
        elif method_ == "cross_attention":
            self.gca = GatedCrossAttention(
                c_query=c_atom, c_kv=c_token // self.n_split, **cross_attention_block
            )
        elif method_ == "residual_cross_attention":
            self.gca = ResidualGatedCrossAttention(
                c_query=c_atom, c_kv=c_token // self.n_split, **cross_attention_block
            )
        else:
            raise ValueError(f"Unknown upcast method: {method_}")

    def forward(self, Q_L, A_I, tok_idx, n_atoms_per_tok_max: int | None = None):
        """Forward pass of the upcast module."""
        valid_mask = build_valid_mask(
            # NOTE: This implementation of `build_valid_mask()` is compatible with `torch.compile()`
            tok_idx,
            n_atoms_per_tok_max=n_atoms_per_tok_max
            if n_atoms_per_tok_max is not None
            else self.n_atoms_per_tok_max,
        )
        Q_IA = ungroup_atoms(Q_L, valid_mask)
        Q_IA = self.forward_(Q_IA, A_I, valid_mask)
        Q_L = group_atoms(Q_IA, valid_mask)
        return Q_L


class Downcast(RFD3Downcast):
    """Downcast modules for when atoms are already reshaped from N_atoms -> (N_tokens, 1/14/22/23)"""

    def __init__(
        self,
        c_atom,
        c_token,
        c_s=None,
        method="mean",
        cross_attention_block=None,
        n_atoms_per_tok_max=23,
    ):
        method_ = method
        super().__init__(
            c_atom=c_atom,
            c_token=c_token,
            c_s=c_s,
            method=method.replace("residual_", ""),
            cross_attention_block=cross_attention_block,
        )
        self.n_atoms_per_tok_max = n_atoms_per_tok_max

        if method_ == "mean":
            self.project = linearNoBias(c_atom, c_token)
        elif method_ == "cross_attention":
            self.gca = GatedCrossAttention(
                c_query=c_token,
                c_kv=c_atom,
                **cross_attention_block,
            )
        elif method_ == "residual_cross_attention":
            self.gca = ResidualGatedCrossAttention(
                c_query=c_token,
                c_kv=c_atom,
                **cross_attention_block,
            )
        else:
            raise ValueError(f"Unknown downcast method: {method_}")

    def forward(
        self, Q_L, A_I, S_I=None, tok_idx=None, n_atoms_per_tok_max: int | None = None
    ):
        """Forward pass of the downcast module."""
        valid_mask = build_valid_mask(
            # NOTE: This implementation of `build_valid_mask()` is compatible with `torch.compile()`
            tok_idx,
            n_atoms_per_tok_max=n_atoms_per_tok_max
            if n_atoms_per_tok_max is not None
            else self.n_atoms_per_tok_max,
        )
        if Q_L.ndim == 2:
            squeeze = True
            Q_L = Q_L.unsqueeze(0)
        else:
            squeeze = False

        A_I = A_I.unsqueeze(0) if exists(A_I) and A_I.ndim == 2 else A_I
        S_I = S_I.unsqueeze(0) if exists(S_I) and S_I.ndim == 2 else S_I

        Q_IA = ungroup_atoms(Q_L, valid_mask)

        A_I = self.forward_(Q_IA, A_I, S_I, valid_mask=valid_mask)

        if squeeze:
            A_I = A_I.squeeze(0)
        return A_I


class LinearEmbedWithPool(RFD3LinearEmbedWithPool):
    """Linear embedding layer that pools atom features into token features using scatter mean."""

    def __init__(self, c_token):
        super().__init__(c_token=c_token)

    def forward(self, R_L, tok_idx, num_tok):
        """Forward pass of the linear embedding layer with pooling."""
        B = R_L.shape[0]
        I = num_tok  # NOTE: This implementation supports `torch.compile()`
        A_I_shape = (
            B,
            I,
            self.c_token,
        )
        Q_L = self.linear(R_L)
        A_I = scatter_mean(
            torch.zeros(A_I_shape, device=R_L.device, dtype=Q_L.dtype),
            -2,
            tok_idx.long(),
            Q_L,
        )
        return A_I


##################################################################################
# Zatom blocks
##################################################################################


class OneDFeatureBottleneckEmbedder(nn.Module):
    """
    Embed 1D features into a single vector using a bottleneck structure.

    Inspired by Algorithms 10 and 12 in Emyx.
    Reference: https://arxiv.org/abs/2606.19377v1.

    Args:
        features (dict): Dictionary of feature names and their number of channels.
        bottleneck_channels (int): Number of channels in the bottleneck layer.
        output_channels (int): Output dimension of the projected embedding.
        use_allscaip_charge_spin_embeddings (bool): Replace the scalar linear
            projections for ``total_charge`` and ``spin_multiplicity`` with the
            trainable lookup embeddings used by AllScAIP/UMA.
        row_count_key (str | None): Feature whose length defines the number of
            rows when every configured feature is absent. Token and atom
            embedders should use ``batch_token_lvl`` and ``batch_atom_lvl``,
            respectively.
        residual_features (list[str] | tuple[str, ...] | None): Numeric features
            to exclude from the shared bottleneck and project jointly through a
            dedicated linear embedding that is added to the bottleneck output.
    """

    def __init__(
        self,
        features,
        bottleneck_channels,
        output_channels,
        use_allscaip_charge_spin_embeddings=False,
        row_count_key=None,
        residual_features=None,
    ):
        super().__init__()
        self.features = {k: v for k, v in features.items() if exists(v)}
        self.bottleneck_channels = bottleneck_channels
        self.output_channels = output_channels
        self.use_allscaip_charge_spin_embeddings = use_allscaip_charge_spin_embeddings
        self.row_count_key = row_count_key

        residual_feature_names = tuple(dict.fromkeys(residual_features or ()))
        unknown_residual_features = set(residual_feature_names) - set(self.features)
        if unknown_residual_features:
            raise ValueError(
                "residual_features contains features that are not configured: "
                f"{sorted(unknown_residual_features)}"
            )
        self.residual_features = {
            feature: self.features[feature] for feature in residual_feature_names
        }
        self.bottleneck_features = {
            feature: n_channels
            for feature, n_channels in self.features.items()
            if feature not in self.residual_features
        }

        self.embedders = nn.ModuleDict(
            {
                feature: self._make_embedder(
                    feature,
                    n_channels,
                    output_channels,
                )
                for feature, n_channels in self.bottleneck_features.items()
            }
        )
        self.bottleneck_mlp = (
            nn.Sequential(
                # Each feature is already projected to output_channels, so apply
                # an initial nonlinearity before the bottleneck layer.
                nn.SiLU(),
                linearNoBias(
                    len(self.bottleneck_features) * output_channels,
                    bottleneck_channels,
                ),
                nn.SiLU(),
                linearNoBias(bottleneck_channels, output_channels),
            )
            if self.bottleneck_features
            else None
        )
        self.residual_embedder = (
            linearNoBias(sum(self.residual_features.values()), output_channels)
            if self.residual_features
            else None
        )

    def _make_embedder(self, feature, n_channels, output_channels):
        """Create the feature-specific input projection."""
        embedding_target = None
        if self.use_allscaip_charge_spin_embeddings:
            embedding_target = {
                "total_charge": "charge",
                "spin_multiplicity": "spin",
            }.get(feature)
        if embedding_target is not None:
            if n_channels != 1:
                raise ValueError(f"{feature} must have exactly one channel.")
            return ChgSpinEmbedding(
                "rand_emb",
                embedding_target,
                output_channels,
                grad=True,
            )
        return linearNoBias(n_channels, output_channels)

    def _num_rows(self, f):
        """Infer the number of feature rows for optional charge/spin defaults."""
        for feature, n_channels in self.features.items():
            value = f.get(feature)
            if value is not None:
                return value.numel() // n_channels
        if self.row_count_key is not None and self.row_count_key in f:
            return f[self.row_count_key].numel()
        raise ValueError(
            "Cannot infer feature count: no configured feature is present and "
            f"row_count_key={self.row_count_key!r} is unavailable."
        )

    def _embed_feature(self, f, feature, n_channels):
        """Embed one feature, applying UMA's missing metadata defaults."""
        embedder = self.embedders[feature]
        value = f.get(feature)
        if isinstance(embedder, ChgSpinEmbedding):
            if value is None:
                value = embedder.rand_emb.weight.new_zeros(
                    self._num_rows(f),
                    dtype=torch.long,
                )
            return embedder(value.reshape(-1))
        return embedder(value.float().reshape(-1, n_channels))

    def forward(self, f):
        """
        Forward pass of the 1D feature embedder.

        Args:
            f (dict): Dictionary of feature tensors.

        Returns:
            Tensor: Embedded features of shape [B, L, output_channels].
        """
        output = None
        if self.bottleneck_features:
            embeddings = torch.cat(
                [
                    self._embed_feature(f, feature, n_channels)
                    for feature, n_channels in self.bottleneck_features.items()
                ],
                dim=-1,
            )
            output = self.bottleneck_mlp(embeddings)

        if self.residual_features:
            residual_inputs = torch.cat(
                [
                    f[feature].float().reshape(-1, n_channels)
                    for feature, n_channels in self.residual_features.items()
                ],
                dim=-1,
            )
            residual_embedding = self.residual_embedder(residual_inputs)
            output = (
                residual_embedding if output is None else output + residual_embedding
            )

        if output is None:
            raise ValueError("At least one 1D feature must be configured.")
        return output


class TwoDFeatureBottleneckEmbedder(nn.Module):
    """
    Embed 2D features into a single vector using a bottleneck structure.

    Inspired by Algorithms 10 and 12 in Emyx.
    Reference: https://arxiv.org/abs/2606.19377v1.

    Args:
        features (dict): Dictionary of feature names and their number of channels.
        bottleneck_channels (int): Number of channels in the bottleneck layer.
        output_channels (int): Output dimension of the projected embedding.
    """

    def __init__(self, features, bottleneck_channels, output_channels):
        super().__init__()
        self.features = {k: v for k, v in features.items() if exists(v)}
        self.bottleneck_channels = bottleneck_channels
        self.output_channels = output_channels

        self.embedders = nn.ModuleDict(
            {
                feature: linearNoBias(n_channels, output_channels)
                for feature, n_channels in self.features.items()
            }
        )
        self.bottleneck_mlp = nn.Sequential(
            # NOTE: Each feature is already embedded with a linear layer,
            # so we apply an initial nonlinearity before the bottleneck layer.
            nn.SiLU(),
            linearNoBias(len(self.features) * output_channels, bottleneck_channels),
            nn.SiLU(),
            linearNoBias(bottleneck_channels, output_channels),
        )

    def forward(self, f, L):
        """
        Forward pass of the 2D feature embedder.

        Args:
            f (dict): Dictionary of feature tensors.
            L (int): Number of tokens (or atoms) in the sequence,
                used to reshape the 2D features into [L, L, n_channels]
                before embedding.

        Returns:
            Tensor: Embedded features of shape [L, L, output_channels].
        """
        embeddings = torch.cat(
            [
                self.embedders[feature](f[feature].float().reshape(L, L, n_channels))
                for feature, n_channels in self.features.items()
                if exists(n_channels)
            ],
            dim=-1,
        )
        return self.bottleneck_mlp(embeddings)
