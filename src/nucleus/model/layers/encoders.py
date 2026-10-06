import functools

import torch
import torch.nn as nn
from foundry.common import exists
from foundry.training.checkpoint import activation_checkpointing

from nucleus import DISABLE_CHECKPOINTING
from nucleus.model.layers.blocks import (
    Downcast,
    LocalAtomTransformer,
    OneDFeatureBottleneckEmbedder,
    RelativePositionEncodingWithIndexRemoval,
    TwoDFeatureBottleneckEmbedder,
)
from nucleus.model.layers.layer_utils import Transition
from nucleus.model.layers.pairformer_layers import (
    PairformerBlock,
)
from nucleus.utils.attention import maybe_create_flex_block_mask
from nucleus.utils.distogram import bucketize_scaled_distogram

try:
    from rfd3.model.layers.block_utils import (
        pairwise_mean_pool,
    )
    from rfd3.model.layers.blocks import (
        OneDFeatureEmbedder,
        PositionPairDistEmbedder,
        SinusoidalDistEmbed,
    )
    from rfd3.model.layers.chunked_pairwise import (
        ChunkedPairwiseEmbedder,
        ChunkedPositionPairDistEmbedder,
        ChunkedSinusoidalDistEmbed,
    )
    from rfd3.model.layers.layer_utils import (
        RMSNorm,
        linearNoBias,
    )
    from rfd3na.model.layers.blocks import TwoDFeatureEmbedder
except ImportError:
    raise ImportError(
        "Could not import RFD3/RFD3NA modules. Please ensure that the `rc-foundry[all]` package is installed and up to date."
    )


def _token_pair_mask(f, *, device: torch.device) -> torch.Tensor:
    """Shared token-level attention mask for token init and diffusion token encoding."""
    same_example_mask = f["batch_token_lvl"].to(device=device).unsqueeze(-1) == f[
        "batch_token_lvl"
    ].to(device=device).unsqueeze(-2)

    if "unindexing_pair_mask" not in f:
        return same_example_mask

    return ~f["unindexing_pair_mask"].to(device=device, dtype=torch.bool)


def _embed_optional_token_bonds(
    process_token_bonds: nn.Module,
    token_bonds: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Embed optional token bonds while keeping the projection in the graph."""
    if token_bonds is None:
        token_bonds = torch.zeros(
            (*reference.shape[:-1], 1),
            dtype=torch.float32,
            device=reference.device,
        )
    else:
        token_bonds = token_bonds.unsqueeze(-1).float()
    return process_token_bonds(token_bonds)


class TokenInitializer(nn.Module):
    """
    Token embedding module.

    Args:
        c_s: Channel dimension for token features
        c_z: Channel dimension for pairwise token features
        c_atom: Channel dimension for atom features
        c_atompair: Channel dimension for pairwise atom features
        relative_position_encoding: Config for relative position encoding module
        n_pairformer_blocks: Number of pairformer blocks to run in the initializer
        pairformer_block: Config for pairformer blocks
        downcast: Config for downcasting module to go from atom features to token features
        token_1d_features: List of 1D token features to embed and provide as input to the model trunk
        atom_1d_features: List of 1D atom features to embed and provide as input to the model trunk
        atom_transformer: Config for optional local transformer block to mix atom features with pairwise features in the initializer
        token_2d_features: Optional list of 2D token features to embed and provide as input to the model trunk
        use_chunked_pll: Whether to use chunked processing for the P_LL computation, which can reduce memory usage at inference time with large numbers of atoms.
        use_pll_only: Whether to only compute P_LL and not project it to Z_II. This can be useful if you want to provide P_LL directly to the model trunk without mixing it into Z_II, which can reduce memory usage and computation time.
        attn_backend: Must be one of ("manual", "megafold", "flex").
    """

    def __init__(
        self,
        c_s,
        c_z,
        c_atom,
        c_atompair,
        relative_position_encoding,
        n_pairformer_blocks,
        pairformer_block,
        downcast,
        token_1d_features,
        atom_1d_features,
        atom_transformer,
        token_2d_features=None,
        use_chunked_pll=False,  # Parameter for memory optimization
        use_pll_only=False,  # New parameter for customizing focus of pairwise representation
        attn_backend="manual",
    ):
        super().__init__()

        # Store chunked mode flag
        self.use_chunked_pll = use_chunked_pll
        self.use_pll_only = use_pll_only

        # Features
        self.atom_1d_embedder_1 = OneDFeatureEmbedder(atom_1d_features, c_s)
        self.atom_1d_embedder_2 = OneDFeatureEmbedder(atom_1d_features, c_atom)
        self.token_1d_embedder = OneDFeatureEmbedder(token_1d_features, c_s)
        if exists(token_2d_features):
            self.token_2d_embedder = TwoDFeatureEmbedder(token_2d_features, c_z)
        else:
            self.token_2d_embedder = None

        self.downcast_atom = Downcast(c_atom=c_s, c_token=c_s, c_s=None, **downcast)
        self.transition_post_token = Transition(c=c_s, n=2)
        self.transition_post_atom = Transition(c=c_s, n=2)
        self.process_s_init = nn.Sequential(
            RMSNorm(c_s),
            linearNoBias(c_s, c_s),
        )

        # Operations to mix into Z_II and S_I
        self.to_z_init_i = linearNoBias(c_s, c_z)
        self.to_z_init_j = linearNoBias(c_s, c_z)
        self.relative_position_encoding = RelativePositionEncodingWithIndexRemoval(
            c_z=c_z, **relative_position_encoding
        )
        self.relative_position_encoding2 = RelativePositionEncodingWithIndexRemoval(
            c_z=c_z, **relative_position_encoding
        )
        self.process_token_bonds = linearNoBias(1, c_z)

        # Processing of Z_init
        self.process_z_init = nn.Sequential(
            RMSNorm(c_z * 2),
            linearNoBias(c_z * 2, c_z),
        )
        self.transition_1 = nn.ModuleList(
            [
                Transition(c=c_z, n=2),
                Transition(c=c_z, n=2),
            ]
        )
        self.ref_pos_embedder_tok = PositionPairDistEmbedder(c_z, embed_frame=False)

        # Pairformer without triangle updates
        if n_pairformer_blocks > 0:
            self.transformer_stack = nn.ModuleList(
                [
                    PairformerBlock(c_s=c_s, c_z=c_z, **pairformer_block)
                    for _ in range(n_pairformer_blocks)
                ]
            )
        else:
            self.transformer_stack = None

        #############################################################################
        # Token track processing
        self.process_s_trunk = nn.Sequential(RMSNorm(c_s), linearNoBias(c_s, c_atom))
        self.process_single_l = nn.Sequential(
            nn.ReLU(), linearNoBias(c_atom, c_atompair)
        )
        self.process_single_m = nn.Sequential(
            nn.ReLU(), linearNoBias(c_atom, c_atompair)
        )
        self.process_z = nn.Sequential(RMSNorm(c_z), linearNoBias(c_z, c_atompair))

        # ALWAYS create these MLPs - they will be shared between chunked and standard modes
        self.motif_pos_embedder = SinusoidalDistEmbed(c_atompair=c_atompair)
        self.ref_pos_embedder = PositionPairDistEmbedder(c_atompair, embed_frame=False)
        self.pair_mlp = nn.Sequential(
            nn.ReLU(),
            linearNoBias(c_atompair, c_atompair),
            nn.ReLU(),
            linearNoBias(c_atompair, c_atompair),
            nn.ReLU(),
            linearNoBias(c_atompair, c_atompair),
        )

        # Atom pair feature processing
        if self.use_chunked_pll:
            # Share trained components so chunked inference uses same weights as full training.
            motif_pos_embedder = ChunkedSinusoidalDistEmbed(
                embedder_instance=self.motif_pos_embedder
            )
            ref_pos_embedder = ChunkedPositionPairDistEmbedder(
                embedder_instance=self.ref_pos_embedder
            )
            self.chunked_pairwise_embedder = ChunkedPairwiseEmbedder(
                c_atompair=c_atompair,
                motif_pos_embedder=motif_pos_embedder,
                ref_pos_embedder=ref_pos_embedder,
                process_single_l=self.process_single_l,
                process_single_m=self.process_single_m,
                process_z=self.process_z,
                pair_mlp=self.pair_mlp,
            )
        if not self.use_pll_only:
            self.process_pll = linearNoBias(c_atompair, c_atompair)
            self.project_pll = linearNoBias(c_atompair, c_z)

        if atom_transformer["n_blocks"] > 0:
            raise NotImplementedError(
                "Atom transformer is currently not compatible with sparse input batches."
            )
            self.atom_transformer = LocalAtomTransformer(
                c_atom=c_atom, c_s=None, c_atompair=c_atompair, **atom_transformer
            )
        else:
            self.atom_transformer = None

        self.use_flex_attention = attn_backend == "flex"
        self.use_checkpointing = not DISABLE_CHECKPOINTING

    def forward(self, f) -> dict:
        """
        Provide an initial representation for atom and token representations.

        Args:
            f: Dict containing input features. Expected to contain at least:
                - atom_to_token_map: Long tensor of shape (L,) mapping each atom to its corresponding token index
                - restype: Long tensor of shape (I,) containing the residue type for each token
                - ref_atom_name_chars: Long tensor of shape (L, max_atom_name_length) containing character-level encoding of atom names for each atom
                - token_bonds: Optional long tensor of shape (I, I) containing number of bonds between tokens (0 if not in same molecule, 1 if covalently bonded, 2 if in same residue but not covalently bonded)
                - ref_space_uid: Long tensor of shape (L,) containing unique identifiers for groups of atoms that share reference coordinates (e.g. all atoms in the same residue will have the same ref_space_uid)
                - is_ca: Bool tensor of shape (L,) indicating which atoms are C-alpha atoms, which are used for embedding reference coordinate information at the token level
                - batch_token_lvl: Long tensor of shape (I,) indicating batch indices for each token (used to avoid mixing information between examples in the batch)
                - batch_atom_lvl: Long tensor of shape (L,) indicating batch indices for each atom (used to avoid mixing information between examples in the batch)
                - is_motif_atom_with_fixed_coord: Bool tensor of shape (L,) indicating which atoms have fixed coordinates that should be embedded into P_LL via motif_pos_embedder
                - is_motif_atom_with_fixed_seq: Bool tensor of shape (L,) indicating which atoms have fixed sequence that should be considered when applying ref_pos_embedder to P_LL

        Returns:
            Dict containing:
                - Q_L_init: Float tensor of shape (L, c_atom) containing initial atom features for each atom
                - C_L: Float tensor of shape (L, c_atom) containing processed atom features for each atom, mixed with token features
                - chunked_pairwise_embedder: If use_chunked_pll is True, this will contain the initialized ChunkedPairwiseEmbedder instance with pre-cached static projections. This should be passed to the forward_chunked method at inference time to compute P_LL in a memory-efficient manner. If use_chunked_pll is False, this will be None.
                - P_LL: Float tensor of shape (L, L, c_atompair) containing pairwise atom features for each atom pair (only computed if use_chunked_pll is False, otherwise will be None)
                - S_I: Float tensor of shape (I, c_s) containing token features after mixing with atom features via P_LL (only computed if use_chunked_pll is False, otherwise will be None)
                - Z_II: Float tensor of shape (I, I, c_z) containing pairwise token features after mixing with atom features via P_LL (only computed if use_chunked_pll is False, otherwise will be None)
                - token_flex_block_mask: Optional cached token-level FlexAttention block mask for downstream token encoders.
                - token_flex_block_mask_ready: Whether the token-level FlexAttention block mask is already cached or intentionally unavailable.
        """
        tok_idx = f["atom_to_token_map"]
        L = len(tok_idx)
        f["ref_atom_name_chars"] = f["ref_atom_name_chars"].reshape(L, -1)
        I = len(f["restype"])

        def init_tokens():
            token_pair_mask = _token_pair_mask(f, device=tok_idx.device)

            # Embed token features
            S_I = self.token_1d_embedder(f, I)
            S_I = S_I + self.transition_post_token(S_I)

            # Embed atom features and downcast to token features
            S_I = self.downcast_atom(
                Q_L=self.atom_1d_embedder_1(f, L), A_I=S_I, tok_idx=tok_idx
            )
            S_I = S_I + self.transition_post_atom(S_I)
            S_I = self.process_s_init(S_I)

            # Embed Z_II
            Z_init_II = self.to_z_init_i(S_I).unsqueeze(-3) + self.to_z_init_j(
                S_I
            ).unsqueeze(-2)
            Z_init_II = Z_init_II + self.relative_position_encoding(f)
            Z_init_II = Z_init_II + _embed_optional_token_bonds(
                self.process_token_bonds,
                f.get("token_bonds"),
                Z_init_II,
            )

            # Embed reference coordinates of ligands
            token_id = f["ref_space_uid"][f["is_ca"]]
            valid_mask = (token_id.unsqueeze(-1) == token_id.unsqueeze(-2)) & (
                token_pair_mask
            )
            Z_init_II = Z_init_II + self.ref_pos_embedder_tok(
                f["ref_pos"][f["is_ca"]], valid_mask.unsqueeze(-1)
            )
            if exists(self.token_2d_embedder):
                Z_init_II = Z_init_II + self.token_2d_embedder(f, I)

            # Maybe run a small transformer to provide position encodings to single.
            flex_block_mask = None
            flex_block_mask_ready = False
            if self.use_flex_attention:
                flex_block_mask, flex_block_mask_ready = maybe_create_flex_block_mask(
                    token_pair_mask,
                    S_I.shape[-2],
                    S_I.device,
                    use_flex_attention=self.use_flex_attention,
                )
            if exists(self.transformer_stack):
                for block in self.transformer_stack:
                    S_I, Z_init_II = block(
                        S_I,
                        Z_init_II,
                        A_mask=token_pair_mask,
                        flex_block_mask=flex_block_mask,
                        flex_block_mask_ready=flex_block_mask_ready,
                    )

            # Also cat the relative position encoding and mix
            Z_init_II = torch.cat(
                [
                    Z_init_II,
                    self.relative_position_encoding2(f),
                ],
                dim=-1,
            )
            Z_init_II = self.process_z_init(Z_init_II)
            for b in range(2):
                Z_init_II = Z_init_II + self.transition_1[b](Z_init_II)

            # Mask out invalid positions in Z_init_II to avoid providing any information about them to the model
            Z_init_II = Z_init_II.masked_fill(~token_pair_mask.unsqueeze(-1), 0.0)

            return {
                "S_init_I": S_I,
                "Z_init_II": Z_init_II,
                "token_flex_block_mask": flex_block_mask,
                "token_flex_block_mask_ready": flex_block_mask_ready,
            }

        def init_atoms_impl(
            S_init_I,
            Z_init_II,
            token_flex_block_mask=None,
            token_flex_block_mask_ready=False,
        ):
            Q_L_init = self.atom_1d_embedder_2(f, L)
            C_L = Q_L_init + self.process_s_trunk(S_init_I)[..., tok_idx, :]

            if self.use_chunked_pll:
                # Precompute static MLP projections once so forward_chunked can
                # skip those MLP calls at every subsequent diffusion step.
                self.chunked_pairwise_embedder.cache_static_projections(C_L, Z_init_II)
                return {
                    "Q_L_init": Q_L_init,
                    "C_L": C_L,
                    "chunked_pairwise_embedder": self.chunked_pairwise_embedder,
                    "S_I": S_init_I,
                    "Z_II": Z_init_II,
                    "token_flex_block_mask": token_flex_block_mask,
                    "token_flex_block_mask_ready": token_flex_block_mask_ready,
                }
            else:
                # Original full P_LL computation
                ##################################################################################
                # Embed motif coordinates
                same_example_mask = f["batch_atom_lvl"].unsqueeze(-1) == f[
                    "batch_atom_lvl"
                ].unsqueeze(-2)
                valid_mask = same_example_mask.unsqueeze(-1) & (
                    f["is_motif_atom_with_fixed_coord"].unsqueeze(-1)
                    & f["is_motif_atom_with_fixed_coord"].unsqueeze(-2)
                ).unsqueeze(-1)
                P_LL = self.motif_pos_embedder(
                    f["motif_pos"], valid_mask
                )  # (L, L, c_atompair)

                # Embed ref pos
                atoms_in_same_token = same_example_mask.unsqueeze(-1) & (
                    f["ref_space_uid"].unsqueeze(-1) == f["ref_space_uid"].unsqueeze(-2)
                ).unsqueeze(-1)
                # Only consider ref_pos for atoms given seq (otherwise ref_pos is 0, doesn't make sense to compute)
                atoms_has_seq = same_example_mask.unsqueeze(-1) & (
                    f["is_motif_atom_with_fixed_seq"].unsqueeze(-1)
                    & f["is_motif_atom_with_fixed_seq"].unsqueeze(-2)
                ).unsqueeze(-1)
                valid_mask = atoms_in_same_token & atoms_has_seq
                P_LL = P_LL + self.ref_pos_embedder(f["ref_pos"], valid_mask)

                ##################################################################################

                P_LL = P_LL + (
                    self.process_single_l(C_L).unsqueeze(-2)
                    + self.process_single_m(C_L).unsqueeze(-3)
                )
                P_LL = (
                    P_LL
                    + self.process_z(Z_init_II)[..., tok_idx, :, :][..., tok_idx, :]
                )
                P_LL = P_LL + self.pair_mlp(P_LL)
                P_LL = P_LL.contiguous()

                # Mask out invalid positions in P_LL to avoid providing any information about them to the model
                P_LL = P_LL.masked_fill(~same_example_mask.unsqueeze(-1), 0.0)

                if not self.use_pll_only:
                    # Pool P_LL to token level to provide atom-level resolution for token track
                    pooled_atom_level_features = pairwise_mean_pool(
                        pairwise_atom_features=self.process_pll(P_LL).unsqueeze(0),
                        atom_to_token_map=tok_idx,
                        I=I,
                        dtype=P_LL.dtype,
                    ).squeeze(0)
                    Z_init_II = Z_init_II + self.project_pll(pooled_atom_level_features)

                # Mix atom conditioning features via sequence-local attention
                if exists(self.atom_transformer):
                    C_L = self.atom_transformer(
                        C_L.unsqueeze(0), None, P_LL, indices=None, f=f, X_L=None
                    ).squeeze(0)

                return {
                    "Q_L_init": Q_L_init,
                    "C_L": C_L,
                    "P_LL": P_LL,
                    "S_I": S_init_I,
                    "Z_II": Z_init_II,
                    "token_flex_block_mask": token_flex_block_mask,
                    "token_flex_block_mask_ready": token_flex_block_mask_ready,
                }

        init_atoms = (
            activation_checkpointing(init_atoms_impl)
            if self.use_checkpointing
            else init_atoms_impl
        )

        tokens = init_tokens()
        return init_atoms(**tokens)


class ZatomInitializer(nn.Module):
    """
    Atom and token embedding module for Zatom.

    A diffusion geometry/topology/time-agnostic flavor of Algorithm 3 in Emyx.
    Reference: https://arxiv.org/abs/2606.19377v1.

    Args:
        c_s: Channel dimension for token features
        c_z: Channel dimension for pairwise token features
        c_atom: Channel dimension for atom features
        c_bottleneck: Channel dimension for feature embedding bottleneck
        n_token_head: Number of attention heads for token features
        n_atom_head: Number of attention heads for atom features
        relative_position_encoding: Config for relative position encoding module
        token_1d_features: List of 1D token features to embed and provide as input to the model trunk
        atom_1d_features: List of 1D atom features to embed and provide as input to the model trunk
        token_2d_features: Optional list of 2D token features to embed and provide as input to the model trunk
        use_allscaip_charge_spin_embeddings: Whether to use AllScAIP/UMA lookup
            embeddings for charge and spin. If false, use the legacy linear
            feature projections.
        force_conditioning_residual_embedding: Whether force-statistic and
            presence features bypass the shared token-feature bottleneck through
            a dedicated projection added residually to its output.
    """

    def __init__(
        self,
        c_s,
        c_z,
        c_atom,
        c_bottleneck,
        n_token_head,
        n_atom_head,
        relative_position_encoding,
        token_1d_features,
        atom_1d_features,
        token_2d_features=None,
        use_allscaip_charge_spin_embeddings=True,
        force_conditioning_residual_embedding=False,
        **kwargs,
    ):
        super().__init__()

        # Single token features
        force_conditioning_features = (
            "mean_force_norm",
            "has_mean_force_norm",
            "force_rms_p95",
            "has_force_rms_p95",
        )
        residual_features = (
            tuple(
                feature
                for feature in force_conditioning_features
                if exists(token_1d_features.get(feature))
            )
            if force_conditioning_residual_embedding
            else None
        )
        if force_conditioning_residual_embedding and not residual_features:
            raise ValueError(
                "force_conditioning_residual_embedding=True requires configured "
                "force-conditioning token features."
            )
        self.token_1d_bottleneck_embedder = OneDFeatureBottleneckEmbedder(
            token_1d_features,
            c_bottleneck,
            c_s,
            use_allscaip_charge_spin_embeddings=use_allscaip_charge_spin_embeddings,
            row_count_key="batch_token_lvl",
            residual_features=residual_features,
        )
        self.process_s = RMSNorm(c_s)

        ## Pairwise token features
        self.relative_position_encoding = RelativePositionEncodingWithIndexRemoval(
            c_z=c_z, **relative_position_encoding
        )
        self.process_token_bonds = linearNoBias(1, c_z)
        self.ref_pos_embedder_tok = PositionPairDistEmbedder(c_z, embed_frame=False)
        self.token_2d_embedder = (
            TwoDFeatureBottleneckEmbedder(token_2d_features, c_bottleneck, c_z)
            if exists(token_2d_features)
            else None
        )
        self.process_z = linearNoBias(c_z, n_token_head)

        # Single atom features
        self.atom_1d_embedder = OneDFeatureBottleneckEmbedder(
            atom_1d_features,
            c_bottleneck,
            c_atom,
            row_count_key="batch_atom_lvl",
        )
        self.process_c_init = linearNoBias(c_atom, c_atom)
        self.process_c_from_s = linearNoBias(c_s, c_atom)
        self.process_c = RMSNorm(c_atom)

        # Pairwise atom features
        self.motif_pos_embedder = SinusoidalDistEmbed(c_atompair=n_token_head)
        self.ref_pos_embedder = PositionPairDistEmbedder(
            c_atompair=n_token_head, embed_frame=False
        )
        self.process_p = linearNoBias(n_token_head, n_atom_head)

    def forward(self, f) -> dict:
        """
        Provide an initial representation for atoms and tokens.

        Args:
            f: Dict containing input features. Expected to contain at least:
                - atom_to_token_map: Long tensor of shape (L,) mapping each atom to its corresponding token index
                - restype: Long tensor of shape (I,) containing the residue type for each token
                - ref_atom_name_chars: Long tensor of shape (L, max_atom_name_length) containing character-level encoding of atom names for each atom
                - token_bonds: Optional long tensor of shape (I, I) containing number of bonds between tokens (0 if not in same molecule, 1 if covalently bonded, 2 if in same residue but not covalently bonded)
                - ref_space_uid: Long tensor of shape (L,) containing unique identifiers for groups of atoms that share reference coordinates (e.g. all atoms in the same residue will have the same ref_space_uid)
                - is_ca: Bool tensor of shape (L,) indicating which atoms are C-alpha atoms, which are used for embedding reference coordinate information at the token level
                - batch_token_lvl: Long tensor of shape (I,) indicating batch indices for each token (used to avoid mixing information between examples in the batch)
                - batch_atom_lvl: Long tensor of shape (L,) indicating batch indices for each atom (used to avoid mixing information between examples in the batch)
                - is_motif_atom_with_fixed_coord: Bool tensor of shape (L,) indicating which atoms have fixed coordinates that should be embedded into P_LL via motif_pos_embedder
                - is_motif_atom_with_fixed_seq: Bool tensor of shape (L,) indicating which atoms have fixed sequence that should be considered when applying ref_pos_embedder to P_LL

        Returns:
            Dict containing:
                - S_I: Float tensor of shape (I, c_s) containing processed single token features
                - Z_II: Float tensor of shape (I, I, c_z) containing processed pairwise token features
                - C_L: Float tensor of shape (L, c_atom) containing processed single atom features
                - P_LL: Float tensor of shape (L, L, c_atompair) containing processed pairwise atom features
        """
        # Initialize metadata
        I = len(f["restype"])  # I = number of tokens; L = number of atoms
        tok_idx = f["atom_to_token_map"]
        token_id = f["ref_space_uid"][f["is_ca"]]

        # ... Create masks, caching the pair mask for use in downstream modules
        tokens_in_same_example_mask = f["token_pair_mask"] = _token_pair_mask(
            f, device=tok_idx.device
        )
        tokens_in_same_token_mask = (
            token_id.unsqueeze(-1) == token_id.unsqueeze(-2)
        ) & (tokens_in_same_example_mask)

        atoms_in_same_example_mask = f["batch_atom_lvl"].unsqueeze(-1) == f[
            "batch_atom_lvl"
        ].unsqueeze(-2)
        atoms_have_struct_mask = atoms_in_same_example_mask.unsqueeze(-1) & (
            # Only consider motif_pos for atoms with fixed coordinates
            # (Otherwise motif_pos is 0, which doesn't make sense to compute)
            f["is_motif_atom_with_fixed_coord"].unsqueeze(-1)
            & f["is_motif_atom_with_fixed_coord"].unsqueeze(-2)
        ).unsqueeze(-1)

        atoms_in_same_token_mask = atoms_in_same_example_mask.unsqueeze(-1) & (
            f["ref_space_uid"].unsqueeze(-1) == f["ref_space_uid"].unsqueeze(-2)
        ).unsqueeze(-1)
        atoms_have_seq_mask = atoms_in_same_example_mask.unsqueeze(-1) & (
            # Only consider ref_pos for atoms given sequence
            # (otherwise ref_pos is 0, which doesn't make sense to compute)
            f["is_motif_atom_with_fixed_seq"].unsqueeze(-1)
            & f["is_motif_atom_with_fixed_seq"].unsqueeze(-2)
        ).unsqueeze(-1)
        atoms_in_same_token_and_have_seq_mask = (
            atoms_in_same_token_mask & atoms_have_seq_mask
        )

        def init_tokens_impl():
            """Embed token features."""
            # ... Embed single token features
            S_I_init = self.token_1d_bottleneck_embedder(f)
            S_I = self.process_s(S_I_init)  # (I, c_s), processed single token features

            # ... Embed pairwise token features
            Z_II_rpe = self.relative_position_encoding(f)
            Z_II_bonds = _embed_optional_token_bonds(
                self.process_token_bonds,
                f.get("token_bonds"),
                Z_II_rpe,
            )
            Z_II_ref_pos = self.ref_pos_embedder_tok(
                # Embed reference coordinates of ligands
                f["ref_pos"][f["is_ca"]],
                tokens_in_same_token_mask.unsqueeze(-1),
            )
            Z_II_list = [Z_II_rpe, Z_II_bonds, Z_II_ref_pos]

            if exists(self.token_2d_embedder):
                # ... Embed additional 2D token features if provided
                Z_II_2d = self.token_2d_embedder(f, I)
                Z_II_list.append(Z_II_2d)

            Z_II = sum(Z_II_list) / len(
                Z_II_list
            )  # (I, I, c_z), average processed pairwise token features

            Z_II = self.process_z(
                Z_II
            )  # (I, I, n_token_head), processed pairwise per-token-head token features

            Z_II = Z_II.masked_fill(
                # Mask out inter-batch positions in Z_II to avoid
                # providing any information about them to the model
                ~tokens_in_same_example_mask.unsqueeze(-1),
                0.0,
            )

            return {"S_I": S_I, "Z_II": Z_II}

        def init_atoms_impl(S_I, Z_II, **kwargs):
            """Embed atom features."""
            # ... Embed single atom features
            C_L_init = self.atom_1d_embedder(f)
            C_L_init = self.process_c_init(C_L_init)
            C_L_from_S = self.process_c_from_s(S_I)[..., tok_idx, :]
            C_L = self.process_c(
                C_L_init + C_L_from_S
            )  # (L, c_atom), processed single atom features

            # ... Embed pairwise atom features
            P_LL_motif_pos = self.motif_pos_embedder(
                # Embed motif coordinates
                f["motif_pos"],
                atoms_have_struct_mask,
            )
            P_LL_ref_pos = self.ref_pos_embedder(
                # Embed reference coordinates of atoms
                f["ref_pos"],
                atoms_in_same_token_and_have_seq_mask,
            )
            P_LL_from_Z = Z_II[..., tok_idx, :, :][..., tok_idx, :]

            P_LL_list = [P_LL_motif_pos, P_LL_ref_pos, P_LL_from_Z]
            P_LL = (
                sum(P_LL_list) / len(P_LL_list)
            )  # (L, L, n_token_head), average processed pairwise per-token-head atom features

            P_LL = self.process_p(
                P_LL
            )  # (L, L, n_atom_head), processed pairwise per-atom-head atom features

            P_LL = P_LL.masked_fill(
                # Mask out inter-batch positions in P_LL to avoid
                # providing any information about them to the model
                ~atoms_in_same_example_mask.unsqueeze(-1),
                0.0,
            )

            return {
                # Token features
                "S_I": S_I,
                "Z_II": Z_II,
                # Atom features
                "C_L": C_L,
                "P_LL": P_LL,
            }

        tokens = init_tokens_impl()
        return init_atoms_impl(**tokens)


class DiffusionTokenEncoder(nn.Module):
    """
    Token encoder module for diffusion model trunk. Takes initial token features from TokenInitializer and applies further processing to prepare them for input to the model trunk.

    Args:
        c_s: Channel dimension for token features
        c_z: Channel dimension for pairwise token features
        c_token: Channel dimension for token features output by the encoder (input to model trunk)
        c_atompair: Channel dimension for pairwise atom features (used if use_distogram is True to embed distance information into Z_II)
        sigma_data: Sigma for the data distribution, used for bucketizing distances if use_sinusoidal_distogram_embedder is False
        n_pairformer_blocks: Number of pairformer blocks to run in the encoder
        pairformer_block: Config for pairformer blocks in the encoder
        use_distogram: Whether to embed distance information into Z_II using either a sinusoidal embedding or bucketized distances. This can provide useful geometric information to the token features at each diffusion step, and can improve performance of the model, but comes with an additional computational cost, especially if use_sinusoidal_distogram_embedder is False and we need to compute bucketized distances at each step.
        use_self: Whether to embed self-distance information into Z_II, which can provide information about the number of atoms in each token and can help the model learn to predict this correctly, but comes with an additional computational cost.
        use_sinusoidal_distogram_embedder: If use_distogram is True, whether to use a sinusoidal embedding of distances, which is more computationally efficient at inference time since it doesn't require recomputing bucketized distances at each step, but may not perform as well as learned bucket embeddings.
        attn_backend: Must be one of ("manual", "megafold", "flex").
    """

    def __init__(
        self,
        c_s,
        c_z,
        c_token,
        c_atompair,
        sigma_data,
        n_pairformer_blocks,
        pairformer_block,
        use_distogram,
        use_self,
        use_sinusoidal_distogram_embedder=True,
        attn_backend="manual",
        **_,
    ):
        super().__init__()

        # Sequence processing
        self.transition_1 = nn.ModuleList(
            [
                Transition(c=c_s, n=2),
                Transition(c=c_s, n=2),
            ]
        )

        # Post-processing of z
        self.n_bins_distogram = 65  # n bins for both self distogram and distogram
        n_bins_noise = self.n_bins_distogram
        self.use_self = use_self
        self.use_distogram = use_distogram
        self.use_sinusoidal_distogram_embedder = use_sinusoidal_distogram_embedder
        if self.use_distogram:
            if self.use_sinusoidal_distogram_embedder:
                self.dist_embedder = SinusoidalDistEmbed(c_atompair=c_z)
                n_bins_noise = c_z
            else:
                self.bucketize_fn = functools.partial(
                    bucketize_scaled_distogram,
                    min_dist=1,
                    max_dist=30,
                    sigma_data=sigma_data,
                    n_bins=self.n_bins_distogram,
                )
        cat_c_z = (
            c_z
            + int(self.use_distogram) * n_bins_noise
            + int(self.use_self) * self.n_bins_distogram
        )
        self.process_z = nn.Sequential(
            RMSNorm(cat_c_z),
            linearNoBias(cat_c_z, c_z),
        )

        self.transition_2 = nn.ModuleList(
            [
                Transition(c=c_z, n=2),
                Transition(c=c_z, n=2),
            ]
        )

        # Pairformer without triangle updates
        if n_pairformer_blocks > 0:
            self.pairformer_stack = nn.ModuleList(
                [
                    PairformerBlock(c_s=c_s, c_z=c_z, **pairformer_block)
                    for _ in range(n_pairformer_blocks)
                ]
            )
        else:
            self.pairformer_stack = None

        self.use_flex_attention = attn_backend == "flex"
        self.use_checkpointing = not DISABLE_CHECKPOINTING

    def forward(
        self,
        f,
        R_L,
        S_init_I,
        Z_init_II,
        C_L,
        P_LL,
        token_flex_block_mask=None,
        token_flex_block_mask_ready=False,
        **kwargs,
    ) -> tuple:
        """
        Pool atom-level features to token-level features, encoding them into Z_II and S_I, and prepare A_I.

        Args:
            f: Dict containing input features. Expected to contain at least:
                - atom_to_token_map: Long tensor of shape (L,) mapping each atom to its corresponding token index
                - batch_token_lvl: Long tensor of shape (I,) indicating batch indices for each token (used to avoid mixing information between examples in the batch)
                - batch_atom_lvl: Long tensor of shape (L,) indicating batch indices for each atom (used to avoid mixing information between examples in the batch)
                - is_motif_atom_with_fixed_coord: Bool tensor of shape (L,) indicating which atoms have fixed coordinates that should be embedded into P_LL via motif_pos_embedder (if use_distogram is True, these atoms will also be used to compute distance information to embed into Z_II)
                - is_motif_atom_with_fixed_seq: Bool tensor of shape (L,) indicating which atoms have fixed sequence that should be considered when applying ref_pos_embedder to P_LL (if use_distogram is True, these atoms will also be used to compute distance information to embed into Z_II)
            R_L: Float tensor of shape (B, L, 3) containing coordinates for each atom in the batch, where B is the batch size and L is the number of atoms. These are used to compute distance information to embed into Z_II if use_distogram is True.
            S_init_I: Float tensor of shape (I, c_s) containing initial token features from the TokenInitializer, where I is the number of tokens. These are further processed and mixed with atom features in this encoder before being passed to the model trunk.
            Z_init_II: Float tensor of shape (I, I, c_z) containing initial pairwise token features from the TokenInitializer, where I is the number of tokens. These are further processed and mixed with atom features in this encoder before being passed to the model trunk.
            C_L: Float tensor of shape (L, c_atom) containing atom features from the TokenInitializer, where L is the number of atoms. These are pooled to the token level and mixed into Z_II in this encoder before being passed to the model trunk.
            P_LL: Float tensor of shape (L, L, c_atompair) containing pairwise atom features from the TokenInitializer, where L is the number of atoms. These are used to compute distance information to embed into Z_II if use_distogram is True, and can also be mixed into Z_II in this encoder before being passed to the model trunk if use_distogram is False (in which case we rely on the model trunk to learn to extract useful geometric information from the atom features without explicit distance embeddings).
            token_flex_block_mask: Optional cached token-level FlexAttention block mask for the encoder. If not provided, it will be created on the fly if needed for the pairformer blocks in this encoder.
            token_flex_block_mask_ready: Whether the token_flex_block_mask is already cached and ready to be used. If False, a new mask will be created if needed for the pairformer blocks in this encoder.
            **kwargs: May contain additional features needed for the distogram embedding, such as D_II_self if use_self is True

        Returns:
            Tuple containing:
                - S_I: Float tensor of shape (I, c_s) containing processed token features after mixing with atom features via P_LL, ready to be passed to the model trunk
                - Z_II: Float tensor of shape (I, I, c_z) containing processed pairwise token features after mixing with atom features via P_LL, ready to be passed to the model trunk
                - token_flex_block_mask: Optional cached token-level FlexAttention block mask for the diffusion token encoder.
                - token_flex_block_mask_ready: Whether the token-level FlexAttention block mask is already cached or intentionally unavailable.
        """

        B = R_L.shape[0]

        # Build masks
        token_pair_mask = _token_pair_mask(f, device=S_init_I.device)
        flex_block_mask = None
        flex_block_mask_ready = False
        if self.use_flex_attention:
            flex_block_mask, flex_block_mask_ready = maybe_create_flex_block_mask(
                token_pair_mask,
                S_init_I.shape[-2],
                S_init_I.device,
                use_flex_attention=self.use_flex_attention,
                flex_block_mask=token_flex_block_mask,
                flex_block_mask_ready=token_flex_block_mask_ready,
            )

        def token_embed_impl(S_init_I, Z_init_II):
            S_I = S_init_I
            for b in range(2):
                S_I = S_I + self.transition_1[b](S_I)

            Z_II = Z_init_II.unsqueeze(0).expand(B, -1, -1, -1)  # B, I, I, c_z

            Z_II_list = [Z_II]
            if self.use_distogram:
                # Noise / self conditioning pair
                if self.use_sinusoidal_distogram_embedder:
                    mask = f["is_motif_atom_with_fixed_coord"][f["is_ca"]]
                    mask = (mask[None, :] != mask[:, None]).unsqueeze(
                        -1
                    )  # remove off-diagonals where distances don't make sense across time
                    D_LL = self.dist_embedder(R_L[..., f["is_ca"], :], ~mask)
                else:
                    D_LL = self.bucketize_fn(
                        R_L[..., f["is_ca"], :]
                    )  # [B, I, I, n_bins]
                Z_II_list.append(D_LL)
            if self.use_self:
                D_II_self = kwargs.get("D_II_self")
                if D_II_self is None:
                    D_II_self = torch.zeros(
                        Z_II.shape[:-1] + (self.n_bins_distogram,),
                        device=Z_II.device,
                        dtype=Z_II.dtype,
                    )
                Z_II_list.append(D_II_self)
            Z_II = torch.cat(Z_II_list, dim=-1)

            # Mask out invalid pairwise token features before token pooling.
            Z_II = Z_II.masked_fill(~token_pair_mask[None, ..., None], 0.0)

            # Flatten concatenated dims
            Z_II = self.process_z(Z_II)

            for b in range(2):
                Z_II = Z_II + self.transition_2[b](Z_II)

            # Pairformer to mix
            if exists(self.pairformer_stack):
                for block in self.pairformer_stack:
                    S_I, Z_II = block(
                        S_I,
                        Z_II,
                        A_mask=token_pair_mask,
                        flex_block_mask=flex_block_mask,
                        flex_block_mask_ready=flex_block_mask_ready,
                    )

            # Mask out invalid pairwise token features before the diffusion token block.
            Z_II = Z_II.masked_fill(~token_pair_mask[None, ..., None], 0.0)

            return S_I, Z_II

        token_embed = (
            activation_checkpointing(token_embed_impl)
            if self.use_checkpointing and not self.use_flex_attention
            else token_embed_impl
        )

        return token_embed(S_init_I, Z_init_II)
