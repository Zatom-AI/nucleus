from __future__ import annotations

import torch
import torch.nn as nn


def _xavier_linear_(module: nn.Module) -> None:
    """Apply Xavier uniform initialization to all linear layers in a module."""
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class SelfEmbeddingConditioner(nn.Module):
    """SCD-style graph embedding head and AdaLN conditioning projection.

    The embedding head mirrors the SCD reference implementation: project
    invariant per-node features, sum-pool them to one vector per graph, then
    pass the pooled vector through a two-layer MLP bottleneck. During the
    conditioned denoising pass, the graph embedding is dropped by replacing it
    with a learned mask token with probability ``condition_dropout``.
    """

    def __init__(
        self,
        *,
        c_token: int,
        c_s: int,
        c_atom: int | None = None,
        atom_probability: float = 0.5,
        condition_dropout: float = 0.2,
        clip_value: float = 100.0,
    ):
        super().__init__()
        if not 0.0 <= atom_probability <= 1.0:
            raise ValueError("atom_probability must be between 0 and 1.")
        self.c_token = c_token
        self.c_s = c_s
        self.c_atom = c_token if c_atom is None else c_atom
        self.atom_probability = atom_probability
        self.condition_dropout = condition_dropout
        self.clip_value = clip_value

        self.pre_proj = nn.Sequential(
            nn.LayerNorm(c_token),
            nn.Linear(c_token, c_token),
            nn.SiLU(),
            nn.LayerNorm(c_token),
        )
        self.atom_pre_proj = nn.Sequential(
            nn.LayerNorm(self.c_atom),
            nn.Linear(self.c_atom, c_token),
            nn.SiLU(),
            nn.LayerNorm(c_token),
        )
        self.post_pool_mlp = nn.Sequential(
            nn.LayerNorm(c_token),
            nn.Linear(c_token, c_token),
            nn.SiLU(),
            nn.LayerNorm(c_token),
            nn.Linear(c_token, c_token),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, c_token))
        self.condition_norm = nn.LayerNorm(c_token)
        self.token_norm = nn.LayerNorm(c_token)
        self.s_norm = nn.LayerNorm(c_s)
        self.to_token_adaln = nn.Linear(c_token, 2 * c_token)
        self.to_s_adaln = nn.Linear(c_token, 2 * c_s)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        _xavier_linear_(self.pre_proj)
        _xavier_linear_(self.atom_pre_proj)
        _xavier_linear_(self.post_pool_mlp)
        _xavier_linear_(self.to_token_adaln)
        _xavier_linear_(self.to_s_adaln)
        nn.init.xavier_uniform_(self.mask_token)
        self.condition_norm.reset_parameters()
        self.token_norm.reset_parameters()
        self.s_norm.reset_parameters()

    @staticmethod
    def _sum_pool(x: torch.Tensor, batch_idx: torch.Tensor, batch_size: int):
        """Sum-pool a [D, N, C] tensor to [D, B, C] by graph index."""
        out = x.new_zeros((x.shape[0], batch_size, x.shape[-1]))
        index = batch_idx.view(1, -1, 1).expand(x.shape[0], -1, x.shape[-1])
        return out.scatter_add(1, index, x)

    def embed(
        self,
        embeddings: torch.Tensor,
        *,
        batch_idx: torch.Tensor | None = None,
        batch_token_lvl: torch.Tensor | None = None,
        batch_size: int,
        input_level: str = "token",
    ) -> torch.Tensor:
        """Create one self-conditioning vector per diffusion sample and graph."""
        if batch_idx is None:
            batch_idx = batch_token_lvl
        if batch_idx is None:
            raise ValueError("Self-embedding pooling requires a batch index tensor.")

        if input_level == "token":
            embeddings = self.pre_proj(embeddings)
        elif input_level == "atom":
            embeddings = self.atom_pre_proj(embeddings)
        else:
            raise ValueError(f"Unknown self-embedding input level: {input_level}")

        graph_embeddings = self._sum_pool(
            embeddings,
            batch_idx=batch_idx,
            batch_size=batch_size,
        )
        return self.post_pool_mlp(graph_embeddings)

    def embed_random_level(
        self,
        *,
        token_embeddings: torch.Tensor,
        atom_embeddings: torch.Tensor,
        batch_token_lvl: torch.Tensor,
        batch_atom_lvl: torch.Tensor,
        batch_size: int,
        atom_probability: float | None = None,
    ) -> torch.Tensor:
        """Pool token or atom clean-pass features into a graph condition."""
        atom_probability = (
            self.atom_probability if atom_probability is None else atom_probability
        )
        token_condition_B = self.embed(
            token_embeddings,
            batch_idx=batch_token_lvl,
            batch_size=batch_size,
            input_level="token",
        )
        atom_condition_B = self.embed(
            atom_embeddings,
            batch_idx=batch_atom_lvl,
            batch_size=batch_size,
            input_level="atom",
        )
        if atom_probability <= 0.0:
            return token_condition_B
        if atom_probability >= 1.0:
            return atom_condition_B

        use_atom = (
            torch.rand(token_condition_B.shape[:2], device=token_condition_B.device)
            < atom_probability
        )
        return torch.where(use_atom[..., None], atom_condition_B, token_condition_B)

    def _drop_condition(self, condition_B: torch.Tensor) -> torch.Tensor:
        if self.condition_dropout <= 0.0 or not self.training:
            return condition_B
        drop_mask = (
            torch.rand(condition_B.shape[:2], device=condition_B.device)
            < self.condition_dropout
        )
        mask_token = self.mask_token.to(
            device=condition_B.device, dtype=condition_B.dtype
        )
        return torch.where(drop_mask[..., None], mask_token, condition_B)

    def _zero_parameter_dependency(self, reference: torch.Tensor) -> torch.Tensor:
        """Return scalar zero whose graph touches each conditioner parameter."""
        zero = reference.new_zeros(())
        for parameter in self.parameters():
            zero = zero + parameter.reshape(-1)[0].to(dtype=reference.dtype) * 0.0
        return zero

    def condition_token_inputs(
        self,
        *,
        A_I: torch.Tensor,
        S_I: torch.Tensor,
        condition_B: torch.Tensor | None,
        batch_token_lvl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Modulate token-track inputs with graph-level SCD conditioning."""
        if condition_B is None:
            zero = self._zero_parameter_dependency(A_I)
            return A_I + zero, S_I + zero

        condition_B = self._drop_condition(condition_B)
        condition_B = condition_B.clamp(
            min=-self.clip_value,
            max=self.clip_value,
        )
        condition_B = self.condition_norm(condition_B)

        token_scale, token_shift = self.to_token_adaln(condition_B).chunk(2, dim=-1)
        s_scale, s_shift = self.to_s_adaln(condition_B).chunk(2, dim=-1)

        token_scale = token_scale[:, batch_token_lvl, :].to(A_I.dtype)
        token_shift = token_shift[:, batch_token_lvl, :].to(A_I.dtype)
        s_scale = s_scale[:, batch_token_lvl, :].to(S_I.dtype)
        s_shift = s_shift[:, batch_token_lvl, :].to(S_I.dtype)

        A_I = self.token_norm(A_I).to(A_I.dtype) * (1 + token_scale) + token_shift
        S_I = self.s_norm(S_I).to(S_I.dtype) * (1 + s_scale) + s_shift
        return A_I, S_I
