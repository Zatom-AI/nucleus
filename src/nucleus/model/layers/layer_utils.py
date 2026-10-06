import torch.nn as nn
from foundry.training.checkpoint import activation_checkpointing
from rfd3.model.layers.layer_utils import (
    RMSNorm,
    linearNoBias,
)
from torch.nn.functional import silu

from nucleus import DISABLE_CHECKPOINTING


class Transition(nn.Module):
    """
    A simple transition layer that applies a linear transformation followed by a
    non-linear activation and another linear transformation. This is often used in
    transformer architectures to process token representations.
    """

    def __init__(self, n, c):
        super().__init__()
        self.layer_norm_1 = RMSNorm(c)
        self.linear_1 = linearNoBias(c, n * c)
        self.linear_2 = linearNoBias(c, n * c)
        self.linear_3 = linearNoBias(n * c, c)

        self.use_checkpointing = not DISABLE_CHECKPOINTING

    def forward_impl(
        self,
        X,
    ):
        """Forward pass of the transition layer."""
        X = self.layer_norm_1(X)
        A = self.linear_1(X)
        B = self.linear_2(X)
        X = self.linear_3(silu(A) * B)
        return X

    def forward(self, X):
        """Forward method that wraps the forward_impl with activation checkpointing."""
        forward = (
            activation_checkpointing(self.forward_impl)
            if self.use_checkpointing
            else self.forward_impl
        )
        return forward(X)
