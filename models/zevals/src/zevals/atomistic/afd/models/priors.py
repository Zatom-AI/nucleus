"""
Minimal `BasePrior` class kept only so the `prior_model` ModuleList type-check
in CET's BaseModel works. The pretrained `ct-scd-pcq` checkpoint stores an empty
prior_model, so no concrete prior subclass is needed.

If you later want to attach an Atomref or similar, copy the relevant class
out of `SelfConditionedDenoisingAtoms/models/ET_models/priors.py`.
"""

from typing import Dict, Optional

from torch import Tensor, nn


class BasePrior(nn.Module):
    """Base class for prior models. Identity by default."""

    def __init__(self, dataset=None):
        super().__init__()

    def get_init_args(self):
        return {}

    def pre_reduce(self, x, z, pos, batch, extra_args: Optional[Dict[str, Tensor]]):
        return x

    def post_reduce(
        self,
        y,
        z,
        pos,
        batch,
        box: Optional[Tensor],
        extra_args: Optional[Dict[str, Tensor]],
    ):
        return y


__all__ = ["BasePrior"]
