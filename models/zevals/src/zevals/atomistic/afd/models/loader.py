"""
Pretrained CT loading helpers.

The upstream `model_helper.load_model` does several things we don't need
(prior model construction, EMA loading, head-resetting, weight-decay groups).
Here we keep only what is required to:

  1. Download a checkpoint from HuggingFace under `Ty-Perez/<model_name>`.
  2. Build a `CET` with the hyper-parameters baked into the checkpoint.
  3. Load the state-dict (stripping the Lightning "model." prefix).

Two entry points:
  - `load_pretrained_ct(model_name="ct-scd-pcq", cache_dir=..., device=...)`
  - `load_ct_from_ckpt(filepath, device=...)`
"""

from __future__ import annotations

import os
import re

import torch

from .scd_model import CET

_CET_CONFIG_KEYS = (
    "emb_dim",
    "num_layers",
    "num_heads",
    "num_rbf",
    "rbf_type",
    "trainable_rbf",
    "neighbor_embedding",
    "max_num_neighbors",
    "distance_influence",
    "cutoff_lower",
    "cutoff_upper",
    "max_z",
    "layernorm_on_vec",
    "check_errors",
    "vector_cutoff",
    "p_droppath",
    "p_dropcond",
    "inv_post_norm",
    "vec_post_norm",
    "vec_prenorm",
    "derivative",
    "activation",
    "aggregation",
    "emb_agg",
)


def _ct_kwargs_from_hparams(hparams: dict) -> dict:
    """Pull just the CET-relevant entries out of a Lightning hparams dict.

    The upstream Lightning checkpoint stores a large hparams dict that mixes
    architecture config, data-loader config, optimiser config, etc. We pluck
    out only what `CET.__init__` accepts.
    """
    return {k: hparams[k] for k in _CET_CONFIG_KEYS if k in hparams}


def load_ct_from_ckpt(
    filepath: str,
    device: str = "cpu",
    strict: bool = False,
) -> CET:
    """Build a `CET` from a Lightning .ckpt file and load its weights.

    Mirrors `models/model_helper.py:load_model`, minus prior/EMA/head-reset logic.
    """
    ckpt = torch.load(filepath, map_location="cpu", weights_only=False)

    hparams = ckpt.get("hyper_parameters", {})
    assert (
        hparams.get("model", "CET") == "CET"
    ), f"This loader only supports CET checkpoints; got model={hparams.get('model')!r}"

    cet_kwargs = _ct_kwargs_from_hparams(hparams)
    model = CET(**cet_kwargs)

    # Split EMA from raw weights; we only use the raw weights — README
    # explicitly states ema_model on HF is untrained.
    state_dict_raw = {
        k: v for k, v in ckpt["state_dict"].items() if not k.startswith("ema_model.")
    }
    state_dict = {re.sub(r"^model\.", "", k): v for k, v in state_dict_raw.items()}

    result = model.load_state_dict(state_dict, strict=strict)

    # The pretrained CET has no prior_model; allow only prior_model.* in the
    # set of missing keys.
    non_prior_missing = [
        k for k in result.missing_keys if not k.startswith("prior_model.")
    ]
    if non_prior_missing:
        raise RuntimeError(
            f"Missing keys when loading CET state_dict (not in prior_model): "
            f"{non_prior_missing}"
        )
    if result.unexpected_keys:
        # Don't raise — these usually correspond to optimizer/scheduler state
        # leaking into the dict, or to training-time-only modules removed in
        # this minimal extraction (e.g. the FRAD variant).
        import warnings

        warnings.warn(
            f"Unexpected keys in CET state_dict (ignored): {result.unexpected_keys}",
            RuntimeWarning,
        )

    model.eval()
    return model.to(device)


def load_pretrained_ct(
    model_name: str = "ct-scd-pcq",
    cache_dir: str = "./experiments",
    device: str = "cpu",
    repo_prefix: str = "Ty-Perez/",
    filename: str = "last.ckpt",
) -> CET:
    """Download a CT checkpoint from HuggingFace and load it.

    Args:
        model_name: HuggingFace model repo suffix under `repo_prefix`, such as
            `"ct-scd-pcq"`, or a full repo ID, such as
            `"Ty-Perez/ct-scd-pcq"`.
        cache_dir: Where `huggingface_hub` should cache the download.
        device: Device to place the model on after loading. Default "cpu".
        repo_prefix: HuggingFace org/user prefix. Default Ty-Perez/.
        filename: Checkpoint filename on the HF repo. Default last.ckpt.
    """
    # Lazy import so unit tests that mock out the network can avoid the dependency.
    from huggingface_hub import hf_hub_download

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    repo_id = model_name if "/" in model_name else f"{repo_prefix}{model_name}"
    ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
    )
    return load_ct_from_ckpt(ckpt_path, device=device)


__all__ = ["load_pretrained_ct", "load_ct_from_ckpt"]
