"""
Feature extraction for AFD.

Runs atomistic records through the pretrained CT model and collects graph-level
``mol_emb`` embeddings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from zevals.atomistic.afd.models import AddStandardKeys, load_pretrained_ct

_MOLECULE_KEYS = ("qm9", "molecules", "small_molecules")
_MATERIAL_KEYS = ("mp20", "mp_20", "materials", "crystals", "periodic_crystals")


def _default_model_cache_dir() -> Path:
    """Return the default Hugging Face hub cache directory."""
    if hf_hub_cache := os.environ.get("HF_HUB_CACHE"):
        return Path(hf_hub_cache).expanduser()
    if hf_home := os.environ.get("HF_HOME"):
        return Path(hf_home).expanduser() / "hub"
    if xdg_cache_home := os.environ.get("XDG_CACHE_HOME"):
        return Path(xdg_cache_home).expanduser() / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _get_model_name(data_keyword: str | None) -> str:
    """Select the canonical pretrained CT checkpoint for a data modality."""
    if data_keyword is None:
        raise ValueError("data_keyword is required when model_name is not provided")

    data_keyword = data_keyword.lower()
    if data_keyword in _MOLECULE_KEYS:
        return "ct-scd-pcq"
    if data_keyword in _MATERIAL_KEYS:
        return "ct-scd-amp20"

    valid = [*_MOLECULE_KEYS, *_MATERIAL_KEYS]
    raise ValueError(
        f"Unrecognized data keyword {data_keyword!r}. Please use one of {valid}."
    )


def _build_model(
    model_name: str,
    cache_dir: str | Path | None = None,
    device: str | torch.device | None = "cpu",
):
    """Load a pretrained CT checkpoint."""
    if cache_dir is None:
        cache_dir = _default_model_cache_dir()
    if device is None:
        device = "cpu"
    return load_pretrained_ct(
        model_name,
        cache_dir=str(cache_dir),
        device=device,
    ).eval()


class GenerationDataset(Dataset):
    """In-memory wrapper for atomistic records plus an optional transform."""

    def __init__(self, data: Any, transform=None) -> None:
        if not hasattr(data, "__getitem__") or not hasattr(data, "__len__"):
            raise ValueError(
                "data must support __getitem__ and __len__ "
                "(for example a list of Data objects or a PyG Dataset)"
            )
        if len(data) == 0:
            raise ValueError("data must contain at least one record")

        self.data = data
        self.transform = transform

        sample = data[0]
        self.dict_data = isinstance(sample, dict)
        if self.dict_data:
            required_keys = {"z", "pos"}
            if not required_keys.issubset(sample):
                raise ValueError(f"Each dict record must include keys: {required_keys}")
        elif not isinstance(sample, Data):
            raise ValueError(
                "Each item must be a torch_geometric.data.Data object or a dict"
            )
        else:
            required_keys = {"z", "pos"}
            if not all(hasattr(sample, key) for key in required_keys):
                raise ValueError(f"Each Data object must have keys: {required_keys}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Data:
        data = self.data[idx]
        if self.dict_data:
            data = Data(**data)
        return self.transform(data) if self.transform is not None else data


@torch.no_grad()
def compute_features(
    dataset,
    model,
    transform=None,
    batch_size: int = 512,
    device: str | torch.device | None = None,
    show_progress: bool = True,
    feature_key: str = "mol_emb",
    desc: str = "features",
) -> torch.Tensor:
    """Run ``dataset`` through ``model`` and stack per-graph embeddings."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = torch.device(device)

    model = model.to(device).eval()
    generation_dataset = GenerationDataset(dataset, transform=transform)
    loader = DataLoader(generation_dataset, batch_size=batch_size, shuffle=False)

    chunks = []
    iterator = tqdm(
        loader,
        total=len(loader),
        desc=desc,
        unit="batch",
        disable=not show_progress,
        leave=False,
    )
    for batch in iterator:
        batch = batch.to(device)
        out = model(
            z=batch.z,
            pos=batch.pos,
            batch=batch.batch,
            graph_batch=batch,
        )
        chunks.append(out[feature_key].detach().to("cpu"))

    model = model.to("cpu")
    if not chunks:
        raise RuntimeError("compute_features got zero records")
    return torch.cat(chunks, dim=0)


class FeatureExtractor:
    """Hold a pretrained CT model and extract AFD features from records."""

    def __init__(
        self,
        data_keyword: str | None = None,
        model_name: str | None = None,
        batch_size: int = 512,
        device: str | torch.device | None = None,
        model_cache_dir: str | Path | None = None,
    ) -> None:
        if model_name is None:
            model_name = _get_model_name(data_keyword)
        self.device = device
        self.batch_size = batch_size
        self.model = _build_model(
            model_name,
            cache_dir=model_cache_dir,
            device=device,
        )
        self.standard_transform = AddStandardKeys()

    def _compute_features(self, dataset, **kwargs) -> torch.Tensor:
        kwargs.setdefault("batch_size", self.batch_size)
        kwargs.setdefault("device", self.device)
        return compute_features(
            dataset,
            self.model,
            transform=self.standard_transform,
            **kwargs,
        )

    def __call__(self, dataset, **kwargs) -> torch.Tensor:
        return self._compute_features(dataset, **kwargs)


__all__ = ["FeatureExtractor", "GenerationDataset", "compute_features"]
