"""AFD: Atomistic Fréchet Distance for atomistic generative models."""

from __future__ import annotations

from importlib import import_module

_EXPORT_MODULES = {
    "AFDScore": "zevals.atomistic.afd.score",
    "afd_score": "zevals.atomistic.afd.score",
    "frechet_distance": "zevals.atomistic.afd.score",
    "gaussian_stats": "zevals.atomistic.afd.score",
    "FeatureExtractor": "zevals.atomistic.afd.features",
    "compute_features": "zevals.atomistic.afd.features",
    "GenerationDataset": "zevals.atomistic.afd.data",
    "dict_to_data": "zevals.atomistic.afd.data",
    "load_generations": "zevals.atomistic.afd.data",
    "load_records_pt": "zevals.atomistic.afd.data",
    "register_loader": "zevals.atomistic.afd.data",
    "resolve_records": "zevals.atomistic.afd.data",
}

__all__ = [
    # High-level
    "AFDScore",
    "FeatureExtractor",
    "compute_features",
    "afd_score",
    "frechet_distance",
    "gaussian_stats",
    # Data
    "resolve_records",
    "load_records_pt",
    "load_generations",
    "dict_to_data",
    "register_loader",
    "GenerationDataset",
]


def __getattr__(name: str):
    if name not in _EXPORT_MODULES:
        raise AttributeError(name)
    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
