"""UMA: near-DFT energies and forces for generated 3D structures.

An optional module. It needs ``fairchem-core`` (the ``uma`` extra) and gated
Hugging Face weights, so nothing here is imported until a pipeline is told to
run it (``compute_uma_forces: true``). Every name below is resolved lazily, so
``import zevals`` stays cheap and works without fairchem installed.

Use the modality wrapper rather than the core, because it sets the correct UMA
task head for the chemistry:

* molecules -> ``zevals.molecules.uma_forces`` (task ``omol``, aperiodic,
  charge aware)
* materials -> ``zevals.materials.uma_forces`` (task ``omat``, periodic cell)

See ``docs/UMA-guide.md`` for setup, metric meanings and caveats.
"""

from __future__ import annotations

from importlib import import_module

_EXPORT_MODULES = {
    "UMAModel": "zevals.atomistic.uma.model",
    "UMA_TASKS": "zevals.atomistic.uma.model",
    "UMA_PERIODIC_TASKS": "zevals.atomistic.uma.model",
    "load_fairchem": "zevals.atomistic.uma.model",
    "normalize_device": "zevals.atomistic.uma.model",
    "pyg_batch_to_atoms": "zevals.atomistic.uma.model",
    "clear_model_cache": "zevals.atomistic.uma.adapters",
    "get_uma_model": "zevals.atomistic.uma.adapters",
    "mol_to_atoms": "zevals.atomistic.uma.adapters",
    "sample_to_atoms": "zevals.atomistic.uma.adapters",
    "samples_to_atoms": "zevals.atomistic.uma.adapters",
    "structure_to_atoms": "zevals.atomistic.uma.adapters",
    "DEFAULT_CHECKPOINT": "zevals.atomistic.uma.metrics",
    "UMA_KEYS": "zevals.atomistic.uma.metrics",
    "compute_uma_force_metrics": "zevals.atomistic.uma.metrics",
    "score_uma_samples": "zevals.atomistic.uma.metrics",
    "uma_metric_keys": "zevals.atomistic.uma.metrics",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str):
    if name not in _EXPORT_MODULES:
        raise AttributeError(name)
    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
