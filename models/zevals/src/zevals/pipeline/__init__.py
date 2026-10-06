from __future__ import annotations

from importlib import import_module
from typing import Any

from zevals.pipeline.base import EvalPipeline, EvalResults

__all__ = [
    "EvalPipeline",
    "EvalResults",
    "MoleculeEvalPipeline",
    "LigandEvalPipeline",
    "ProteinEvalPipeline",
    "RNAEvalPipeline",
    "MaterialEvalPipeline",
]

_LAZY_IMPORTS = {
    "MoleculeEvalPipeline": "zevals.pipeline.molecules",
    "LigandEvalPipeline": "zevals.pipeline.ligands",
    "ProteinEvalPipeline": "zevals.pipeline.proteins",
    "RNAEvalPipeline": "zevals.pipeline.rna",
    "MaterialEvalPipeline": "zevals.pipeline.materials",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
