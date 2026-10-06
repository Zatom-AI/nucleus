"""Smoke tests: every module in `zevals` imports cleanly, and every
major class constructs without touching the network or loading real model
weights (all torch-based oracles lazy-load on first `predict()`/`design()`
call, never in `__init__`).

This is a broad, cheap regression net — it won't catch logic bugs, but it
catches import-time errors (circular imports, typos, missing dependencies
in a code path that should be lazy) across the whole package in one pass.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _iter_submodules(package_name: str) -> list[str]:
    """All importable submodule names under `package_name`, excluding the
    vendored `staging/` tree (external repos, not zevals's own code)."""
    package = importlib.import_module(package_name)
    names = []
    for _finder, name, _is_pkg in pkgutil.walk_packages(
        package.__path__, prefix=f"{package_name}."
    ):
        if ".staging" in name or name.endswith(".staging"):
            continue
        names.append(name)
    return names


@pytest.mark.parametrize("module_name", _iter_submodules("zevals"))
def test_module_imports_cleanly(module_name: str) -> None:
    """Every non-staging module under zevals imports without error."""
    importlib.import_module(module_name)


def test_all_five_pipeline_classes_are_importable() -> None:
    """The lazy-import registry in pipeline/__init__.py resolves every entry."""
    from zevals.pipeline import (
        LigandEvalPipeline,
        MaterialEvalPipeline,
        MoleculeEvalPipeline,
        ProteinEvalPipeline,
        RNAEvalPipeline,
    )

    for cls in (
        MoleculeEvalPipeline,
        LigandEvalPipeline,
        ProteinEvalPipeline,
        RNAEvalPipeline,
        MaterialEvalPipeline,
    ):
        assert isinstance(cls, type)


def test_torch_oracles_construct_without_loading_weights() -> None:
    """ESMFold/ProteinMPNN/LigandMPNN construction is cheap and lazy — no
    checkpoint download or load should happen until predict()/design()."""
    from zevals.oracles.inverse_folders.ligand_mpnn import LigandMPNNFolder
    from zevals.oracles.inverse_folders.na_mpnn import NAMPNNFolder
    from zevals.oracles.inverse_folders.protein_mpnn import ProteinMPNNFolder
    from zevals.oracles.structure_predictors.esmfold import ESMFoldPredictor

    esmfold = ESMFoldPredictor(device="cpu")
    assert esmfold._model is None

    pmpnn = ProteinMPNNFolder(weights_dir="/nonexistent", device="cpu")
    assert pmpnn._model is None

    lmpnn = LigandMPNNFolder(checkpoint_path="/nonexistent.pt", device="cpu")
    assert lmpnn._model is None

    nmpnn = NAMPNNFolder(checkpoint_path="/nonexistent.pt", device="cpu")
    assert nmpnn._model is None


def test_foldseek_searcher_constructs() -> None:
    from zevals.oracles.structure_search.foldseek import FoldSeekSearcher

    searcher = FoldSeekSearcher(binary_path=None)
    assert searcher.binary_path is None


def test_usalign_searcher_constructs() -> None:
    from zevals.oracles.structure_search.usalign import USalignSearcher

    searcher = USalignSearcher(binary_path=None)
    assert searcher.binary_path is None


def test_zevals_suite_constructs_with_no_args() -> None:
    from zevals.pipeline.suite import ZevalSuite

    suite = ZevalSuite()
    assert suite._pipelines == {}
    assert suite.results_dir is None


def test_cli_module_has_main_entrypoint() -> None:
    from zevals import cli

    assert callable(cli.main)
