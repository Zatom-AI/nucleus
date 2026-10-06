from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeAFDScorer:
    def __init__(self, score: float = 1.25):
        self.score = score
        self.records = None
        self.kwargs = None

    def __call__(self, records, **kwargs) -> float:
        self.records = list(records)
        self.kwargs = kwargs
        return self.score


def _load_afd_loader_module(monkeypatch):
    """Load loader.py directly so this test does not require PyG model imports."""
    module_name = "zevals.atomistic.afd.models.loader"
    package_name = "zevals.atomistic.afd.models"
    loader_path = ROOT / "zevals" / "atomistic" / "afd" / "models" / "loader.py"

    package = ModuleType(package_name)
    package.__path__ = [str(loader_path.parent)]
    scd_model = ModuleType(f"{package_name}.scd_model")
    scd_model.CET = object

    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, f"{package_name}.scd_model", scd_model)
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    spec = importlib.util.spec_from_file_location(module_name, loader_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_afd_score_is_zero_for_identical_features():
    from zevals.atomistic.afd.score import afd_score

    features = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ],
        dtype=np.float64,
    )

    assert afd_score(features, features) == pytest.approx(0.0, abs=1e-6)


def test_atomistic_afd_public_api_is_lazy_importable():
    import zevals.atomistic.afd as afd

    assert "AFDScore" in afd.__all__
    assert "afd_score" in afd.__all__
    assert hasattr(afd.AFDScore, "embeddings")


def test_add_standard_keys_normalizes_numpy_periodic_metadata():
    """NumPy material records are converted to CT-compatible tensors."""
    from torch_geometric.data import Data

    from zevals.atomistic.afd.models.transforms import AddStandardKeys

    data = Data(
        z=np.asarray([14, 8], dtype=np.int64),
        pos=np.asarray([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]], dtype=np.float32),
        cell=np.diag([5.0, 6.0, 7.0])[None, :, :],
        pbc=np.ones((1, 3), dtype=bool),
        natoms=np.asarray([2], dtype=np.int64),
    )

    transformed = AddStandardKeys()(data)

    assert transformed.z.dtype == torch.long
    assert transformed.pos.dtype == torch.float32
    assert transformed.cell.dtype == torch.float32
    assert transformed.pbc.dtype == torch.bool
    assert transformed.natoms.dtype == torch.long
    assert transformed.cell.shape == (1, 3, 3)
    assert transformed.pbc.shape == (1, 3)
    assert transformed.natoms.shape == (1,)


def test_lattice_offset_builder_rejects_excessive_cartesian_product():
    """Degenerate periodic cells fail before torch allocates the offset grid."""
    from zevals.atomistic.afd.models.graph_utils.fast import _build_lattice_offsets

    with pytest.raises(
        ValueError,
        match=r"8,120,601 lattice offsets.*\(100, 100, 100\).*1,000,000",
    ):
        _build_lattice_offsets(
            torch.tensor([100.0, 100.0, 100.0]),
            device="cpu",
            dtype=torch.float32,
        )


def test_lattice_offset_builder_preserves_normal_periodic_grid():
    """The safety guard preserves ordinary periodic offset construction."""
    from zevals.atomistic.afd.models.graph_utils.fast import _build_lattice_offsets

    offsets = _build_lattice_offsets(
        torch.tensor([1.0, 2.0, 0.0]),
        device="cpu",
        dtype=torch.float32,
    )

    assert offsets.shape == (15, 3)
    assert torch.any(torch.all(offsets == 0.0, dim=1))


def test_molecule_afd_metrics_match_reference_size_by_default():
    from zevals.molecules.afd import AFD_KEYS, compute_afd_metrics

    scorer = FakeAFDScorer(score=0.42)
    records = [
        {"z": [6], "pos": [[0.0, 0.0, 0.0]]},
        {"z": [8], "pos": [[1.0, 0.0, 0.0]]},
    ]

    metrics = compute_afd_metrics(
        records,
        scorer=scorer,
        seed=123,
        show_progress=False,
    )

    assert list(metrics) == list(AFD_KEYS)
    assert metrics["afd_score"] == pytest.approx(0.42)
    assert metrics["afd_num_samples"] == pytest.approx(2.0)
    assert metrics["afd_ref_sample_size"] == pytest.approx(2.0)
    assert scorer.records == records
    assert scorer.kwargs == {
        "ref_sample_size": 2,
        "seed": 123,
        "show_progress": False,
    }


def test_molecule_afd_metrics_can_disable_reference_matching():
    from zevals.molecules.afd import compute_afd_metrics

    scorer = FakeAFDScorer()
    metrics = compute_afd_metrics(
        [{"z": [6], "pos": [[0.0, 0.0, 0.0]]}],
        scorer=scorer,
        match_reference_size=False,
    )

    assert metrics["afd_ref_sample_size"] == pytest.approx(0.0)
    assert scorer.kwargs == {}


def test_empty_molecule_afd_metrics_return_zeroes():
    from zevals.molecules.afd import AFD_KEYS, compute_afd_metrics

    assert compute_afd_metrics([]) == {key: 0.0 for key in AFD_KEYS}


def test_load_pretrained_ct_accepts_full_huggingface_repo_id(monkeypatch, tmp_path):
    loader = _load_afd_loader_module(monkeypatch)

    captured = {}

    def fake_hf_hub_download(repo_id, filename, cache_dir):
        captured["repo_id"] = repo_id
        captured["filename"] = filename
        captured["cache_dir"] = cache_dir
        return str(tmp_path / filename)

    def fake_load_ct_from_ckpt(filepath, device):
        captured["filepath"] = filepath
        captured["device"] = device
        return "model"

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_hf_hub_download),
    )
    monkeypatch.setattr(loader, "load_ct_from_ckpt", fake_load_ct_from_ckpt)

    model = loader.load_pretrained_ct(
        "Ty-Perez/ct-scd-omol25",
        cache_dir=str(tmp_path),
        device="cpu",
    )

    assert model == "model"
    assert captured["repo_id"] == "Ty-Perez/ct-scd-omol25"
    assert captured["filename"] == "last.ckpt"
    assert captured["cache_dir"] == str(tmp_path)
    assert captured["device"] == "cpu"


def test_afd_model_cache_prefers_huggingface_env_vars(monkeypatch, tmp_path):
    from zevals.atomistic.afd.features import _default_model_cache_dir

    hf_hub_cache = tmp_path / "hub-cache"
    hf_home = tmp_path / "hf-home"
    xdg_cache_home = tmp_path / "xdg-cache"

    monkeypatch.setenv("HF_HUB_CACHE", str(hf_hub_cache))
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_cache_home))
    assert _default_model_cache_dir() == hf_hub_cache

    monkeypatch.delenv("HF_HUB_CACHE")
    assert _default_model_cache_dir() == hf_home / "hub"

    monkeypatch.delenv("HF_HOME")
    assert _default_model_cache_dir() == xdg_cache_home / "huggingface" / "hub"


def test_custom_reference_feature_cache_sanitizes_full_repo_model_name(tmp_path):
    from zevals.atomistic.afd.score import AFDScore

    class FakeFeaturizer:
        def __call__(self, dataset):
            return torch.ones((len(dataset), 2))

    scorer = AFDScore.__new__(AFDScore)
    scorer.featurizer = FakeFeaturizer()

    features = scorer._load_or_compute_ref_features(
        [{"z": [6], "pos": [[0.0, 0.0, 0.0]]}],
        model_name="Ty-Perez/ct-scd-omol25",
        feature_cache_dir=tmp_path,
    )

    assert features.shape == (1, 2)
    assert (tmp_path / "Ty-Perez__ct-scd-omol25_reference_features.pt").exists()
