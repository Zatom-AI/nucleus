"""Tests for restartable MLIP evaluation artifacts and aggregation."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from foundry.training.EMA import EMA
from lightning.fabric.strategies import SingleDeviceStrategy
from lightning.fabric.wrappers import _FabricModule
from omegaconf import OmegaConf
from zatom2.mlip_evaluation import (
    _PREDICTION_SCHEMA,
    MetricSums,
    _make_rank_local_inference_model,
    _prediction_rows,
    _reference_rows,
    aggregate_prediction_parts,
    resolve_aselmdb_paths,
)

from nucleus.utils import evaluation_manifest
from nucleus.utils.torch import as_cpu_float64_tensor

MLIP_MODULE_NAMES = (
    "zatom2.mlip_evaluation",
)


def test_as_cpu_float64_tensor_transfers_before_unsupported_cast(
    monkeypatch: pytest.MonkeyPatch,
):
    """The float64 conversion must happen only after leaving an MPS-like device."""
    calls = []

    class OrderCheckingTensor:
        def detach(self):
            calls.append("detach")
            return self

        def cpu(self):
            calls.append("cpu")
            return self

        def double(self):
            if not calls or calls[-1] != "cpu":
                raise RuntimeError("float64 cast attempted before CPU transfer")
            calls.append("double")
            return self

    tensor = OrderCheckingTensor()
    monkeypatch.setattr(torch, "as_tensor", lambda value: tensor)

    assert as_cpu_float64_tensor(object(), detach=True) is tensor
    assert calls == ["detach", "cpu", "double"]


def test_source_tree_fingerprint_detects_uncommitted_source_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Source identity changes without requiring a package-version bump."""
    package_dir = tmp_path / "manifest_fingerprint_target"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    before = evaluation_manifest.source_tree_fingerprint(
        ("manifest_fingerprint_target",)
    )
    (package_dir / "__init__.py").write_text("VALUE = 2\n")
    after = evaluation_manifest.source_tree_fingerprint(
        ("manifest_fingerprint_target",)
    )

    assert before["sha256"] != after["sha256"]


def test_prediction_identity_resolves_checkpoint_model_interpolations(
    monkeypatch: pytest.MonkeyPatch,
):
    """Environment-backed model settings cannot reuse predictions after changing."""
    monkeypatch.setattr(
        evaluation_manifest,
        "source_tree_fingerprint",
        lambda package_names: {"sha256": "source"},
    )
    monkeypatch.setattr(
        evaluation_manifest,
        "_runtime_versions",
        lambda: {"python": "test-runtime"},
    )
    run_config = OmegaConf.create({"evaluation": {}})
    checkpoint_config = OmegaConf.create(
        {
            "model": {"width": "${oc.env:MLIP_TEST_MODEL_WIDTH}"},
            "trainer": {"precision": "32-true"},
        }
    )

    monkeypatch.setenv("MLIP_TEST_MODEL_WIDTH", "8")
    before = evaluation_manifest.build_mlip_prediction_identity(
        config=run_config,
        checkpoint_config=checkpoint_config,
        preprocessing_configs={},
        model_package="unused-in-test",
        prediction_schema="schema",
    )
    monkeypatch.setenv("MLIP_TEST_MODEL_WIDTH", "16")
    after = evaluation_manifest.build_mlip_prediction_identity(
        config=run_config,
        checkpoint_config=checkpoint_config,
        preprocessing_configs={},
        model_package="unused-in-test",
        prediction_schema="schema",
    )

    assert (
        before["checkpoint_config"]["unresolved_sha256"]
        == after["checkpoint_config"]["unresolved_sha256"]
    )
    assert (
        before["checkpoint_config"]["resolved_sections"]["model"]
        != after["checkpoint_config"]["resolved_sections"]["model"]
    )


def _row(*, data_id: str = "biomolecules") -> dict:
    return {
        "global_index": 0,
        "example_id": "data0000:1",
        "data_id": data_id,
        "n_atoms": 2,
        "energy_eV": -8.0,
        "forces_eV_per_angstrom": [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        "has_reference": True,
        "energy_absolute_error_eV": 2.0,
        "energy_squared_error_eV2": 4.0,
        "energy_per_atom_absolute_error_eV": 1.0,
        "force_component_absolute_error_sum": 2.0,
        "force_component_squared_error_sum": 2.0,
        "force_vector_error_sum": 2.0,
        "force_vector_squared_error_sum": 2.0,
        "force_norm_absolute_error_sum": 2.0,
        "force_cosine_similarity_sum": 2.0,
        "force_cosine_similarity_count": 2,
        "num_force_components": 6,
        "num_force_atoms": 2,
    }


def test_metric_sums_report_paper_units():
    sums = MetricSums()
    sums.update(_row())
    metrics = sums.finalize()

    assert metrics["energy_mae_meV"] == pytest.approx(2000.0)
    assert metrics["energy_per_atom_mae_meV"] == pytest.approx(1000.0)
    assert metrics["force_mae_meV_per_angstrom"] == pytest.approx(1000 / 3)
    assert metrics["force_cosine_similarity"] == pytest.approx(1.0)


def test_metric_sums_mask_energy_without_masking_forces():
    row = _row()
    row.update(
        {
            "energy_eV": None,
            "has_reference": False,
            "energy_absolute_error_eV": None,
            "energy_squared_error_eV2": None,
            "energy_per_atom_absolute_error_eV": None,
        }
    )
    sums = MetricSums()
    sums.update(row)

    metrics = sums.finalize()

    assert metrics["num_examples"] == 1
    assert metrics["num_energy_systems"] == 0
    assert metrics["energy_mae_eV"] is None
    assert metrics["num_force_atoms"] == 2
    assert metrics["force_mae_meV_per_angstrom"] == pytest.approx(1000 / 3)


def test_prediction_row_masks_energy_and_preserves_force_schema():
    rows = _prediction_rows(
        examples=[
            {
                "example_id": "val-5:2114",
                "extra_info": {"calc_id": "rattled-1000-subsampled"},
            }
        ],
        global_indices=[73251],
        prediction={
            "energy_B": torch.tensor([float("nan")]),
            "energy_reconstruction_valid_B": torch.tensor([False]),
            "forces_L": torch.zeros(2, 3),
        },
        collated={
            "feats": {"batch_atom_lvl": torch.tensor([0, 0])},
            "ground_truth": {
                "energy_gt_B": torch.tensor([-40.0]),
                "energy_valid_mask_B": torch.tensor([True]),
                "force_gt_D_L": torch.ones(1, 2, 3),
                "force_valid_mask_L": torch.tensor([True, True]),
                "source_atom_count_B": torch.tensor([2]),
            },
        },
    )

    assert len(rows) == 1
    assert rows[0]["energy_eV"] is None
    assert rows[0]["energy_absolute_error_eV"] is None
    assert rows[0]["has_reference"] is False
    assert rows[0]["force_component_absolute_error_sum"] == pytest.approx(6.0)
    assert pa.Table.from_pylist(rows, schema=_PREDICTION_SCHEMA).schema == (
        _PREDICTION_SCHEMA
    )


@pytest.mark.parametrize("module_name", MLIP_MODULE_NAMES)
def test_prediction_row_rejects_nonfinite_energy_with_reference(module_name: str):
    """A failed labeled prediction cannot silently disappear from energy metrics."""
    mlip_module = importlib.import_module(module_name)

    with pytest.raises(
        ValueError,
        match="non-finite energy prediction.*nonfinite-energy",
    ):
        mlip_module._prediction_rows(
            examples=[{"example_id": "nonfinite-energy", "extra_info": {}}],
            global_indices=[0],
            prediction={
                "energy_B": torch.tensor([float("nan")]),
                "energy_reconstruction_valid_B": torch.tensor([True]),
                "forces_L": torch.zeros(1, 3),
            },
            collated={
                "feats": {"batch_atom_lvl": torch.tensor([0])},
                "ground_truth": {
                    "energy_gt_B": torch.tensor([0.0]),
                    "energy_valid_mask_B": torch.tensor([True]),
                    "force_gt_D_L": torch.zeros(1, 1, 3),
                    "force_valid_mask_L": torch.tensor([True]),
                    "source_atom_count_B": torch.tensor([1]),
                },
            },
        )


@pytest.mark.parametrize("module_name", MLIP_MODULE_NAMES)
def test_prediction_row_rejects_nonfinite_forces_with_reference(module_name: str):
    """A failed labeled prediction cannot silently disappear from force metrics."""
    mlip_module = importlib.import_module(module_name)

    with pytest.raises(
        ValueError,
        match="non-finite force prediction.*nonfinite-force",
    ):
        mlip_module._prediction_rows(
            examples=[{"example_id": "nonfinite-force", "extra_info": {}}],
            global_indices=[0],
            prediction={
                "energy_B": torch.tensor([0.0]),
                "energy_reconstruction_valid_B": torch.tensor([True]),
                "forces_L": torch.tensor([[float("nan"), 0.0, 0.0]]),
            },
            collated={
                "feats": {"batch_atom_lvl": torch.tensor([0])},
                "ground_truth": {
                    "energy_gt_B": torch.tensor([0.0]),
                    "energy_valid_mask_B": torch.tensor([True]),
                    "force_gt_D_L": torch.zeros(1, 1, 3),
                    "force_valid_mask_L": torch.tensor([True]),
                    "source_atom_count_B": torch.tensor([1]),
                },
            },
        )


@pytest.mark.parametrize("module_name", MLIP_MODULE_NAMES)
def test_prediction_error_is_propagated_from_a_peer_rank(module_name: str):
    """Every rank must leave evaluation when one rank sees an invalid prediction."""
    mlip_module = importlib.import_module(module_name)

    class PeerFailureFabric:
        device = torch.device("cpu")

        @staticmethod
        def all_gather(local_failed):
            assert not bool(local_failed.item())
            return torch.tensor([[0], [1]], dtype=torch.int64)

        @staticmethod
        def broadcast(local_detail, *, src):
            assert src == 1
            assert local_detail is None
            return {
                "rank": 1,
                "type": "_NonFinitePredictionError",
                "message": "non-finite force prediction for 'peer-example'",
            }

    with pytest.raises(
        RuntimeError,
        match="rank 1.*non-finite force prediction.*peer-example",
    ):
        mlip_module._synchronize_prediction_error(
            fabric=PeerFailureFabric(),
            local_error=None,
            rank=0,
            world_size=2,
        )


@pytest.mark.parametrize("module_name", MLIP_MODULE_NAMES)
@pytest.mark.parametrize(
    "failure_kind",
    ["nonfinite", "wrong_energy_count", "wrong_force_count"],
)
def test_invalid_prediction_does_not_flush_an_incomplete_part(
    module_name: str,
    failure_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A failed batch must remain missing so resume cannot treat it as complete."""
    mlip_module = importlib.import_module(module_name)
    examples = [
        {"example_id": "finite", "extra_info": {}},
        {"example_id": "nonfinite", "extra_info": {}},
    ]
    collated = {
        "feats": {"batch_atom_lvl": torch.tensor([0])},
        "ground_truth": {
            "energy_gt_B": torch.tensor([0.0]),
            "energy_valid_mask_B": torch.tensor([True]),
            "force_gt_D_L": torch.zeros(1, 1, 3),
            "force_valid_mask_L": torch.tensor([True]),
            "source_atom_count_B": torch.tensor([1]),
        },
    }

    class PredictionModel:
        calls = 0

        def forward(self, **kwargs):
            del kwargs
            self.calls += 1
            if self.calls == 1:
                energy = torch.tensor([0.0])
            elif failure_kind == "nonfinite":
                energy = torch.tensor([float("nan")])
            else:
                energy = (
                    torch.empty(0)
                    if failure_kind == "wrong_energy_count"
                    else torch.tensor([0.0])
                )
            return {
                "energy_B": energy,
                "energy_reconstruction_valid_B": torch.tensor([True]),
                "forces_L": (
                    torch.empty(0, 3)
                    if failure_kind == "wrong_force_count" and self.calls > 1
                    else torch.zeros(1, 3)
                ),
            }

    class SingleRankFabric:
        global_rank = 0
        world_size = 1
        device = torch.device("cpu")
        strategy = None

        @staticmethod
        def to_device(value):
            return value

        @staticmethod
        def barrier():
            raise AssertionError("A failed evaluation must not reach the barrier.")

    model = PredictionModel()
    trainer = SimpleNamespace(
        fabric=SingleRankFabric(),
        state={"model": model},
        _assemble_network_inputs=lambda batch: batch,
        _single_view_property_input=lambda network_input: network_input,
    )
    monkeypatch.setattr(
        mlip_module,
        "_build_dataset",
        lambda train_cfg, domain, path, max_atoms: examples,
    )
    monkeypatch.setattr(mlip_module, "SparseCollate", lambda: lambda batch: collated)
    evaluation_cfg = OmegaConf.create(
        {
            "max_system_atoms": 32,
            "max_examples_per_domain": None,
            "part_size": 8,
            "resume": False,
            "batch_size": 1,
            "num_workers": 0,
            "prefetch_factor": 2,
            "progress_every": 100,
        }
    )

    expected_error = {
        "nonfinite": "non-finite energy prediction.*nonfinite",
        "wrong_energy_count": "Expected 1 energy predictions, got 0",
        "wrong_force_count": (
            "Expected one force vector per collated atom.*shape \\(0, 3\\).*1 atoms"
        ),
    }[failure_kind]
    with pytest.raises(
        mlip_module._PredictionEvaluationError,
        match=expected_error,
    ):
        mlip_module._evaluate_domain(
            engine=SimpleNamespace(trainer=trainer),
            train_cfg=OmegaConf.create({}),
            domain="omol25",
            dataset_path="unused",
            output_dir=tmp_path,
            evaluation_cfg=evaluation_cfg,
        )

    assert not list((tmp_path / "predictions").rglob("part-*.parquet"))


@pytest.mark.parametrize("module_name", MLIP_MODULE_NAMES)
def test_distributed_error_polling_continues_after_local_work_is_exhausted(
    module_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A shorter rank must keep polling while peers finish their workloads."""
    mlip_module = importlib.import_module(module_name)
    examples = [{"example_id": "finite", "extra_info": {}}]
    collated = {
        "feats": {"batch_atom_lvl": torch.tensor([0])},
        "ground_truth": {
            "energy_gt_B": torch.tensor([0.0]),
            "energy_valid_mask_B": torch.tensor([True]),
            "force_gt_D_L": torch.zeros(1, 1, 3),
            "force_valid_mask_L": torch.tensor([True]),
            "source_atom_count_B": torch.tensor([1]),
        },
    }

    class PredictionModel:
        calls = 0

        def forward(self, **kwargs):
            del kwargs
            self.calls += 1
            return {
                "energy_B": torch.tensor([0.0]),
                "energy_reconstruction_valid_B": torch.tensor([True]),
                "forces_L": torch.zeros(1, 3),
            }

    class UnevenWorkFabric:
        global_rank = 0
        world_size = 2
        device = torch.device("cpu")
        strategy = None

        def __init__(self):
            self.all_gather_calls = 0
            self.barrier_calls = 0

        def all_gather(self, value):
            self.all_gather_calls += 1
            if self.all_gather_calls == 1:
                assert value.item() == 1
                return torch.tensor([[1], [3]], dtype=torch.int64)
            assert value.item() == 0
            return torch.tensor([[0], [0]], dtype=torch.int64)

        @staticmethod
        def broadcast(local_detail, *, src):
            raise AssertionError((local_detail, src))

        @staticmethod
        def to_device(value):
            return value

        def barrier(self):
            self.barrier_calls += 1

    model = PredictionModel()
    fabric = UnevenWorkFabric()
    trainer = SimpleNamespace(
        fabric=fabric,
        state={"model": model},
        _assemble_network_inputs=lambda batch: batch,
        _single_view_property_input=lambda network_input: network_input,
    )
    monkeypatch.setattr(
        mlip_module,
        "_build_dataset",
        lambda train_cfg, domain, path, max_atoms: examples,
    )
    monkeypatch.setattr(
        mlip_module,
        "_make_rank_local_inference_model",
        lambda model, strategy, world_size: model,
    )
    monkeypatch.setattr(mlip_module, "SparseCollate", lambda: lambda batch: collated)
    evaluation_cfg = OmegaConf.create(
        {
            "max_system_atoms": 32,
            "max_examples_per_domain": None,
            "part_size": 8,
            "resume": False,
            "batch_size": 1,
            "num_workers": 0,
            "prefetch_factor": 2,
            "progress_every": 1,
        }
    )

    dataset_size = mlip_module._evaluate_domain(
        engine=SimpleNamespace(trainer=trainer),
        train_cfg=OmegaConf.create({}),
        domain="omol25",
        dataset_path="unused",
        output_dir=tmp_path,
        evaluation_cfg=evaluation_cfg,
    )

    assert dataset_size == 1
    assert model.calls == 1
    assert fabric.all_gather_calls == 4
    assert fabric.barrier_calls == 1
    assert len(list((tmp_path / "predictions").rglob("part-*.parquet"))) == 1


def test_prediction_row_subtracts_large_energies_in_float64():
    """Serialized benchmark errors retain sub-ULP residual energy differences."""
    baseline = -1_000_000_000.0
    rows = _prediction_rows(
        examples=[{"example_id": "large-energy", "extra_info": {}}],
        global_indices=[0],
        prediction={
            "energy_B": torch.tensor([[baseline + 0.25]], dtype=torch.float64),
            "energy_reconstruction_valid_B": torch.tensor([[True]]),
            "forces_L": torch.zeros(1, 3),
        },
        collated={
            "feats": {"batch_atom_lvl": torch.tensor([0])},
            "ground_truth": {
                "energy_gt_B": torch.tensor([baseline], dtype=torch.float64),
                "energy_valid_mask_B": torch.tensor([True]),
                "force_gt_D_L": torch.zeros(1, 1, 3),
                "force_valid_mask_L": torch.tensor([True]),
                "source_atom_count_B": torch.tensor([1]),
            },
        },
    )

    assert rows[0]["energy_absolute_error_eV"] == 0.25
    assert rows[0]["energy_per_atom_absolute_error_eV"] == 0.25


def test_aggregate_prediction_parts_builds_omol_paper_group(tmp_path: Path):
    prediction_dir = tmp_path / "omol25"
    part_dir = prediction_dir / "rank-00000"
    part_dir.mkdir(parents=True)
    rows = [_row(data_id="biomolecules"), _row(data_id="elytes")]
    rows[1]["global_index"] = 1
    pq.write_table(
        pa.Table.from_pylist(rows, schema=_PREDICTION_SCHEMA),
        part_dir / "part-000000.parquet",
    )

    metrics = aggregate_prediction_parts(prediction_dir)

    assert metrics["all"]["num_examples"] == 2
    assert metrics["biomolecules"]["num_examples"] == 1
    assert metrics["electrolytes"]["num_examples"] == 1


def test_reference_rows_include_allscaip_validation_results():
    rows = [row for row in _reference_rows() if row["model"].startswith("AllScAIP")]
    by_key = {(row["benchmark"], row["subset"], row["model"]): row for row in rows}

    assert len(rows) == 29
    assert by_key[
        (
            "omol25_4m_val",
            "all",
            "AllScAIP-md-ft-cons. (OMol25-4M)",
        )
    ] == {
        "benchmark": "omol25_4m_val",
        "subset": "all",
        "model": "AllScAIP-md-ft-cons. (OMol25-4M)",
        "energy_per_atom_mae_meV": None,
        "energy_mae_meV": 44.40,
        "force_mae_meV_per_angstrom": 7.51,
        "source": "AllScAIP arXiv:2603.06567v1 Table 7 (OMol25 validation)",
    }
    assert (
        by_key[
            (
                "omol25_4m_val",
                "metal_complexes",
                "AllScAIP-md-d. (OMol25-102M)",
            )
        ]["energy_mae_meV"]
        == 105.59
    )
    assert by_key[
        (
            "omat24_1m_val",
            "all",
            "AllScAIP-md-ft-cons.",
        )
    ] == {
        "benchmark": "omat24_1m_val",
        "subset": "all",
        "model": "AllScAIP-md-ft-cons.",
        "energy_per_atom_mae_meV": 10.7,
        "energy_mae_meV": None,
        "force_mae_meV_per_angstrom": 54.3,
        "source": "AllScAIP arXiv:2603.06567v1 Table 8 (OMat24 validation)",
    }


def test_resolve_aselmdb_paths_ignores_macos_resource_forks(tmp_path: Path):
    (tmp_path / "data.aselmdb").touch()
    (tmp_path / "._data.aselmdb").touch()

    assert resolve_aselmdb_paths(tmp_path) == [(tmp_path / "data.aselmdb").resolve()]


def _minimal_dataset_train_config(
    *,
    loader: dict,
    validation_sidecar_path: Path | None = None,
):
    config = {
        "datasets": {
            "train": {
                "omol25": {
                    "dataset": {
                        "paths": ["unused-training-path"],
                        "name": "omol25-training",
                        "loader": loader,
                        "transform": {
                            "train_conditions": {
                                "unconditional": {"frequency": 0.5},
                                "mlip": {"frequency": 0.5},
                            },
                            "meta_conditioning_probabilities": {"example": 0.5},
                        },
                    }
                }
            },
            "val": {},
        }
    }
    if validation_sidecar_path is not None:
        config["datasets"]["val"]["omol25_structure_prediction"] = {
            "dataset": {"bond_sidecar_path": str(validation_sidecar_path)}
        }
    return OmegaConf.create(config)


def _minimal_manifest_run_config(tmp_path: Path):
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.write_bytes(b"checkpoint")
    dataset_dir = tmp_path / "validation"
    dataset_dir.mkdir()
    (dataset_dir / "data.aselmdb").touch()
    return OmegaConf.create(
        {
            "ckpt_path": str(checkpoint_path),
            "out_dir": str(tmp_path / "output"),
            "use_ema_weights": True,
            "evaluation": {
                "domains": ["omol25"],
                "datasets": {"omol25": {"path": str(dataset_dir)}},
                "batch_size": 1,
                "num_workers": 0,
                "prefetch_factor": 2,
                "part_size": 8,
                "max_system_atoms": 192,
                "max_examples_per_domain": None,
                "resume": True,
                "progress_every": 100,
            },
        }
    )


@pytest.mark.parametrize("module_name", MLIP_MODULE_NAMES)
def test_resume_manifest_rejects_prediction_identity_changes(
    module_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Stale parts cannot survive config, preprocessing, or source changes."""
    mlip_module = importlib.import_module(module_name)
    config = _minimal_manifest_run_config(tmp_path)
    train_config = _minimal_dataset_train_config(
        loader={"_target_": "atomworks.ml.datasets.loaders.create_ase_atoms_loader"}
    )
    output_dir = Path(config.out_dir)
    output_dir.mkdir()
    source_identity = {
        "algorithm": "sha256",
        "sha256": "source-v1",
        "packages": {},
    }
    monkeypatch.setattr(
        evaluation_manifest,
        "source_tree_fingerprint",
        lambda package_names: dict(source_identity),
    )
    monkeypatch.setattr(
        evaluation_manifest,
        "_runtime_versions",
        lambda: {"python": "test-runtime"},
    )

    def write_manifest():
        mlip_module._write_manifest(
            output_dir,
            cfg=config,
            train_cfg=train_config,
            rank=0,
            world_size=1,
        )

    write_manifest()
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["manifest_version"] == 2
    assert manifest["prediction_identity"]["source"]["sha256"] == "source-v1"

    # Resume-throughput and logging options do not change predictions.
    config.evaluation.num_workers = 3
    config.evaluation.prefetch_factor = 7
    config.evaluation.progress_every = 1
    write_manifest()

    config.evaluation.batch_size = 2
    with pytest.raises(ValueError, match="prediction-affecting configuration"):
        write_manifest()
    config.evaluation.batch_size = 1

    train_config.datasets.train.omol25.dataset.transform.new_preprocessing_flag = True
    with pytest.raises(ValueError, match="prediction-affecting configuration"):
        write_manifest()
    del train_config.datasets.train.omol25.dataset.transform.new_preprocessing_flag

    source_identity["sha256"] = "source-v2"
    with pytest.raises(ValueError, match="prediction-affecting configuration"):
        write_manifest()


@pytest.mark.parametrize("module_name", MLIP_MODULE_NAMES)
def test_build_dataset_replaces_training_bond_sidecar_for_omol25_validation(
    module_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    validation_shard = tmp_path / "data.aselmdb"
    validation_shard.touch()
    validation_sidecar = tmp_path / "val_all_bonds.lmdb"
    train_cfg = _minimal_dataset_train_config(
        loader={
            "_target_": "nucleus.utils.bond_sidecar.create_bond_sidecar_ase_atoms_loader",
            "bond_sidecar_path": str(tmp_path / "train_4M_all_bonds.lmdb"),
        },
        validation_sidecar_path=validation_sidecar,
    )
    instantiated = {}

    def capture_dataset_config(dataset_cfg):
        instantiated["config"] = dataset_cfg
        return dataset_cfg

    mlip_module = importlib.import_module(module_name)
    monkeypatch.setattr(mlip_module.hydra.utils, "instantiate", capture_dataset_config)

    mlip_module._build_dataset(train_cfg, "omol25", str(tmp_path), max_atoms=192)

    dataset_cfg = instantiated["config"]
    assert dataset_cfg.loader.bond_sidecar_path == str(validation_sidecar)
    assert dataset_cfg.paths == [str(validation_shard.resolve())]


@pytest.mark.parametrize("module_name", MLIP_MODULE_NAMES)
def test_build_dataset_preserves_loader_without_bond_sidecar(
    module_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    validation_shard = tmp_path / "data.aselmdb"
    validation_shard.touch()
    train_cfg = _minimal_dataset_train_config(
        loader={"_target_": "atomworks.ml.datasets.loaders.create_ase_atoms_loader"}
    )
    instantiated = {}

    def capture_dataset_config(dataset_cfg):
        instantiated["config"] = dataset_cfg
        return dataset_cfg

    mlip_module = importlib.import_module(module_name)
    monkeypatch.setattr(mlip_module.hydra.utils, "instantiate", capture_dataset_config)

    mlip_module._build_dataset(train_cfg, "omol25", str(tmp_path), max_atoms=192)

    assert "bond_sidecar_path" not in instantiated["config"].loader


@pytest.mark.parametrize("training", [False, True])
def test_rank_local_inference_model_bypasses_distributed_forward_target(training):
    original_model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Dropout())
    strategy = SingleDeviceStrategy(device="cpu")
    distributed_forward_target = torch.nn.Sequential(original_model)
    fabric_model = _FabricModule(
        forward_module=distributed_forward_target,
        strategy=strategy,
        original_module=original_model,
    )
    fabric_model.train(training)
    # Rewrapping must also preserve intentionally different child modes.
    original_model[-1].eval()
    original_modes = [module.training for module in original_model.modules()]

    local_model = _make_rank_local_inference_model(
        fabric_model,
        strategy=strategy,
        world_size=16,
    )

    assert isinstance(local_model, _FabricModule)
    assert local_model.module is original_model
    assert local_model._forward_module is original_model
    assert local_model._forward_module is not distributed_forward_target
    assert local_model.training is training
    assert fabric_model.training is training
    assert [module.training for module in original_model.modules()] == original_modes


def test_rank_local_inference_model_preserves_eval_ema_dispatch():
    original_model = EMA(torch.nn.Linear(2, 1, bias=False), decay=0.9)
    with torch.no_grad():
        original_model.model.weight.fill_(2.0)
        original_model.shadow.weight.fill_(3.0)
    strategy = SingleDeviceStrategy(device="cpu")
    fabric_model = _FabricModule(
        forward_module=torch.nn.Sequential(original_model),
        strategy=strategy,
        original_module=original_model,
    ).eval()

    local_model = _make_rank_local_inference_model(
        fabric_model, strategy=strategy, world_size=16
    )

    assert not local_model.training
    assert not original_model.training
    torch.testing.assert_close(
        local_model(torch.tensor([[1.0, 2.0]])), torch.tensor([[9.0]]), rtol=0, atol=0
    )


def test_single_rank_inference_preserves_existing_fabric_model():
    original_model = torch.nn.Linear(2, 1)
    strategy = SingleDeviceStrategy(device="cpu")
    fabric_model = _FabricModule(original_model, strategy=strategy)

    assert (
        _make_rank_local_inference_model(
            fabric_model,
            strategy=strategy,
            world_size=1,
        )
        is fabric_model
    )
