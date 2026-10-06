import builtins
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch
from atomworks.io.utils.bonds import has_ambiguous_annotation_set
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.utils.token import get_token_starts
from biotite import structure as struc
from hydra import compose, initialize_config_dir
from nucleus.inference_input_parsing import (
    ZatomDesignInputSpecification,
    _coerce_zatom_design_specification,
    accumulate_components_with_molecules,
    create_diffused_molecule_atoms,
    install_molecule_contig_support,
    uniquify_atom_names_for_cif,
)
from nucleus.metrics.molecule_generation import _atom_array_to_mol
from nucleus.transforms.molecule_transforms import AddMeanForceNormConditioning
from nucleus.utils.batching import SparseCollate
from nucleus.utils.datasets import _coerce_ligand_design_specification
from nucleus.utils.torch import expand_t_to_level
from omegaconf import OmegaConf
from rfd3.inference.input_parsing import DesignInputSpecification
from rfd3.utils.inference import set_indices
from rfd3.utils.io import build_stack_from_atom_array_and_batched_coords

import zatom2.run_inference as zatom2_run_inference
from zatom2.model.Zatom2 import Zatom2
from zatom2.run_inference import (
    BondSidecarPipeline,
    _build_inference_transform_from_checkpoint_cfg,
)

zatom2_model_module = importlib.import_module("zatom2.model.Zatom2")


def test_serial_inference_does_not_import_optional_batching(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "zatom2.inference_batching":
            pytest.fail("Serial inference imported optional batching helpers")
        return original_import(name, *args, **kwargs)

    engine = object.__new__(zatom2_run_inference.Zatom2InferenceEngine)
    engine.inference_batch_size = 1
    engine.inference_num_workers = 0
    specs = {"example": {}}
    expected = {"example": ["output"]}

    def serial_forward(self, supplied):
        assert self is engine
        assert supplied is specs
        return expected

    monkeypatch.setattr(
        zatom2_run_inference.RFD3InferenceEngine, "_run_multi", serial_forward
    )
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert engine._run_multi(specs) is expected


@pytest.mark.parametrize(("batch_size", "num_workers"), [(2, 0), (1, 1)])
def test_parallel_inference_reports_missing_batching_module(
    monkeypatch, batch_size, num_workers
):
    original_import = builtins.__import__

    def missing_batching(name, *args, **kwargs):
        if name == "zatom2.inference_batching":
            raise ModuleNotFoundError("missing batching helpers", name=name)
        return original_import(name, *args, **kwargs)

    engine = object.__new__(zatom2_run_inference.Zatom2InferenceEngine)
    engine.inference_batch_size = batch_size
    engine.inference_num_workers = num_workers
    monkeypatch.setattr(builtins, "__import__", missing_batching)
    with pytest.raises(RuntimeError, match="inference_batch_size=1"):
        engine._run_multi({})


class _FakeCalc:
    def __init__(self, forces):
        self.results = {"forces": forces}


class _FakeASEAtoms:
    def __init__(self, forces):
        self.calc = _FakeCalc(forces)


class _FakeASEAtomsWithoutForces:
    calc = None

    def get_forces(self):
        raise RuntimeError("no calculator")


@pytest.mark.parametrize(
    ("specification", "expected_example_id"),
    [
        ({"example_id": "source-example"}, "source-example"),
        ({}, "outer-example"),
    ],
)
def test_bond_sidecar_pipeline_attaches_before_transform(
    monkeypatch,
    specification,
    expected_example_id,
):
    calls = []

    class FakeBondSidecarLoader:
        def __init__(self, *, bond_sidecar_path):
            calls.append(("init", bond_sidecar_path))

        def attach_bonds(self, atom_array, *, example_id):
            calls.append(("attach", example_id))
            atom_array.bonds_attached = True
            return True

    monkeypatch.setattr(
        zatom2_run_inference,
        "BondSidecarASEAtomsLoader",
        FakeBondSidecarLoader,
    )

    def transform(data):
        assert data["atom_array"].bonds_attached is True
        assert data["has_disconnected_fragments"] is True
        calls.append(("transform", None))
        return data

    atom_array = type("AtomArrayStub", (), {})()
    pipeline = BondSidecarPipeline(
        transform,
        bond_sidecar_path="validation-bonds.lmdb",
    )
    output = pipeline(
        {
            "atom_array": atom_array,
            "example_id": "outer-example",
            "specification": specification,
        }
    )

    assert output["atom_array"] is atom_array
    assert calls == [
        ("init", "validation-bonds.lmdb"),
        ("attach", expected_example_id),
        ("transform", None),
    ]


def test_inference_config_wires_local_paths_outside_rfd3_config(tmp_path):
    config_dir = Path(__file__).parents[1] / "configs"
    sidecar_path = tmp_path / "validation-bonds.lmdb"
    statistics_path = tmp_path / "statistics.json"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(
            config_name="inference",
            overrides=[
                f"bond_sidecar_path={sidecar_path}",
                f"property_statistics_path={statistics_path}",
                f"ckpt_path={tmp_path / 'model.ckpt'}",
                f"inputs={tmp_path / 'inputs.json'}",
                f"out_dir={tmp_path / 'outputs'}",
            ],
        )

    engine = zatom2_run_inference.build_inference_engine(config)

    assert engine.bond_sidecar_path == str(sidecar_path)
    assert engine.property_statistics_path == str(statistics_path)


@pytest.mark.parametrize("enabled", [True, False, None])
@pytest.mark.parametrize("use_override", [True, False])
def test_inference_property_statistics_path_override(tmp_path, enabled, use_override):
    original_path = "/private/training/statistics.json"
    local_path = tmp_path / "statistics.json"
    cfg = OmegaConf.create(
        {
            "trainer": {"accelerator": "cpu"},
            "model": {
                "net": {
                    "diffusion_module": {
                        "n_recycle": 3,
                        "atomistic_property_head": (
                            {
                                "enabled": enabled,
                                "statistics_path": original_path,
                                "energy_hidden_dim": 768,
                            }
                            if enabled is not None
                            else None
                        ),
                    }
                }
            },
        }
    )
    OmegaConf.set_struct(cfg.model.net.diffusion_module, True)
    engine = zatom2_run_inference.Zatom2InferenceEngine(
        **zatom2_run_inference.RFD3InferenceConfig(
            ckpt_path=str(tmp_path / "model.ckpt")
        ),
        property_statistics_path=local_path if use_override else None,
    )

    updated = engine._override_checkpoint_config(cfg)

    assert updated.model.net.diffusion_module.n_recycle == 3
    head = updated.model.net.diffusion_module.atomistic_property_head
    if enabled is None:
        assert head is None
    else:
        expected_path = str(local_path) if enabled and use_override else original_path
        assert head.statistics_path == expected_path
        assert head.enabled is enabled
        assert head.energy_hidden_dim == 768
        assert (
            cfg.model.net.diffusion_module.atomistic_property_head.statistics_path
            == original_path
        )


@pytest.mark.parametrize("existing", [True, False])
def test_published_property_statistics_relocate_only_missing_paths(tmp_path, existing):
    saved_path = tmp_path / "atomistic_property_statistics.v1.json"
    if existing:
        saved_path.write_text("{}")
    cfg = OmegaConf.create(
        {
            "trainer": {"accelerator": "cpu"},
            "model": {
                "net": {
                    "diffusion_module": {
                        "n_recycle": 3,
                        "atomistic_property_head": {
                            "enabled": True,
                            "statistics_path": str(saved_path),
                        },
                    }
                }
            },
        }
    )
    engine = zatom2_run_inference.Zatom2InferenceEngine(
        **zatom2_run_inference.RFD3InferenceConfig(
            ckpt_path=str(tmp_path / "model.ckpt")
        )
    )
    updated = engine._override_checkpoint_config(cfg)
    path = Path(
        updated.model.net.diffusion_module.atomistic_property_head.statistics_path
    )
    expected = (
        saved_path if existing else Path(__file__).parents[3] / "data" / saved_path.name
    )
    assert path == expected
    assert path.is_file()
    assert (
        cfg.model.net.diffusion_module.atomistic_property_head.statistics_path
        == str(saved_path)
    )


@pytest.mark.parametrize(
    ("filename", "domain", "count"),
    [
        ("demo.json", "molecule", (7, 9)),
        ("material.json", "material", (12, 12)),
        ("protein.json", "protein", (100, 100)),
    ],
)
def test_public_quickstart_examples_generate_the_documented_domain(
    tmp_path, filename, domain, count
):
    install_molecule_contig_support()
    engine = zatom2_run_inference.Zatom2InferenceEngine(
        **zatom2_run_inference.RFD3InferenceConfig(
            ckpt_path=str(tmp_path / "model.ckpt")
        )
    )
    examples = Path(__file__).parents[1] / "docs/examples" / filename
    for spec in engine._canonicalize_inputs(str(examples)).values():
        atoms = spec.build()
        if domain == "protein":
            assert struc.filter_amino_acids(atoms).all()
            token_count = len(struc.get_residue_starts(atoms))
        else:
            assert (atoms.res_name == "LIG").all()
            assert atoms.atomize.all()
            assert (atoms.is_material == (domain == "material")).all()
            token_count = len(get_token_starts(atoms))
        assert count[0] <= token_count <= count[1]


@pytest.mark.parametrize("has_validation", [True, False])
@pytest.mark.parametrize(
    ("training_tasks", "transform_overrides", "expected_tasks"),
    [
        (
            ["unconditional", "structure_prediction"],
            {},
            {"unconditional", "structure_prediction"},
        ),
        ([], {}, set()),
        (["unconditional"], {}, {"unconditional"}),
        (None, {}, {"unconditional", "structure_prediction", "mlip"}),
        (
            ["unconditional", "structure_prediction"],
            {"mean_force_norm_conditioning_tasks": ["structure_prediction"]},
            {"structure_prediction"},
        ),
        (
            ["unconditional", "structure_prediction"],
            {"mean_force_norm_conditioning_tasks": []},
            set(),
        ),
    ],
)
def test_inference_force_scope_uses_training_defaults_and_explicit_overrides(
    monkeypatch,
    tmp_path,
    has_validation,
    training_tasks,
    transform_overrides,
    expected_tasks,
):
    """A joint checkpoint's first validation loader must not disable a task."""
    cfg = OmegaConf.create(
        {
            "datasets": {
                "global_transform_args": {
                    "mean_force_norm_conditioning_tasks": training_tasks,
                },
                "val": (
                    {
                        "unconditional": None,
                        "omol25_generation": {
                            "dataset": {
                                "transform": {
                                    "mean_force_norm_conditioning_tasks": [
                                        "unconditional"
                                    ],
                                }
                            }
                        },
                    }
                    if has_validation
                    else None
                ),
            },
        }
    )

    # Exercise the real engine config merge and force transform without loading
    # a checkpoint, running the denoiser, or constructing unrelated transforms.
    monkeypatch.setattr(
        zatom2_run_inference.hydra.utils,
        "instantiate",
        lambda config: AddMeanForceNormConditioning(
            conditioning_tasks=config.get("mean_force_norm_conditioning_tasks"),
        ),
    )
    engine = zatom2_run_inference.Zatom2InferenceEngine(
        **zatom2_run_inference.RFD3InferenceConfig(
            ckpt_path=str(tmp_path / "model.ckpt")
        ),
        transform_overrides=transform_overrides,
    )
    engine._construct_pipeline(cfg)

    for task in ("unconditional", "structure_prediction", "mlip"):
        for requested in (True, False):
            atom_array = create_diffused_molecule_atoms(3)
            atom_array.set_annotation(
                "is_motif_atom_with_fixed_seq",
                np.full(3, task != "unconditional", dtype=bool),
            )
            atom_array.set_annotation(
                "is_motif_atom_with_fixed_coord",
                np.full(3, task == "mlip", dtype=bool),
            )
            output = engine.pipeline(
                {
                    "atom_array": atom_array,
                    "is_inference": True,
                    "specification": {
                        "extra": {
                            "mean_force_norm": 10.0,
                            "has_mean_force_norm": requested,
                        }
                    },
                }
            )
            enabled = requested and task in expected_tasks
            expected = torch.full((3, 1), float(enabled))
            torch.testing.assert_close(output["feats"]["mean_force_norm"], expected)
            torch.testing.assert_close(output["feats"]["has_mean_force_norm"], expected)


@pytest.mark.parametrize("global_transform_args", [None, {}])
def test_inference_preserves_legacy_validation_force_scope(
    monkeypatch, tmp_path, global_transform_args
):
    cfg = OmegaConf.create(
        {
            "datasets": {
                "global_transform_args": global_transform_args,
                "val": {
                    "omol25_generation": {
                        "dataset": {
                            "transform": {
                                "mean_force_norm_conditioning_tasks": ["unconditional"],
                            }
                        }
                    },
                },
            },
        }
    )
    instantiated_configs = []
    monkeypatch.setattr(
        zatom2_run_inference.hydra.utils,
        "instantiate",
        instantiated_configs.append,
    )
    engine = zatom2_run_inference.Zatom2InferenceEngine(
        **zatom2_run_inference.RFD3InferenceConfig(
            ckpt_path=str(tmp_path / "model.ckpt")
        ),
    )

    engine._construct_pipeline(cfg)

    assert instantiated_configs[0].mean_force_norm_conditioning_tasks == [
        "unconditional"
    ]


def test_cfg_feature_stripping_preserves_non_tensor_metadata():
    features = {
        "is_motif_token_unindexed": torch.zeros(2, dtype=torch.bool),
        "is_motif_atom_unindexed": torch.zeros(3, dtype=torch.bool),
        "mean_force_norm": torch.ones((2, 1)),
        "metadata": ["example"],
    }

    stripped = Zatom2._strip_cfg_features(features, ["mean_force_norm"])

    torch.testing.assert_close(stripped["mean_force_norm"], torch.zeros((2, 1)))
    assert stripped["metadata"] is features["metadata"]


def test_force_only_cfg_config_enables_named_reference_mode(tmp_path):
    config_dir = Path(__file__).parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(
            config_name="inference",
            overrides=[
                "classifier_free_guidance=force_only",
                f"ckpt_path={tmp_path / 'model.ckpt'}",
                f"inputs={tmp_path / 'inputs.json'}",
                f"out_dir={tmp_path / 'outputs'}",
            ],
        )

    assert config.inference_sampler.use_classifier_free_guidance is True
    assert config.inference_sampler.cfg_feature_mode == "force_only"


@pytest.mark.parametrize(
    ("cfg_scale", "expected_enabled"),
    [(1.0, False), (1.5, True)],
)
def test_force_only_cfg_constructor_resolves_sampler_config(
    monkeypatch,
    cfg_scale,
    expected_enabled,
):
    sampler_config = {
        "kind": "default",
        "use_classifier_free_guidance": True,
        "cfg_scale": cfg_scale,
        "cfg_feature_mode": "force_only",
        "cfg_features": ["active_donor", "ref_atomwise_rasa"],
    }
    captured_sampler_config = {}

    monkeypatch.setattr(
        zatom2_model_module.hydra.utils,
        "instantiate",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        zatom2_model_module,
        "ZatomConditionalDiffusionSampler",
        lambda **kwargs: captured_sampler_config.update(kwargs),
    )

    model = Zatom2(
        c_s=1,
        c_z=1,
        c_atom=1,
        token_initializer={},
        diffusion_module={},
        inference_sampler=sampler_config,
    )

    assert model.use_classifier_free_guidance is expected_enabled
    assert model.cfg_feature_mode == "force_only"
    assert model.cfg_features == [
        "mean_force_norm",
        "has_mean_force_norm",
        "force_rms_p95",
        "has_force_rms_p95",
    ]
    assert captured_sampler_config["use_classifier_free_guidance"] is expected_enabled
    assert "cfg_feature_mode" not in captured_sampler_config
    assert "cfg_features" not in captured_sampler_config
    assert sampler_config["cfg_feature_mode"] == "force_only"
    assert sampler_config["cfg_features"] == ["active_donor", "ref_atomwise_rasa"]


def test_force_only_cfg_reference_preserves_all_non_force_conditioning():
    features = {
        # An unindexed motif triggers cropping in the legacy CFG helper. Force-only
        # CFG must preserve it and every other non-force condition exactly.
        "is_motif_token_unindexed": torch.tensor([False, True]),
        "is_motif_atom_unindexed": torch.tensor([False, False, True]),
        "mean_force_norm": torch.full((2, 1), 1.25),
        "has_mean_force_norm": torch.ones((2, 1)),
        "force_rms_p95": torch.full((2, 2), 2.5),
        "has_force_rms_p95": torch.ones((2, 1)),
        "total_charge": torch.full((2, 1), -1.0),
        "spin_multiplicity": torch.full((2, 1), 2.0),
        "ref_element": torch.arange(6).reshape(3, 2),
        "token_bonds": torch.eye(2),
        "task_condition": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        "metadata": ["example"],
    }
    cfg_features = Zatom2._resolve_cfg_features(
        "force_only",
        ["active_donor", "ref_atomwise_rasa"],
    )

    reference = Zatom2._build_cfg_reference_features(
        features,
        cfg_feature_mode="force_only",
        cfg_features=cfg_features,
    )

    for feature_name in (
        "mean_force_norm",
        "has_mean_force_norm",
        "force_rms_p95",
        "has_force_rms_p95",
    ):
        torch.testing.assert_close(
            reference[feature_name], torch.zeros_like(features[feature_name])
        )
    for feature_name in (
        "is_motif_token_unindexed",
        "is_motif_atom_unindexed",
        "total_charge",
        "spin_multiplicity",
        "ref_element",
        "token_bonds",
        "task_condition",
    ):
        assert reference[feature_name] is features[feature_name]
    assert reference["metadata"] is features["metadata"]


def test_cfg_feature_mode_rejects_unknown_mode():
    with pytest.raises(ValueError, match="configured.*force_only"):
        Zatom2._resolve_cfg_features("all_but_force", [])


def test_diffused_molecule_atoms_match_omol25_single_residue_layout():
    """
    Test that the diffused molecule atoms have the same layout as an
    OMol25 single residue, which is what the model was trained on.
    """
    atom_array = create_diffused_molecule_atoms(7)
    atom_array = set_indices(
        array=atom_array,
        chain="A",
        res_id_start=3,
        molecule_id=0,
        component="7m",
    )

    assert np.all(atom_array.res_id == 3)
    assert np.all(atom_array.res_name == "LIG")
    assert np.all(atom_array.atomize)
    assert len(get_token_starts(atom_array)) == 7


def test_mean_force_norm_conditioning_broadcasts_to_molecule_tokens():
    atom_array = create_diffused_molecule_atoms(4)
    forces = np.array(
        [
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 12.0],
            [0.0, 0.0, 0.0],
        ]
    )

    data = AddMeanForceNormConditioning()(
        {"atom_array": atom_array, "atoms": _FakeASEAtoms(forces)}
    )

    expected = torch.full(
        (4, 1),
        np.log10((5.0 + 12.0 + 0.0) / 3.0),
        dtype=torch.float32,
    )
    assert torch.allclose(data["feats"]["mean_force_norm"], expected)
    assert torch.allclose(data["feats"]["has_mean_force_norm"], torch.ones((4, 1)))


def test_rms_and_p95_conditioning_broadcasts_transformed_vector():
    atom_array = create_diffused_molecule_atoms(4)
    forces = np.array(
        [
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 12.0],
            [0.0, 0.0, 0.0],
        ]
    )

    data = AddMeanForceNormConditioning(conditioning_method="rms_and_p95")(
        {"atom_array": atom_array, "atoms": _FakeASEAtoms(forces)}
    )

    force_rms = np.sqrt((5.0**2 + 12.0**2) / 3.0)
    force_p95 = 11.3
    expected = torch.log(
        torch.tensor([force_rms, force_p95], dtype=torch.float32) + 1e-8
    ).expand(4, -1)
    torch.testing.assert_close(data["feats"]["force_rms_p95"], expected)
    torch.testing.assert_close(data["feats"]["has_force_rms_p95"], torch.ones((4, 1)))


def test_mean_force_norm_conditioning_defaults_to_zero_without_forces():
    data = AddMeanForceNormConditioning()(
        {"atom_array": create_diffused_molecule_atoms(2)}
    )

    assert data["feats"]["mean_force_norm"].shape == (2, 1)
    assert torch.count_nonzero(data["feats"]["mean_force_norm"]).item() == 0
    assert data["feats"]["has_mean_force_norm"].shape == (2, 1)
    assert torch.count_nonzero(data["feats"]["has_mean_force_norm"]).item() == 0


def test_mean_force_norm_conditioning_probability_zero_uses_null_conditioning():
    atom_array = create_diffused_molecule_atoms(4)
    forces = np.array(
        [
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 12.0],
        ]
    )

    data = AddMeanForceNormConditioning(conditioning_probability=0.0)(
        {"atom_array": atom_array, "atoms": _FakeASEAtoms(forces)}
    )

    assert data["feats"]["mean_force_norm"].shape == (4, 1)
    assert torch.count_nonzero(data["feats"]["mean_force_norm"]).item() == 0
    assert data["feats"]["has_mean_force_norm"].shape == (4, 1)
    assert torch.count_nonzero(data["feats"]["has_mean_force_norm"]).item() == 0


def test_mean_force_norm_conditioning_tolerates_atoms_without_forces():
    data = AddMeanForceNormConditioning()(
        {
            "atom_array": create_diffused_molecule_atoms(2),
            "atoms": _FakeASEAtomsWithoutForces(),
        }
    )

    assert torch.count_nonzero(data["feats"]["mean_force_norm"]).item() == 0
    assert torch.count_nonzero(data["feats"]["has_mean_force_norm"]).item() == 0


def test_mean_force_norm_conditioning_spec_scalar_defaults_to_null():
    data = AddMeanForceNormConditioning()(
        {
            "atom_array": create_diffused_molecule_atoms(3),
            "specification": {"mean_force_norm": 2.0},
        }
    )

    assert torch.count_nonzero(data["feats"]["mean_force_norm"]).item() == 0
    assert torch.count_nonzero(data["feats"]["has_mean_force_norm"]).item() == 0


def test_mean_force_norm_conditioning_uses_enabled_spec_scalar():
    data = AddMeanForceNormConditioning()(
        {
            "atom_array": create_diffused_molecule_atoms(3),
            "specification": {
                "mean_force_norm": 2.0,
                "has_mean_force_norm": True,
            },
        }
    )

    assert torch.allclose(
        data["feats"]["mean_force_norm"],
        torch.full((3, 1), np.log10(2.0), dtype=torch.float32),
    )
    assert torch.allclose(data["feats"]["has_mean_force_norm"], torch.ones((3, 1)))


def test_mean_force_norm_conditioning_uses_enabled_extra_spec_scalar():
    data = AddMeanForceNormConditioning()(
        {
            "atom_array": create_diffused_molecule_atoms(3),
            "specification": {
                "extra": {
                    "mean_force_norm": 2.0,
                    "has_mean_force_norm": True,
                }
            },
        }
    )

    assert torch.allclose(
        data["feats"]["mean_force_norm"],
        torch.full((3, 1), np.log10(2.0), dtype=torch.float32),
    )
    assert torch.allclose(data["feats"]["has_mean_force_norm"], torch.ones((3, 1)))


def test_rms_and_p95_inference_conditioning_uses_raw_spec_values_deterministically():
    data = AddMeanForceNormConditioning(
        conditioning_probability=0.0,
        conditioning_method="rms_and_p95",
    )(
        {
            "atom_array": create_diffused_molecule_atoms(3),
            "is_inference": True,
            "specification": {
                "force_rms": 2.0,
                "force_p95": 4.0,
                "has_force_rms_p95": True,
            },
        }
    )

    expected = torch.log(torch.tensor([2.0, 4.0]) + 1e-8).expand(3, -1)
    torch.testing.assert_close(data["feats"]["force_rms_p95"], expected)
    torch.testing.assert_close(data["feats"]["has_force_rms_p95"], torch.ones((3, 1)))


def test_rms_and_p95_inference_conditioning_requires_both_statistics():
    transform = AddMeanForceNormConditioning(conditioning_method="rms_and_p95")

    with pytest.raises(ValueError, match="requires both force_rms and force_p95"):
        transform(
            {
                "atom_array": create_diffused_molecule_atoms(3),
                "is_inference": True,
                "specification": {
                    "force_rms": 2.0,
                    "has_force_rms_p95": True,
                },
            }
        )


def test_zatom_design_spec_accepts_mean_force_norm_field():
    spec = ZatomDesignInputSpecification.safe_init(
        length="3m",
        mean_force_norm=1.5,
    )

    assert spec.mean_force_norm == 1.5
    assert spec.has_mean_force_norm is False


def test_zatom_design_spec_accepts_has_mean_force_norm_field():
    spec = ZatomDesignInputSpecification.safe_init(
        length="3m",
        mean_force_norm=1.5,
        has_mean_force_norm=True,
    )

    assert spec.mean_force_norm == 1.5
    assert spec.has_mean_force_norm is True


def test_zatom_design_spec_accepts_electronic_conditioning_fields():
    spec = ZatomDesignInputSpecification.safe_init(
        length="3m",
        total_charge=-1,
        spin_multiplicity=2,
    )

    assert spec.total_charge == -1
    assert spec.spin_multiplicity == 2


def test_zatom_design_spec_accepts_rms_and_p95_fields():
    spec = ZatomDesignInputSpecification.safe_init(
        length="3m",
        force_rms=1.5,
        force_p95=2.5,
        has_force_rms_p95=True,
    )

    assert spec.force_rms == 1.5
    assert spec.force_p95 == 2.5
    assert spec.has_force_rms_p95 is True


def test_ligand_contig_dataset_coercion_uses_zatom_spec():
    spec = _coerce_ligand_design_specification(
        {
            "length": "3m",
            "mean_force_norm": 1.5,
        }
    )

    assert isinstance(spec, ZatomDesignInputSpecification)
    assert spec.mean_force_norm == 1.5


def test_zatom_design_spec_coercion_accepts_mean_force_norm_field():
    spec = _coerce_zatom_design_specification(
        {
            "length": "3m",
            "mean_force_norm": 1.5,
        }
    )

    assert isinstance(spec, ZatomDesignInputSpecification)
    assert spec.mean_force_norm == 1.5


def test_zatom_design_spec_build_metadata_includes_mean_force_norm():
    install_molecule_contig_support()
    spec = ZatomDesignInputSpecification.safe_init(
        length="3m",
        mean_force_norm=1.5,
        has_mean_force_norm=True,
    )

    _, metadata = spec.build(return_metadata=True)

    assert metadata["mean_force_norm"] == 1.5
    assert metadata["has_mean_force_norm"] is True


def test_zatom_design_spec_build_metadata_includes_electronic_conditioning():
    install_molecule_contig_support()
    spec = ZatomDesignInputSpecification.safe_init(
        length="3m",
        total_charge=-1,
        spin_multiplicity=2,
    )

    _, metadata = spec.build(return_metadata=True)

    assert metadata["total_charge"] == -1
    assert metadata["spin_multiplicity"] == 2


def test_zatom_design_spec_build_metadata_includes_rms_and_p95():
    install_molecule_contig_support()
    spec = ZatomDesignInputSpecification.safe_init(
        length="3m",
        force_rms=1.5,
        force_p95=2.5,
        has_force_rms_p95=True,
    )

    _, metadata = spec.build(return_metadata=True)

    assert metadata["force_rms"] == 1.5
    assert metadata["force_p95"] == 2.5
    assert metadata["has_force_rms_p95"] is True


def test_multiple_diffused_molecules_use_consecutive_residue_ids():
    """
    Test that when multiple diffused molecules are created, they are assigned
    consecutive residue IDs, which is important for the model to correctly
    infer the contig layout.
    """
    atom_array = accumulate_components_with_molecules(
        ["7m", "7m"],
        indexed_tokens={},
        unindexed_tokens={},
    )

    assert atom_array.res_id.tolist() == [1] * 7 + [2] * 7
    assert len(get_token_starts(atom_array)) == 14


def test_input_conditioned_molecule_design_uses_indexed_ligand_contig():
    """
    Test that fixed-structure atom-type design selects the input ligand residue,
    rather than creating fresh placeholder molecule tokens.
    """
    install_molecule_contig_support()
    atom_array = struc.array(
        [
            struc.Atom(
                [0.0, 0.0, 0.0],
                chain_id="A",
                res_id=1,
                res_name="LIG",
                atom_name="C1",
                element="C",
            ),
            struc.Atom(
                [1.0, 0.0, 0.0],
                chain_id="A",
                res_id=1,
                res_name="LIG",
                atom_name="N1",
                element="N",
            ),
            struc.Atom(
                [2.0, 0.0, 0.0],
                chain_id="A",
                res_id=1,
                res_name="LIG",
                atom_name="BR1",
                element="Br",
            ),
        ]
    )

    indexed_spec = DesignInputSpecification.safe_init(
        input=None,
        atom_array_input=atom_array,
        contig="A1",
        select_unfixed_sequence=True,
    )
    indexed = indexed_spec.build()

    assert indexed.element.tolist() == ["C", "N", "Br"]
    assert indexed.is_motif_atom_with_fixed_coord.tolist() == [True, True, True]
    assert indexed.is_motif_atom_with_fixed_seq.tolist() == [False, False, False]
    np.testing.assert_allclose(
        indexed.coord,
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )
    assert len(get_token_starts(indexed)) == 1

    uncentered_spec = DesignInputSpecification.safe_init(
        input=None,
        atom_array_input=atom_array,
        contig="A1",
        ori_token=[0, 0, 0],
        select_unfixed_sequence=True,
    )
    uncentered = uncentered_spec.build()

    np.testing.assert_allclose(
        uncentered.coord,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )

    placeholder_spec = DesignInputSpecification.safe_init(
        input=None,
        atom_array_input=atom_array,
        contig="3m",
        select_unfixed_sequence=True,
    )
    placeholder = placeholder_spec.build()

    assert placeholder.element.tolist() == ["C", "C", "C"]
    assert placeholder.is_motif_atom_with_fixed_coord.tolist() == [False, False, False]
    assert len(get_token_starts(placeholder)) == 3


def test_input_cif_hydrogens_reach_pipeline_input(tmp_path):
    """
    Test that input hydrogens are optionally not removed before the transform pipeline.
    """
    install_molecule_contig_support()
    atom_array = struc.array(
        [
            struc.Atom(
                [0.0, 0.0, 0.0],
                chain_id="A",
                res_id=1,
                res_name="LIG",
                atom_name="C1",
                element="C",
            ),
            struc.Atom(
                [1.0, 0.0, 0.0],
                chain_id="A",
                res_id=1,
                res_name="LIG",
                atom_name="H1",
                element="H",
            ),
        ]
    )
    atom_array.set_annotation("hetero", np.ones(2, dtype=bool))

    input_path = tmp_path / "ligand_with_hydrogen.cif"
    to_cif_file(atom_array, input_path, file_type="cif", include_entity_poly=False)

    spec = DesignInputSpecification.safe_init(
        input=str(input_path),
        contig="A1",
        ori_token=[0, 0, 0],
        select_unfixed_sequence=True,
    )
    built = spec.build()
    data = spec.to_pipeline_input("ligand_with_hydrogen")

    assert built.element.tolist() == ["C", "H"]
    assert data["atom_array"].element.tolist() == ["C", "H"]


def test_atom_array_to_mol_preserves_atoms_and_conformer():
    """
    Test that validation molecule metrics do not accept a zero-atom RDKit mol
    when the source AtomArray has valid coordinates for a molecule RDKit can
    perceive.
    """
    atom_array = struc.array(
        [
            struc.Atom(
                [0.0, 0.0, 0.0],
                chain_id="A",
                res_id=1,
                res_name="LIG",
                atom_name="C1",
                element="C",
            ),
            struc.Atom(
                [1.54, 0.0, 0.0],
                chain_id="A",
                res_id=1,
                res_name="LIG",
                atom_name="C2",
                element="C",
            ),
            struc.Atom(
                [2.94, 0.0, 0.0],
                chain_id="A",
                res_id=1,
                res_name="LIG",
                atom_name="O1",
                element="O",
            ),
            struc.Atom(
                [-0.63, 0.63, 0.63],
                chain_id="A",
                res_id=1,
                res_name="LIG",
                atom_name="H1",
                element="H",
            ),
            struc.Atom(
                [-0.63, -0.63, 0.63],
                chain_id="A",
                res_id=1,
                res_name="LIG",
                atom_name="H2",
                element="H",
            ),
            struc.Atom(
                [-0.63, 0.0, -0.89],
                chain_id="A",
                res_id=1,
                res_name="LIG",
                atom_name="H3",
                element="H",
            ),
            struc.Atom(
                [1.9, 0.89, 0.51],
                chain_id="A",
                res_id=1,
                res_name="LIG",
                atom_name="H4",
                element="H",
            ),
            struc.Atom(
                [1.9, -0.89, 0.51],
                chain_id="A",
                res_id=1,
                res_name="LIG",
                atom_name="H5",
                element="H",
            ),
            struc.Atom(
                [3.25, 0.8, 0.0],
                chain_id="A",
                res_id=1,
                res_name="LIG",
                atom_name="H6",
                element="H",
            ),
        ]
    )

    mol = _atom_array_to_mol(atom_array)

    assert mol is not None
    assert mol.GetNumAtoms() == len(atom_array)
    assert mol.GetNumBonds() > 0
    assert mol.GetNumConformers() == 1


def test_expand_t_to_level_uses_current_sampler_time():
    """
    Test that expand_t_to_level() correctly expands the
    current sampler time t to the correct shape for the model,
    and that it uses the current sampler time for each batch
    element when batch_key is provided.
    """
    t = torch.tensor([2.0, 3.0])
    expanded = expand_t_to_level(t, {}, level_len=4, batch_key="batch_atom_lvl")

    assert expanded.tolist() == [
        [2.0, 2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0, 3.0],
    ]


def test_overfit_inference_transform_uses_train_conformer_settings():
    """
    Test that no-val overfit inference preserves train transform conformer settings.
    """
    cfg = OmegaConf.create(
        {
            "datasets": {
                "pipeline_target": "nucleus.transforms.pipelines.build_atom14_base_pipeline",
                "diffusion_batch_size_train": 32,
                "randomly_augment": False,
                "enable_atomistic_property_targets": True,
                "global_transform_args": {
                    "central_atom": "CB",
                    "n_atoms_per_token": 14,
                    "association_scheme": "dense",
                    "center_option": "diffuse",
                    "generate_conformers": True,
                    "generate_conformers_for_non_protein_only": True,
                    "provide_reference_conformer_when_unmasked": True,
                    "ground_truth_conformer_policy": "IGNORE",
                    "provide_elements_for_unindexed_components": True,
                    "use_element_for_atom_names_of_atomized_tokens": True,
                },
                "train": {
                    "pdb": None,
                    "omol25": {
                        "dataset": {
                            "transform": {
                                "remove_hydrogens": False,
                                "generate_conformers": False,
                                "generate_conformers_for_non_protein_only": False,
                                "provide_reference_conformer_when_unmasked": False,
                            }
                        }
                    },
                },
            },
            "model": {
                "net": {
                    "diffusion_module": {"sigma_data": 16},
                    "token_initializer": {
                        "atom_1d_features": [],
                        "token_1d_features": [],
                    },
                }
            },
            "paths": {"data": {"residue_cache_dir": None}},
        }
    )

    transform = _build_inference_transform_from_checkpoint_cfg(cfg)

    assert transform.generate_conformers is False
    assert transform.generate_conformers_for_non_protein_only is False
    assert transform.provide_reference_conformer_when_unmasked is False
    assert transform.remove_hydrogens is False
    assert transform.enable_atomistic_property_targets is True


def test_expand_t_to_level_gathers_batched_training_times():
    """
    Test that expand_t_to_level() correctly gathers the
    batched training times when batch_key is provided.
    """
    t = torch.tensor([[2.0, 5.0], [3.0, 7.0]])
    f = {"batch_atom_lvl": torch.tensor([0, 0, 1, 1])}

    expanded = expand_t_to_level(
        t,
        f,
        level_len=4,
        batch_key="batch_atom_lvl",
    )

    assert expanded.tolist() == [
        [2.0, 2.0, 5.0, 5.0],
        [3.0, 3.0, 7.0, 7.0],
    ]


def test_sparse_collate_keeps_t_system_level_when_length_matches_atom_count():
    """
    t has shape [D], so it must be stacked to [D, B] even when D happens to
    match the number of atoms in the first example.
    """

    def make_example(t, *, n_atoms, n_tokens):
        return {
            "t": torch.tensor(t, dtype=torch.float32),
            "noise": torch.zeros((4, n_atoms, 3), dtype=torch.float32),
            "ground_truth": {
                "coord_atom_lvl": torch.zeros(n_atoms, 3),
                "sequence_gt_I": torch.zeros(n_tokens, dtype=torch.long),
                "is_original_unindexed_token": torch.zeros(n_tokens, dtype=torch.bool),
            },
            "feats": {
                "atom_to_token_map": torch.arange(n_atoms) % n_tokens,
            },
        }

    collated = SparseCollate()(
        [
            make_example([1.0, 2.0, 3.0, 4.0], n_atoms=4, n_tokens=2),
            make_example([10.0, 20.0, 30.0, 40.0], n_atoms=3, n_tokens=2),
        ]
    )

    assert collated["t"].shape == (4, 2)
    assert collated["t_L"].shape == (4, 7)
    assert collated["t_I"].shape == (4, 4)
    assert collated["t_L"].tolist() == [
        [1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0],
        [2.0, 2.0, 2.0, 2.0, 20.0, 20.0, 20.0],
        [3.0, 3.0, 3.0, 3.0, 30.0, 30.0, 30.0],
        [4.0, 4.0, 4.0, 4.0, 40.0, 40.0, 40.0],
    ]
    assert collated["t_I"].tolist() == [
        [1.0, 1.0, 10.0, 10.0],
        [2.0, 2.0, 20.0, 20.0],
        [3.0, 3.0, 30.0, 30.0],
        [4.0, 4.0, 40.0, 40.0],
    ]


def test_uniquify_atom_names_for_cif_allows_molecule_trajectory_export(tmp_path):
    """
    Test that uniquify_atom_names_for_cif() correctly modifies the atom names
    to be unique, which allows for exporting molecule trajectories to CIF files
    without name collisions.
    """
    atom_array = create_diffused_molecule_atoms(7)
    atom_array = set_indices(
        array=atom_array,
        chain="A",
        res_id_start=1,
        molecule_id=0,
        component="7m",
    )

    unique_atom_array = uniquify_atom_names_for_cif(atom_array)
    coords = np.zeros((3, atom_array.array_length(), 3), dtype=np.float32)
    stack = build_stack_from_atom_array_and_batched_coords(coords, atom_array)
    stack.atom_name = unique_atom_array.atom_name.copy()

    out_path = tmp_path / "traj.cif.gz"
    to_cif_file(stack, out_path, file_type="cif.gz", include_entity_poly=False)

    assert out_path.exists()


def test_uniquify_atom_names_for_cif_updates_atom_id_for_prediction_export(tmp_path):
    """
    Generated molecule outputs can carry duplicate atom IDs within a single LIG
    residue. The CIF writer requires unique bond annotation identifiers.
    """
    atom_array = create_diffused_molecule_atoms(7)
    atom_array = set_indices(
        array=atom_array,
        chain="A",
        res_id_start=1,
        molecule_id=0,
        component="7m",
    )
    atom_array.set_annotation(
        "atom_id",
        np.full(atom_array.array_length(), "C", dtype="<U4"),
    )

    assert has_ambiguous_annotation_set(atom_array)

    atom_array = uniquify_atom_names_for_cif(atom_array)

    assert not has_ambiguous_annotation_set(atom_array)
    assert atom_array.atom_id.tolist() == atom_array.atom_name.tolist()

    out_path = tmp_path / "prediction.cif.gz"
    to_cif_file(atom_array, out_path, file_type="cif.gz", include_entity_poly=False)

    assert out_path.exists()
