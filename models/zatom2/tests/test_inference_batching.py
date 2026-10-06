"""Independent-example batching, output isolation and distributed shard coverage."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning.fabric.strategies import SingleDeviceStrategy
from lightning.fabric.wrappers import _FabricModule
from nucleus.inference_input_parsing import (
    create_diffused_material_atoms,
    create_diffused_molecule_atoms,
    install_molecule_contig_support,
)
from nucleus.model.inference_sampler import ZatomConditionalDiffusionSampler
from nucleus.trainers.zatom import ZatomDesignTrainer
from nucleus.utils.batching import SparseCollate
from omegaconf import OmegaConf

from zatom2.inference_batching import (
    batching_fallback_reason,
    build_inference_loader,
    split_prediction_outputs,
)
from zatom2.model.Zatom2 import Zatom2
from zatom2.run_inference import (
    RFD3InferenceConfig,
    Zatom2InferenceEngine,
    build_inference_engine,
)


def _example(index, num_atoms, *, diffusion_batch_size=1, material=False):
    make_atoms = (
        create_diffused_material_atoms if material else create_diffused_molecule_atoms
    )
    array = make_atoms(num_atoms)
    array.set_annotation("is_ligand", np.ones(num_atoms, dtype=bool))
    # Atomized ligands keep their residue layout during sequence readout;
    # element generation is controlled separately by atom_type_generation_mask.
    array.is_motif_atom_with_fixed_seq[:] = True
    for key in ("is_protein", "is_dna", "is_rna"):
        array.set_annotation(key, np.zeros(num_atoms, dtype=bool))
    for key in ("active_donor", "active_acceptor"):
        if key in array.get_annotation_categories():
            array.del_annotation(key)
    coordinates = torch.linspace(0.1, 0.9, num_atoms * 3).reshape(num_atoms, 3) + index
    return {
        "example_id": f"example_{index}_0",
        "atom_array": array,
        "specification": {"example_id": f"source_{index}", f"optional_{index}": index},
        "t": torch.ones(diffusion_batch_size),
        "noise": torch.zeros(diffusion_batch_size, num_atoms, 3),
        "coord_atom_lvl_to_be_noised": torch.zeros(diffusion_batch_size, num_atoms, 3),
        "ground_truth": {
            "coord_atom_lvl": coordinates,
            "sequence_gt_I": torch.zeros(num_atoms, dtype=torch.long),
            "is_original_unindexed_token": torch.zeros(num_atoms, dtype=torch.bool),
        },
        "feats": {
            "atom_to_token_map": torch.arange(num_atoms),
            "ref_pos": coordinates,
            "ref_element": torch.full((num_atoms,), 6 + index, dtype=torch.long),
            "is_motif_atom_with_fixed_coord": torch.zeros(num_atoms, dtype=torch.bool),
            "is_motif_atom_unindexed": torch.zeros(num_atoms, dtype=torch.bool),
            "atom_type_generation_mask": torch.ones(num_atoms, dtype=torch.bool),
            "material_atom_mask": torch.full((num_atoms,), material),
            "material_token_mask": torch.full((num_atoms,), material),
            "mean_force_norm": torch.full((num_atoms, 1), 0.5 + index),
            "has_mean_force_norm": torch.ones(num_atoms, 1, dtype=torch.bool),
        },
    }


class _PredictionModel(torch.nn.Module):
    """Deterministic predictions that distinguish every atom and system."""

    def __init__(self):
        super().__init__()
        self.calls = []
        self.eval()

    def forward(self, *, input, coord_atom_lvl_to_be_noised):
        features = input["f"]
        diffusion_size = coord_atom_lvl_to_be_noised.shape[0]
        batch_size = features["batch_size"]
        self.calls.append((batch_size, diffusion_size))
        coord = features["ref_pos"].unsqueeze(0).expand(diffusion_size, -1, -1)
        elements = features["ref_element"].unsqueeze(0).expand(diffusion_size, -1)
        energy = torch.zeros(batch_size).index_add_(
            0, features["batch_atom_lvl"], features["mean_force_norm"].squeeze(-1)
        )
        energy = energy.unsqueeze(0).expand(diffusion_size, -1)
        return {
            "X_L": coord,
            "atom_type_indices_I": elements,
            "fractional_coord_L": coord / 10,
            "lattice_lengths_B": energy.unsqueeze(-1).expand(-1, -1, 3),
            "lattice_angles_B": torch.full(
                (diffusion_size, batch_size, 3), torch.pi / 2
            ),
            "energy_B": energy,
            "forces_L": coord / 100,
        }


def _engine(tmp_path, *, batch_size=2, **kwargs):
    engine = Zatom2InferenceEngine(
        **RFD3InferenceConfig(ckpt_path=tmp_path / "model.ckpt"),
        inference_batch_size=batch_size,
        **kwargs,
    )
    trainer = ZatomDesignTrainer.__new__(ZatomDesignTrainer)
    trainer.fabric = SimpleNamespace(
        device=torch.device("cpu"),
        to_device=lambda value: value,
        strategy=None,
        global_rank=0,
        world_size=1,
    )
    trainer.state = {"model": _PredictionModel()}
    trainer.tensor_validation_every_n_steps = None
    trainer.allow_sequence_outputs = True
    trainer.read_sequence_from_sequence_head = False
    trainer.association_scheme = "allatom"
    trainer.seed = 42
    trainer.output_full_json = True
    trainer.cleanup_guideposts = True
    trainer.cleanup_virtual_atoms = True
    trainer.material_coordinate_output_source = "fractional"
    trainer.wrap_material_euclidean_coordinates = True
    # Metric preparation is irrelevant to inference; leave the actual legacy
    # validation forward and structure/metadata builder in place for comparison.
    trainer._assemble_metrics_extra_info = lambda example, _output: {
        "X_gt_L": example["coord_atom_lvl_to_be_noised"]
    }
    engine.trainer = trainer
    engine.out_dir = None
    return engine


@pytest.mark.parametrize("material", [False, True])
@pytest.mark.parametrize("diffusion_size", [1, 2])
def test_packed_outputs_match_legacy_single_example_writer(
    tmp_path, material, diffusion_size
):
    examples = [
        _example(i, count, diffusion_batch_size=diffusion_size, material=material)
        for i, count in enumerate((2, 3))
    ]
    engine = _engine(tmp_path)
    serial = [engine._model_forward(example) for example in deepcopy(examples)]
    engine.trainer.state["model"].calls.clear()

    groups = engine._model_forward_batch(deepcopy(examples))

    assert engine.trainer.state["model"].calls == [(2, diffusion_size)]
    assert [len(group) for group in groups] == [diffusion_size, diffusion_size]
    for index, (actual_group, expected_group) in enumerate(zip(groups, serial)):
        for actual, expected in zip(actual_group, expected_group):
            assert actual.example_id == expected.example_id
            assert actual.metadata == expected.metadata
            assert f"optional_{index}" in actual.metadata["specification"]
            np.testing.assert_array_equal(
                actual.atom_array.element, expected.atom_array.element
            )
            np.testing.assert_allclose(
                actual.atom_array.coord, expected.atom_array.coord, rtol=0, atol=0
            )
            assert actual.atom_array.element.tolist() == [
                "C" if index == 0 else "N"
            ] * (index + 2)


def test_split_uses_explicit_axes_when_diffusion_atoms_tokens_and_systems_overlap():
    examples = [
        _example(0, 2, diffusion_batch_size=2),
        _example(1, 3, diffusion_batch_size=2),
    ]
    examples[0]["ground_truth"]["sequence_gt_I"] = torch.zeros(1)
    examples[1]["ground_truth"]["sequence_gt_I"] = torch.zeros(2)
    outputs = {
        "X_L": torch.arange(30).reshape(2, 5, 3),
        "sequence_logits_I": torch.arange(24).reshape(2, 3, 4),
        "energy_B": torch.tensor([[10, 20], [30, 40]]),
        "lattice_lengths_B": torch.arange(12).reshape(2, 2, 3),
        "X_noisy_L_traj": [torch.empty(2, 5, 3)],
    }
    first, second = split_prediction_outputs(outputs, examples)
    torch.testing.assert_close(first["X_L"], outputs["X_L"][:, :2])
    torch.testing.assert_close(second["X_L"], outputs["X_L"][:, 2:])
    torch.testing.assert_close(
        second["sequence_logits_I"], outputs["sequence_logits_I"][:, 1:]
    )
    torch.testing.assert_close(second["energy_B"], torch.tensor([[20], [40]]))
    torch.testing.assert_close(
        second["lattice_lengths_B"], outputs["lattice_lengths_B"][:, 1:]
    )
    assert "X_noisy_L_traj" not in first


def test_split_rejects_predictions_that_omit_a_system():
    with pytest.raises(ValueError, match="Packed prediction 'energy_B'"):
        list(
            split_prediction_outputs(
                {"energy_B": torch.zeros(1, 1)}, [_example(0, 2), _example(1, 3)]
            )
        )


class _EchoPipeline:
    def __call__(self, data):
        return {"example_id": data["example_id"], "num_atoms": len(data["atom_array"])}


@pytest.mark.parametrize("world_size", [1, 2, 8])
def test_loader_covers_inputs_once_with_uneven_and_empty_shards(world_size):
    install_molecule_contig_support()
    specs = {f"example_{i}": {"length": f"{i + 2}m"} for i in range(5)}
    seen = []
    for rank in range(world_size):
        loader = build_inference_loader(
            specs,
            pipeline=_EchoPipeline(),
            rank=rank,
            world_size=world_size,
            batch_size=2,
            num_workers=0,
            prefetch_factor=2,
            seed=42,
        )
        batches = list(loader)
        assert all(1 <= len(batch) <= 2 for batch in batches)
        seen.extend(item["example_id"] for batch in batches for item in batch)
    assert sorted(seen) == sorted(specs)


def test_spawned_worker_installs_molecule_input_support():
    loader = build_inference_loader(
        {"example": {"length": "3m"}},
        pipeline=_EchoPipeline(),
        rank=0,
        world_size=1,
        batch_size=2,
        num_workers=1,
        prefetch_factor=2,
        seed=42,
    )
    assert list(loader) == [[{"example_id": "example", "num_atoms": 3}]]


@pytest.mark.parametrize("world_size", [1, 16])
@pytest.mark.parametrize("material", [False, True])
def test_engine_preserves_tail_batch_and_output_names(
    tmp_path, monkeypatch, world_size, material
):
    examples = [_example(i, i + 2, material=material) for i in range(3)]
    engine = _engine(tmp_path)
    engine.pipeline = _EchoPipeline()
    prediction_model = engine.trainer.state["model"]
    if world_size > 1:
        strategy = SingleDeviceStrategy(device="cpu")
        engine.trainer.fabric.strategy = strategy
        engine.trainer.fabric.world_size = world_size
        engine.trainer.fabric.global_rank = 14
        engine.trainer.state["model"] = _FabricModule(
            # No forward implementation: using this target instead of the
            # original model fails without launching a distributed process group.
            forward_module=torch.nn.Module(),
            strategy=strategy,
            original_module=prediction_model,
        ).eval()
    original_model = engine.trainer.state["model"]
    monkeypatch.setattr(
        "zatom2.inference_batching.build_inference_loader",
        lambda *args, **kwargs: [examples[:2], examples[2:]],
    )
    outputs = engine._run_multi({})
    assert prediction_model.calls == [(2, 1), (1, 1)]
    assert engine.trainer.state["model"] is original_model
    assert not original_model.training
    assert list(outputs) == [example["example_id"] for example in examples]
    assert [group[0].example_id for group in outputs.values()] == [
        f"{example['example_id']}_model_0" for example in examples
    ]


@pytest.mark.parametrize(
    "feature", ["partial_t", "is_motif_atom_unindexed", "has_mean_force_norm"]
)
def test_incompatible_inputs_fall_back_without_losing_examples(
    tmp_path, monkeypatch, feature
):
    examples = [_example(0, 2), _example(1, 3)]
    if feature == "partial_t":
        examples[1]["feats"][feature] = torch.tensor([5.0])
    else:
        examples[1]["feats"][feature].logical_not_()
    assert batching_fallback_reason(examples) is not None
    engine = _engine(tmp_path)
    engine.pipeline = _EchoPipeline()
    monkeypatch.setattr(
        "zatom2.inference_batching.build_inference_loader",
        lambda *args, **kwargs: [examples],
    )
    monkeypatch.setattr(
        engine, "_model_forward", lambda example: [example["example_id"]]
    )
    outputs = engine._run_multi({})
    assert outputs == {
        example["example_id"]: [example["example_id"]] for example in examples
    }


def test_batching_config_keeps_sampling_settings(tmp_path):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(
            config_name="generation_evaluation",
            overrides=[
                f"ckpt_path={tmp_path / 'model.ckpt'}",
                f"out_dir={tmp_path}",
                "inputs=inputs.json",
                "inference_batch_size=8",
                "inference_num_workers=4",
            ],
        )
    sampler_config = OmegaConf.to_container(config.inference_sampler, resolve=True)
    engine = build_inference_engine(config)
    assert engine.inference_batch_size == 8
    assert engine.inference_num_workers == 4
    assert config.diffusion_batch_size == config.n_batches == 1
    assert engine.transform_overrides["diffusion_batch_size"] == 1
    assert engine.inference_sampler_overrides == sampler_config


@pytest.mark.parametrize(
    "setting,value",
    [
        ("inference_batch_size", 0),
        ("inference_batch_size", 1.5),
        ("inference_num_workers", -1),
        ("inference_prefetch_factor", 0),
    ],
)
def test_invalid_batching_settings_are_rejected(tmp_path, setting, value):
    with pytest.raises(ValueError, match=setting):
        Zatom2InferenceEngine(
            **RFD3InferenceConfig(ckpt_path=tmp_path / "model.ckpt"), **{setting: value}
        )


def test_real_sampler_keeps_cfg_conditions_separate_and_recycling_unchanged(
    monkeypatch,
):
    # Zero Gaussian draws isolate batching arithmetic from changes in RNG order.
    monkeypatch.setattr(
        torch,
        "normal",
        lambda *, mean, std, size, device: torch.zeros(size, device=device),
    )
    sampler = ZatomConditionalDiffusionSampler(
        num_timesteps=4,
        n_recycle=2,
        allow_realignment=False,
        use_classifier_free_guidance=True,
        cfg_scale=2.0,
    )
    calls = []

    def denoiser(*, X_noisy_L, f, n_recycle, **kwargs):
        calls.append(n_recycle)
        return {"X_L": (f["ref_pos"] + f["mean_force_norm"]).unsqueeze(0)}

    def run(examples):
        batch = SparseCollate()(examples)
        features = batch["feats"]
        reference = Zatom2._build_cfg_reference_features(
            features,
            cfg_feature_mode="force_only",
            cfg_features=["mean_force_norm", "has_mean_force_norm"],
        )
        with torch.no_grad():
            return sampler.sample_diffusion_like_af3(
                f=features,
                f_ref=reference,
                diffusion_module=denoiser,
                diffusion_batch_size=1,
                coord_atom_lvl_to_be_noised=batch["coord_atom_lvl_to_be_noised"],
                initializer_outputs={},
                ref_initializer_outputs={},
            )["X_L"]

    examples = [_example(0, 2), _example(1, 3)]
    expected = torch.cat([run([example]) for example in examples], dim=1)
    calls.clear()
    actual = run(examples)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert calls == [2] * (2 * (4 - 1))


def test_real_denoiser_keeps_packed_molecules_independent(tmp_path):
    """Exercise actual token/atom attention and cell pooling with two recycles."""
    config_path = (
        Path(__file__).resolve().parents[1] / "configs/model/components/zatom2_net.yaml"
    )
    config = OmegaConf.create({"model": {"net": OmegaConf.load(config_path)}})
    net = config.model.net
    net.c_s, net.c_z, net.c_atom = 64, 16, 64
    net.attn_backend = "manual"
    net.diffusion_module.upcast.n_split = 2
    net.token_initializer.c_bottleneck = 4
    net.token_initializer.token_1d_features = {
        "mean_force_norm": 1,
        "has_mean_force_norm": 1,
    }
    net.token_initializer.atom_1d_features = {"ref_pos": 3}
    for key in ("atom_encoder", "token_trunk", "atom_decoder"):
        block = net.diffusion_module[key]
        block.n_blocks, block.n_head, block.c_bottleneck = 1, 2, 4
    net.inference_sampler = {"use_classifier_free_guidance": False, "cfg_scale": 1.0}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = instantiate(net, _recursive_=False).eval()
        # Give zero-initialized output/residual projections a signal so isolation
        # is tested with nontrivial coordinate and head predictions.
        for module in model.modules():
            if (
                isinstance(module, torch.nn.Linear)
                and not module.weight.count_nonzero()
            ):
                torch.nn.init.normal_(module.weight, std=0.02)
    engine = _engine(tmp_path)
    examples = [_example(0, 2), _example(1, 3)]
    for example in examples:
        count = len(example["atom_array"])
        features = example["feats"]
        features.update(
            {
                "restype": torch.zeros(count, 32),
                "ref_space_uid": torch.zeros(count, dtype=torch.long),
                "motif_pos": torch.zeros(count, 3),
                "is_ca": torch.ones(count, dtype=torch.bool),
                "is_motif_atom_with_fixed_seq": torch.zeros(count, dtype=torch.bool),
                "is_motif_token_with_fully_fixed_coord": torch.zeros(
                    count, dtype=torch.bool
                ),
                "unindexing_pair_mask": torch.zeros(count, count, dtype=torch.bool),
                "token_index": torch.arange(count),
            }
        )
        for key in ("asym_id", "entity_id", "residue_index", "sym_id"):
            features[key] = torch.zeros(count, dtype=torch.long)

    def predict(batch):
        features = engine.trainer._assemble_network_inputs(
            SparseCollate()(deepcopy(batch))
        )["f"]
        with torch.no_grad():
            initial = model.token_initializer(features)
            return model.diffusion_module(
                X_noisy_L=features["ref_pos"].unsqueeze(0),
                t=torch.tensor([0.35]),
                f=features,
                **initial,
            )

    expected = [predict([example]) for example in examples]
    actual = list(split_prediction_outputs(predict(examples), examples))
    for per_example, reference in zip(actual, expected):
        for key in per_example:
            torch.testing.assert_close(
                per_example[key], reference[key], rtol=1e-5, atol=1e-6
            )
