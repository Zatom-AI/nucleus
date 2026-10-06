"""Regression tests for production Zatom-2 pretraining configs."""

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from nucleus.inference_input_parsing import create_diffused_molecule_atoms
from nucleus.transforms.molecule_transforms import AddMeanForceNormConditioning

CONFIG_DIR = Path(__file__).parents[1] / "models" / "zatom2" / "configs"
MODEL_CONFIG_DIRS = tuple(
    Path(__file__).parents[1] / "models" / model_name / "configs"
    for model_name in ("zatom2",)
)
ATOMIZED_DATASETS = ("omol25", "omat24")
PROTEIN_ONLY_META_FEATURES = (
    "calculate_hbonds",
    "calculate_rasa",
    "keep_protein_motif_rasa",
    "hbond_subsample",
    "unindex_leak_global_index",
    "unindex_insert_random_break",
    "unindex_remove_random_break",
    "add_1d_ss_features",
    "featurize_plddt",
    "add_global_is_non_loopy_feature",
    "add_ppi_hotspots",
    "full_binder_crop",
)


def _compose(
    experiment: str,
    *,
    debug: str | None = None,
    force_conditioning: str | None = None,
    force_conditioning_architecture: str | None = None,
    extra_overrides: list[str] | None = None,
    config_dir: Path = CONFIG_DIR,
):
    """Compose a config for a given experiment and optional debug mode."""
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        overrides = [f"experiment={experiment}"]
        if debug is not None:
            overrides.append(f"debug={debug}")
        if force_conditioning is not None:
            overrides.append(f"force_conditioning={force_conditioning}")
        if force_conditioning_architecture is not None:
            overrides.append(
                f"force_conditioning_architecture={force_conditioning_architecture}"
            )
        overrides.extend(extra_overrides or [])
        return compose(config_name="train", overrides=overrides)


def _force_norm_statistics(forces):
    """Compute RMS and linear-interpolated P95 force magnitudes."""
    force_norms = sorted(math.dist(force, (0.0, 0.0, 0.0)) for force in forces)
    force_rms = math.sqrt(
        sum(force_norm**2 for force_norm in force_norms) / len(force_norms)
    )
    quantile_index = 0.95 * (len(force_norms) - 1)
    lower_index = math.floor(quantile_index)
    upper_index = math.ceil(quantile_index)
    fraction = quantile_index - lower_index
    force_p95 = force_norms[lower_index] + fraction * (
        force_norms[upper_index] - force_norms[lower_index]
    )
    return force_rms, force_p95


def _assert_broad_mlip_validation_inputs(config):
    """Assert that paper property runs use MLIP views of the broad cohorts."""
    expected_filenames = {
        "omol25": "omol25_val_energy_force_prediction_500.json",
        "omat24": "omat24_val_energy_force_prediction_500.json",
    }
    for domain, expected_filename in expected_filenames.items():
        assert (
            Path(config.datasets.val[f"{domain}_mlip"].dataset.data).name
            == expected_filename
        )


def test_force_conditioning_config_switches_model_mlp_input_features():
    legacy = _compose("pretrain")
    rms_p95 = _compose("pretrain", force_conditioning="rms_and_p95")

    legacy_features = legacy.model.net.token_initializer.token_1d_features
    assert legacy_features.mean_force_norm == 1
    assert legacy_features.has_mean_force_norm == 1
    assert legacy_features.get("force_rms_p95") is None

    rms_p95_features = rms_p95.model.net.token_initializer.token_1d_features
    assert rms_p95_features.mean_force_norm is None
    assert rms_p95_features.has_mean_force_norm is None
    assert rms_p95_features.force_rms_p95 == 2
    assert rms_p95_features.has_force_rms_p95 == 1


@pytest.mark.parametrize(
    ("architecture", "expected_residual"),
    [
        ("shared_bottleneck", False),
        ("residual_embedding", True),
    ],
)
def test_force_conditioning_architecture_presets_are_independently_selectable(
    architecture,
    expected_residual,
):
    config = _compose(
        "pretrain",
        force_conditioning_architecture=architecture,
    )

    assert (
        config.model.net.token_initializer.force_conditioning_residual_embedding
        is expected_residual
    )


def test_force_conditioning_architecture_group_contains_only_supported_presets():
    """Every selectable preset must correspond to a live model architecture."""
    config_group = CONFIG_DIR / "force_conditioning_architecture"

    assert {path.stem for path in config_group.glob("*.yaml")} == {
        "residual_embedding",
        "shared_bottleneck",
    }


@pytest.mark.parametrize(
    ("experiment", "expected_frequencies"),
    [
        (
            "pretrain",
            {
                "unconditional": 1.0,
                "structure_prediction": 1.0,
                "sequence_design": 0.0,
                "island": 0.0,
            },
        ),
        (
            "pretrain-unconditional",
            {
                "unconditional": 1.0,
                "structure_prediction": 0.0,
                "sequence_design": 0.0,
                "island": 0.0,
            },
        ),
        (
            "pretrain-structure-prediction",
            {
                "unconditional": 0.0,
                "structure_prediction": 1.0,
                "sequence_design": 0.0,
                "island": 0.0,
            },
        ),
        (
            "pretrain-joint-properties",
            {
                "unconditional": 0.34,
                "structure_prediction": 0.33,
                "sequence_design": 0.0,
                "island": 0.0,
                "mlip": 0.33,
            },
        ),
    ],
)
def test_atomized_pretrain_experiments_have_explicit_task_mixtures(
    experiment, expected_frequencies
):
    """Test that the pretraining experiments have the expected task mixture frequencies."""
    config = _compose(experiment)
    conditions = config.datasets.global_transform_args.train_conditions

    assert {
        name: float(condition.frequency) for name, condition in conditions.items()
    } == expected_frequencies


@pytest.mark.parametrize(
    "experiment",
    [
        "pretrain",
        "pretrain-unconditional",
        "pretrain-structure-prediction",
        "pretrain-joint-properties",
    ],
)
def test_atomized_pretrain_experiments_use_fast_path_and_disable_protein_features(
    experiment,
):
    """Test that the pretraining experiments use the fast path and disable protein features."""
    config = _compose(experiment)

    for dataset_name in ATOMIZED_DATASETS:
        dataset_config = config.datasets.train[dataset_name]
        assert dataset_config.probability == 0.5
        assert dataset_config.dataset.transform.optimize_atomized_ligand_pipeline
        transform = dataset_config.dataset.transform
        assert not transform.generate_conformers
        assert not transform.generate_conformers_for_non_protein_only
        assert not transform.provide_reference_conformer_when_unmasked
    assert config.datasets.train.pdb is None

    probabilities = (
        config.datasets.global_transform_args.meta_conditioning_probabilities
    )
    assert all(float(probabilities[name]) == 0.0 for name in PROTEIN_ONLY_META_FEATURES)


@pytest.mark.parametrize(
    "config_dir",
    MODEL_CONFIG_DIRS,
    ids=("zatom2",),
)
def test_omol_pretraining_loads_fragment_status_for_every_model(config_dir):
    """All released model packages apply the shared OMol task-local filter."""
    config = _compose("pretrain", config_dir=config_dir)
    loader = config.datasets.train.omol25.dataset.loader

    assert loader._target_ == (
        "nucleus.utils.bond_sidecar.create_bond_sidecar_ase_atoms_loader"
    )
    assert loader.bond_sidecar_path == config.paths.data.omol25_bond_sidecar_path

    validation_config = OmegaConf.load(
        config_dir / "datasets" / "val" / "omol25_subset_dataset.yaml"
    )
    validation_loader = validation_config.dataset.dataset.loader
    assert validation_loader._target_ == loader._target_
    assert (
        OmegaConf.to_container(validation_loader, resolve=False)["bond_sidecar_path"]
        == "${paths.data.omol25_bond_sidecar_path}"
    )




@pytest.mark.parametrize(
    "config_dir",
    MODEL_CONFIG_DIRS,
    ids=("zatom2",),
)
@pytest.mark.parametrize("reduction", [None, "system_mean", "active_mean"])
def test_material_geometry_loss_reduction_is_configurable(config_dir, reduction):
    """All models preserve system means by default and can restore active means."""
    overrides = (
        []
        if reduction is None
        else [f"trainer.loss.material_geometry_loss.reduction={reduction}"]
    )
    config = _compose("pretrain", config_dir=config_dir, extra_overrides=overrides)
    material_loss = config.trainer.loss.material_geometry_loss
    expected_reduction = reduction or "system_mean"
    assert material_loss.reduction == expected_reduction
    assert instantiate(material_loss).reduction == expected_reduction


@pytest.mark.parametrize(
    "config_dir",
    MODEL_CONFIG_DIRS,
    ids=("zatom2",),
)
def test_joint_property_pretrain_is_opt_in_and_preserves_sampler_contract(config_dir):
    """Test that the joint property pretraining experiment enables property targets and the property head, and preserves the sampler contract."""
    config = _compose("pretrain-joint-properties", config_dir=config_dir)

    assert config.datasets.enable_atomistic_property_targets
    assert config.datasets.diffusion_batch_size_train == 2
    assert config.datasets.crop_size == 512
    expected_max_atoms = 5120 if config_dir == CONFIG_DIR else 512
    assert config.datasets.max_atoms_in_crop == expected_max_atoms
    assert not config.datasets.randomly_translate_fractional_coordinates
    assert config.model.net.diffusion_module.f_pred == "flow_matching"
    assert config.model.net.diffusion_module.atomistic_property_head.enabled
    assert config.trainer.loss.atomistic_property_loss.energy_weight == 0.1
    assert config.trainer.loss.atomistic_property_loss.force_weight == 0.3
    assert config.trainer.loss.atomistic_property_loss.energy_max_sigma == 1.0
    assert config.trainer.loss.atomistic_property_loss.force_max_sigma == 0.25
    assert config.trainer.loss.atomistic_property_loss.warmup_steps == 10_000
    material_loss = config.trainer.loss.material_geometry_loss
    assert material_loss.fractional_coord_weight == 0.25
    assert material_loss.lattice_lengths_weight == 0.05
    assert material_loss.lattice_angles_weight == 0.05
    assert (
        config.datasets.global_transform_args.mean_force_norm_conditioning_probability
        == 0.8
    )
    assert list(
        config.datasets.global_transform_args.mean_force_norm_conditioning_tasks
    ) == ["unconditional", "structure_prediction"]
    condition_frequencies = {
        name: float(condition.frequency)
        for name, condition in config.datasets.global_transform_args.train_conditions.items()
    }
    expected_shared_frequencies = {
        "unconditional": 0.34,
        "island": 0.0,
        "sequence_design": 0.0,
        "structure_prediction": 0.33,
        "mlip": 0.33,
    }
    assert {
        name: condition_frequencies[name] for name in expected_shared_frequencies
    } == expected_shared_frequencies
    assert all(
        frequency == 0.0
        for name, frequency in condition_frequencies.items()
        if name not in expected_shared_frequencies
    )
    mlip = config.datasets.global_transform_args.train_conditions.mlip
    assert mlip.name == "mlip"
    assert mlip.molecule_conditioning_mode == "mlip"
    assert mlip.p_fix_motif_coordinates == 1.0
    assert mlip.p_fix_motif_sequence == 1.0
    assert config.datasets.train.omol25.probability == 0.5
    assert config.datasets.train.omat24.probability == 0.5
    assert config.datasets.train.pdb is None
    assert config.datasets.train.omol25.dataset.transform.atomistic_property_domain == (
        "omol25"
    )
    assert config.datasets.train.omat24.dataset.transform.atomistic_property_domain == (
        "omat24"
    )
    assert (
        config.datasets.train.omol25.dataset.transform.enable_atomistic_property_targets
    )
    assert (
        config.datasets.train.omat24.dataset.transform.enable_atomistic_property_targets
    )

    token_features = config.model.net.token_initializer.token_1d_features
    for key in (
        "total_charge",
        "spin_multiplicity",
        "has_total_charge",
        "has_spin_multiplicity",
    ):
        assert token_features[key] == 1

    assert set(config.datasets.val) == {
        "unconditional",
        "unconditional_deep",
        "indexed",
        "omol25_generation",
        "omat24_generation",
        "omol25_structure_prediction",
        "omat24_structure_prediction",
        "omol25_mlip",
        "omat24_mlip",
    }
    assert config.datasets.val.unconditional is None
    assert config.datasets.val.unconditional_deep is None
    assert config.datasets.val.indexed is None
    if config_dir == CONFIG_DIR:
        expected_generation_inputs = {
            "omol25": (
                "omol25_train_4m_mean_force_norm_1000_with_hydrogens_with_forces.json"
            ),
            "omat24": "omat24_train_1m_mean_force_norm_1000_with_forces.json",
        }
        expected_structure_prediction_inputs = {
            "omol25": "omol25_val_structure_prediction_500.json",
            "omat24": "omat24_val_structure_prediction_500.json",
        }
        expected_mlip_inputs = {
            "omol25": "omol25_val_energy_force_prediction_500.json",
            "omat24": "omat24_val_energy_force_prediction_500.json",
        }
    else:
        expected_generation_inputs = {
            "omol25": (
                "omol25_geom_orca6_val_low_force_representative_1000_"
                "with_hydrogens_with_forces.json"
            ),
            "omat24": "omat24_val_rattled_relax_atom_count_1000_with_forces.json",
        }
        expected_structure_prediction_inputs = {
            "omol25": "omol25_geom_orca6_structure_prediction_500.json",
            "omat24": "omat24_rattled_300_structure_prediction_500.json",
        }
        expected_mlip_inputs = {
            "omol25": "omol25_geom_orca6_energy_force_prediction_500.json",
            "omat24": "omat24_rattled_300_energy_force_prediction_500.json",
        }
    for domain in ("omol25", "omat24"):
        assert (
            Path(config.datasets.val[f"{domain}_generation"].dataset.data).name
            == expected_generation_inputs[domain]
        )
    assert (
        config.datasets.val.omol25_generation.dataset.transform.mean_force_norm_conditioning_probability
        == 1.0
    )
    assert list(
        config.datasets.val.omol25_generation.dataset.transform.mean_force_norm_conditioning_tasks
    ) == ["unconditional"]
    assert (
        config.datasets.val.omat24_generation.dataset.transform.mean_force_norm_conditioning_probability
        == 1.0
    )
    assert list(
        config.datasets.val.omat24_generation.dataset.transform.mean_force_norm_conditioning_tasks
    ) == ["unconditional"]
    for domain in ("omol25", "omat24"):
        assert (
            Path(
                config.datasets.val[f"{domain}_structure_prediction"].dataset.data
            ).name
            == expected_structure_prediction_inputs[domain]
        )
        structure_transform = config.datasets.val[
            f"{domain}_structure_prediction"
        ].dataset.transform
        assert structure_transform.mean_force_norm_conditioning_probability == 1.0
        assert list(structure_transform.mean_force_norm_conditioning_tasks) == [
            "structure_prediction"
        ]
    for domain in ("omol25", "omat24"):
        assert (
            Path(config.datasets.val[f"{domain}_mlip"].dataset.data).name
            == expected_mlip_inputs[domain]
        )
    assert config.datasets.val.omol25_mlip.dataset.subset_to_keys is None
    assert config.datasets.val.omat24_mlip.dataset.subset_to_keys is None
    for dataset in config.datasets.val.values():
        if dataset is not None:
            assert (
                dataset.dataset.transform.diffusion_batch_size
                == config.datasets.diffusion_batch_size_inference
            )
    expected_metrics = {
        "molecule_generation_metrics",
        "material_generation_metrics",
        "atomistic_property_metrics",
    }
    if config_dir == CONFIG_DIR:
        expected_metrics.add("structure_prediction_metrics")
    assert set(config.trainer.metrics) == expected_metrics
    assert list(
        config.trainer.metrics.molecule_generation_metrics.required_tags_all
    ) == ["omol25", "unconditional"]
    assert list(
        config.trainer.metrics.material_generation_metrics.required_tags_all
    ) == ["omat24", "unconditional"]
    assert list(
        config.trainer.metrics.atomistic_property_metrics.required_tags_all
    ) == ["mlip"]
    if config_dir == CONFIG_DIR:
        structure_metrics = config.trainer.metrics.structure_prediction_metrics
        assert structure_metrics._target_ == (
            "nucleus.metrics.structure_prediction.ZevalsStructurePredictionMetrics"
        )
        assert list(structure_metrics.required_tags_all) == ["structure_prediction"]
        assert structure_metrics.compute_uma_forces
        assert structure_metrics.uma_molecule_task == "omol"
        assert structure_metrics.uma_material_task == "omat"


def test_zatom2_joint_property_paper_hyperparameters_are_config_owned():
    """The canonical Table 3 run must compose without shell-only settings."""
    config = _compose("pretrain-joint-properties")

    assert config.name == (
        "paper-pretrain-omol25-omat24-joint-properties-force-conditioned"
    )
    assert "csv" in config.logger
    assert "wandb" not in config.logger
    assert config.model.compile_model is False
    assert config.model.lr_scheduler is None
    assert config.model.optimizer.lr == pytest.approx(1e-4)
    assert config.model.net.attn_backend == "cueq"

    train_loader = config.dataloader.train.dataloader_params
    assert train_loader.batch_size == 256
    assert train_loader.num_workers == 8
    assert train_loader.prefetch_factor == 1
    assert train_loader.coalesce_worker_tensors
    assert train_loader.persistent_workers
    assert train_loader.sanitize_coordinate_nans
    assert train_loader.worker_chunk_size == 32
    assert config.dataloader.val.dataloader_params.num_workers == 8
    assert config.datasets.diffusion_batch_size_inference == 1
    assert config.datasets.max_num_tokens == 7168
    assert config.datasets.randomly_augment
    assert config.trainer.grad_accum_steps == 1
    assert config.trainer.max_epochs == 80
    assert config.trainer.n_examples_per_epoch == 4_996_604
    assert config.trainer.checkpoint_every_n_epochs is None
    assert config.trainer.checkpoint_every_n_steps == 750
    assert config.trainer.checkpoint_before_validation
    assert config.trainer.checkpoint_keep_last_n == 100
    assert not config.trainer.allow_world_size_change_on_resume
    assert config.trainer.tensor_validation_every_n_steps == 1000
    assert not config.trainer.validate_first_epoch
    assert not config.trainer.get(
        "collate_validation_batch_before_device_transfer", False
    )
    assert dict(config.trainer.validate_loader_every_n_epochs) == {
        "omol25_generation": 5,
        "omat24_generation": 5,
        "omol25_structure_prediction": 5,
        "omat24_structure_prediction": 5,
        "omol25_mlip": 1,
        "omat24_mlip": 1,
    }

    assert config.callbacks.log_molecule_afd_validation_metrics_callback.enabled
    assert config.callbacks.log_materials_afd_validation_metrics_callback.enabled
    assert config.callbacks.log_grouped_zevals_structure_prediction_metrics_callback.enabled
    composite_callback = config.callbacks.log_composite_validation_metrics_callback
    assert composite_callback.enabled
    assert composite_callback.active_tasks is None
    assert set(composite_callback.task_metrics) == {
        "generation",
        "structure_prediction",
        "mlip",
    }
    callback_order = list(config.callbacks)
    assert callback_order.index(
        "log_composite_validation_metrics_callback"
    ) > callback_order.index("log_design_validation_metrics_callback")
    assert config.trainer.metrics.molecule_generation_metrics.compute_uma_forces
    assert config.trainer.metrics.material_generation_metrics.compute_uma_forces
    assert config.trainer.metrics.structure_prediction_metrics.compute_uma_forces










@pytest.mark.parametrize(
    (
        "experiment",
        "expected_train_domains",
        "expected_task_frequencies",
        "expected_force_probability",
    ),
    [
        (
            "paper-table3-joint-force-conditioned",
            {"omol25": 0.5, "omat24": 0.5},
            {"unconditional": 0.5, "structure_prediction": 0.5},
            0.8,
        ),
        (
            "paper-table1-zatom2-s-omol25-joint-force-conditioned",
            {"omol25": 1.0},
            {"unconditional": 0.5, "structure_prediction": 0.5},
            0.8,
        ),
        (
            "paper-table1-zatom2-s-omol25-unconditional-force-conditioned",
            {"omol25": 1.0},
            {"unconditional": 1.0, "structure_prediction": 0.0},
            0.8,
        ),
        (
            "paper-table1-zatom2-s-omol25-unconditional-without-force-conditioning",
            {"omol25": 1.0},
            {"unconditional": 1.0, "structure_prediction": 0.0},
            0.0,
        ),
    ],
)
def test_paper_pretrain_experiment_matrix(
    experiment,
    expected_train_domains,
    expected_task_frequencies,
    expected_force_probability,
):
    """Table 1/3 launch configs differ only along the declared ablations."""
    config = _compose(experiment)

    actual_train_domains = {
        domain: float(dataset.probability)
        for domain, dataset in config.datasets.train.items()
        if dataset is not None
    }
    assert actual_train_domains == expected_train_domains

    conditions = config.datasets.global_transform_args.train_conditions
    for task, expected_frequency in expected_task_frequencies.items():
        assert float(conditions[task].frequency) == expected_frequency
    assert float(conditions.island.frequency) == 0.0
    assert float(conditions.sequence_design.frequency) == 0.0
    assert (
        config.datasets.global_transform_args.mean_force_norm_conditioning_probability
        == expected_force_probability
    )
    assert config.trainer.max_epochs == 80
    assert not config.datasets.enable_atomistic_property_targets
    assert "atomistic_property_head" not in config.model.net.diffusion_module

    if "zatom2-s" in experiment:
        assert config.model.net.diffusion_module.token_trunk.n_blocks == 9
    else:
        assert config.model.net.diffusion_module.token_trunk.n_blocks == 18


@pytest.mark.parametrize(
    ("force_conditioning", "value_key", "presence_key", "n_channels"),
    [
        ("mean_force_norm", "mean_force_norm", "has_mean_force_norm", 1),
        ("rms_and_p95", "force_rms_p95", "has_force_rms_p95", 2),
    ],
)
def test_table1_no_force_ablation_disables_validation_force_metadata(
    force_conditioning,
    value_key,
    presence_key,
    n_channels,
):
    """The no-force ablation must stay unconditioned during inference."""
    config = _compose(
        "paper-table1-zatom2-s-omol25-unconditional-without-force-conditioning",
        force_conditioning=force_conditioning,
    )

    train_transform = config.datasets.train.omol25.dataset.transform
    validation = config.datasets.val.omol25_generation.dataset
    validation_transform = validation.transform
    assert train_transform.mean_force_norm_conditioning_probability == 0.0
    assert list(train_transform.mean_force_norm_conditioning_tasks) == []
    assert validation_transform.mean_force_norm_conditioning_probability == 0.0
    assert list(validation_transform.mean_force_norm_conditioning_tasks) == []
    assert validation_transform.is_inference

    pipeline = instantiate(validation_transform)
    force_transforms = [
        transform
        for transform in pipeline.transforms
        if isinstance(transform, AddMeanForceNormConditioning)
    ]
    assert len(force_transforms) == 1

    validation_examples = json.loads(Path(validation.data).read_text(encoding="utf-8"))
    specification = next(iter(validation_examples.values()))
    original_extra = specification["extra"].copy()
    assert specification["extra"][presence_key] is True

    output = force_transforms[0](
        {
            "atom_array": create_diffused_molecule_atoms(3),
            "is_inference": True,
            "sampled_condition_name": "unconditional",
            "specification": specification,
        }
    )

    torch.testing.assert_close(output["feats"][value_key], torch.zeros(3, n_channels))
    torch.testing.assert_close(output["feats"][presence_key], torch.zeros(3, 1))
    assert specification["extra"] == original_extra


def test_table2_benchmark_pretraining_config():
    """Table 2 mixes only QM9/MP20 and disables force conditioning."""
    config = _compose("paper-table2-qm9-mp20-unconditional-without-force-conditioning")

    active_train_domains = {
        domain: float(dataset.probability)
        for domain, dataset in config.datasets.train.items()
        if dataset is not None
    }
    assert active_train_domains == {"qm9": 0.5, "mp20": 0.5}
    assert config.datasets.train.qm9.dataset.name == "qm9"
    assert config.datasets.train.mp20.dataset.name == "mp20"
    assert Path(config.datasets.train.qm9.dataset.paths).parts[-4:] == (
        "data",
        "qm9",
        "raw",
        "train_100k",
    )
    assert Path(config.datasets.train.mp20.dataset.paths).parts[-4:] == (
        "data",
        "mp_20",
        "raw",
        "train_27138",
    )
    assert config.datasets.val is None
    assert config.trainer.metrics is None
    assert config.trainer.n_examples_per_epoch == 127_138
    assert config.trainer.max_epochs == 10000
    assert config.trainer.checkpoint_every_n_epochs == 25
    assert config.trainer.checkpoint_every_n_steps is None
    assert config.trainer.checkpoint_keep_last_n is None
    assert config.datasets.max_num_tokens == 3584

    conditions = config.datasets.global_transform_args.train_conditions
    assert float(conditions.unconditional.frequency) == 1.0
    assert float(conditions.structure_prediction.frequency) == 0.0
    assert (
        config.datasets.global_transform_args.mean_force_norm_conditioning_probability
        == 0.0
    )
    assert config.model.net.diffusion_module.token_trunk.n_blocks == 18
    assert not config.datasets.enable_atomistic_property_targets

    for domain in ("qm9", "mp20"):
        pipeline = instantiate(config.datasets.train[domain].dataset.transform)
        assert pipeline is not None


def test_table3_property_finetuning_config():
    """Table 3 MLIP fine-tuning retains generation and structure rehearsal."""
    config = _compose("paper-table3-finetune-joint-properties")
    _assert_broad_mlip_validation_inputs(config)

    active_train_domains = {
        domain
        for domain, dataset in config.datasets.train.items()
        if dataset is not None
    }
    assert active_train_domains == {"omol25", "omat24"}
    assert config.datasets.train.omol25.probability == 0.5
    assert config.datasets.train.omat24.probability == 0.5
    assert config.datasets.max_num_tokens == 3584
    assert config.trainer.n_examples_per_epoch == 4_996_604
    assert config.trainer.max_epochs == 120
    assert config.trainer.precision == "32-true"

    active_val_loaders = {
        name for name, dataset in config.datasets.val.items() if dataset is not None
    }
    assert active_val_loaders == {
        "omol25_generation",
        "omat24_generation",
        "omol25_structure_prediction",
        "omat24_structure_prediction",
        "omol25_mlip",
        "omat24_mlip",
    }
    assert dict(config.trainer.validate_loader_every_n_epochs) == {
        "omol25_generation": 5,
        "omat24_generation": 5,
        "omol25_structure_prediction": 5,
        "omat24_structure_prediction": 5,
        "omol25_mlip": 1,
        "omat24_mlip": 1,
    }

    assert config.trainer.metrics.molecule_generation_metrics is not None
    assert config.trainer.metrics.material_generation_metrics is not None
    assert config.trainer.metrics.structure_prediction_metrics is not None
    assert config.trainer.metrics.atomistic_property_metrics is not None
    assert config.callbacks.log_molecule_afd_validation_metrics_callback.enabled
    assert config.callbacks.log_materials_afd_validation_metrics_callback.enabled
    assert config.callbacks.log_grouped_zevals_structure_prediction_metrics_callback.enabled

    frequencies = {
        name: float(condition.frequency)
        for name, condition in config.datasets.global_transform_args.train_conditions.items()
    }
    assert frequencies == {
        "unconditional": 0.1,
        "island": 0.0,
        "sequence_design": 0.0,
        "structure_prediction": 0.1,
        "mlip": 0.8,
    }
    assert (
        config.datasets.global_transform_args.mean_force_norm_conditioning_probability
        == 0.8
    )
    assert config.datasets.enable_atomistic_property_targets
    assert config.model.net.diffusion_module.atomistic_property_head.enabled
    assert config.trainer.loss.atomistic_property_loss.weight > 0.0
    assert config.trainer.loss.atomistic_property_loss.warmup_steps == 0
    assert config.trainer.loss.diffusion_loss.weight == 4.0
    assert config.trainer.loss.sequence_loss.sequence_weight == 0.1
    assert config.trainer.loss.sequence_loss.atom_type_weight == 0.1
    assert config.trainer.loss.material_geometry_loss.fractional_coord_weight == 0.25
    assert config.trainer.loss.material_geometry_loss.lattice_lengths_weight == 0.05
    assert config.trainer.loss.material_geometry_loss.lattice_angles_weight == 0.05


def test_table3_joint_property_pretraining_config():
    """The second Table 3 pretrain jointly supervises generation and MLIP."""
    config = _compose("paper-table3-joint-properties-force-conditioned")
    _assert_broad_mlip_validation_inputs(config)

    active_train_domains = {
        domain: float(dataset.probability)
        for domain, dataset in config.datasets.train.items()
        if dataset is not None
    }
    assert active_train_domains == {"omol25": 0.5, "omat24": 0.5}
    frequencies = {
        name: float(condition.frequency)
        for name, condition in config.datasets.global_transform_args.train_conditions.items()
    }
    assert frequencies == {
        "unconditional": 0.34,
        "island": 0.0,
        "sequence_design": 0.0,
        "structure_prediction": 0.33,
        "mlip": 0.33,
    }
    assert (
        config.datasets.global_transform_args.mean_force_norm_conditioning_probability
        == 0.8
    )
    assert config.datasets.enable_atomistic_property_targets
    assert config.model.net.diffusion_module.atomistic_property_head.enabled
    assert config.trainer.max_epochs == 80


def test_table3_large_property_pretraining_has_no_finetuning_changes():
    """The Table 3 Zatom-2-L row changes depth, not tasks or schedule."""
    config = _compose("paper-table3-zatom2-l-joint-properties-force-conditioned")
    _assert_broad_mlip_validation_inputs(config)

    assert config.model.net.diffusion_module.token_trunk.n_blocks == 36
    assert config.model.net.diffusion_module.atomistic_property_head.enabled
    assert config.datasets.max_num_tokens == 3584
    assert config.trainer.max_epochs == 80
    assert config.trainer.n_examples_per_epoch == 4_996_604
    assert config.datasets.train.omol25.probability == 0.5
    assert config.datasets.train.omat24.probability == 0.5


@pytest.mark.parametrize(
    (
        "experiment",
        "expected_trunk_blocks",
        "expected_subset_size",
        "expected_max_num_tokens",
    ),
    [
        (
            "paper-figure1-zatom2-500k-each-joint-properties-force-conditioned",
            18,
            500_000,
            7168,
        ),
        (
            "paper-figure1-zatom2-l-500k-each-joint-properties-force-conditioned",
            36,
            500_000,
            3584,
        ),
    ],
)
def test_figure1_scaling_pretraining_configs(
    experiment,
    expected_trunk_blocks,
    expected_subset_size,
    expected_max_num_tokens,
):
    """Figure 1 varies only fixed data support and token-trunk depth."""
    config = _compose(experiment)
    _assert_broad_mlip_validation_inputs(config)

    active_train_domains = {
        domain: float(dataset.probability)
        for domain, dataset in config.datasets.train.items()
        if dataset is not None
    }
    assert active_train_domains == {"omol25": 0.5, "omat24": 0.5}
    assert config.datasets.max_num_tokens == expected_max_num_tokens

    frequencies = {
        name: float(condition.frequency)
        for name, condition in config.datasets.global_transform_args.train_conditions.items()
    }
    assert frequencies == {
        "unconditional": 0.34,
        "island": 0.0,
        "sequence_design": 0.0,
        "structure_prediction": 0.33,
        "mlip": 0.33,
    }
    assert (
        config.datasets.global_transform_args.mean_force_norm_conditioning_probability
        == 0.8
    )
    assert config.datasets.enable_atomistic_property_targets
    assert config.model.net.diffusion_module.atomistic_property_head.enabled
    assert (
        config.model.net.diffusion_module.token_trunk.n_blocks == expected_trunk_blocks
    )
    assert config.trainer.max_epochs == 40
    assert config.trainer.n_examples_per_epoch == 4_996_604
    assert dict(config.trainer.validate_loader_every_n_epochs) == {
        "omol25_generation": 19,
        "omat24_generation": 19,
        "omol25_structure_prediction": 19,
        "omat24_structure_prediction": 19,
        "omol25_mlip": 1,
        "omat24_mlip": 1,
    }

    for domain in ("omol25", "omat24"):
        dataset = config.datasets.train[domain]
        assert dataset.get("uniform_subset_size") == expected_subset_size
        expected_seed = 42 if expected_subset_size is not None else None
        assert dataset.get("uniform_subset_seed") == expected_seed
        expected_source_size = None
        if expected_subset_size is not None:
            expected_source_size = 3_986_754 if domain == "omol25" else 1_009_850
        assert dataset.get("uniform_subset_source_size") == expected_source_size
        expected_sha256 = None
        if expected_subset_size is not None:
            expected_sha256 = {
                "omol25": (
                    "36b7366932ae1dbe3941743373610a063018697a2f2cd16664efbca114bdf01e"
                ),
                "omat24": (
                    "73d46f5304147e419a81c6df15f9a0fa628c199a525566637bf2b6fc887a5227"
                ),
            }[domain]
        assert dataset.get("uniform_subset_expected_sha256") == expected_sha256


@pytest.mark.parametrize(
    "config_dir",
    MODEL_CONFIG_DIRS,
    ids=("zatom2",),
)
def test_metrics_callbacks_expose_dataset_filters(config_dir):
    """All model variants expose the shared AFD dataset-filter arguments."""
    callbacks = OmegaConf.load(config_dir / "callbacks" / "metrics_logging.yaml")

    for callback_name in (
        "log_molecule_afd_validation_metrics_callback",
        "log_materials_afd_validation_metrics_callback",
    ):
        assert "dataset_names" in callbacks[callback_name]
        assert callbacks[callback_name].dataset_names is None


def test_joint_property_generation_validation_is_count_matched_and_force_conditioned():
    """OMol25 and OMat24 generation use 1,000 force-conditioned inputs each."""
    config = _compose("pretrain-joint-properties")

    counts = {}
    for domain in ("omol25", "omat24"):
        data_path = Path(config.datasets.val[f"{domain}_generation"].dataset.data)
        with data_path.open(encoding="utf-8") as handle:
            examples = json.load(handle)
        counts[domain] = len(examples)
        for specification in examples.values():
            metadata = specification["extra"]
            assert metadata["has_mean_force_norm"] is True
            assert math.isfinite(metadata["mean_force_norm"])
            assert metadata["mean_force_norm"] > 0.0
            assert metadata["has_force_rms_p95"] is True
            assert math.isfinite(metadata["force_rms"])
            assert math.isfinite(metadata["force_p95"])
            assert metadata["force_rms"] >= metadata["mean_force_norm"] > 0.0
            assert metadata["force_p95"] > 0.0
            if domain == "omol25":
                assert isinstance(specification["total_charge"], int)
                assert isinstance(specification["spin_multiplicity"], int)
                assert specification["spin_multiplicity"] >= 1
                assert metadata["source_split"] == "all"

    assert counts == {"omol25": 1000, "omat24": 1000}


def test_rms_p95_paper_validation_inputs_are_fully_conditioned():
    """Every loader advertised as conditioned must provide both statistics."""
    config = _compose(
        "paper-table3-joint-force-conditioned",
        force_conditioning="rms_and_p95",
    )
    expected_counts = {
        "omol25_generation": 1000,
        "omat24_generation": 1000,
        "omol25_structure_prediction": 500,
        "omat24_structure_prediction": 500,
    }

    for loader_name, expected_count in expected_counts.items():
        dataset = config.datasets.val[loader_name].dataset
        with Path(dataset.data).open(encoding="utf-8") as handle:
            examples = json.load(handle)
        assert len(examples) == expected_count
        assert dataset.transform.mean_force_norm_conditioning_probability == 1.0
        for specification in examples.values():
            extra = specification.get("extra", {})
            has_force_rms_p95 = specification.get(
                "has_force_rms_p95", extra.get("has_force_rms_p95")
            )
            force_rms = specification.get("force_rms", extra.get("force_rms"))
            force_p95 = specification.get("force_p95", extra.get("force_p95"))
            assert has_force_rms_p95 is True
            assert math.isfinite(force_rms)
            assert math.isfinite(force_p95)
            assert force_rms > 0.0
            assert force_p95 > 0.0


def test_omat_generation_force_statistics_provenance_digest():
    """Lock the ASE-LMDB row identities and statistics used by the fixture."""
    path = (
        CONFIG_DIR.parent
        / "docs/examples/omat24_val_rattled_relax_atom_count_1000_with_forces.json"
    )
    with path.open(encoding="utf-8") as handle:
        examples = json.load(handle)

    digest_rows = []
    for example_key, specification in examples.items():
        metadata = specification["extra"]
        assert metadata["source_shard"] == "rattled-relax/val.aselmdb"
        assert metadata["source_db_id"] == metadata["source_row_index"] + 1
        digest_rows.append(
            {
                "example_key": example_key,
                **{
                    key: metadata[key]
                    for key in (
                        "source_shard",
                        "source_row_index",
                        "source_db_id",
                        "sid",
                        "mean_force_norm",
                        "force_rms",
                        "force_p95",
                        "has_force_rms_p95",
                    )
                },
            }
        )

    digest = hashlib.sha256(
        json.dumps(digest_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert len(digest_rows) == 1000
    assert digest == "c591d837cd372bfc723f81a3d8904e6c82ad2d36462eb5ea67781a3c980c511e"


@pytest.mark.parametrize(
    ("domain", "structure_filename", "property_filename"),
    [
        (
            "omol25",
            "omol25_val_structure_prediction_500.json",
            "omol25_val_energy_force_prediction_500.json",
        ),
        (
            "omat24",
            "omat24_val_structure_prediction_500.json",
            "omat24_val_energy_force_prediction_500.json",
        ),
        (
            "omat24",
            "omat24_rattled_300_structure_prediction_500.json",
            "omat24_rattled_300_energy_force_prediction_500.json",
        ),
    ],
)
def test_joint_property_validation_uses_matched_structure_and_mlip_examples(
    domain,
    structure_filename,
    property_filename,
):
    """Every property example must match its structure-prediction counterpart."""
    examples_dir = CONFIG_DIR.parent / "docs" / "examples"
    with (examples_dir / structure_filename).open(encoding="utf-8") as handle:
        structure_examples = json.load(handle)
    with (examples_dir / property_filename).open(encoding="utf-8") as handle:
        property_examples = json.load(handle)

    assert len(structure_examples) == 500
    assert property_examples.keys() == structure_examples.keys()
    for example_id, structure_spec in structure_examples.items():
        property_spec = property_examples[example_id]
        metadata = property_spec["extra"]
        forces = metadata["forces_gt_eV_per_angstrom"]

        assert property_spec["input"] == structure_spec["input"]
        assert property_spec["contig"] == structure_spec["contig"]
        if "example_id" in structure_spec:
            assert property_spec["example_id"] == structure_spec["example_id"]
        for key, value in structure_spec.get("extra", {}).items():
            assert metadata[key] == value
        assert structure_spec["has_mean_force_norm"] is True
        for conditioning_key in (
            "mean_force_norm",
            "has_mean_force_norm",
            "force_rms",
            "force_p95",
            "has_force_rms_p95",
        ):
            assert conditioning_key not in property_spec
            assert conditioning_key not in metadata
        assert property_spec["select_fixed_atoms"] is True
        assert property_spec["select_unfixed_sequence"] is False
        assert metadata["atomistic_domain"] == domain
        assert metadata["atomistic_property_task"] == "mlip"
        assert math.isfinite(metadata["energy_gt_eV"])
        assert forces
        assert all(len(force) == 3 for force in forces)
        assert all(math.isfinite(component) for force in forces for component in force)
        atom_count_key = (
            "num_atoms_with_hydrogens" if domain == "omol25" else "num_atoms"
        )
        expected_atom_count = structure_spec.get("extra", {}).get(atom_count_key)
        if expected_atom_count is not None:
            assert len(forces) == expected_atom_count
        mean_force_norm = sum(math.dist(force, (0.0, 0.0, 0.0)) for force in forces)
        mean_force_norm /= len(forces)
        assert mean_force_norm == pytest.approx(
            structure_spec["mean_force_norm"], abs=1e-12
        )
        force_rms, force_p95 = _force_norm_statistics(forces)
        assert structure_spec["has_force_rms_p95"] is True
        assert structure_spec["force_rms"] == pytest.approx(force_rms, abs=1e-12)
        assert structure_spec["force_p95"] == pytest.approx(force_p95, abs=1e-12)

        if domain == "omol25":
            assert property_spec["example_id"] == structure_spec["example_id"]
            assert metadata["source_example_id"] == structure_spec["example_id"]
            assert (
                metadata["source_row_index"]
                == structure_spec["extra"]["source_row_index"]
            )
            assert isinstance(metadata["total_charge"], int)
            assert metadata["spin_multiplicity"] >= 1
        else:
            expected_source_id = structure_spec.get("extra", {}).get(
                "source_example_id", example_id
            )
            assert metadata["source_example_id"] == expected_source_id
            assert metadata["source_split"] == "val"


@pytest.mark.parametrize(
    ("filename", "expected_digest"),
    [
        (
            "omol25_val_energy_force_prediction_500.json",
            "1aff2b39ad73217addaa525a2225f9060b5fc4d6e024cf371331554f35bb29ad",
        ),
        (
            "omat24_val_energy_force_prediction_500.json",
            "1fd509101c2fdf2bf43291d3b0a59a497331b23ee72451d0e18925870ac4f34f",
        ),
    ],
)
def test_broad_mlip_validation_labels_have_source_digest(filename, expected_digest):
    """Lock the source identities and authoritative energy/force labels."""
    examples_dir = CONFIG_DIR.parent / "docs" / "examples"
    entries = json.loads((examples_dir / filename).read_text(encoding="utf-8"))
    digest_rows = []
    for entry_id, specification in entries.items():
        metadata = specification["extra"]
        source_identity = {
            key: metadata[key]
            for key in (
                "source_example_id",
                "source_shard",
                "source_row_index",
                "source_db_id",
                "sid",
            )
            if key in metadata
        }
        digest_rows.append(
            {
                "entry_id": entry_id,
                "source_identity": source_identity,
                "energy_gt_eV": metadata["energy_gt_eV"],
                "forces_gt_eV_per_angstrom": metadata["forces_gt_eV_per_angstrom"],
            }
        )

    payload = json.dumps(
        digest_rows,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert hashlib.sha256(payload).hexdigest() == expected_digest


def test_legacy_pretrain_does_not_enable_property_targets_or_head():
    """Test that the legacy pretraining experiment does not enable property targets or the property head."""
    config = _compose("pretrain")

    assert not config.datasets.enable_atomistic_property_targets
    assert "atomistic_property_head" not in config.model.net.diffusion_module
    assert "atomistic_property_loss" not in config.trainer.loss










