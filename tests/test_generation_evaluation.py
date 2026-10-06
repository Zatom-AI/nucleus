"""Focused tests for the checkpoint generation-evaluation runner."""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ZATOM2_SRC = Path(__file__).resolve().parents[1] / "models" / "zatom2" / "src"
if str(ZATOM2_SRC) not in sys.path:
    sys.path.insert(0, str(ZATOM2_SRC))

from zatom2.generation_evaluation import (  # noqa: E402
    _compute_structure_prediction_metrics,
    _conditioned_graph_features_from_reference,
    _conditioning_force,
    _flatten_outputs,
    _load_conditioned_molecule_structure_prediction_samples,
    _load_input_specifications,
    _load_saved_samples,
    _merge_output_shards,
    _molecule_from_shared_conditioned_graph,
    _select_input_specifications,
    compute_force_consistency_metrics,
)


def test_force_consistency_metrics_perfect_ranking_and_regression():
    metrics = compute_force_consistency_metrics(
        [0.0, 0.5, 1.0, 2.0],
        [0.0, 0.5, 1.0, 2.0],
        cutoff=1.0,
    )

    assert metrics["num_paired_samples"] == 4
    assert metrics["positive_target_count"] == 2
    assert metrics["negative_target_count"] == 2
    assert metrics["mae_eV_per_angstrom"] == 0.0
    assert metrics["auroc"] == 1.0
    assert metrics["auroc_by_ground_truth_cutoff_eV_per_angstrom"] == {
        "0.5": {
            "high_force_target_count": 3,
            "low_force_target_count": 1,
            "auroc": 1.0,
        },
        "0.75": {
            "high_force_target_count": 2,
            "low_force_target_count": 2,
            "auroc": 1.0,
        },
        "1.0": {
            "high_force_target_count": 2,
            "low_force_target_count": 2,
            "auroc": 1.0,
        },
    }
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["pearson"] == pytest.approx(1.0)


def test_force_consistency_metrics_report_undefined_statistics_as_none():
    metrics = compute_force_consistency_metrics([0.2, 0.2], [0.1, 0.3])

    assert metrics["auroc"] is None
    assert metrics["spearman"] is None
    assert metrics["pearson"] is None


def test_force_consistency_reports_auroc_for_each_ground_truth_bin_cutoff():
    metrics = compute_force_consistency_metrics(
        [0.2, 0.6, 0.8, 1.2],
        [0.9, 0.1, 0.8, 0.7],
        auroc_cutoffs=[0.5, 0.75, 1.0],
    )

    by_cutoff = metrics["auroc_by_ground_truth_cutoff_eV_per_angstrom"]
    assert by_cutoff["0.5"] == {
        "high_force_target_count": 3,
        "low_force_target_count": 1,
        "auroc": 0.0,
    }
    assert by_cutoff["0.75"] == {
        "high_force_target_count": 2,
        "low_force_target_count": 2,
        "auroc": pytest.approx(0.5),
    }
    assert by_cutoff["1.0"] == {
        "high_force_target_count": 1,
        "low_force_target_count": 3,
        "auroc": pytest.approx(1 / 3),
    }


def test_force_consistency_single_cutoff_override_skips_multi_cutoff_metrics():
    metrics = compute_force_consistency_metrics(
        [0.2, 0.8, 1.0, 2.0],
        [0.1, 0.7, 0.9, 1.8],
        cutoff=0.75,
        auroc_cutoffs=None,
    )

    assert metrics["cutoff_eV_per_angstrom"] == 0.75
    assert metrics["auroc"] == pytest.approx(1.0)
    assert "auroc_by_ground_truth_cutoff_eV_per_angstrom" not in metrics


def test_conditioning_force_supports_top_level_and_extra_layouts():
    assert _conditioning_force({"mean_force_norm": 0.25}) == 0.25
    assert (
        _conditioning_force(
            {"extra": {"mean_force_norm": 0.75, "has_mean_force_norm": True}}
        )
        == 0.75
    )
    assert (
        _conditioning_force({"mean_force_norm": 0.5, "has_mean_force_norm": False})
        is None
    )
    assert (
        _conditioning_force(
            {"extra": {"mean_force_norm": [0.9], "has_mean_force_norm": [True]}}
        )
        == 0.9
    )


def test_input_loader_matches_rfd3_global_args_and_path_resolution(tmp_path):
    input_path = tmp_path / "inputs.json"
    input_path.write_text(
        json.dumps(
            {
                "global_args": {"mean_force_norm": 0.8},
                "target": {"input": "reference.cif", "mean_force_norm": 0.2},
            }
        )
    )

    specification = _load_input_specifications(input_path)["target"]

    assert specification["mean_force_norm"] == 0.8
    assert specification["input"] == str((tmp_path / "reference.cif").resolve())


def test_expected_sample_count_limits_unconditional_inputs_before_inference():
    specifications = {f"target_{index}": {} for index in range(1000)}

    selected = _select_input_specifications(
        specifications,
        expected_num_samples=10,
        n_batches=1,
        diffusion_batch_size=1,
    )

    assert list(selected) == [f"target_{index}" for index in range(10)]


def test_expected_sample_count_accounts_for_candidates_per_target():
    specifications = {f"target_{index}": {} for index in range(500)}

    selected = _select_input_specifications(
        specifications,
        expected_num_samples=10,
        n_batches=2,
        diffusion_batch_size=1,
    )

    assert list(selected) == [f"target_{index}" for index in range(5)]


def test_expected_sample_count_is_capped_by_available_inputs():
    specifications = {f"target_{index}": {} for index in range(500)}

    selected = _select_input_specifications(
        specifications,
        expected_num_samples=1000,
        n_batches=1,
        diffusion_batch_size=1,
    )

    assert selected == specifications


def test_expected_sample_count_must_be_divisible_by_outputs_per_target():
    with pytest.raises(ValueError, match="must be divisible"):
        _select_input_specifications(
            {"target_a": {}, "target_b": {}},
            expected_num_samples=3,
            n_batches=2,
            diffusion_batch_size=1,
        )


def test_expected_sample_count_respects_explicit_json_key_order():
    specifications = {f"target_{index}": {} for index in range(5)}

    selected = _select_input_specifications(
        specifications,
        expected_num_samples=2,
        n_batches=1,
        diffusion_batch_size=1,
        json_keys_subset="target_4,target_2,target_0",
    )

    assert list(selected) == ["target_4", "target_2"]


def test_flatten_outputs_retains_target_batch_and_model_order():
    specifications = {"target_a": {"mean_force_norm": 0.1}}
    outputs = {
        "target_a_0": [
            SimpleNamespace(
                example_id="target_a_0_model_0",
                atom_array=object(),
                metadata={"specification": {"mean_force_norm": 0.1}},
            ),
            SimpleNamespace(
                example_id="target_a_0_model_1",
                atom_array=object(),
                metadata={"specification": {"mean_force_norm": 0.1}},
            ),
        ],
        "target_a_1": [
            SimpleNamespace(
                example_id="target_a_1_model_0",
                atom_array=object(),
                metadata={"specification": {"mean_force_norm": 0.1}},
            )
        ],
    }

    samples = _flatten_outputs(
        outputs,
        specifications,
        input_path="inputs.json",
        global_prefix="",
        n_batches=2,
    )

    assert [sample.sample_id for sample in samples] == [
        "target_a_0_model_0",
        "target_a_0_model_1",
        "target_a_1_model_0",
    ]
    assert [sample.batch_index for sample in samples] == [0, 0, 1]
    assert [sample.model_index for sample in samples] == [0, 1, 0]


def test_flatten_outputs_rejects_partial_sample_sets():
    with pytest.raises(ValueError, match="do not match"):
        _flatten_outputs(
            {},
            {"target_a": {}},
            input_path="inputs.json",
            global_prefix="",
            n_batches=1,
        )


def test_merge_output_shards_deduplicates_distributed_sampler_padding(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    rank_outputs = [
        {"target_a_0": ["rank-0-a"], "target_c_0": ["rank-0-c"]},
        {"target_b_0": ["rank-1-b"], "target_a_0": ["rank-1-a"]},
    ]
    for rank, outputs in enumerate(rank_outputs):
        with (shard_dir / f"rank-{rank:05d}.pkl").open("wb") as handle:
            pickle.dump(outputs, handle)

    merged = _merge_output_shards(shard_dir, world_size=2)

    assert merged == {
        "target_a_0": ["rank-0-a"],
        "target_c_0": ["rank-0-c"],
        "target_b_0": ["rank-1-b"],
    }


def test_load_saved_samples_reconstructs_manifest_and_conditioning(
    tmp_path, monkeypatch
):
    import nucleus.inference_input_parsing as input_parsing

    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    cif_path = samples_dir / "sample-0.cif.gz"
    cif_path.touch()
    metadata_path = samples_dir / "sample-0.json"
    metadata_path.write_text(
        json.dumps(
            {
                "ckpt_path": str(tmp_path / "checkpoint.ckpt"),
                "specification": {
                    "input": ["reference.cif"],
                    "example_id": ["data0001:2"],
                    "mean_force_norm": [0.25],
                },
            }
        )
    )
    (tmp_path / "sample_manifest.json").write_text(
        json.dumps(
            [
                {
                    "sample_index": 0,
                    "sample_id": "sample-0",
                    "target_id": "target-0",
                    "batch_index": 1,
                    "model_index": 0,
                    "cif_path": "samples/sample-0.cif.gz",
                    "metadata_path": "samples/sample-0.json",
                }
            ]
        )
    )
    atom_array = object()
    monkeypatch.setattr(
        input_parsing,
        "inference_load_preserving_hydrogens",
        lambda path: {"atom_array": atom_array},
    )

    samples = _load_saved_samples(
        tmp_path,
        expected_checkpoint=tmp_path / "checkpoint.ckpt",
    )

    assert len(samples) == 1
    assert samples[0].sample_id == "sample-0"
    assert samples[0].target_id == "target-0"
    assert samples[0].batch_index == 1
    assert samples[0].atom_array is atom_array
    assert _conditioning_force(samples[0].specification) == 0.25


def test_load_saved_samples_rejects_checkpoint_mismatch(tmp_path, monkeypatch):
    import nucleus.inference_input_parsing as input_parsing

    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    (samples_dir / "sample.cif.gz").touch()
    (samples_dir / "sample.json").write_text(
        json.dumps(
            {
                "ckpt_path": str(tmp_path / "original.ckpt"),
                "specification": {},
            }
        )
    )
    (tmp_path / "sample_manifest.json").write_text(
        json.dumps(
            [
                {
                    "sample_index": 0,
                    "sample_id": "sample",
                    "target_id": "target",
                    "batch_index": 0,
                    "model_index": 0,
                    "cif_path": "samples/sample.cif.gz",
                    "metadata_path": "samples/sample.json",
                }
            ]
        )
    )
    monkeypatch.setattr(
        input_parsing,
        "inference_load_preserving_hydrogens",
        lambda path: {"atom_array": object()},
    )

    with pytest.raises(ValueError, match="not the requested checkpoint"):
        _load_saved_samples(
            tmp_path,
            expected_checkpoint=tmp_path / "other.ckpt",
        )


def test_metrics_only_mode_skips_generation_and_sample_dump(tmp_path, monkeypatch):
    import zatom2.generation_evaluation as generation_evaluation

    input_path = tmp_path / "inputs.json"
    input_path.write_text(json.dumps({"target": {}}))

    sample = SimpleNamespace(
        sample_index=0,
        sample_id="sample",
        target_id="target",
        batch_index=0,
        model_index=0,
        specification={},
        atom_array=object(),
    )
    monkeypatch.setattr(
        generation_evaluation,
        "_load_saved_samples",
        lambda output_dir, expected_checkpoint: [sample],
    )
    monkeypatch.setattr(
        generation_evaluation,
        "_compute_generation_metrics",
        lambda samples, cfg: ({"validity": 1.0}, [object()]),
    )

    def fail_if_dumped(*args, **kwargs):
        raise AssertionError("metrics-only mode must not dump or regenerate samples")

    monkeypatch.setattr(generation_evaluation, "_dump_samples", fail_if_dumped)
    cfg = OmegaConf.create(
        {
            "out_dir": str(tmp_path),
            "ckpt_path": "checkpoint.ckpt",
            "inputs": str(input_path),
            "n_batches": 1,
            "diffusion_batch_size": 1,
            "evaluation": {
                "name": "test",
                "domain": "omol25",
                "task": "unconditional",
                "overwrite": False,
                "reuse_saved_samples": True,
                "expected_num_samples": 2,
                "afd": {"enabled": False},
                "structure_prediction": {"enabled": False},
                "uma": {"enabled": False},
            },
        }
    )

    metrics = generation_evaluation.run_generation_evaluation(cfg)

    assert metrics["generation"] == {"validity": 1.0}
    assert metrics["evaluation"]["reused_saved_samples"] is True
    assert cfg.evaluation.expected_num_samples == 1
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "metrics_resolved_config.yaml").is_file()
    assert not (tmp_path / "resolved_config.yaml").exists()


def test_material_evaluation_rejects_flat_cells_without_dropping_sample_indices(
    monkeypatch, caplog
):
    """The epoch-50 flat cell must not reach matching or periodic UMA graphs."""
    import zatom2.generation_evaluation as evaluation
    from biotite.structure import AtomArray
    from zevals.atomistic.uma import metrics as uma_metrics

    from nucleus.utils.materials import annotate_material_properties

    samples = []
    for index in range(3):
        array = AtomArray(2)
        array.element = np.asarray(["Na", "Cl"])
        lengths = np.asarray([4.0, 4.0, 4.0])
        angles = np.asarray([90.0, 90.0, 90.0])
        if index == 1:
            # Sample 047 from the reported run: long edges, nearly zero volume.
            lengths = np.asarray(
                [14.35395123559461, 18.14445663462707, 17.781171301989]
            )
            angles = np.asarray(
                [128.02024841308594, 133.3917236328125, 102.50572204589844]
            )
        annotate_material_properties(
            array,
            fractional_coordinates=np.asarray([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
            lattice_lengths=lengths,
            lattice_angles=angles,
        )
        samples.append(
            SimpleNamespace(
                sample_id=f"sample-{index}",
                target_id=f"target-{index}",
                atom_array=array,
                specification={},
            )
        )
    flat_cell = samples[1].atom_array.material_lattice_vectors.copy()
    assert abs(np.linalg.det(flat_cell[0])) > 1e-8

    converted = evaluation._convert_samples(samples, "omat24")

    assert len(converted) == 3
    assert converted[0] is not None and converted[2] is not None
    assert converted[1] is None
    assert "sample-1 has an unsafe periodic cell" in caplog.text
    np.testing.assert_array_equal(
        samples[1].atom_array.material_lattice_vectors, flat_cell
    )
    monkeypatch.setattr(
        evaluation,
        "_load_reference_samples",
        lambda samples, domain: {
            sample.target_id: [converted[0]] for sample in samples
        },
    )
    cfg = OmegaConf.create(
        {
            "domain": "omat24",
            "structure_prediction": {
                "ltol": 0.3,
                "stol": 0.5,
                "angle_tol": 10.0,
                "scale": True,
                "top_k": [1],
            },
            "uma": {"checkpoint": "unused", "device": "cpu", "batch_size": 8},
        }
    )
    result = evaluation._compute_structure_prediction_metrics(samples, converted, cfg)
    assert result["material_match_rate"] == pytest.approx(2 / 3)
    assert result["material_candidate_count"] == 3
    assert result["material_invalid_candidate_count"] == 1
    assert result["material_matcher_error_count"] == 0

    def score_valid_samples(atoms_inputs, **kwargs):
        assert len(atoms_inputs) == 3
        assert atoms_inputs[1] is None
        assert atoms_inputs[0].pbc.all() and atoms_inputs[2].pbc.all()
        return [
            {
                "sample_index": index,
                "force_max": 1.0,
                "mean_force_norm": 0.5,
                "force_rms": 0.5,
                "energy_per_atom": -1.0,
            }
            for index in (0, 2)
        ]

    monkeypatch.setattr(uma_metrics, "score_uma_samples", score_valid_samples)
    metrics, scores = evaluation._score_uma(samples, converted, cfg)
    assert metrics["uma_scored_rate"] == pytest.approx(2 / 3)
    assert [score["sample_index"] for score in scores] == [0, 2]


@pytest.mark.parametrize("partial_t", [None, 0.0, 1.5])
def test_structure_prediction_uses_one_conditioned_graph_for_reference_and_candidates(
    monkeypatch,
    partial_t,
):
    import zatom2.generation_evaluation as generation_evaluation

    import nucleus.inference_input_parsing as input_parsing
    import nucleus.utils.atom_array as atom_array_utils

    class FakeAtomArray:
        def __len__(self):
            return 2

    reference_atom_array = FakeAtomArray()
    candidate_atom_array = FakeAtomArray()
    shared_features = {"token_bonds": object(), "atom_to_token_map": object()}
    attached = []

    monkeypatch.setattr(
        input_parsing,
        "inference_load_preserving_hydrogens",
        lambda path: {"atom_array": reference_atom_array},
    )

    class FakeSidecarLoader:
        def attach_bonds(self, atom_array, *, example_id):
            attached.append((atom_array, example_id))

    graph_inputs = []

    def fake_graph_features(atom_array, *, target_id, partial_t):
        graph_inputs.append((atom_array, target_id, partial_t))
        return shared_features

    monkeypatch.setattr(
        generation_evaluation,
        "_conditioned_graph_features_from_reference",
        fake_graph_features,
    )
    monkeypatch.setattr(
        atom_array_utils,
        "_atomic_numbers_from_atom_array",
        lambda atom_array, count: [6, 8],
    )
    built = []

    def fake_build(
        atom_array,
        features,
        *,
        reference_atomic_numbers,
        target_id,
        role,
    ):
        built.append((atom_array, features, reference_atomic_numbers, target_id, role))
        return f"{role}-molecule"

    monkeypatch.setattr(
        generation_evaluation,
        "_molecule_from_shared_conditioned_graph",
        fake_build,
    )
    sample = SimpleNamespace(
        target_id="target-037",
        specification={
            "input": ["reference.cif"],
            "example_id": ["data0014:3558"],
            "partial_t": [partial_t],
        },
        atom_array=candidate_atom_array,
    )

    generated, references = _load_conditioned_molecule_structure_prediction_samples(
        [sample],
        sidecar_loader=FakeSidecarLoader(),
    )

    assert attached == [(reference_atom_array, "data0014:3558")]
    assert graph_inputs == [(reference_atom_array, "target-037", partial_t)]
    assert references == {"target-037": ["reference-molecule"]}
    assert generated == {"target-037": ["candidate-0-molecule"]}
    assert [row[0] for row in built] == [reference_atom_array, candidate_atom_array]
    assert all(row[1] is shared_features for row in built)
    assert [row[4] for row in built] == ["reference", "candidate-0"]


def test_shared_conditioned_graph_reconstruction_does_not_perceive_candidate_bonds():
    from biotite.structure import AtomArray, BondList
    from zevals.molecules.structure_prediction import _heavy_atom_graph_id

    reference = AtomArray(3)
    reference.element = np.asarray(["C", "O", "H"])
    reference.coord = np.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [-0.8, 0.0, 0.0]])
    reference.bonds = BondList(3, np.asarray([[0, 1], [0, 2]]))
    candidate = reference.copy()
    candidate.coord = np.asarray([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
    candidate.bonds = BondList(3)

    features = _conditioned_graph_features_from_reference(
        reference,
        target_id="target",
    )
    atomic_numbers = np.asarray([6, 8, 1])
    reference_molecule = _molecule_from_shared_conditioned_graph(
        reference,
        features,
        reference_atomic_numbers=atomic_numbers,
        target_id="target",
        role="reference",
    )
    candidate_molecule = _molecule_from_shared_conditioned_graph(
        candidate,
        features,
        reference_atomic_numbers=atomic_numbers,
        target_id="target",
        role="candidate-0",
    )

    assert _heavy_atom_graph_id(candidate_molecule) == _heavy_atom_graph_id(
        reference_molecule
    )


def test_conditioned_graph_preserves_long_sidecar_bonds_during_reconstruction():
    """Zero-coordinate inference retains bonds beyond the native 2.4-A cutoff."""
    from biotite.structure import AtomArray, BondList

    reference = AtomArray(3)
    reference.coord = np.asarray([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    reference.bonds = BondList(3, np.asarray([[0, 1]]))

    features = _conditioned_graph_features_from_reference(reference, target_id="target")

    assert features["token_bonds"].tolist() == [
        [False, True, False],
        [True, False, False],
        [False, False, False],
    ]
    assert features["atom_to_token_map"].tolist() == [0, 1, 2]


@pytest.mark.parametrize("partial_t", [0.0, 1.5])
def test_partial_diffusion_graph_uses_native_distance_cutoff(partial_t):
    """Even zero partial noise preserves native coordinates and filters long bonds."""
    from biotite.structure import AtomArray, BondList

    reference = AtomArray(3)
    reference.coord = np.asarray([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    reference.bonds = BondList(3, np.asarray([[0, 1], [0, 2]]))
    annotations_before = reference.get_annotation_categories()
    coordinates_before = reference.coord.copy()

    features = _conditioned_graph_features_from_reference(
        reference, target_id="partial-target", partial_t=partial_t
    )

    assert features["token_bonds"].tolist() == [
        [False, False, True],
        [False, False, False],
        [True, False, False],
    ]
    assert features["atom_to_token_map"].tolist() == [0, 1, 2]
    assert reference.get_annotation_categories() == annotations_before
    np.testing.assert_array_equal(reference.coord, coordinates_before)


def test_structure_prediction_does_not_attach_sidecar_to_generated_molecules(
    monkeypatch,
):
    import zatom2.generation_evaluation as generation_evaluation
    import zevals.molecules.structure_prediction as structure_prediction

    import nucleus.utils.bond_sidecar as bond_sidecar_utils

    class FakeSidecarLoader:
        def __init__(self, *, bond_sidecar_path):
            self.bond_sidecar_path = bond_sidecar_path

        def close(self):
            pass

    loader_seen = {}

    def fake_load_conditioned(samples, *, sidecar_loader):
        loader_seen["loader"] = sidecar_loader
        return (
            {"target": [converted_candidate]},
            {"target": ["reference-molecule"]},
        )

    captured = {}

    def fake_metrics(generated, references, *, top_k):
        captured["generated"] = generated
        captured["references"] = references
        captured["top_k"] = top_k
        return {"molecule_best_rmsd_mean_top1": 0.5}

    monkeypatch.setattr(
        bond_sidecar_utils,
        "BondSidecarASEAtomsLoader",
        FakeSidecarLoader,
    )
    monkeypatch.setattr(
        generation_evaluation,
        "_load_conditioned_molecule_structure_prediction_samples",
        fake_load_conditioned,
    )
    monkeypatch.setattr(
        structure_prediction,
        "compute_molecule_structure_prediction_metrics",
        fake_metrics,
    )
    sample = SimpleNamespace(target_id="target")
    converted_candidate = object()
    cfg = OmegaConf.create(
        {
            "domain": "omol25",
            "bond_sidecar_path": "bonds.lmdb",
            "structure_prediction": {"top_k": [1]},
        }
    )

    result = _compute_structure_prediction_metrics(
        [sample],
        [converted_candidate],
        cfg,
    )

    assert loader_seen["loader"].bond_sidecar_path == "bonds.lmdb"
    assert captured == {
        "generated": {"target": [converted_candidate]},
        "references": {"target": ["reference-molecule"]},
        "top_k": [1],
    }
    assert result == {"molecule_best_rmsd_mean_top1": 0.5}


@pytest.mark.parametrize(
    ("preset", "num_targets", "structure_prediction"),
    [
        ("omol25_unconditional", 1000, False),
        ("omol25_structure_prediction", 500, True),
        ("omat24_unconditional", 1000, False),
        ("omat24_structure_prediction", 500, True),
    ],
)
def test_evaluation_presets_request_one_sample_per_target(
    preset, num_targets, structure_prediction
):
    repository_root = Path(__file__).resolve().parents[1]
    config_dir = repository_root / "models" / "zatom2" / "configs"
    os.environ["PROJECT_ROOT"] = str(repository_root)
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="generation_evaluation",
            overrides=[
                f"evaluation={preset}",
                "ckpt_path=checkpoint.ckpt",
                "out_dir=outputs",
            ],
        )

    specifications = _load_input_specifications(cfg.inputs)
    assert len(specifications) == num_targets
    assert cfg.n_batches == 1
    assert cfg.diffusion_batch_size == 1
    assert cfg.inference_sampler.use_classifier_free_guidance is True
    assert cfg.inference_sampler.cfg_feature_mode == "force_only"
    assert cfg.inference_sampler.cfg_scale == 2.0
    assert cfg.evaluation.reuse_saved_samples is False
    assert cfg.evaluation.force_consistency.auroc_cutoff_override is None
    assert list(cfg.evaluation.force_consistency.auroc_cutoffs) == [0.5, 0.75, 1.0]
    assert cfg.evaluation.structure_prediction.enabled is structure_prediction
    assert len(specifications) * cfg.n_batches * cfg.diffusion_batch_size == num_targets
    expected_bond_sidecar = (
        str(repository_root / "data/omol25/sidecars/val_all_bonds.lmdb")
        if preset == "omol25_structure_prediction"
        else None
    )
    assert cfg.bond_sidecar_path == expected_bond_sidecar


@pytest.mark.parametrize(
    ("overrides", "enabled", "feature_mode", "scale"),
    [
        (
            ["classifier_free_guidance=force_only", "inference_sampler.cfg_scale=3.5"],
            True,
            "force_only",
            3.5,
        ),
        (
            ["inference_sampler.use_classifier_free_guidance=false"],
            False,
            "force_only",
            2.0,
        ),
        (["~classifier_free_guidance"], False, "configured", 2.0),
    ],
)
def test_evaluation_guidance_overrides_take_precedence(
    overrides, enabled, feature_mode, scale
):
    config_dir = Path(__file__).resolve().parents[1] / "models" / "zatom2" / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="generation_evaluation", overrides=overrides)

    assert cfg.inference_sampler.use_classifier_free_guidance is enabled
    assert cfg.inference_sampler.cfg_feature_mode == feature_mode
    assert cfg.inference_sampler.cfg_scale == scale
