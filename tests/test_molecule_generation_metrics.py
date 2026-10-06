from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest


@dataclass
class FakeMol:
    """A fake molecule class for testing purposes."""

    label: str

    def GetNumAtoms(self) -> int:
        """Return the number of atoms based on the label length."""
        return max(len(self.label.replace(".", "")), 1)

    def GetNumConformers(self) -> int:
        """Return a fixed number of conformers for testing."""
        return 1


def _patch_metric_helpers(monkeypatch: pytest.MonkeyPatch, module) -> None:
    """Replace the heavy helper functions with deterministic fakes."""

    def fake_smiles_from_atom_array(atom_array, *, example_id):
        """Convert an atom array to a SMILES string, with deterministic behavior."""
        del example_id
        if atom_array.label == "bad":
            return None
        return atom_array.label.lower()

    def fake_atom_array_to_mol(atom_array):
        """Convert an atom array to a molecule object, with deterministic behavior."""
        if atom_array.label == "bad":
            return None
        return FakeMol(atom_array.label.lower())

    def fake_validity(molecules, frag_check=True):
        """Compute validity metrics for a list of molecule objects."""
        del frag_check
        valid_mols = [mol for mol in molecules if mol is not None]
        return {
            "num_valid": len(valid_mols),
            "valid_mols": valid_mols,
            "valid_smiles": [mol.label for mol in valid_mols],
        }

    def fake_uniqueness(smiles):
        """Compute uniqueness metrics for a list of SMILES strings."""
        assert smiles == ["a", "a", "b"]
        return {"num_unique": len(set(smiles))}

    def fake_diversity(smiles):
        """Compute diversity metrics for a list of SMILES strings."""
        assert smiles == ["a", "a", "b"]
        return {"diversity_rate": 1.0 if smiles else 0.0}

    def fake_novelty(smiles, reference):
        """Compute novelty metrics for a list of SMILES strings against a reference set."""
        assert smiles == ["a", "a", "b"]
        return {"num_novel": len(set(smiles) - set(reference))}

    def fake_posebusters(valid_mols):
        """Compute posebusters metrics for a list of valid molecule objects."""
        clash_rate = (len(valid_mols) - 1) / len(valid_mols) if valid_mols else 0.0
        metrics = {key: 1.0 for key in module.POSEBUSTERS_KEYS}
        metrics["internal_steric_clash"] = clash_rate
        metrics["posebusters_rate"] = clash_rate
        return metrics

    def fake_uff_mean_force_norm(mol, *, row_id="generated_molecule"):
        """Compute a deterministic fake force norm for valid molecule objects."""
        del row_id
        return {"a": 1.0, "b": 10.0}[mol.label]

    monkeypatch.setattr(
        module, "_rdkit_smiles_for_valid_atom_array", fake_smiles_from_atom_array
    )
    monkeypatch.setattr(module, "_atom_array_to_mol", fake_atom_array_to_mol)
    monkeypatch.setattr(module, "compute_mol_validity", fake_validity)
    monkeypatch.setattr(module, "compute_smiles_uniqueness", fake_uniqueness)
    monkeypatch.setattr(module, "compute_smiles_diversity", fake_diversity)
    monkeypatch.setattr(module, "compute_smiles_novelty", fake_novelty)
    monkeypatch.setattr(module, "compute_posebusters_metrics", fake_posebusters)
    monkeypatch.setattr(module, "_mol_uff_mean_force_norm", fake_uff_mean_force_norm)


def test_molecule_generation_metrics_match_nucleus_adapter_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the molecule generation metrics compute the expected values and match the Nucleus adapter contract."""
    import nucleus.metrics.molecule_generation as molecule_generation

    _patch_metric_helpers(monkeypatch, molecule_generation)

    predicted = [
        SimpleNamespace(label="A", element=np.asarray(["C"], dtype=object)),
        SimpleNamespace(label="a", element=np.asarray(["C"], dtype=object)),
        SimpleNamespace(label="B", element=np.asarray(["C"], dtype=object)),
        SimpleNamespace(label="bad", element=np.asarray(["C"], dtype=object)),
    ]
    reference = [SimpleNamespace(label="B", element=np.asarray(["C"], dtype=object))]

    metrics = molecule_generation.compute_molecule_generation_metrics(
        predicted_atom_array_stack=predicted,
        ground_truth_atom_array_stack=reference,
    )

    assert metrics["valid_rate"] == pytest.approx(0.75)
    assert metrics["unique_rate"] == pytest.approx(2 / 3)
    assert metrics["novel_rate"] == pytest.approx(1 / 2)
    assert metrics["posebusters_rate"] == pytest.approx(2 / 3)
    assert metrics["uff_median_force_norm"] == pytest.approx(1.0)
    assert metrics["uff_median_force_norm_success_rate"] == pytest.approx(0.75)
    assert "afd_score" not in metrics
    assert "afd_input_success_rate" not in metrics
    assert "internal_steric_clash" in molecule_generation.POSEBUSTERS_KEYS

    metric = molecule_generation.MoleculeGenerationMetrics()
    via_metric = metric.compute_from_kwargs(
        predicted_atom_array_stack=predicted,
        ground_truth_atom_array_stack=reference,
        extra_info={"example_id": "example-1"},
    )

    assert via_metric == metrics


def test_molecule_generation_metrics_compute_uff_mae_from_inference_target(
    monkeypatch: pytest.MonkeyPatch,
):
    """The inference JSON mean-force target is applied to each UFF score."""
    import nucleus.metrics.molecule_generation as molecule_generation

    _patch_metric_helpers(monkeypatch, molecule_generation)

    predicted = [
        SimpleNamespace(label="A", element=np.asarray(["C"], dtype=object)),
        SimpleNamespace(label="a", element=np.asarray(["C"], dtype=object)),
        SimpleNamespace(label="B", element=np.asarray(["C"], dtype=object)),
        SimpleNamespace(label="bad", element=np.asarray(["C"], dtype=object)),
    ]
    metrics = molecule_generation.compute_molecule_generation_metrics(
        predicted_atom_array_stack=predicted,
        extra_info={
            "specification": {
                "has_mean_force_norm": [True],
                "mean_force_norm": [0.2],
            }
        },
    )

    # Fake UFF norms are [1, 1, 10] kcal/mol/A and the target is in eV/A.
    conversion = 0.0433641153087705
    assert metrics["uff_mean_force_norm_mae_eV_per_angstrom"] == pytest.approx(
        (abs(conversion - 0.2) * 2 + abs(10 * conversion - 0.2)) / 3
    )


def test_atom_array_to_afd_record_uses_coordinates_and_atomic_numbers():
    """Test conversion from AtomArray-like objects to AFD featurization records."""
    from nucleus.utils.atom_array import _atom_array_to_afd_record

    class FakeAtomArray:
        coord = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        atomic_number = np.asarray([6, 6], dtype=np.int64)
        element = np.asarray(
            ["C", "Br"], dtype=object
        )  # Include a non-carbon atom to test element handling

    record = _atom_array_to_afd_record(FakeAtomArray())

    assert record is not None
    assert record["z"].tolist() == [6, 6]
    assert record["pos"].shape == (2, 3)


def test_atom_array_to_material_afd_record_preserves_periodic_cell():
    """Test conversion of material atoms, lattice vectors, and PBC for AFD."""
    from biotite.structure import AtomArray

    from nucleus.utils.atom_array import _atom_array_to_material_afd_record
    from nucleus.utils.materials import annotate_material_properties

    atom_array = AtomArray(3)
    atom_array.coord = np.asarray(
        [[0.0, 0.0, 0.0], [1.5, 1.5, 1.5], [4.0, 4.0, 4.0]],
        dtype=np.float32,
    )
    atom_array.element = np.asarray(["Si", "O", "C"], dtype=object)
    lattice_vectors = np.diag([5.0, 6.0, 7.0])
    annotate_material_properties(
        atom_array,
        fractional_coordinates=np.asarray(
            [[0.0, 0.0, 0.0], [0.3, 0.25, 1.5 / 7.0], [0.8, 2 / 3, 4 / 7]],
        ),
        lattice_lengths=np.asarray([5.0, 6.0, 7.0]),
        lattice_angles=np.asarray([90.0, 90.0, 90.0]),
        lattice_vectors=lattice_vectors,
        is_material=np.asarray([True, True, False]),
    )

    record = _atom_array_to_material_afd_record(atom_array)

    assert record is not None
    assert record["z"].tolist() == [14, 8]
    assert record["pos"].shape == (2, 3)
    np.testing.assert_allclose(record["cell"], lattice_vectors[None, :, :])
    np.testing.assert_array_equal(record["pbc"], np.ones((1, 3), dtype=bool))


def test_atom_array_to_material_afd_record_requires_periodic_annotations():
    """Test that molecule-like AtomArrays are rejected by the materials adapter."""
    from biotite.structure import AtomArray

    from nucleus.utils.atom_array import _atom_array_to_material_afd_record

    atom_array = AtomArray(1)
    atom_array.coord = np.zeros((1, 3), dtype=np.float32)
    atom_array.element = np.asarray(["C"], dtype=object)

    assert _atom_array_to_material_afd_record(atom_array) is None


def test_atom_array_to_material_afd_record_rejects_near_singular_cell():
    """AFD skips generated cells that would require excessive PBC replicas."""
    from biotite.structure import AtomArray

    from nucleus.utils.atom_array import _atom_array_to_material_afd_record
    from nucleus.utils.materials import annotate_material_properties

    atom_array = AtomArray(1)
    atom_array.coord = np.zeros((1, 3), dtype=np.float32)
    atom_array.element = np.asarray(["K"], dtype=object)
    annotate_material_properties(
        atom_array,
        fractional_coordinates=np.zeros((1, 3)),
        lattice_lengths=np.asarray(
            [10.667398282094606, 15.989673645197463, 14.183183546629506]
        ),
        lattice_angles=np.asarray(
            [132.49649047851562, 128.02024841308594, 106.08670806884766]
        ),
    )

    # This is the epoch-120 regression cell. Its determinant passed the old
    # absolute 1e-8 check despite the cell being effectively two-dimensional.
    lattice_vectors = atom_array.material_lattice_vectors[0]
    assert abs(float(np.linalg.det(lattice_vectors))) > 1e-8
    assert _atom_array_to_material_afd_record(atom_array) is None


def test_molecule_generation_metrics_omit_disabled_afd_keys(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that per-step molecule-generation metrics never emit AFD keys."""
    import nucleus.metrics.molecule_generation as molecule_generation

    _patch_metric_helpers(monkeypatch, molecule_generation)

    metrics = molecule_generation.compute_molecule_generation_metrics(
        predicted_atom_array_stack=[FakeMol("A"), FakeMol("a"), FakeMol("B")],
    )

    assert "afd_score" not in metrics
    assert "afd_num_samples" not in metrics
    assert "afd_ref_sample_size" not in metrics
    assert "afd_input_success_rate" not in metrics


def test_epoch_molecule_afd_callback_accumulates_and_appends_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    """Test epoch-level AFD accumulation over all validation batches for one dataset."""
    import pandas as pd
    import zevals.molecules.afd as afd_module

    import nucleus.callbacks.metrics_logging as metrics_logging

    captured = {}

    def fake_atom_array_to_afd_record(atom_array):
        if atom_array.label == "bad":
            return None
        return {
            "z": np.asarray([6], dtype=np.int64),
            "pos": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            "label": atom_array.label,
        }

    def fake_get_afd_scorer(self):
        captured["scorer_model_name"] = self.afd_model_name
        return "cached-scorer"

    def fake_afd(records, **kwargs):
        captured["records"] = list(records)
        captured["kwargs"] = kwargs
        return {
            "afd_score": 0.75,
            "afd_num_samples": float(len(records)),
            "afd_ref_sample_size": float(len(records)),
        }

    monkeypatch.setattr(
        metrics_logging,
        "_atom_array_to_afd_record",
        fake_atom_array_to_afd_record,
    )
    monkeypatch.setattr(
        metrics_logging.LogMoleculeAFDValidationMetricsCallback,
        "_get_afd_scorer",
        fake_get_afd_scorer,
    )
    monkeypatch.setattr(afd_module, "compute_afd_metrics", fake_afd)

    class FakeFabric:
        global_rank = 0
        world_size = 1
        is_global_zero = True

        def __init__(self):
            self.logged = []

        def log_dict(self, metrics, step=None):
            self.logged.append((metrics, step))

    validation_results_path = tmp_path / "validation_output_all_epochs.csv"
    pd.DataFrame(
        [
            {
                "example_id": "example-1",
                "epoch": 5,
                "dataset": "omol25",
                "valid_rate": 1.0,
            }
        ]
    ).to_csv(validation_results_path, index=False)
    fabric = FakeFabric()
    trainer = SimpleNamespace(
        fabric=fabric,
        state={"current_epoch": 5, "global_step": 123},
        validation_results_path=validation_results_path,
    )
    callback = metrics_logging.LogMoleculeAFDValidationMetricsCallback(
        enabled=True,
        afd_reference_records_path="reference.pt",
        afd_model_name="test-model",
    )

    callback.on_validation_epoch_start(trainer)
    callback.on_validation_batch_end(
        trainer,
        outputs={
            "predicted_atom_array_stack": [
                SimpleNamespace(label="A", element=np.asarray(["C"], dtype=object)),
                SimpleNamespace(label="bad", element=np.asarray(["C"], dtype=object)),
            ]
        },
        batch=None,
        batch_idx=0,
        num_batches=2,
        dataset_name="omol25",
    )
    callback.on_validation_batch_end(
        trainer,
        outputs={
            "predicted_atom_array_stack": [
                SimpleNamespace(label="B", element=np.asarray(["C"], dtype=object))
            ]
        },
        batch=None,
        batch_idx=1,
        num_batches=2,
        dataset_name="omol25",
    )
    callback.on_validation_epoch_end(trainer)

    assert [record["label"] for record in captured["records"]] == ["A", "B"]
    assert captured["kwargs"]["scorer"] == "cached-scorer"
    assert captured["kwargs"]["model_name"] == "test-model"
    assert captured["scorer_model_name"] == "test-model"
    assert fabric.logged == []

    df = pd.read_csv(validation_results_path)
    afd_row = df[df["example_id"] == metrics_logging.EPOCH_AFD_EXAMPLE_ID].iloc[0]
    assert afd_row["dataset"] == "omol25"
    assert afd_row["afd_score"] == pytest.approx(0.75)
    assert afd_row["afd_num_samples"] == pytest.approx(2.0)
    assert afd_row["afd_ref_sample_size"] == pytest.approx(2.0)
    assert afd_row["afd_input_success_rate"] == pytest.approx(2 / 3)
    assert afd_row["atom_fraction_o"] == pytest.approx(0.0)
    assert afd_row["atom_fraction_n"] == pytest.approx(0.0)
    assert afd_row["atom_fraction_c"] == pytest.approx(1.0)


def test_epoch_materials_afd_callback_uses_periodic_scorer(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test materials AFD collection and the explicit Hugging Face model ID."""
    import zevals.materials.afd as afd_module

    import nucleus.callbacks.metrics_logging as metrics_logging

    captured = {}

    def fake_material_record(atom_array):
        if atom_array.label == "bad":
            return None
        return {
            "z": np.asarray([14], dtype=np.int64),
            "pos": np.zeros((1, 3), dtype=np.float32),
            "cell": np.eye(3, dtype=np.float32)[None, :, :],
            "pbc": np.ones((1, 3), dtype=bool),
            "label": atom_array.label,
        }

    def fake_get_afd_scorer(self):
        captured["scorer_model_name"] = self.afd_model_name
        return "materials-scorer"

    def fake_afd(records, **kwargs):
        captured["records"] = list(records)
        captured["kwargs"] = kwargs
        return {
            "afd_score": 1.25,
            "afd_num_samples": float(len(records)),
            "afd_ref_sample_size": float(len(records)),
        }

    monkeypatch.setattr(
        metrics_logging,
        "_atom_array_to_material_afd_record",
        fake_material_record,
    )
    monkeypatch.setattr(
        metrics_logging.LogMaterialsAFDValidationMetricsCallback,
        "_get_afd_scorer",
        fake_get_afd_scorer,
    )
    monkeypatch.setattr(afd_module, "compute_afd_metrics", fake_afd)

    class FakeFabric:
        global_rank = 0
        world_size = 1
        is_global_zero = True

        def __init__(self):
            self.logged = []

        def log_dict(self, metrics, step=None):
            self.logged.append((metrics, step))

    fabric = FakeFabric()
    trainer = SimpleNamespace(
        fabric=fabric,
        state={"current_epoch": 2, "global_step": 20},
    )
    callback = metrics_logging.LogMaterialsAFDValidationMetricsCallback(
        enabled=True,
        append_to_validation_results=False,
    )

    callback.on_validation_epoch_start(trainer)
    callback.on_validation_batch_end(
        trainer,
        outputs={
            "predicted_atom_array_stack": [
                SimpleNamespace(label="A"),
                SimpleNamespace(label="bad"),
                SimpleNamespace(label="B"),
            ]
        },
        batch=None,
        batch_idx=0,
        num_batches=1,
        dataset_name="omat24",
    )
    callback.on_validation_epoch_end(trainer)

    assert [record["label"] for record in captured["records"]] == ["A", "B"]
    assert captured["scorer_model_name"] == "Ty-Perez/ct-scd-amp20"
    assert captured["kwargs"]["scorer"] == "materials-scorer"
    assert captured["kwargs"]["reference"] == "mp20"
    assert captured["kwargs"]["data_modality"] == "materials"
    assert captured["kwargs"]["model_name"] == "Ty-Perez/ct-scd-amp20"

    assert len(fabric.logged) == 1
    logged_metrics, step = fabric.logged[0]
    assert step == 20
    assert logged_metrics == {
        "val/omat24/afd_score": pytest.approx(1.25),
        "val/omat24/afd_num_samples": pytest.approx(2.0),
        "val/omat24/afd_ref_sample_size": pytest.approx(2.0),
        "val/omat24/afd_input_success_rate": pytest.approx(2 / 3),
    }


def test_materials_afd_callback_builds_mp20_scorer(monkeypatch):
    """Test the materials callback's reference and checkpoint selection."""
    import zevals.materials.afd as afd_module

    import nucleus.callbacks.metrics_logging as metrics_logging

    captured = {}

    class FakeAFDScore:
        def __init__(self, reference, **kwargs):
            captured["reference"] = reference
            captured["kwargs"] = kwargs

    monkeypatch.setattr(afd_module, "AFDScore", FakeAFDScore, raising=False)
    callback = metrics_logging.LogMaterialsAFDValidationMetricsCallback(enabled=True)

    scorer = callback._get_afd_scorer()

    assert isinstance(scorer, FakeAFDScore)
    assert captured["reference"] == "mp20"
    assert captured["kwargs"] == {
        "data_modality": "materials",
        "model_name": "Ty-Perez/ct-scd-amp20",
        "batch_size": 256,
        "device": None,
        "feature_cache_dir": None,
    }


def test_epoch_molecule_atom_fraction_metrics_count_all_generated_atoms(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test O/N/C fractions aggregate over all generated atoms at epoch end."""
    import nucleus.callbacks.metrics_logging as metrics_logging

    class FakeFabric:
        global_rank = 0
        world_size = 1
        is_global_zero = True

        def __init__(self):
            self.logged = []

        def log_dict(self, metrics, step=None):
            self.logged.append((metrics, step))

    def fake_atom_array_to_afd_record(atom_array):
        if atom_array.label == "not_afd_compatible":
            return None
        return {
            "z": np.asarray([6, 8, 1], dtype=np.int64),
            "pos": np.zeros((3, 3), dtype=np.float32),
        }

    monkeypatch.setattr(
        metrics_logging,
        "_atom_array_to_afd_record",
        fake_atom_array_to_afd_record,
    )

    fabric = FakeFabric()
    trainer = SimpleNamespace(
        fabric=fabric,
        state={"current_epoch": 0, "global_step": 10},
    )
    callback = metrics_logging.LogMoleculeAFDValidationMetricsCallback(
        enabled=True,
        append_to_validation_results=False,
    )

    callback.on_validation_epoch_start(trainer)
    callback.on_validation_batch_end(
        trainer,
        outputs={
            "predicted_atom_array_stack": [
                SimpleNamespace(
                    label="afd_compatible",
                    coord=np.zeros((3, 3), dtype=np.float32),
                    element=np.asarray(["C", "O", "H"], dtype=object),
                    atomic_number=np.asarray([6, 6, 6], dtype=np.int64),
                ),
                SimpleNamespace(
                    label="not_afd_compatible",
                    coord=np.zeros((2, 3), dtype=np.float32),
                    element=np.asarray(["N", "C"], dtype=object),
                    atomic_number=np.asarray([6, 6], dtype=np.int64),
                ),
            ]
        },
        batch=None,
        batch_idx=0,
        num_batches=1,
        dataset_name="omol25",
    )
    callback.on_validation_epoch_end(trainer)

    assert len(fabric.logged) == 1
    logged_metrics, step = fabric.logged[0]
    assert step == 10
    assert logged_metrics["val/omol25/afd_num_samples"] == pytest.approx(1.0)
    assert logged_metrics["val/omol25/afd_input_success_rate"] == pytest.approx(1 / 2)
    assert logged_metrics["val/omol25/atom_fraction_o"] == pytest.approx(1 / 5)
    assert logged_metrics["val/omol25/atom_fraction_n"] == pytest.approx(1 / 5)
    assert logged_metrics["val/omol25/atom_fraction_c"] == pytest.approx(2 / 5)


def test_epoch_afd_callbacks_filter_validation_datasets(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test molecule and material AFD collect only allowlisted loaders."""
    import nucleus.callbacks.metrics_logging as metrics_logging

    monkeypatch.setattr(
        metrics_logging.LogMoleculeAFDValidationMetricsCallback,
        "_atom_array_to_afd_record",
        staticmethod(lambda atom_array: {"label": atom_array.label}),
    )
    monkeypatch.setattr(
        metrics_logging.LogMaterialsAFDValidationMetricsCallback,
        "_atom_array_to_afd_record",
        staticmethod(lambda atom_array: {"label": atom_array.label}),
    )

    trainer = SimpleNamespace()
    molecule_callback = metrics_logging.LogMoleculeAFDValidationMetricsCallback(
        enabled=True,
        dataset_names=["omol25_generation"],
    )
    material_callback = metrics_logging.LogMaterialsAFDValidationMetricsCallback(
        enabled=True,
        dataset_names=["omat24_generation"],
    )
    callbacks = (molecule_callback, material_callback)
    for callback in callbacks:
        callback.on_validation_epoch_start(trainer)

    for dataset_name in (
        "omol25_generation",
        "omat24_generation",
        "omol25_structure_prediction",
        "omat24_structure_prediction",
        "omol25_mlip",
        "omat24_mlip",
    ):
        outputs = {"predicted_atom_array_stack": [SimpleNamespace(label=dataset_name)]}
        for callback in callbacks:
            callback.on_validation_batch_end(
                trainer,
                outputs=outputs,
                batch=None,
                batch_idx=0,
                num_batches=1,
                dataset_name=dataset_name,
            )

    assert molecule_callback._local_payload() == {
        "omol25_generation": {
            "records": [{"label": "omol25_generation"}],
            "total_samples": 1,
            "atom_counts": {
                "atom_fraction_o": 0,
                "atom_fraction_n": 0,
                "atom_fraction_c": 0,
            },
            "total_atoms": 0,
        }
    }
    assert material_callback._local_payload() == {
        "omat24_generation": {
            "records": [{"label": "omat24_generation"}],
            "total_samples": 1,
            "atom_counts": {
                "atom_fraction_o": 0,
                "atom_fraction_n": 0,
                "atom_fraction_c": 0,
            },
            "total_atoms": 0,
        }
    }


def test_epoch_molecule_atom_fraction_metrics_can_run_without_afd(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test atom-fraction metrics can run without AFD conversion or scoring."""
    import nucleus.callbacks.metrics_logging as metrics_logging

    class FakeFabric:
        global_rank = 0
        world_size = 1
        is_global_zero = True

        def __init__(self):
            self.logged = []

        def log_dict(self, metrics, step=None):
            self.logged.append((metrics, step))

    def fail_atom_array_to_afd_record(atom_array):
        del atom_array
        raise AssertionError("AFD record conversion should be disabled")

    monkeypatch.setattr(
        metrics_logging,
        "_atom_array_to_afd_record",
        fail_atom_array_to_afd_record,
    )

    fabric = FakeFabric()
    trainer = SimpleNamespace(
        fabric=fabric,
        state={"current_epoch": 0, "global_step": 10},
    )
    callback = metrics_logging.LogMoleculeAFDValidationMetricsCallback(
        enabled=True,
        compute_afd=False,
        compute_atom_fractions=True,
        append_to_validation_results=False,
    )

    callback.on_validation_epoch_start(trainer)
    callback.on_validation_batch_end(
        trainer,
        outputs={
            "predicted_atom_array_stack": [
                SimpleNamespace(element=np.asarray(["C", "O"], dtype=object)),
                SimpleNamespace(element=np.asarray(["N", "C"], dtype=object)),
            ]
        },
        batch=None,
        batch_idx=0,
        num_batches=1,
        dataset_name="omol25",
    )
    callback.on_validation_epoch_end(trainer)

    assert len(fabric.logged) == 1
    logged_metrics, step = fabric.logged[0]
    assert step == 10
    assert set(logged_metrics) == {
        "val/omol25/atom_fraction_o",
        "val/omol25/atom_fraction_n",
        "val/omol25/atom_fraction_c",
    }
    assert logged_metrics["val/omol25/atom_fraction_o"] == pytest.approx(1 / 4)
    assert logged_metrics["val/omol25/atom_fraction_n"] == pytest.approx(1 / 4)
    assert logged_metrics["val/omol25/atom_fraction_c"] == pytest.approx(2 / 4)


def test_epoch_molecule_afd_metrics_can_run_without_atom_fractions():
    """Test AFD metrics can run without logging atom-fraction keys."""
    import nucleus.callbacks.metrics_logging as metrics_logging

    class FakeFabric:
        global_rank = 0
        world_size = 1
        is_global_zero = True

        def __init__(self):
            self.logged = []

        def log_dict(self, metrics, step=None):
            self.logged.append((metrics, step))

    fabric = FakeFabric()
    trainer = SimpleNamespace(
        fabric=fabric,
        state={"current_epoch": 0, "global_step": 10},
    )
    callback = metrics_logging.LogMoleculeAFDValidationMetricsCallback(
        enabled=True,
        compute_afd=True,
        compute_atom_fractions=False,
        append_to_validation_results=False,
    )
    callback._records_by_dataset = {"omol25": [{"label": "only-record"}]}
    callback._total_samples_by_dataset = {"omol25": 3}

    callback.on_validation_epoch_end(trainer)

    assert len(fabric.logged) == 1
    logged_metrics, step = fabric.logged[0]
    assert step == 10
    assert set(logged_metrics) == {
        "val/omol25/afd_score",
        "val/omol25/afd_num_samples",
        "val/omol25/afd_ref_sample_size",
        "val/omol25/afd_input_success_rate",
    }
    assert logged_metrics["val/omol25/afd_score"] == pytest.approx(0.0)
    assert logged_metrics["val/omol25/afd_num_samples"] == pytest.approx(1.0)
    assert logged_metrics["val/omol25/afd_ref_sample_size"] == pytest.approx(0.0)
    assert logged_metrics["val/omol25/afd_input_success_rate"] == pytest.approx(1 / 3)


def test_epoch_molecule_afd_callback_gathers_payloads_to_rank_zero(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test multi-rank AFD record gathering and rank-zero merging."""
    import nucleus.callbacks.metrics_logging as metrics_logging

    callback = metrics_logging.LogMoleculeAFDValidationMetricsCallback(enabled=True)
    callback._records_by_dataset = {"omol25": [{"label": "rank0"}]}
    callback._total_samples_by_dataset = {"omol25": 1}
    callback._atom_counts_by_dataset = {
        "omol25": {
            "atom_fraction_o": 0,
            "atom_fraction_n": 0,
            "atom_fraction_c": 1,
        }
    }
    callback._total_atoms_by_dataset = {"omol25": 1}
    trainer = SimpleNamespace(
        fabric=SimpleNamespace(global_rank=0, world_size=2, is_global_zero=True)
    )

    def fake_gather_object(local_payload, gathered_payloads, dst):
        assert dst == 0
        gathered_payloads[0] = local_payload
        gathered_payloads[1] = {
            "omol25": {
                "records": [{"label": "rank1-a"}, {"label": "rank1-b"}],
                "total_samples": 3,
                "atom_counts": {
                    "atom_fraction_o": 1,
                    "atom_fraction_n": 2,
                    "atom_fraction_c": 1,
                },
                "total_atoms": 4,
            },
            "other": {
                "records": [{"label": "rank1-other"}],
                "total_samples": 1,
                "atom_counts": {
                    "atom_fraction_o": 0,
                    "atom_fraction_n": 1,
                    "atom_fraction_c": 0,
                },
                "total_atoms": 1,
            },
        }

    monkeypatch.setattr(metrics_logging.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(
        metrics_logging.torch.distributed, "is_initialized", lambda: True
    )
    monkeypatch.setattr(
        metrics_logging.torch.distributed, "gather_object", fake_gather_object
    )

    should_compute, payload = callback._gather_payload_to_rank_zero(trainer)

    assert should_compute is True
    assert payload["omol25"]["total_samples"] == 4
    assert [record["label"] for record in payload["omol25"]["records"]] == [
        "rank0",
        "rank1-a",
        "rank1-b",
    ]
    assert payload["omol25"]["atom_counts"] == {
        "atom_fraction_o": 1,
        "atom_fraction_n": 2,
        "atom_fraction_c": 2,
    }
    assert payload["omol25"]["total_atoms"] == 5
    assert payload["other"]["total_samples"] == 1


def test_epoch_molecule_afd_callback_uses_feature_cache_before_raw_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    """Test that cached reference features avoid loading raw reference records."""
    import torch

    import nucleus.callbacks.metrics_logging as metrics_logging

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached_features = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    torch.save(
        cached_features,
        cache_dir / "Ty-Perez__ct-scd-omol25_reference_features.pt",
    )
    captured = {}

    class FakeCachedReferenceAFDScorer:
        def __init__(self, ref_features, **kwargs):
            captured["ref_features"] = ref_features
            captured["kwargs"] = kwargs

    def fail_get_afd_reference(self):
        del self
        raise AssertionError("Raw reference records should not be loaded")

    monkeypatch.setattr(
        metrics_logging,
        "_CachedReferenceAFDScorer",
        FakeCachedReferenceAFDScorer,
    )
    monkeypatch.setattr(
        metrics_logging.LogMoleculeAFDValidationMetricsCallback,
        "_get_afd_reference",
        fail_get_afd_reference,
    )

    callback = metrics_logging.LogMoleculeAFDValidationMetricsCallback(
        enabled=True,
        afd_reference_records_path=tmp_path / "raw_reference.pt",
        afd_model_name="Ty-Perez/ct-scd-omol25",
        afd_feature_cache_dir=cache_dir,
    )

    scorer = callback._get_afd_scorer()

    assert isinstance(scorer, FakeCachedReferenceAFDScorer)
    assert torch.equal(captured["ref_features"], cached_features)
    assert captured["kwargs"] == {
        "data_modality": "molecules",
        "model_name": "Ty-Perez/ct-scd-omol25",
        "batch_size": 256,
        "device": None,
    }


def test_epoch_molecule_afd_callback_missing_feature_cache_computes_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    """Test that missing reference features fall back to AFDScore computation."""
    import zevals.molecules.afd as afd_module

    import nucleus.callbacks.metrics_logging as metrics_logging

    captured = {}

    class FakeAFDScore:
        def __init__(self, reference, **kwargs):
            captured["reference"] = reference
            captured["kwargs"] = kwargs

    monkeypatch.setattr(afd_module, "AFDScore", FakeAFDScore, raising=False)

    callback = metrics_logging.LogMoleculeAFDValidationMetricsCallback(
        enabled=True,
        afd_reference_records_path=tmp_path / "raw_reference.pt",
        afd_model_name="Ty-Perez/ct-scd-omol25",
        afd_feature_cache_dir=tmp_path / "cache",
    )
    callback._afd_reference = "loaded-raw-reference"

    scorer = callback._get_afd_scorer()

    assert isinstance(scorer, FakeAFDScore)
    assert captured["reference"] == "loaded-raw-reference"
    assert captured["kwargs"] == {
        "data_modality": "molecules",
        "model_name": "Ty-Perez/ct-scd-omol25",
        "batch_size": 256,
        "device": None,
        "feature_cache_dir": tmp_path / "cache",
    }


def test_epoch_molecule_afd_callback_skips_scorer_for_too_few_records(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that tiny epoch-level AFD payloads avoid expensive scorer setup."""
    import nucleus.callbacks.metrics_logging as metrics_logging

    class FakeFabric:
        global_rank = 0
        world_size = 1
        is_global_zero = True

        def __init__(self):
            self.logged = []

        def log_dict(self, metrics, step=None):
            self.logged.append((metrics, step))

    def fail_get_afd_scorer(self):
        del self
        raise AssertionError("AFD scorer should not be constructed for N < 2")

    monkeypatch.setattr(
        metrics_logging.LogMoleculeAFDValidationMetricsCallback,
        "_get_afd_scorer",
        fail_get_afd_scorer,
    )

    fabric = FakeFabric()
    trainer = SimpleNamespace(
        fabric=fabric,
        state={"current_epoch": 0, "global_step": 10},
    )
    callback = metrics_logging.LogMoleculeAFDValidationMetricsCallback(
        enabled=True,
        append_to_validation_results=False,
    )
    callback._records_by_dataset = {"omol25": [{"label": "only-record"}]}
    callback._total_samples_by_dataset = {"omol25": 3}

    callback.on_validation_epoch_end(trainer)

    assert len(fabric.logged) == 1
    logged_metrics, step = fabric.logged[0]
    assert step == 10
    assert logged_metrics["val/omol25/afd_score"] == pytest.approx(0.0)
    assert logged_metrics["val/omol25/afd_num_samples"] == pytest.approx(1.0)
    assert logged_metrics["val/omol25/afd_ref_sample_size"] == pytest.approx(0.0)
    assert logged_metrics["val/omol25/afd_input_success_rate"] == pytest.approx(1 / 3)
    assert logged_metrics["val/omol25/atom_fraction_o"] == pytest.approx(0.0)
    assert logged_metrics["val/omol25/atom_fraction_n"] == pytest.approx(0.0)
    assert logged_metrics["val/omol25/atom_fraction_c"] == pytest.approx(0.0)


def test_metric_class_rejects_removed_afd_kwargs():
    """Test that stale Hydra configs fail clearly after the AFD config cleanup."""
    import nucleus.metrics.molecule_generation as molecule_generation

    with pytest.raises(TypeError, match="afd_reference"):
        molecule_generation.MoleculeGenerationMetrics(afd_reference="qm9")

    with pytest.raises(TypeError, match="compute_afd"):
        molecule_generation.MoleculeGenerationMetrics(compute_afd=True)

    with pytest.raises(TypeError, match="afd_reference_records_path"):
        molecule_generation.MoleculeGenerationMetrics(
            afd_reference_records_path="reference.pt"
        )


def test_molecule_generation_metrics_use_fpsim2_reference_path(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that molecule generation metrics forward the FPSim2 DB path."""
    import nucleus.metrics.molecule_generation as molecule_generation

    _patch_metric_helpers(monkeypatch, molecule_generation)

    captured = {}

    def fake_novelty(smiles, reference=None, *, reference_fp=None, n_threads=1):
        captured["smiles"] = list(smiles)
        captured["reference"] = reference
        captured["reference_fp"] = reference_fp
        captured["n_threads"] = n_threads
        return {"novel_rate": 0.25}

    monkeypatch.setattr(molecule_generation, "compute_smiles_novelty", fake_novelty)

    metrics = molecule_generation.compute_molecule_generation_metrics(
        predicted_atom_array_stack=[FakeMol("A"), FakeMol("a"), FakeMol("B")],
        reference_fingerprint_path="reference.h5",
        novelty_n_threads=4,
        compute_posebusters=False,
        compute_uff_force_norm=False,
    )

    assert metrics["novel_rate"] == pytest.approx(0.25)
    assert captured == {
        "smiles": ["a", "a", "b"],
        "reference": None,
        "reference_fp": molecule_generation.Path("reference.h5"),
        "n_threads": 4,
    }


def test_atom_array_to_mol_requires_rdkit_molecule(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that failed RDKit molecule perception is counted as invalid."""
    import nucleus.metrics.molecule_generation as molecule_generation

    def fake_rdkit_smiles_from_atom_array(atom_array, *, example_id, return_mol_only):
        del atom_array, example_id, return_mol_only
        return None

    monkeypatch.setattr(
        molecule_generation,
        "_rdkit_smiles_from_atom_array",
        fake_rdkit_smiles_from_atom_array,
    )

    assert molecule_generation._atom_array_to_mol(object()) is None


def test_molecule_generation_metrics_omit_uma_keys_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    """UMA is opt-in, so the default metric set must be byte-for-byte unchanged.

    A zero-filled placeholder would be worse than an absent key here: every
    existing run would start logging `uma_*: 0.0` for a model that never ran.
    """
    import nucleus.metrics.molecule_generation as molecule_generation

    _patch_metric_helpers(monkeypatch, molecule_generation)

    def explode(*args, **kwargs):
        raise AssertionError("UMA must not be called when the flag is off")

    monkeypatch.setattr(molecule_generation, "_compute_uma_force_metrics", explode)

    predicted = [
        SimpleNamespace(label="A", element=np.asarray(["C"], dtype=object)),
        SimpleNamespace(label="a", element=np.asarray(["C"], dtype=object)),
        SimpleNamespace(label="B", element=np.asarray(["C"], dtype=object)),
        SimpleNamespace(label="bad", element=np.asarray(["C"], dtype=object)),
    ]
    metrics = molecule_generation.compute_molecule_generation_metrics(
        predicted_atom_array_stack=predicted,
    )

    assert not [key for key in metrics if key.startswith("uma_")]


def test_molecule_generation_metrics_forward_uma_config_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
):
    """The Hydra config values must reach the zevals metric unchanged.

    `uma_task` in particular decides the physics: `omol` is the aperiodic,
    charge-aware head. A default that silently drifted to a periodic head would
    score molecules against the wrong reference without raising.
    """
    import nucleus.metrics.molecule_generation as molecule_generation

    _patch_metric_helpers(monkeypatch, molecule_generation)

    captured = {}

    def fake_uma(mol_list, **kwargs):
        captured.update(kwargs)
        captured["n_mols"] = len(mol_list)
        return {"uma_scored_rate": 0.5, "uma_force_max_mean": 1.25}

    monkeypatch.setattr(molecule_generation, "_compute_uma_force_metrics", fake_uma)

    predicted = [
        SimpleNamespace(label="A", element=np.asarray(["C"], dtype=object)),
        SimpleNamespace(label="a", element=np.asarray(["C"], dtype=object)),
        SimpleNamespace(label="B", element=np.asarray(["C"], dtype=object)),
        SimpleNamespace(label="bad", element=np.asarray(["C"], dtype=object)),
    ]
    metrics = molecule_generation.compute_molecule_generation_metrics(
        predicted_atom_array_stack=predicted,
        extra_info={
            "has_mean_force_norm": [True],
            "mean_force_norm": [2.75],
            # Match the metadata shape assembled by ZatomTrainer: the original
            # inference request is retained below ``specification``.
            "specification": {
                "total_charge": -1,
                "spin_multiplicity": 2,
            },
        },
        compute_uma_forces=True,
        uma_device="cuda",
        uma_batch_size=4,
    )

    assert captured["task"] == "omol"
    assert captured["checkpoint"] == "uma-s-1p2"
    assert captured["device"] == "cuda"
    assert captured["batch_size"] == 4
    assert captured["target_mean_force_norm"] == pytest.approx(2.75)
    assert captured["total_charge"] == -1
    assert captured["spin_multiplicity"] == 2
    # Scored over valid molecules (3 of 4), but the rate counts against
    # everything generated, so the denominator stays at the full 4.
    assert captured["n_mols"] == 3
    assert captured["total_samples"] == 4
    assert metrics["uma_scored_rate"] == pytest.approx(0.5)


def test_molecule_generation_metrics_survive_a_uma_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    """A missing fairchem / gated-weights 401 must not take down validation.

    The guard lives inside `_compute_uma_force_metrics`, so this exercises the
    real function with the zevals import forced to fail.
    """
    import sys

    import nucleus.metrics.molecule_generation as molecule_generation

    _patch_metric_helpers(monkeypatch, molecule_generation)
    monkeypatch.setitem(sys.modules, "zevals.molecules.uma_forces", None)

    predicted = [
        SimpleNamespace(label="A", element=np.asarray(["C"], dtype=object)),
        SimpleNamespace(label="a", element=np.asarray(["C"], dtype=object)),
        SimpleNamespace(label="B", element=np.asarray(["C"], dtype=object)),
        SimpleNamespace(label="bad", element=np.asarray(["C"], dtype=object)),
    ]
    metrics = molecule_generation.compute_molecule_generation_metrics(
        predicted_atom_array_stack=predicted,
        compute_uma_forces=True,
    )

    assert not [key for key in metrics if key.startswith("uma_")]
    assert metrics["valid_rate"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("adapter_module", "zevals_module", "sample"),
    [
        (
            "nucleus.metrics.molecule_generation",
            "zevals.molecules.uma_forces",
            object(),
        ),
        (
            "nucleus.metrics.material_generation",
            "zevals.materials.uma_forces",
            object(),
        ),
    ],
)
def test_validation_uma_adapter_reenables_gradients(
    monkeypatch: pytest.MonkeyPatch,
    adapter_module: str,
    zevals_module: str,
    sample: object,
):
    """UMA keeps autograd enabled inside Fabric's no-grad validation loop."""
    import importlib

    import torch

    adapter = importlib.import_module(adapter_module)
    uma_forces = importlib.import_module(zevals_module)
    captured = {}

    def fake_uma(samples, total_samples, **kwargs):
        captured["grad_enabled"] = torch.is_grad_enabled()
        captured["samples"] = samples
        captured["total_samples"] = total_samples
        captured["target_mean_force_norm"] = kwargs["target_mean_force_norm"]
        captured["total_charge"] = kwargs.get("total_charge")
        captured["spin_multiplicity"] = kwargs.get("spin_multiplicity")
        return {"uma_scored_rate": 1.0}

    monkeypatch.setattr(uma_forces, "compute_uma_force_metrics", fake_uma)

    adapter_kwargs = {}
    if "molecule" in adapter_module:
        adapter_kwargs.update(total_charge=-1, spin_multiplicity=2)
    with torch.no_grad():
        result = adapter._compute_uma_force_metrics(
            [sample],
            total_samples=1,
            checkpoint="uma-s-1p2",
            task="omol" if "molecule" in adapter_module else "omat",
            device="cpu",
            batch_size=1,
            target_mean_force_norm=2.5,
            **adapter_kwargs,
        )

    expected = {
        "grad_enabled": True,
        "samples": [sample],
        "total_samples": 1,
        "target_mean_force_norm": 2.5,
        "total_charge": None,
        "spin_multiplicity": None,
    }
    if "molecule" in adapter_module:
        expected.update(total_charge=-1, spin_multiplicity=2)
    assert captured == expected
    assert result == {"uma_scored_rate": 1.0}
