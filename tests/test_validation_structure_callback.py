"""Tests for material-aware validation structure export."""

import gzip
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray
from omegaconf import OmegaConf

from nucleus.callbacks.dump_validation_structures import (
    DumpValidationStructuresCallback,
)
from nucleus.inference_input_parsing import create_diffused_material_atoms
from nucleus.utils.materials import (
    annotate_material_properties,
    fractional_to_cartesian,
    lattice_vectors_from_lengths_angles,
)


class _Trainer:
    state = {"current_epoch": 5}


def _material_structure() -> AtomArray:
    lengths = np.array([5.0, 6.0, 7.0])
    angles = np.array([90.0, 90.0, 120.0])
    fractional_coordinates = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    lattice_vectors = lattice_vectors_from_lengths_angles(lengths, angles)

    atom_array = create_diffused_material_atoms(2)
    atom_array.atom_name = np.array(["C1", "O1"])
    atom_array.element = np.array(["C", "O"])
    atom_array.coord = fractional_to_cartesian(fractional_coordinates, lattice_vectors)
    return annotate_material_properties(
        atom_array,
        fractional_coordinates=fractional_coordinates,
        lattice_lengths=lengths,
        lattice_angles=angles,
        lattice_vectors=lattice_vectors,
    )


def _molecule_structure() -> AtomArray:
    atom_array = AtomArray(1)
    atom_array.coord = np.array([[1.0, 2.0, 3.0]])
    atom_array.chain_id = np.array(["A"])
    atom_array.res_id = np.array([1])
    atom_array.res_name = np.array(["LIG"])
    atom_array.atom_name = np.array(["C1"])
    atom_array.element = np.array(["C"])
    atom_array.hetero = np.array([True])
    return atom_array


def _run_callback(tmp_path: Path, structure: AtomArray, example_id: str) -> Path:
    callback = DumpValidationStructuresCallback(
        save_dir=tmp_path,
        dump_predictions=True,
        dump_prediction_metadata_json=False,
        one_model_per_file=True,
        dump_every_n=5,
        verbose=False,
    )
    callback.on_validation_batch_end(
        trainer=_Trainer(),
        outputs={
            "network_output": {},
            "predicted_atom_array_stack": [structure],
            "prediction_metadata": {},
        },
        batch=[{"example_id": example_id}],
        dataset_name="validation",
    )
    return (
        tmp_path
        / "predictions"
        / "epoch_5"
        / "validation"
        / f"{example_id}_model_0.cif.gz"
    )


def test_validation_material_cif_contains_fractional_coordinates_and_cell(tmp_path):
    """Validation exports retain predicted fractional and lattice outputs."""
    output_path = _run_callback(tmp_path, _material_structure(), "material")

    with gzip.open(output_path, "rt") as handle:
        text = handle.read()

    for tag in (
        "_atom_site.fract_x",
        "_atom_site.fract_y",
        "_atom_site.fract_z",
        "_cell.length_a",
        "_cell.length_b",
        "_cell.length_c",
        "_cell.angle_alpha",
        "_cell.angle_beta",
        "_cell.angle_gamma",
    ):
        assert tag in text


def test_validation_molecule_cif_does_not_gain_material_fields(tmp_path):
    """The material-aware callback preserves ordinary molecular exports."""
    output_path = _run_callback(tmp_path, _molecule_structure(), "molecule")

    with gzip.open(output_path, "rt") as handle:
        text = handle.read()

    assert "_atom_site.Cartn_x" in text
    assert "_atom_site.fract_x" not in text
    assert "_cell.length_a" not in text


def test_model_configs_use_material_aware_validation_callback():
    """All released Zatom variants share the corrected validation exporter."""
    repository_root = Path(__file__).parents[1]
    expected_target = (
        "nucleus.callbacks.dump_validation_structures."
        "DumpValidationStructuresCallback"
    )

    for model_name in ("zatom2",):
        config = OmegaConf.load(
            repository_root
            / "models"
            / model_name
            / "configs"
            / "callbacks"
            / "metrics_logging.yaml"
        )
        assert config.dump_validation_structures_callback._target_ == expected_target
