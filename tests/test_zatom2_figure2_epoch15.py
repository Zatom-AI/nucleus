"""Check target selection and scientific aggregation for the saved-output figure."""

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def figure(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("plot_zatom2_figure2_epoch15")


def test_common_targets_are_identical_sorted_and_not_imputed(figure):
    pools = {"base": dict.fromkeys("abc"), "large": dict.fromkeys("ac")}
    assert figure.common_targets(pools, set("abc"), False) == ["a", "c"]
    with pytest.raises(ValueError, match="Incomplete target coverage"):
        figure.common_targets(pools, set("abc"), True)
    with pytest.raises(ValueError, match="Fewer than two"):
        figure.common_targets({"base": {"a": {}}, "large": {"b": {}}}, set("ab"), False)


@pytest.mark.parametrize("name", ["a_model_1.json", "unknown_model_0.json"])
def test_unexpected_candidates_are_rejected(figure, tmp_path, name):
    (tmp_path / name).write_text("{}")
    with pytest.raises(ValueError, match="Unexpected target or candidate"):
        figure.saved_pairs(tmp_path, {"a"})


def test_orphan_file_is_not_silently_dropped(figure, tmp_path):
    (tmp_path / "a_model_0.json").write_text("{}")
    with pytest.raises(ValueError, match="Incomplete CIF/JSON pair"):
        figure.saved_pairs(tmp_path, {"a"})


def test_fixture_global_overrides_and_relative_reference(figure, tmp_path):
    path = tmp_path / "fixture.json"
    path.write_text(
        json.dumps(
            {"global_args": {"fixed": True}, "a": {"input": "a.cif", "fixed": False}}
        )
    )
    assert figure.load_fixture(path) == {
        "a": {"fixed": True, "input": str(tmp_path / "a.cif")}
    }


@pytest.fixture
def mlip_case(tmp_path, monkeypatch, figure):
    pool, specs, atoms = {}, {}, {}
    # Unequal atom counts distinguish component pooling from a system mean.
    for target, count, force, energy in [("a", 1, 1.0, 2.0), ("b", 3, 3.0, 6.0)]:
        path = tmp_path / f"{target}.json"
        path.write_text(
            json.dumps(
                {
                    "atomistic_properties": {
                        "energy_eV": energy,
                        "forces_eV_per_angstrom": [[force] * 3] * count,
                    }
                }
            )
        )
        pool[target] = {".json": {"path": str(path)}, ".cif.gz": {"path": target}}
        specs[target] = {
            "input": target,
            "extra": {
                "energy_gt_eV": 0.0,
                "forces_gt_eV_per_angstrom": [[0.0] * 3] * count,
            },
        }

        class Atoms(SimpleNamespace):
            def __len__(self):
                return len(self.coord)

        atoms[target] = Atoms(
            element=np.array(["C"] * count), coord=np.zeros((count, 3))
        )
    module = importlib.import_module("recompute_zatom2_figure2_metrics")
    monkeypatch.setattr(module, "load_prediction", lambda path: atoms[str(path)])
    return pool, specs


def test_force_errors_pool_over_components_energy_over_systems(figure, mlip_case):
    pool, specs = mlip_case
    result = figure.score_mlip(pool, specs, ["a", "b"])
    assert result["num_force_components"] == 12
    assert result["force_component_absolute_error_sum"] == 30
    assert result["value"] == 2.5  # Per-system averaging would incorrectly give 2.
    assert result["energy_mae_eV"] == 4
    assert result["energy_per_atom_mae_eV"] == 2


def test_missing_targets_do_not_enter_mlip_denominator(figure, mlip_case):
    pool, specs = mlip_case
    result = figure.score_mlip(pool, specs, ["b"])
    assert result["value"] == 3
    assert result["num_force_components"] == 9


def test_nonfinite_saved_prediction_fails(figure, mlip_case):
    pool, specs = mlip_case
    path = Path(pool["a"][".json"]["path"])
    metadata = json.loads(path.read_text())
    metadata["atomistic_properties"]["forces_eV_per_angstrom"][0][0] = float("nan")
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="must all be finite"):
        figure.score_mlip(pool, specs, ["a"])


def test_changed_mlip_coordinate_frame_is_rejected(figure, mlip_case, monkeypatch):
    pool, specs = mlip_case
    module = importlib.import_module("recompute_zatom2_figure2_metrics")
    original_loader = module.load_prediction
    changed = copy.deepcopy(original_loader(Path("b")))
    changed.coord[0, 0] = 1.0
    pool["b"][".cif.gz"]["path"] = "changed"
    monkeypatch.setattr(
        module,
        "load_prediction",
        lambda path: changed if str(path) == "changed" else original_loader(path),
    )
    with pytest.raises(ValueError, match="atom order/coordinate frame differs"):
        figure.score_mlip(pool, specs, ["b"])


@pytest.mark.parametrize("change_fractional", [False, True])
def test_periodic_mlip_uses_fractional_atom_order(
    figure, mlip_case, monkeypatch, change_fractional
):
    pool, specs = mlip_case
    module = importlib.import_module("recompute_zatom2_figure2_metrics")
    reference = module.load_prediction(Path("b"))
    reference.box = np.eye(3) * 4
    reference.material_frac_coords = np.array(
        [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]]
    )
    prediction = copy.deepcopy(reference)
    prediction.coord[0, 0] = 1.0  # Different Cartesian representation is allowed.
    prediction.material_frac_coords[0] += 1  # Periodic wrapping is equivalent.
    if change_fractional:
        prediction.material_frac_coords[0, 0] += 0.2
    pool["b"][".cif.gz"]["path"] = "prediction"
    monkeypatch.setattr(
        module,
        "load_prediction",
        lambda path: prediction if str(path) == "prediction" else reference,
    )
    if change_fractional:
        with pytest.raises(ValueError, match="atom order/coordinate frame differs"):
            figure.score_mlip(pool, specs, ["b"])
    else:
        assert figure.score_mlip(pool, specs, ["b"])["value"] == 3


def test_wrong_encoder_reference_cache_is_rejected(figure, tmp_path):
    path = tmp_path / "features.pt"
    path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "signature": {
                    "model_name": "Ty-Perez/ct-scd-geom10",
                    "model_revision": "wrong",
                }
            }
        )
    )
    with pytest.raises(ValueError, match="wrong AFD encoder"):
        figure.reference_identity(path)


@pytest.mark.parametrize(
    "angles",
    [
        [46.32900619506836, 46.32900619506836, 93.55326080322266],
        [120.0, 120.0, 120.0],
        [0.0, 90.0, 90.0],
        [90.0, float("nan"), 90.0],
    ],
)
def test_impossible_or_degenerate_cell_angles_are_rejected(figure, angles):
    assert not figure.realizable_cell([9.47, 9.24, 15.68], angles)


def test_valid_cells_have_no_arbitrary_volume_cutoff(figure):
    assert figure.realizable_cell([1e-4, 1e-4, 1e-4], [90.0, 90.0, 90.0])
    assert figure.realizable_cell([4.0, 5.0, 6.0], [80.0, 95.0, 105.0])
    assert not figure.realizable_cell([-4.0, 5.0, 6.0], [90.0, 90.0, 90.0])


def test_source_content_changes_fingerprint(figure, tmp_path):
    path = tmp_path / "sample.json"
    path.write_text("first")
    original = figure.digest(figure.file_record(path))
    path.write_text("second")
    assert figure.digest(figure.file_record(path)) != original
