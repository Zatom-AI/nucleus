"""Numerical checks for the frozen-model relaxation experiment."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _probe_module():
    path = Path(__file__).resolve().parents[1] / "scripts/probe_zero_shot_relaxation.py"
    spec = importlib.util.spec_from_file_location("relaxation_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flow_euler_endpoint_uses_physical_velocity():
    module = _probe_module()
    state = np.array([[1.0, -2.0, 3.0]])
    normalized_velocity = np.array([[0.25, -0.5, 0.75]])
    # The checkpoint's endpoint formula is x + (1-t) * sigma_data * V_L.
    endpoint = module.flow_euler_step(state, normalized_velocity, 0.75, 1.0, 16.0)
    np.testing.assert_array_equal(endpoint, [[2.0, -4.0, 6.0]])
    first = module.flow_euler_step(state, normalized_velocity, 0.75, 0.875, 16.0)
    second = module.flow_euler_step(first, normalized_velocity, 0.875, 1.0, 16.0)
    np.testing.assert_array_equal(second, endpoint)


def test_summary_pairs_by_structure_and_counts_failures(tmp_path):
    rows = []
    for index, input_force, output_force in ((0, 1.0, 2.0), (1, 100.0, 25.0)):
        base = dict(domain="omol25", index=index, noise=0.1, elements=["H", "H"])
        rows.extend(
            [
                dict(
                    base,
                    method="input",
                    sigma=None,
                    target=None,
                    mean_force=input_force,
                    max_force=input_force,
                    energy_eV=0.0,
                ),
                dict(
                    base,
                    method="one_step",
                    sigma=0.1,
                    target=None,
                    mean_force=output_force,
                    max_force=output_force,
                    energy_eV=-1.0,
                ),
                dict(
                    base,
                    method="one_step",
                    sigma=0.1,
                    target=0.0,
                    error="oracle failed",
                ),
            ]
        )
    # Reverse the order to catch accidental positional pairing.
    serialized = "".join(json.dumps(row) + "\n" for row in reversed(rows))
    (tmp_path / "structures.jsonl").write_text(serialized)
    (tmp_path / "scores.jsonl").write_text(serialized)
    _probe_module().summarize(SimpleNamespace(output=tmp_path, seed=42))
    summaries = json.loads((tmp_path / "summary.json").read_text())
    control = next(row for row in summaries if row["target"] is None)
    assert control["median_force_ratio"] == 1.125
    assert control["both_improved"] == 1
    failed = next(row for row in summaries if row["target"] == 0.0)
    assert failed["attempts"] == 2
    assert failed["scored"] == 0


def test_summary_refuses_incomplete_scoring(tmp_path):
    (tmp_path / "structures.jsonl").write_text("{}\n")
    (tmp_path / "scores.jsonl").write_text("")
    with pytest.raises(RuntimeError, match="Scoring is incomplete"):
        _probe_module().summarize(SimpleNamespace(output=tmp_path, seed=42))


def test_one_step_baseline_matches_table_inputs():
    rows = [
        dict(domain=domain, index=0, method=method, noise=noise)
        for domain in ("omol25", "omat24")
        for method in ("input", "one_step")
        for noise in (0.0, 0.1, 0.3)
    ]
    selected = _probe_module().select_one_step_inputs(rows)
    assert [(r["domain"], r["noise"]) for r in selected] == [
        ("omol25", 0.1),
        ("omat24", 0.0),
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        _probe_module().select_one_step_inputs(selected + selected)


@pytest.mark.parametrize("curvature", [35.0, 140.0])
def test_one_bfgs_update_is_one_force_evaluation_with_global_cap(curvature):
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes

    class HarmonicCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self):
            super().__init__()
            self.calls = 0

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.calls += 1
            self.results = {
                "energy": 0.5 * curvature * np.square(self.atoms.positions).sum(),
                "forces": -curvature * self.atoms.positions,
            }

    atoms = Atoms(
        "HC", positions=[[0.03, 0, 0], [0, 0.01, 0]], cell=[4, 5, 6], pbc=True
    )
    atoms.info.update(charge=1, spin=2)
    calculator = HarmonicCalculator()
    atoms.calc = calculator
    initial = atoms.copy()
    displacement = atoms.get_forces() / 70.0
    displacement *= min(1.0, 0.04 / np.linalg.norm(displacement, axis=-1).max())
    _probe_module().one_bfgs_update(atoms)
    assert calculator.calls == 1  # no endpoint force or line-search evaluation
    np.testing.assert_allclose(
        atoms.positions, initial.positions + displacement, atol=1e-14, rtol=0
    )
    np.testing.assert_array_equal(atoms.cell, initial.cell)
    np.testing.assert_array_equal(atoms.numbers, initial.numbers)
    assert atoms.info == initial.info
    atoms.get_forces()  # separate evaluation of the resulting geometry
    assert calculator.calls == 2
