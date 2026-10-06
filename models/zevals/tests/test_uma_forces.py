"""Tests for the UMA force-metric module.

Two groups:

1. Contract and adapter tests. They run everywhere — no fairchem, no weights,
   no network. A fake model stands in for UMA, so the flags, the rate
   arithmetic, the failure handling and the pipeline wiring are all checked.
2. Real-model tests, marked with the ``uma_model`` fixture. They need
   ``fairchem-core`` and the gated ``uma-s-1p2`` weights, and are skipped when
   either is missing.

Run:  pytest -q tests/test_uma_forces.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOL = 2e-3  # eV or eV/A; float32 inference is not tighter than this


# --------------------------------------------------------------- fake model


class FakeUMAModel:
    """Stand-in for ``UMAModel`` that records what it was given.

    Forces are deterministic and per-atom: atom ``i`` of system ``j`` gets a
    force of magnitude ``(i + 1) * scale``. That makes the max/mean/rms
    arithmetic checkable by hand.
    """

    def __init__(self, scale: float = 1.0, fail: bool = False):
        self.scale = scale
        self.fail = fail
        self.seen: list[list] = []

    def predict(self, chunk):
        import torch

        chunk = list(chunk)
        self.seen.append(chunk)
        if self.fail:
            raise RuntimeError("simulated CUDA OOM")
        natoms = [len(atoms) for atoms in chunk]
        forces = torch.cat(
            [
                torch.tensor(
                    [[(i + 1) * self.scale, 0.0, 0.0] for i in range(n)],
                    dtype=torch.float32,
                )
                for n in natoms
            ]
        )
        return {
            "forces": forces,
            "energy": torch.tensor([-10.0 * n for n in natoms]),
            "natoms": torch.tensor(natoms),
        }

    @staticmethod
    def split_per_system(result, key="forces"):
        import torch

        return list(torch.split(result[key], result["natoms"].tolist()))


def _ethanol():
    """An MMFF-relaxed ethanol with a real 3D conformer."""
    Chem = pytest.importorskip("rdkit.Chem")
    AllChem = pytest.importorskip("rdkit.Chem.AllChem")

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=0) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _nacl_structure():
    """Rock-salt NaCl at its equilibrium lattice constant."""
    pmg = pytest.importorskip("pymatgen.core")

    a = 5.64
    return pmg.Structure(
        lattice=[[a / 2, 0, 0], [0, a / 2, 0], [0, 0, a / 2]],
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )


# ------------------------------------------------------------------ adapters


def test_molecule_conversion_is_aperiodic_and_carries_formal_charge():
    """Guards the molecule flags.

    The ``omol`` head is charge aware and expects no cell. A leaked cell or a
    dropped formal charge changes the predicted forces while leaving the output
    shape correct, so only a flag check catches it.
    """
    Chem = pytest.importorskip("rdkit.Chem")
    AllChem = pytest.importorskip("rdkit.Chem.AllChem")
    from zevals.atomistic.uma.adapters import mol_to_atoms

    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-]"))
    assert AllChem.EmbedMolecule(mol, randomSeed=0) == 0

    atoms = mol_to_atoms(mol)
    assert atoms is not None
    assert not atoms.pbc.any(), "a molecule must not be periodic"
    assert atoms.get_cell().rank == 0, "a molecule must not carry a cell"
    assert atoms.info["charge"] == -1


def test_material_conversion_is_periodic_and_keeps_the_lattice():
    """Guards the material flags.

    Losing the cell or ``pbc`` turns a relaxed crystal into isolated atoms in
    vacuum, which then report large spurious forces.
    """
    pytest.importorskip("pymatgen.core")
    import numpy as np

    from zevals.atomistic.uma.adapters import structure_to_atoms

    structure = _nacl_structure()
    atoms = structure_to_atoms(structure)

    assert atoms is not None
    assert atoms.pbc.all(), "a crystal must be periodic on all three axes"
    assert np.allclose(atoms.get_cell()[:], structure.lattice.matrix)


def test_conversion_returns_none_instead_of_raising_on_bad_input():
    """Guards the never-raise rule at the adapter level."""
    Chem = pytest.importorskip("rdkit.Chem")
    from zevals.atomistic.uma.adapters import sample_to_atoms, samples_to_atoms

    conformerless = Chem.MolFromSmiles("CCO")  # parses, but has no 3D geometry

    assert sample_to_atoms(None) is None
    assert sample_to_atoms(conformerless) is None
    assert sample_to_atoms("not a molecule") is None
    assert samples_to_atoms([None, conformerless, "junk"]) == []


# ------------------------------------------------------- metric contract


def test_metric_survives_bad_samples_and_returns_flat_floats():
    """Guards the zevals metric contract.

    A sample list may hold ``None`` and conformer-less molecules. The metric
    must not raise, must return a flat dict of plain floats, and must not load
    a model when there is nothing to score.
    """
    Chem = pytest.importorskip("rdkit.Chem")
    from zevals.molecules.uma_forces import compute_uma_force_metrics

    out = compute_uma_force_metrics(
        [None, Chem.MolFromSmiles("CCO"), "not a molecule"], total_samples=10
    )

    assert out["uma_scored_rate"] == 0.0
    assert all(isinstance(key, str) for key in out)
    assert all(isinstance(value, float) for value in out.values())


def test_metric_normalises_the_rate_by_total_samples():
    """Guards the rate arithmetic.

    Two scorable molecules out of ``total_samples=8`` is a rate of 0.25, not
    1.0. Normalising by the scorable count instead would hide exactly the
    failure this metric exists to report.
    """
    pytest.importorskip("rdkit.Chem")
    from zevals.molecules.uma_forces import compute_uma_force_metrics

    mol = _ethanol()
    out = compute_uma_force_metrics(
        [mol, mol, None], total_samples=8, model=FakeUMAModel()
    )

    assert out["uma_scored_rate"] == pytest.approx(0.25)


def test_per_sample_scores_preserve_input_indices_after_skips():
    """Per-sample output must remain alignable with heterogeneous targets."""
    pytest.importorskip("ase")
    from ase import Atoms

    from zevals.atomistic.uma.metrics import score_uma_samples

    first = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    first.info.update(charge=0, spin=1)
    third = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    third.info.update(charge=1, spin=2)

    scores = score_uma_samples(
        [first, None, third],
        task="omol",
        model=FakeUMAModel(),
        batch_size=2,
    )

    assert [score["sample_index"] for score in scores] == [0, 2]
    assert scores[0]["mean_force_norm"] == pytest.approx(1.0)
    assert scores[1]["mean_force_norm"] == pytest.approx(1.5)
    assert first.info == {"charge": 0, "spin": 1}
    assert third.info == {"charge": 1, "spin": 2}


def test_metric_force_arithmetic_matches_the_known_fake_forces():
    """Guards the max/mean/rms/energy reduction.

    Ethanol with hydrogens has 9 atoms, and the fake model puts a force of
    ``i + 1`` eV/A on atom ``i``. So the max is 9, the mean force norm is 5,
    the rms over atoms is sqrt(mean(1..81)) = sqrt(28.33), and the energy per
    atom is -10.
    """
    pytest.importorskip("rdkit.Chem")
    import math

    from zevals.molecules.uma_forces import compute_uma_force_metrics

    mol = _ethanol()
    assert mol.GetNumAtoms() == 9

    out = compute_uma_force_metrics([mol], model=FakeUMAModel())

    assert out["uma_force_max_mean"] == pytest.approx(9.0)
    assert out["uma_force_max_median"] == pytest.approx(9.0)
    assert out["uma_mean_force_norm_mean"] == pytest.approx(5.0)
    assert out["uma_force_rms_mean"] == pytest.approx(math.sqrt(285.0 / 9.0))
    assert out["uma_energy_per_atom_mean"] == pytest.approx(-10.0)
    assert "uma_mean_force_norm_mae_eV_per_angstrom" not in out


def test_metric_compares_each_uma_mean_force_norm_to_a_shared_target():
    """Conditioning MAE is reduced after per-structure force norms are computed."""
    Chem = pytest.importorskip("rdkit.Chem")
    from zevals.molecules.uma_forces import compute_uma_force_metrics

    ethanol = _ethanol()  # 9 atoms -> fake mean force norm 5 eV/A
    heavy_atom_ethanol = Chem.RemoveHs(ethanol)  # 3 atoms -> mean 2 eV/A

    out = compute_uma_force_metrics(
        [ethanol, heavy_atom_ethanol],
        target_mean_force_norm=3.0,
        model=FakeUMAModel(),
    )

    # mean(|[5, 2] - 3|) = 1.5 eV/A. This intentionally differs from
    # |mean([5, 2]) - 3| = 0.5 eV/A.
    assert out["uma_mean_force_norm_mae_eV_per_angstrom"] == pytest.approx(1.5)


def test_target_metric_is_zero_when_no_structure_can_be_scored():
    """A requested target keeps a stable key even when the scored rate is zero."""
    from zevals.molecules.uma_forces import compute_uma_force_metrics

    out = compute_uma_force_metrics(
        [],
        target_mean_force_norm=2.0,
        model=FakeUMAModel(),
    )

    assert out["uma_mean_force_norm_mae_eV_per_angstrom"] == 0.0


def test_metric_keeps_going_when_a_batch_fails():
    """Guards batch-level error isolation.

    A GPU OOM in one forward pass must not lose the whole epoch's metric — it
    must drop that batch only, and the scored rate must report the loss.
    """
    pytest.importorskip("rdkit.Chem")
    from zevals.molecules.uma_forces import compute_uma_force_metrics

    out = compute_uma_force_metrics(
        [_ethanol()] * 4, total_samples=4, model=FakeUMAModel(fail=True)
    )

    assert out["uma_scored_rate"] == 0.0
    assert out["uma_force_max_mean"] == 0.0


def test_metric_respects_batch_size():
    """Guards the batching loop: 5 samples at batch_size 2 is 3 forward passes."""
    pytest.importorskip("rdkit.Chem")
    from zevals.molecules.uma_forces import compute_uma_force_metrics

    model = FakeUMAModel()
    compute_uma_force_metrics([_ethanol()] * 5, batch_size=2, model=model)

    assert [len(chunk) for chunk in model.seen] == [2, 2, 1]


def test_metric_applies_inference_charge_and_spin_before_predict():
    """Ground-truth electronic state overrides generated-graph defaults."""
    pytest.importorskip("rdkit.Chem")
    from zevals.molecules.uma_forces import compute_uma_force_metrics

    model = FakeUMAModel()
    compute_uma_force_metrics(
        [_ethanol(), _ethanol()],
        total_charge=-1,
        spin_multiplicity=2,
        model=model,
    )

    assert len(model.seen) == 1
    assert [atoms.info["charge"] for atoms in model.seen[0]] == [-1, -1]
    assert [atoms.info["spin"] for atoms in model.seen[0]] == [2, 2]


def test_prefix_renames_every_key():
    """Guards `uma_metric_keys`: a custom prefix must rename every key."""
    pytest.importorskip("rdkit.Chem")
    from zevals.atomistic.uma.metrics import UMA_KEYS, uma_metric_keys
    from zevals.molecules.uma_forces import compute_uma_force_metrics

    out = compute_uma_force_metrics([_ethanol()], prefix="umaS", model=FakeUMAModel())

    assert set(out) == set(uma_metric_keys("umaS"))
    assert len(out) == len(UMA_KEYS)


# ------------------------------------------------------------ task flags


def test_molecule_and_material_defaults_use_different_task_heads():
    """Guards the flag that decides the physics.

    ``omol`` is the aperiodic, charge-aware head; ``omat`` is the periodic one.
    Swapping them scores a crystal as a molecule in vacuum, or a molecule
    against a periodic-DFT reference.
    """
    pytest.importorskip("zevals.materials")  # needs the `materials` extra
    from zevals.materials.uma_forces import UMA_MATERIAL_TASK
    from zevals.molecules.uma_forces import UMA_MOLECULE_TASK

    assert UMA_MOLECULE_TASK == "omol"
    assert UMA_MATERIAL_TASK == "omat"


def test_materials_wrapper_rejects_the_aperiodic_head():
    """Guards against the silent-wrong-answer case.

    ``omol`` ignores the cell, so it would return plausible-looking but
    meaningless forces for a crystal. That must be a loud error, not a metric.
    """
    pytest.importorskip("zevals.materials")  # needs the `materials` extra
    from zevals.materials.uma_forces import compute_uma_force_metrics

    with pytest.raises(ValueError, match="periodic task head"):
        compute_uma_force_metrics([], task="omol")


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("charge", 1),
        ("spin", 2),
        ("total_charge", -1),
        ("spin_multiplicity", 3),
    ],
)
def test_materials_wrapper_rejects_nonzero_charge_and_spin(keyword, value):
    """Periodic UMA heads only support the fixed charge=0, spin=0 inputs."""
    pytest.importorskip("zevals.materials")
    from zevals.materials.uma_forces import compute_uma_force_metrics

    with pytest.raises(ValueError, match="requires .*0"):
        compute_uma_force_metrics(
            [_nacl_structure()],
            model=FakeUMAModel(),
            **{keyword: value},
        )


def test_materials_wrapper_overrides_ase_charge_and_spin_with_zero():
    """Pre-populated ASE metadata cannot leak electronic state into OMat."""
    pytest.importorskip("ase")
    from zevals.materials.uma_forces import compute_uma_force_metrics

    atoms = _nacl_structure().to_ase_atoms()
    atoms.info.update(charge=-1, spin=3)
    model = FakeUMAModel()

    compute_uma_force_metrics([atoms], model=model)

    assert len(model.seen) == 1
    assert model.seen[0][0].info["charge"] == 0
    assert model.seen[0][0].info["spin"] == 0


def test_indexed_cuda_device_is_reduced_to_plain_cuda():
    """Guards the fairchem device rule.

    ``zevals.utils.devices.resolve_device`` hands out ``"cuda:2"``, which
    fairchem rejects outright. It must be reduced, not passed through.
    """
    from zevals.atomistic.uma.model import normalize_device

    assert normalize_device("cuda:2") == "cuda"
    assert normalize_device("cuda") == "cuda"
    assert normalize_device("cpu") == "cpu"
    assert normalize_device("auto") is None
    assert normalize_device(None) is None


def test_missing_fairchem_raises_an_actionable_error(monkeypatch):
    """Guards the optional-dependency message.

    When fairchem is absent the error must name the extra to install, not
    surface a bare ModuleNotFoundError from deep inside the wrapper.
    """
    from zevals.atomistic.uma.model import load_fairchem

    # A None entry in sys.modules makes `import fairchem.core` raise, which is
    # what an uninstalled fairchem looks like from inside load_fairchem.
    monkeypatch.setitem(sys.modules, "fairchem.core", None)
    monkeypatch.setitem(sys.modules, "fairchem", None)

    with pytest.raises(ImportError, match="fairchem-core"):
        load_fairchem()


# --------------------------------------------------------- pipeline wiring


def test_molecule_pipeline_skips_uma_by_default():
    """Guards the opt-in switch.

    Default config must not emit a single ``uma_*`` key — otherwise every
    existing run would start logging zeros for a metric it never ran.
    """
    pytest.importorskip("rdkit.Chem")
    from zevals.pipeline.molecules import MoleculeEvalPipeline

    pipeline = MoleculeEvalPipeline(compute_posebusters=False)
    assert pipeline.compute_uma_forces is False

    results = pipeline.run([_ethanol()])
    assert not [key for key in results.aggregate if key.startswith("uma_")]


def test_molecule_pipeline_runs_uma_when_switched_on(monkeypatch):
    """Guards the on path end to end, with the model faked out.

    Checks that the flag reaches the metric, that the config values (task,
    checkpoint, batch size) are forwarded, and that the keys land in the
    aggregate dict.
    """
    pytest.importorskip("rdkit.Chem")
    import zevals.atomistic.uma.adapters as adapters
    from zevals.pipeline.molecules import MoleculeEvalPipeline

    captured = {}

    def fake_get_uma_model(checkpoint, task, device, **kwargs):
        captured.update(checkpoint=checkpoint, task=task, device=device)
        return FakeUMAModel()

    monkeypatch.setattr(adapters, "get_uma_model", fake_get_uma_model)

    pipeline = MoleculeEvalPipeline(
        compute_posebusters=False,
        compute_uma_forces=True,
        uma_device="cpu",
        uma_batch_size=4,
    )
    results = pipeline.run([_ethanol(), _ethanol()])

    assert captured == {"checkpoint": "uma-s-1p2", "task": "omol", "device": "cpu"}
    assert results.aggregate["uma_scored_rate"] == pytest.approx(1.0)
    assert results.aggregate["uma_force_max_mean"] == pytest.approx(9.0)


def test_material_pipeline_runs_uma_with_the_periodic_head(monkeypatch):
    """Guards the materials half of the wiring, including the ``omat`` flag."""
    pytest.importorskip("pymatgen.core")
    pytest.importorskip("zevals.materials")  # needs the `materials` extra
    import zevals.atomistic.uma.adapters as adapters
    from zevals.pipeline.materials import MaterialEvalPipeline

    captured = {}

    def fake_get_uma_model(checkpoint, task, device, **kwargs):
        captured.update(checkpoint=checkpoint, task=task, device=device)
        return FakeUMAModel()

    monkeypatch.setattr(adapters, "get_uma_model", fake_get_uma_model)

    pipeline = MaterialEvalPipeline(compute_uma_forces=True, uma_device="cpu")
    results = pipeline.run([_nacl_structure()])

    assert captured["task"] == "omat"
    assert "uma_scored_rate" in results.aggregate


def test_pipeline_isolates_a_uma_failure_from_every_other_metric(monkeypatch):
    """Guards graceful degradation.

    A missing fairchem install, an HTTP 401 on the gated weights, or a dead GPU
    must cost the run its UMA keys only — validity, uniqueness and diversity
    must still be reported.
    """
    pytest.importorskip("rdkit.Chem")
    import zevals.atomistic.uma.adapters as adapters
    from zevals.pipeline.molecules import MoleculeEvalPipeline

    def explode(*args, **kwargs):
        raise ImportError("UMA force metrics require fairchem-core")

    monkeypatch.setattr(adapters, "get_uma_model", explode)

    pipeline = MoleculeEvalPipeline(compute_posebusters=False, compute_uma_forces=True)
    results = pipeline.run([_ethanol()])

    assert not [key for key in results.aggregate if key.startswith("uma_")]
    assert results.aggregate["valid_rate"] == pytest.approx(1.0)


def test_configs_expose_the_uma_switch():
    """Guards the shipped YAML: the flag must exist and default to off."""
    yaml = pytest.importorskip("yaml")

    configs = Path(__file__).resolve().parents[1] / "configs"
    expected_task = {"molecules.yaml": "omol", "materials.yaml": "omat"}
    for name, task in expected_task.items():
        config = yaml.safe_load((configs / name).read_text())["pipeline"]
        assert config["compute_uma_forces"] is False, name
        assert config["uma_task"] == task, name


# ------------------------------------------------------ real UMA weights


@pytest.fixture(scope="module")
def uma_molecule_model():
    """The real ``omol`` model, or a skip when it cannot be loaded."""
    pytest.importorskip("fairchem.core")
    from zevals.atomistic.uma.model import UMAModel

    try:
        return UMAModel(checkpoint="uma-s-1p2", task="omol")
    except Exception as err:  # no HF token, no network, gated access pending
        pytest.skip(f"UMA omol model unavailable: {err}")


@pytest.fixture(scope="module")
def uma_material_model():
    """The real ``omat`` model, or a skip when it cannot be loaded."""
    pytest.importorskip("fairchem.core")
    from zevals.atomistic.uma.model import UMAModel

    try:
        return UMAModel(checkpoint="uma-s-1p2", task="omat")
    except Exception as err:
        pytest.skip(f"UMA omat model unavailable: {err}")


def test_real_forces_are_the_negative_energy_gradient(uma_molecule_model):
    """Guards a wrong sign, wrong units, or a wrong atom order.

    A finite difference of the energy along one coordinate must equal minus the
    predicted force on that coordinate. All three of those bugs break this
    equality while leaving the output shape correct.
    """
    import torch
    from ase import Atoms

    h = 5e-3
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.2396, 0.9273, 0.0]],
    )
    forces = uma_molecule_model.predict([atoms])["forces"].double().cpu()

    for atom_i, axis in [(0, 0), (1, 1), (2, 0)]:
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions[atom_i, axis] += h
        minus.positions[atom_i, axis] -= h
        e_plus = float(uma_molecule_model.predict([plus])["energy"].reshape(-1)[0])
        e_minus = float(uma_molecule_model.predict([minus])["energy"].reshape(-1)[0])
        numeric = -(e_plus - e_minus) / (2 * h)
        assert torch.isclose(
            torch.tensor(numeric, dtype=torch.float64),
            forces[atom_i, axis],
            atol=5e-2,
        ), f"atom {atom_i} axis {axis}: fd={numeric:.4f} model={forces[atom_i, axis]:.4f}"


def test_real_batched_result_matches_one_at_a_time(uma_molecule_model):
    """Guards a wrong batch index or graph leakage across systems.

    Two molecules in one batch must give the same numbers as two separate
    calls. A wrong batch vector joins them into one graph and the energies
    drift.
    """
    from ase import Atoms

    water = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.2396, 0.9273, 0.0]],
    )
    d = 0.629
    methane = Atoms(
        "CH4",
        positions=[[0, 0, 0], [d, d, d], [-d, -d, d], [d, -d, -d], [-d, d, -d]],
    )

    together = uma_molecule_model.predict([water, methane])
    apart = [uma_molecule_model.predict([m]) for m in (water, methane)]

    assert together["natoms"].tolist() == [3, 5]
    per_system = uma_molecule_model.split_per_system(together, "forces")
    for i, single in enumerate(apart):
        assert abs(float(together["energy"][i]) - float(single["energy"][0])) < 5e-2
        assert (per_system[i].cpu() - single["forces"].cpu()).abs().max() < 5e-2


def test_real_relaxed_molecule_scores_lower_than_a_strained_one(uma_molecule_model):
    """Guards that the metric measures what it claims to.

    A relaxed ethanol conformer must report a smaller maximum residual force
    than the same molecule with one bond stretched. If the number did not track
    strain, the metric would be useless as a geometry check.
    """
    pytest.importorskip("rdkit.Chem")
    from zevals.molecules.uma_forces import compute_uma_force_metrics

    relaxed = _ethanol()
    strained = _ethanol()
    conf = strained.GetConformer()
    pos = conf.GetAtomPosition(0)
    conf.SetAtomPosition(0, (pos.x + 0.6, pos.y, pos.z))

    good = compute_uma_force_metrics([relaxed], model=uma_molecule_model)
    bad = compute_uma_force_metrics([strained], model=uma_molecule_model)

    assert good["uma_scored_rate"] == pytest.approx(1.0)
    assert good["uma_force_max_mean"] < bad["uma_force_max_mean"]


def test_real_periodic_crystal_runs_on_the_omat_head(uma_material_model):
    """Guards the periodic path end to end.

    NaCl at its equilibrium lattice constant sits near a force-free minimum.
    If the cell were dropped, the two ions would stand alone in vacuum and feel
    a large binding force instead.
    """
    pytest.importorskip("pymatgen.core")
    pytest.importorskip("zevals.materials")  # needs the `materials` extra
    from zevals.materials.uma_forces import compute_uma_force_metrics

    out = compute_uma_force_metrics([_nacl_structure()], model=uma_material_model)

    assert out["uma_scored_rate"] == pytest.approx(1.0)
    assert out["uma_force_max_mean"] < 0.5
