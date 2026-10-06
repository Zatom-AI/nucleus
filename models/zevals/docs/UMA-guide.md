# UMA force metrics

## Words used in this document

| Word | What it means |
|---|---|
| **UMA** | Universal Model for Atoms. A model from Meta FAIR that gives the energy and the forces of a set of atoms. |
| **DFT** | Density Functional Theory. A quantum-mechanics calculation. It is accurate but very slow. UMA is trained on DFT results, so it gives almost the same answer much more quickly. |
| **UFF** | Universal Force Field. A classical force field in RDKit. It calculates forces from simple rules about bonds and angles, not from quantum mechanics. It is fast but much less accurate than UMA. |
| **Force** | How hard each atom is pushed. The unit is eV/A (electron-volts per Angstrom). A structure at rest has almost no force on its atoms. |
| **Residual force** | The force that stays on an atom in a structure that should be at rest. A large residual force shows a bad structure. |
| **Task head** | One of several output layers in the UMA model. Each head is trained on a different type of chemistry. You must select the correct head. |
| **Periodic** | The structure repeats in all directions, as in a crystal. A molecule is not periodic. |

## What this module does

UMA gives energies and forces in one forward pass. `zevals` uses it as a
**geometry check**. A generated 3D structure that is near a true energy minimum
has small residual forces on its atoms. A strained or a badly generated
structure has a large maximum force.

This is the near-DFT equivalent of the UFF force norm
(`zevals.molecules.conformer_quality`). UMA is much more accurate. UMA also
works for periodic crystals, which UFF does not do. But UMA needs a GPU, one
more dependency, and a gated model download. Therefore this module is
**optional, and it is off by default**.

Contents:

1. [Install](#1-install)
2. [Switch it on](#2-switch-it-on)
3. [What you get](#3-what-you-get)
4. [Molecules vs materials — the task flag](#4-molecules-vs-materials--the-task-flag)
5. [Direct use in Python](#5-direct-use-in-python)
6. [Performance and device](#6-performance-and-device)
7. [Caveats](#7-caveats)
8. [Test](#8-test)
9. [Sources](#9-sources)

---

## 1. Install

### 1.1 The package

```bash
uv pip install -e '.[uma]'      # fairchem-core + torch~=2.8 + numpy<2.5 + ase
```

**Use a separate environment from your trainer.** `fairchem-core` 2.21 pins
`torch~=2.8.0` and `numpy<2.5`. Install it next to a newer torch and it
downgrades the environment, which silently breaks CUDA support elsewhere. For
this reason `zevals[all]` deliberately does **not** include `uma`.

Three ways to work around the pin:

| Situation | What to do |
|---|---|
| Evaluation runs in its own job or environment | Install `zevals[uma]` there. Simplest, recommended. |
| You want UMA **inside** a training loop that needs a newer torch | Install fairchem with the pins overridden — see §1.3. Verified working. |
| You cannot change the training environment | Dump generated structures to disk and score them in a second job. |

### 1.3 Installing beside a newer torch (verified recipe)

fairchem's `torch~=2.8` / `numpy<2.5` pins are **metadata only**: the UMA
inference path itself runs correctly on newer torch and numpy versions. To keep
the versions already installed in your environment, override the pins rather
than letting the resolver downgrade them:

```bash
# Read the installed distribution versions, which are the versions uv resolves.
torch_version=$(python -c 'import importlib.metadata as m; print(m.version("torch"))')
numpy_version=$(python -c 'import importlib.metadata as m; print(m.version("numpy"))')
printf 'torch==%s\nnumpy==%s\nlmdb>=2.3.0\n' \
    "$torch_version" "$numpy_version" > /tmp/uma-overrides.txt

# From the zevals directory:
uv pip install --overrides /tmp/uma-overrides.txt -e '.[uma]'
# From the Nucleus repository root, use -e './models/zevals[uma]' instead.
uv pip install 'setuptools<81'   # torchtnt still imports pkg_resources
```

Pin torch exactly. A lower bound such as `torch>=2.11` still permits the
resolver to upgrade it and potentially replace a working CUDA build. Also use
the distribution version returned by `importlib.metadata`, **not**
`torch.__version__`: CUDA builds can report a runtime version such as
`2.12.1+cu130` even though their resolvable package metadata version is
`2.12.1`. Pinning the runtime string can produce a misleading "no version of
torch" resolution error.

Check the result before trusting it:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Two traps worth knowing:

* **`--no-deps` does not work.** fairchem needs `ray`, `e3nn`, `torchtnt`,
  `submitit` and more at import time, and hand-chasing that chain dead-ends.
  Use the overrides file instead.
* **A conda env may need `LD_LIBRARY_PATH`.** fairchem imports `ase.db.sqlite`,
  hence `sqlite3`, whose extension can resolve the system `libstdc++` instead of
  the env's and fail with ``version `CXXABI_1.3.15' not found``. Fix by
  exporting `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` for the run. This is a
  pre-existing property of such an environment, not something UMA introduces.

### 1.2 The weights

UMA weights are gated on Hugging Face. Do this one time:

1. Request access at <https://huggingface.co/facebook/UMA>. Approval usually
   takes a few minutes.
2. Make a read token at <https://huggingface.co/settings/tokens>.
3. Log in:

```bash
hf auth login          # or: export HF_TOKEN=hf_xxxxxxxx
```

Without a token the first model load fails with HTTP 401. `zevals` turns that
into a warning and skips the UMA metrics — every other metric in the run still
reports (see [§7](#7-caveats)).

### 1.3 Checkpoints

| `uma_checkpoint` | Active / total parameters | Note |
|---|---|---|
| `uma-s-1p2` | 6.6 M / 290 M | the recommended small model, the default |
| `uma-s-1p1` | 6.6 M / 150 M | the earlier small model |
| `uma-m-1p1` | 50 M / 1.4 B | the most accurate public model |

A path that ends in `.pt` loads a local inference checkpoint instead of
downloading one.

---

## 2. Switch it on

One flag per pipeline, `compute_uma_forces`. It is `false` by default, and
while it is false **no `uma_*` key appears in the results** and `fairchem` is
never imported.

### 2.1 Molecules — `configs/molecules.yaml`

```yaml
pipeline:
  _target_: zevals.pipeline.molecules.MoleculeEvalPipeline
  compute_uma_forces: true
  uma_checkpoint: uma-s-1p2
  uma_task: omol       # molecules head: aperiodic, charge aware
  uma_device: null     # null/"auto" picks CUDA when available
  uma_batch_size: 8
```

### 2.2 Materials — `configs/materials.yaml`

```yaml
pipeline:
  _target_: zevals.pipeline.materials.MaterialEvalPipeline
  compute_uma_forces: true
  uma_checkpoint: uma-s-1p2
  uma_task: omat       # inorganic crystals head: periodic
  uma_device: null
  uma_batch_size: 8
```

### 2.3 Through `ZevalSuite`

```python
from zevals.pipeline.suite import ZevalSuite

suite = ZevalSuite(
    modality_overrides={
        "molecules": ["pipeline.compute_uma_forces=true"],
        "materials": ["pipeline.compute_uma_forces=true", "pipeline.uma_batch_size=4"],
    }
)
results = suite.evaluate({"molecules": "path/to/generated/", "materials": "path/to/cifs/"})
# results["molecules/uma_force_max_mean"], results["materials/uma_force_max_mean"], ...
```

### 2.4 Nucleus structure-prediction validation

`ZevalsStructurePredictionMetrics` dispatches each validation example to the
aperiodic molecule or periodic material UMA wrapper. When the inference
specification enables `has_mean_force_norm`, it also logs
`uma_mean_force_norm_mae_eV_per_angstrom` against that example's conditioning
target.

```yaml
structure_prediction_metrics:
  _target_: nucleus.metrics.structure_prediction.ZevalsStructurePredictionMetrics
  required_tags_all: [structure_prediction]
  compute_uma_forces: true
  uma_checkpoint: uma-s-1p2
  uma_molecule_task: omol
  uma_material_task: omat
  uma_device: null
  uma_batch_size: 8
```

The molecule and material task settings are separate because the same metric
adapter handles both domains. UMA remains off by default in the reusable
trainer metric presets.

---

## 3. What you get

Six keys per modality, plus an optional conditioning-adherence key when a
reference mean force norm is supplied. `ZevalSuite` namespaces them, e.g.
`molecules/uma_force_max_mean`.

| Key | Unit | Meaning |
|---|---|---|
| `uma_force_max_mean` | eV/A | Mean over samples of the **largest force on any atom**. The main signal. Lower is better. A large value means at least one atom is badly placed. |
| `uma_force_max_median` | eV/A | The same quantity, median instead of mean. Use it when a few very bad samples would otherwise dominate. |
| `uma_mean_force_norm_mean` | eV/A | Mean over samples of each structure's mean per-atom force norm, `mean_i ||F_i||_2`. This matches the repository's `mean_force_norm` conditioning convention. |
| `uma_mean_force_norm_mae_eV_per_angstrom` | eV/A | Mean over scored structures of the absolute difference between the UMA mean per-atom force norm and `target_mean_force_norm`. Nucleus validation supplies this target automatically when the inference JSON sets `has_mean_force_norm: true`. |
| `uma_force_rms_mean` | eV/A | Mean over samples of the root-mean-square force across atoms. A whole-structure measure, not a worst-atom one. |
| `uma_energy_per_atom_mean` | eV/atom | Reference only. UMA energies are **absolute**, so compare this number only inside one chemical system. |
| `uma_scored_rate` | 0–1 | Fraction of `total_samples` that UMA could score. |

Read `uma_scored_rate` first. It is the denominator-aware honesty check: it
counts against **everything generated**, not against the samples that happened
to convert. A rate of 0.0 means the other requested values are all 0.0
placeholders and nothing was actually measured.

Rough reading of `uma_force_max_mean` for small organic molecules:

| Value | Reading |
|---|---|
| < 0.1 | at or very near a relaxed minimum |
| 0.1 – 0.5 | a plausible, slightly strained conformer |
| 0.5 – 2 | visibly distorted geometry |
| > 2 | broken: a clash, a wrong bond length, or a wrong element |

These are guides, not thresholds. Compare runs against each other, and against
the same metric computed on your training set, rather than against an absolute
number.

## 4. Molecules vs materials — the task flag

One UMA checkpoint holds several **task heads**. `uma_task` picks the head, and
the right value depends on the chemistry. The wrong head does not raise — it
returns plausible-looking numbers that mean nothing.

| `uma_task` | Use it for | Periodic | Training data |
|---|---|---|---|
| `omol` | molecules, protein-ligand systems | no | OMol25, DFT wB97M-V/def2-TZVPD |
| `omat` | inorganic crystals | yes | OMat24, DFT PBE |
| `omc` | molecular crystals | yes | OMC25 |
| `odac` | metal-organic frameworks | yes | ODAC23 |
| `oc20` | catalyst surfaces and adsorbates | yes | OC20 |

`zevals` sets the flag for you, and each modality wrapper also sets the input
flags that go with it:

| | molecules (`zevals.molecules.uma_forces`) | materials (`zevals.materials.uma_forces`) |
|---|---|---|
| Default task | `omol` | `omat` |
| Input type | RDKit `Chem.Mol` | pymatgen `Structure` |
| Cell | none | the lattice matrix |
| `pbc` | `False` | `True` |
| Charge | the molecule's formal charge by default; Nucleus inference evaluation overrides it with the ground-truth `total_charge` | not used by the periodic heads |
| Spin | multiplicity 1 by default; Nucleus inference evaluation overrides it with the ground-truth `spin_multiplicity` | not used |

Two guards are built in:

* The materials wrapper **rejects** an aperiodic head (`uma_task: omol`) with a
  `ValueError`. Without the cell, a perfectly relaxed crystal looks like
  isolated atoms in vacuum and reports large fake forces — verified: relaxed
  NaCl scores `2e-7` eV/A with the cell and `0.35` eV/A without it.
* A molecule is never given a cell and never gets `pbc=True`, so it cannot
  interact with a periodic image of itself.

For a molecular crystal, use the **materials** pipeline with `uma_task: omc` —
it is periodic, so it belongs there, not in the molecules pipeline.

---

## 5. Direct use in Python

### 5.1 The metric functions

```python
from zevals.molecules.uma_forces import compute_uma_force_metrics as mol_uma
from zevals.materials.uma_forces import compute_uma_force_metrics as mat_uma

mol_uma(mols, total_samples=len(mols))            # RDKit Mols, task omol
mat_uma(structures, total_samples=len(structures))  # pymatgen Structures, task omat
# {'uma_scored_rate': 0.8, 'uma_force_max_mean': 0.928, 'uma_force_max_median': 0.902,
#  'uma_mean_force_norm_mean': 0.441, 'uma_force_rms_mean': 0.496,
#  'uma_energy_per_atom_mean': -636.5}

mol_uma(mols, target_mean_force_norm=0.25)
# also reports 'uma_mean_force_norm_mae_eV_per_angstrom'
```

Both follow the `zevals` metric contract: they take a plain list, they never
raise because of one bad sample, they accept `total_samples`, and they return a
flat dict of plain floats. A `None` item, a molecule with no 3D conformer, or a
structure that fails to convert is skipped and shows up in `uma_scored_rate`.

### 5.2 The model itself

```python
from zevals.atomistic.uma import UMAModel

model = UMAModel(checkpoint="uma-s-1p2", task="omol", device="cuda")

out = model.predict([atoms_a, atoms_b])   # ASE Atoms, a list of them, or a PyG Batch/Data
out["energy"]   # tensor [2]      eV
out["forces"]   # tensor [8, 3]   eV/A
out["natoms"]   # tensor [3, 5]
out["batch"]    # tensor [8]      which system each atom belongs to

per_system = UMAModel.split_per_system(out, "forces")   # [ [3,3], [5,3] ]
```

### 5.3 Helpers

| Function | Does |
|---|---|
| `zevals.atomistic.uma.get_uma_model(...)` | Returns a **cached** `UMAModel`, one per (checkpoint, task, device). A training loop calling a metric each validation epoch therefore loads the weights once. |
| `zevals.atomistic.uma.clear_model_cache()` | Drops every cached model and frees its GPU memory. |
| `mol_to_atoms(mol)` | RDKit `Mol` -> aperiodic ASE `Atoms`, or `None`. |
| `structure_to_atoms(structure)` | pymatgen `Structure` -> periodic ASE `Atoms`, or `None`. |
| `samples_to_atoms(samples)` | A mixed list -> ASE `Atoms`, dropping failures. |
| `pyg_batch_to_atoms(batch)` | A `torch_geometric` `Batch`/`Data` -> a list of ASE `Atoms`. |

Pass an already-built model with `model=` to skip the cache entirely:

```python
model = UMAModel(task="omol", device="cuda")
mol_uma(batch_1, model=model)
mol_uma(batch_2, model=model)
```

---

## 6. Performance and device

* **Device.** `uma_device: null` (or `"auto"`) picks CUDA when it is available,
  otherwise CPU. `"cuda"` and `"cpu"` force the choice.
* **fairchem cannot target a GPU index.** `"cuda:1"` is rejected by fairchem
  itself. `zevals` reduces an indexed string to `"cuda"` and logs a warning, so
  `zevals.utils.devices.resolve_device`'s round-robin output does not crash the
  run. To pin a specific GPU, set `CUDA_VISIBLE_DEVICES` before the process
  starts.
* **Batch size.** `uma_batch_size` is systems per forward pass, 8 by default.
  Large unit cells build large neighbour graphs — lower it to 2 or 4 if you hit
  GPU out-of-memory. A batch that fails is dropped and the run continues; the
  loss shows up in `uma_scored_rate`.
* **Cost.** UMA is one forward pass per structure, so it is far cheaper than
  DFT, and far more expensive than UFF or PoseBusters. On CPU it is usable for
  a few hundred small molecules; for thousands, use a GPU.

---

## 7. Caveats

* **Do not wrap a call in `torch.no_grad()` or `torch.inference_mode()`.** UMA
  is a *conservative* model: forces are the gradient of the energy, so the
  forward pass must build an autograd graph. Under `no_grad` the forces come
  back as zeros or the call fails outright.
* **Energies are absolute.** `uma_energy_per_atom_mean` is a total DFT-reference
  energy per atom, not a formation energy. Comparing it across different
  elements is meaningless.
* **A conformer is required.** Molecules parsed from bare SMILES have no 3D
  geometry and are skipped, exactly like every other conformer-level check in
  `zevals`. `uma_scored_rate` is the tell: `0.0` means your input had no
  geometry to score.
* **Failure is isolated, not fatal.** A missing `fairchem`, an HTTP 401 on the
  gated weights, or a dead GPU costs the run its `uma_*` keys only — validity,
  uniqueness, diversity and every other metric still report. The failure is
  logged as a warning by `pipeline/base.py::safe_update`, so check the log if
  the keys are absent when you expected them.
* **Out-of-domain chemistry.** Each head covers the chemistry of its training
  set. Elements or bonding far outside it give unreliable forces rather than an
  error.

---

## 8. Test

```bash
pytest -q tests/test_uma_forces.py
```

The tests split in two:

* **Contract and wiring tests** run everywhere — no `fairchem`, no weights, no
  network. A fake model stands in for UMA, so the task flags, the periodic
  flags, the rate arithmetic, the batching, the failure isolation and the
  pipeline switch are all checked in a normal environment.
* **Real-model tests** need `fairchem-core` and the gated `uma-s-1p2` weights.
  They are skipped, not failed, when either is missing.

What each real-model test guards:

| Test | Failure it catches |
|---|---|
| `test_real_forces_are_the_negative_energy_gradient` | wrong sign, wrong units, or wrong atom order |
| `test_real_batched_result_matches_one_at_a_time` | a wrong batch index, systems merged into one graph |
| `test_real_relaxed_molecule_scores_lower_than_a_strained_one` | a metric that does not actually track strain |
| `test_real_periodic_crystal_runs_on_the_omat_head` | cell or `pbc` lost on the pymatgen path |

Tests that touch the materials pipeline also need the `materials` extra
(`smact`, `matminer`) and skip without it.

---

## 9. Sources

* fairchem: <https://github.com/facebookresearch/fairchem>
* UMA weights and checkpoint list: <https://huggingface.co/facebook/UMA>
* UMA paper: <https://arxiv.org/abs/2506.23971>
