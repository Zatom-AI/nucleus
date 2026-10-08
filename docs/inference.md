# Inference with Zatom-2

Install the monorepo using the root README. Run examples from its root.

## Checkpoints

```bash
nucleus list-available
nucleus install zatom2
```

The short name `zatom2` aliases
`zatom2-table1-zatom2-l-joint-properties-ablation`. There are 27 distinct paper
checkpoints, each with a SHA-256 in the
[registry](../src/nucleus/inference_engines/checkpoint_registry.py). `install all`
downloads all 27 (many gigabytes); the default example needs only one.
Hub filenames retain their published directory names. `hf auth login` is needed
only for restricted repositories; see [release status](release_status.md).

To use another disk, set the search path in the inference shell too:

```bash
export NUCLEUS_CHECKPOINT_DIRS=/path/to/checkpoints
nucleus install zatom2 --checkpoint-dir "$NUCLEUS_CHECKPOINT_DIRS"
```

`ckpt_path` accepts a registered name or a local checkpoint path. No training
data download is needed for inputless generation. Property normalization values
are bundled, so published checkpoints do not require the author's data paths.

## Molecules, materials, and proteins

```bash
zatom2 design inputs=models/zatom2/docs/examples/demo.json out_dir=logs/molecule
zatom2 design inputs=models/zatom2/docs/examples/material.json out_dir=logs/material
nucleus install zatom2-table3-scope-from-joint-properties-force-conditioned
zatom2 design \
  ckpt_path=zatom2-table3-scope-from-joint-properties-force-conditioned \
  inputs=models/zatom2/docs/examples/protein.json out_dir=logs/protein
```

The model generates CIF files and prediction metadata beneath `out_dir`.
`n_batches=4` generates four batches per input. `diffusion_batch_size` controls
samples per input; `inference_batch_size` controls independent inputs per GPU.
Set `dump_trajectories=true` to retain the denoising trajectory and
`skip_existing=false` to regenerate existing outputs. Trajectories can be large.

The demo is a smoke example. Paper generation uses frozen request manifests:

```bash
zatom2 design ckpt_path=zatom2 \
  inputs=models/zatom2/docs/examples/omol25_train_4m_mean_force_norm_1000_with_hydrogens_with_forces.json \
  out_dir=logs/omol25 classifier_free_guidance=force_only \
  inference_sampler.cfg_scale=2.0
```

The standard paper sampler uses EMA weights, 200 steps, two denoiser passes,
step scale 1.5, noise scale 1.003, schedule exponent 7, and churn 0.6. QM9/MP20
uses its dedicated 100-step, no-guidance evaluation protocol.

## Structure prediction

Structure prediction generates 3D coordinates while keeping the chemical composition and elemental identities fixed from an input structure. In the input JSON specification, this corresponds to:

- `select_fixed_atoms: false` (atomic coordinates are initialized from noise and diffused).
- `select_unfixed_sequence: false` (composition and atom types are preserved from the input structure).

Relative paths in `input` are automatically resolved relative to the directory containing the JSON manifest.

### Material structure prediction

Predict coordinates for a single material structure:

```bash
zatom2 design ckpt_path=zatom2 \
  inputs=models/zatom2/docs/examples/omat24_rattled_300_train_0001_structure_prediction.json \
  out_dir=logs/structure_prediction/omat24_single
```

The manifest specifies the input CIF, contig, and selection flags:

```json
{
  "omat24_rattled_300_train_0001_structure_prediction": {
    "input": "input_cifs/omat24_rattled_300_train_0001.cif",
    "contig": "A1",
    "select_fixed_atoms": false,
    "select_unfixed_sequence": false,
    "mean_force_norm": 1.3231431893640304,
    "has_mean_force_norm": true
  }
}
```

To run a specific target from the 500-sample test cohort:

```bash
zatom2 design ckpt_path=zatom2 \
  inputs=models/zatom2/docs/examples/omat24_rattled_300_structure_prediction_500.json \
  json_keys_subset=omat24_rattled_300_test_000 \
  out_dir=logs/structure_prediction/omat24_500
```

### Molecule structure prediction

Predict coordinates for molecular structures from the 500-target validation cohort:

```bash
zatom2 design ckpt_path=zatom2 \
  inputs=models/zatom2/docs/examples/omol25_geom_orca6_structure_prediction_500.json \
  json_keys_subset=omol25_geom_orca6_val_force_sample_000 \
  out_dir=logs/structure_prediction/omol25_500
```

Molecular inputs provide `total_charge` and `spin_multiplicity` in addition to coordinate and sequence flags:

```json
{
  "omol25_geom_orca6_val_force_sample_000": {
    "example_id": "data0058:4584",
    "input": "input_cifs/omol25_geom_orca6_structure_prediction_500/000_data0058_4584.cif",
    "contig": "A1",
    "select_fixed_atoms": false,
    "select_unfixed_sequence": false,
    "total_charge": 0,
    "spin_multiplicity": 1
  }
}
```

By default, molecule structure prediction perceives covalent bonds on the fly via Open Babel (`bond_sidecar_path=null`). To reproduce the exact paper benchmark using precomputed reference bonds from LMDB sidecars, supply `bond_sidecar_path`:

```bash
zatom2 design ckpt_path=zatom2 \
  inputs=models/zatom2/docs/examples/omol25_geom_orca6_structure_prediction_500.json \
  json_keys_subset=omol25_geom_orca6_val_force_sample_000 \
  bond_sidecar_path=data/omol25/sidecars/val_all_bonds.lmdb \
  out_dir=logs/structure_prediction/omol25_500
```

## MLIP energy and force prediction

Zatom-2 can evaluate Machine Learning Interatomic Potential (MLIP) total potential energy and per-atom forces for fixed input structures without running diffusion rollout. In the input JSON specification, MLIP inference is specified by:

- `select_fixed_atoms: true` (coordinates are fixed; no noise is added).
- `select_unfixed_sequence: false` (composition and atom types are fixed).
- Alternatively, `"atomistic_property_task": "mlip"` can be explicitly set under `extra`.

When these flags are present, the inference engine bypasses generative sampling entirely and executes a single clean-endpoint evaluation through the shared transformer backbone at flow-matching $t=1$.

### Material MLIP prediction

Evaluate energy and forces on a material structure:

```bash
zatom2 design ckpt_path=zatom2 \
  inputs=models/zatom2/docs/examples/omat24_rattled_300_energy_force_prediction_500.json \
  json_keys_subset=omat24_rattled_300_test_000 \
  out_dir=logs/mlip/omat24
```

### Molecule MLIP prediction

Evaluate energy and forces on a molecular structure:

```bash
zatom2 design ckpt_path=zatom2 \
  inputs=models/zatom2/docs/examples/omol25_geom_orca6_energy_force_prediction_500.json \
  json_keys_subset=omol25_geom_orca6_val_force_sample_000 \
  out_dir=logs/mlip/omol25
```

### Inspecting outputs and comparing against ground truth

For each input, the model writes the structure CIF and a prediction metadata JSON to `out_dir`. Predicted physical properties are stored under `"atomistic_properties"`:

```json
{
  "atomistic_properties": {
    "energy_eV": -16.92788575,
    "forces_eV_per_angstrom": [
      [-0.008865, 0.112996, -0.305205],
      [-0.111019, 0.085175, -0.172062]
    ]
  }
}
```

The example manifests embed authoritative DFT reference values under `extra` (`energy_gt_eV` and `forces_gt_eV_per_angstrom`). You can score predictions against these references using the evaluation script:

```bash
python scripts/compare_atomistic_property_predictions.py logs/mlip/omat24/*.json
```

This prints a JSON summary reporting:
- `energy_error_eV` (predicted minus reference energy)
- `force_mae_eV_per_angstrom` (mean absolute error of force vector components)
- `force_rmse_eV_per_angstrom` (root mean square error of force components)
- `force_mean_cosine_alignment` (cosine similarity between predicted and reference force vectors)

## Attaching energy and force predictions to generated samples

Checkpoints with joint-property prediction heads enabled (such as `zatom2`, which aliases `zatom2-table1-zatom2-l-joint-properties-ablation`, or `zatom2-table3-scope-from-joint-properties-force-conditioned`) can automatically compute physical energies and forces for newly generated structures at the final denoising step.

### Requirements for property attachment

1. **Compatible checkpoint**: The checkpoint must include trained property heads (`atomistic_property_prediction_enabled: true`). Generative-only ablation checkpoints do not predict properties.
2. **Force conditioning must NOT be active**: If the input specification conditions on target force statistics (`has_mean_force_norm: true` or `has_force_rms_p95: true`), property reconstruction is deliberately disabled to avoid circular conditioning and information leakage. **Omit force statistics or set `has_mean_force_norm: false`** to enable property prediction on generated structures.
3. **Molecular charge and spin**: Molecule generation requests must include integer `total_charge` (e.g., `0`) and positive integer `spin_multiplicity` (e.g., `1`).
4. **Atom type handling**:
   - For **unconditional generation**, atom types are softly predicted during denoising, and property reconstruction uses masked mode (`atom_type_logits_I`) to weight elemental reference energies.
   - For **structure prediction**, atom types are known from the input structure, and property reconstruction uses known elemental references (`ref_element`).

### Unconditional generation with attached properties

Create an input JSON without force-conditioning fields:

```json
{
  "unconditional_molecule": {
    "contig": "38m",
    "total_charge": 0,
    "spin_multiplicity": 1
  }
}
```

Run generation:

```bash
zatom2 design ckpt_path=zatom2 inputs=my_unconditional_molecule.json out_dir=logs/unconditional_properties
```

The output JSON `<sample_name>.json` will contain `"atomistic_properties"` with `energy_eV` and `forces_eV_per_angstrom` alongside the generated `.cif` coordinates.

### Structure prediction with attached properties

To attach energy and force predictions to structure prediction samples, ensure `has_mean_force_norm` is omitted or `false`:

```json
{
  "material_structure_prediction": {
    "input": "input_cifs/omat24_rattled_300_train_0001.cif",
    "contig": "A1",
    "select_fixed_atoms": false,
    "select_unfixed_sequence": false,
    "has_mean_force_norm": false
  }
}
```

Run structure prediction:

```bash
zatom2 design ckpt_path=zatom2 inputs=my_structure_prediction.json out_dir=logs/structure_properties
```

The resulting `<sample_name>.json` metadata file includes both the structure generation metrics and the decoded `atomistic_properties`.

### Post-hoc property evaluation on existing CIFs

If structures were previously generated with force conditioning, or come from another workflow, you can attach MLIP properties post-hoc without re-running generative diffusion. Simply point an MLIP specification to the generated CIF:

```json
{
  "post_hoc_mlip_evaluation": {
    "input": "/path/to/generated_structure.cif",
    "contig": "A1",
    "select_fixed_atoms": true,
    "select_unfixed_sequence": false
  }
}
```

And run `zatom2 design`:

```bash
zatom2 design ckpt_path=zatom2 inputs=post_hoc_eval.json out_dir=logs/post_hoc_mlip
```

This evaluates the exact generated geometry in a single $t=1$ forward pass and writes `<sample_name>.json` with the predicted energy and forces.

## Score generated samples

```bash
zatom2 evaluate-generation ckpt_path=zatom2 out_dir=logs/evaluation/omol25 \
  evaluation=omol25_unconditional
zatom2 evaluate-generation ckpt_path=zatom2 out_dir=logs/evaluation/omat24 \
  evaluation=omat24_unconditional
```

These commands need the training-reference fingerprints and external AFD/UMA
oracles described in [data preparation](data.md). They save individual samples,
manifests, and computed metrics. `evaluation=omol25_structure_prediction` and
`evaluation=omat24_structure_prediction` use the included 500-target manifests;
set `evaluation.expected_num_samples=500` for these protocols.

Energy/force evaluation streams the original validation shards:

```bash
zatom2 evaluate-mlip ckpt_path=zatom2 out_dir=logs/evaluation/mlip
```

Set `OMOL25_VAL_DIR` and `OMAT24_VAL_DIR` first. GPU execution and the complete
datasets are required for full-paper evaluation. `evaluation.max_examples_per_domain=10`
is a pipeline smoke check, not a reproduction result.
