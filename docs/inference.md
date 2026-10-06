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
