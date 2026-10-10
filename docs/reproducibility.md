# Reproducing Zatom-2

This guide follows the included **arXiv preprint**, including its appendices.
Some Hydra filenames retain earlier table numbering: `paper-table3-*` pretraining
configs correspond to final Table 1, `paper-table4-scope` to final Table 3, and
`paper-figure1-*` to final Figure 3. Checkpoint CLI identifiers use the final
numbering. Consult [release status](release_status.md) for external artifacts
still needed for a complete public reproduction.

## Environment and data

Use the root lockfile, Python 3.12, and [data preparation](data.md). Training uses
80 GB A100 GPUs; complete runs require thousands of GPU hours. The manuscript
contains per-run compute estimates. CSV logging is the default; W&B is optional.
Record the Git revision, `uv.lock`, input hashes, output configuration, RNG seeds,
and checkpoint SHA-256 with every reproduction run.

The portable launcher preserves the approximate effective optimizer token batch
and prints the exact command. It does not allocate cluster resources:

```bash
python scripts/train_paper.py paper-table3-joint-properties-force-conditioned \
  --nodes 1 --devices 4 --dry-run
python scripts/train_paper.py paper-table3-joint-properties-force-conditioned \
  --nodes 1 --devices 4
```

For multiple nodes, run within an allocation configured for Lightning Fabric DDP
with one process per GPU and consistent rendezvous settings. Specify the actual
node/device counts. With 64 GPUs, base/small runs use one accumulation step and
7,168 tokens/GPU; large, QM9/MP20, and MLIP finetunes use two accumulation steps
and 3,584 tokens/GPU. Smaller allocations increase accumulation to retain that
budget. Dynamic packing means the token cap is an upper bound, not a fixed
number of systems. Smoke overrides after `--` are useful for debugging but
change the paper protocol.

## Pretraining and scaling

| Manuscript setting | Hydra experiment | Checkpoint identifier |
| --- | --- | --- |
| Table 1, I: small, generation, no FC | `paper-table1-zatom2-s-omol25-unconditional-without-force-conditioning` | `zatom2-table1-unconditional-no-force-ablation` |
| Table 1, II: small, generation, FC | `paper-table1-zatom2-s-omol25-unconditional-force-conditioned` | `zatom2-table1-unconditional-ablation` |
| Table 1, III: small, generation + structure, FC | `paper-table1-zatom2-s-omol25-joint-force-conditioned` | `zatom2-table1-ablation` |
| Table 1, IV: base, mixed domains, generation + structure, FC | `paper-table3-joint-force-conditioned` | `zatom2-table1-joint-ablation` |
| Table 1, V: base, mixed domains, all tasks, FC | `paper-table3-joint-properties-force-conditioned` | `zatom2-table1-joint-properties-ablation` |
| Table 1, VI: large, mixed domains, all tasks, FC | `paper-table3-zatom2-l-joint-properties-force-conditioned` | `zatom2-table1-zatom2-l-joint-properties-ablation` |
| Base, generation + structure, no FC | `paper-table3-joint-without-force-conditioning` | `zatom2-table1-joint-no-force-ablation` |
| Figure 3, base 500k/domain | `paper-figure1-zatom2-500k-each-joint-properties-force-conditioned` | `zatom2-figure3-500k-joint-properties-ablation` |
| Figure 3, large 500k/domain | `paper-figure1-zatom2-l-500k-each-joint-properties-force-conditioned` | `zatom2-figure3-zatom2-l-500k-joint-properties-ablation` |

Use any experiment above with `scripts/train_paper.py`. Pretraining uses 80
epochs, Adam at 1e-4, equal domain sampling with replacement, and 4,996,604 draws
per mixed-domain epoch. Task probabilities are 0.5/0.5 for generation/structure
or 0.34/0.33/0.33 with property prediction. Force conditioning is applied to
80% of eligible generation/structure examples. The config files preserve loss
weights, warmup, augmentations, crop budgets, and architecture sizes.

Scaling uses a fixed 500k support per domain with the same full-corpus-equivalent
sampling budget. Compare matched checkpoint epochs; do not interpret an epoch as
one traversal of the smaller support. Registered checkpoints include raw epochs
7, 15, 19, and 39; CLI suffixes 8, 16, and 40 count completed epochs.

## Generation, force conditioning, and structure prediction

Use the [inference/evaluation commands](inference.md) with each Table 1 checkpoint.
The frozen generation requests contain 1,000 targets per domain; structure
prediction uses 500 targets and one candidate per target. Force consistency uses
UMA-S-1p2 without relaxation, with the `omol`/`omat` heads and cutoff 1 eV/Å
(ANI2x uses 2.5). Inference uses 200 steps and force-only CFG scale 2.

For structure prediction, supply the validation bond sidecar for OMol25. Use
`evaluation.expected_num_samples=500` with the structure-prediction evaluation
presets. The evaluator exports exact denominators, computed scores, and sample
provenance. Missing or failed oracle evaluations must be reported, not replaced
by successful-only denominators.

## Standard benchmarks

```bash
python scripts/train_paper.py \
  paper-table2-qm9-mp20-unconditional-without-force-conditioning --devices 4
python scripts/train_paper.py train-geom-drugs --devices 4
```

QM9/MP20 training runs for up to 10,000 epochs; the released benchmark checkpoint
and 100-step EMA protocol are evaluated by `zatom2 evaluate-table2`. Follow the
[full evaluation guide](../models/zatom2/docs/table2_evaluation.md), including
RDKit/OpenBabel/PoseBusters pins and the separate locked LeMat-GenBench environment.
This attempts 10,000 QM9 molecules and 2,500 MP20 structures with seed 42.

GEOM generation uses the pinned processed GEOM training split, 80 completed
epochs, and the 10,000-request manifest. Use
`scripts/evaluate_geom_zatom1.py` and `scripts/evaluate_geom_zatom1_supplement.py`
for the Zatom-1-compatible validity, PoseBusters, novelty, and diversity protocol.
They require the public [Zatom-1 evaluator](https://github.com/Zatom-AI/zatom).
The separate GEOM conformer and Crystalite MP20 structure-prediction appendix
experiments require their own splits/checkpoints; see release status.

## Energy and force prediction (Table 10)

Evaluate pretrained models with `zatom2 evaluate-mlip`. For finetuning, use
`paper-table3-finetune-joint-properties` and bootstrap from the corresponding
**epoch checkpoint** of configurations IV or V:

```bash
python scripts/train_paper.py paper-table3-finetune-joint-properties \
  --devices 4 --checkpoint /path/to/epoch-0079.ckpt --bootstrap
```

The config retains the pretraining epoch counter and stops at 120 completed
epochs (40 additional epochs from raw epoch 79). It uses full precision,
0.1/0.1/0.8 generation/structure/MLIP probabilities, and zero MLIP loss warmup.
The extended V finetune adds another 120 epochs. Match the raw epoch named in the
checkpoint registry; the `epoch240` identifier is raw epoch 236, not 240.
Resume the same branch with `--checkpoint` without `--bootstrap`; exact mid-epoch
resume requires the original world size, token budget, and accumulation.

## Protein transfer (Table 3)

```bash
python scripts/train_paper.py paper-table4-scope --devices 4
python scripts/train_paper.py paper-table4-scope --devices 4 \
  --checkpoint /path/to/epoch-0080.ckpt --bootstrap
```

Use configuration IV, V, and IV without force conditioning for the three
pretrained initializations. The launcher preserves 64 input systems/update (16
GPU equivalents × four systems), four views/system, 128-token crops, and a
50,000-update budget. Bootstrap discards incompatible optimizer/rank state and
adjusts the stopping epoch; it is a new branch, not exact pretraining continuation.

Paper-selected global steps are 25,000 (scratch), 120,000 (IV), 130,000 (V), and
125,000 (IV without FC). They correspond to 25k/20k/30k/25k protein updates.
Checkpoint sweeps use 96 generated backbones and best-of-eight refolding.

Follow [protein evaluation](protein_evaluation.md) for the 1,027 in-range and
1,024 extrapolation cohorts, repeated with inference seeds 42, 43, and 44.
Training data seed is 0; inference replicates are not independent training runs.
The abstract/table report means and standard deviations across these repeats.

## Representation analyses, refinement, and figures

| Analysis | Included implementation |
| --- | --- |
| Figure 3 scaling | `analyze_zatom2_figure2.py`, `recompute_zatom2_figure2_metrics.py`, `plot_zatom2_figure2_epochs.py` |
| Protein t-SNE and localized perturbations | `probe_protein_embeddings.py`, `analyze_protein_embeddings.py` |
| Molecular/material subset and element embeddings | `plot_zatom2_subset_embeddings.py`, `plot_zatom2_element_embeddings.py` |
| Representation alignment | `zatom2_representation_alignment.py` |
| Zero-shot denoising/refinement | `probe_zero_shot_relaxation.py` |
| Property rotation/translation equivariance | `probe_atomistic_property_equivariance.py` |
| Protein novelty/diversity and checkpoint plots | `compute_scope_advanced_novelty_diversity.py`, `plot_scope_self_consistency_comparison.py`, `plot_scope_novelty_diversity_comparison.py` |
| Molecular/material/protein galleries | `models/zatom2/docs/manuscript/visualization/render.py` |
| Interactive trajectories | `prepare_project_page_trajectories.py` |

Scripts are under `scripts/` unless otherwise shown. Each command's `--help`
describes required local inputs. Compact embedding arrays, selected structure
IDs, result tables, and manuscript figures are included under
`models/zatom2/docs/manuscript/figs`. Analysis scripts that read training histories
need the outputs of your reproduction run or the original archived histories;
those histories are not fabricated from manuscript tables.

Build the paper with `latexmk -pdf arxiv_preprint.tex` in its manuscript
directory. The preprint is available on [arXiv](https://arxiv.org/abs/2610.11454). To preview the
page, run `python -m http.server 8000 --directory models/zatom2/docs`.
