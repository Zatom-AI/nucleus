# SCOPe-2k training-set designability

**Measured September 19, 2026: 58.20% (1,164 / 2,000 domains).** All 16,000
refolds completed successfully. The pass count was independently recomputed
from the exported per-refold CSV and matched the JSON aggregate exactly.

| Cohort | Domains | Designability | Mean best-of-8 scRMSD (Å) | Mean best-of-8 scTM |
| --- | ---: | ---: | ---: | ---: |
| SCOPe-2k training structures | 2,000 | **0.5820** | **2.1824** | **0.8812** |
| 96-backbone checkpoint sweep: no pretraining | 96 | 0.760 | 1.317 | 0.915 |
| 96-backbone checkpoint sweep: generation + structure | 96 | 0.865 | 1.084 | 0.929 |
| 96-backbone checkpoint sweep: generation + structure + MLIP | 96 | 0.896 | 1.057 | 0.939 |

The training-set median best-of-eight scRMSD is 1.2041 Å, and median
best-of-eight scTM is 0.93245. RMSD is minimized and TM-score maximized
independently across the eight refolds, as in the existing evaluator.

The complete-N/CA/C/O subset gives **58.48% (1,158 / 1,980)**. The remaining
20 domains, each missing one O atom, give **30.0% (6 / 20)**; their treatment
is described below. These are point estimates from one sequence-design run
over the full training cohort. The generated cohorts in 96-backbone checkpoint sweep use a
separate held-out generation-length distribution.

This evaluation measures the designability of all 2,000 domains in
`data/scope/rungs/seed0/scope_2000.csv`, using the protocol for the manuscript's
SCOPe-2k transfer table. It refolds eight SolubleMPNN candidate sequences per
experimental backbone rather than refolding the native sequence.

## Protocol

- SolubleMPNN `solublempnn_v_48_020`: eight sequences per domain, temperature
  0.1, backbone noise 0.02.
- ESMFold2-Fast revision `c6c7958d63f5f2f1f0fed0bb9462316f8ccceea6`:
  three trunk loops, 50 sampling steps, one refold per sequence, seed 0.
- A domain passes if its minimum of eight Cα Kabsch RMSDs is **strictly below
  1.5 Å**. No pLDDT gate is applied. The denominator is all 2,000 domains.
- Sequence-design seeds are the first four bytes of SHA256(domain ID),
  interpreted as an unsigned big-endian integer. This makes resumption and
  GPU sharding independent of sequence-design randomness. ESMFold2 keeps its
  original fixed seed for each refold.

Inputs use the first structural model and Biopython's selected alternate
conformations. Only observed N, CA, C, and O atoms are retained. Residues are
renumbered consecutively in chain A to avoid ProteinMPNN inserting unobserved
residues at source-numbering gaps. Coordinates are unchanged; an exact Cα
coordinate comparison is required for every prepared domain. All observed
lengths match the training CSV. The 20 domains missing one O atom are retained
without imputation. All domains have complete N, CA, and C coordinates.
The existing SolubleMPNN featurizer masks a residue if any N/CA/C/O atom is
missing; the sampler retains its native amino acid at that position. Thus,
one residue is fixed in each of these 20 domains. The complete-backbone
subset (1,980 domains) is also reported to make this input-specific behavior
visible, without excluding domains from the primary denominator.

The cohort covers 488 SCOPe folds, with lengths 20–128 residues (mean 94.882,
median 98). Domain IDs and lengths exactly match the training CSV in the
checkout used for the saved 96-backbone checkpoint sweep evaluations. This is the full
training-cohort base rate, rather than a
subsample matched to the held-out generation-length distribution.

## Reproduction and artifacts

The evaluation driver is `scripts/evaluate_scope_training_designability.py`.
From the repository root, with a compatible zevals ESMFold2 environment:

```bash
python scripts/evaluate_scope_training_designability.py prepare \
  --output logs/analysis/scope_training_base_rate --shards 16

# Run each shard on an allocated GPU, substituting its index and model path.
python scripts/evaluate_scope_training_designability.py run \
  --output logs/analysis/scope_training_base_rate --shard 0 \
  --model /path/to/pinned/ESMFold2-Fast

# Requires all 16 shards and exactly eight refolds for every domain.
python scripts/evaluate_scope_training_designability.py summarize \
  --output logs/analysis/scope_training_base_rate
```

The driver writes the input manifest, coordinate audit, per-refold CSV,
per-domain measurements, subset summaries, and coverage checks under the chosen
output directory. Preserve these with your reproduction run. Original runtime
logs and cluster accounting records are not included in the public source.

The historical Biohub Transformers commit
`ef32577f55da19a4989cd7b22e004dc43a4998cb` was recovered from the
`huggingface/transformers` GitHub archive because the original Biohub fork URL
was unavailable. The ESM commit is
`26b0bc2b771e3e419ea74f445a5f35cc094a1509`; the ESMC-6B snapshot is
`45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a`. A run-local copy of the
historical ESMFold2 config points to that cached stem. Use the pinned sources and model revisions when comparing to the reported scores.
