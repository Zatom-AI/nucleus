# Protein generation and transfer evaluation

Prepare the frozen SCOPe lists and original structures as described in
[data.md](data.md). Run generation in the Nucleus environment; run ESMFold2
refolding in a separate Python 3.12 environment installed with:

```bash
uv venv --python 3.12 .venv-protein-eval
uv pip install --python .venv-protein-eval/bin/python \
  -e './models/zevals[proteins-esmfold2]'
.venv-protein-eval/bin/zevals setup --tools usalign,mmseqs
.venv-protein-eval/bin/zevals setup --checkpoints soluble_mpnn
```

Download ESMFold2-Fast, its ESMC stem, and CCD assets through the pinned oracle
before an offline cluster run. See the Zevals predictor config. Model access
must be available to the account running evaluation. AtomWorks uses Biotite’s
built-in CCD by default. The paper’s seeded generation driver additionally
requires a readable local mirror through `CCD_MIRROR_PATH`; prepare it using the
[AtomWorks setup guide](https://github.com/amorehead/atomworks).

## Frozen cohorts

Use `nucleus install zatom2-table3-scope-from-joint-properties-force-conditioned`
(or another Table 3 identifier). The following creates the 1,027-backbone
in-range cohort, 13 examples at each length 50–128:

```bash
mkdir -p logs/protein-paper
python scripts/make_scope_unconditional_inputs.py \
  --scope-csv data/scope/rungs/seed0/scope_4000.csv \
  --exclude-csv data/scope/rungs/seed0/scope_2000.csv \
  --output-json logs/protein-paper/inputs.json \
  --manifest-json logs/protein-paper/inputs.manifest.json \
  --lengths "$(seq -s, 50 128)" --samples-per-length 13 --seed 1
python scripts/generate_scope_seeded.py --package zatom2 \
  --checkpoint /path/to/installed/protein-checkpoint.ckpt \
  --inputs logs/protein-paper/inputs.json \
  --output logs/protein-paper/generated --seed 42
python scripts/extract_protein_backbones.py \
  --input-dir logs/protein-paper/generated \
  --output-dir logs/protein-paper/backbones
```

Repeat with seeds **43 and 44** in new output roots, using the same checkpoint
and length cohort. Per-sample RNG derivation is stable across sharding and
resumption; completion records bind the request, RNG seed, and CIF hash.
For the extrapolation cohort use `--lengths "$(seq -s, 129 256)" --samples-per-length 8
--allow-ood-fixed-lengths`, yielding **1,024** backbones. The manuscript caption's
wording is less precise than the archived evaluator on this count; do not
silently change the denominator. Optional stress cohorts are lengths 384 and
512 with 64 samples each.

## Refolding and scoring

```bash
.venv-protein-eval/bin/python scripts/run_scope_self_consistency_shard.py \
  --shard-index 0 --zevals-dir models/zevals \
  --backbone-dir logs/protein-paper/backbones \
  --zevals-output-dir logs/protein-paper/refolds \
  --progress-json logs/protein-paper/progress.json \
  --structure-predictor esmfold2 --esmfold-model-name biohub/ESMFold2-Fast \
  --mpnn-num-sequences 8 --mpnn-temperature 0.1 --mpnn-backbone-noise 0.02 \
  --mpnn-seed 42 --esmfold2-num-loops 3 --esmfold2-num-sampling-steps 50 \
  --esmfold2-num-samples 1 --esmfold2-seed 0 \
  --sc-rmsd-threshold 1.5 --sc-atom CA --tmalign-bin TMalign
python scripts/collect_zevals_self_consistency.py \
  --backbone-dir logs/protein-paper/backbones \
  --zevals-output-dir logs/protein-paper/refolds \
  --out-json logs/protein-paper/self_consistency.json \
  --out-csv logs/protein-paper/refolds.csv \
  --sc-rmsd-threshold 1.5 --sc-atom CA
```

Set the MPNN seed to the generation replicate seed. Retain every backbone and
all eight refolds in denominator audits. Designability means the minimum
C-alpha scRMSD is strictly below 1.5 Å. No pLDDT filter is applied.

Prepare canonical novelty targets with
`prepare_scope2k_canonical_tmalign_targets.py --input-dir
 data/scope/rungs/seed0/pdb_2000 --output-dir artifacts/scope-targets`.
`compute_scope_advanced_novelty_diversity.py` consumes the self-consistency
summary, computes query-normalized exact TM-align novelty against those
training targets, and computes designable structural/joint clusters. Its
`--help` exposes parallel workers and merge stages. Novelty uses maximum TM
<0.5; distinct clusters use TM≥0.5. Joint sequence/structure clusters additionally
require 10% identity and 70% bidirectional coverage.

Checkpoint selection uses a 96-backbone sweep before freezing the four selected
Zatom-2 steps. Do not select a different checkpoint for each final inference
replicate. Report the mean and sample standard deviation across seeds, and retain
per-seed results and failures. The RFdiffusion3 comparison checkpoint used by
the manuscript is outside the released Zatom-2 registry; see release status.
