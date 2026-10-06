# Reproducing the Zatom-2 Table 2 evaluation

The `zatom2 evaluate-table2` command generates and scores the QM9 and MP20
samples used for the Zatom-2 manuscript's Table 2. Its checked-in defaults are
the paper protocol: 10,000 QM9 attempts, 2,500 MP20 attempts, seed 42, EMA
weights, and the checkpoint-native 100-step AF3 sampler. The command saves the
samples, exact denominators, hashes, resolved configuration, and complete
LeMat-GenBench output; it does not infer or insert manuscript results.

Run the commands below from the Nucleus repository root. Prepare the QM9 and
MP20 ASE-LMDB training partitions first by following the
[QM9/MP20 data guide](../../../docs/qm9_mp20.md). The evaluation
uses those exact training partitions for empirical atom-count sampling.

## Table 2 chemistry environment

QM9 scoring follows [Zatom-1 Table 2 and Appendix B.3](https://arxiv.org/html/2602.22251v4#S3.T2)
and its [released evaluator](https://github.com/Zatom-AI/zatom/blob/0d4676571d9958e1bc072fe57144451474d2e3ac/zatom/eval/molecule_generation.py).
Use the four chemistry package versions declared by that release. In particular,
PoseBusters 0.6 changes the internal-energy calculation relative to 0.5.1.
The Table 2 scorer rejects a different chemistry stack.

Use a temporary `uv` overlay on the already installed Nucleus environment for
every Table 2 invocation. Keep the chemistry pins out of the main `.venv`:
do not install this requirements file with `uv pip install` or add its pins to
the project's dependencies. Define this helper from the repository root:

```bash
table2() {
    uv run --no-sync \
        --with-requirements models/zatom2/docs/table2_qm9.requirements.txt \
        python -m zatom2.cli evaluate-table2 "$@"
}
```

`--no-sync` prevents syncing the project environment and updating `uv.lock`.
`--with-requirements` layers the requested packages in a separate temporary
environment; it does not replace the installed packages in `.venv`, even
temporarily. Downloaded packages may remain in the uv cache. See the
[uv environment documentation](https://docs.astral.sh/uv/reference/cli/#uv-run).

You can inspect the environment without running an evaluation:

```bash
# Main environment: run again after the overlay to confirm its version is unchanged.
uv run --no-sync python -c \
    'import sys; from importlib.metadata import version; print(sys.prefix, version("posebusters"))'

# Temporary Table 2 environment: this must report PoseBusters 0.5.1.
uv run --no-sync \
    --with-requirements models/zatom2/docs/table2_qm9.requirements.txt \
    python -c \
    'import sys; from importlib.metadata import version; print(sys.prefix, version("posebusters"))'
```

Use the helper for Table 2 generation, scoring and reporting; reporting
recomputes the QM9 results. The requirements pin RDKit 2025.3.5,
OpenBabel-wheel 3.1.1.22, pymatgen 2024.11.13, and PoseBusters 0.5.1.
The scorer records package versions and hashes the PoseBusters configuration
and implementation. Its dedicated QM9 path does not alter the shared molecule
evaluators or their PoseBusters configuration.

## One-time LeMat-GenBench setup

LeMat-GenBench has a PyTorch, PyG, MACE, ORB, and UMA stack that is independent
of the Nucleus environment. Keep it in the isolated Python 3.11 environment
created by the setup helper; do not install it into the Nucleus Python 3.12
environment.

Choose a new absolute path. The helper refuses to overwrite an existing path:

```bash
export LEMAT_GENBENCH_DIR=$HOME/lemat-genbench-table2

(
    cd models/zevals
    bash scripts/setup_lematgenbench.sh "$LEMAT_GENBENCH_DIR"
)
export ZEVALS_LEMAT_GENBENCH_REPO="$LEMAT_GENBENCH_DIR"
```

The helper checks out LeMat-GenBench commit
`58e6eae3e4a6c87c22171cf069123ecc4e2fa7e6`, verifies the committed `uv.lock`
SHA-256, and runs `uv sync --python 3.11 --frozen`. It skips Git LFS smudging
because the upstream LFS quota is currently exhausted; the Table 2
`structure-matcher` protocol does not use the affected augmented-BAWL files.
Do not run `uv lock` in this checkout.

The comprehensive benchmark requires gated UMA weights. Request access to
[`facebook/UMA`](https://huggingface.co/facebook/UMA), authenticate the account
visible to the evaluation job, and verify that account before scoring:

```bash
hf auth login
hf auth whoami
```

Use the CLI from an installed environment (for example `.venv/bin/hf`) if a
stale `hf` executable on `PATH` reports a missing Python interpreter. The
launcher does not modify authentication or repair external environments.

Before allocating a GPU, follow the
[manual checkpoint download instructions](table2_checkpoints.md) to populate
and hash-check the exact ORB, MACE, and UMA files, including UMA's atomic
references. Keep ORB's `CACHED_PATH_CACHE_ROOT` on shared scratch to avoid
home-filesystem lock stalls.

Leave the checkout's `data/convex_hulls/` directory absent. Files placed there
are unversioned overrides of the pinned Hugging Face hull references, so the
adapter rejects them for manuscript runs. ORB, MACE, UMA, and reference data
may be fetched on first use through verified download hooks; the resulting
standard caches can be reused on compute nodes without outbound network access.
The similarly named upstream `download_above_hull_datasets.py` script fetches
legacy reference files and does not prepare the comprehensive multi-MLIP hull
inputs used here. See the
[zevals LeMat-GenBench guide](../../zevals/docs/LeMat-GenBench-guide.md) for the
frozen lock hash, upstream data revisions, adapter artifacts, and current
upstream limitations.

## Evaluate a paper checkpoint

Point the harness at the exact training partitions and the installed paper
checkpoint. Use a fresh output directory for a new checkpoint or protocol.

```bash
export QM9_DATA_DIR="$(pwd)/data/qm9/raw/train_100k"
export MP20_DATA_DIR="$(pwd)/data/mp_20/raw/train_27138"
export ZEVALS_LEMAT_GENBENCH_REPO="$LEMAT_GENBENCH_DIR"

CHECKPOINT=/absolute/path/to/installed/paper-checkpoint.ckpt
OUTPUT=logs/evaluation/paper-table2-qm9-mp20-unconditional-without-force-conditioning/epoch-0100

table2 \
    "ckpt_path=$CHECKPOINT" \
    "out_dir=$OUTPUT"
```

The harness requires the checkpoint file to exist and records its SHA-256;
never reuse one epoch's output directory for another checkpoint.

With no stage override, this runs `prepare`, `sample`, `score_qm9`,
`score_mp20`, and `report` in that order. Generation is GPU-heavy and the
comprehensive LeMat score is a separate, long-running multi-MLIP workload. For
batch scheduling, the staged form is usually more practical.

## Optional CHGNet pre-relaxation

Install the isolated [CHGNet environment](../../zevals/docs/CHGNet-prerelaxation.md)
once, then opt in when scoring materials:

```bash
bash models/zevals/scripts/setup_chgnet_prerelax.sh /absolute/path/to/chgnet-env
export ZEVALS_CHGNET_PYTHON=/absolute/path/to/chgnet-env/bin/python

table2 \
    "ckpt_path=$CHECKPOINT" "out_dir=$OUTPUT" \
    evaluation.mp20.prerelax.enabled=true \
    evaluation.resume=true
```

This uses FlowMM's CHGNet 0.3.1/model 0.3.0, FIRE, historical `ExpCellFilter`,
cell relaxation, `fmax=0.1` eV/Å and at most 1,500 steps. The wrapper leaves
`mp20/samples/` untouched and passes `mp20/chgnet_prerelax/cifs/` to all LeMat
families. Each CIF and completion record is committed atomically. Parse or
scientific relaxation failures retain the original bytes with an explicit
fallback status; unconverged finite endpoints are retained and marked. No
attempt is removed from the 2,500-attempt denominator. Infrastructure errors
abort the invocation instead of silently becoming scientific failures.

The JSON and CSV distinguish `unrelaxed` from `chgnet-flowmm`; use separate
output directories for these variants, and keep the same opt-in setting for
scoring and reporting. The CHGNet manifest binds source/output bytes, software,
checkpoint and configuration, and records convergence, steps and energies.
This is a separate preprocessing step from LeMat's own MLIP relaxations.

## Resume evaluation

Run the staged CLI in your GPU allocation. No cluster-specific launcher is required.

`evaluation.resume=true` verifies and reuses completed stages. Sampling commits one
eight-replica group at a time; CHGNet commits each CIF; LeMat commits each
individual ORB/MACE/UMA relaxation and each completed full-set benchmark family.
Novelty additionally commits each structure's reference comparison, and
uniqueness/SUN commit comparison rows while preserving the full-set algorithm.
An interrupted sampling group, CIF, novelty query or active comparison row is
recomputed. Other currently running families (such as distribution) restart
from their family boundary.
Global distribution, novelty and uniqueness are always evaluated on the full
generated set, never averaged from chunk-level scores. Interrupted adapter
artifacts remain under `mp20/lematgenbench/resume_state/attempts/`. Changed checkpoint,
inputs, numerical configuration, software or GPU type are rejected on resume.

## Run individual stages

Use identical checkpoint, output, data, seed, and protocol paths in every
stage. The first stage scans and hash-stamps the complete training partitions;
it may take several minutes but does not load the checkpoint.

```bash
table2 \
    "ckpt_path=$CHECKPOINT" "out_dir=$OUTPUT" \
    evaluation.stages=prepare

table2 \
    "ckpt_path=$CHECKPOINT" "out_dir=$OUTPUT" \
    evaluation.stages=sample

table2 \
    "ckpt_path=$CHECKPOINT" "out_dir=$OUTPUT" \
    evaluation.stages=score_qm9

table2 \
    "ckpt_path=$CHECKPOINT" "out_dir=$OUTPUT" \
    evaluation.stages=score_mp20

table2 \
    "ckpt_path=$CHECKPOINT" "out_dir=$OUTPUT" \
    evaluation.stages=report
```

The scoring stages consume the persisted CIF and metadata pairs. They do not
rerun diffusion. A second `sample` invocation requires `evaluation.resume=true`
to reuse verified progress. Do not combine `resume` and `overwrite`; use a new
`out_dir` for a changed sampling protocol.

If only the LeMat table projection changes and the samples and upstream
configuration are unchanged, the existing raw result can be reparsed without
rerunning the MLIPs:

```bash
table2 \
    "ckpt_path=$CHECKPOINT" "out_dir=$OUTPUT" \
    evaluation.stages=score_mp20 \
    evaluation.mp20.lematgenbench.reuse_raw_results=true
```

Reuse is accepted only when the saved structure hashes, sample manifest,
attempt/load/drop accounting, upstream commit and lock, config, family list,
fingerprint method, reference revisions, ORB/MACE/UMA identities and complete
coverage, CUDA policy, and every archived adapter artifact still match the
original run. Attempt-normalized rates are recomputed from the raw counts with
2,500 as the denominator even on a reuse pass.

### Rescore an existing EDM-only Table 2 run

Completed version-3 samples and protocol assets can be reused. The corrected
QM9 scorer has separate scoring provenance; it rejects the old five-metric
JSON instead of accepting it as a Zatom-1-compatible row. Using the `table2`
helper above and the original checkpoint, output, protocol path and material
pre-relaxation setting, run:

```bash
table2 "ckpt_path=$CHECKPOINT" "out_dir=$OUTPUT" \
    evaluation.stages=score_qm9 evaluation.resume=false

table2 "ckpt_path=$CHECKPOINT" "out_dir=$OUTPUT" \
    evaluation.stages=score_mp20 evaluation.resume=false \
    evaluation.mp20.lematgenbench.reuse_raw_results=true

table2 "ckpt_path=$CHECKPOINT" "out_dir=$OUTPUT" \
    evaluation.stages=report
```

The MP20 step refreshes its report provenance from the verified raw LeMat
result because the harness source and reporting environment changed; it does
not rerun the oracles or change the material metric definitions. It requires
a complete, reusable raw result. Sampling is not rerun. Incomplete sampling
still obeys the existing source/runtime checks and cannot be resumed across
an implementation or environment change.

## Reuse one frozen protocol across checkpoints

The protocol assets are checkpoint-independent. Preparing them once avoids
rescanning the same 100,000-molecule QM9 and 27,138-structure MP20 training
partitions for every epoch. Put them at a stable absolute path and use that path
for every checkpoint:

```bash
PROTOCOL="$(pwd)/logs/evaluation/table2-protocol-seed42"

table2 \
    "ckpt_path=$CHECKPOINT" "out_dir=$OUTPUT" \
    "evaluation.protocol_dir=$PROTOCOL" \
    evaluation.stages=prepare

NEXT_CHECKPOINT=/absolute/path/to/another/checkpoint.ckpt
NEXT_OUTPUT="$(pwd)/logs/evaluation/table2/another-checkpoint"

table2 \
    "ckpt_path=$NEXT_CHECKPOINT" "out_dir=$NEXT_OUTPUT" \
    "evaluation.protocol_dir=$PROTOCOL" \
    evaluation.stages=sample
```

Then run the three scoring/report commands for `NEXT_OUTPUT`, passing the same
`evaluation.protocol_dir` each time. The loader verifies every protocol
artifact against `protocol_manifest.json`. Reuse a protocol only for
checkpoints trained against the recorded dataset partitions and do not mutate
the protocol directory.

## Metric definitions and denominators

Both domains draw atom counts with replacement from the empirical atom-count
histogram of their exact training partition, using a dedicated CPU PyTorch
generator seeded with 42. QM9 samples are neutral singlets and retain explicit
hydrogens. The exact pre-drawn sequences are grouped by atom count because one
RFD3 input specification must use one size. Each forward pass produces eight
replicas. For the frozen data this means 1,261 QM9 specifications (10,088 raw
outputs, 88 discarded tail replicas) and 320 MP20 specifications (2,560 raw
outputs, 60 discarded tail replicas). The recorded plan restores original
sample-index order and persists only the requested 10,000/2,500 attempts; the
padding outputs never enter a metric denominator.

QM9 uses Zatom-1's geometry-to-molecule path: construct a pymatgen molecule
from the generated elements and Cartesian coordinates, perceive bonds with
OpenBabel, serialize to PDB, and parse with RDKit using `removeHs=False`.
This preserves the upstream PDB coordinate precision and explicit hydrogens.
Canonical isomeric SMILES use the largest fragment. PoseBusters receives the
**full valid molecule**, including every fragment, with its pinned `mol`
configuration. Generated coordinates are not relaxed before evaluation.

| Table metric | Numerator | Denominator |
| --- | --- | --- |
| Valid | Attempts with a sanitizable canonical SMILES | 10,000 attempts |
| Unique | Distinct canonical SMILES | Valid attempts |
| Atoms connected | Valid molecules passing `all_atoms_connected` | Valid attempts |
| Bond angles | Valid molecules passing `bond_angles` | Valid attempts |
| Bond lengths | Valid molecules passing `bond_lengths` | Valid attempts |
| Aromatic ring flat | Valid molecules passing `aromatic_ring_flatness` | Valid attempts |
| Double bond flat | Valid molecules passing `double_bond_flatness` | Valid attempts |
| Internal energy | Valid molecules passing `internal_energy` | Valid attempts |
| No steric clash | Valid molecules passing `internal_steric_clash` | Valid attempts |

PoseBusters checks are averaged over valid attempts, including duplicates,
matching the released Zatom-1 evaluator. As in its native pandas means,
missing check results are excluded from that check's denominator. The JSON
records these missing counts and the actual denominators; a zero denominator
produces zero with an explicit `zero_denominator_metrics` entry. Missing
PoseBusters rows or columns are errors. Connectivity is a separate check;
a disconnected molecule can still pass largest-fragment SMILES validity.
Atom/molecule stability and novelty are not Table 2 molecule columns.
The legacy EDM scorer remains available separately. Version-3 preparation
still preserves its historical training-SMILES artifact so existing sampling
manifests remain usable, but the corrected QM9 score does not use that reference.
Preparation requires the recorded source, selected-index, and per-shard SHA-256
values, so another same-sized partition cannot silently replace the checkpoint's
data.

MP20 is evaluated by the pinned LeMat-GenBench
`comprehensive_multi_mlip_hull` configuration. To match the Zatom-1 table
convention, all rate columns in the compact Table 2 projection divide counts by
the original 2,500 generation attempts, not by LeMat's validity-filtered
subset:

| Table metric | Table 2 value |
| --- | --- |
| Overall valid | Overall-valid count / 2,500 |
| Unique | Unique count / 2,500 |
| Novel | Novel count / 2,500 |
| Stable | Count with ensemble energy above hull at or below 0 eV/atom / 2,500 |
| SUN | Stable, unique, novel count / 2,500 |
| MetaSUN | Unique, novel count with ensemble energy above hull in `(0, 0.1]` eV/atom (`msun`) / 2,500 |

The native LeMat ratios, including validity-conditioned downstream ratios, are
also retained in `mp20/metrics.json`; do not substitute them for the
attempt-normalized manuscript columns.

The harness gives LeMat the persisted CIF directory rather than reparsing it in
the Nucleus process. If upstream cannot parse a saved CIF, that artifact remains
one of the 2,500 attempts and contributes zero to the validity numerator;
attempted, loaded, and parser-drop counts are all recorded. The adapter stages
and hashes every source CIF, then rehashes the staged bytes after scoring.

The manuscript path also requires the pinned LeMat-Bulk, multi-MLIP hull, and
UMA Hugging Face revisions; exact ORB, `medium-mpa-0`, and `uma-s-1` checkpoint
digests; and complete ORB/MACE/UMA energy-above-hull coverage for every valid
structure. Unversioned local hull overrides and a MACE fallback to `small` are
hard failures.

At the pinned revision LeMat emits three distribution components
(`JSDistance`, `MMD`, and `FrechetDistance`) and four diversity components
(`element_diversity`, `space_group_diversity`, `site_number_diversity`, and
`physical_size_diversity`). It defines no scalar aggregate for either family.
The checked-in harness explicitly projects `distribution/JSDistance` and
`diversity/element_diversity` into the manuscript's singular columns while
preserving every component in `mp20/metrics.json`. This is a local manuscript
choice, not an upstream aggregate; freeze the same two keys for every compared
method or override them explicitly before any table is populated.

## Output layout

The important artifacts are:

```text
OUTPUT/
├── protocol/
│   ├── protocol_manifest.json
│   ├── qm9_inputs.json
│   ├── mp20_inputs.json
│   └── *_train_*.json.gz
├── resolved_config.yaml
├── run_manifest.json
├── qm9/
│   ├── samples/
│   ├── sample_manifest.json
│   └── metrics.json
├── mp20/
│   ├── samples/
│   ├── sample_manifest.json
│   ├── metrics.json
│   └── lematgenbench/
│       ├── lematgenbench_raw.json
│       ├── lematgenbench_metrics.json
│       ├── lematgenbench_manifest.json
│       ├── lematgenbench.uv.lock
│       ├── cif_paths.txt
│       ├── input_cifs/
│       ├── lematgenbench_runtime.json
│       ├── lematgenbench_reference_pin_audit.jsonl
│       ├── lematgenbench_reference_pin_shim/sitecustomize.py
│       └── lematgenbench*.log
├── table2_metrics.json
└── table2_metrics.csv
```

If `evaluation.protocol_dir` points elsewhere, `OUTPUT/protocol` is absent and
`run_manifest.json` records the external manifest path and hash. The CSV is the
compact manuscript row; use the JSON files to audit counts, denominators,
component metrics, sample hashes, checkpoint hash and metadata, Git state,
software versions, CUDA devices, command line, and LeMat provenance.
Each scoring stage also hashes its exact local implementation modules and
records relevant package versions. The report recomputes those fingerprints and
requires an explicit rescore after scorer code or its environment changes.

## Reproducibility boundaries

Protocol version 3 requires one node and one device (`num_nodes=1`,
`devices_per_node=1`). After model initialization, each sampling group receives
its own seed: the first four bytes of
`SHA256("zatom2-table2-v3:42:<domain>:<group_id>")`, interpreted as an unsigned
big-endian integer. The atom-count stream remains seed 42. Fresh and resumed
runs therefore use the same diffusion stream for each group, independent of
where allocation boundaries fall. Input, checkpoint, editable source and
runtime fingerprints protect progress reuse. CUDA kernel and hardware changes
can still affect bitwise outputs; cross-hardware equivalence is not claimed.

Version 2 used one continuous diffusion RNG stream per domain. Its artifacts
cannot be mixed with version 3. Start new protocol/output directories when
upgrading, and evaluate all compared checkpoints with the same protocol version.

LeMat scoring has a separate frozen `runtime_device=cuda` policy. The adapter
removes inherited Python import paths, verifies that CUDA is visible in its
isolated environment, and archives the interpreter, Torch/CUDA/cuDNN versions,
GPU identity, visibility mask, determinism flags, and TF32 settings. Keep a
consistent GPU/runtime when comparing checkpoints: the audit makes numerical
differences attributable, but does not claim bitwise equivalence across GPU
models.

Do not override the checked-in sampler when comparing checkpoints: EMA weights,
AF3 solver, 100 time steps, `sigma_data=16`, `gamma_0=0.8`, `gamma_min=1.0`,
`noise_scale=1.003`, `step_scale=1.5`, `p=7`, `s_min=4e-4`, `s_max=160`, two
recycles, no motif-fixing steps, and classifier-free guidance disabled. The
entire sampler and input/transform override mappings are frozen, so added Hydra
keys are noncanonical as well. The generic design command has different
defaults. The harness rejects changes to these values, sample counts, single-GPU
topology, or checkpoint hashing. Set
`evaluation.allow_noncanonical_protocol=true` only for an explicitly diagnostic
smoke run; such output is marked noncanonical and must not populate Table 2.

Finally, the pinned LeMat version calls MACE without an explicit model argument.
The adapter forces and hashes `medium-mpa-0` at that call site and rejects the
upstream fallback that can select `small`; however, equivalence to the
undocumented MACE checkpoint behind the published hull remains unresolved. This
is an [open upstream reproducibility issue](https://github.com/LeMaterial/lemat-genbench/issues/87).
Archive the adapter lockfile and logs, and do not compare results produced by
different dependency locks or fallback paths.
