# LeMat-GenBench evaluation

`zevals.materials.lematgenbench` runs the official LeMat-GenBench CIF pipeline
from an isolated, revision-checked checkout and converts its output into stable
`family/metric` keys. It does not reimplement the benchmark.

## Frozen upstream protocol

The adapter supports
[`LeMaterial/lemat-genbench`](https://github.com/LeMaterial/lemat-genbench) at
commit `58e6eae3e4a6c87c22171cf069123ecc4e2fa7e6` (the tip of `main` when this
integration was frozen on 2026-09-04). Upstream has no tags or GitHub releases;
its `pyproject.toml` reports `0.1.0` while `lemat_genbench.__version__` reports
`0.2.0`, so a package version is not a reproducible identifier.

The supported entry point is the repository's canonical CIF runner:

```text
scripts/run_benchmarks.py --cifs ... \
  --config comprehensive_multi_mlip_hull \
  --name ...
```

Do not substitute the installed `lemat-genbench` console command. At the pinned
revision it has a different input/API path and is not the runner documented for
leaderboard-compatible CIF evaluation.

LeMat-GenBench brings its own PyTorch, PyG, MACE, ORB, and UMA dependency stack.
It must not be installed into a Nucleus training environment. This is why it is
not a `zevals` package extra: the integration boundary is a subprocess and CIF
files.

## One-time environment setup

After preparing the environment, use the
[manual checkpoint download guide](../../zatom2/docs/table2_checkpoints.md) to
cache the pinned ORB, MACE, and UMA weights before starting evaluation.

From `models/zevals`, provide a new absolute destination:

```bash
bash scripts/setup_lematgenbench.sh /path/to/lemat-genbench
export ZEVALS_LEMAT_GENBENCH_REPO=/path/to/lemat-genbench
```

The setup helper:

1. clones the official repository without Git LFS smudging;
2. checks out the supported commit in detached-HEAD mode;
3. verifies the committed `uv.lock` SHA-256; and
4. creates the Python 3.11 environment with `uv sync --frozen`.

Normal clones can currently fail because the upstream Git LFS quota is
exhausted ([upstream issue #82](https://github.com/LeMaterial/lemat-genbench/issues/82)).
The `comprehensive_multi_mlip_hull` configuration uses `structure-matcher`, not
the affected augmented-BAWL parquet files, so skipping their LFS smudge is the
appropriate workaround for this protocol.

Stability and distribution evaluation require access to the upstream MLIP
weights and reference data. In particular, request access to
[`facebook/UMA`](https://huggingface.co/facebook/UMA) and authenticate the
account used by the evaluation job before launching a full run. Leave
`data/convex_hulls/` absent. The pinned runner fetches missing hull data through
Hugging Face into the datasets cache or `data/.cache`; it does not materialize
downloads in `data/convex_hulls/`. In revision-enforced mode the adapter rejects
files placed there because no audited checksum manifest exists for those local
overrides. The upstream `download_above_hull_datasets.py` helper is not suitable
for this configuration: at the supported commit it downloads legacy Entalpic
PKLs, not the six ORB/MACE/UMA parquet/NPZ inputs used here.

The pinned upstream revision commits `uv.lock`; do not run `uv lock`, which
would mutate the protocol and make the checkout fail the clean-worktree check.
Its expected SHA-256 is
`74970323f6697d50e60251de9182e7a559031bce67ab2f693986742e7c286850`.
Every adapter run uses `uv run --frozen`, verifies that digest, and archives a
copy of the lockfile, so dependency drift cannot occur silently.

The distribution and novelty references below are ordinary Git blobs at the
pinned commit—not Git LFS objects. Revision-enforced runs require these exact
SHA-256 values before launching, and the manifest repeats the observed values:

| Repository path | SHA-256 |
|---|---|
| `data/lematbulk_jsdistance_distributions.json` | `9603f06e5f86d11877e81735e89de3e39f2f6c5c0395d61d69cd20ea88108748` |
| `data/lematbulk_mmd_values_15k.pkl` | `1361fa4c1e7ac402cac16f9428ab77e32c753262a771374eda7e166cb9f71fd0` |
| `data/orb_mu.npy` | `72c6c25763e91065d79068b1018a624ed0350c97963fef58e02490da226d1b7a` |
| `data/orb_sigma.npy` | `f6625a5548218992875db07a55093a7a1de5d415e296cd2885097dbfd7eaca80` |
| `data/mace_mu.npy` | `a39c081db7c8db67bce2a23c28add8edc7dccbcf9c7bc80230fecddca3f1befd` |
| `data/mace_sigma.npy` | `644801230ae4bf86982e57394fddbd82414cd5db23ac644040722c3008644aea` |
| `data/uma_mu.npy` | `1685dad90fa2a5c670731ee7e07e63a926ae595745f50e01b337b030427ea28a` |
| `data/uma_sigma.npy` | `adec18113f0e44b06fed966387d2800f9d49464318fdb745534efdab4eba431b` |
| `data/all_compositions.npz` | `28b362f9a558e25d6ba3b846060b56da2ec5caada299bc0737e7b7256325f38f` |

For MLIP-dependent families, the adapter also resolves `medium-mpa-0`
explicitly in that environment before benchmarking and requires checkpoint
SHA-256
`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`.
This catches model-cache drift while retaining the canonical clean upstream
runner. Development runs can set `verify_mace_checkpoint=False`; manuscript
runs should not.

The other two oracle identities are enforced at first use in the benchmark
interpreter:

- ORB `orb_v3_conservative_inf_omat` from the URL embedded in locked
  `orb-models==0.5.4`, SHA-256
  `0a41ef1132ad9c41ee0c7b9d855fccabfefb97abe04625c97ec0d07930b6c0f0`;
- UMA `uma-s-1`, task `omat`, from `facebook/UMA` revision
  `f611b917d9c68566bbbeccbb0aa0f7cad1696cb2`, SHA-256
  `02f83e20547294d3d47dd9dfbe4a4ee51eebca21b2e1febe63210e55b764cc78`.

To do this without modifying the upstream checkout, `zevals` generates an
archived `sitecustomize.py` and prepends its directory to `PYTHONPATH` only for
the subprocess. It forces audited revisions on `datasets.load_dataset`,
`huggingface_hub.hf_hub_download`, and
`huggingface_hub.snapshot_download`; it also hashes the ORB file returned by
`cached_path`, the UMA checkpoint returned by the Hub, and the MACE checkpoint
used by a wrapped `mace.calculators.mace_mp`. That wrapper forces
`medium-mpa-0`, so the benchmark-time invocation is tied to the preflighted
file. The startup preflight and role-labelled post-run JSONL audit prove that
the benchmark interpreter loaded every hook. Inherited `PYTHONPATH` and
`PYTHONHOME` are removed rather than silently extending the locked environment.
Revision enforcement is on by default. Setting
`enforce_reference_revisions=False` is an explicit diagnostic opt-out and must
not be used for a manuscript result.

## Running the adapter

The input can be a persisted CIF directory (preferred), a single CIF, a text
file listing CIFs, or a list of `pymatgen.Structure` objects:

```python
from pathlib import Path

from zevals.materials.lematgenbench import (
    LEMAT_GENBENCH_TABLE2_FAMILIES,
    run_lematgenbench,
)

result = run_lematgenbench(
    Path("outputs/checkpoint-0100/mp20"),
    repo_path=Path("/path/to/lemat-genbench"),
    run_name="epoch-0100-seed-0",
    output_dir=Path("scores/epoch-0100/lemat-genbench"),
    families=LEMAT_GENBENCH_TABLE2_FAMILIES,
    runtime_device="cuda",
    # Optional: freeze a scheduler-approved visibility mask explicitly.
    cuda_visible_devices=None,
    parse_failures_as_invalid=True,
)

print(result.metrics)
print(result.raw_result_path)
```

Use a fresh `output_dir` for each evaluation, or explicitly enable `resume=True`
to continue the same evaluation. The adapter otherwise refuses to overwrite
existing results. It verifies the exact upstream commit, committed lockfile,
tracked files, and untracked code/config before running, while allowing known
cache/result locations. Opting out with
`expected_commit=None`, `expected_lock_sha256=None`, or
`require_clean_repo=False` is available for development, but such results
should not be placed in a manuscript table.

Regardless of whether the source is a directory or an existing list, the
adapter copies exact bytes into `input_cifs/` and passes upstream a sorted,
deduplicated, one-absolute-staged-path-per-line manifest. Directory discovery
includes both `.cif` and `.cif.gz`, and a caller-provided text manifest can list
either extension. Pre- and post-run hashes detect mutation while upstream is
reading those self-contained staged files. The adapter compares
`validity_filtering.total_input_structures` with the number of staged paths.

AtomWorks/PDBx mmCIF inputs are normalized in memory at the isolated runner's
materials-parser boundary. The adapter maps equivalent cell, symmetry and
atom-site tags, retaining fractional coordinates, species and occupancies;
it does not rewrite source or staged CIFs. The conversion helper is recorded
in the archived runner command and resume contract. Ordinary materials CIFs
continue through the unmodified parser. This applies with and without CHGNet
pre-relaxation, including original-byte fallbacks from unsuccessful relaxations.

For legacy Table 2 runs whose CHGNet stage reported every mmCIF as a parse
failure, see [the targeted recovery procedure](CHGNet-prerelaxation.md#recovering-the-legacy-table-2-mmcif-parse-failures)
before resuming. Old conversion results and downstream caches must not be mixed
with the corrected parser.

Strict mode (`parse_failures_as_invalid=False`, the adapter default) raises on
any mismatch after preserving the raw result and diagnostics. For checkpoint
evaluation, set `parse_failures_as_invalid=True`: paths that the permissive
upstream parser drops remain invalid generation attempts in the staged
denominator. The native upstream metrics still retain their loaded-structure
denominators, while `attempt_normalized/...` rates use every staged artifact.
`input_accounting/cif_parse_failure_count` and both counts are prominent in the
metrics and manifest. This permits a valid manuscript row even when a saved CIF
is malformed, without silently rewarding that failure.

The pinned runner ignores the CPU values in its YAML and selects CUDA whenever
`torch.cuda.is_available()`. The adapter therefore requires an explicit
`runtime_device` policy (`"cuda"` by default; pass `"cpu"` to hide CUDA).
`cuda_visible_devices=None` preserves the scheduler-provided mask, while an
explicit string overrides it only in child processes. A preflight archives the
isolated Python and Torch versions, CUDA/cuDNN builds, effective visibility,
device names/capabilities/memory, and determinism/TF32 flags. These records make
hardware drift visible; they do not claim bitwise identity across GPU models.

The output directory contains:

| Artifact | Contents |
|---|---|
| `lematgenbench_raw.json` | Byte-for-byte copy of the complete upstream result |
| `lematgenbench_metrics.json` | Sorted numeric `family/metric` map plus explicitly named `attempt_normalized/...` derived rates |
| `lematgenbench_manifest.json` | Commit, config, families, exact command, lock/config hashes, and SHA-256 for every input CIF |
| `lematgenbench.stdout.log` | Complete upstream standard output |
| `lematgenbench.stderr.log` | Complete upstream standard error |
| `lematgenbench.uv.lock` | Copy of the exact upstream dependency lockfile |
| `cif_paths.txt` | Sorted absolute paths actually supplied to upstream |
| `input_cifs/` | Exact staged `.cif`/`.cif.gz` bytes evaluated upstream |
| `lematgenbench_runtime.json` | Isolated Python/Torch/CUDA/GPU provenance and selected device policy |
| `lematgenbench_runtime_preflight.{stdout,stderr}.log` | Runtime-probe output and diagnostics |
| `lematgenbench_mace_preflight.stdout.log` | MACE checkpoint resolution and identity log (MLIP families only) |
| `lematgenbench_mace_preflight.stderr.log` | MACE checkpoint preflight diagnostics (MLIP families only) |
| `lematgenbench_reference_pin_shim/sitecustomize.py` | Exact generated revision/checkpoint enforcement shim; its SHA-256 is in the manifest |
| `lematgenbench_reference_pin_preflight.{stdout,stderr}.log` | Proof and diagnostics for Python startup-hook injection |
| `lematgenbench_reference_pin_audit.jsonl` | Role-labelled shim loads, forced repository revisions/models, resolved paths, and ORB/MACE/UMA checkpoint digests |

`result.provenance["artifact_sha256"]` hash-binds the manifest, raw result,
metrics, archived lock, CIF manifest, audit, shim, and every produced preflight
or benchmark log. The manifest also inventories local reference files both
before and after the run.

The raw file matters because the pinned upstream runner currently serializes
each `BenchmarkResult` dataclass as a string inside JSON. The adapter's parser
uses Python's AST and `literal_eval` only on the `final_scores` node; it never
executes the string. It also accepts a future structured `final_scores` object.
The upstream script catches family failures and can still exit with status zero;
the adapter treats those error objects (and unreadable family schemas) as a
failed run by default, after preserving the diagnostics. Set
`fail_on_family_error=False` only when intentionally collecting partial results;
`result.family_errors` and the manifest retain the failures.

The canonical upstream configuration allows ensemble statistics with only two
of three MLIPs (`min_mlips_required: 2`, `require_all_mlips: false`). For a
manuscript run, the adapter's default
`require_complete_oracle_coverage=True` additionally checks the stability
family's public `n_valid_structures_{orb,mace,uma}` final scores against the
valid-structure count. Missing or incomplete all-three coverage is recorded and
fails the run. If the stability family was not selected, upstream exposes no
equivalent per-model coverage counters; the manifest reports
`not_reported_without_stability_family` rather than inferring coverage from
logs. Model-load markers and checkpoint identity are recorded separately from
per-structure coverage.

The same options are exposed in `configs/materials.yaml`. Set
`run_lematgenbench=true`, `ZEVALS_LEMAT_GENBENCH_REPO`, and a unique
`ZEVALS_LEMAT_GENBENCH_OUTPUT_DIR` to invoke it through
`MaterialEvalPipeline`. The flattened keys are merged into the aggregate result
and artifact paths/digests are recorded under
`EvalResults.metadata["lematgenbench"]`. Protocol failures propagate by default
(`lematgenbench_strict=true`); disable that only when partial generic pipeline
results are intentionally acceptable.

## Resuming across allocation time limits

Set `resume=True` on the first invocation and subsequent invocations, keep the
same persistent `output_dir` and input CIFs, and provide the time remaining in
the current allocation as `timeout_seconds`. Through `MaterialEvalPipeline`,
the corresponding options are `lematgenbench_resume` and
`lematgenbench_timeout_seconds`. Resume requires reference-revision enforcement.
The timeout covers preflights and the benchmark together. Standard output and
error stream to disk while the child runs; timeouts and scheduler termination
stop the entire child process group, including upstream workers. A timeout raises
`LeMatGenBenchTimeout` (`TimeoutError`); Ctrl-C remains cancellation.

The archived startup hook leaves the pinned upstream checkout unchanged. It
caches each completed ORB/MACE/UMA calculation independently for each structure,
each per-structure novelty comparison against its pinned reference, and rows
of the structure-matcher comparisons used by uniqueness and SUN. Novelty work
is also reused inside SUN. The uniqueness hook preserves upstream's comparison
order and representative convention; it stores at most one row record per
structure for each ordered input set, rather than millions of pair files.
It then caches each completed benchmark evaluation over the **entire** valid input
set. Distribution, uniqueness, novelty, SUN, and other global metrics are always
computed over that full set. Chunk-level metric averages are never used. A
calculation interrupted before its atomic commit is retried, as are oracle
results with missing requested outputs. A family interrupted before completion
runs again using the committed oracle, novelty, and comparison-row work. A hard
kill loses at most the active comparison row; full-set distribution calculations
are still committed only when that family finishes.

The cache lives in `resume_state/cache/`. Its immutable request and runtime
contracts bind generated CIF bytes, upstream revision and lock, oracle and
reference identities, configuration, adapter/hook source, Python/Torch builds,
GPU model/memory, and numerical settings. A newly allocated GPU UUID and
visibility mask may differ; changes to the scientific or numerical contract
require a new output directory. Upstream's non-scientific
`properties.validity_timestamp` is excluded from cache identity, including
copies attached to relaxed structures. Every other structure property remains
part of the cache key.

Cache payloads are local Python objects with verified SHA-256 records, atomic
replacement, file/directory synchronization, and per-key locks. Treat
`resume_state/` as local generated state; do not substitute cache directories
from untrusted sources. One writer holds an exclusive output-directory lock.
Prior logs, staged inputs, manifests, and shims are preserved under
`resume_state/attempts/`; their verified reference-access audits carry forward
to document the origin of reused work. The final manifest records that history
and the copied `_lemat_resume_runtime.py` hash. Once a complete adapter result
has passed all checks, `resume_state/completed.json` allows the enclosing Table 2
workflow to recover even if the allocation ends before it writes its own report.
Reuse rechecks the original artifacts and staged CIF hashes.

## Mapping to the Zatom-2 Table 2 material columns

Use `select_table2_metrics` after deciding which upstream distribution and
diversity component the manuscript's singular columns denote:

```python
from zevals.materials.lematgenbench import select_table2_metrics

table2 = select_table2_metrics(
    result.metrics,
    distribution_key="distribution/FrechetDistance",
    diversity_key="diversity/space_group_diversity",
)
```

There is deliberately no default for those two choices. At the pinned revision,
LeMat-GenBench emits three distribution scalars (`JSDistance`, `MMD`, and
`FrechetDistance`) and four diversity scalars (`element_diversity`,
`space_group_diversity`, `site_number_diversity`, and
`physical_size_diversity`). It does not define a single benchmark-level scalar
for either family. Inventing an average in `zevals` would no longer be the
upstream metric. Freeze and report the chosen keys before filling the table.

The selector returns fractions for:

| Table column | Upstream source |
|---|---|
| `overall_valid` | `validity/overall_validity_count / N_input` |
| `distribution` | caller-selected distribution key |
| `diversity` | caller-selected diversity key |
| `unique` | `uniqueness/unique_structures_count / N_input` |
| `novel` | `novelty/novel_structures_count / N_input` |
| `stable` | `stability/stable_count / N_input` |
| `sun` | `sun/sun_count / N_input` |
| `metasun` | `sun/msun_count / N_input` |

The original input count is essential. The latest runner first filters invalid
structures and then passes only the valid subset to downstream families, so its
family-local ratio fields can have a different denominator. Published
LeMat-GenBench/Zatom-1 table rates use counts divided by the original sample
count (2,500 in that evaluation); the selector reproduces that convention.
Both representations remain available: for example,
`uniqueness/uniqueness_ratio` is the unmodified backend-native ratio, while
`attempt_normalized/unique_rate` is the
`uniqueness/unique_structures_count` value divided by
`validity_filtering/total_input_structures`. The adapter
also derives `attempt_normalized/metastable_rate` even though Zatom-2 Table 2
only requests MetaSUN, so the complete published protocol remains auditable.

## Reproducibility caveats that must be reported

- Use exactly 2,500 MP20 samples if matching the Zatom-1 evaluation. Record the
  generation seed and hash-stamped CIF manifest produced by this adapter.
- Record the LeMat-Bulk reference configuration, relaxation settings, MLIPs,
  and energy-above-hull thresholds from
  `comprehensive_multi_mlip_hull.yaml` in the pinned checkout. Upstream omits
  `revision=`, so the adapter's audited startup shim supplies and verifies
  these exact revisions: LeMat-Bulk
  `0dc17eea904b860ad7288141e9870f67f8e6bb2c`, MLIP hulls
  `70d505bb294c16658a9551e2b81ace69a28d9790`, and embeddings
  `0d5895780c53fa54347ffc076cc72c7748e692e4`. The canonical runtime does not
  fetch the embeddings repository; it uses the committed and hash-inventoried
  `{orb,mace,uma}_{mu,sigma}.npy` files. The manifest explains that absence
  rather than falsely claiming a runtime pin event.
- Upstream calls `mace.calculators.mace_mp()` without an explicit model
  argument. The adapter's runtime hook forces `medium-mpa-0`, verifies its
  digest at the actual benchmark-time call, and rehashes it after the run.
  Whether that checkpoint matches the
  published MACE-specific hull is an unresolved question
  ([upstream issue #87](https://github.com/LeMaterial/lemat-genbench/issues/87)).
  The adapter pins and hashes that current effective model, but cannot establish
  its equivalence to the hull's undocumented checkpoint. It rejects the
  upstream alternative loading path, which silently changes the model to
  `small`.
- Upstream may download weights and reference datasets on first use. Warm and
  freeze caches before batch evaluation, especially on compute nodes without
  outbound network access.
- The full configuration is compute- and model-heavy. Run it as a separate
  score stage, not inside a trainer process or a multi-GPU training launch.
