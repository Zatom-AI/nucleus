# Zero-shot coordinate relaxation probe

This experiment asks whether the frozen Table 3 Zatom-2-L joint-properties
checkpoint can improve an existing geometry in one denoiser pass, while retaining
every atom and its element. No fine-tuning, force-gradient guidance, candidate
ranking, or best-of-k selection is used.

## Results (2026-09-17)

**Partial success: one pass can reduce energy and force, but it does not replace
MLIP relaxation.** All 2,496 input/output geometries were scored successfully.
None of the 2,304 generated candidates, including 16-pass controls, reached a
maximum atomic force below 0.05 eV/Å.

The primary finite target is 0.05 eV/Å. Ratios below are medians of paired
output/input mean-force norms, with 95% bootstrap intervals over the 32 source
structures. Energy changes are medians in eV/atom.

| Input / target | Force ratio [95% interval] | ΔE/atom | Both improve |
| --- | ---: | ---: | ---: |
| Molecules, 0.1 Å perturbation, no force target | 0.173 [0.157, 0.201] | −0.431 | 32/32 |
| Molecules, 0.1 Å perturbation, target 0.05 | 0.164 [0.147, 0.184] | −0.439 | 32/32 |
| Molecules, native geometry, target 0.05 | 0.859 [0.711, 1.291] | −0.035 | 18/32 |
| Materials, native rattled geometry, no force target | 0.454 [0.439, 0.567] | −0.276 | 31/32 |
| Materials, native rattled geometry, target 0.05 | 0.374 [0.339, 0.484] | −0.318 | 31/32 |
| Materials, additional 0.1 Å perturbation, target 0.05 | 0.367 [0.289, 0.433] | −0.368 | 31/32 |

The molecular conditioning advantage over unconditioned denoising is uncertain:
the paired conditioned/unconditioned force ratio is 0.898 [0.864, 1.007]. For
native rattled materials it is 0.795 [0.718, 0.890], with improvement in 27/32
cases. Literal zero conditioning is substantially worse than the finite target.
For native molecules, a zero target raises median energy by 0.025 eV/atom and
has median force ratio 1.407.

Primary output mean-force norms remain around 1.12 eV/Å in both domains. All
32 primary molecular outputs pass the source-bond distance diagnostic, which
does not establish complete chemical or stereochemical validity. The actual
confirmation atom-count ranges are 7–117 for molecules and 10–20 for materials.
Composition is preserved by fixed identity conditioning and retaining source
elements when constructing the output geometry.

For the same 0.1 Å inputs, UMA/BFGS reaches the maximum-force threshold in
**30/32 molecules and 29/32 materials**, versus 0/32 per domain for one-pass
refinement. Across all attempts, median iteration counts are 97 and 117.5;
median final mean-force norms are 0.024 and 0.027 eV/Å. The two molecular and
three material failures hit the unchanged 200-step cap and remain in these
statistics.

The [complete result grid and provenance](manuscript/zero_shot_refinement_results.json)
include the full conditioning/noise grid, bootstrap intervals, source identities,
checkpoint hashes, environment versions, and BFGS summary. The
[manuscript](manuscript/arxiv_preprint.tex) contains the qualified result and
control table. Targeted numerical/aggregation tests pass; the new LaTeX
environments, references, and citation were checked, but no LaTeX compiler was
available for a full build.

## One-step UMA baseline (2026-09-24)

One fresh UMA-S-1p2/BFGS update was evaluated on the same 64 inputs used in the
manuscript's one-pass comparison. All attempts succeeded and reduced both energy
and mean force; none reached maximum force below 0.05 eV/Å.

| Input | UMA force ratio [95% interval] | UMA output mean force (eV/Å) | Zatom-2 output mean force, target 0.05 (eV/Å) |
| --- | ---: | ---: | ---: |
| Molecules, 0.1 Å perturbation | 0.764 [0.745, 0.771] | 5.506 | 1.116 |
| Materials, native rattled geometry | 0.853 [0.830, 0.873] | 2.253 | 1.117 |

The output columns are medians across structures. This untuned first BFGS step
reduces force less than Zatom-2's single denoiser pass under the specified
settings; it does not measure the best achievable UMA step size or matched
compute. Its median energy changes are −0.118 and −0.065 eV/atom, respectively.
The checkpoint hash and package versions match the original oracle. Recomputed
input mean forces agree within 1.91e-6 eV/Å and energies within 3.57e-6 eV.
The complete result artifact includes the baseline protocol, summary, per-sample
metrics, and raw-file hashes.

## Reproduction

Install the checkpoint from the repository environment:

```bash
DEBUG=false .venv/bin/nucleus install zatom2-table1-zatom2-l-joint-properties-ablation
```

The checkpoint SHA256 is
`ebaa4afa55b7ab4d51df819ad45a95182cf03435a05460a37d1e64242c79b386`.
The experiment uses EMA weights and one denoiser recycle, with the checkpoint's
bfloat16 mixed precision. Dataset paths come from `.env`. UMA-S-1p2 and
`fairchem-core` must be available in the evaluation environment.

On one allocated A100 80 GB, run these commands in order:

```bash
export DEBUG=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
.venv/bin/python scripts/probe_zero_shot_relaxation.py generate \
  --count 32 --seed 20260918 --output logs/zero_shot_relaxation/confirmation
.venv/bin/python scripts/probe_zero_shot_relaxation.py score \
  --output logs/zero_shot_relaxation/confirmation
.venv/bin/python scripts/probe_zero_shot_relaxation.py summarize \
  --output logs/zero_shot_relaxation/confirmation
.venv/bin/python scripts/probe_zero_shot_relaxation.py relax \
  --output logs/zero_shot_relaxation/confirmation
.venv/bin/python scripts/probe_zero_shot_relaxation.py one-step-uma \
  --output logs/zero_shot_relaxation/confirmation
```

The generation command refuses to overwrite existing structures. Oracle scoring
and the BFGS reference can resume completed records. Raw structures, per-candidate
scores, the manifest, and BFGS endpoints remain in the output directory.
The one-step baseline writes separate `uma_one_step_baseline.jsonl` and
`uma_one_step_summary.json` files and refuses to overwrite existing records.
Use `--uma-checkpoint /path/to/uma-s-1p2.pt` to pin a local UMA checkpoint.

## Protocol

- A separate pilot uses four structures per domain and seed 20260917. The
  confirmation uses 32 structures per domain and seed 20260918. Candidates are
  selected in seeded random order from OMol25 validation and OMat24 validation
  `rattled-300`. Only complete structures with 3–128 atoms, valid property labels,
  and a supported structure-prediction condition are eligible. Disconnected
  molecular graphs are excluded by the checkpoint's normal preprocessing.
- Before confirmation scoring, the pilot selects `sigma=0.1` Å for the molecular
  0.1 Å corruption comparison and `sigma=0.3` Å for materials, with the finite
  force target 0.05 eV/Å. The entire grid remains in the summary rather than
  selecting a different setting for each confirmation structure.
- Each structure is tested at its native coordinates and after centered Gaussian
  coordinate perturbations with per-component standard deviations of 0.1 and
  0.3 Å. The same perturbation direction is used across severities and conditions.
- The conditioning target is absent, 0, or 0.05 eV/Å. Literal zero is encoded as
  `log10(max(target, 1e-8))`, matching the existing input transform. The finite
  target checks whether a result depends on extrapolation to that extreme input.
- For denoiser noise parameter `sigma` in {0.1, 0.3, 1.0} Å, initialize the flow
  state at `t0 * X_input`, where `t0 = 16 / (16 + sigma)`. Use the predicted
  clean endpoint directly. This makes **one denoiser pass**, with no additional
  noise injection or force evaluation. The 16-pass control integrates the same
  flow from `sigma=0.3` using Euler steps and the physical velocity `16 * V_L`.
- Atom identities, molecular bonds used as conditioning, charge, and spin are
  retained. Material outputs use Cartesian coordinates and the source cell.
  This is a fixed-cell coordinate experiment, not cell relaxation or evaluation
  of the model's separate fractional-coordinate/lattice readouts.
- UMA-S-1p2 scores every input and output with the appropriate `omol` or `omat`
  head. Energies are paired within the same composition and electronic state.
  The model's own property predictions are never used for scoring or selection.
- Report the median per-structure output/input mean-force ratio, paired energy
  change per atom, counts improving both, and the number with maximum atomic
  force below 0.05 eV/Å. The force-ratio interval is a 10,000-resample percentile
  bootstrap over source structures, separately for each condition. It is not a
  confidence interval over atoms or repeated variants treated as independent.
- A source-bond distance check flags molecular bonds outside 0.65–1.5 times the
  sum of the two covalent radii. This is a geometry diagnostic, not a complete
  chemical-validity or stereochemistry test.
- For 0.1 Å inputs, a fixed-cell UMA/BFGS reference uses `maxstep=0.04` Å,
  `fmax=0.05` eV/Å, and at most 200 steps. Failed or unconverged attempts remain
  in the denominator. This comparison distinguishes coordinate improvement from
  replacing an optimization that reaches a stationary geometry.
- The one-step UMA baseline in Table 10 uses exactly its 32 molecular 0.1 Å
  inputs and 32 native rattled material inputs. A fresh ASE BFGS optimizer takes
  one update with `alpha=70` eV/Å² and `maxstep=0.04` Å, matching the iterative
  reference's initial Hessian and displacement cap. Thus the displacement is
  `F/70`, uniformly rescaled if the largest atomic displacement exceeds 0.04 Å.
  There is no optimizer history, line search, or step-size tuning. One initial
  UMA force evaluation determines the update; a separate endpoint evaluation
  measures its quality. This compares single structural updates, not equal
  FLOPs or wall time. UMA is both the baseline potential and the scoring oracle.

## Scope

Validation structures are held out by the source dataset split; this does not
establish molecule-identity or material-composition novelty. Gaussian corruption
is not a substitute for evaluating an arbitrary high-energy conformer. UMA is
an independent surrogate, not a new DFT calculation, and both models share
training-domain coverage. Lower mean force does not by itself establish a local
minimum, preserved stereochemistry, or chemical validity. Native molecular
geometries are an essential negative control for unnecessary structural damage.

The pilot results in `logs/zero_shot_relaxation/pilot_v2/` are not pooled into
confirmation statistics.
