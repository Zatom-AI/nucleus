# AFD — Atomistic Fréchet Distance: a user's guide

A practical guide to the `zevals.atomistic.afd` scoring tools: what AFD measures, when it is a
valid measure, how many samples you need, and how to read a score. The empirical
claims here are grounded in the validation suite under `re_val/` (see
[Evidence](#evidence-the-re_val-validation-suite)).

> This guide covers the tools in `zevals.atomistic.afd`. The CT model code it depends
> on lives in `zevals.atomistic.afd.models` (a minimal copy of the Conditional Equivariant
> Transformer); you do not call it directly to score a model.

---

## 1. What AFD is

AFD is **FID for atoms**. It is the exact same recipe as the Fréchet Inception
Distance used to evaluate image generators (Heusel et al. 2017), with one
substitution: the Inception-v3 feature extractor is replaced by a **Conditional
Equivariant Transformer (CT)** pretrained on atomistic data. So instead of scoring
generated *images* in an image-feature space, AFD scores generated *3D atomistic
structures* (molecules or periodic crystals) in a chemistry-aware feature space.

The procedure:

1. Embed every structure — reference and generated — through the pretrained CT,
   harvesting its 256-d graph-level embedding (`mol_emb`).
2. Fit a Gaussian (mean μ, covariance Σ) to each set of embeddings.
3. Report the closed-form Fréchet (Wasserstein-2) distance between the two
   Gaussians:

   ```
   AFD²  =  ‖μ_ref − μ_gen‖²  +  Tr( Σ_ref + Σ_gen − 2·(Σ_ref·Σ_gen)^½ )
   ```

**Lower is better.** AFD = 0 means the generated set is distributionally
indistinguishable from the reference *in the CT feature space*. A larger AFD means
the generated distribution has drifted from the reference — in geometry, in
composition, or in both.

The CT checkpoint **is part of the metric.** Two AFD numbers are only comparable
if they were produced with the same checkpoint against the same reference. There
are two canonical setups:

| Modality | Reference keyword | Checkpoint | Reference set |
|---|---|---|---|
| Molecules | `"qm9"` | `ct-scd-pcq` | 130,831 QM9 molecules |
| Materials (periodic crystals) | `"mp20"` | `ct-scd-amp20` | 45,229 mp20 crystals |

Reference features and model weights are fetched automatically from HuggingFace
(`Ty-Perez/AtomisticEval`) the first time you use a canonical reference — no local
embedding pass and no GPU cost for the reference side.

---

## 2. What AFD is for

Use AFD to **rank atomistic generative models** (or generation pipelines) by how
faithfully their output reproduces a known ground-truth distribution. It is a
single, sample-based, distributional score — it does not need per-sample labels,
matching, or a likelihood.

Good uses:

- **Comparing generative models** trained on the same dataset (e.g. ranking
  several QM9 molecule generators), provided you score them all the same way.
- **Assessing a generation pipeline's geometric fidelity** — AFD reads systematic
  geometry error, not just chemistry. In `re_val`, RDKit force-field conformers of
  *correct* molecules still score ≈ 0.19 (far above the floor) purely from their
  geometry being off relative to the DFT reference.
- **Ranking post-processing / relaxation steps.** AFD correctly ordered two
  relaxers: GFN2-xTB pulled conformers *toward* the DFT reference (≈ 0.106, an
  improvement on raw RDKit ≈ 0.19), while the cheaper GFN-FF pushed them *further
  away* (≈ 0.24, a regression). AFD registered both the win and the regression.

What AFD does **not** do — pair it with other checks:

- **It is blind to chirality.** A molecule and its mirror image embed identically
  (the CT embedding is reflection-invariant / E(3)). A chiral set and its full
  enantiomeric flip score the same (0.0060 vs 0.0061 in `re_val`). If handedness
  matters, add an explicit chirality check.
- **It ignores unit-cell multiplicity (materials).** `mol_emb` is effectively
  *intensive* (sum-pool + LayerNorm), so tiling a crystal into an 8× supercell
  leaves AFD near the floor. AFD won't flag a wrong cell multiplicity.
- **It is not an absolute validity / stability checker.** AFD measures
  distributional distance, not whether any single structure is physically valid,
  stable, or unique. Use it alongside validity/uniqueness/novelty metrics, not
  instead of them.
- **It does not localize errors.** A single number can't tell you *which* atoms or
  *what* property drifted — use the calibration table below to translate the
  magnitude into an intuition.

---

## 3. When AFD is a valid measure (sample size)

AFD inherits FID's central caveat: **the estimator is biased upward, and the bias
depends on the sample size N.** Concretely, even comparing the reference against a
held-out, *unperturbed* fold of true ground truth gives a non-zero "floor" that
shrinks roughly as `1/N`.

Two consequences:

### Rule 1 — Never compare AFD across different N.

A model scored with 1,000 samples and another with 5,000 samples are **not
comparable**: the smaller-N score is inflated by bias alone. Always match the
sample count on both sides. The tools make this easy — pass `ref_sample_size=N`,
which subsamples the reference to exactly your generated-set size (see §5).

### Rule 2 — Use enough samples.

Below the 256-d feature dimension the covariance is rank-degenerate and scores are
meaningless artifacts (e.g. N=10 and N=100 produced *smaller* numbers than larger,
better-resolved N). Above the feature rank the floor decays ≈ `1/N`. From the
`re_val` fold-vs-fold study:

**QM9 (molecules) finite-N floor:**

| N | Floor (unperturbed fold) | Verdict |
|---|---|---|
| 10, 100 | rank-degenerate (artifact) | unusable |
| 500 | 0.0105 | minimum usable floor |
| 1,000 | 0.0050 | minimum for ranking |
| 5,000 | 5.5·10⁻⁴ | resolves ~0.001 gaps |
| 10,000 | ≈ 0 | tightest |

**mp20 (materials)** follows the same law: floor ≈ 0.022 at N=500, 0.011 at
N=1,000, 0.0015 at N=5,000.

**Practical sample-size guidance:**

| Goal | Minimum N (each side) |
|---|---|
| Any meaningful score at all | **≥ 500** |
| Ranking models / detecting moderate differences | **≥ 1,000** |
| Resolving small gaps (~0.001) between strong models | **≥ 5,000** |

If your generated set is smaller than 500, AFD is not a valid measure — generate
more samples. If two models differ by less than the floor at your N, the
difference is not resolvable; either increase N or treat them as tied.

---

## 4. How to read a score

A raw AFD number is unitless. To build intuition, `re_val` calibrated AFD against
controlled, physically-interpretable perturbations of ground truth (QM9). Read any
AFD by the nearest row:

| AFD level | ≈ coordinate jitter (Å) | ≈ # atoms mutated | ≈ # atoms removed |
|---|---|---|---|
| 0.01 | 0.011 | 0.05 | 0.05 |
| 0.05 | 0.016 | 0.26 | 0.25 |
| 0.10 | 0.023 | 0.53 | 0.51 |
| 0.20 | 0.036 | 1.12 | 1.04 |
| 0.30 | 0.047 | 2.18 | 2.15 |

So **an AFD of 0.1 is about 0.023 Å of position noise, or about half an atom per
molecule mutated or missing.** A single mutated atom registers ≈ 0.19 (in-set) /
≈ 0.29 (out-of-distribution element); a single vacancy ≈ 0.20.

Key properties established in `re_val` that make these readings trustworthy:

- **Monotone and sensitive.** AFD rises monotonically with every degradation
  tested — position noise, atom mutation, vacancies, bond-length scaling, lattice
  strain (Spearman ρ ≈ 0.95–1.00). It resolves position noise as small as
  ≈ 0.03 Å (QM9) / ≈ 0.01 Å (mp20).
- **Reacts more strongly to out-of-distribution chemistry.** Mutating an atom to an
  element absent from the reference (e.g. Si/P/S in QM9, transuranics in mp20)
  scores higher than an in-set mutation at every count.
- **Consistent across folds.** Repeated independent folds cluster tightly (the
  RDKit-conformer experiment scored 0.187–0.196 across 11 folds + holdout), so a
  single number is reproducible.

### Interpreting the magnitude in practice

- **At or near the floor** for your N (see §3 table): indistinguishable from ground
  truth at this sample size.
- **~0.01–0.05:** small but real drift — sub-0.02 Å geometry error or a rare
  composition slip.
- **~0.1–0.2:** substantial — on the order of half an atom per structure wrong, or
  systematic force-field-level geometry error.
- **≥ 0.3:** large distributional drift (multiple atoms' worth of error, or
  out-of-distribution chemistry).

Always interpret a score *relative to the floor at your N* and *relative to other
models scored identically* — AFD is a comparative metric first.

---

## 5. How to use the tools

The core scorer lives in `zevals.atomistic.afd`, with domain-facing metric
wrappers in `zevals.molecules.afd` and `zevals.materials.afd`. Use the
project's `zatom` conda env. The AFD tools need `torch_geometric` +
`huggingface_hub` (both pure-Python) and, optionally, `torch_cluster` for fast
neighbour search:

```bash
pip install -e '.[atomistic]'

# Optional fast neighbour-search backend:
pip install torch_cluster -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
```

`torch_cluster` is **optional**. The neighbour-search backend is selected by the
`AFD_GRAPH_BACKEND` env var: `auto` (default — use `torch_cluster` if installed,
else a pure-torch fallback), `torch_cluster` (force), or `torch` (force the
dependency-free fallback). The two backends are numerically identical; on an A100
`torch_cluster` is ~3× faster at featurization, so it is preferred when available.

Reference embeddings and CT checkpoints are fetched from HuggingFace
(`Ty-Perez/AtomisticEval`); set `HF_HOME=$SCRATCH/hf_cache` so the cache lives on
scratch rather than the space-limited home dir.

### 5.1 The high-level scorer (recommended): `AFDScore`

```python
from zevals.atomistic.afd import AFDScore

# Construct once per reference. Reference features + checkpoint are fetched
# from HuggingFace automatically; no GPU cost for the reference side.
scorer = AFDScore("qm9", device="cuda:0")     # molecules  (ct-scd-pcq)
# scorer = AFDScore("mp20", device="cuda:0")   # materials  (ct-scd-amp20)

# `gen_records` is a list of structures — either torch_geometric Data objects
# or plain dicts, each carrying at minimum:
#     z   : (N,)    atomic numbers (long)
#     pos : (N, 3)  Cartesian coordinates in Å (float)
# For materials, also include:
#     cell : (1, 3, 3) lattice vectors,  pbc : (1, 3) periodic flags
score = scorer(gen_records)
print("AFD:", score)
```

**Always match N** so the comparison is valid (§3, Rule 1):

```python
# Subsample the reference to exactly your generated-set size.
score = scorer(gen_records, ref_sample_size=5000, seed=0)
```

`seed` makes the reference subsample reproducible. Use the same `ref_sample_size`
(and ideally the same `seed`) for every model you intend to compare.

#### Custom reference distributions

To score against your own ground-truth set instead of canonical QM9/mp20, pass the
records and tell the scorer which checkpoint to use via `data_modality`:

```python
scorer = AFDScore(my_ref_records, data_modality="materials", device="cuda:0")
#   data_modality "molecules" -> ct-scd-pcq ; "materials" -> ct-scd-amp20
```

For references outside the standard chemistries (a wider element range, or
non-ground-state geometries) you can name a checkpoint explicitly with
`model_name=...`. Otherwise prefer the matched canonical checkpoint — `re_val`
showed the in-domain specialist checkpoint dominates every alternative, so
**always use the checkpoint matched to your data modality.**

#### Extracting embeddings without scoring

```python
feats = scorer.embeddings(my_records)   # (N, 256) tensor of mol_emb features
```

Useful for caching, clustering, or feeding a different distance metric.

#### zevals metric wrappers

For integration with zevals/nucleus metric code, use the domain wrapper.
It returns a metric dictionary and, by default, subsamples the reference to the
generated record count:

```python
from zevals.molecules.afd import compute_afd_metrics

metrics = compute_afd_metrics(gen_records, device="cuda:0", seed=0)
# {"afd_score": ..., "afd_num_samples": ..., "afd_ref_sample_size": ...}
```

For materials, import the analogous wrapper:

```python
from zevals.materials.afd import compute_afd_metrics
```

### 5.2 The command-line driver

```bash
# QM9 reference, one or more candidate .pt files of records:
python -m zevals.atomistic.afd score --ref qm9 --gen qm9_gens/ADiT_qm9.pt --device cuda:0

# Match N across raw reference and candidate records (always do this for real comparisons):
python -m zevals.atomistic.afd score --ref qm9 --gen qm9_gens/*.pt \
    --ref-n 5000 --gen-n 5000 --device cuda:0

# Materials:
python -m zevals.atomistic.afd score --ref mp20 --gen my_crystals.pt --device cuda:0

# Custom reference from a .pt file of records:
python -m zevals.atomistic.afd score --ref my_ref.pt --gen my_gens.pt --device cuda:0
```

Each `--gen` is a `.pt` file holding a Python list of record dicts/`Data`. The
CLI prints a per-candidate AFD and a summary table. Useful flags:
`--gen-n` / `--ref-n` (cap records), `--model-name` (override checkpoint),
`--batch-size`, `--device`, and `--ref-cache`. For canonical precomputed
reference embeddings plus reproducible `ref_sample_size` subsampling, prefer the
Python `AFDScore` API or zevals wrappers above.

### 5.3 Low-level building blocks

If you want to manage features yourself:

```python
from zevals.atomistic.afd import FeatureExtractor, afd_score

ext = FeatureExtractor(data_keyword="qm9", device="cuda:0")
feats_ref = ext(ref_records)     # (N, 256)
feats_gen = ext(gen_records)     # (M, 256)
score = afd_score(feats_ref, feats_gen)   # match N yourself before calling
```

`afd_score(features_ref, features_gen)` is the pure feature-space distance; pair it
with `gaussian_stats` / `frechet_distance` if you need the intermediate μ, Σ.

---

## 6. A correct end-to-end comparison (checklist)

To rank a set of models fairly:

1. Pick the reference matching your modality (`"qm9"` or `"mp20"`), and let the
   tool pick the matched checkpoint.
2. Generate **≥ 1,000** samples per model (≥ 5,000 to resolve close calls).
3. Score every model with the **same** `ref_sample_size` and `seed`, set to the
   common sample count.
4. Compare scores **only** against each other and against the floor at that N — a
   gap smaller than the floor is not resolvable.
5. Translate magnitudes with the calibration table (§4) for intuition.
6. Add a chirality check (molecules) and a cell-multiplicity / validity check
   (materials), since AFD is blind to both.

---

## 7. Evidence: the `re_val/` validation suite

The empirical claims above come from `re_val/`, a self-contained validation suite
that uses these exact AFD tools (it calls `AFDScore` directly — there is no
separate scoring path). It is organized in three blocks, all reproducible via
`python -m re_val.run_all` (≈ 44 min on one RTX A5000); full per-figure detail is
in `re_val/FIGURES.md`.

- **Block A — Sensitivity (`re_val/sensitivity.py`).** Perturbs *ground-truth
  folds* by known amounts, so the answer is known: an unperturbed fold sits at the
  finite-N floor, and each degradation rung's entire AFD rise is attributable to
  it. This block established the **sample-size floor** (§3), the **monotonic
  sensitivity** to position noise / mutation / vacancy / scaling / strain, the
  **calibration table** (§4), and the **chirality** and **supercell** blind spots.
- **Block B — Checkpoints (`re_val/checkpoints.py`).** Reruns the ladders across
  four CT checkpoints. The in-domain specialist (`ct-scd-pcq` for QM9,
  `ct-scd-amp20` for mp20) dominated every alternative by a wide margin — the
  basis for "**always use the matched checkpoint.**" Feature spaces are not
  comparable across checkpoints, so the checkpoint is part of the metric.
- **Block C — Generated data (`re_val/generated.py`).** The downstream use case:
  scoring real generated geometries (RDKit conformers, xtb-relaxed conformers)
  against ground truth. Showed AFD reads systematic geometry error consistently
  across folds and correctly **ranks generation/relaxation pipelines**.

If you change the scoring tools, rerun `re_val` to confirm the floor, sensitivity,
and calibration numbers still hold — the guidance in this document is only as valid
as those results.
