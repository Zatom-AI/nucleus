# Appendix C: visualizations

The five `../figs/generated_*_molstar.pdf` files contain actual Mol* renders
of 48 saved Zatom-2 outputs. Labels remain vector text. No model inference,
relaxation, sequence design, or refolding is needed to rebuild the figures.

## Sources and selection

`manifest.json` records the original repository-relative paths, checksums,
checkpoint provenance, sample seeds where available, and exact selection rules.
`structures/` contains small, unmodified copies of the decompressed molecule
and material mmCIFs and the generated protein backbone PDBs. These copies make
the galleries reproducible independently of the local `logs/` tree. Source
logs were read only.

- **QM9 molecules:** joint QM9+MP20 epoch 9,925, from
  `logs/evaluation/table2/paper-table2-qm9-mp20-unconditional-without-force-conditioning-epoch-9925-unrelaxed/qm9/`.
  The first nine distinct formulae with connected, sanitizable neutral RDKit
  graphs were selected in ascending sample-ID order. The selection used
  RDKit bond inference only as an illustration filter, independently of the
  published benchmark scorer. Mol* perceives the displayed bonds from the
  original mmCIF coordinates; its displayed bonds are not a chemical validation.
- **MP20 materials:** samples from the same checkpoint/run. The first nine
  distinct unit-cell compositions in the archived LeMat-GenBench validity list
  were selected. Each staged structure was matched byte-for-byte after
  decompression to its original output. Inputs are unrelaxed; no stability or
  novelty filter was applied. Formulae count atoms in the saved cell and use
  alphabetical element order (Hill order for carbon-containing compositions).
- **OMol25 molecules and OMat24 materials:** the `epoch_75` validation outputs
  of `paper-table3-omol25-omat24-joint-properties-force-conditioned`, job
  `2026-09-04_16-35_JOB_57928157`, under `logs/train/`. The run configuration
  and its checksum, the checkpoint path, and each selected generation input
  are recorded in the manifest. These are the saved `omol25_generation` and
  `omat24_generation` predictions, not reference structures or structure
  prediction outputs. Each source cohort contains 1,000 samples. The model
  uses force conditioning, and the inputs specify atom counts and molecular
  charge/spin where applicable; compositions and coordinates are generated.
  OMol25 selection is stratified by the source subset of the conditioning
  values, as detailed below. OMat24 selection
  takes the first nine distinct unit-cell compositions with at most 40 atoms
  (for legibility), checking finite lattice parameters and positive cell
  volume only. No LeMat-GenBench validity, stability, or novelty filter is
  applied to this gallery. Both retain the original generated coordinates;
  OMat24 lattices are unrelaxed. Short labels `omol25-NNNN` and `omat24-NNNN`
  use the input manifest's sample index; full IDs and source hashes are saved.
- **Proteins:**
  `logs/analysis/scope_best_20260919/joint_properties_fc/`, raw step 130,000
  (30,000 local SCOPe steps). The source cohort has 1,027 generated backbones,
  13 at each integer length 50–128. From designable backbones (saved best-of-eight
  C-alpha scRMSD < 1.5 Å), select the first sample ID nearest each target length
  60, 90, 120 in each category: helix-rich (alpha fraction >= 0.6, beta < 0.1),
  mixed (alpha >= 0.2, beta >= 0.2), and sheet-rich (beta >= 0.4, alpha < 0.1).
  Fractions and scRMSDs come from the saved `per_backbone_metrics.csv`;
  no metrics were recomputed. Short labels omit `scope_` and `_0_model_0`.
  The PDBs are the generated backbones used for evaluation, not ESMFold2 refolds.

This is a selected qualitative gallery, not a random sample, a new evaluation,
or evidence of novel folds or experimental stability.

### OMol25 source stratification

The twelve panels cover all ten `data_id` values found among the 1,000
generation inputs. For biomolecules, panel (a) retains the first source filename
and panel (b) uses the largest generated output by atom count, including
hydrogens, with ascending source-filename order breaking ties. In ascending
source-filename order, take the first two electrolyte samples and the first eligible
sample from each remaining subset. Eligibility includes the connectivity
policy below. The resulting panel order is:

| Panels | Conditioning-source subset | Raw `data_id` | Inputs in cohort |
| --- | --- | --- | ---: |
| a, b | Biomolecules | `biomolecules` | 210 |
| c | SPICE | `spice` | 11 |
| d, e | Electrolytes | `elytes` | 195 |
| f | Metal complexes | `metal_complexes` | 211 |
| g | Reactivity | `reactivity` | 181 |
| h | Transition-1x | `trans1x` | 57 |
| i | RGD1 | `rgd` | 21 |
| j | ANI-2x | `ani2x` | 46 |
| k | GEOM | `geom_orca6` | 55 |
| l | OrbNet Denali | `orbnet_denali` | 13 |

Panel (b), `omol25-0878`, contains 177 atoms, compared with 74 in panel (a).
It is the largest of all 210 biomolecule-source outputs in this saved
1,000-sample generation cohort. Its conditioning reference comes from the
`rna/4v5s_DB100_eqeq_frame0_-5_1` source. Selection ranks the generated
atom counts, not reference molecular sizes or a different checkpoint's outputs.
The manifest records all 210 candidates in descending size order and the
selected structure's full source metadata. All candidate outputs were parsed,
checked for finite coordinates, and matched to the reference atom count,
charge, spin, and mean atomic force norm. No connectivity filter is applied
to this biomolecule panel.

Join each generation input's `extra.source_row_index` to the OMol25 `train_4M`
dataset using `atomworks.ml.datasets.ASELMDBDataset` with lexicographically
sorted `*.aselmdb` shards under `OMOL25_DATA_DIR` (excluding `._*` files),
`return_type="record"`, `readonly=True`, `readahead=False`, and
`include_data=True`, matching the original input-generation script
`scripts/make_omol25_atom_count_contig_json.py`. Read `extra_info.data_id`
and `extra_info.source`. The join was checked for all 1,000 inputs: reference
atom counts, charge, and spin match exactly, and mean atomic force norms recomputed from the
source forces match within an absolute tolerance of 1e-10 eV/angstrom.
Each selected panel records the source row index, shard, database row ID,
original source identifier, reference formula, and checked conditioning values
in `manifest.json`; source and generated formulae are explicitly separate.

Subset names describe **conditioning-source provenance**. The model receives
atom count, charge, spin, and mean atomic force norm, not a subset label or fixed
reference composition/geometry. Therefore the labels do not classify the
generated structures. In particular, Transition-1x and RGD1 include reaction
paths; these outputs are not verified transition states. No transition-state
search or vibrational analysis was performed. Parsed atom counts and finite
coordinates are checked for every selected output. All generated atoms and
coordinates are retained without optimization or fragment removal.

**Connectivity policy.** GEOM outputs must be a single connected component.
For ANI-2x, OrbNet Denali, and SPICE, require a single generated component when
the corresponding source reference is connected under the same graph rule.
These mixed datasets also contain noncovalent systems, so their disconnected
references remain eligible without a connectivity requirement. Biomolecules,
electrolytes, metal complexes, reactivity, Transition-1x, and RGD1 remain
unfiltered to allow clusters, complexes, and reaction-path geometries.

The filter uses RDKit `DetermineConnectivity(useHueckel=False, useVdw=True,
covFactor=1.3)` and `GetMolFrags`, counting **all atoms including hydrogens**.
The distance threshold is 1.3 times the sum of covalent radii. This is a
coordinate-based illustration filter, not a chemical-validity assessment;
no bond-order assignment or sanitization is required. Mol* independently
perceives the displayed bonds. `manifest.json` records the RDKit version,
reference and generated component sizes, and accepted/rejected candidates in
filename order for the four filtered subsets.

This replaces GEOM `omol25-0002` (component sizes 35, 16, 16, 16) with
`omol25-0003` (59 connected atoms), and ANI-2x `omol25-0034` (12 + 1 atoms)
with `omol25-0040` (9 connected atoms). The selected SPICE and OrbNet Denali
outputs already pass and are retained. The other eight panels are unchanged.

To repeat selection using the original local `train_4M` data and generation
logs, run from the repository root with `OMOL25_DATA_DIR` configured:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python \
  models/zatom2/docs/manuscript/visualization/filter_omol25_connectivity.py
```

Then rerender with `render.py --galleries omol25` as described below. Selection
only updates the four affected subset entries and their bundled structures;
reference datasets and original generation logs remain read-only.

## Rebuild

From the repository root, install the isolated rendering environment and the
pinned Mol* viewer. These commands download public software only:

```bash
uv venv --python 3.12 /tmp/zatom2-visualization-venv
uv pip install --python /tmp/zatom2-visualization-venv/bin/python \
  playwright==1.63.0 pillow==12.3.0 pymupdf==1.28.2
PLAYWRIGHT_BROWSERS_PATH=/tmp/zatom2-playwright \
  /tmp/zatom2-visualization-venv/bin/python -m playwright install chromium
mkdir -p /tmp/zatom2-molstar
curl --fail --location https://registry.npmjs.org/molstar/-/molstar-5.11.0.tgz \
  --output /tmp/zatom2-molstar/molstar.tgz
tar -xzf /tmp/zatom2-molstar/molstar.tgz -C /tmp/zatom2-molstar
PLAYWRIGHT_BROWSERS_PATH=/tmp/zatom2-playwright \
  /tmp/zatom2-visualization-venv/bin/python \
  models/zatom2/docs/manuscript/visualization/render.py \
  --molstar-viewer /tmp/zatom2-molstar/package/build/viewer
```

The script verifies the Mol* JavaScript bundle and all structure checksums.
It renders through headless Chromium with SwiftShader (CPU WebGL), blocks
network requests during rendering, and exports 1400 × 1100 pixel Mol* images.
Only white margins are trimmed before PDF composition. Images with no visible
geometry or geometry touching the canvas edge fail the build. `--output DIR`
can redirect PDFs and the camera/atom-count audit to a temporary directory.
Append `--galleries omol25 omat24` to render only the new galleries, preserving
other gallery PDFs and existing audit entries in the output directory.
GPU/driver differences may change pixels slightly; coordinate files are fixed.

Molecules use element-colored ball-and-stick representations, including
hydrogens. Materials use element-colored spheres in a 2 × 2 × 2 repetition
(eight copies, verified by atom count), with the original unit cell outlined
in gray. The mmCIFs record P1; no higher symmetry is inferred or imposed.
Proteins use Mol* cartoons with a blue-to-red sequence-position gradient, not
a confidence color scale. Molecule and protein cameras align the structure principal axes to the screen;
crystals use a fixed oblique viewing direction. Each panel is independently
framed and scaled.
`../figs/molstar_render_audit.json` records camera snapshots and rendered atom
counts. The browser session never uploads the structures to molstar.org.

After rendering, compile `../iclr2027_conference.tex` with its bibliography
and inspect Appendix C. The renderer is separate from the manuscript build;
LaTeX needs only the five checked-in PDFs.

## OMol25/OMat24 latent-space comparison

`../figs/subset_embeddings_pca.pdf` compares base Zatom-2 (IV), without
energy/force prediction pretraining, against base Zatom-2 (V), with it.
Columns identify the checkpoints; rows show the joint sample, OMol25, and
OMat24. Colors and validation structures are identical across columns.
Both checkpoints use EMA weights at step 97,500 (zero-based epoch 79).
Both were pretrained with force conditioning; it is disabled for this analysis.
The supervised recipe also changes task sampling from 0.5/0.5 generation/
structure prediction to 0.34/0.33/0.33 generation/structure prediction/MLIP and
includes latent-equivariance regularization. This compares the two pretraining
recipes, rather than isolating a single loss term.

Rebuild from the repository root without checkpoints, GPUs, or source datasets:

```bash
OPENBLAS_NUM_THREADS=4 uv run --no-sync python \
  scripts/plot_zatom2_subset_embeddings.py plot
```

The `subset_embeddings_without_properties.npz` and
`subset_embeddings_with_properties.npz` archives contain the mean-pooled final
atom-decoder representations, element fractions, atom counts, labels, and
per-example input checksums. Their corresponding `_provenance.json` files record
checkpoint, common preprocessing, manifest, and archive hashes. Plotting refuses
to compare mismatched inputs or sample metadata. `subset_embeddings_results.json`
and `subset_embeddings_pca.csv` contain the diagnostics and plotted coordinates.
`subset_embeddings_checkpoint_comparison.json` records the saved training
counters, task frequencies, and model-configuration differences.
The original large-model archive (`subset_embeddings.npz`) and its provenance
remain archival outputs and are not used by the comparison.

The main-text comparison uses exactly the saved joint-panel coordinates and
variance percentages, exported in a compact two-panel layout without refitting:

```bash
uv run --no-sync python scripts/plot_zatom2_subset_embeddings.py plot-main
```

This writes `subset_embeddings_joint_pca.pdf` and `.png`, preserving the appendix
panel colors and model order.

`subset_embeddings_samples.json` fixes the original 1,472 validation examples:
128 per molecular group and 64 per material subset, sampled without replacement
with seed 20260922 and an eligibility range of 2–512 atoms. Neutral organics pools
ANI2x, OrbNet Denali, and GEOM-ORCA6 without balancing these three sources.
Both new extractions replay those exact shard/row IDs and retain every atom,
including hydrogens. No new example selection depends on either model.

To repeat extraction into a new directory, configure the existing validation
paths and set `BASE_WITHOUT_EF` and `BASE_WITH_EF` to the local paths of the
registered `zatom2-table1-joint-ablation` and
`zatom2-table1-joint-properties-ablation` checkpoints, respectively:

```bash
for variant in without_properties with_properties; do
  checkpoint="$BASE_WITHOUT_EF"
  if [ "$variant" = with_properties ]; then checkpoint="$BASE_WITH_EF"; fi
  DEBUG=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run --no-sync python \
    scripts/plot_zatom2_subset_embeddings.py extract --variant "$variant" \
    --checkpoint "$checkpoint" --preprocessing-checkpoint "$BASE_WITH_EF" \
    --output logs/subset_embeddings_repeat
done
OPENBLAS_NUM_THREADS=4 uv run --no-sync python \
  scripts/plot_zatom2_subset_embeddings.py plot \
  --output logs/subset_embeddings_repeat
```

Extraction uses one GPU per invocation, never downloads weights, and refuses to
overwrite an archive. Both models use the supervised base checkpoint's identical
MLIP preprocessing configuration. Clean reference coordinates, known elements,
and native charge/spin metadata enter one forward pass at flow time 1, with no
force conditioning or random augmentation. A forward hook captures `Q_L` from
`diffusion_module.backbone`, before task heads, so the model without property
heads is evaluated identically. For the supervised model, this capture is checked
for exact equality with `predict_properties(..., return_embeddings=True)` on one
example from each domain, after removing the singleton diffusion dimension.

Every panel fits a centered PCA independently, with equal structure weights,
no channel standardization, and no vector normalization. Axes are not aligned
between checkpoints; positions, signs, or apparent cluster area cannot be
compared directly. Full-space Euclidean 10-neighbor subset purity excludes the
query itself and is compared with element-fraction vectors and the exact
random-label expectation for the group sizes. These are descriptive statistics
on one matched sample and one checkpoint per recipe. They do not establish
physical similarity independently of composition, size, periodicity, electronic
metadata, or related conformers/materials among the neighbors.

## References

- [Zatom-1, Appendix C.2](https://arxiv.org/html/2602.22251v4#A3.SS2)
- [Mol* Viewer](https://molstar.org/)
- [Sehnal et al., 2021](https://doi.org/10.1093/nar/gkab314)
- [OMol25: source domains and community datasets](https://arxiv.org/html/2505.08762v2)
