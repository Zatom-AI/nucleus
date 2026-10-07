# Data and reference preparation

Run from the repository root after `uv sync --locked --extra all`. Copy
`.env.example` to `.env` and set the dataset locations. Raw corpora, external
model weights, and generated caches are not stored in Git. The release contains
frozen selection indices, normalization statistics, evaluation requests, and
1,500 reference CIFs used by the paper's structure/property evaluations.

## OMol25 and OMat24

Obtain **OMol25 train 4M and validation** from the download links in the
[official OMol25 model/data card](https://huggingface.co/facebook/OMol25), and
**OMat24 train 1M and validation** from the
[official OMat24 card](https://huggingface.co/datasets/facebook/OMAT24).
Accept the upstream dataset terms where required. Extract the original ASE-LMDB
archives without renaming or merging shards. The data cards are the source of
record for the archive URLs and their licenses.

Expected training cardinalities are **3,986,754 OMol25** and **1,009,850 OMat24**.
Keep OMat24's subset directories (`rattled-*`, `aimd-*`) intact. The loader sorts
shard paths; replacing or reordering source shards changes the frozen index
selection. Validation must remain separate: paper MLIP evaluation uses
2.76 million OMol25 and 107,732 OMat24 validation records.

```bash
export OMOL25_DATA_DIR=/path/to/omol25/train_4M
export OMOL25_VAL_DIR=/path/to/omol25/val
export OMAT24_DATA_DIR=/path/to/omat24/train
export OMAT24_VAL_DIR=/path/to/omat24/val
python scripts/validate_data.py
```

[`ASELMDBDataset` configurations](../models/zatom2/configs/datasets/train/omol25/omol25.yaml)
use AtomWorks' ASE loaders to construct atom arrays and energy/force labels.
Periodic structures use the [materials loader](../models/zatom2/configs/datasets/train/omat24/omat24.yaml).
The shared training builder applies deterministic subsetting before sampling.
The 500k/domain scaling selections use separate NumPy PCG64 generators with seed
42 and enforce these index hashes:

| Dataset | Selected-index SHA-256 |
| --- | --- |
| OMol25 | `36b7366932ae1dbe3941743373610a063018697a2f2cd16664efbca114bdf01e` |
| OMat24 | `73d46f5304147e419a81c6df15f9a0fa628c199a525566637bf2b6fc887a5227` |

### Molecular bond conditioning

The paper infers bonds with Open Babel and excludes disconnected examples from
the structure-prediction task (60.32% of the OMol25 training pool). The reference
release attachment includes the original training and validation bond sidecars
(see [Reference data release attachment (Release 1.0.0)](#reference-data-release-attachment-release-100)).
Install them with:

```bash
python scripts/install_reference_data.py --archive zatom2-reference-data.tar.gz
```

To regenerate sidecars for both source splits, use the pinned environment:

```bash
python scripts/prepare_omol25_bonds_lmdb.py \
  --input-dir "$OMOL25_DATA_DIR" --data-id all \
  --output data/omol25/sidecars/train_4M_all_bonds.lmdb
python scripts/prepare_omol25_bonds_lmdb.py \
  --input-dir "$OMOL25_VAL_DIR" --data-id all \
  --output data/omol25/sidecars/val_all_bonds.lmdb
```

Sidecar keys are `example_id`s, not random row indices. Failure reports are
written beside each LMDB. The training loader applies the sidecar's connected
fragment mask only to the structure task. Checked-in subset files under
`data/omol25/subsets/` retain the original source ordering.

### Normalization and novelty references

`data/atomistic_property_statistics.v1.json` contains the training-only elemental
energy references and domain normalization scales. Never fit these on validation
or test data. To recompute them on the same source shards:

```bash
python scripts/fit_atomistic_property_statistics.py \
  --output artifacts/atomistic_property_statistics.v1.json
python scripts/prepare_omol25_reference_smiles.py \
  --input-dir "$OMOL25_DATA_DIR" --partition all \
  --output data/omol25/subsets/omol25_train_4M_smiles_obabel.smi
python scripts/prepare_omol25_reference_fpsim2.py \
  --input data/omol25/subsets/omol25_train_4M_smiles_obabel.smi \
  --output data/omol25/subsets/omol25_train_4M_smiles_obabel_fpsim2.h5
python scripts/prepare_omat24_reference_fingerprints.py \
  --input-dir "$OMAT24_DATA_DIR" \
  --output data/omat24/subsets/train_1M_fingerprints.npy
```

AFD requires its separate CT checkpoints and reference embeddings; consult the
[AFD guide](../models/zevals/docs/AFD-guide.md). The OMol25 protocol points to
`data/omol25/geom_orca6_rdkit_conformers.pt`, a custom reference artifact. Its
file size and SHA-256 are recorded in the
[reference manifest](reference_data_manifest.json). The release attachment includes
this file, both OMol25 bond sidecars, the original OMol25/OMat24 novelty caches,
the GEOM-subset training SMILES and fingerprints, and the cached `ct-scd-geom10`
and `ct-scd-omol25` AFD reference features. The latter includes the checksum and
protocol metadata required by the manuscript figure scripts.

Do not silently substitute QM9 or MP20 reference features for this molecular reference.
Materials AFD uses the `Ty-Perez/ct-scd-amp20` model and MP20 reference.

### Reference data release attachment (Release 1.0.0)

Because GitHub Releases enforces a 2 GB per-file asset limit, the 2.26 GB
(2,255,986,830 bytes) reference archive `zatom2-reference-data.tar.gz` is published
for release **`1.0.0`** (tag: `1.0.0`) split into two equal parts:
- `zatom2-reference-data.tar.gz.part00` (1,127,993,415 bytes)
- `zatom2-reference-data.tar.gz.part01` (1,127,993,415 bytes)
accompanied by `checksums.sha256`.

#### 1. Download

Download the split parts and checksums from release `1.0.0`:

Using GitHub CLI:
```bash
gh release download 1.0.0 -R Zatom-AI/nucleus \
  -p "zatom2-reference-data.tar.gz.part*" \
  -p "checksums.sha256"
```

Or using `curl`:
```bash
curl -LO https://github.com/Zatom-AI/nucleus/releases/download/1.0.0/zatom2-reference-data.tar.gz.part00
curl -LO https://github.com/Zatom-AI/nucleus/releases/download/1.0.0/zatom2-reference-data.tar.gz.part01
curl -LO https://github.com/Zatom-AI/nucleus/releases/download/1.0.0/checksums.sha256
```

#### 2. Verify and reassemble

Verify the downloaded part checksums:
```bash
sha256sum -c checksums.sha256 --ignore-missing
```

Reassemble the parts into the complete archive:
- **Linux / macOS**:
  ```bash
  cat zatom2-reference-data.tar.gz.part* > zatom2-reference-data.tar.gz
  ```
- **Windows (Command Prompt)**:
  ```cmd
  copy /b zatom2-reference-data.tar.gz.part00 + zatom2-reference-data.tar.gz.part01 zatom2-reference-data.tar.gz
  ```

Verify the reassembled archive matches the expected checksum:
```bash
sha256sum -c checksums.sha256
```

#### 3. Process and install

Install the verified reference artifacts into the project `data/` directory:
```bash
python scripts/install_reference_data.py --archive zatom2-reference-data.tar.gz
```

The script extracts the archive into `data/` (or `--destination /path/to/data`) and
strictly verifies each file's size and SHA-256 against
[reference_data_manifest.json](reference_data_manifest.json). Existing identical
reference files are preserved; any differing files will abort installation.

The archive supplies derived reference artifacts; obtain the raw ASE-LMDB
datasets and external model checkpoints separately as described above. Frozen
selection indices and normalization statistics are already included in Git.

See [release status](release_status.md) for attachment publication status.

## QM9 and MP20

Follow the [pinned download and ASE-LMDB conversion guide](qm9_mp20.md).
The source revisions are QM9 `fbf62fcfdcd61960160b04c3056d161ebb62724e` and MP20
`9d9e762a5590105b591e237cf3bc61bd5b671890`. Training contains exactly 100,000 and
27,138 records, respectively. The converter writes source and index hashes and
checks the resulting loader cardinality. It refuses to mix new shards into an
existing conversion.

## GEOM-Drugs

```bash
python scripts/download_geom.py --data-dir data/geom
python scripts/make_geom_unconditional_json.py --data-dir data/geom \
  --num-samples 10000 --seed 42 \
  --output models/zatom2/docs/examples/geom_drugs_unconditional_10000.json
```

The downloader verifies dataset revision `855cd36632b30787b913743ec21903283dba41c8`
and archive SHA-256 `3183cae6e8eaaa63500cc67fd1e949d62d4c97b27148c922486ef008359856d5`.
The training split has 243,480 molecules and 1,172,577 conformers (up to five
released conformers per molecule). All are separate training samples. The
ConfGF/GO-Flow conformer-generation experiment is a separate protocol; it must
not be evaluated with unconditional GEOM requests.

## SCOPe-2k

The exact domain lists and CSVs are included under `data/scope/rungs/seed0`.
Seeds 1 and 2 are also retained when present in the source snapshot. Obtain
SCOPe 2.08 ASTRAL-40 coordinates:

```bash
mkdir -p data
curl -L --fail https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-sel-gs-bib-40-2.08.tgz \
  -o data/pdbstyle-sel-gs-bib-40-2.08.tgz
tar -xf data/pdbstyle-sel-gs-bib-40-2.08.tgz -C data
python scripts/link_scope_structures.py --pdbstyle-dir data/pdbstyle-2.08
```

The linking helper consumes the frozen CSV rather than resampling the domains.
To reconstruct the balanced selection itself, use
[`prepare_scope_rungs.py`](../scripts/prepare_scope_rungs.py) with Genie's
`coords/*.npy`, `classes.txt`, and original pdbstyle files. It cycles through
shuffled SCOPe fold groups, excludes the six parser-incompatible domains, and
constructs nested 1k/2k/4k selections. The short pool is 3,936 domains.

Protein preprocessing also needs AtomWorks' CCD resources. Configure
`CCD_MIRROR_PATH` according to the
[AtomWorks setup guide](https://github.com/amorehead/atomworks); seeded protein
sampling rejects a missing mirror before it loads a model.

## Licenses

OMol25 and OMat24: CC BY 4.0. QM9: CC0. MP20: follow Materials Project and
upstream redistribution terms. GEOM: CC0. SCOPe/ASTRAL: upstream SCOPe terms;
PDB coordinates: CC0. LeMat-Bulk: CC BY 4.0. See the preprint's dataset-license
table and upstream source cards for exact attribution and usage conditions.
