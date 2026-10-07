# Release status and provenance

The public source was prepared from development snapshot
`05f6ce04a73143fc65e4b916fb321c8d74c25cf3`, preserving the Zatom-2 paper code and
Zevals in one Nucleus monorepo. The release excludes unreleased model packages,
experimental StructureCloud training, cluster account/reservation launchers,
private environment files, run logs, W&B/Overleaf administration, scratch oracle
experiments, unfinished oracle wrappers, and LaTeX build intermediates.

## Publication dependencies

1. **Weights:** `https://huggingface.co/Zatom-AI/Zatom-2` returned HTTP 401 during
   release preparation, including an anonymous request. The registry retains
   27 filenames and SHA-256 values from the recorded Hub inventory at revision
   `74b6811f6356d701b6fa5ad627d529a2ef51540d`. Make those assets publicly readable
   before advertising anonymous downloads. No model weights were downloaded or
   uploaded by this preparation task.
2. **Reference data:** release preparation produced and verified
   `dist/zatom2-reference-data.tar.gz` for attachment to the GitHub release.
   Its size (2,255,986,830 bytes), SHA-256, and individual file hashes are recorded in
   `docs/reference_data_manifest.json`. It contains the original OMol25 training
   and validation bond sidecars, custom molecular AFD records and cached features
   (including the audited OMol25 feature metadata), molecular/material novelty
   caches, and GEOM-subset training SMILES and fingerprints.
   These large artifacts belong in a release attachment, not Git history.
   Because GitHub Releases enforces a 2 GB per-file asset limit, the archive for
   release `1.0.0` (tag: `1.0.0`) is published as two equal parts:
   - `zatom2-reference-data.tar.gz.part00` (1,127,993,415 bytes)
   - `zatom2-reference-data.tar.gz.part01` (1,127,993,415 bytes)
   accompanied by `checksums.sha256`.

   Users can download and reassemble the release data using the GitHub CLI:
   ```bash
   gh release download 1.0.0 -R Zatom-AI/nucleus \
     -p "zatom2-reference-data.tar.gz.part*" \
     -p "checksums.sha256"
   sha256sum -c checksums.sha256 --ignore-missing
   cat zatom2-reference-data.tar.gz.part* > zatom2-reference-data.tar.gz
   sha256sum -c checksums.sha256
   ```
   or via direct download:
   ```bash
   curl -LO https://github.com/Zatom-AI/nucleus/releases/download/1.0.0/zatom2-reference-data.tar.gz.part00
   curl -LO https://github.com/Zatom-AI/nucleus/releases/download/1.0.0/zatom2-reference-data.tar.gz.part01
   curl -LO https://github.com/Zatom-AI/nucleus/releases/download/1.0.0/checksums.sha256
   cat zatom2-reference-data.tar.gz.part* > zatom2-reference-data.tar.gz
   ```
   Users then install the reassembled reference data with:
   ```bash
   python scripts/install_reference_data.py --archive zatom2-reference-data.tar.gz
   ```
3. **arXiv:** `https://arxiv.org/abs/XXXX.XXXXX` is an intentional placeholder
   requested by the authors. Replace it after submission. The preprint PDF is
   included and linked independently. Its project URL was corrected to the public
   site, with all other extracted text verified unchanged across all 49 pages.
4. **Remaining reproducibility gaps in the source snapshot:** the manuscript's
   separate ConfGF/GO-Flow GEOM conformer-generation and Crystalite MP20
   structure-prediction experiments do not have a complete matched set of
   experiment configs, split manifests, and selected checkpoints in this
   snapshot. The RFdiffusion3 SCOPe baseline checkpoint is also absent from the
   public registry. Supply these materials before claiming complete reproduction
   of every appendix and baseline result. Existing generic metrics are not a
   substitute for those experiment artifacts.

Public source preparation does not publish GitHub commits, enable Pages in
repository settings, change Hub visibility, or publish a GitHub release.
The included Pages workflow can be dispatched once GitHub Pages uses GitHub
Actions as its source. It publishes only the project-page files.

## Validation boundaries

The release audit checks Python syntax, retained fixture references, local page
assets, checkpoint metadata, package entry points, and excluded file patterns.
Targeted tests exercise download checksum failures, data subsetting/conversion,
config composition, sampling protocols, and inference/evaluation plumbing.
Builds verify both the source archive and wheel. The paper's multi-GPU training
runs, gated external oracles, and full checkpoint inference are not rerun by
these offline checks. Numerical reproduction still requires the datasets,
weights, compute, and artifacts described above.

The offline audit passed for 2,283 release files. The lightweight CI suite
passed all 24 tests, the selected Zevals suite passed all 205 tests, and the
GEOM suite passed all 33 tests. All 73 inference and logging regression tests
passed, including the three documented quickstart inputs and checkpoint
normalization-path relocation. The installed-wheel smoke check ran outside the
checkout, bypassing editable installs: package imports, all three CLIs, Hydra
configuration, the inference target, and bundled normalization statistics
passed. The reference archive was installed into an empty directory and
every file checksum matched. Both split parts (`part00`, `part01`) and the reassembled
archive were verified against `dist/checksums.sha256` and `docs/reference_data_manifest.json`.
Both Python distribution formats built successfully; `uv lock --check` passed. The
manuscript was built with TeX Live 2024; the bibliography was unchanged and all references resolved.
