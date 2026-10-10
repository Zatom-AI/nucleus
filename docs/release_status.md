# Release status and provenance

This public version of `nucleus` was prepared from development snapshot
`05f6ce04a73143fc65e4b916fb321c8d74c25cf3`, preserving the Zatom-2 paper code and
Zevals in one monorepo. This release excludes unreleased model packages,
experimental StructureCloud training, cluster account/reservation launchers,
private environment files, run logs, W&B/Overleaf administration, scratch oracle
experiments, unfinished oracle wrappers, and LaTeX build intermediates.

## Publication dependencies

1. **Weights:** model checkpoints can be found at `https://huggingface.co/Zatom-AI/Zatom-2`.
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
3. **arXiv:** the preprint is available at `https://arxiv.org/abs/2610.11454`.
4. **Remaining reproducibility gaps in the public repository:** the MP20
   structure prediction and GEOM-Drugs experiments do not have selected checkpoints 
   in this snapshot. The RFdiffusion3 SCOPe baseline checkpoint is also absent 
   from the public registry.
