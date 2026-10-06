# Third-party notices

Nucleus, Zatom-2, and Zevals retain their original license files. The root BSD
license does not override third-party software, checkpoint, dataset, or webpage
terms.

- **AtomWorks** and **RosettaCommons Foundry/RFdiffusion3** are installed from
  the public revisions pinned in `pyproject.toml` and `uv.lock`. Their licenses
  remain with their distributions.
- **ProteinMPNN and NA-MPNN** code is vendored under
  `models/zevals/src/zevals/oracles/inverse_folders/`; accompanying LICENSE files
  are retained. Follow upstream terms for downloaded weights.
- **Mol\*** is distributed with its embedded MIT copyright/license header in
  `models/zatom2/docs/static/js/molstar.js`.
- **Academic Project Page Template** by Eliahu Horwitz is licensed CC BY-SA 4.0.
  The adapted Zatom-2 page retains its attribution and links to
  https://github.com/eliahuhorwitz/Academic-project-page-template and
  https://creativecommons.org/licenses/by-sa/4.0/ .
- **Bulma** and its carousel/slider assets retain embedded MIT notices.
  **Font Awesome** and **Academicons** retain their bundled headers and font
  licensing terms (Font Awesome Free: CC BY 4.0 icons, SIL OFL 1.1 fonts, MIT code;
  Academicons: SIL OFL 1.1 font and MIT CSS).
- LaTeX style files in the manuscript directory retain their original notices.
- Datasets and derived reference structures remain subject to the source
  dataset licenses in [data.md](docs/data.md) and the preprint's license table.
- External UMA, ESMFold2, AFD/CT, Boltz, LeMat, CHGNet, ORB, and MACE weights and
  tools retain their respective upstream terms. Some require gated model access.

The model checkpoint registry records filenames and hashes; distributing code
under BSD does not itself grant additional rights to third-party weights.
