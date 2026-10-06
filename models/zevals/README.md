# Zevals

Zevals evaluates generated molecules, materials, proteins, RNA, and ligands.
It ships in the shared [Nucleus monorepo](../../README.md). Install the root
`all` extra for Zatom-2 training/inference and standard evaluation dependencies:

```bash
uv sync --locked --extra all
source .venv/bin/activate
zevals --help
```

For evaluation alone, install `models/zevals` with the needed domain extra
(`molecules`, `materials`, `proteins`, `rna`, or `all`) in a separate Python 3.12
environment. The `uma` and `proteins-esmfold2` extras have separate dependency
stacks and are intentionally not part of the root installation.

## Evaluate structures

```bash
zevals evaluate --modality molecules=outputs/molecules --results-dir outputs/metrics
zevals evaluate --modality materials=outputs/materials --results-dir outputs/metrics
zevals evaluate --modality proteins=outputs/backbones \
  --override proteins:oracle/inverse_folder=soluble_mpnn \
  --override proteins:oracle/structure_predictor=esmfold2 \
  --override proteins:pipeline.sc_rmsd_threshold=1.5 \
  --results-dir outputs/metrics
```

Molecule inputs use SDF/SMILES; material inputs use CIF; proteins and RNA accept
PDB/CIF. Reference conformer ensembles use matching `<target_id>/*.sdf`
directories passed with `--ground-truth molecules=references/conformers`.
See [`configs/`](configs/) for supported parameters and default protocols.

The generic protein preset uses ESMFold v1 and a 2 Å threshold. The Zatom-2 paper
uses SolubleMPNN, ESMFold2-Fast, eight sequences/backbone, a 1.5 Å C-alpha RMSD
threshold, and no pLDDT gate. Follow the
[paper protein protocol](../../docs/protein_evaluation.md) for full reproduction.
Do not treat generic defaults as manuscript settings.

```python
from zevals.pipeline.suite import ZevalSuite

suite = ZevalSuite()
metrics = suite.evaluate({"molecules": "outputs/molecules"})
```

Construct a suite once when integrating it with training validation. Scores are
returned with domain prefixes. Unavailable optional oracles produce explicit
metric status; inspect coverage and failures before reporting aggregate results.

## External oracles and references

```bash
zevals setup --tools foldseek,mmseqs,usalign
zevals setup --checkpoints soluble_mpnn
```

Setup downloads binaries/weights only when explicitly invoked. Large reference
databases are a separate step, for example:

```bash
zevals setup --tools foldseek --build-db PDB --db-dir /path/to/foldseek-databases
```

- [AFD guide](docs/AFD-guide.md): CT checkpoints, reference features, sample counts.
- [UMA guide](docs/UMA-guide.md): gated UMA weights, task heads, force metrics.
- [LeMat-GenBench](docs/LeMat-GenBench-guide.md): pinned separate environment,
  data revisions, energy-hull protocol, and upstream provenance limitations.
- [CHGNet pre-relaxation](docs/CHGNet-prerelaxation.md): optional relaxed MP20 path.
- [Protein metric definitions](src/zevals/proteins/METRICS.md).

The protein oracle source revisions are pinned in `pyproject.toml`; ESMFold2-Fast
weights use revision `c6c7958d63f5f2f1f0fed0bb9462316f8ccceea6`. RNA's working default
is Boltz-2 plus NA-MPNN and USalign; its executables and weights require their own
setup. Unfinished oracle stubs and unreleased model-specific benchmarks are
excluded from this release. Third-party licenses are retained with vendored code.
