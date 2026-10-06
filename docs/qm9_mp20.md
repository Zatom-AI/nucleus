# QM9 And MP20 ASE-LMDB Preparation

This guide prepares QM9 and MP20 for the Zatom-2 paper Table 2 experiment.
The training configs expect directories containing ASE database-compatible
`*.aselmdb` shards, not the PyTorch Geometric `*.pt` files used by Zatom-1.

Run all commands from the Nucleus repository root. Install the release
environment first if needed:

```bash
uv sync --locked --extra all
source .venv/bin/activate
```

## Canonical Benchmark Splits

The released [Zatom-1 data module][zatom-datamodule] contains 130,831 QM9
molecules and 45,229 MP20 structures. Its generative benchmark splits are:

| Dataset | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| QM9 | 100,000 | 18,000 | 12,831 | 130,831 |
| MP20 | 27,138 | 9,046 | 9,045 | 45,229 |

Table 2 uses exactly the Zatom-1 training cardinalities: 100,000 QM9 molecules
and 27,138 MP20 structures. The MP20 validation and test partitions are never
included in training. The converter records the source-artifact SHA-256,
selection rule, seed, record count, shard list, and selected-index SHA-256 in
`manifest.json`.

## Download The Zatom-1 Source Artifacts

The minimal route downloads the official preprocessed artifacts from the
[Zatom-AI QM9][zatom-qm9-hf] and [Zatom-AI MP20][zatom-mp20-hf] Hugging Face
dataset repositories. These are serialized PyTorch Geometric datasets; only
load artifacts obtained from a trusted source.

```bash
export ZATOM1_DATA_ROOT=/path/to/zatom1_sources

mkdir -p "$ZATOM1_DATA_ROOT/qm9"
mkdir -p "$ZATOM1_DATA_ROOT/mp_20"

hf download Zatom-AI/qm9 processed.tar.gz \
  --type dataset \
  --revision fbf62fcfdcd61960160b04c3056d161ebb62724e \
  --local-dir "$ZATOM1_DATA_ROOT/qm9"
hf download Zatom-AI/mp_20 processed.tar.gz \
  --type dataset \
  --revision 9d9e762a5590105b591e237cf3bc61bd5b671890 \
  --local-dir "$ZATOM1_DATA_ROOT/mp_20"

tar -C "$ZATOM1_DATA_ROOT/qm9" \
  -xzf "$ZATOM1_DATA_ROOT/qm9/processed.tar.gz"
tar -C "$ZATOM1_DATA_ROOT/mp_20" \
  -xzf "$ZATOM1_DATA_ROOT/mp_20/processed.tar.gz"

test -f "$ZATOM1_DATA_ROOT/qm9/data_v3.pt"
test -f "$ZATOM1_DATA_ROOT/mp_20/mp20.pt"
```

The `processed.tar.gz` archives extract their files directly into these dataset
roots. The converter also accepts the `processed/data_v3.pt` and
`processed/mp20.pt` layout produced by some Zatom-1 setup paths.

Alternatively, use the [official Zatom-1 downloader][zatom-downloader] from an
installed Zatom-1 checkout. Note that the accepted MP20 dataset name is
`mp_20`, with an underscore:

```bash
cd /path/to/zatom
python zatom/data/download_data.py --dataset qm9
python zatom/data/download_data.py --dataset mp_20
```

That route creates `data/qm9` and `data/mp_20` in the Zatom-1 checkout and also
downloads raw artifacts not needed by the LMDB converter.

## Prepare The Table 2 Training Splits

`scripts/prepare_qm9_mp20_lmdb.py` preserves hydrogens in QM9, stores MP20 as
fully periodic ASE `Atoms` objects, and retains MP20 space-group metadata. It
writes new shards through the same ASE-LMDB backend read by AtomWorks and then
performs a read-back cardinality check.

QM9 is shuffled with a dedicated PyTorch generator seeded with 42 before the
first 100,000 records are selected. MP20 retains source order, matching the
ordering on which Zatom-1 defines its split boundaries.

```bash
python scripts/prepare_qm9_mp20_lmdb.py qm9 \
  --source-root "$ZATOM1_DATA_ROOT/qm9" \
  --output-dir data/qm9/raw/train_100k \
  --num-examples 100000 \
  --seed 42 \
  --shard-size 10000

python scripts/prepare_qm9_mp20_lmdb.py mp20 \
  --source-root "$ZATOM1_DATA_ROOT/mp_20" \
  --output-dir data/mp_20/raw/train_27138 \
  --num-examples 27138 \
  --shard-size 10000
```

The converter refuses to write into a directory that already contains shards,
a manifest, or a partial shard. This prevents an interrupted or changed
selection from being mixed silently with an earlier conversion. Inspect any
existing artifacts and choose a new output directory when rerunning with a
different split.

The Table 2 epoch sampling budget is 127,138 examples, the exact sum of the two
training-set cardinalities. Dataset probabilities remain half-and-half, so the
mixed sampler draws each domain with equal probability and may resample source
records within an epoch.

## Validate The AtomWorks Contract

The converter already reopens all output shards with `ASELMDBDataset`. The
following independent smoke test also applies the same loaders selected by the
Table 2 Hydra configs:

```bash
export QM9_DATA_DIR="$(pwd)/data/qm9/raw/train_100k"
export MP20_DATA_DIR="$(pwd)/data/mp_20/raw/train_27138"

python - <<'PY'
import os

from atomworks.ml.datasets import ASELMDBDataset
from atomworks.ml.datasets.loaders import (
    create_ase_atoms_loader,
    create_ase_materials_loader,
)

with ASELMDBDataset(
    paths=os.environ["QM9_DATA_DIR"],
    name="qm9_smoke",
    loader=create_ase_atoms_loader(res_name="LIG"),
    readonly=True,
    readahead=False,
) as qm9:
    assert len(qm9) == 100_000
    assert "atom_array" in qm9[0]
    assert not qm9[0]["atoms"].pbc.any()

with ASELMDBDataset(
    paths=os.environ["MP20_DATA_DIR"],
    name="mp20_smoke",
    loader=create_ase_materials_loader(
        res_name="LIG",
        wrap_fractional_coordinates=True,
        include_atom_array=True,
    ),
    readonly=True,
    readahead=False,
) as mp20:
    assert len(mp20) == 27_138
    record = mp20[0]
    assert record["atoms"].pbc.all()
    assert record["fractional_coordinates"].shape[1] == 3

print("QM9 and MP20 ASE-LMDB smoke checks passed.")
PY
```

## Run The Paper Experiment

The repository-local paths above are already the defaults in
`models/zatom2/configs/paths/data/default.yaml`. For data stored elsewhere,
export the two directories before launching:

```bash
export QM9_DATA_DIR=/path/to/qm9/train_100k
export MP20_DATA_DIR=/path/to/mp_20/train_27138

python scripts/train_paper.py paper-table2-qm9-mp20-unconditional-without-force-conditioning --devices 4
```

The Hydra experiment samples QM9 and MP20 with equal probability. It has no
in-training validation loader; generate samples from the trained checkpoint and
evaluate them with the paper's QM9 and LeMat-GenBench metric groups. The
[Zatom-2 Table 2 evaluation guide](../models/zatom2/docs/table2_evaluation.md)
documents the frozen sampler, isolated LeMat setup, checkpoint command, metric
denominators, and reusable protocol artifacts.

[zatom-datamodule]: https://github.com/Zatom-AI/zatom/blob/main/zatom/data/joint_datamodule.py
[zatom-downloader]: https://github.com/Zatom-AI/zatom/blob/main/zatom/data/download_data.py
[zatom-qm9-hf]: https://huggingface.co/datasets/Zatom-AI/qm9
[zatom-mp20-hf]: https://huggingface.co/datasets/Zatom-AI/mp_20
