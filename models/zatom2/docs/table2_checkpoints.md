# Download the Table 2 LeMat-GenBench checkpoints

Run these commands from the Nucleus repository root with network access, before
starting evaluation. They populate the caches used by the pinned LeMat
environment without loading models onto a GPU or changing installed packages.
The model identities and hashes below are enforced by the
[adapter](../../zevals/src/zevals/materials/lematgenbench.py).

## Environment and cache paths

First prepare the isolated LeMat environment as described in the
[Table 2 guide](table2_evaluation.md). Set its actual installation path:

```bash
export ZEVALS_LEMAT_GENBENCH_REPO="/path/to/lemat-genbench-table2"
export LEMAT_PYTHON="$ZEVALS_LEMAT_GENBENCH_REPO/.venv/bin/python"
export CACHED_PATH_CACHE_ROOT="$HOME/.cache/cached_path-table2"
mkdir -p "$CACHED_PATH_CACHE_ROOT"
```

Keep these variables exported when launching evaluation. On shared clusters,
choose persistent cache directories that support file locking and are accessible
to all evaluation workers. Optional `HF_DATASETS_CACHE`, `HF_HUB_CACHE`, and
`HF_XET_CACHE` variables relocate the benchmark reference caches.

The pinned MACE loader uses `~/.cache/mace`; the pinned FAIRChem loader passes
`~/.cache/fairchem` explicitly to Hugging Face. `HF_HUB_CACHE` does not redirect
that FAIRChem argument. Prepare these caches before running offline workers.

## ORB

Use `cached_path`, rather than downloading to an arbitrary filename, to create
the URL/ETag cache layout that the ORB loader expects:

```bash
"$LEMAT_PYTHON" - <<'PY'
import hashlib
from pathlib import Path
from cached_path import cached_path

url = (
    "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/"
    "forcefields/orb-v3/orb-v3-conservative-inf-omat-20250404.ckpt"
)
path = Path(cached_path(url))
with path.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
expected = "0a41ef1132ad9c41ee0c7b9d855fccabfefb97abe04625c97ec0d07930b6c0f0"
if digest != expected:
    raise RuntimeError(f"ORB checkpoint hash mismatch: {digest}")
print(f"ORB verified: {path} ({path.stat().st_size} bytes)")
PY
```

This is `orb_v3_conservative_inf_omat`, approximately 102 MB.

## MACE

Select `medium-mpa-0` explicitly; `medium` identifies a different checkpoint.
The loader reuses an existing download when available:

```bash
"$LEMAT_PYTHON" - <<'PY'
import hashlib
from pathlib import Path
from mace.calculators.foundations_models import download_mace_mp_checkpoint

path = Path(download_mace_mp_checkpoint("medium-mpa-0"))
with path.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
expected = "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638"
if digest != expected:
    raise RuntimeError(f"MACE checkpoint hash mismatch: {digest}")
print(f"MACE verified: {path} ({path.stat().st_size} bytes)")
PY
```

The checkpoint is approximately 79 MB. Its usual cache filename is
`~/.cache/mace/macempa0mediummodel`.

## UMA and its atomic reference file

The account must have access to the gated
[`facebook/UMA`](https://huggingface.co/facebook/UMA) repository. Use the working
Hugging Face CLI in Nucleus, rather than an unrelated `hf` on `PATH`:

```bash
.venv/bin/hf auth whoami
# If needed, authenticate interactively with: .venv/bin/hf auth login

.venv/bin/hf download facebook/UMA checkpoints/uma-s-1.pt \
    --revision f611b917d9c68566bbbeccbb0aa0f7cad1696cb2 \
    --cache-dir "$HOME/.cache/fairchem"

.venv/bin/hf download facebook/UMA references/iso_atom_elem_refs.yaml \
    --revision f611b917d9c68566bbbeccbb0aa0f7cad1696cb2 \
    --cache-dir "$HOME/.cache/fairchem"
```

Both files are required by FAIRChem's `uma-s-1` loader. The Table 2 adapter pins
its Hugging Face requests to this revision and uses the `omat` task. Do not
substitute `main` or a newer UMA model. Verify the download from the isolated
environment, using only local files:

```bash
"$LEMAT_PYTHON" - <<'PY'
import hashlib
from pathlib import Path
from huggingface_hub import hf_hub_download

revision = "f611b917d9c68566bbbeccbb0aa0f7cad1696cb2"
for filename in ("checkpoints/uma-s-1.pt", "references/iso_atom_elem_refs.yaml"):
    path = Path(hf_hub_download(
        "facebook/UMA", filename=filename, revision=revision,
        cache_dir=str(Path.home() / ".cache/fairchem"), local_files_only=True,
    ))
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if filename.endswith(".pt"):
        expected = "02f83e20547294d3d47dd9dfbe4a4ee51eebca21b2e1febe63210e55b764cc78"
        if digest != expected:
            raise RuntimeError(f"UMA checkpoint hash mismatch: {digest}")
    print(f"{filename}: {path} ({path.stat().st_size} bytes), sha256={digest}")
PY
```

## Use the prepared caches

Restart a worker that was launched before setting `CACHED_PATH_CACHE_ROOT`;
exporting it in another shell cannot change the running worker's environment.
Keep the same checkpoint identities, scoring configuration, and output
directory to reuse completed CHGNet/LeMat progress.

These steps prepare the three model checkpoints and UMA's atomic references.
They do not preload the separate LeMat benchmark datasets, hull references, or
embedding references. Do not enable global Hugging Face offline mode until
those required reference datasets are also cached at their pinned revisions.
The [LeMat guide](../../zevals/docs/LeMat-GenBench-guide.md) lists those revisions.
