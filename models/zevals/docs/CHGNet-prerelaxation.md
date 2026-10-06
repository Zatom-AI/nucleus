# Optional CHGNet pre-relaxation

This adapter follows the [released FlowMM pre-relaxation protocol](https://github.com/facebookresearch/flowmm#prerelaxation):
CHGNet 0.3.1, bundled model 0.3.0, FIRE, force tolerance 0.1 eV/Å,
cell relaxation, and a maximum of 1,500 steps. Its historical ASE 3.22.1
`ExpCellFilter` is explicit; it is not silently replaced with the newer
`FrechetCellFilter`. CPU and CUDA are supported, with CUDA used by the
Table 2 evaluation. Pre-relaxed scores are a separate protocol variant
from scores on raw generated structures.

Set up a separate Python 3.11 environment on a login node:

```bash
bash models/zevals/scripts/setup_chgnet_prerelax.sh /absolute/path/to/chgnet-env
export ZEVALS_CHGNET_PYTHON=/absolute/path/to/chgnet-env/bin/python
```

The setup pins the primary scientific packages, limits transitive dependency
releases to before 2024-06-01 for this historical stack, and writes a fully
resolved, hash-locked `requirements.lock` in that environment. Archive it to
recreate the same transitive dependencies. CHGNet's checkpoint ships in its
package; the worker verifies it against the package's wheel RECORD and the
explicit SHA-256
`d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1`.
The LeMat and training environments are unaffected.

The reusable API is:

```python
from zevals.materials.chgnet_prerelax import run_chgnet_prerelaxation

result = run_chgnet_prerelaxation(
    "generated/cifs",
    output_dir="evaluation/chgnet",
    runner_command=["/absolute/path/to/chgnet-env/bin/python"],
    device="cuda",
    resume=True,
)
if result.complete:
    # Pass result.cif_dir to run_lematgenbench(...).
    print(result.manifest_path)
```

Both `.cif` and `.cif.gz` inputs are supported. Every source filename has
exactly one output; relaxed gzip files use a fixed zero timestamp.
Generated AtomWorks/PDBx mmCIF tags (for example, `_atom_site.fract_x` and
`_atom_site.label_atom_id`) are converted in memory to the equivalent materials
CIF tags before parsing. Cell parameters, fractional coordinates, species,
occupancies, and atom-site rows are preserved; Cartesian coordinates are not
used to reconstruct the fractional positions. Original sample bytes and hashes
are never changed. The shared conversion implementation is hashed into the
worker contract, and CHGNet adapter protocol version 2 rejects results from the
earlier parser that treated these mmCIFs as invalid.
Parse or scientific relaxation
errors keep the original CIF bytes and record an explicit fallback status;
unconverged but finite relaxations retain their final structure. All attempts
therefore remain in the scoring denominator. CUDA, memory, and filesystem
errors stop the worker and preserve completed work for resumption.

Each final CIF is written atomically, followed by an atomic JSON completion
record. A timeout or SIGTERM loses at most the current CIF's relaxation;
the next run verifies hashes and skips completed CIFs. An unfinished output
without its completion record is recomputed. Completed numerical work is
never accepted after changes to input hashes, configuration, worker code,
checkpoint, installed package versions/source hashes, or recorded device
properties. `validate_chgnet_prerelaxation` performs read-only validation
without requiring CHGNet or a GPU. It checks saved provenance; starting new
numerical work additionally verifies the active runtime.

Artifacts are `cifs/`, `records/<input-name>.json`,
`chgnet_prerelax.manifest.json`, and append-only worker stdout/stderr logs.
Records contain source/output hashes, status, actual ASE convergence and
optimizer steps, trajectory frame count, initial/final energies (eV and
eV/atom), and final atomic/filter forces. Actual ASE convergence includes
cell forces and is more precise than FlowMM's trajectory-length heuristic.
Per-filename seeds and deterministic PyTorch algorithms are fixed; the
runtime records CUDA and GPU properties. Bitwise agreement across different
hardware or software environments is not assumed.

## Recovering the legacy Table 2 mmCIF parse failures

An earlier adapter passed generated mmCIF directly to pymatgen's materials CIF
reader. Runs affected by this bug have all entries recorded as
`fallback_parse_error` with `ValueError: Invalid CIF file with no structures!`,
and worker warnings mention `_atom_site_label`. A completed manifest in that
state means every attempt was processed, not that any structure was relaxed.
Do not reuse those fallback results after updating the converter.

For the specific Table 2 case where sampling and QM9 scoring completed but
LeMat stopped during preflight, follow the
[guarded recovery script](../scripts/archive_table2_cif_fallbacks.py). It archives only
the affected scoring stages, keeping raw samples, protocol assets, QM9 metrics,
and the preserved checkpoint in place. Merely changing a cache environment
variable does not invalidate the old CHGNet fallback records.
