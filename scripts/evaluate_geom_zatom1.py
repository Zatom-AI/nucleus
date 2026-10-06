#!/usr/bin/env python3
"""Score saved GEOM-Drugs CIFs with the unmodified released Zatom-1 evaluator.

Run inside the Zatom-1 chemistry environment on a CPU allocation. Predictions
are matched to the inference JSON, so extra/unzipped CIFs cannot be double-counted.
Chunks save per-sample checks for restart and exact global aggregation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "0d4676571d9958e1bc072fe57144451474d2e3ac"
UPSTREAM_SHA256 = "0bbb6ee104a9ea20abe9cc6540e6f4832f4776c3ff4eb0eb1bc7adf1d6cd8b1e"
VERSIONS = {
    "rdkit": "2025.3.5",
    "openbabel-wheel": "3.1.1.22",
    "posebusters": "0.5.1",
    "pymatgen": "2024.11.13",
}
PB_KEYS = (
    "mol_pred_loaded",
    "sanitization",
    "inchi_convertible",
    "all_atoms_connected",
    "bond_lengths",
    "bond_angles",
    "internal_steric_clash",
    "aromatic_ring_flatness",
    "double_bond_flatness",
    "internal_energy",
    "posebusters_rate",
)


def sha256(path: Path) -> str:
    """Hash an artifact without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def initialize_backend(zatom_root: str) -> None:
    """Pin the chemistry versions and evaluator before interpreting validity."""
    os.environ["DEBUG"] = "false"
    for name, expected in VERSIONS.items():
        actual = metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"Requires {name}=={expected}; found {actual}")
    sys.path.insert(0, zatom_root)
    from zatom.eval import molecule_generation

    if sha256(Path(molecule_generation.__file__)) != UPSTREAM_SHA256:
        raise RuntimeError("The imported evaluator differs from the pinned release")
    global EVALUATOR
    EVALUATOR = molecule_generation.MoleculeGenerationEvaluator


class RecordingBuster:
    """Retain upstream's full result frame without changing its checks."""

    def __init__(self, buster):
        self.buster = buster
        self.frame = None

    def bust(self, *args, **kwargs):
        self.frame = self.buster.bust(*args, **kwargs)
        return self.frame


def score_chunk(task):
    """Run the released evaluator, including its PDB writer and hydrogen policy."""
    import gemmi
    import numpy as np
    import pandas as pd

    index, samples, output_root = task
    root = Path(output_root)
    chunk_path = root / "chunks" / f"chunk_{index:05d}.json"
    if chunk_path.exists():
        saved = json.loads(chunk_path.read_text())
        if [r["source"] for r in saved] != samples:
            raise RuntimeError(f"Chunk inputs changed: {chunk_path}")
        return index, len(saved), sum(r["valid"] for r in saved)

    records = []
    readable_records = []
    evaluator = EVALUATOR(dataset_smiles_list=None, removeHs=False, device="cpu")
    recorder = RecordingBuster(evaluator.buster)
    evaluator.buster = recorder
    for sample in samples:
        row = {"source": sample, "valid": False, "smiles": None, "checks": {}}
        records.append(row)
        try:
            path = Path(sample["path"])
            if sha256(path) != sample["sha256"]:
                raise RuntimeError(
                    f"Prediction changed after manifest creation: {path}"
                )
            block = gemmi.cif.read(str(path)).sole_block()
            symbols = list(block.find_values("_atom_site.type_symbol"))
            numbers = np.array([gemmi.Element(s).atomic_number for s in symbols])
            positions = np.array(
                [list(block.find_values(f"_atom_site.Cartn_{axis}")) for axis in "xyz"],
                dtype=float,
            ).T
            models = set(block.find_values("_atom_site.pdbx_PDB_model_num"))
            if (
                len(numbers) != sample["num_atoms"]
                or positions.shape != (len(numbers), 3)
                or not np.isfinite(positions).all()
                or not ((numbers >= 1) & (numbers <= 118)).all()
                or len(models) != 1
            ):
                raise ValueError("Invalid CIF coordinates, atom count, or model count")
        except (ValueError, OSError) as error:
            row["read_error"] = str(error)
            continue
        evaluator.append_pred_array(
            {"atom_types": numbers, "pos": positions, "sample_idx": sample["index"]}
        )
        readable_records.append(row)

    if readable_records:
        # One process per chunk; get_metrics uses its own sequential conversion.
        evaluator.get_metrics(
            save=True,
            save_dir=str(root / "pdb" / f"chunk_{index:05d}"),
            n_jobs=1,
            visualize=False,
        )
        valid_index = 0
        for row, prediction in zip(readable_records, evaluator.pred_rdkit_list):
            _, smiles, _, valid, _ = prediction
            row["valid"] = bool(valid)
            if valid:
                row["smiles"] = smiles
                checks = recorder.frame.iloc[valid_index]
                missing = set(PB_KEYS) - set(checks.index)
                if missing:
                    raise RuntimeError(f"PoseBusters omitted checks: {missing}")
                row["checks"] = {
                    key: None if pd.isna(checks[key]) else bool(checks[key])
                    for key in PB_KEYS
                }
                valid_index += 1
        if valid_index and len(recorder.frame) != valid_index:
            raise RuntimeError("PoseBusters row count does not match valid molecules")
    write_json(chunk_path, records)
    return index, len(records), sum(row["valid"] for row in records)


def aggregate(records, reference_smiles):
    """Aggregate counts with upstream's global uniqueness and native NaN policy."""
    valid = [row for row in records if row["valid"]]
    smiles = [row["smiles"] for row in valid]
    unique = set(smiles)
    novel = unique - set(reference_smiles)
    numerators = {
        "valid_rate": len(valid),
        "unique_rate": len(unique),
        "novel_rate": len(novel),
    }
    denominators = {
        "valid_rate": len(records),
        "unique_rate": len(valid),
        "novel_rate": len(valid),
    }
    missing_checks = {}
    for key in PB_KEYS:
        observed = [
            row["checks"][key] for row in valid if row["checks"][key] is not None
        ]
        numerators[key] = sum(observed)
        denominators[key] = len(observed)
        missing_checks[key] = len(valid) - len(observed)
    numerators["posebusters_valid_over_all_generated"] = numerators["posebusters_rate"]
    denominators["posebusters_valid_over_all_generated"] = len(records)
    rates = {key: numerators[key] / n if n else None for key, n in denominators.items()}
    return {
        "counts": {
            "generated": len(records),
            "read_errors": sum("read_error" in r for r in records),
            "valid": len(valid),
            "unique": len(unique),
            "novel_unique": len(novel),
            "posebusters_valid": numerators["posebusters_rate"],
            "posebusters_missing_checks": missing_checks,
        },
        "numerators": numerators,
        "denominators": denominators,
        "rates": rates,
        "percent": {key: None if v is None else 100 * v for key, v in rates.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument(
        "--input-json",
        type=Path,
        default=REPO
        / "models/zatom2/docs/examples/geom_drugs_unconditional_10000.json",
    )
    parser.add_argument("--zatom-root", type=Path, default=REPO.parent / "zatom")
    parser.add_argument("--reference-smiles", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_size < 1:
        parser.error("workers and chunk-size must be positive")
    output = (args.output_dir or args.generated_dir / "zatom1_evaluation").resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "chunks").mkdir(exist_ok=True)
    reference_path = (
        args.reference_smiles or args.zatom_root / "data/geom/smiles.pt"
    ).resolve()
    initialize_backend(str(args.zatom_root.resolve()))
    import torch
    from posebusters import PoseBusters

    requests = json.loads(args.input_json.read_text())
    samples = []
    checkpoints = set()
    for index, (key, spec) in enumerate(requests.items()):
        name = f"{args.input_json.stem}_{key}_0_model_0"
        path = (args.generated_dir / f"{name}.cif.gz").resolve()
        meta_path = path.with_suffix("").with_suffix(".json")
        metadata_json = json.loads(meta_path.read_text())
        checkpoints.add(metadata_json["ckpt_path"])
        samples.append(
            {
                "index": index,
                "name": name,
                "num_atoms": int(spec["length"][:-1]),
                "path": str(path),
                "sha256": sha256(path),
            }
        )
    if len(checkpoints) != 1:
        raise ValueError("The inputs contain predictions from multiple checkpoints")
    config = PoseBusters(config="mol").config
    manifest = {
        "protocol": "Zatom-1 GEOM-Drugs Table 3, released molecule evaluator",
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_source_sha256": UPSTREAM_SHA256,
        "script_sha256": sha256(Path(__file__)),
        "input_json": str(args.input_json.resolve()),
        "input_json_sha256": sha256(args.input_json),
        "checkpoint": next(iter(checkpoints)),
        "package_versions": {
            p: metadata.version(p) for p in (*VERSIONS, "numpy", "pandas", "gemmi")
        },
        "reference_smiles": str(reference_path),
        "reference_smiles_sha256": sha256(reference_path),
        "pdb_remove_hydrogens": False,
        "conversion": "pymatgen Molecule.to(PDB) -> RDKit MolFromPDBFile",
        "predicted_geometry_optimized": False,
        "validity_and_uniqueness_fragment_policy": "largest_fragment_smiles",
        "posebusters_fragment_policy": "full_valid_molecule",
        "posebusters_missing_policy": "pandas mean/all with skipna=True (upstream)",
        "posebusters_config": config,
        "novelty_definition": "unique valid SMILES absent from reference / valid samples",
        "chunk_size": args.chunk_size,
        "samples": samples,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(
                "Existing evaluation inputs or protocol changed; use a new output directory"
            )
    else:
        write_json(manifest_path, manifest)
    tasks = [
        (i // args.chunk_size, samples[i : i + args.chunk_size], str(output))
        for i in range(0, len(samples), args.chunk_size)
    ]
    started = time.monotonic()
    completed = 0
    valid = 0
    print(
        f"Scoring {len(samples)} samples in {len(tasks)} chunks with {args.workers} workers",
        flush=True,
    )
    if args.workers == 1:
        for task in tasks:
            _, n, v = score_chunk(task)
            completed += n
            valid += v
            print(
                f"Completed {completed}/{len(samples)}; valid {valid}; elapsed {time.monotonic() - started:.1f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("fork")
        ) as pool:
            for future in as_completed(
                [pool.submit(score_chunk, task) for task in tasks]
            ):
                _, n, v = future.result()
                completed += n
                valid += v
                print(
                    f"Completed {completed}/{len(samples)}; valid {valid}; elapsed {time.monotonic() - started:.1f}s",
                    flush=True,
                )
    records = [
        row
        for task in tasks
        for row in json.loads(
            (output / "chunks" / f"chunk_{task[0]:05d}.json").read_text()
        )
    ]
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    result = aggregate(records, reference)
    result["protocol_manifest"] = str(manifest_path)
    result["protocol_manifest_sha256"] = sha256(manifest_path)
    result["reference_smiles_count"] = len(reference)
    result["elapsed_seconds"] = time.monotonic() - started
    write_json(output / "metrics.json", result)
    with (output / "per_sample.jsonl").open("w") as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "numerator", "denominator", "rate", "percent"])
        for key, rate in result["rates"].items():
            writer.writerow(
                [
                    key,
                    result["numerators"][key],
                    result["denominators"][key],
                    rate,
                    result["percent"][key],
                ]
            )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
