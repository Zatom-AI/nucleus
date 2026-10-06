#!/usr/bin/env python3
"""Supplement Zatom-1's released metrics with documented Table 15 definitions.

The Zatom-1 release does not contain Table 15's extra metric implementation.
This script uses the cited TABASCO definitions and pinned PoseCheck strain
source, and labels the results as a reconstruction rather than exact reproduction.
It consumes the PDBs and validity decisions from evaluate_geom_zatom1.py.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import math
import multiprocessing
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

from evaluate_geom_zatom1 import REPO, VERSIONS, sha256, write_json

POSECHECK_COMMIT = "56799eb242dbeb0df94076a9a36774613e3010b1"
POSECHECK_SHA256 = "c2ecb52c8b0066b0ada403df5bcfbe6c8ccb3b173d86e9e6abc73af93223d423"
TABASCO_COMMIT = "097eaae106e90a88294a93d42a3f91f1798f383f"


def initialize(root):
    root = Path(root)
    sys.path.insert(0, str(root / "chemistry_overlay"))
    for name, expected in (VERSIONS | {"datamol": "0.12.5"}).items():
        if metadata.version(name) != expected:
            raise RuntimeError(f"Requires {name}=={expected}")
    path = root / "sources/posecheck_strain.py"
    if sha256(path) != POSECHECK_SHA256:
        raise RuntimeError("PoseCheck source does not match the pinned commit")
    spec = importlib.util.spec_from_file_location("pinned_posecheck_strain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    global POSECHECK
    POSECHECK = module
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")


def seeded_strain(mol, seed, num_confs=50):
    """Supply explicit embedding seeds; leave PoseCheck's energy method intact."""
    from rdkit.Chem import AllChem

    rng = random.Random(seed)

    def embed(*args, **kwargs):
        kwargs["randomSeed"] = rng.randrange(1, 2**31 - 1)
        return AllChem.EmbedMolecule(*args, **kwargs)

    # The pinned source uses only these two AllChem functions. A module-local
    # proxy avoids altering RDKit globally or changing the archived source.
    POSECHECK.AllChem = SimpleNamespace(
        EmbedMolecule=embed, UFFOptimizeMolecule=AllChem.UFFOptimizeMolecule
    )
    return POSECHECK.calculate_strain_energy(mol, maxDispl=0.1, num_confs=num_confs)


def score_sample(task):
    import datamol as dm
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Crippen, Descriptors, Lipinski

    record, root, chunk_size, seed = task
    index = record["source"]["index"]
    result_path = Path(root) / "supplement_samples" / f"sample_{index:05d}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    path = (
        Path(root)
        / "pdb"
        / f"chunk_{index // chunk_size:05d}"
        / f"molecule_{index}.pdb"
    )
    mol = Chem.MolFromPDBFile(str(path), removeHs=False)
    if mol is None:
        raise ValueError(f"Previously valid molecule cannot be reread: {path}")
    largest = max(
        Chem.GetMolFrags(mol, asMols=True), key=lambda frag: frag.GetNumAtoms()
    )
    if Chem.MolToSmiles(largest, isomericSmiles=True) != record["smiles"]:
        raise ValueError(f"Molecule changed after core scoring: {path}")
    sample_seed = seed + index
    logp = Crippen.MolLogP(largest)
    lipinski = sum(
        [
            Descriptors.ExactMolWt(largest) < 500,
            Lipinski.NumHDonors(largest) <= 5,
            Lipinski.NumHAcceptors(largest) <= 10,
            -2 <= logp <= 5,
            Chem.rdMolDescriptors.CalcNumRotatableBonds(largest) <= 10,
        ]
    )
    fp = dm.to_fp(mol, fp_type="ecfp", fpSize=2048, as_array=False)
    energy = seeded_strain(mol, sample_seed)
    result = {
        "index": index,
        "name": record["source"]["name"],
        "pdb_sha256": sha256(path),
        "smiles": record["smiles"],
        "qed": Descriptors.qed(largest),
        "logp": logp,
        "lipinski": lipinski,
        "strain_energy_kcal_mol": float(energy)
        if energy is not None and math.isfinite(energy)
        else None,
        "strain_random_seed": sample_seed,
        "full_molecule_ecfp6_2048": base64.b64encode(
            DataStructs.BitVectToBinaryText(fp)
        ).decode("ascii"),
    }
    write_json(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument(
        "--training-smiles", type=Path, default=REPO / "data/geom/train_smiles.pt"
    )
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    root = args.evaluation_dir.resolve()
    initialize(str(root))
    import datamol
    import numpy as np
    import torch
    from rdkit import DataStructs

    core_manifest = json.loads((root / "manifest.json").read_text())
    records = [
        json.loads(line)
        for line in (root / "per_sample.jsonl").read_text().splitlines()
    ]
    core_metrics = json.loads((root / "metrics.json").read_text())
    if len(records) != len(core_manifest["samples"]) or core_metrics["counts"][
        "generated"
    ] != len(records):
        raise ValueError("Core evaluation is incomplete")
    valid = [record for record in records if record["valid"]]
    protocol = {
        "status": "supplementary reconstruction; not a released Zatom-1 Table 15 implementation",
        "core_manifest_sha256": sha256(root / "manifest.json"),
        "core_per_sample_sha256": sha256(root / "per_sample.jsonl"),
        "script_sha256": sha256(Path(__file__)),
        "tabasco_commit": TABASCO_COMMIT,
        "tabasco_source_sha256": sha256(root / "sources/tabasco_metrics.py"),
        "posecheck_commit": POSECHECK_COMMIT,
        "posecheck_strain_source_sha256": POSECHECK_SHA256,
        "datamol_version": datamol.__version__,
        "chemistry_versions": VERSIONS,
        "training_smiles_path": str(args.training_smiles.resolve()),
        "training_smiles_sha256": sha256(args.training_smiles),
        "property_fragment_policy": "largest fragment, explicit H retained",
        "diversity": "mean pairwise Jaccard/Tanimoto distance of full valid molecule ECFP6 fingerprints; 2048 bits, radius 3, no chirality, no deduplication",
        "novelty": "raw isomeric largest-fragment SMILES compared to official training SMILES; both sample and unique denominators reported",
        "lipinski": "sum of ExactMolWt<500, NumHDonors<=5, NumHAcceptors<=10, -2<=Crippen logP<=5, rotatable bonds<=10",
        "strain": {
            "forcefield": "UFF",
            "max_displacement_angstrom": 0.1,
            "num_conformers": 50,
            "fragment_policy": "full valid molecule",
            "seed": args.seed,
            "seed_policy": "EmbedMolecule receives explicit randomSeed values from Python Random(seed + sample index), drawn uniformly from [1, 2**31 - 1)",
            "source_adjustment": "module-local AllChem proxy supplies explicit embedding seeds; default upstream EmbedMolecule is stochastic",
            "missing_policy": "exclude None/nonfinite, report counts",
            "aggregates": ["mean", "median", "torch_lower_median"],
        },
    }
    manifest_path = root / "supplement_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != protocol:
        raise ValueError(
            "Supplement protocol changed; use a fresh evaluation directory"
        )
    write_json(manifest_path, protocol)
    (root / "supplement_samples").mkdir(exist_ok=True)
    tasks = [
        (record, str(root), core_manifest["chunk_size"], args.seed) for record in valid
    ]
    started = time.monotonic()
    results = []
    print(
        f"Scoring appendix properties and 50-conformer strain for {len(tasks)} valid molecules",
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("fork")
    ) as pool:
        for future in as_completed([pool.submit(score_sample, task) for task in tasks]):
            results.append(future.result())
            if len(results) % 100 == 0 or len(results) == len(tasks):
                print(
                    f"Supplement {len(results)}/{len(tasks)} complete; {time.monotonic() - started:.1f}s",
                    flush=True,
                )
    results.sort(key=lambda result: result["index"])
    fps = [
        DataStructs.CreateFromBinaryText(
            base64.b64decode(row["full_molecule_ecfp6_2048"])
        )
        for row in results
    ]
    pair_count = len(fps) * (len(fps) - 1) // 2
    similarity_sum = math.fsum(
        math.fsum(DataStructs.BulkTanimotoSimilarity(fp, fps[i + 1 :]))
        for i, fp in enumerate(fps[:-1])
    )
    reference = set(
        torch.load(args.training_smiles, weights_only=False, map_location="cpu")
    )
    smiles = [row["smiles"] for row in results]
    unique_smiles = set(smiles)
    novel_unique = unique_smiles - reference
    energies = [
        row["strain_energy_kcal_mol"]
        for row in results
        if row["strain_energy_kcal_mol"] is not None
    ]
    metrics = {
        "protocol_manifest": str(manifest_path),
        "counts": {
            "generated": len(records),
            "valid": len(results),
            "fingerprint_pairs": pair_count,
            "training_unique_smiles": len(reference),
            "novel_unique": len(novel_unique),
            "strain_evaluated": len(energies),
            "strain_missing": len(results) - len(energies),
        },
        "diversity": 1 - similarity_sum / pair_count if pair_count else None,
        "novel_fraction_of_valid_samples": sum(s not in reference for s in smiles)
        / len(smiles)
        if smiles
        else None,
        "novel_unique_over_valid_samples": len(novel_unique) / len(smiles)
        if smiles
        else None,
        "novel_fraction_of_unique_valid": len(novel_unique) / len(unique_smiles)
        if unique_smiles
        else None,
        "qed_mean": float(np.mean([row["qed"] for row in results]))
        if results
        else None,
        "logp_mean": float(np.mean([row["logp"] for row in results]))
        if results
        else None,
        "lipinski_mean": float(np.mean([row["lipinski"] for row in results]))
        if results
        else None,
        "strain_energy_mean_kcal_mol": float(np.mean(energies)) if energies else None,
        "strain_energy_median_kcal_mol": float(np.median(energies))
        if energies
        else None,
        "strain_energy_torch_lower_median_kcal_mol": sorted(energies)[
            (len(energies) - 1) // 2
        ]
        if energies
        else None,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(root / "supplement_metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
