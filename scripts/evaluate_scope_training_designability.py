#!/usr/bin/env python3
"""Evaluate every SCOPe training domain with the manuscript's best-of-8 metric.

Prepare coordinate-preserving, consecutively numbered, first-model backbones;
run independent resumable GPU shards; then summarize only a complete cohort.
The existing zevals oracles and self-consistency collector define the metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path


def write_json(path: Path, payload: object) -> None:
    """Atomically save progress so interrupted workers can be resumed."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def prepare(args: argparse.Namespace) -> None:
    """Preserve observed backbone coordinates and record source residue mapping."""
    import numpy as np
    from Bio.PDB import PDBIO, PDBParser
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Structure import Structure

    rows = list(csv.DictReader(args.scope_csv.open()))
    assert len(rows) == 2000 and len({r["domain_id"] for r in rows}) == 2000
    assert all(r["split"] == "train" for r in rows)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Manifest already exists; use run to resume")
    domains = []
    # Balance lengths across workers without depending on CSV order.
    rows.sort(key=lambda row: (-int(row["length"]), row["domain_id"]))
    for index, row in enumerate(rows):
        domain = row["domain_id"]
        source = args.pdb_dir / f"{domain}.pdb"
        original = PDBParser(QUIET=True).get_structure(domain, source)
        residues = [
            r for r in original[0].get_residues() if "CA" in r and r.resname != "HOH"
        ]
        assert len(residues) == int(row["length"]), domain
        assert len({r.parent.id for r in residues}) == 1, domain
        structure = Structure(domain)
        model = Model(0)
        chain = Chain("A")
        structure.add(model)
        model.add(chain)
        mapping, missing = [], []
        for number, residue in enumerate(residues, 1):
            new = Residue((" ", number, " "), residue.resname, " ")
            chain.add(new)
            mapping.append([residue.parent.id, list(residue.id), number])
            for name in ("N", "CA", "C", "O"):
                if name not in residue:
                    missing.append([number, name])
                    continue
                atom = residue[name]
                if atom.is_disordered():
                    atom = atom.selected_child
                atom = atom.copy()
                atom.set_altloc(" ")
                atom.disordered_flag = 0
                new.add(atom)
        shard = index % args.shards
        destination = args.output / "shards" / f"shard_{shard:02d}" / "backbones"
        destination.mkdir(parents=True, exist_ok=True)
        backbone = destination / f"{domain}.pdb"
        writer = PDBIO()
        writer.set_structure(structure)
        writer.save(str(backbone))
        reread = PDBParser(QUIET=True).get_structure(domain, backbone)
        np.testing.assert_array_equal(
            [r["CA"].coord for r in residues],
            [r["CA"].coord for r in reread.get_residues()],
        )
        domains.append(
            dict(
                row,
                shard=shard,
                source=str(source.resolve()),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                backbone=str(backbone.resolve()),
                backbone_sha256=hashlib.sha256(backbone.read_bytes()).hexdigest(),
                source_models=len(original),
                missing_backbone_atoms=missing,
                residue_mapping=mapping,
            )
        )
    write_json(
        args.output / "manifest.json",
        {
            "scope_csv": str(args.scope_csv.resolve()),
            "scope_csv_sha256": hashlib.sha256(args.scope_csv.read_bytes()).hexdigest(),
            "n_domains": len(domains),
            "n_shards": args.shards,
            "preparation": "First model, selected altloc, observed N/CA/C/O only; "
            "single chain A, consecutive residue numbers; no coordinate changes "
            "or missing-atom imputation. All domains retained.",
            "domains": domains,
        },
    )
    print(f"Prepared {len(domains)} domains across {args.shards} shards", flush=True)


def run(args: argparse.Namespace) -> None:
    """Run the same oracles as Table 5, retaining all eight refolds per domain."""
    import numpy as np
    import torch
    from collect_zevals_self_consistency import collect
    from run_scope_self_consistency_shard import _build_oracles, _complete_sample
    from zevals.proteins.designability import _atom_coords, compute_designability

    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    shard = args.output / "shards" / f"shard_{args.shard:02d}"
    output = shard / "zevals"
    output.mkdir(exist_ok=True)
    backbones = sorted((shard / "backbones").glob("*.pdb"))
    assert backbones
    predictor, folder = _build_oracles(
        Path(__file__).resolve().parents[1] / "models" / "zevals",
        output,
        mpnn_num_sequences=8,
        mpnn_temperature=0.1,
        mpnn_backbone_noise=0.02,
        structure_predictor="esmfold2",
        esmfold_model_name=str(args.model.resolve()),
        esmfold_chunk_size=64,
        esmfold2_num_loops=3,
        esmfold2_num_sampling_steps=50,
        esmfold2_num_samples=1,
        esmfold2_seed=0,
        sc_rmsd_threshold=1.5,
        sc_atom="CA",
    )
    # Model initialization consumes RNG even though checkpoint weights replace it.
    # Load once before applying per-domain seeds.
    folder._get_model(ca_only=False)
    for index, backbone in enumerate(backbones):
        complete = _complete_sample(output, backbone, 8)
        if complete:
            # A preempted write may leave a final, truncated CIF on disk.
            try:
                cached = output / "designability" / backbone.stem / "structures"
                length = len(_atom_coords(backbone))
                complete = all(
                    len(_atom_coords(pred)) == length for pred in cached.glob("*.cif")
                )
            except Exception:
                complete = False
        if not complete:
            # Domain-specific seeds make results independent of sharding/resume.
            seed = int.from_bytes(hashlib.sha256(backbone.stem.encode()).digest()[:4])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            compute_designability(
                [backbone],
                predictor,
                folder,
                output / "designability",
                sc_rmsd_threshold=1.5,
                sc_atom="CA",
            )
        if not _complete_sample(output, backbone, 8):
            raise RuntimeError(f"Incomplete refolds: {backbone}")
        predictions = sorted(
            (output / "designability" / backbone.stem / "structures").glob("*.cif")
        )
        assert len(predictions) == 8
        length = len(_atom_coords(backbone))
        assert all(len(_atom_coords(pred)) == length for pred in predictions), backbone
        write_json(
            shard / "progress.json",
            {
                "completed": index + 1,
                "total": len(backbones),
                "current": backbone.stem,
                "updated_at_unix": time.time(),
            },
        )
        print(f"{index + 1}/{len(backbones)} {backbone.stem}", flush=True)
    collect(
        shard / "backbones",
        output,
        output / "self_consistency_details.json",
        output / "self_consistency_refolds.csv",
        args.tmalign,
        1.5,
        "CA",
    )


def summarize(args: argparse.Namespace) -> None:
    """Require complete, unique coverage before reporting the training base rate."""
    from aggregate_scope_self_consistency_shards import aggregate

    manifest = json.loads((args.output / "manifest.json").read_text())
    observed = []
    for shard in range(manifest["n_shards"]):
        path = args.output / "shards" / f"shard_{shard:02d}" / "zevals"
        payload = json.loads((path / "self_consistency_details.json").read_text())
        observed.extend(payload["per_backbone"])
    expected = {row["domain_id"] for row in manifest["domains"]}
    assert len(observed) == len(expected) == 2000
    assert {row["sample"] for row in observed} == expected
    assert all(row["n_sequences"] == 8 for row in observed)
    aggregate(
        args.output / "shards",
        args.output / "self_consistency_details.json",
        args.output / "self_consistency_refolds.csv",
    )
    with (args.output / "per_domain.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(observed[0]))
        writer.writeheader()
        writer.writerows(sorted(observed, key=lambda row: row["sample"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "summarize"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scope-csv", type=Path, default=Path("data/scope/rungs/seed0/scope_2000.csv")
    )
    parser.add_argument(
        "--pdb-dir", type=Path, default=Path("data/scope/rungs/seed0/pdb_2000")
    )
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--tmalign", default="TMalign")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.mode == "run":
        import fcntl

        lock = args.output / "shards" / f"shard_{args.shard:02d}" / "worker.lock"
        with lock.open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"Shard {args.shard} is already running in another worker")
                return
            run(args)
    else:
        {"prepare": prepare, "summarize": summarize}[args.mode](args)


if __name__ == "__main__":
    main()
