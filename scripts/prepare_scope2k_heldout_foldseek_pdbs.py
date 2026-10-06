#!/usr/bin/env python3
"""Prepare the non-SCOPe-2k domains from the 4k rung as FoldSeek inputs."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def domain_ids(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        return {row["domain_id"] for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--all-csv", type=Path, required=True)
    parser.add_argument("--all-pdb-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    train = domain_ids(args.train_csv)
    with args.all_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    heldout = [row["domain_id"] for row in rows if row["domain_id"] not in train]
    if args.force and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for domain_id in heldout:
        source = args.all_pdb_dir / (domain_id + ".pdb")
        if not source.is_file():
            missing.append(str(source))
            continue
        destination = args.output_dir / source.name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source.resolve())
    if missing:
        raise FileNotFoundError(
            "missing {} held-out PDBs, first: {}".format(len(missing), missing[0])
        )
    print("wrote {} held-out PDB links to {}".format(len(heldout), args.output_dir))


if __name__ == "__main__":
    main()
