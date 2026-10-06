#!/usr/bin/env python3
"""Link original SCOPe coordinates using the frozen paper CSV domain lists."""

import argparse
import csv
from pathlib import Path


def link_structures(pdbstyle_dir: Path, rung_root: Path) -> int:
    planned = []
    for csv_path in sorted(rung_root.glob("seed*/scope_*.csv")):
        size = csv_path.stem.removeprefix("scope_")
        with csv_path.open() as handle:
            for row in csv.DictReader(handle):
                domain = row["domain_id"]
                if Path(domain).name != domain or Path(row["path"]).name != row["path"]:
                    raise ValueError(f"Unsafe domain path in {csv_path}")
                source = (pdbstyle_dir / domain[2:4] / f"{domain}.ent").resolve()
                destination = csv_path.parent / f"pdb_{size}" / f"{row['path']}.pdb"
                if not source.is_file():
                    raise FileNotFoundError(source)
                if destination.exists() or destination.is_symlink():
                    if not destination.is_symlink() or destination.resolve() != source:
                        raise FileExistsError(destination)
                else:
                    planned.append((source, destination))
    if not list(rung_root.glob("seed*/scope_*.csv")):
        raise FileNotFoundError(f"No frozen SCOPe CSVs in {rung_root}")
    for source, destination in planned:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source)
    return len(planned)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdbstyle-dir", type=Path, required=True)
    parser.add_argument("--rung-root", type=Path, default=Path("data/scope/rungs"))
    args = parser.parse_args()
    print(f"Linked {link_structures(args.pdbstyle_dir, args.rung_root)} structures")
