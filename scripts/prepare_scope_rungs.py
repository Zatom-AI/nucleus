#!/usr/bin/env python3
"""Build nested SCOPe/ASTRAL dataset rungs for Genie and Nucleus.

Inputs are the outputs from Genie's scripts/generate_scope_coords.py plus the
SCOPe pdbstyle tree. The script creates:

* Genie-compatible data roots:
  <genie-rung-dir>/seed0/scope_1000/scope/coords/*.npy
* The user's requested direct coordinate symlink dirs:
  <genie-scope-dir>/seed0/coords_1000/*.npy
* Nucleus GenericDFParser CSVs and .pdb symlinks:
  <nucleus-rung-dir>/seed0/scope_1000.csv
  <nucleus-rung-dir>/seed0/pdb_1000/*.pdb -> pdbstyle .ent files
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENIE_SCOPE_DIR = Path(
    os.environ.get("GENIE_SCOPE_DIR", REPO_ROOT / "data" / "scope" / "genie")
)
DEFAULT_GENIE_RUNG_DIR = Path(
    os.environ.get("GENIE_RUNG_DIR", REPO_ROOT / "data" / "scope" / "genie_rungs")
)
DEFAULT_PDBSTYLE_DIR = REPO_ROOT / "data" / "pdbstyle-2.08"
DEFAULT_NUCLEUS_RUNG_DIR = REPO_ROOT / "data" / "scope" / "rungs"
DEFAULT_BLACKLISTED_DOMAINS = {
    # Bad for the Nucleus/AtomWorks PDB parser and train transform pipeline.
    "d1enwa_",
    "d1k0ra4",
    "d1kfta_",
    "d1r8ha_",
    "d1sv7a_",
    "d1y6da_",
}


def _read_classes(path: Path) -> dict[str, tuple[str, str, str, str]]:
    classes: dict[str, tuple[str, str, str, str]] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            domain_id, sccs = line.split(",", 1)
            parts = sccs.split(".")
            if len(parts) != 4:
                raise ValueError(f"Malformed SCOPe class line: {line}")
            classes[domain_id] = (parts[0], parts[1], parts[2], parts[3])
    return classes


def _is_numpy_binary(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(6) == b"\x93NUMPY"


def _coord_row_count_from_array(coords: np.ndarray, path: Path) -> int:
    if coords.ndim == 1:
        return 1 if coords.size == 3 else coords.size
    if coords.ndim >= 2 and coords.shape[-1] == 3:
        return int(np.prod(coords.shape[:-1]))
    raise ValueError(f"Coordinate array must end in xyz dimension 3: {path}")


def _coord_length(path: Path, atoms_per_residue: int = 3) -> int:
    if atoms_per_residue <= 0:
        raise ValueError("--coord-atoms-per-residue must be positive")

    if _is_numpy_binary(path):
        coords = np.asarray(np.load(path, allow_pickle=False))
        n_coord_rows = _coord_row_count_from_array(coords, path)
    else:
        with path.open() as handle:
            n_coord_rows = sum(1 for line in handle if line.strip())

    if n_coord_rows % atoms_per_residue != 0:
        raise ValueError(
            f"Coordinate row count is not divisible by {atoms_per_residue}: {path}"
        )
    return n_coord_rows // atoms_per_residue


def _scope_pdbstyle_path(pdbstyle_dir: Path, domain_id: str) -> Path:
    return pdbstyle_dir / domain_id[2:4] / f"{domain_id}.ent"


def _group_key(parts: tuple[str, str, str, str], balance_level: str) -> str:
    if balance_level == "class":
        return parts[0]
    if balance_level == "fold":
        return ".".join(parts[:2])
    if balance_level == "superfamily":
        return ".".join(parts[:3])
    if balance_level == "family":
        return ".".join(parts[:4])
    raise ValueError(f"Unsupported balance level: {balance_level}")


def _balanced_order(
    domains: list[dict[str, object]],
    seed: int,
    balance_level: str,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for domain in domains:
        groups[str(domain["balance_key"])].append(domain)

    keys = sorted(groups)
    rng.shuffle(keys)
    for key in keys:
        groups[key].sort(key=lambda item: str(item["domain_id"]))
        rng.shuffle(groups[key])

    ordered: list[dict[str, object]] = []
    remaining = True
    while remaining:
        remaining = False
        for key in keys:
            if groups[key]:
                ordered.append(groups[key].pop())
                remaining = True
    return ordered


def _symlink(src: Path, dst: Path, *, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        if Path(os.readlink(dst)) == src:
            return
        if force:
            dst.unlink()
        else:
            raise FileExistsError(f"Existing symlink points elsewhere: {dst}")
    elif dst.exists():
        if force:
            if dst.is_dir():
                raise IsADirectoryError(f"Refusing to replace directory: {dst}")
            dst.unlink()
        else:
            raise FileExistsError(f"Path already exists: {dst}")
    dst.symlink_to(src)


def _remove_stale_symlinks(directory: Path, keep_names: set[str]) -> None:
    """Remove stale generated symlinks without touching source data files."""
    if not directory.exists():
        return
    for path in directory.iterdir():
        if path.name in keep_names:
            continue
        if path.is_symlink():
            path.unlink()


def _write_classes(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(f"{row['domain_id']},{row['sccs']}\n")


def _write_txt(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(f"{row['domain_id']}\n")


def _write_nucleus_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "example_id",
        "pdb_id",
        "domain_id",
        "scope_class",
        "scope_fold",
        "scope_superfamily",
        "scope_family",
        "length",
        "split",
        "path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            domain_id = str(row["domain_id"])
            scope_class, scope_fold, scope_superfamily, scope_family = row["sccs_parts"]
            writer.writerow(
                {
                    "example_id": f"scope_{domain_id}",
                    "pdb_id": domain_id[1:5],
                    "domain_id": domain_id,
                    "scope_class": scope_class,
                    "scope_fold": scope_fold,
                    "scope_superfamily": scope_superfamily,
                    "scope_family": scope_family,
                    "length": row["length"],
                    "split": "train",
                    "path": domain_id,
                }
            )


def _parse_rungs(values: list[str], pool_size: int) -> list[tuple[str, int, int]]:
    rungs: list[tuple[str, int, int]] = []
    seen_labels: set[str] = set()
    for value in values:
        if value.lower() == "all":
            label = "all"
            requested = pool_size
            actual = pool_size
        else:
            requested = int(value)
            if requested <= 0:
                raise ValueError(f"Rung sizes must be positive, got {value}")
            label = str(requested)
            actual = min(requested, pool_size)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        rungs.append((label, requested, actual))
    return rungs


def _load_pool(args: argparse.Namespace) -> list[dict[str, object]]:
    coords_dir = args.genie_scope_dir / "coords"
    classes_path = args.genie_scope_dir / "classes.txt"
    if not coords_dir.exists():
        raise FileNotFoundError(f"Missing Genie SCOPe coords directory: {coords_dir}")
    if not classes_path.exists():
        raise FileNotFoundError(f"Missing Genie classes file: {classes_path}")

    classes = _read_classes(classes_path)
    blacklisted_domains = set(args.blacklist_domain)
    pool: list[dict[str, object]] = []
    skipped = defaultdict(int)
    for domain_id, parts in sorted(classes.items()):
        if domain_id in blacklisted_domains:
            skipped["blacklisted"] += 1
            continue
        coord_path = coords_dir / f"{domain_id}.npy"
        pdbstyle_path = _scope_pdbstyle_path(args.pdbstyle_dir, domain_id)
        if not coord_path.exists():
            skipped["missing_coords"] += 1
            continue
        if not pdbstyle_path.exists():
            skipped["missing_pdbstyle"] += 1
            continue
        length = _coord_length(coord_path, args.coord_atoms_per_residue)
        if length > args.max_n_res:
            skipped["too_long"] += 1
            continue
        balance_key = _group_key(parts, args.balance_level)
        pool.append(
            {
                "domain_id": domain_id,
                "coord_path": coord_path,
                "pdbstyle_path": pdbstyle_path,
                "length": length,
                "sccs_parts": parts,
                "sccs": ".".join(parts),
                "balance_key": balance_key,
            }
        )

    print(
        f"SCOPe pool <= {args.max_n_res} residues: {len(pool)} "
        f"(skipped: {dict(skipped)})"
    )
    if args.expected_pool_size is not None:
        delta = abs(len(pool) - args.expected_pool_size)
        if delta > args.pool_size_tolerance:
            raise SystemExit(
                "SCOPe pool sanity check failed: "
                f"expected about {args.expected_pool_size} +/- "
                f"{args.pool_size_tolerance}, got {len(pool)}."
            )
    return pool


def _materialize_seed(
    *,
    args: argparse.Namespace,
    seed: int,
    ordered: list[dict[str, object]],
    rungs: list[tuple[str, int, int]],
) -> None:
    seed_nucleus_dir = args.nucleus_rung_dir / f"seed{seed}"
    seed_genie_direct_dir = args.genie_scope_dir / f"seed{seed}"
    seed_genie_rung_dir = args.genie_rung_dir / f"seed{seed}"
    seed_nucleus_dir.mkdir(parents=True, exist_ok=True)
    seed_genie_direct_dir.mkdir(parents=True, exist_ok=True)
    seed_genie_rung_dir.mkdir(parents=True, exist_ok=True)

    previous: set[str] = set()
    for label, requested, actual in rungs:
        rows = ordered[:actual]
        domain_ids = {str(row["domain_id"]) for row in rows}
        if previous and not previous.issubset(domain_ids):
            raise AssertionError(f"Rungs are not nested for seed {seed}, label {label}")
        previous = domain_ids

        direct_coords_dir = seed_genie_direct_dir / f"coords_{label}"
        dropin_scope_dir = seed_genie_rung_dir / f"scope_{label}" / "scope"
        dropin_coords_dir = dropin_scope_dir / "coords"
        nucleus_pdb_dir = seed_nucleus_dir / f"pdb_{label}"

        direct_coords_dir.mkdir(parents=True, exist_ok=True)
        dropin_coords_dir.mkdir(parents=True, exist_ok=True)
        nucleus_pdb_dir.mkdir(parents=True, exist_ok=True)

        if args.force:
            keep_coord_names = {f"{row['domain_id']}.npy" for row in rows}
            keep_pdb_names = {f"{row['domain_id']}.pdb" for row in rows}
            _remove_stale_symlinks(direct_coords_dir, keep_coord_names)
            _remove_stale_symlinks(dropin_coords_dir, keep_coord_names)
            _remove_stale_symlinks(nucleus_pdb_dir, keep_pdb_names)

        for row in rows:
            domain_id = str(row["domain_id"])
            _symlink(
                Path(row["coord_path"]).resolve(),
                direct_coords_dir / f"{domain_id}.npy",
                force=args.force,
            )
            _symlink(
                Path(row["coord_path"]).resolve(),
                dropin_coords_dir / f"{domain_id}.npy",
                force=args.force,
            )
            _symlink(
                Path(row["pdbstyle_path"]).resolve(),
                nucleus_pdb_dir / f"{domain_id}.pdb",
                force=args.force,
            )

        _write_classes(dropin_scope_dir / "classes.txt", rows)
        _write_txt(seed_nucleus_dir / f"train_{label}.txt", rows)
        _write_nucleus_csv(seed_nucleus_dir / f"scope_{label}.csv", rows)

        fold_count = len({str(row["balance_key"]) for row in rows})
        manifest = {
            "seed": seed,
            "label": label,
            "requested_size": requested,
            "actual_size": actual,
            "max_n_res": args.max_n_res,
            "balance_level": args.balance_level,
            "blacklisted_domains": sorted(args.blacklist_domain),
            "distinct_balance_groups": fold_count,
            "nucleus_csv": str(seed_nucleus_dir / f"scope_{label}.csv"),
            "nucleus_pdb_base_path": str(nucleus_pdb_dir),
            "genie_data_dir": str(seed_genie_rung_dir / f"scope_{label}"),
            "genie_direct_coords": str(direct_coords_dir),
        }
        (seed_nucleus_dir / f"manifest_{label}.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(
            f"seed={seed} rung={label}: actual={actual}, "
            f"{args.balance_level}s={fold_count}, csv={manifest['nucleus_csv']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genie-scope-dir", type=Path, default=DEFAULT_GENIE_SCOPE_DIR)
    parser.add_argument("--genie-rung-dir", type=Path, default=DEFAULT_GENIE_RUNG_DIR)
    parser.add_argument("--pdbstyle-dir", type=Path, default=DEFAULT_PDBSTYLE_DIR)
    parser.add_argument("--nucleus-rung-dir", type=Path, default=DEFAULT_NUCLEUS_RUNG_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rungs", nargs="+", default=["1000", "2000", "4000"])
    parser.add_argument("--max-n-res", type=int, default=128)
    parser.add_argument(
        "--coord-atoms-per-residue",
        type=int,
        default=3,
        help=(
            "Number of coordinate rows per residue in the Genie SCOPe coordinate "
            "files. Genie's generate_scope_coords.py writes N, CA, C rows."
        ),
    )
    parser.add_argument(
        "--balance-level",
        choices=["class", "fold", "superfamily", "family"],
        default="fold",
    )
    parser.add_argument("--expected-pool-size", type=int, default=3936)
    parser.add_argument("--pool-size-tolerance", type=int, default=150)
    parser.add_argument(
        "--blacklist-domain",
        action="append",
        default=sorted(DEFAULT_BLACKLISTED_DOMAINS),
        help=(
            "SCOPe domain ID to exclude from all generated rungs. "
            "May be repeated; defaults to known Nucleus-incompatible domains."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pool = _load_pool(args)
    if not pool:
        raise SystemExit("No SCOPe domains passed the filters.")
    rungs = _parse_rungs(args.rungs, len(pool))
    for seed in args.seeds:
        ordered = _balanced_order(pool, seed, args.balance_level)
        _materialize_seed(args=args, seed=seed, ordered=ordered, rungs=rungs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
