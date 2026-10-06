#!/usr/bin/env python3
"""Extract protein backbone PDBs from Zatom-2 generated structure files."""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

BACKBONE_ATOMS = {"N", "CA", "C", "O"}


def _load_structure(path: Path):
    from Bio.PDB import MMCIFParser, PDBParser

    suffixes = "".join(path.suffixes[-2:])
    if suffixes == ".cif.gz":
        with gzip.open(path, "rt") as handle:
            return MMCIFParser(QUIET=True).get_structure(path.stem, handle)
    if path.suffix == ".cif":
        return MMCIFParser(QUIET=True).get_structure(path.stem, str(path))
    if path.suffix == ".pdb":
        return PDBParser(QUIET=True).get_structure(path.stem, str(path))
    raise ValueError(f"Unsupported structure extension: {path}")


class BackboneSelect:
    """PDBIO selector retaining protein backbone atoms only."""

    def accept_model(self, model) -> bool:
        return int(getattr(model, "id", 0)) == 0

    def accept_chain(self, chain) -> bool:
        return True

    def accept_residue(self, residue) -> bool:
        if residue.get_resname().strip() == "HOH":
            return False
        atom_names = {atom.get_name().strip() for atom in residue.get_atoms()}
        return "CA" in atom_names and bool(atom_names & BACKBONE_ATOMS)

    def accept_atom(self, atom) -> bool:
        return atom.get_name().strip() in BACKBONE_ATOMS


def _structure_paths(input_dir: Path) -> list[Path]:
    paths = [
        *input_dir.rglob("*.cif.gz"),
        *input_dir.rglob("*.cif"),
        *input_dir.rglob("*.pdb"),
    ]
    filtered = []
    for path in paths:
        name = path.name
        if "_denoised_model_" in name or "_noisy_model_" in name:
            continue
        filtered.append(path)
    return sorted(filtered)


def _output_name(path: Path) -> str:
    if "".join(path.suffixes[-2:]) == ".cif.gz":
        return path.name[: -len(".cif.gz")] + ".pdb"
    return path.stem + ".pdb"


def extract_backbones(input_dir: Path, output_dir: Path, overwrite: bool) -> int:
    from Bio.PDB import PDBIO

    output_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0
    selector = BackboneSelect()

    for source in _structure_paths(input_dir):
        dest = output_dir / _output_name(source)
        if dest.exists() and not overwrite:
            n_skipped += 1
            continue

        structure = _load_structure(source)
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(dest), selector)

        if dest.stat().st_size == 0:
            dest.unlink(missing_ok=True)
            print(f"warning: no backbone atoms found in {source}", file=sys.stderr)
            continue
        n_written += 1

    print(
        f"backbone extraction complete: written={n_written} skipped={n_skipped} "
        f"output_dir={output_dir}",
        flush=True,
    )
    return n_written + n_skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    n_total = extract_backbones(args.input_dir, args.output_dir, args.overwrite)
    if n_total == 0:
        raise SystemExit(f"No structure files found under {args.input_dir}")


if __name__ == "__main__":
    main()
