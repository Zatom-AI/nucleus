#!/usr/bin/env python3
"""Create one canonical, first-model PDB per SCOPe-2k training domain."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def first_model(text: str) -> str:
    lines = text.splitlines()
    model_positions = [i for i, line in enumerate(lines) if line.startswith("MODEL")]
    if not model_positions:
        return "\n".join(lines) + "\n"
    start = model_positions[0] + 1
    end = next(
        (i for i in range(start, len(lines)) if lines[i].startswith("ENDMDL")),
        len(lines),
    )
    atom_lines = [line for line in lines[start:end] if line.startswith(("ATOM", "HETATM", "TER"))]
    return "\n".join(atom_lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.force and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(args.input_dir.glob("*.pdb"))
    if not inputs:
        raise ValueError(f"no PDB files found in {args.input_dir}")
    for source in inputs:
        destination = args.output_dir / source.name
        destination.write_text(first_model(source.read_text()))
    print(f"wrote {len(inputs)} canonical targets to {args.output_dir}")


if __name__ == "__main__":
    main()
