#!/usr/bin/env python3
"""Reselect OMol25 illustration panels using source-aware atomic connectivity.

Run with the repository environment and OMOL25_DATA_DIR pointing to train_4M.
Reads reference records and original generated outputs; never modifies them.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from atomworks.ml.datasets import ASELMDBDataset
from biotite.structure.io.pdbx import CIFFile, get_structure
from dotenv import load_dotenv
from rdkit import Chem, rdBase
from rdkit.Chem import rdDetermineBonds

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
FILTER_SUBSETS = {"geom_orca6", "ani2x", "orbnet_denali", "spice"}


def component_sizes(elements, coordinates) -> list[int]:
    """Infer a coordinate-only graph, retaining hydrogens and all components."""
    xyz = f"{len(elements)}\n\n" + "\n".join(
        f"{element} {x:.12f} {y:.12f} {z:.12f}"
        for element, (x, y, z) in zip(elements, coordinates, strict=True)
    )
    molecule = Chem.MolFromXYZBlock(xyz)
    if molecule is None:
        raise ValueError("Could not parse coordinates")
    rdDetermineBonds.DetermineConnectivity(
        molecule, useHueckel=False, useVdw=True, covFactor=1.3
    )
    return sorted((len(c) for c in Chem.GetMolFrags(molecule)), reverse=True)


def main() -> None:
    load_dotenv(ROOT / ".env")
    manifest_path = HERE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    selected = {
        s["source_subset"]: dict(s)
        for s in manifest["samples"]
        if s.get("gallery") == "omol25" and s["source_subset"] in FILTER_SUBSETS
    }
    input_path = ROOT / next(iter(selected.values()))["input_manifest"]
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if any(s["input_manifest_sha256"] != input_hash for s in selected.values()):
        raise ValueError("Generation input manifest checksum mismatch")
    inputs = json.loads(input_path.read_text())
    paths = sorted(
        p
        for p in Path(os.environ["OMOL25_DATA_DIR"]).rglob("*.aselmdb")
        if not p.name.startswith("._")
    )
    dataset = ASELMDBDataset(
        paths=paths,
        name="omol25_gallery_connectivity",
        return_type="record",
        readonly=True,
        readahead=False,
        include_data=True,
    )
    source_dir = ROOT / Path(next(iter(selected.values()))["source"]).parent
    replacements, payloads, decisions = {}, {}, []
    for target_id, specification in sorted(inputs.items()):
        record = dataset[specification["extra"]["source_row_index"]]
        metadata = record["extra_info"]
        subset = metadata["data_id"]
        if subset not in selected or subset in replacements:
            continue
        reference = record["atoms"]
        mean_force = float(np.linalg.norm(reference.get_forces(), axis=1).mean())
        assert len(reference) == specification["extra"]["num_atoms_with_hydrogens"]
        assert metadata["charge"] == specification["total_charge"]
        assert metadata["spin"] == specification["spin_multiplicity"]
        assert abs(mean_force - specification["extra"]["mean_force_norm"]) < 1e-10
        reference_sizes = component_sizes(reference.symbols, reference.positions)
        required = subset == "geom_orca6" or len(reference_sizes) == 1
        sample_id = target_id + "_model_0"
        source = source_dir / (sample_id + ".cif.gz")
        compressed = source.read_bytes()
        raw = gzip.decompress(compressed)
        atoms = get_structure(CIFFile.read(io.StringIO(raw.decode())), model=1)
        assert len(atoms) == len(reference) and np.isfinite(atoms.coord).all()
        sizes = component_sizes(atoms.element, atoms.coord)
        accepted = not required or len(sizes) == 1
        decision = dict(
            sample_id=sample_id,
            source_subset=subset,
            reference_component_sizes=reference_sizes,
            generated_component_sizes=sizes,
            connectivity_required=required,
            accepted=accepted,
        )
        decisions.append(decision)
        print(json.dumps(decision), flush=True)
        if not accepted:
            continue
        counts = Counter(atoms.element)
        elements = sorted(counts)
        if "C" in counts:
            elements = ["C"] + (["H"] if "H" in counts else [])
            elements += sorted(set(counts) - {"C", "H"})
        formula = "".join(
            e + (str(counts[e]) if counts[e] != 1 else "") for e in elements
        )
        sample = dict(selected[subset])
        sample.update(
            sample_id=sample_id,
            display_id=f"omol25-{specification['extra']['sample_index']:04d}",
            source=str(source.relative_to(ROOT)),
            source_sha256=hashlib.sha256(compressed).hexdigest(),
            file=f"structures/{sample_id}.cif",
            sha256=hashlib.sha256(raw).hexdigest(),
            formula=formula,
            atom_count=len(atoms),
            generation_input=specification,
            conditioning_source=dict(
                source_row_index=specification["extra"]["source_row_index"],
                example_id=record["example_id"],
                shard=Path(record["db_path"]).name,
                db_id=record["db_id"],
                data_id=subset,
                source=metadata["source"],
                reference_formula=reference.get_chemical_formula(),
                reference_atom_count=len(reference),
                reference_mean_force_norm_eV_per_angstrom=mean_force,
            ),
            connectivity_filter=decision,
        )
        replacements[subset] = sample
        payloads[sample["file"]] = raw
        if replacements.keys() == selected.keys():
            break
    if replacements.keys() != selected.keys():
        raise ValueError("Could not select all filtered subsets")
    for sample in manifest["samples"]:
        subset = sample.get("source_subset")
        if sample.get("gallery") == "omol25" and subset in replacements:
            sample.clear()
            sample.update(replacements[subset])
    manifest["omol25_connectivity_filter"] = dict(
        rdkit_version=rdBase.rdkitVersion,
        method="DetermineConnectivity(useHueckel=False, useVdw=True, covFactor=1.3); "
        "GetMolFrags; all atoms including hydrogens. No bond-order assignment, "
        "sanitization, fragment removal, or coordinate optimization.",
        policy="Require one generated component for GEOM. For ANI-2x, OrbNet Denali "
        "and SPICE require one component when the reference graph is connected. "
        "Other subsets remain unfiltered to allow complexes, clusters and reactions.",
        candidates_in_filename_order=decisions,
    )
    manifest["selection"]["omol25"] = (
        "Stratified by conditioning-source OMol25 data_id: first source filename and "
        "largest saved output by atom count for biomolecules (filename order breaks "
        "size ties), first two source filenames for elytes, first eligible filename "
        "for the other eight "
        "subsets. Connectivity eligibility is defined in omol25_connectivity_filter. "
        "Twelve outputs cover all ten subsets; no sanitization, formula uniqueness, "
        "stability or transition-state filter. Generated coordinates and atom "
        "identities unchanged. Labels describe source provenance, not classification."
    )
    for filename, raw in payloads.items():
        (HERE / filename).write_bytes(raw)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    current_files = {s["file"] for s in manifest["samples"]}
    for sample in selected.values():
        if sample["file"] not in current_files:
            (HERE / sample["file"]).unlink()


if __name__ == "__main__":
    main()
