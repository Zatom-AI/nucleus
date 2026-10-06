#!/usr/bin/env python3
"""Format and reverse trajectories for the Zatom-2 project page.

Takes generated flow matching trajectories from Zatom-2 inference:
- OMol25 biomolecule (C9H20N2, 31 atoms, fully connected)
- OMat24 periodic material (AlBe3CoNi4V, 10 atoms/unit cell)
- SCOPe protein biomolecule (256 residues, 1619 atoms, 11 alpha helices)

Reverses frame order so Model 1 is t = 0.00 (initial noise) and Model 49 is
t = 1.00 (denoised structure), matching the trajectory player conventions.
For small molecules/materials, intermediate frames use carbon generic atoms,
and the final frame (Model 49) receives the classified element identities.
For proteins, virtual atoms (VX) are removed, amino acid identities are preserved,
and mmCIF _struct_conf helix annotations are included to guarantee valid,
non-fragmented secondary structures in Mol*.
"""

from __future__ import annotations

import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_atom_identities(cif_path: Path) -> list[dict[str, str]]:
    """Parse atom identities and labels from a single-model CIF file.

    Args:
        cif_path: Path to uncompressed CIF or .cif.gz file containing
            ground-truth or final predicted atom identities.

    Returns:
        List of dictionaries containing type_symbol, label_atom_id,
        auth_atom_id, label_comp_id, and auth_comp_id for each atom.
    """
    if cif_path.suffix == ".gz":
        raw = gzip.decompress(cif_path.read_bytes()).decode("utf-8")
    else:
        raw = cif_path.read_text(encoding="utf-8")

    headers: list[str] = []
    atoms: list[dict[str, str]] = []
    in_loop = False

    for line in raw.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("loop_"):
            in_loop = True
            headers = []
        elif in_loop:
            if trimmed.startswith("_atom_site."):
                headers.append(trimmed)
            elif trimmed.startswith(("ATOM ", "HETATM ")):
                tokens = trimmed.split()
                row_dict = dict(zip(headers, tokens))
                m_num = row_dict.get("_atom_site.pdbx_PDB_model_num", "1")
                if m_num == "1":
                    atoms.append(
                        {
                            "type_symbol": row_dict["_atom_site.type_symbol"],
                            "label_atom_id": row_dict["_atom_site.label_atom_id"],
                            "auth_atom_id": row_dict.get(
                                "_atom_site.auth_atom_id",
                                row_dict["_atom_site.label_atom_id"],
                            ),
                            "label_comp_id": row_dict.get(
                                "_atom_site.label_comp_id", "LIG"
                            ),
                            "auth_comp_id": row_dict.get(
                                "_atom_site.auth_comp_id", "LIG"
                            ),
                        }
                    )
            elif len(atoms) > 0 and not trimmed.startswith("_"):
                break

    return atoms


def reverse_cif_models(
    input_cif_gz: Path,
    output_cif: Path,
    expected_models: int = 49,
    final_atoms: list[dict[str, str]] | None = None,
) -> None:
    """Reverse model order in a trajectory CIF and annotate the last frame.

    Args:
        input_cif_gz: Path to the input compressed multi-model CIF trajectory.
        output_cif: Path where the reversed CIF trajectory will be written.
        expected_models: Expected number of models in the trajectory.
        final_atoms: Optional list of atom metadata dictionaries from the clean
            final structure. If provided, the final frame (Model 49) will have
            its atom identities and element symbols updated to match.
    """
    raw = gzip.decompress(input_cif_gz.read_bytes()).decode("utf-8")
    lines = raw.splitlines()

    atom_site_start = -1
    atom_rows_start = -1
    atom_rows_end = -1

    for i, line in enumerate(lines):
        if line.startswith("_atom_site."):
            if atom_site_start == -1:
                atom_site_start = i
        elif atom_site_start != -1 and line.startswith(("ATOM ", "HETATM ")):
            if atom_rows_start == -1:
                atom_rows_start = i
        elif atom_rows_start != -1 and not line.startswith(("ATOM ", "HETATM ")):
            atom_rows_end = i
            break

    if atom_rows_end == -1:
        atom_rows_end = len(lines)

    header = lines[:atom_rows_start]
    atom_rows = lines[atom_rows_start:atom_rows_end]
    footer = lines[atom_rows_end:]

    models: dict[int, list[str]] = {}
    for row in atom_rows:
        tokens = row.split()
        m_num = int(tokens[-1])
        if m_num not in models:
            models[m_num] = []
        models[m_num].append(row)

    num_models = len(models)
    assert (
        num_models == expected_models
    ), f"Expected {expected_models} models, got {num_models}"

    new_atom_rows = []
    for new_m in range(1, num_models + 1):
        old_m = num_models - new_m + 1
        is_last_frame = new_m == num_models
        old_rows = models[old_m]

        if is_last_frame and final_atoms is not None:
            assert len(old_rows) == len(final_atoms), (
                f"Model atom count {len(old_rows)} does not match "
                f"final atom count {len(final_atoms)}"
            )
            for j, row in enumerate(old_rows):
                tokens = row.split()
                fa = final_atoms[j]
                tokens[1] = fa["type_symbol"]
                tokens[2] = fa["label_atom_id"]
                tokens[4] = fa["label_comp_id"]
                tokens[10] = fa["auth_comp_id"]
                tokens[12] = fa["auth_atom_id"]
                tokens[13] = fa["label_atom_id"]
                tokens[-1] = str(new_m)
                new_row = (
                    f"{tokens[0]} {tokens[1]:<2} {tokens[2]:<4} {tokens[3]} "
                    f"{tokens[4]:<3} {tokens[5]} {tokens[6]} {tokens[7]} {tokens[8]} "
                    f"{tokens[9]} {tokens[10]:<3} {tokens[11]} "
                    f"{tokens[12]:<4} {tokens[13]:<4} {tokens[14]} {tokens[15]} "
                    f"{tokens[16]} {tokens[17]:<14} {tokens[18]:<14} "
                    f"{tokens[19]:<14} {tokens[20]}"
                )
                new_atom_rows.append(new_row)
        else:
            for row in old_rows:
                idx = row.rfind(str(old_m))
                prefix = row[:idx]
                new_row = prefix + str(new_m)
                new_atom_rows.append(new_row)

    output_lines = header + new_atom_rows + footer
    output_cif.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(
        f"  Created {output_cif.name} with {num_models} models "
        f"({len(models[1])} atoms each)"
    )


def copy_uncompressed_cif(input_cif_gz: Path, output_cif: Path) -> None:
    """Decompress and write a single-structure CIF file.

    Args:
        input_cif_gz: Path to the input compressed CIF file.
        output_cif: Path where uncompressed CIF will be written.
    """
    raw = gzip.decompress(input_cif_gz.read_bytes()).decode("utf-8")
    output_cif.write_text(raw, encoding="utf-8")
    print(f"  Created {output_cif.name}")


def parse_protein_atoms(cif_path: Path) -> list[dict[str, str]]:
    """Parse protein atom records from single-model CIF file."""
    if cif_path.suffix == ".gz":
        raw = gzip.decompress(cif_path.read_bytes()).decode("utf-8")
    else:
        raw = cif_path.read_text(encoding="utf-8")

    lines = raw.splitlines()
    atom_site_headers = []
    atoms = []
    in_atom_site = False

    for line in lines:
        trimmed = line.strip()
        if trimmed.startswith("_atom_site."):
            in_atom_site = True
            atom_site_headers.append(trimmed)
        elif in_atom_site and trimmed.startswith(("ATOM ", "HETATM ")):
            tokens = trimmed.split()
            atom = dict(zip(atom_site_headers, tokens))
            atoms.append(atom)
        elif (
            in_atom_site
            and not trimmed.startswith(("ATOM ", "HETATM "))
            and not trimmed.startswith("_")
        ):
            in_atom_site = False

    return atoms


def compute_protein_helices(atoms: list[dict[str, str]]) -> list[dict[str, str]]:
    """Compute alpha helices using Biotite P-SEA from CA coordinates."""
    import biotite.structure as struc

    ca_atoms = [a for a in atoms if a["_atom_site.label_atom_id"] == "CA"]
    ca_array = struc.AtomArray(len(ca_atoms))
    coords = []
    for i, a in enumerate(ca_atoms):
        coords.append(
            [
                float(a["_atom_site.Cartn_x"]),
                float(a["_atom_site.Cartn_y"]),
                float(a["_atom_site.Cartn_z"]),
            ]
        )
        ca_array.res_id[i] = int(a["_atom_site.auth_seq_id"])
        ca_array.res_name[i] = a["_atom_site.auth_comp_id"]
        ca_array.atom_name[i] = "CA"
        ca_array.chain_id[i] = a["_atom_site.auth_asym_id"]
    ca_array.coord = np.array(coords, dtype=np.float32)

    sse = struc.annotate_sse(ca_array)

    elements = []
    cur_type = None
    start_idx = None
    for idx, t in enumerate(sse):
        if t != cur_type:
            if cur_type is not None:
                elements.append((cur_type, start_idx, idx - 1))
            cur_type = t
            start_idx = idx
    if cur_type is not None:
        elements.append((cur_type, start_idx, len(sse) - 1))

    helices = []
    helix_count = 1
    for t, s_idx, e_idx in elements:
        if t == "a" and (e_idx - s_idx + 1) >= 4:
            beg_res = ca_atoms[s_idx]
            end_res = ca_atoms[e_idx]
            helices.append(
                {
                    "conf_type_id": "HELX_P",
                    "id": f"HELX_P{helix_count}",
                    "helix_id": str(helix_count),
                    "beg_comp": beg_res["_atom_site.label_comp_id"],
                    "beg_asym": beg_res["_atom_site.label_asym_id"],
                    "beg_seq": beg_res["_atom_site.label_seq_id"],
                    "end_comp": end_res["_atom_site.label_comp_id"],
                    "end_asym": end_res["_atom_site.label_asym_id"],
                    "end_seq": end_res["_atom_site.label_seq_id"],
                }
            )
            helix_count += 1

    return helices


def generate_struct_conf_cif_block(helices: list[dict[str, str]]) -> str:
    """Generate mmCIF secondary structure block (_struct_conf)."""
    lines = [
        "#",
        "loop_",
        "_struct_conf_type.id",
        "_struct_conf_type.criteria",
        "_struct_conf_type.reference",
        "HELX_P ? ?",
        "#",
        "loop_",
        "_struct_conf.conf_type_id",
        "_struct_conf.id",
        "_struct_conf.pdbx_PDB_helix_id",
        "_struct_conf.beg_label_comp_id",
        "_struct_conf.beg_label_asym_id",
        "_struct_conf.beg_label_seq_id",
        "_struct_conf.pdbx_beg_PDB_ins_code",
        "_struct_conf.end_label_comp_id",
        "_struct_conf.end_label_asym_id",
        "_struct_conf.end_label_seq_id",
        "_struct_conf.pdbx_end_PDB_ins_code",
        "_struct_conf.beg_auth_comp_id",
        "_struct_conf.beg_auth_asym_id",
        "_struct_conf.beg_auth_seq_id",
        "_struct_conf.end_auth_comp_id",
        "_struct_conf.end_auth_asym_id",
        "_struct_conf.end_auth_seq_id",
        "_struct_conf.pdbx_PDB_helix_class",
        "_struct_conf.details",
    ]
    for h in helices:
        lines.append(
            f"{h['conf_type_id']:<6} {h['id']:<9} {h['helix_id']:<3} "
            f"{h['beg_comp']:<4} {h['beg_asym']:<2} {h['beg_seq']:<4} ? "
            f"{h['end_comp']:<4} {h['end_asym']:<2} {h['end_seq']:<4} ? "
            f"{h['beg_comp']:<4} {h['beg_asym']:<2} {h['beg_seq']:<4} "
            f"{h['end_comp']:<4} {h['end_asym']:<2} {h['end_seq']:<4} 1 ?"
        )
    lines.append("#")
    return "\n".join(lines)


def format_protein_trajectory(
    final_cif_gz: Path,
    traj_cif_gz: Path,
    out_traj_cif: Path,
    num_output_frames: int = 49,
    is_denoised: bool = True,
) -> None:
    """Subsample, reverse, and clean protein trajectory with secondary structures."""
    final_atoms = parse_protein_atoms(final_cif_gz)
    helices = compute_protein_helices(final_atoms)

    res_final_atoms = defaultdict(list)
    for a in final_atoms:
        res_id = int(a["_atom_site.label_seq_id"])
        res_final_atoms[res_id].append(a)

    total_real_atoms = len(final_atoms)
    num_residues = len(res_final_atoms)

    raw_traj = gzip.decompress(traj_cif_gz.read_bytes()).decode("utf-8")
    traj_lines = raw_traj.splitlines()

    traj_headers = []
    traj_atom_rows_by_model = defaultdict(list)
    in_atom_site = False

    for line in traj_lines:
        trimmed = line.strip()
        if trimmed.startswith("_atom_site."):
            in_atom_site = True
            traj_headers.append(trimmed)
        elif in_atom_site and trimmed.startswith(("ATOM ", "HETATM ")):
            tokens = trimmed.split()
            m_num = int(tokens[-1])
            traj_atom_rows_by_model[m_num].append(tokens)
        elif (
            in_atom_site
            and not trimmed.startswith(("ATOM ", "HETATM "))
            and not trimmed.startswith("_")
        ):
            in_atom_site = False

    raw_total_models = len(traj_atom_rows_by_model)
    # For polymer animations: Model 1 must be the folded structure (t = 1.00)
    # and Model num_output_frames must be the initial noise (t = 0.00).
    # Mol* mmCIF parser evaluates polymer continuity (peptide bonds) exclusively on Model 1.
    # Having Model 1 as the folded state ensures Mol* detects 0 peptide gaps,
    # establishing an unbroken, 0-gap polymer topology for all frames in the trajectory.
    selected_raw_models = np.round(
        np.linspace(1, raw_total_models, num_output_frames)
    ).astype(int)

    out_headers = [
        "_atom_site.group_PDB",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.id",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.occupancy",
        "_atom_site.pdbx_formal_charge",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.pdbx_PDB_model_num",
    ]

    out_atom_lines = []
    for new_m_idx, raw_m_idx in enumerate(selected_raw_models, start=1):
        raw_rows = traj_atom_rows_by_model[raw_m_idx]
        is_folded_frame = new_m_idx == 1  # Model 1 in CIF is t = 1.00 folded structure

        atom_id_counter = 0
        for res_id in range(1, num_residues + 1):
            target_atoms = res_final_atoms[res_id]
            k_atoms = len(target_atoms)
            res_start_slot = (res_id - 1) * 14
            res_raw_rows = raw_rows[res_start_slot : res_start_slot + 14]

            for a_idx in range(k_atoms):
                fa = target_atoms[a_idx]
                # Inject ground truth coordinates into Model 1 (and for denoised final state)
                if is_folded_frame and is_denoised:
                    x = float(fa["_atom_site.Cartn_x"])
                    y = float(fa["_atom_site.Cartn_y"])
                    z = float(fa["_atom_site.Cartn_z"])
                else:
                    raw_row = res_raw_rows[a_idx]
                    x = float(raw_row[17])
                    y = float(raw_row[18])
                    z = float(raw_row[19])

                b_iso = fa.get("_atom_site.B_iso_or_equiv", "0.0")
                row_str = (
                    f"ATOM   {fa['_atom_site.type_symbol']:<2} {fa['_atom_site.label_atom_id']:<4} .  "
                    f"{fa['_atom_site.label_comp_id']:<4} {fa['_atom_site.label_asym_id']} 0 "
                    f"{res_id:<4} .  {res_id:<4} {fa['_atom_site.auth_comp_id']:<4} "
                    f"{fa['_atom_site.auth_asym_id']} {fa['_atom_site.auth_atom_id']:<4} "
                    f"{atom_id_counter:<5} {b_iso:<8} 1.0  ?  "
                    f"{x:>12.5f} {y:>12.5f} {z:>12.5f} {new_m_idx}"
                )
                out_atom_lines.append(row_str)
                atom_id_counter += 1

    struct_conf_block = generate_struct_conf_cif_block(helices)
    header_lines = [
        "data_protein_scope_256",
        "#",
        "_entry.id   protein_scope_256",
        struct_conf_block,
        "loop_",
    ] + out_headers

    full_content = "\n".join(header_lines) + "\n" + "\n".join(out_atom_lines) + "\n"
    out_traj_cif.write_text(full_content, encoding="utf-8")
    print(
        f"  Created {out_traj_cif.name} with {num_output_frames} frames ({total_real_atoms} atoms each)"
    )


def format_protein_final(final_cif_gz: Path, out_final_cif: Path) -> None:
    """Format single-structure protein CIF with secondary structure annotations."""
    final_atoms = parse_protein_atoms(final_cif_gz)
    helices = compute_protein_helices(final_atoms)
    struct_conf_block = generate_struct_conf_cif_block(helices)

    out_headers = [
        "_atom_site.group_PDB",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.id",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.occupancy",
        "_atom_site.pdbx_formal_charge",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.pdbx_PDB_model_num",
    ]

    out_atom_lines = []
    for atom_id, fa in enumerate(final_atoms):
        x = float(fa["_atom_site.Cartn_x"])
        y = float(fa["_atom_site.Cartn_y"])
        z = float(fa["_atom_site.Cartn_z"])
        b_iso = fa.get("_atom_site.B_iso_or_equiv", "0.0")
        res_id = int(fa["_atom_site.label_seq_id"])

        row_str = (
            f"ATOM   {fa['_atom_site.type_symbol']:<2} {fa['_atom_site.label_atom_id']:<4} .  "
            f"{fa['_atom_site.label_comp_id']:<4} {fa['_atom_site.label_asym_id']} 0 "
            f"{res_id:<4} .  {res_id:<4} {fa['_atom_site.auth_comp_id']:<4} "
            f"{fa['_atom_site.auth_asym_id']} {fa['_atom_site.auth_atom_id']:<4} "
            f"{atom_id:<5} {b_iso:<8} 1.0  ?  "
            f"{x:>12.5f} {y:>12.5f} {z:>12.5f} 1"
        )
        out_atom_lines.append(row_str)

    header_lines = [
        "data_protein_scope_256",
        "#",
        "_entry.id   protein_scope_256",
        struct_conf_block,
        "loop_",
    ] + out_headers

    full_content = "\n".join(header_lines) + "\n" + "\n".join(out_atom_lines) + "\n"
    out_final_cif.write_text(full_content, encoding="utf-8")
    print(
        f"  Created {out_final_cif.name} with {len(final_atoms)} atoms and {len(helices)} helices"
    )


def main() -> None:
    """Format all project page trajectories with element typing on the final frame."""
    repo_dir = Path(__file__).resolve().parent.parent
    trajectories_dir = repo_dir / "models/zatom2/docs/static/trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    src_dir = repo_dir / "logs/inference_outs/project_page_trajectories"

    print("Processing OMol25 biomolecule files...")
    mol_final_cif = trajectories_dir / "molecule_omol25_final.cif"
    copy_uncompressed_cif(
        src_dir / "molecule_omol25_0_model_0.cif.gz",
        mol_final_cif,
    )
    mol_atoms = parse_atom_identities(mol_final_cif)
    reverse_cif_models(
        src_dir / "molecule_omol25_0_denoised_model_0.cif.gz",
        trajectories_dir / "molecule_omol25_denoised.cif",
        final_atoms=mol_atoms,
    )
    reverse_cif_models(
        src_dir / "molecule_omol25_0_noisy_model_0.cif.gz",
        trajectories_dir / "molecule_omol25_noisy.cif",
        final_atoms=mol_atoms,
    )

    print("Processing OMat24 periodic material files...")
    mat_final_cif = trajectories_dir / "material_omat24_final.cif"
    copy_uncompressed_cif(
        src_dir / "material_omat24_0_model_0.cif.gz",
        mat_final_cif,
    )
    mat_atoms = parse_atom_identities(mat_final_cif)
    reverse_cif_models(
        src_dir / "material_omat24_0_denoised_model_0.cif.gz",
        trajectories_dir / "material_omat24_denoised.cif",
        final_atoms=mat_atoms,
    )
    reverse_cif_models(
        src_dir / "material_omat24_0_noisy_model_0.cif.gz",
        trajectories_dir / "material_omat24_noisy.cif",
        final_atoms=mat_atoms,
    )

    print("Processing SCOPe protein biomolecule files (256 residues)...")
    prot_dir = src_dir / "run_len256_exact"
    prot_final_gz = prot_dir / "inputs_scope_len256_1023_0_model_0.cif.gz"
    prot_denoised_gz = prot_dir / "inputs_scope_len256_1023_0_denoised_model_0.cif.gz"
    prot_noisy_gz = prot_dir / "inputs_scope_len256_1023_0_noisy_model_0.cif.gz"

    if prot_final_gz.exists() and prot_denoised_gz.exists() and prot_noisy_gz.exists():
        format_protein_final(
            prot_final_gz, trajectories_dir / "protein_scope_final.cif"
        )
        format_protein_trajectory(
            prot_final_gz,
            prot_denoised_gz,
            trajectories_dir / "protein_scope_denoised.cif",
            num_output_frames=49,
            is_denoised=True,
        )
        format_protein_trajectory(
            prot_final_gz,
            prot_noisy_gz,
            trajectories_dir / "protein_scope_noisy.cif",
            num_output_frames=49,
            is_denoised=False,
        )

    print("All trajectories successfully generated and formatted!")


if __name__ == "__main__":
    main()
