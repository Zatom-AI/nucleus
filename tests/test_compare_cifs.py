from pathlib import Path

import numpy as np
from biotite.structure.io.pdbx import CIFFile, get_structure, set_structure

from nucleus.scripts.compare_cifs import compare_cifs


def test_compare_cifs_handles_periodic_translation(tmp_path):
    """The comparison should be invariant to a uniform periodic shift."""
    reference = (
        Path(__file__).resolve().parents[0] / "examples/input_cifs/train_0001.cif"
    )
    shifted = tmp_path / "train_0001_shifted.cif"

    cif = CIFFile.read(str(reference))
    stack = get_structure(cif)
    shift = np.array([0.31, -0.17, 0.42], dtype=float)
    stack.coord[0] += shift @ stack.box[0]
    shifted_cif = CIFFile()
    set_structure(shifted_cif, stack)
    shifted_cif.write(str(shifted))

    report = compare_cifs(reference, shifted)

    assert report.n_atoms_reference == report.n_atoms_candidate
    assert report.formula_reference == report.formula_candidate
    assert report.matched_rmsd_post_align < 1e-6
    assert report.max_atom_displacement_post_align < 1e-6
