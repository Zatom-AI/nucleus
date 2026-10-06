import re

from atomworks.constants import AF3_EXCLUDED_LIGANDS, CRYSTALLIZATION_AIDS
from beartype.typing import Literal

_RNA_CRYSTALLIZATION_METALS = [
    "MG",  # magnesium ion — ubiquitous in RNA crystal structures
    "ZN",  # zinc ion
    "CA",  # calcium ion
    "MN",  # manganese ion
    "K",  # potassium ion
    "CO",  # cobalt ion (cobalt hexammine used in RNA crystallography)
]

RNA_EXCLUDED_CCDS: list[str] = list(
    dict.fromkeys(
        CRYSTALLIZATION_AIDS + AF3_EXCLUDED_LIGANDS + _RNA_CRYSTALLIZATION_METALS
    )
)
"""AtomWorks and AF3 exclusions extended with common RNA crystallization metals."""

RNA_EXCLUDED_CCDS_REGEX = (
    rf"(?:^|,)\s*(?:{'|'.join(re.escape(ccd) for ccd in RNA_EXCLUDED_CCDS)})"
    r"\s*(?:,|$)"
)
"""Regex matching the same comma-separated CCD names removed by the PDB parser."""

UNCONDITIONAL_PROPERTY_TASK_ID = 0
STRUCTURE_PREDICTION_PROPERTY_TASK_ID = 1
MLIP_PROPERTY_TASK_ID = 2
UNKNOWN_PROPERTY_TASK_ID = -1

MOLECULE_CONDITIONING_MODE = Literal[
    "unconditional_design",
    "structure_fixed_sequence_design",
    "sequence_fixed_structure_design",
    "mlip",
    "protein_conditioned_molecule_design",
    "nucleotide_conditioned_molecule_design",
    "polymer_conditioned_molecule_design",
    "protein_conditioned_nucleotide_design",
    "nucleotide_conditioned_protein_design",
    "ligand_conditioned_nucleotide_design",
    "ligand_conditioned_protein_design",
    "protein_ligand_conditioned_nucleotide_design",
    "nucleotide_ligand_conditioned_protein_design",
]
