"""Nucleus training condition adapters."""

import numpy as np
from atomworks.ml.utils.token import get_token_starts, spread_token_wise

from nucleus.constants import MOLECULE_CONDITIONING_MODE

try:
    from rfd3na.transforms.training_conditions import IslandCondition, random_condition
except ImportError:
    raise ImportError(
        "Could not import RFD3NA training conditions. Please ensure that the `rc-foundry[all]` package is installed and up to date."
    )


class LigandAwareIslandCondition(IslandCondition):
    """RFD3NA island condition valid for protein/NA polymer and ligand-only examples."""

    MOLECULE_ONLY_MODES = {
        "unconditional_design",
        "structure_fixed_sequence_design",
        "sequence_fixed_structure_design",
        "mlip",
    }

    def __init__(
        self,
        *args,
        molecule_conditioning_mode: MOLECULE_CONDITIONING_MODE,
        p_protein_conditioned_molecule_design: float = 0.5,
        p_nucleotide_conditioned_molecule_design: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.molecule_conditioning_mode = molecule_conditioning_mode
        self.p_protein_conditioned_molecule_design = (
            p_protein_conditioned_molecule_design
        )
        self.p_nucleotide_conditioned_molecule_design = (
            p_nucleotide_conditioned_molecule_design
        )

    @staticmethod
    def _entity_presence(atom_array):
        has_protein = bool(np.any(atom_array.is_protein))
        has_ligand = bool(np.any(atom_array.is_ligand))
        has_nucleotide = bool(np.any(atom_array.is_dna | atom_array.is_rna))
        has_polymer = has_protein or has_nucleotide
        return has_protein, has_ligand, has_nucleotide, has_polymer

    @staticmethod
    def _is_valid_for_named_condition(
        condition_name,
        *,
        has_protein,
        has_ligand,
        has_nucleotide,
        has_polymer,
    ):
        if condition_name == "protein_conditioned_molecule":
            return has_protein and has_ligand and not has_nucleotide
        if condition_name == "nucleotide_conditioned_molecule":
            return has_nucleotide and has_ligand and not has_protein
        if condition_name == "polymer_conditioned_molecule":
            return has_ligand and has_polymer
        if condition_name == "protein_conditioned_nucleotide":
            return has_protein and has_nucleotide and not has_ligand
        if condition_name == "nucleotide_conditioned_protein":
            return has_protein and has_nucleotide and not has_ligand
        if condition_name == "ligand_conditioned_nucleotide":
            return has_ligand and has_nucleotide and not has_protein
        if condition_name == "ligand_conditioned_protein":
            return has_ligand and has_protein and not has_nucleotide
        if condition_name == "protein_ligand_conditioned_nucleotide":
            return has_protein and has_ligand and has_nucleotide
        if condition_name == "nucleotide_ligand_conditioned_protein":
            return has_protein and has_ligand and has_nucleotide
        return None

    def is_valid_for_example(self, data) -> bool:
        """Returns True if the example is valid for this training condition."""
        if self.name == "structure_prediction":
            has_disconnected_fragments = data.get("has_disconnected_fragments", False)
            if not isinstance(has_disconnected_fragments, (bool, np.bool_)):
                raise ValueError("has_disconnected_fragments must be a scalar boolean.")
            if has_disconnected_fragments:
                return False

        has_protein, has_ligand, has_nucleotide, has_polymer = self._entity_presence(
            data["atom_array"]
        )
        named_condition_valid = self._is_valid_for_named_condition(
            self.name,
            has_protein=has_protein,
            has_ligand=has_ligand,
            has_nucleotide=has_nucleotide,
            has_polymer=has_polymer,
        )
        if named_condition_valid is not None:
            return named_condition_valid

        if has_ligand and not has_polymer:
            return self.molecule_conditioning_mode in self.MOLECULE_ONLY_MODES

        # Delegate to the base condition, which accepts protein (and DNA/RNA under
        # the atom23 association scheme). The ligand-only case is handled above so
        # polymer-context modes cannot defer a deterministic error until sampling.
        if super().is_valid_for_example(data):
            return True
        return False

    def _set_conditioning_annotations(
        self,
        atom_array,
        *,
        is_motif_token,
        is_motif_atom,
        fixed_seq,
        fixed_coord,
        unindexed=None,
    ):
        """Set required conditioning annotations from atom-level masks."""
        if unindexed is None:
            unindexed = np.zeros(atom_array.array_length(), dtype=bool)
        atom_array.set_annotation("is_motif_token", is_motif_token)
        atom_array.set_annotation("is_motif_atom", is_motif_atom)
        atom_array.set_annotation("is_motif_atom_with_fixed_seq", fixed_seq)
        atom_array.set_annotation("is_motif_atom_with_fixed_coord", fixed_coord)
        atom_array.set_annotation("is_motif_atom_unindexed", unindexed)
        atom_array.set_annotation(
            "is_motif_atom_unindexed_motif_breakpoint",
            np.zeros(atom_array.array_length(), dtype=bool),
        )
        return atom_array

    def _sample_molecule_only(self, atom_array):
        """Sample molecule generation conditions for ligand-only examples."""
        false_atom = np.zeros(atom_array.array_length(), dtype=bool)
        token_level_array = atom_array[get_token_starts(atom_array)]
        all_tokens = np.ones(token_level_array.array_length(), dtype=bool)

        mode = self.molecule_conditioning_mode
        if mode == "unconditional_design":
            motif_tokens = np.zeros_like(all_tokens, dtype=bool)
            fixed_seq = false_atom.copy()
            fixed_coord = false_atom.copy()
        elif mode == "structure_fixed_sequence_design":
            motif_tokens = all_tokens
            fixed_seq = false_atom.copy()
            fixed_coord = np.ones(atom_array.array_length(), dtype=bool)
        elif mode == "sequence_fixed_structure_design":
            motif_tokens = all_tokens
            fixed_seq = np.ones(atom_array.array_length(), dtype=bool)
            fixed_coord = false_atom.copy()
        elif mode == "mlip":
            # A clean property-prediction task: exact atom types and coordinates
            # are inputs, so neither generative branch contributes a target.
            motif_tokens = all_tokens
            fixed_seq = np.ones(atom_array.array_length(), dtype=bool)
            fixed_coord = np.ones(atom_array.array_length(), dtype=bool)
        else:
            raise ValueError(f"Invalid molecule_conditioning_mode: {mode}")

        is_motif_token = spread_token_wise(atom_array, motif_tokens)
        is_motif_atom = fixed_seq | fixed_coord
        return self._set_conditioning_annotations(
            atom_array,
            is_motif_token=is_motif_token,
            is_motif_atom=is_motif_atom,
            fixed_seq=fixed_seq,
            fixed_coord=fixed_coord,
        )

    def _sample_protein_conditioned_molecule(self, atom_array):
        """Fix protein context while diffusing molecule coordinates and atom types."""
        is_protein = np.asarray(atom_array.is_protein, dtype=bool)
        false_atom = np.zeros(atom_array.array_length(), dtype=bool)
        return self._set_conditioning_annotations(
            atom_array,
            is_motif_token=is_protein.copy(),
            is_motif_atom=is_protein.copy(),
            fixed_seq=is_protein.copy(),
            fixed_coord=is_protein.copy(),
            unindexed=false_atom,
        )

    def _sample_nucleotide_conditioned_molecule(self, atom_array):
        """Fix nucleotide (DNA/RNA) context while diffusing molecule/material coords and atom types."""
        is_nucleotide = np.asarray(atom_array.is_dna | atom_array.is_rna, dtype=bool)
        false_atom = np.zeros(atom_array.array_length(), dtype=bool)
        return self._set_conditioning_annotations(
            atom_array,
            is_motif_token=is_nucleotide.copy(),
            is_motif_atom=is_nucleotide.copy(),
            fixed_seq=is_nucleotide.copy(),
            fixed_coord=is_nucleotide.copy(),
            unindexed=false_atom,
        )

    def _sample_polymer_conditioned_molecule(self, atom_array):
        """Fix protein and nucleotide context while diffusing molecule coordinates and atom types."""
        is_polymer = np.asarray(
            atom_array.is_protein | atom_array.is_dna | atom_array.is_rna, dtype=bool
        )
        false_atom = np.zeros(atom_array.array_length(), dtype=bool)
        return self._set_conditioning_annotations(
            atom_array,
            is_motif_token=is_polymer.copy(),
            is_motif_atom=is_polymer.copy(),
            fixed_seq=is_polymer.copy(),
            fixed_coord=is_polymer.copy(),
            unindexed=false_atom,
        )

    def _sample_protein_conditioned_nucleotide(self, atom_array):
        """Fix protein context while diffusing nucleotide coordinates and sequence."""
        is_protein = np.asarray(atom_array.is_protein, dtype=bool)
        false_atom = np.zeros(atom_array.array_length(), dtype=bool)
        return self._set_conditioning_annotations(
            atom_array,
            is_motif_token=is_protein.copy(),
            is_motif_atom=is_protein.copy(),
            fixed_seq=is_protein.copy(),
            fixed_coord=is_protein.copy(),
            unindexed=false_atom,
        )

    def _sample_nucleotide_conditioned_protein(self, atom_array):
        """Fix nucleotide context while diffusing protein coordinates and sequence."""
        is_nucleotide = np.asarray(atom_array.is_dna | atom_array.is_rna, dtype=bool)
        false_atom = np.zeros(atom_array.array_length(), dtype=bool)
        return self._set_conditioning_annotations(
            atom_array,
            is_motif_token=is_nucleotide.copy(),
            is_motif_atom=is_nucleotide.copy(),
            fixed_seq=is_nucleotide.copy(),
            fixed_coord=is_nucleotide.copy(),
            unindexed=false_atom,
        )

    def _sample_ligand_conditioned_design(self, atom_array):
        """Fix ligand context while diffusing the requested polymer entity."""
        is_ligand = np.asarray(atom_array.is_ligand, dtype=bool)
        false_atom = np.zeros(atom_array.array_length(), dtype=bool)
        return self._set_conditioning_annotations(
            atom_array,
            is_motif_token=is_ligand.copy(),
            is_motif_atom=is_ligand.copy(),
            fixed_seq=is_ligand.copy(),
            fixed_coord=is_ligand.copy(),
            unindexed=false_atom,
        )

    def _sample_protein_ligand_conditioned_nucleotide(self, atom_array):
        """Fix protein and ligand context while diffusing nucleotide coordinates and sequence."""
        is_protein_ligand = np.asarray(atom_array.is_protein, dtype=bool) | np.asarray(
            atom_array.is_ligand, dtype=bool
        )
        false_atom = np.zeros(atom_array.array_length(), dtype=bool)
        return self._set_conditioning_annotations(
            atom_array,
            is_motif_token=is_protein_ligand.copy(),
            is_motif_atom=is_protein_ligand.copy(),
            fixed_seq=is_protein_ligand.copy(),
            fixed_coord=is_protein_ligand.copy(),
            unindexed=false_atom,
        )

    def _sample_nucleotide_ligand_conditioned_protein(self, atom_array):
        """Fix nucleotide and ligand context while diffusing protein coordinates and sequence."""
        is_nucleotide_ligand = np.asarray(
            atom_array.is_dna | atom_array.is_rna, dtype=bool
        ) | np.asarray(atom_array.is_ligand, dtype=bool)
        false_atom = np.zeros(atom_array.array_length(), dtype=bool)
        return self._set_conditioning_annotations(
            atom_array,
            is_motif_token=is_nucleotide_ligand.copy(),
            is_motif_atom=is_nucleotide_ligand.copy(),
            fixed_seq=is_nucleotide_ligand.copy(),
            fixed_coord=is_nucleotide_ligand.copy(),
            unindexed=false_atom,
        )

    def sample(self, data):
        """Sample protein or molecule-aware motif/fixed annotations."""
        atom_array = data["atom_array"]
        has_protein, has_ligand, has_nucleotide, has_polymer = self._entity_presence(
            atom_array
        )
        # Nucleic acids count as polymers, so an NA(+ligand) example must go
        # through the polymer island-sampling path, not the molecule-only path.

        if has_ligand and not has_polymer:
            # ..., e.g., training using OMol25
            return self._sample_molecule_only(atom_array)
        if (
            has_ligand
            and has_polymer
            and self.molecule_conditioning_mode == "polymer_conditioned_molecule_design"
        ):
            return self._sample_polymer_conditioned_molecule(atom_array)
        if (
            has_ligand
            and has_protein
            and not has_nucleotide
            and self.molecule_conditioning_mode == "protein_conditioned_molecule_design"
            and random_condition(self.p_protein_conditioned_molecule_design)
        ):
            # ..., e.g., training using PDB
            return self._sample_protein_conditioned_molecule(atom_array)
        if (
            has_ligand
            and has_nucleotide
            and not has_protein
            and self.molecule_conditioning_mode
            == "nucleotide_conditioned_molecule_design"
            and random_condition(self.p_nucleotide_conditioned_molecule_design)
        ):
            # Fix the nucleotide (DNA/RNA) context and generate the molecule/material
            return self._sample_nucleotide_conditioned_molecule(atom_array)
        if (
            has_protein
            and has_nucleotide
            and not has_ligand
            and self.molecule_conditioning_mode
            == "protein_conditioned_nucleotide_design"
        ):
            return self._sample_protein_conditioned_nucleotide(atom_array)
        if (
            has_protein
            and has_nucleotide
            and not has_ligand
            and self.molecule_conditioning_mode
            == "nucleotide_conditioned_protein_design"
        ):
            return self._sample_nucleotide_conditioned_protein(atom_array)
        if (
            has_ligand
            and has_nucleotide
            and not has_protein
            and self.molecule_conditioning_mode
            == "ligand_conditioned_nucleotide_design"
        ):
            return self._sample_ligand_conditioned_design(atom_array)
        if (
            has_ligand
            and has_protein
            and not has_nucleotide
            and self.molecule_conditioning_mode == "ligand_conditioned_protein_design"
        ):
            return self._sample_ligand_conditioned_design(atom_array)
        if (
            has_protein
            and has_ligand
            and has_nucleotide
            and self.molecule_conditioning_mode
            == "protein_ligand_conditioned_nucleotide_design"
        ):
            return self._sample_protein_ligand_conditioned_nucleotide(atom_array)
        if (
            has_protein
            and has_ligand
            and has_nucleotide
            and self.molecule_conditioning_mode
            == "nucleotide_ligand_conditioned_protein_design"
        ):
            return self._sample_nucleotide_ligand_conditioned_protein(atom_array)

        # ..., e.g., training using PDB (and protein/NA polymers in general)
        return super().sample(data)
