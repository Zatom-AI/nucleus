"""Molecule-aware transforms for atomized examples."""

import numpy as np
import torch
from atomworks.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER, GAP
from atomworks.io.utils.atom_array_plus import insert_atoms
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import get_token_starts, spread_token_wise
from biotite import structure as struc
from foundry.common import exists

from nucleus.transforms.atomistic_property_transforms import (
    canonical_property_task_name,
    infer_property_task_name,
)
from nucleus.utils.forces import (
    FORCE_CONDITIONING_METHODS,
    FORCE_LOG_EPSILON,
    MEAN_FORCE_NORM_CONDITIONING,
    RMS_AND_P95_CONDITIONING,
    force_rms_and_p95,
    log_force_rms_and_p95,
    mean_force_norm,
)

try:
    from rfd3na.constants import (
        ATOM14_ATOM_NAMES,
        ATOM23_ATOM_NAMES_DNA,
        ATOM23_ATOM_NAMES_RNA,
        VIRTUAL_ATOM_ELEMENT_NAME,
    )
    from rfd3na.transforms.design_transforms import AddGroundTruthSequence
    from rfd3na.transforms.util_transforms import (
        assert_single_representative,
        get_af3_token_representative_masks,
    )
    from rfd3na.transforms.virtual_atoms import (
        generate_atom_mappings_,
        map_to_association_scheme,
        permute_symmetric_atom_names_,
    )
except ImportError:
    raise ImportError(
        "Could not import RFD3NA modules. Please ensure that the `rc-foundry[all]` package is installed and up to date."
    )


def _atomic_numbers_from_elements(elements: np.ndarray) -> np.ndarray:
    """
    Map element symbols to atomic numbers, using 0 for unknown elements.

    Args:
        elements: Array of element symbols (e.g., "C", "N", "O").

    Returns:
        Array of atomic numbers corresponding to the input elements.
    """

    atomic_numbers = []
    for element in elements:
        key = str(element).strip().upper()
        atomic_numbers.append(ELEMENT_NAME_TO_ATOMIC_NUMBER.get(key, 0))
    return np.asarray(atomic_numbers, dtype=np.int64)


def _token_atomic_numbers(atom_array) -> np.ndarray:
    """Return token-level atomic numbers, using zero for unknown elements."""
    token_starts = get_token_starts(atom_array)
    if "atomic_number" in atom_array.get_annotation_categories():
        atomic_numbers = np.asarray(atom_array.atomic_number, dtype=np.int64)
    else:
        atomic_numbers = _atomic_numbers_from_elements(atom_array.element)
    return atomic_numbers[token_starts]


def _molecule_atom_type_generation_mask(atom_array) -> np.ndarray:
    """Return token mask for atomized molecule atoms whose atom type is a target."""
    token_starts = get_token_starts(atom_array)
    token_array = atom_array[token_starts]
    return (
        np.asarray(token_array.is_ligand, dtype=bool)
        & np.asarray(token_array.atomize, dtype=bool)
        & ~np.asarray(token_array.is_motif_atom_with_fixed_seq, dtype=bool)
    )


def _force_array_from_ase_atoms(atoms):
    """Return retained ASE forces, or ``None`` when no forces are available."""
    calc = getattr(atoms, "calc", None)
    if calc is not None and "forces" in getattr(calc, "results", {}):
        return calc.results["forces"]
    try:
        return atoms.get_forces()
    except Exception:
        return None


def _log10_mean_force_norm(value) -> float:
    """Convert a raw mean force norm to the model's log10-scaled feature."""
    if hasattr(value, "item"):
        value = value.item()
    try:
        raw_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"mean_force_norm must be numeric; got {value!r}.") from exc
    if not np.isfinite(raw_value):
        raise ValueError(f"mean_force_norm must be finite; got {value!r}.")
    if raw_value < 0:
        raise ValueError(f"mean_force_norm must be non-negative; got {raw_value!r}.")
    return float(np.log10(max(raw_value, 1e-8)))


def _fix_multidimensional_annotations_in_pad_array(atomarray, padarray):
    """Normalize multidimensional annotation dtypes after Biotite array expansion."""
    for annotation in atomarray.get_annotation_categories():
        if len(atomarray.get_annotation(annotation).shape) > 1:
            stacked = np.stack(padarray.get_annotation(annotation)).astype(float)
            padarray.del_annotation(annotation)
            padarray.set_annotation(annotation, stacked)
    return padarray


class MoleculeAwarePadTokensWithVirtualAtoms(Transform):
    """
    RFD3NA token padding that leaves atomized molecule tokens as atom1 tokens.

    Protein residue tokens retain the RFD3NA atom14 behavior, and under the
    ``atom23`` association scheme nucleic-acid residues are padded to the NA
    layout (DNA -> 22 atoms, RNA -> 23 atoms, both centered on ``C1'``) exactly
    as in ``rfd3na``. Atomized ligand tokens are already one atom per token after
    ``AtomizeByCCDName`` and are kept unpadded so molecule generation uses
    atom-wise tokens. This unified handling allows protein/NA/ligand to be mixed
    in a single example (e.g. RNA-ligand co-generation).
    """

    requires_previous_transforms = ["UnindexFlaggedTokens"]

    def __init__(
        self,
        n_atoms_per_token,
        atom_to_pad_from,
        association_scheme,
    ):
        self.n_atoms_per_token = n_atoms_per_token
        self.atom_to_pad_from = atom_to_pad_from
        self.association_scheme = association_scheme
        if exists(association_scheme):
            self.association_map_, self.symmetry_map_ = generate_atom_mappings_(
                association_scheme
            )

    def forward(self, data: dict) -> dict:
        """Pad protein/NA residue tokens and preserve atomized ligand atom tokens."""
        atom_array = data["atom_array"]
        starts = get_token_starts(atom_array, add_exclusive_stop=True)
        token_starts = starts[:-1]
        token_level_array = atom_array[token_starts]
        is_motif_atom_with_fixed_seq = token_level_array.is_motif_atom_with_fixed_seq
        is_motif_token_unindexed = token_level_array.is_motif_atom_unindexed

        token_ids = np.unique(atom_array.token_id)
        assert len(token_ids) == len(
            is_motif_atom_with_fixed_seq
        ), "Token ids and token level array have different lengths!"

        # Atomized molecule tokens are kept as atom1 tokens and are never padded.
        is_atomized_molecule = np.asarray(
            token_level_array.is_ligand, dtype=bool
        ) & np.asarray(token_level_array.atomize, dtype=bool)
        token_lengths = np.diff(starts)
        if np.any(is_atomized_molecule & (token_lengths != 1)):
            raise ValueError(
                "Atomized molecule tokens must contain exactly one atom. "
                f"Got token lengths: {token_lengths[is_atomized_molecule].tolist()}"
            )

        # Unindexed tokens are never fully atomized, but may be assigned as atomized to have repr atoms:
        is_residue = (
            token_level_array.is_protein & ~token_level_array.atomize
        ) | is_motif_token_unindexed

        if self.association_scheme == "atom23":
            # Under atom23, DNA/RNA residues are also paddable residues (to 22/23 atoms).
            is_residue_na = (
                (token_level_array.is_dna | token_level_array.is_rna)
                & ~token_level_array.atomize
            ) | is_motif_token_unindexed
            is_residue = is_residue | is_residue_na

        # Unindexed tokens are never padded, and so are treated as residues with fixed sequence.
        is_paddable = is_residue & ~(
            is_motif_atom_with_fixed_seq | is_motif_token_unindexed
        )
        is_non_paddable_residue = is_residue & (
            is_motif_atom_with_fixed_seq | is_motif_token_unindexed
        )

        # Never pad atomized small-molecule tokens, even when unindexed, so they stay atom1.
        is_paddable = is_paddable & ~is_atomized_molecule
        is_non_paddable_residue = is_non_paddable_residue & ~is_atomized_molecule

        # Collect virtual atoms to insert (we will insert them all at once)
        virtual_atoms_to_insert = []
        insert_positions = []

        # First pass: collect virtual atoms for insertion
        for token_id, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
            if not is_paddable[token_id]:
                continue
            token = atom_array[start:end]

            # Per-token atom budget and central atom: NA uses C1' with 22 (DNA) / 23 (RNA) atoms.
            if self.association_scheme == "atom23" and atom_array[start].is_dna:
                n_atoms_per_token = 22
                central_atom = "C1'"
            elif self.association_scheme == "atom23" and atom_array[start].is_rna:
                n_atoms_per_token = 23
                central_atom = "C1'"
            else:
                n_atoms_per_token = self.n_atoms_per_token
                central_atom = self.atom_to_pad_from

            # First, pad with virtual atoms if needed
            n_pad = n_atoms_per_token - len(token)
            if n_pad <= 0:
                continue

            mask = get_af3_token_representative_masks(token, central_atom=central_atom)
            assert_single_representative(token, central_atom=central_atom)

            # ... Create virtual atoms
            pad_atoms = token[mask].copy()
            pad_atoms = (
                pad_atoms[0] if isinstance(pad_atoms, struc.AtomArray) else pad_atoms
            )
            pad_atoms.element = VIRTUAL_ATOM_ELEMENT_NAME

            # ... Expand to desired number of atoms
            pad_array = struc.array([pad_atoms] * n_pad)

            # ... Change occupancy | if any atom in the token has occupancy, set to 1.0
            occ = 1.0 if pad_atoms.occupancy.sum() > 0.0 else 0.0
            pad_array.occupancy = np.full(n_pad, occ)

            # ... Even if the input pad_atoms are all motif, we don't ever want padded atoms to be motif
            pad_array.is_motif_atom = np.zeros(n_pad, dtype=bool)

            # Handle multidimensional annotations
            pad_array = _fix_multidimensional_annotations_in_pad_array(token, pad_array)

            # Collect virtual atoms for later insertion
            virtual_atoms_to_insert.append(pad_array)
            insert_positions.append(end)

        # Insert all virtual atoms at once using insert_atoms
        if virtual_atoms_to_insert:
            atom_array_padded = insert_atoms(
                atom_array, virtual_atoms_to_insert, insert_positions
            )
        else:
            atom_array_padded = atom_array

        # Initialize gt_atom_name annotation if it doesn't exist
        if "gt_atom_name" not in atom_array_padded.get_annotation_categories():
            atom_array_padded.set_annotation(
                "gt_atom_name", np.empty(len(atom_array_padded), dtype="U4")
            )

        # Second pass: process tokens with proper atom name assignment after padding
        # Get updated token starts after padding
        starts_padded = get_token_starts(atom_array_padded, add_exclusive_stop=True)
        for token_id, (start, end) in enumerate(
            zip(starts_padded[:-1], starts_padded[1:])
        ):
            # Atom-name table depends on polymer type (atom23 NA vs atom14).
            if atom_array_padded[start].is_dna:
                atom_name_table = ATOM23_ATOM_NAMES_DNA
            elif atom_array_padded[start].is_rna:
                atom_name_table = ATOM23_ATOM_NAMES_RNA
            else:
                atom_name_table = ATOM14_ATOM_NAMES

            if is_paddable[token_id]:
                # ... Permutation of atom names during training
                if not data["is_inference"] and exists(self.association_scheme):
                    atom_names = permute_symmetric_atom_names_(
                        atom_name_table,
                        atom_array_padded.res_name[start],
                        association_map=self.association_map_,
                        symmetry_map=self.symmetry_map_,
                    )
                else:
                    atom_names = atom_name_table
                atom_array_padded.atom_name[start:end] = atom_names
                atom_array_padded.get_annotation("gt_atom_name")[start:end] = atom_names
            elif is_non_paddable_residue[token_id]:
                # When sequence-constrained, we want to directly map the residue name based on the sequence
                atom_names, res_name = (
                    atom_array_padded.atom_name[start:end],
                    atom_array_padded.res_name[start],
                )
                atom_array_padded.get_annotation("gt_atom_name")[start:end] = atom_names
                atom_names = map_to_association_scheme(
                    atom_names,
                    res_name,
                    scheme=self.association_scheme,
                    ATOM_NAMES=atom_name_table,
                )
                atom_array_padded.atom_name[start:end] = atom_names
            else:
                # ... Add gt_atom_name annotation to other tokens
                atom_names = atom_array_padded.atom_name[start:end]
                atom_array_padded.get_annotation("gt_atom_name")[start:end] = atom_names

            # ... Update atom array
            assert {VIRTUAL_ATOM_ELEMENT_NAME} != set(
                atom_array_padded.element[start:end].tolist()
            ), (
                "Padded atoms should be virtual atoms, but found: "
                f"{set(atom_array_padded.element[start:end].tolist())}"
            )

        data["atom_array"] = atom_array_padded
        return data


class EnsureMotifAtomAnnotation(Transform):
    """Ensure derived motif atom annotation exists for inference reference features."""

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        if "is_motif_atom" not in atom_array.get_annotation_categories():
            atom_array.set_annotation(
                "is_motif_atom",
                np.asarray(atom_array.is_motif_atom_with_fixed_seq, dtype=bool)
                | np.asarray(atom_array.is_motif_atom_with_fixed_coord, dtype=bool),
            )
        data["atom_array"] = atom_array
        return data


class AddMeanForceNormConditioning(Transform):
    """Add the configured force-statistic conditioning as token features.

    ``mean_force_norm`` preserves the original one-dimensional log10 feature.
    ``rms_and_p95`` computes ``F_RMS`` and the linearly interpolated 95th
    percentile of per-atom force magnitudes, then applies ``log(epsilon + F)``
    to each value (forces are numerically expressed in eV/Angstrom). The result
    is broadcast to ligand molecule tokens as the two-channel ``force_rms_p95``
    feature and is accompanied by ``has_force_rms_p95``.

    Inference accepts raw ``force_rms`` and ``force_p95`` values and requires
    ``has_force_rms_p95=true`` to enable conditioning on both.
    """

    def __init__(
        self,
        conditioning_probability: float = 1.0,
        conditioning_tasks: tuple[str, ...] | list[str] | None = None,
        conditioning_method: str = MEAN_FORCE_NORM_CONDITIONING,
        epsilon: float = FORCE_LOG_EPSILON,
    ):
        conditioning_probability = float(conditioning_probability)
        if not 0.0 <= conditioning_probability <= 1.0:
            raise ValueError(
                "conditioning_probability must be between 0.0 and 1.0; "
                f"got {conditioning_probability}."
            )
        self.conditioning_probability = conditioning_probability
        self.conditioning_tasks = (
            None
            if conditioning_tasks is None
            else frozenset(
                canonical_property_task_name(task) or str(task)
                for task in conditioning_tasks
            )
        )
        if conditioning_method not in FORCE_CONDITIONING_METHODS:
            raise ValueError(
                f"Unknown force conditioning method {conditioning_method!r}; "
                f"expected one of {sorted(FORCE_CONDITIONING_METHODS)}."
            )
        epsilon = float(epsilon)
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError(f"epsilon must be finite and positive; got {epsilon}.")
        self.conditioning_method = conditioning_method
        self.epsilon = epsilon

    @property
    def _feature_spec(self) -> tuple[str, str, int]:
        """Return value key, presence key, and channel count for this method."""
        if self.conditioning_method == RMS_AND_P95_CONDITIONING:
            return "force_rms_p95", "has_force_rms_p95", 2
        return "mean_force_norm", "has_mean_force_norm", 1

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        token_starts = get_token_starts(atom_array)
        n_tokens = len(token_starts)

        token_array = atom_array[token_starts]
        token_categories = set(token_array.get_annotation_categories())
        if "is_ligand" in token_categories:
            molecule_token_mask = torch.as_tensor(
                np.asarray(token_array.is_ligand, dtype=bool),
                dtype=torch.bool,
            )
        else:
            molecule_token_mask = torch.ones(n_tokens, dtype=torch.bool)

        feature_key, presence_key, n_channels = self._feature_spec
        condition = torch.zeros((n_tokens, n_channels), dtype=torch.float32)
        has_condition = torch.zeros((n_tokens, 1), dtype=torch.float32)
        feats = data.setdefault("feats", {})
        is_inference = data.get("is_inference", False)
        if self.conditioning_tasks is not None and (
            infer_property_task_name(data) not in self.conditioning_tasks
        ):
            feats[feature_key] = condition
            feats[presence_key] = has_condition
            return data
        if not is_inference and (
            self.conditioning_probability == 0.0
            or (
                self.conditioning_probability < 1.0
                and np.random.random() >= self.conditioning_probability
            )
        ):
            feats[feature_key] = condition
            feats[presence_key] = has_condition
            return data

        specification = data.get("specification", {})
        if isinstance(specification, dict):
            extra = specification.get("extra")
            if not isinstance(extra, dict):
                extra = {}
            if self.conditioning_method == RMS_AND_P95_CONDITIONING:
                force_rms = specification.get("force_rms", extra.get("force_rms"))
                force_p95 = specification.get("force_p95", extra.get("force_p95"))
                has_flag = (
                    "has_force_rms_p95" in specification or "has_force_rms_p95" in extra
                )
                enabled = specification.get(
                    "has_force_rms_p95",
                    extra.get("has_force_rms_p95", False),
                )
                if enabled:
                    if force_rms is None or force_p95 is None:
                        raise ValueError(
                            "has_force_rms_p95=True requires both force_rms and "
                            "force_p95 to be provided."
                        )
                    transformed = log_force_rms_and_p95(
                        force_rms,
                        force_p95,
                        epsilon=self.epsilon,
                    )
                    condition[molecule_token_mask] = torch.tensor(
                        transformed, dtype=condition.dtype
                    )
                    has_condition[molecule_token_mask, 0] = 1.0
                    feats[feature_key] = condition
                    feats[presence_key] = has_condition
                    return data
                if has_flag or force_rms is not None or force_p95 is not None:
                    feats[feature_key] = condition
                    feats[presence_key] = has_condition
                    return data
            else:
                mean_force_norm_value = specification.get("mean_force_norm")
                if mean_force_norm_value is None:
                    mean_force_norm_value = extra.get("mean_force_norm")
                has_flag = (
                    "has_mean_force_norm" in specification
                    or "has_mean_force_norm" in extra
                )
                enabled = specification.get(
                    "has_mean_force_norm",
                    extra.get("has_mean_force_norm", False),
                )
                if enabled:
                    if mean_force_norm_value is None:
                        raise ValueError(
                            "has_mean_force_norm=True requires mean_force_norm to be "
                            "provided."
                        )
                    condition[molecule_token_mask, 0] = _log10_mean_force_norm(
                        mean_force_norm_value
                    )
                    has_condition[molecule_token_mask, 0] = 1.0
                    feats[feature_key] = condition
                    feats[presence_key] = has_condition
                    return data
                if has_flag or mean_force_norm_value is not None:
                    feats[feature_key] = condition
                    feats[presence_key] = has_condition
                    return data

        atoms = data.get("atoms")
        if atoms is not None:
            row_id = data.get("example_id", data.get("id", "unknown"))
            force_array = _force_array_from_ase_atoms(atoms)
            if force_array is not None:
                if self.conditioning_method == RMS_AND_P95_CONDITIONING:
                    force_rms, force_p95 = force_rms_and_p95(
                        force_array,
                        row_id=row_id,
                    )
                    transformed = log_force_rms_and_p95(
                        force_rms,
                        force_p95,
                        epsilon=self.epsilon,
                    )
                    condition[molecule_token_mask] = torch.tensor(
                        transformed, dtype=condition.dtype
                    )
                else:
                    condition[molecule_token_mask, 0] = _log10_mean_force_norm(
                        mean_force_norm(force_array, row_id=row_id)
                    )
                has_condition[molecule_token_mask, 0] = 1.0

        feats[feature_key] = condition
        feats[presence_key] = has_condition
        return data


class AddUnifiedTokenTargets(AddGroundTruthSequence):
    """
    Add protein sequence targets and molecule atom-type targets.

    Protein sequence supervision follows RFD3NA. Atomized molecule tokens use a
    disabled sequence target and an atomic-number target when their atom type is
    not fixed by motif annotations.
    """

    def forward(self, data: dict) -> dict:
        """Add unified token targets and mask reference features for generated atoms."""
        atom_array = data["atom_array"]
        token_starts = get_token_starts(atom_array)
        token_array = atom_array[token_starts]

        res_names = atom_array.res_name[token_starts].copy()
        is_atomized_molecule = np.asarray(
            token_array.is_ligand, dtype=bool
        ) & np.asarray(token_array.atomize, dtype=bool)

        sequence_res_names = res_names.copy()
        sequence_res_names[is_atomized_molecule] = GAP
        restype = self.sequence_encoding.encode(sequence_res_names)

        sequence_valid_mask = ~np.isin(res_names, ["UNK", "X", "DX", "<G>"])
        sequence_valid_mask &= ~is_atomized_molecule

        token_atomic_numbers = _token_atomic_numbers(atom_array)
        generation_mask = _molecule_atom_type_generation_mask(atom_array)
        atom_type_valid_mask = generation_mask & (token_atomic_numbers > 0)

        data.setdefault("ground_truth", {})
        data["ground_truth"]["sequence_gt_I"] = torch.from_numpy(restype)
        data["ground_truth"]["sequence_valid_mask"] = torch.from_numpy(
            sequence_valid_mask
        )
        data["ground_truth"]["atom_type_gt_I"] = torch.from_numpy(token_atomic_numbers)
        data["ground_truth"]["atom_type_valid_mask"] = torch.from_numpy(
            atom_type_valid_mask
        )

        data.setdefault("feats", {})
        data["feats"]["atom_type_generation_mask"] = generation_mask

        feats = data["feats"]
        atom_generation_mask = spread_token_wise(atom_array, generation_mask)
        if "ref_atom_name_chars" in feats:
            ref_atom_name_chars = feats["ref_atom_name_chars"].copy()
            ref_atom_name_chars[atom_generation_mask] = 0
            feats["ref_atom_name_chars"] = ref_atom_name_chars
        if "ref_element" in feats:
            ref_element = feats["ref_element"].copy()
            ref_element[atom_generation_mask] = 0
            feats["ref_element"] = ref_element
        if "token_bonds" in feats:
            token_bonds = feats["token_bonds"].copy()
            token_bonds[generation_mask, :] = 0
            token_bonds[:, generation_mask] = 0
            feats["token_bonds"] = token_bonds

        data["feats"] = feats
        return data
