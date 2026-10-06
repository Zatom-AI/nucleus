import numpy as np
from biotite.structure import infer_elements
from foundry.utils.ddp import RankedLogger

try:
    from rfd3na.constants import VIRTUAL_ATOM_ELEMENT_NAME, association_schemes
    from rfd3na.trainer.trainer_utils import (
        _cleanup_virtual_atoms_and_assign_atom_name_elements as _upstream_cleanup_virtual_atoms,
    )
except ImportError:
    raise ImportError(
        "Failed to import RFD3NA modules. Please ensure that the `rc-foundry[all]` package is installed and up to date."
    )


global_logger = RankedLogger(__name__, rank_zero_only=False)


def cleanup_virtual_atoms_and_assign_atom_name_elements(
    atom_array, association_scheme: str = "atom14"
):
    """Local, vocab-robust wrapper around the upstream RFD3 cleanup.

    Follows the nucleus convention of importing upstream functionality and
    adapting only the part that needs tweaking. The upstream implementation
    (``_upstream_cleanup_virtual_atoms``) assumes every diffused residue's
    association scheme has exactly as many entries as the residue has atom
    slots — true only when the predicted residue type matches the modality the
    slots were featurized for.

    During pure unconditional de-novo generation the sequence head diffuses over
    the *whole* allowed vocabulary (protein + RNA + DNA, see ``LinearSequenceHead``),
    so an RNA-featurized token (23 atom slots) can be assigned a protein type
    whose atom23 scheme is only 14 entries (proteins=14, DNA=22, RNA=23). The
    emitted atom-name list then no longer matches the array length and upstream
    raises ``ValueError`` (before mutating the array). We delegate to upstream
    for every normal case and only fall back to per-residue reconciliation when
    that mismatch occurs, so generation stays over the full vocabulary —
    including RNA — without crashing.
    """
    try:
        return _upstream_cleanup_virtual_atoms(
            atom_array, association_scheme=association_scheme
        )
    except ValueError:
        global_logger.warning(
            "Upstream atom-name cleanup failed (cross-modality residue prediction "
            "during de-novo generation); reconciling scheme lengths to atom slot "
            "counts to keep generation over the full vocabulary."
        )

    ret_mask: list[bool] = []
    atom_names: list[str] = []
    # Marks residues whose predicted type is not in the scheme (invalid → UNK).
    invalid_mask: list[bool] = []

    res_ids = atom_array.res_id
    res_start_indices = np.concatenate(
        [[0], np.where(res_ids[1:] != res_ids[:-1])[0] + 1]
    )
    res_end_indices = np.concatenate([res_start_indices[1:], [len(res_ids)]])

    for start, end in zip(res_start_indices, res_end_indices):
        res_array = atom_array[start:end]
        n_slots = len(res_array)

        is_seq_known = all(
            np.array(res_array.is_motif_atom_with_fixed_seq, dtype=bool)
        ) or all(np.array(res_array.is_motif_atom_unindexed, dtype=bool))

        # ... If sequence is known for the original atom array, just keep it.
        if is_seq_known:
            ret_mask += [True] * n_slots
            invalid_mask += [False] * n_slots
            atom_names += res_array.gt_atom_name.tolist()
            continue

        res_name = res_array[0].res_name

        # ... Predicted a residue type with no scheme: mark invalid, keep as-is.
        if res_name not in association_schemes[association_scheme]:
            global_logger.warning(
                f"Model predicted residue type {res_name!r} not in "
                f"{association_scheme!r} scheme; assigning UNK residue token."
            )
            ret_mask += [True] * n_slots
            invalid_mask += [True] * n_slots
            atom_names += res_array.atom_name.tolist()
            continue

        # ... Use the predicted residue's scheme, reconciled to the slot count.
        scheme = association_schemes[association_scheme][res_name]
        slot_names = [
            item.strip() if item is not None else VIRTUAL_ATOM_ELEMENT_NAME
            for item in scheme
        ]
        slot_keep = [item is not None for item in scheme]

        if len(scheme) < n_slots:
            # Cross-modality prediction (e.g. protein type on an RNA slot): pad
            # the remaining slots with virtual atoms so lengths line up; padded
            # atoms are dropped by ret_mask below.
            pad = n_slots - len(scheme)
            slot_names += [VIRTUAL_ATOM_ELEMENT_NAME] * pad
            slot_keep += [False] * pad
        elif len(scheme) > n_slots:
            global_logger.warning(
                f"Predicted residue {res_name!r} scheme length ({len(scheme)}) "
                f"exceeds atom slot count ({n_slots}); truncating."
            )
            slot_names = slot_names[:n_slots]
            slot_keep = slot_keep[:n_slots]

        atom_names += slot_names
        ret_mask += slot_keep
        invalid_mask += [False] * n_slots

    if len(atom_names) != atom_array.array_length():
        # Should be unreachable after per-residue reconciliation; guard anyway.
        global_logger.warning(
            f"Atom name reconciliation failed ({len(atom_names)} vs "
            f"{atom_array.array_length()}); returning array unchanged."
        )
        return atom_array

    atom_array.atom_name = atom_names
    atom_array.element = np.where(
        atom_array.element == VIRTUAL_ATOM_ELEMENT_NAME,
        infer_elements(atom_names),
        atom_array.element,
    )
    invalid_mask = np.array(invalid_mask, dtype=bool)
    if invalid_mask.any():
        atom_array.res_name[invalid_mask] = np.array(["UNK"] * int(invalid_mask.sum()))
    return atom_array[np.array(ret_mask, dtype=bool)]
