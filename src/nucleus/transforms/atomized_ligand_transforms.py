"""Fast transforms for ASE-loaded, fully atomized ligand-only examples."""

from typing import Any

import numpy as np
import torch
from atomworks.ml.transforms.atom_array import get_within_entity_idx
from atomworks.ml.transforms.base import Transform

try:
    from rfd3na.constants import VIRTUAL_ATOM_ELEMENT_NAME
except ImportError:
    raise ImportError(
        "Could not import RFD3NA constants. Please ensure that the "
        "`rc-foundry[all]` package is installed and up to date."
    )


class InitializeAtomizedLigand(Transform):
    """Initialize annotations shared by the OMat24 and OMol25 loaders.

    The two ASE loaders create one resolved ``LIG`` residue with no polymer
    atoms.  The unified pipeline normally discovers these invariants
    through many protein/NA-oriented transforms.  This transform validates the
    loader contract once and initializes the same annotations directly.
    """

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        n_atoms = len(atom_array)

        if n_atoms == 0:
            raise ValueError("Atomized-ligand examples must contain at least one atom.")
        if np.any(atom_array.is_polymer):
            raise ValueError(
                "The atomized-ligand fast path does not support polymer atoms."
            )
        if not np.all(atom_array.res_name == "LIG"):
            raise ValueError(
                "The atomized-ligand fast path requires ASE loader res_name='LIG'."
            )
        if (
            len(np.unique(atom_array.chain_iid)) != 1
            or len(np.unique(atom_array.res_id)) != 1
        ):
            raise ValueError(
                "The atomized-ligand fast path requires one ASE-loaded LIG residue."
            )
        if not np.all(atom_array.occupancy > 0.0):
            raise ValueError(
                "The atomized-ligand fast path requires fully resolved atoms."
            )

        false = np.zeros(n_atoms, dtype=bool)
        true = np.ones(n_atoms, dtype=bool)

        atom_array.set_annotation("is_protein", false.copy())
        atom_array.set_annotation("is_rna", false.copy())
        atom_array.set_annotation("is_dna", false.copy())
        atom_array.set_annotation("is_ligand", true.copy())
        atom_array.set_annotation("is_residue", false.copy())
        atom_array.set_annotation("is_covalent_modification", false.copy())
        atom_array.set_annotation("atomize", true.copy())
        atom_array.set_annotation("atom_id", np.arange(n_atoms, dtype=np.uint32))
        atom_array.set_annotation(
            "within_chain_res_idx",
            np.zeros(n_atoms, dtype=np.int32),
        )
        atom_array.set_annotation(
            "within_poly_res_idx", np.full(n_atoms, -1, dtype=np.int32)
        )
        atom_array.set_annotation("is_N_terminus", false.copy())
        atom_array.set_annotation("is_C_terminus", false.copy())
        atom_array.set_annotation(
            "gt_atom_name", np.asarray(atom_array.atom_name, dtype="U4")
        )

        data["atom_array"] = atom_array
        # The generic NA geometry transform initializes this output even for
        # ligand-only examples. Keep the public pipeline schema identical.
        data.setdefault("log_dict", {})
        return data


class FastAtomizedLigandUnindex(Transform):
    """Create unindexing outputs when every ligand token is already atomized."""

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        n_tokens = len(atom_array)
        is_unindexed = np.asarray(atom_array.is_motif_atom_unindexed, dtype=bool).copy()

        if np.any(is_unindexed):
            raise ValueError(
                "Unindexed atomized-ligand tokens are unsupported by the fast path."
            )

        atom_array.set_annotation("orig_res_id", atom_array.res_id.copy())
        atom_array.set_annotation("token_id", np.arange(n_tokens, dtype=np.uint32))
        data.setdefault("ground_truth", {})["is_original_unindexed_token"] = np.zeros(
            n_tokens, dtype=bool
        )
        feats = data.setdefault("feats", {})
        feats["unindexing_pair_mask"] = np.zeros((n_tokens, n_tokens), dtype=bool)
        feats["is_motif_token_unindexed"] = is_unindexed
        data["atom_array"] = atom_array
        return data


class EncodeAtomizedLigandTokenFeatures(Transform):
    """Encode AF3 token features without rediscovering one-atom tokens."""

    def __init__(self, sequence_encoding, encode_residues_to: int | None = None):
        self.sequence_encoding = sequence_encoding
        self.encode_residues_to = encode_residues_to

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        n_tokens = len(atom_array)

        residue_index = atom_array.within_chain_res_idx.copy()
        token_index = np.arange(n_tokens)
        asym_name, asym_id = np.unique(atom_array.pn_unit_iid, return_inverse=True)
        entity_name, entity_id = np.unique(
            atom_array.pn_unit_entity, return_inverse=True
        )
        sym_name, sym_id = get_within_entity_idx(atom_array, level="pn_unit")

        res_names = atom_array.res_name.copy()
        if self.encode_residues_to is not None:
            is_masked = ~np.asarray(atom_array.is_motif_atom_with_fixed_seq, dtype=bool)
            res_names[is_masked] = self.encode_residues_to
        encoded_sequence = self.sequence_encoding.encode(res_names)
        restype = np.zeros((n_tokens, self.sequence_encoding.n_tokens), dtype=np.int64)
        restype[np.arange(n_tokens), encoded_sequence] = 1

        false = np.zeros(n_tokens, dtype=bool)
        feats = data.setdefault("feats", {})
        feats.update(
            {
                "residue_index": residue_index,
                "token_index": token_index,
                "asym_id": asym_id,
                "entity_id": entity_id,
                "sym_id": sym_id,
                "restype": restype,
                "is_protein": false.copy(),
                "is_rna": false.copy(),
                "is_dna": false.copy(),
                "is_ligand": np.ones(n_tokens, dtype=bool),
                "terminus_type": np.zeros((n_tokens, 2), dtype=np.int64),
                "is_polar": false.copy(),
            }
        )
        data["encoded"] = {"seq": encoded_sequence}
        data.setdefault("feat_metadata", {}).update(
            {
                "asym_name": asym_name,
                "entity_name": entity_name,
                "sym_name": sym_name,
            }
        )
        return data


class AddAtomizedLigandIsXFeatures(Transform):
    """Create the standard IsX features for one-atom ligand tokens."""

    def __init__(self, X, central_atom: str):
        self.X = X
        self.central_atom = central_atom

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        n_atoms = len(atom_array)
        true = np.ones(n_atoms, dtype=bool)
        false = np.zeros(n_atoms, dtype=bool)
        atom_array.set_annotation("is_backbone", true.copy())
        atom_array.set_annotation("is_sidechain", false.copy())

        feats = data.setdefault("feats", {})
        if "is_backbone" in self.X:
            feats["is_backbone"] = torch.ones(n_atoms, dtype=torch.bool)
        if "is_sidechain" in self.X:
            feats["is_sidechain"] = torch.zeros(n_atoms, dtype=torch.bool)
        if "is_virtual" in self.X:
            feats["is_virtual"] = atom_array.element == VIRTUAL_ATOM_ELEMENT_NAME

        for feature_name in (
            "is_motif_atom_with_fixed_coord",
            "is_motif_atom_with_fixed_seq",
            "is_motif_atom_unindexed",
        ):
            if feature_name in self.X:
                feats[feature_name] = np.asarray(
                    atom_array.get_annotation(feature_name), dtype=bool
                ).copy()

        if "is_motif_token_with_fully_fixed_coord" in self.X:
            feats["is_motif_token_with_fully_fixed_coord"] = np.asarray(
                atom_array.is_motif_atom_with_fixed_coord, dtype=bool
            ).copy()
        if "is_central" in self.X:
            feats["is_central"] = true.copy()
        if "is_ca" in self.X:
            feats["is_ca"] = true.copy()

        data["atom_array"] = atom_array
        return data


class FeaturizeUnconditionedAtomizedLigandAtoms(Transform):
    """Create the zero RASA/H-bond features used by material pretraining."""

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        n_atoms = len(data["atom_array"])
        feats = data.setdefault("feats", {})
        feats["ref_atomwise_rasa"] = np.zeros((n_atoms, 3), dtype=np.int64)
        feats["active_donor"] = torch.zeros(n_atoms, dtype=torch.int64)
        feats["active_acceptor"] = torch.zeros(n_atoms, dtype=torch.int64)
        return data


class AddAtomizedLigandAdditional1dFeatures(Transform):
    """Add configured 1D features without token-wise Biotite aggregation."""

    def __init__(
        self,
        token_1d_features,
        atom_1d_features,
        autofill_zeros_if_not_present_in_atomarray: bool = False,
        association_scheme: str = "atom14",
    ):
        self.autofill = autofill_zeros_if_not_present_in_atomarray
        self.token_1d_features = token_1d_features
        self.atom_1d_features = atom_1d_features
        self.association_scheme = association_scheme

    def _add_feature(
        self,
        data: dict[str, Any],
        feature_name: str,
        n_dims: int,
        feature_type: str,
    ) -> None:
        if feature_name in data["feats"]:
            return

        atom_array = data["atom_array"]
        if feature_name in atom_array.get_annotation_categories():
            feature_array = torch.as_tensor(
                atom_array.get_annotation(feature_name)
            ).float()
            if feature_array.ndim == 1 and n_dims == 1:
                feature_array = feature_array.unsqueeze(1)
            elif feature_array.ndim != 2:
                raise ValueError(
                    f"{feature_type} 1d_feature `{feature_name}` must be a 2d "
                    f"array, got {feature_array.ndim}d."
                )
            if feature_array.shape[1] != n_dims:
                raise ValueError(
                    f"{feature_type} 1d_feature `{feature_name}` dimensions in "
                    f"atomarray ({feature_array.shape[-1]}) does not match "
                    f"dimension declared in config ({n_dims})."
                )
        elif self.autofill:
            feature_array = torch.zeros((len(atom_array), n_dims)).float()
        else:
            raise ValueError(
                f"{feature_type} 1d_feature `{feature_name}` is not present in "
                "atomarray, and autofill is False"
            )

        # Each atom is exactly one token, so token-wise mean is the identity.
        data["feats"][feature_name] = feature_array

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        data.setdefault("feats", {})
        if self.association_scheme == "atom23":
            atom_array = data["atom_array"]
            atom_array.set_annotation("is_protein_token", atom_array.is_protein)
            atom_array.set_annotation("is_dna_token", atom_array.is_dna)
            atom_array.set_annotation("is_rna_token", atom_array.is_rna)

        for feature_name, n_dims in self.token_1d_features.items():
            self._add_feature(data, feature_name, n_dims, "token")
        for feature_name, n_dims in self.atom_1d_features.items():
            self._add_feature(data, feature_name, n_dims, "atom")
        return data
