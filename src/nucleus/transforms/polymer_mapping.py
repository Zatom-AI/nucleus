"""Context-aware representation of modified nucleic-acid polymers."""

from __future__ import annotations

from collections import Counter
from functools import cache

import biotite.structure as struc
import numpy as np
from atomworks.constants import STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from atomworks.enums import ChainType
from atomworks.ml.encoding_definitions import (
    UNKNOWN_DNA,
    UNKNOWN_RNA,
    AF3SequenceEncoding,
)
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import (
    get_token_count,
    get_token_starts,
    spread_token_wise,
)
from rfd3na.constants import ATOM23_ATOM_NAMES_RNA

# NAKB parent identities from Stanford3Dfolding_dataprocessing's
# ``nakb_modified_nt_20251014.json`` (2025-10-14).  This vetted RNA reference
# takes precedence over AtomWorks' general CCD parent annotation. AtomWorks'
# CCD parent is retained only when NAKB has no entry, never as a hand-authored
# fallback. The mapping remains gated on AtomWorks polymer context below, so
# free ATP/GDP/UDP compounds always remain on the ligand path.
STANFORD_NAKB_RNA_PARENT_FALLBACK = {
    "0A": "A",
    "0C": "C",
    "0G": "G",
    "0U": "U",
    "1DP": "A",
    "2AD": "A",
    "365": "A",
    "3DA": "A",
    "40A": "A",
    "40C": "C",
    "4AC": "C",
    "50L": "G",
    "50N": "C",
    "56B": "G",
    "574": "A",
    "5AA": "A",
    "5CF": "C",
    "5CG": "G",
    "5GP": "G",
    "5JO": "A",
    "5UD": "U",
    "6F7": "U",
    "6FC": "C",
    "6MD": "A",
    "6OP": "U",
    "6U0": "A",
    "7TE": "A",
    "8AA": "G",
    "8AZ": "G",
    "8GM": "G",
    "8RJ": "U",
    "8XA": "A",
    "8XC": "C",
    "8XG": "G",
    "8XU": "U",
    "9V9": "U",
    "A1A0L": "U",
    "A1B8A": "A",
    "A1ELZ": "A",
    "A1IC0": "A",
    "A1L3P": "U",
    "A1L89": "A",
    "A2P": "A",
    "A66": "A",
    "A7C": "A",
    "ADN": "A",
    "ADP": "A",
    "ADS": "A",
    "AG9": "C",
    "AGD": "G",
    "AI5": "C",
    "AMP": "A",
    "ANZ": "A",
    "APC": "A",
    "ATP": "A",
    "B4P": "A",
    "B8N": "U",
    "C4J": "U",
    "C5P": "C",
    "C66": "C",
    "CAR": "C",
    "CFV": "C",
    "CM0": "U",
    "CTP": "C",
    "CUD": "C",
    "CVC": "C",
    "DJF": "A",
    "DUR": "U",
    "F86": "A",
    "FA2": "A",
    "FMU": "U",
    "G3A": "G",
    "G5J": "G",
    "GCP": "G",
    "GMP": "G",
    "GMU": "U",
    "GMX": "G",
    "GNG": "G",
    "GP3": "G",
    "GTA": "G",
    "GTG": "G",
    "GTP": "G",
    "HGL": "G",
    "I": "G",
    "I2T": "U",
    "IKS": "C",
    "ILK": "U",
    "IMP": "G",
    "IPN": "U",
    "JMC": "C",
    "K1F": "C",
    "KGV": "A",
    "L1J": "G",
    "L2B": "U",
    "L3X": "A",
    "LCC": "C",
    "LCG": "G",
    "LKC": "C",
    "M1Y": "U",
    "M7G": "G",
    "M7M": "G",
    "MGT": "G",
    "MKX": "A",
    "MUM": "U",
    "N79": "A",
    "NF2": "U",
    "NRL": "C",
    "NSF": "U",
    "O2C": "C",
    "OBX": "C",
    "PPS": "A",
    "PQ1": "G",
    "PYY": "C",
    "Q61": "G",
    "QSK": "C",
    "RY": "C",
    "T23": "U",
    "T2T": "U",
    "TKW": "C",
    "TLB": "U",
    "TLN": "U",
    "TM2": "U",
    "U5P": "U",
    "UBB": "U",
    "UMO": "U",
    "UOA": "U",
    "UOB": "U",
    "URU": "U",
    "UTP": "U",
    "UVP": "U",
    "UY1": "U",
    "UY4": "A",
    "UZL": "C",
    "UZR": "U",
    "XAN": "G",
    "XDJ": "G",
    "XE6": "C",
    "XEC": "U",
    "XMP": "G",
    "XNY": "A",
    "Y5P": "U",
    "YA4": "A",
    "ZHP": "U",
    "ZIV": "G",
    "ZJS": "A",
}


class UnsupportedPolymerNucleotideError(ValueError):
    """Raised for a polymer nucleotide without an audited atom23 parent."""


_SEQUENCE_ENCODING = AF3SequenceEncoding()
_RNA_ATOM_BUDGET = len(ATOM23_ATOM_NAMES_RNA)

_STANFORD_RNA_PARENT_ATOM_NAMES = {
    "A": frozenset(
        [
            "P",
            "OP1",
            "OP2",
            "OP3",
            "C1'",
            "C2'",
            "C3'",
            "C4'",
            "O4'",
            "C5'",
            "O5'",
            "O2'",
            "O3'",
            "N9",
            "N1",
            "C2",
            "N3",
            "C4",
            "C5",
            "C6",
            "N7",
            "C8",
            "N6",
        ]
    ),
    "C": frozenset(
        [
            "P",
            "OP1",
            "OP2",
            "OP3",
            "C1'",
            "C2'",
            "C3'",
            "C4'",
            "O4'",
            "C5'",
            "O5'",
            "O2'",
            "O3'",
            "N1",
            "C2",
            "O2",
            "N3",
            "C4",
            "N4",
            "C5",
            "C6",
        ]
    ),
    "G": frozenset(
        [
            "P",
            "OP1",
            "OP2",
            "OP3",
            "C1'",
            "C2'",
            "C3'",
            "C4'",
            "O4'",
            "C5'",
            "O5'",
            "O2'",
            "O3'",
            "N9",
            "N1",
            "C2",
            "N2",
            "N3",
            "C4",
            "C5",
            "C6",
            "O6",
            "N7",
            "C8",
        ]
    ),
    "U": frozenset(
        [
            "P",
            "OP1",
            "OP2",
            "OP3",
            "C1'",
            "C2'",
            "C3'",
            "C4'",
            "O4'",
            "C5'",
            "O5'",
            "O2'",
            "O3'",
            "N1",
            "C2",
            "O2",
            "N3",
            "C4",
            "O4",
            "C5",
            "C6",
        ]
    ),
}


@cache
def _atomworks_rna_parent_by_ccd() -> dict[str, str]:
    """Read canonical RNA parents from AtomWorks' bundled CCD view."""
    chem_comp = struc.info.ccd.get_ccd()["chem_comp"]
    return {
        str(code): str(parent)
        for code, parent in zip(
            chem_comp["id"].as_array(),
            chem_comp["mon_nstd_parent_comp_id"].as_array(),
            strict=True,
        )
        if str(parent) in STANDARD_RNA
    }


def _rna_parent_for_ccd(ccd_code: str) -> str | None:
    """Resolve Stanford/NAKB first, then AtomWorks' curated CCD parent."""
    return STANFORD_NAKB_RNA_PARENT_FALLBACK.get(
        ccd_code, _atomworks_rna_parent_by_ccd().get(ccd_code)
    )


def _nucleotide_parent_for_ccd(ccd_code: str, na_kind: str) -> str | None:
    """Resolve a canonical parent while preserving DNA versus RNA identity."""
    parent = _rna_parent_for_ccd(ccd_code)
    if parent is None or na_kind != "DNA":
        return parent
    return {"A": "DA", "C": "DC", "G": "DG", "U": "DT"}.get(parent, parent)


def _token_is_polymer(atom_array, token_starts: np.ndarray) -> np.ndarray | None:
    if "is_polymer" not in atom_array.get_annotation_categories():
        return None
    return np.asarray(atom_array.is_polymer[token_starts], dtype=bool)


def _nucleic_acid_kind(res_names: np.ndarray) -> np.ndarray:
    """Return RNA, DNA, or empty labels from CCD chemistry."""
    af3_tokens = _SEQUENCE_ENCODING.res_name_to_af3_token(res_names)
    kind = np.full(len(res_names), "", dtype="U3")
    kind[np.isin(af3_tokens, [*STANDARD_RNA, UNKNOWN_RNA])] = "RNA"
    kind[np.isin(af3_tokens, [*STANDARD_DNA, UNKNOWN_DNA])] = "DNA"
    return kind


def _token_has_atom_name(
    atom_array, token_starts: np.ndarray, atom_name: str
) -> np.ndarray:
    """Return whether each token contains a named atom."""
    starts = np.append(token_starts, len(atom_array))
    return np.asarray(
        [
            atom_name in atom_array.atom_name[start:end]
            for start, end in zip(starts[:-1], starts[1:], strict=True)
        ],
        dtype=bool,
    )


def _polymer_nucleic_acid_kind(
    atom_array, token_starts: np.ndarray, res_names: np.ndarray
) -> np.ndarray:
    """Infer nucleotide role from CCD chemistry and guarded chain context.

    ``chain_type`` describes a whole polymer chain, not every individual
    residue.  Aminoacylated RNA and polymer-attached caps can therefore carry
    an RNA chain label despite being amino-acid or ligand tokens.  Use that
    label only to rescue an otherwise unknown residue with a nucleotide C1'
    atom, while retaining residue-level chemistry as the primary signal.
    """
    kind = _nucleic_acid_kind(res_names)
    if "chain_type" not in atom_array.get_annotation_categories():
        return kind

    chain_type = np.asarray(atom_array.chain_type[token_starts])
    af3_tokens = _SEQUENCE_ENCODING.res_name_to_af3_token(res_names)
    is_standard_amino_acid = np.isin(af3_tokens, STANDARD_AA)
    has_c1_prime = _token_has_atom_name(atom_array, token_starts, "C1'")
    unknown_na_candidate = (kind == "") & ~is_standard_amino_acid & has_c1_prime
    kind[unknown_na_candidate & (chain_type == ChainType.RNA.value)] = "RNA"
    kind[unknown_na_candidate & (chain_type == ChainType.DNA.value)] = "DNA"
    # Hybrid chains need per-residue chemistry to distinguish RNA from DNA.
    return kind


class MapPolymerNucleotideParents(Transform):
    """Map audited modified polymer RNA to canonical atom23 parent residues.

    Mapping is gated on AtomWorks' ``is_polymer`` annotation.  Consequently a
    free GDP/ATP/UDP-like component is never relabelled as a polymer nucleotide.
    Modification-only atoms outside the parent atom23 layout are removed, with
    the original CCD code and dropped-atom count retained in annotations and in
    ``data['polymer_mapping_summary']``.  Polymer nucleotides without an
    audited parent raise a named exception so fallback accounting can surface
    them instead of silently atomizing them.
    """

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        starts = get_token_starts(atom_array, add_exclusive_stop=True)
        token_starts = starts[:-1]
        is_polymer = _token_is_polymer(atom_array, token_starts)
        if is_polymer is None:
            # Inference fixtures predating AtomWorks' polymer annotation retain
            # legacy behavior; PDB training inputs always carry this field.
            return data

        res_names = np.asarray(atom_array.res_name[token_starts], dtype="U8")
        na_kind = _polymer_nucleic_acid_kind(atom_array, token_starts, res_names)
        is_noncanonical_na_polymer = (
            is_polymer
            & (na_kind != "")
            & ~np.isin(res_names, [*STANDARD_RNA, *STANDARD_DNA])
        )
        parents = np.asarray(
            [
                _nucleotide_parent_for_ccd(code, kind)
                for code, kind in zip(res_names, na_kind, strict=True)
            ]
        )
        unsupported = sorted(
            set(res_names[is_noncanonical_na_polymer & (parents == None)])  # noqa: E711
        )
        if unsupported:
            raise UnsupportedPolymerNucleotideError(
                "Unsupported polymer nucleotide CCD codes under the audited "
                f"atom23 mapping policy: {', '.join(unsupported)}"
            )

        mapped_token = is_noncanonical_na_polymer
        original_res_name = np.asarray(atom_array.res_name, dtype="U8").copy()
        mapping_status = np.full(len(atom_array), "identity", dtype="U12")
        dropped_atoms = np.zeros(len(atom_array), dtype=np.int16)
        keep = np.ones(len(atom_array), dtype=bool)
        summary = Counter()

        for token_index in np.flatnonzero(mapped_token):
            start, end = starts[token_index], starts[token_index + 1]
            parent = parents[token_index]
            assert parent is not None
            token_atom_names = np.asarray(atom_array.atom_name[start:end], dtype="U4")
            if "C1'" not in token_atom_names:
                raise UnsupportedPolymerNucleotideError(
                    f"Mapped polymer nucleotide {res_names[token_index]} lacks C1'"
                )
            atom_name_parent = {"DA": "A", "DC": "C", "DG": "G", "DT": "U"}.get(
                parent, parent
            )
            keep_token = np.isin(
                token_atom_names,
                list(_STANFORD_RNA_PARENT_ATOM_NAMES[atom_name_parent]),
            )
            retained_atom_names = token_atom_names[keep_token]
            if len(retained_atom_names) > _RNA_ATOM_BUDGET:
                raise UnsupportedPolymerNucleotideError(
                    f"Mapped polymer nucleotide {res_names[token_index]} retains "
                    f"{len(retained_atom_names)} canonical heavy atoms, exceeding "
                    f"the {_RNA_ATOM_BUDGET}-atom RNA layout"
                )
            if len(np.unique(retained_atom_names)) != len(retained_atom_names):
                raise UnsupportedPolymerNucleotideError(
                    f"Mapped polymer nucleotide {res_names[token_index]} has duplicate "
                    "canonical atom names after Stanford/NAKB filtering"
                )
            keep[start:end] = keep_token
            atom_array.res_name[start:end] = parent
            mapping_status[start:end] = "mapped"
            dropped_atoms[start:end] = int((~keep_token).sum())
            summary[f"mapped:{res_names[token_index]}->{parent}"] += 1
            if (~keep_token).any():
                summary[f"dropped_atoms:{res_names[token_index]}"] += int(
                    (~keep_token).sum()
                )

        atom_array.set_annotation("nucleus_original_res_name", original_res_name)
        atom_array.set_annotation("nucleus_mapping_status", mapping_status)
        atom_array.set_annotation("nucleus_mapping_dropped_atoms", dropped_atoms)
        if not np.all(keep):
            atom_array = atom_array[keep]
        data["atom_array"] = atom_array
        data["polymer_mapping_summary"] = dict(summary)
        return data


class ContextAwareAssignTypes(Transform):
    """Assign model types using polymer context rather than CCD name alone."""

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        token_starts = get_token_starts(atom_array)
        res_names = np.asarray(atom_array.res_name[token_starts], dtype="U8")
        token_id = np.arange(get_token_count(atom_array), dtype=np.uint32)
        atom_to_token_map = spread_token_wise(atom_array, token_id)
        is_polymer = _token_is_polymer(atom_array, token_starts)

        if is_polymer is None:
            # Preserve legacy/inference behavior when no parser context exists.
            na_kind = _polymer_nucleic_acid_kind(atom_array, token_starts, res_names)
            is_protein = _SEQUENCE_ENCODING.res_name_to_af3_token(res_names)
            is_protein = np.isin(is_protein, STANDARD_AA)
            is_rna = na_kind == "RNA"
            is_dna = na_kind == "DNA"
        else:
            na_kind = _polymer_nucleic_acid_kind(atom_array, token_starts, res_names)
            aa_tokens = _SEQUENCE_ENCODING.res_name_to_af3_token(res_names)
            is_protein = is_polymer & np.isin(aa_tokens, STANDARD_AA)
            is_rna = is_polymer & ~is_protein & (na_kind == "RNA")
            is_dna = is_polymer & ~is_protein & (na_kind == "DNA")

        is_ligand = ~(is_protein | is_rna | is_dna)
        is_residue = is_protein & np.isin(res_names, STANDARD_AA)
        atom_array.set_annotation("is_protein", is_protein[atom_to_token_map])
        atom_array.set_annotation("is_rna", is_rna[atom_to_token_map])
        atom_array.set_annotation("is_dna", is_dna[atom_to_token_map])
        atom_array.set_annotation("is_ligand", is_ligand[atom_to_token_map])
        atom_array.set_annotation("is_residue", is_residue[atom_to_token_map])
        data["atom_array"] = atom_array
        return data
