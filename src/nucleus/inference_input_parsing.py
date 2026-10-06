"""Zatom inference parsing extensions."""

import importlib
import random
import time
from typing import Dict, List, Optional, Union

import numpy as np
from atomworks import parse
from atomworks.io.parser import parse_atom_array
from atomworks.io.transforms.categories import initialize_chain_info_from_category
from atomworks.io.utils.bonds import has_ambiguous_annotation_set
from atomworks.io.utils.io_utils import get_structure, read_any
from atomworks.io.utils.non_rcsb import initialize_chain_info_from_atom_array
from atomworks.ml.transforms.base import TransformedDict
from atomworks.ml.utils.token import get_token_starts
from biotite import structure as struc
from biotite.structure import AtomArray, BondList
from foundry.common import exists
from foundry.utils.components import (
    ComponentValidationError,
    extract_pn_unit_info,
    fetch_mask_from_idx,
)

from nucleus.utils.materials import (
    annotate_material_properties,
    cartesian_to_fractional,
    copy_material_annotations,
    lattice_vectors_from_lengths_angles,
)

try:
    from rfd3.constants import (
        INFERENCE_ANNOTATIONS,
        REQUIRED_CONDITIONING_ANNOTATIONS,
        REQUIRED_INFERENCE_ANNOTATIONS,
    )
    from rfd3.inference import input_parsing as rfd3_input_parsing
    from rfd3.inference import parsing as rfd3_parsing
    from rfd3.inference.input_parsing import (
        DesignInputSpecification,
        _add_backbone_bonds_for_nonstandard_residues,
        _check_has_backbone_connections_to_nonstandard_residues,
        _restore_component_bonds,
        _sort_bonds,
        create_diffused_residues,
        create_motif_residue,
    )
    from rfd3.transforms.conditioning_base import (
        check_has_required_conditioning_annotations,
        convert_existing_annotations_to_bool,
        set_default_conditioning_annotations,
    )
    from rfd3.utils.inference import (
        STANDARD_PARSER_ARGS,
        set_common_annotations,
        set_indices,
    )
except ImportError:
    raise ImportError(
        "Could not import RFD3 modules. Please ensure that the `rc-foundry[all]` package is installed and up to date."
    )


def _is_generated_atomized_component(component: str) -> bool:
    """Return whether a contig component creates molecule/material atom tokens."""
    if not component or component == "/0" or component[-1] not in {"m", "M"}:
        return False
    span = component[:-1]
    return bool(span) and all(part.isdigit() for part in span.split("-"))


def _is_generated_component(component: str) -> bool:
    """Return whether a contig component creates new polymer or atomized tokens."""
    if not component or component == "/0":
        return False

    span = component[:-1] if component[-1] in {"P", "R", "D", "m", "M"} else component
    bounds = span.split("-")
    return len(bounds) in {1, 2} and all(bound.isdigit() for bound in bounds)


def rewrite_inputless_generated_contig(spec):
    """Route a generated-only contig through the RFD typed-length parser path.

    RFD dialect 2 treats ``contig`` as a selection from an input atom array, even
    when every component is generated. The ``length`` path has no such selection
    requirement and, after installing Nucleus contig support, preserves polymer
    types and chain breaks such as ``32R,/0,90P``.
    """
    if not isinstance(spec, dict):
        return spec
    if spec.get("input") is not None or spec.get("atom_array_input") is not None:
        return spec
    if spec.get("length") is not None:
        return spec

    contig = spec.get("contig")
    if not isinstance(contig, str):
        return spec

    components = [component.strip() for component in contig.split(",")]
    generated_components = [component for component in components if component != "/0"]
    if not generated_components or not all(
        _is_generated_component(component) for component in generated_components
    ):
        return spec

    rewritten = dict(spec)
    rewritten["length"] = rewritten.pop("contig")
    return rewritten


def _strip_generated_atomized_components_for_input_selection(value):
    """Remove generated molecule/material components from input-selection parsing."""
    if not isinstance(value, str):
        return value

    kept_components = []
    for component in value.split(","):
        component = component.strip()
        if component == "/0":
            continue
        if _is_generated_atomized_component(component):
            continue
        kept_components.append(component)

    return ",".join(kept_components)


def from_any_ignoring_molecule_components(v, atom_array):
    """Parse input selections while ignoring generated molecule/material segments."""
    stripped = _strip_generated_atomized_components_for_input_selection(v)
    if stripped == "":
        mask = np.zeros(len(atom_array), dtype=bool)
        return {}, mask, {}
    return rfd3_parsing._ORIGINAL_FROM_ANY(stripped, atom_array)


def _base36_fixed_width(value: int, width: int = 3) -> str:
    """Encode a non-negative integer as a fixed-width base36 string."""
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if value < 0:
        raise ValueError("value must be non-negative")
    chars = []
    while True:
        value, remainder = divmod(value, 36)
        chars.append(digits[remainder])
        if value == 0:
            break
    return "".join(reversed(chars)).rjust(width, "0")[-width:]


def uniquify_atom_names_for_cif(atom_array: AtomArray) -> AtomArray:
    """Make atom names unique when CIF bond annotations would otherwise be ambiguous."""
    if not has_ambiguous_annotation_set(atom_array):
        return atom_array

    atom_array = atom_array.copy()
    prefix = np.asarray(atom_array.element).astype(str)
    unique_atom_names = np.array(
        [
            f"{(element[0] if element and element[0].isalpha() else 'X')}{_base36_fixed_width(i)}"
            for i, element in enumerate(prefix)
        ],
        dtype=atom_array.atom_name.dtype,
    )
    atom_array.atom_name = unique_atom_names
    if "atom_id" in atom_array.get_annotation_categories():
        atom_array.atom_id = unique_atom_names
    return atom_array


def get_design_pattern_with_molecule_constraints(contig, length=None):
    """
    Expand an RFD3 contig string with atomized molecule/material segments.

    Examples:
        ``10m`` creates ten atomized molecule tokens.
        ``10M`` creates ten atomized material tokens.
        ``5-12m`` samples between five and twelve atomized molecule tokens.
    """
    if length == contig and any(
        part[-1] in ["P", "R", "D", "m", "M"]
        for part in contig.split(",")
        if part != "/0"
    ):
        length = None

    contig_parts = contig.split(",")
    variable_ranges = []
    fixed_parts = []
    pos_to_put_motif = []
    suffixes = []

    for part in contig_parts:
        if part == "/0":
            pos_to_put_motif.append(2)
            continue

        if part[-1] in ["P", "R", "D", "m", "M"]:
            suffixes.append(part[-1])
            part = part[:-1]
            if "-" in part:
                start, end = map(int, part.split("-"))
            else:
                start = end = int(part)
            variable_ranges.append([start, end])
            pos_to_put_motif.append(0)
        elif any(c.isalpha() for c in part):
            pn_unit_id, pn_unit_start, pn_unit_end = extract_pn_unit_info(part)
            fixed_parts.append([pn_unit_id, pn_unit_start, pn_unit_end])
            pos_to_put_motif.append(1)
        else:
            if "-" in part:
                start, end = map(int, part.split("-"))
            else:
                start = end = int(part)
            variable_ranges.append([start, end])
            pos_to_put_motif.append(0)
            suffixes.append("P")

    num_motif_residues = sum([i[2] - i[1] + 1 for i in fixed_parts])
    if length is None:
        length_min, length_max = 0, 9999
    elif "-" in length:
        length_min, length_max = map(int, length.split("-"))
    else:
        length_min = length_max = int(length)

    length_min -= num_motif_residues
    length_max -= num_motif_residues
    remaining_length_min = length_min
    remaining_length_max = length_max

    num_free_tokens = []
    for range_idx, range_limits in enumerate(variable_ranges):
        min_value, max_value = range_limits
        valid_min = max(
            min_value,
            remaining_length_min - sum(r[1] for r in variable_ranges[range_idx + 1 :]),
        )
        valid_max = min(
            max_value,
            remaining_length_max - sum(r[0] for r in variable_ranges[range_idx + 1 :]),
        )
        if valid_min > valid_max and length is not None:
            raise ComponentValidationError(
                "No valid selections possible with the given constraints."
            )
        selected_value = random.randint(valid_min, valid_max)
        num_free_tokens.append(selected_value)
        remaining_length_min -= selected_value
        remaining_length_max -= selected_value

    atoms_with_motif = []
    for idx in range(len(pos_to_put_motif)):
        if pos_to_put_motif[idx] == 1:
            pn_unit_id, pn_unit_start, pn_unit_end = fixed_parts.pop(0)
            for index in range(pn_unit_start, pn_unit_end + 1):
                atoms_with_motif.append(f"{pn_unit_id}{index}")
        elif pos_to_put_motif[idx] == 0:
            free_atom = num_free_tokens.pop(0)
            atoms_with_motif.append(str(free_atom) + suffixes.pop(0))
        elif pos_to_put_motif[idx] == 2:
            atoms_with_motif.append("/0")

    return atoms_with_motif


def create_diffused_molecule_atoms(n, additional_annotations=None):
    """Create ``n`` one-atom ligand tokens for molecule generation."""
    if n <= 0:
        raise ValueError(f"Negative/null molecule atom count ({n}) not allowed.")

    atoms = [
        struc.Atom(
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            res_name="LIG",
            res_id=1,
        )
        for _ in range(n)
    ]
    array = struc.array(atoms)
    array.set_annotation("element", np.full(n, "C", dtype="<U2"))
    array.set_annotation("atom_name", np.full(n, "C", dtype="<U4"))
    array = set_default_conditioning_annotations(
        array, motif=False, additional=additional_annotations
    )
    array = set_common_annotations(array)
    array.set_annotation("atomize", np.ones(n, dtype=bool))
    array.set_annotation("is_material", np.zeros(n, dtype=bool))
    return array


def create_diffused_material_atoms(n, additional_annotations=None):
    """Create ``n`` one-atom material tokens for crystal generation."""
    array = create_diffused_molecule_atoms(n, additional_annotations)
    array.res_name = np.full(n, "LIG", dtype=array.res_name.dtype)
    array.set_annotation("is_material", np.ones(n, dtype=bool))
    return array


def accumulate_components_with_molecules(
    components_to_accumulate: List[Union[str, int]],
    *,
    indexed_tokens: Dict[str, AtomArray],
    unindexed_tokens: Dict[str, AtomArray],
    atom_array_accum=None,
    start_chain: str = "A",
    start_resid: int = 1,
    unindexed_breaks: Optional[List[bool]] = None,
    src_atom_array: Optional[AtomArray] = None,
    strip_sidechains_by_default: bool = False,
    **kwargs,
) -> AtomArray:
    """RFD3 component accumulation with molecule/material atomized segments."""
    if not any(
        str(component).endswith(("m", "M")) for component in components_to_accumulate
    ):
        return rfd3_input_parsing._ORIGINAL_ACCUMULATE_COMPONENTS(
            components_to_accumulate,
            indexed_tokens=indexed_tokens,
            unindexed_tokens=unindexed_tokens,
            atom_array_accum=atom_array_accum,
            start_chain=start_chain,
            start_resid=start_resid,
            unindexed_breaks=unindexed_breaks,
            src_atom_array=src_atom_array,
            strip_sidechains_by_default=strip_sidechains_by_default,
            **kwargs,
        )

    assert (
        x := (set(list(indexed_tokens.keys()) + list(unindexed_tokens.keys())))
    ).issubset(
        (y := set(components_to_accumulate))
    ), "Unindexed and indexed set {} is not subset of components to accumulate {}".format(
        x, y
    )
    all_tokens = indexed_tokens | unindexed_tokens
    all_annots = []
    [
        all_annots.extend(list(tok.get_annotation_categories()))
        for tok in all_tokens.values()
    ]
    all_annots = set(all_annots)
    atom_array_accum = [] if atom_array_accum is None else atom_array_accum
    unindexed_breaks = (
        [None] * len(components_to_accumulate)
        if unindexed_breaks is None
        else unindexed_breaks
    )

    unindexed_components_started = False
    chain = start_chain
    res_id = start_resid
    molecule_id = 0
    source_to_accum_idx: Dict[int, int] = {}
    source_idx_to_component: Dict[int, str] = {}
    unindexed_component_names = set(unindexed_tokens.keys())
    current_accum_idx = sum(len(arr) for arr in atom_array_accum)

    assert len(components_to_accumulate) == len(
        unindexed_breaks
    ), "Mismatch in number of components to accumulate and breaks"
    for component, is_break in zip(components_to_accumulate, unindexed_breaks):
        component = str(component)
        src_indices = None
        if exists(is_break) and is_break and not unindexed_components_started:
            chain = start_chain
            res_id = start_resid
            unindexed_components_started = True

        if component == "/0":
            chain = chr(ord(chain) + 1)
            molecule_id += 1
            res_id = 1
            continue

        if component[0].isalpha():
            n = 1
            token = all_tokens[component]
            if src_atom_array is not None:
                src_mask = fetch_mask_from_idx(component, atom_array=src_atom_array)
                src_indices = np.where(src_mask)[0]

            token = create_motif_residue(
                token, strip_sidechains_by_default=strip_sidechains_by_default
            )
            token.set_annotation(
                "is_motif_atom_unindexed_motif_breakpoint",
                np.full(token.shape[0], bool(is_break), dtype=int),
            )
        else:
            suffix = component[-1]
            n = int(component[:-1])
            if n == 0 or unindexed_components_started:
                res_id += n
                continue
            if suffix == "m":
                token = create_diffused_molecule_atoms(n, all_annots)
            elif suffix == "M":
                token = create_diffused_material_atoms(n, all_annots)
            else:
                token = create_diffused_residues(n, all_annots)

        if "is_material" not in token.get_annotation_categories():
            token.set_annotation(
                "is_material",
                np.zeros(token.array_length(), dtype=bool),
            )

        token = set_indices(
            array=token,
            chain=chain,
            res_id_start=res_id,
            molecule_id=molecule_id,
            component=component,
        )

        assert len(get_token_starts(token)) == n, (
            f"Mismatch in number of residues: expected {n}, "
            f"got {len(get_token_starts(token))} in \n{token}"
        )

        if (
            src_atom_array is not None
            and component[0].isalpha()
            and src_indices is not None
            and len(src_indices) == len(token)
        ):
            for i, src_idx in enumerate(src_indices):
                source_to_accum_idx[int(src_idx)] = current_accum_idx + i
                source_idx_to_component[int(src_idx)] = component

        atom_array_accum.append(token)

        # Handle molecules, materials, and polymers differently for residue ID increment
        res_id += (
            1 if not component[0].isalpha() and component[-1].lower() == "m" else n
        )
        current_accum_idx += len(token)

    atom_array_accum = struc.concatenate(atom_array_accum)
    atom_array_accum.set_annotation("pn_unit_iid", atom_array_accum.chain_id)
    should_restore_bonds = (
        src_atom_array is not None
        and bool(source_to_accum_idx)
        and _check_has_backbone_connections_to_nonstandard_residues(
            atom_array_accum, src_atom_array
        )
    )
    if should_restore_bonds:
        atom_array_accum = _restore_component_bonds(
            atom_array_accum=atom_array_accum,
            src_atom_array=src_atom_array,
            source_to_accum_idx=source_to_accum_idx,
            source_idx_to_component=source_idx_to_component,
            unindexed_components=unindexed_component_names,
        )
        atom_array_accum = _add_backbone_bonds_for_nonstandard_residues(
            atom_array_accum=atom_array_accum
        )
        atom_array_accum = _sort_bonds(atom_array_accum)

    if np.any(atom_array_accum.is_motif_atom_unindexed.astype(bool)) and not np.all(
        atom_array_accum.is_motif_atom_unindexed.astype(bool)
    ):
        max_id = np.max(
            atom_array_accum[
                ~atom_array_accum.is_motif_atom_unindexed.astype(bool)
            ].res_id
        )
        min_id_udx = np.min(
            atom_array_accum[
                atom_array_accum.is_motif_atom_unindexed.astype(bool)
            ].res_id
        )
        atom_array_accum.res_id[
            atom_array_accum.is_motif_atom_unindexed.astype(bool)
        ] += max_id - min_id_udx + 1

    if atom_array_accum.bonds is None:
        atom_array_accum.bonds = BondList(atom_array_accum.array_length())
    return atom_array_accum


def install_molecule_contig_support_for(package_name: str):
    """Install nucleus molecule/material contig support into an RFD parser backend."""
    package_name = getattr(package_name, "__name__", package_name)
    if package_name not in {"rfd3", "rfd3na"}:
        raise ValueError(f"Unsupported molecule contig backend: {package_name}")

    parsing_module = importlib.import_module(f"{package_name}.inference.parsing")
    input_parsing_module = importlib.import_module(
        f"{package_name}.inference.input_parsing"
    )
    conditioning_base_module = importlib.import_module(
        f"{package_name}.transforms.conditioning_base"
    )
    inference_utils_module = importlib.import_module(f"{package_name}.utils.inference")

    original_from_any_attr = "_NUCLEUS_ORIGINAL_FROM_ANY"
    if not hasattr(parsing_module, original_from_any_attr):
        setattr(parsing_module, original_from_any_attr, parsing_module.from_any_)

    def from_any_ignoring_molecule_components_for(v, atom_array):
        stripped = _strip_generated_atomized_components_for_input_selection(v)
        if stripped == "":
            mask = np.zeros(len(atom_array), dtype=bool)
            return {}, mask, {}
        return getattr(parsing_module, original_from_any_attr)(stripped, atom_array)

    def create_diffused_atomized_atoms(
        n, additional_annotations=None, *, is_material=False
    ):
        if n <= 0:
            raise ValueError(f"Negative/null molecule atom count ({n}) not allowed.")

        atoms = [
            struc.Atom(
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                res_name="LIG",
                res_id=1,
            )
            for _ in range(n)
        ]
        array = struc.array(atoms)
        array.set_annotation("element", np.full(n, "C", dtype="<U2"))
        array.set_annotation("atom_name", np.full(n, "C", dtype="<U4"))
        array = conditioning_base_module.set_default_conditioning_annotations(
            array, motif=False, additional=additional_annotations
        )
        array = inference_utils_module.set_common_annotations(array)
        array.set_annotation("atomize", np.ones(n, dtype=bool))
        array.set_annotation("is_material", np.full(n, is_material, dtype=bool))
        return array

    def create_diffused_polymer_residues(n, additional_annotations, suffix):
        if package_name == "rfd3na":
            return input_parsing_module.create_diffused_residues(
                n, additional_annotations, suffix
            )
        return input_parsing_module.create_diffused_residues(n, additional_annotations)

    original_accumulate_attr = "_NUCLEUS_ORIGINAL_ACCUMULATE_COMPONENTS"
    if not hasattr(input_parsing_module, original_accumulate_attr):
        setattr(
            input_parsing_module,
            original_accumulate_attr,
            input_parsing_module.accumulate_components,
        )

    def accumulate_components_with_molecules_for(
        components_to_accumulate,
        *,
        indexed_tokens,
        unindexed_tokens,
        atom_array_accum=None,
        start_chain="A",
        start_resid=1,
        unindexed_breaks=None,
        src_atom_array=None,
        strip_sidechains_by_default=False,
        **kwargs,
    ):
        if not any(
            str(component).endswith(("m", "M"))
            for component in components_to_accumulate
        ):
            return getattr(input_parsing_module, original_accumulate_attr)(
                components_to_accumulate,
                indexed_tokens=indexed_tokens,
                unindexed_tokens=unindexed_tokens,
                atom_array_accum=atom_array_accum,
                start_chain=start_chain,
                start_resid=start_resid,
                unindexed_breaks=unindexed_breaks,
                src_atom_array=src_atom_array,
                strip_sidechains_by_default=strip_sidechains_by_default,
                **kwargs,
            )

        assert (
            x := (set(list(indexed_tokens.keys()) + list(unindexed_tokens.keys())))
        ).issubset(
            (y := set(components_to_accumulate))
        ), "Unindexed and indexed set {} is not subset of components to accumulate {}".format(
            x, y
        )
        all_tokens = indexed_tokens | unindexed_tokens
        all_annots = []
        [
            all_annots.extend(list(tok.get_annotation_categories()))
            for tok in all_tokens.values()
        ]
        all_annots = set(all_annots)
        atom_array_accum = [] if atom_array_accum is None else atom_array_accum
        unindexed_breaks = (
            [None] * len(components_to_accumulate)
            if unindexed_breaks is None
            else unindexed_breaks
        )

        unindexed_components_started = False
        chain = start_chain
        res_id = start_resid
        molecule_id = 0
        source_to_accum_idx = {}
        source_idx_to_component = {}
        unindexed_component_names = set(unindexed_tokens.keys())
        current_accum_idx = sum(len(arr) for arr in atom_array_accum)

        assert len(components_to_accumulate) == len(
            unindexed_breaks
        ), "Mismatch in number of components to accumulate and breaks"
        for component, is_break in zip(components_to_accumulate, unindexed_breaks):
            component = str(component)
            src_indices = None
            if exists(is_break) and is_break and not unindexed_components_started:
                chain = start_chain
                res_id = start_resid
                unindexed_components_started = True

            if component == "/0":
                chain = chr(ord(chain) + 1)
                molecule_id += 1
                res_id = 1
                continue

            if component[0].isalpha():
                n = 1
                token = all_tokens[component]
                if src_atom_array is not None:
                    src_mask = fetch_mask_from_idx(component, atom_array=src_atom_array)
                    src_indices = np.where(src_mask)[0]

                token = input_parsing_module.create_motif_residue(
                    token, strip_sidechains_by_default=strip_sidechains_by_default
                )
                token.set_annotation(
                    "is_motif_atom_unindexed_motif_breakpoint",
                    np.full(token.shape[0], bool(is_break), dtype=int),
                )
            else:
                suffix = component[-1]
                if suffix in {"P", "R", "D", "m", "M"}:
                    n = int(component[:-1])
                else:
                    suffix = "P"
                    n = int(component)

                if n == 0 or unindexed_components_started:
                    res_id += n
                    continue

                if suffix == "m":
                    token = create_diffused_atomized_atoms(n, all_annots)
                elif suffix == "M":
                    token = create_diffused_atomized_atoms(
                        n, all_annots, is_material=True
                    )
                else:
                    token = create_diffused_polymer_residues(n, all_annots, suffix)

            if package_name == "rfd3" and (
                "is_material" not in token.get_annotation_categories()
            ):
                token.set_annotation(
                    "is_material",
                    np.zeros(token.array_length(), dtype=bool),
                )

            token = inference_utils_module.set_indices(
                array=token,
                chain=chain,
                res_id_start=res_id,
                molecule_id=molecule_id,
                component=component,
            )

            if package_name == "rfd3":
                assert len(get_token_starts(token)) == n, (
                    f"Mismatch in number of residues: expected {n}, "
                    f"got {len(get_token_starts(token))} in \n{token}"
                )

            if (
                src_atom_array is not None
                and component[0].isalpha()
                and src_indices is not None
                and len(src_indices) == len(token)
            ):
                for i, src_idx in enumerate(src_indices):
                    source_to_accum_idx[int(src_idx)] = current_accum_idx + i
                    source_idx_to_component[int(src_idx)] = component

            atom_array_accum.append(token)
            res_id += (
                1 if not component[0].isalpha() and component[-1].lower() == "m" else n
            )
            current_accum_idx += len(token)

        atom_array_accum = struc.concatenate(atom_array_accum)
        atom_array_accum.set_annotation("pn_unit_iid", atom_array_accum.chain_id)

        should_restore_bonds = (
            src_atom_array is not None
            and bool(source_to_accum_idx)
            and input_parsing_module._check_has_backbone_connections_to_nonstandard_residues(
                atom_array_accum, src_atom_array
            )
        )
        if should_restore_bonds:
            if package_name == "rfd3na":
                atom_array_accum = (
                    input_parsing_module._restore_bonds_for_nonstandard_residues(
                        atom_array_accum, src_atom_array, source_to_accum_idx
                    )
                )
            else:
                atom_array_accum = input_parsing_module._restore_component_bonds(
                    atom_array_accum=atom_array_accum,
                    src_atom_array=src_atom_array,
                    source_to_accum_idx=source_to_accum_idx,
                    source_idx_to_component=source_idx_to_component,
                    unindexed_components=unindexed_component_names,
                )
                atom_array_accum = (
                    input_parsing_module._add_backbone_bonds_for_nonstandard_residues(
                        atom_array_accum=atom_array_accum
                    )
                )
                atom_array_accum = input_parsing_module._sort_bonds(atom_array_accum)

        if np.any(atom_array_accum.is_motif_atom_unindexed.astype(bool)) and not np.all(
            atom_array_accum.is_motif_atom_unindexed.astype(bool)
        ):
            max_id = np.max(
                atom_array_accum[
                    ~atom_array_accum.is_motif_atom_unindexed.astype(bool)
                ].res_id
            )
            min_id_udx = np.min(
                atom_array_accum[
                    atom_array_accum.is_motif_atom_unindexed.astype(bool)
                ].res_id
            )
            atom_array_accum.res_id[
                atom_array_accum.is_motif_atom_unindexed.astype(bool)
            ] += max_id - min_id_udx + 1

        if atom_array_accum.bonds is None:
            atom_array_accum.bonds = BondList(atom_array_accum.array_length())
        return atom_array_accum

    parsing_module.from_any_ = from_any_ignoring_molecule_components_for
    input_parsing_module.get_design_pattern_with_constraints = (
        get_design_pattern_with_molecule_constraints
    )
    input_parsing_module.accumulate_components = (
        accumulate_components_with_molecules_for
    )


def _cif_scalar(category, key):
    """Read a scalar CIF category value as float."""
    if key not in category:
        return None
    values = category[key].as_array(float)
    if len(values) == 0 or not np.isfinite(values[0]):
        return None
    return float(values[0])


def _atom_array_is_nonpolymer_only(atom_array: AtomArray) -> bool:
    """Return whether a parsed input contains no polymer chains."""
    if "is_polymer" in atom_array.get_annotation_categories():
        return not bool(np.any(atom_array.is_polymer.astype(bool)))
    return not (
        np.any(getattr(atom_array, "is_protein", np.zeros(len(atom_array), dtype=bool)))
        or np.any(getattr(atom_array, "is_dna", np.zeros(len(atom_array), dtype=bool)))
        or np.any(getattr(atom_array, "is_rna", np.zeros(len(atom_array), dtype=bool)))
    )


def _add_material_annotations_from_cif(file, atom_array: AtomArray) -> AtomArray:
    """Attach material lattice/fractional-coordinate annotations from a CIF file."""
    try:
        cif_file = read_any(file)
    except Exception:
        return atom_array
    block = getattr(cif_file, "block", None)
    if block is None or "cell" not in block:
        return atom_array

    cell = block["cell"]
    lengths = np.asarray(
        [
            _cif_scalar(cell, "length_a"),
            _cif_scalar(cell, "length_b"),
            _cif_scalar(cell, "length_c"),
        ],
        dtype=float,
    )
    angles = np.asarray(
        [
            _cif_scalar(cell, "angle_alpha"),
            _cif_scalar(cell, "angle_beta"),
            _cif_scalar(cell, "angle_gamma"),
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(lengths)) or not np.all(np.isfinite(angles)):
        return atom_array

    atom_site = block["atom_site"] if "atom_site" in block else {}
    has_fractional = all(key in atom_site for key in ("fract_x", "fract_y", "fract_z"))
    if not has_fractional and not _atom_array_is_nonpolymer_only(atom_array):
        return atom_array

    lattice_vectors = lattice_vectors_from_lengths_angles(lengths, angles)
    if has_fractional:
        fractional_coordinates = np.stack(
            [
                atom_site["fract_x"].as_array(float),
                atom_site["fract_y"].as_array(float),
                atom_site["fract_z"].as_array(float),
            ],
            axis=-1,
        )
        if fractional_coordinates.shape[0] != atom_array.array_length():
            fractional_coordinates = cartesian_to_fractional(
                atom_array.coord,
                lattice_vectors,
            )
    else:
        fractional_coordinates = cartesian_to_fractional(
            atom_array.coord,
            lattice_vectors,
        )

    return annotate_material_properties(
        atom_array,
        fractional_coordinates=fractional_coordinates,
        lattice_lengths=lengths,
        lattice_angles=angles,
        lattice_vectors=lattice_vectors,
    )


def _load_material_cif_without_bonds(file, *, cif_parser_args):
    """Bypass molecular bond construction and file-parser caches for materials."""
    try:
        cif_file = read_any(file, file_type=cif_parser_args.get("file_type"))
    except Exception:
        return None
    block = getattr(cif_file, "block", None)
    if block is None or "cell" not in block:
        return None

    atom_array = get_structure(
        cif_file,
        include_bonds=False,
        model=cif_parser_args.get("model") or 1,
        extra_fields=[
            "label_entity_id",
            "auth_seq_id",
            "atom_id",
            "b_factor",
            "occupancy",
            "charge",
            *cif_parser_args["extra_fields"],
        ],
    )
    has_fractional = all(
        key in block["atom_site"] for key in ("fract_x", "fract_y", "fract_z")
    )
    if not has_fractional:
        # Match AtomWorks' polymer detection: a biological cell is not a material.
        chain_info = (
            initialize_chain_info_from_category(block, atom_array)
            if "entity" in block and "entity_poly" in block
            else initialize_chain_info_from_atom_array(atom_array)
        )
        if any(chain["is_polymer"] for chain in chain_info.values()):
            return None

    if cif_parser_args.get("hydrogen_policy") == "infer":
        raise ValueError("Hydrogen inference is not supported for periodic materials.")
    parser_args = dict(cif_parser_args)
    for key in (
        "file_type",
        "model",
        "cache_dir",
        "load_from_cache",
        "save_to_cache",
        "keep_cif_block",
        "extra_fields",
    ):
        parser_args.pop(key, None)
    parser_args["add_missing_atoms"] = False
    return parse_atom_array(atom_array, build_assembly=None, **parser_args)


def inference_load_preserving_hydrogens(
    file,
    *,
    assembly_id: str = "1",
    cif_parser_args: dict | None = None,
):
    """Load inference inputs without forcing hydrogen removal before transforms."""
    # Default cif_parser_args to an empty dictionary if not provided
    cif_parser_args = {} if cif_parser_args is None else dict(cif_parser_args)

    # Convenience utilities to default to loading from and saving to cache if a cache_dir is provided, unless explicitly overridden
    if cif_parser_args.get("cache_dir"):
        cif_parser_args.setdefault("load_from_cache", True)
        cif_parser_args.setdefault("save_to_cache", True)

    merged_cif_parser_args = {
        **STANDARD_PARSER_ARGS,
        "fix_arginines": False,
        "add_missing_atoms": False,
        "remove_ccds": [],
        **cif_parser_args,
    }
    merged_cif_parser_args.setdefault("hydrogen_policy", "keep")

    # Ensure the required annotations can be loaded
    merged_cif_parser_args["extra_fields"] = list(
        set(
            merged_cif_parser_args.get("extra_fields", [])
            + REQUIRED_CONDITIONING_ANNOTATIONS
        )
    )

    result_dict = _load_material_cif_without_bonds(
        file, cif_parser_args=merged_cif_parser_args
    )
    if result_dict is None:
        result_dict = parse(
            filename=file,
            build_assembly=(assembly_id,),
            **merged_cif_parser_args,
        )

    atom_array = result_dict["assemblies"][assembly_id][0]
    atom_array = convert_existing_annotations_to_bool(atom_array)
    atom_array = _add_material_annotations_from_cif(file, atom_array)

    return {
        "atom_array": atom_array,  # First model
        "chain_info": result_dict["chain_info"],
        "ligand_info": result_dict["ligand_info"],
        "metadata": result_dict["metadata"],
    }


def prepare_pipeline_input_preserving_hydrogens(atom_array_orig) -> dict:
    """Prepare pipeline input while optionally preserving hydrogens for pipeline transforms."""
    _start_parse_time = time.time()
    # HACK: Set empty bond graph:
    if atom_array_orig.bonds is None:
        atom_array_orig.bonds = BondList(atom_array_orig.array_length())

    # Temporary spoof of chain IDs to ensure duplicates aren't dropped:
    result_dict = parse_atom_array(
        atom_array_orig,
        remove_ccds=[],
        fix_arginines=False,
        add_missing_atoms=False,
        extra_fields=INFERENCE_ANNOTATIONS,
        build_assembly=None,
        hydrogen_policy="keep",
    )
    atom_array = result_dict["asym_unit"][0]
    atom_array = copy_material_annotations(atom_array_orig, atom_array)

    # HACK: Set iid information manually
    # We currently do not preserve this information from the input,
    # if you want these we'd need to remove the spoofing here
    check_has_required_conditioning_annotations(
        atom_array, required=REQUIRED_INFERENCE_ANNOTATIONS
    )
    atom_array = convert_existing_annotations_to_bool(atom_array)
    atom_array.set_annotation("chain_iid", [f"{c}_1" for c in atom_array.chain_id])
    atom_array.set_annotation("pn_unit_iid", [f"{c}_1" for c in atom_array.pn_unit_id])

    # Ensure motif annotations are removed
    if "is_motif_token" in atom_array.get_annotation_categories():
        atom_array.del_annotation("is_motif_token")
    if "is_motif_atom" in atom_array.get_annotation_categories():
        atom_array.del_annotation("is_motif_atom")

    data = {
        "atom_array": atom_array,  # First model
        "chain_info": result_dict["chain_info"],
        "ligand_info": result_dict["ligand_info"],
        "metadata": result_dict["metadata"],
    }
    data["metadata"]["parse_time"] = time.time() - _start_parse_time
    return TransformedDict(data)


def design_input_to_pipeline_input_preserving_hydrogens(self, example_id):
    """DesignInputSpecification.to_pipeline_input replacement with optional hydrogen support."""
    atom_array, spec_dict = self.build(return_metadata=True)

    # ... Forward into
    data = prepare_pipeline_input_preserving_hydrogens(atom_array)
    data["example_id"] = example_id

    # ... Wrap up with additional features
    if "extra" not in spec_dict:
        spec_dict["extra"] = {}
    spec_dict["extra"]["example_id"] = example_id
    data["specification"] = spec_dict
    return data


def install_molecule_contig_support():
    """Patch RFD3 inference parsing globals used by DesignInputSpecification."""
    if not hasattr(rfd3_parsing, "_ORIGINAL_FROM_ANY"):
        rfd3_parsing._ORIGINAL_FROM_ANY = rfd3_parsing.from_any_
    rfd3_parsing.from_any_ = from_any_ignoring_molecule_components

    if not hasattr(rfd3_input_parsing, "_ORIGINAL_INFERENCE_LOAD"):
        rfd3_input_parsing._ORIGINAL_INFERENCE_LOAD = rfd3_input_parsing.inference_load_
    rfd3_input_parsing.inference_load_ = inference_load_preserving_hydrogens

    if not hasattr(rfd3_input_parsing, "_ORIGINAL_PREPARE_PIPELINE_INPUT"):
        rfd3_input_parsing._ORIGINAL_PREPARE_PIPELINE_INPUT = (
            rfd3_input_parsing.prepare_pipeline_input_from_atom_array
        )
    rfd3_input_parsing.prepare_pipeline_input_from_atom_array = (
        prepare_pipeline_input_preserving_hydrogens
    )

    if not hasattr(
        rfd3_input_parsing.DesignInputSpecification, "_ORIGINAL_TO_PIPELINE_INPUT"
    ):
        rfd3_input_parsing.DesignInputSpecification._ORIGINAL_TO_PIPELINE_INPUT = (
            rfd3_input_parsing.DesignInputSpecification.to_pipeline_input
        )
    rfd3_input_parsing.DesignInputSpecification.to_pipeline_input = (
        design_input_to_pipeline_input_preserving_hydrogens
    )

    if not hasattr(rfd3_input_parsing, "_ORIGINAL_ACCUMULATE_COMPONENTS"):
        rfd3_input_parsing._ORIGINAL_ACCUMULATE_COMPONENTS = (
            rfd3_input_parsing.accumulate_components
        )
    rfd3_input_parsing.get_design_pattern_with_constraints = (
        get_design_pattern_with_molecule_constraints
    )
    rfd3_input_parsing.accumulate_components = accumulate_components_with_molecules


_ZATOM_DESIGN_SPECIFICATION_CLASSES = {}


class _PreserveNativeCoordinatesMixin:
    """Capture target coordinates before regular inference zeroes generated atoms."""

    def _set_origin(self, atom_array):
        """Save native targets, then delegate coordinate initialization to RFD3."""
        native_coordinates = np.asarray(atom_array.coord, dtype=np.float32).copy()

        # Generated components contain placeholder coordinates, not targets.
        categories = atom_array.get_annotation_categories()
        if "src_component" in categories:
            generated = np.asarray(
                [
                    not component or _is_generated_component(str(component))
                    for component in atom_array.src_component
                ],
                dtype=bool,
            )
            native_coordinates[generated] = np.nan
        elif getattr(self, "atom_array_input", None) is None:
            native_coordinates[:] = np.nan

        object.__setattr__(self, "_nucleus_native_coordinates", native_coordinates)
        return super()._set_origin(atom_array)


def _zatom_design_specification_class_for(design_specification_cls):
    if design_specification_cls is DesignInputSpecification:
        return ZatomDesignInputSpecification
    if design_specification_cls not in _ZATOM_DESIGN_SPECIFICATION_CLASSES:

        class BackendZatomDesignInputSpecification(
            _PreserveNativeCoordinatesMixin,
            design_specification_cls,
        ):
            """Zatom inference specification with source/property metadata."""

            example_id: str | None = None
            total_charge: int | None = None
            spin_multiplicity: int | None = None
            mean_force_norm: float | None = None
            has_mean_force_norm: bool = False
            force_rms: float | None = None
            force_p95: float | None = None
            has_force_rms_p95: bool = False

        _ZATOM_DESIGN_SPECIFICATION_CLASSES[design_specification_cls] = (
            BackendZatomDesignInputSpecification
        )
    return _ZATOM_DESIGN_SPECIFICATION_CLASSES[design_specification_cls]


class ZatomDesignInputSpecification(
    _PreserveNativeCoordinatesMixin,
    DesignInputSpecification,
):
    """Zatom inference specification with source/property metadata."""

    example_id: str | None = None
    total_charge: int | None = None
    spin_multiplicity: int | None = None
    mean_force_norm: float | None = None
    has_mean_force_norm: bool = False
    force_rms: float | None = None
    force_p95: float | None = None
    has_force_rms_p95: bool = False


def _coerce_zatom_design_specification_for(spec, design_specification_cls):
    """Validate dict specs with Zatom-specific fields against a parser backend."""
    if isinstance(spec, design_specification_cls):
        return spec
    if not isinstance(spec, dict):
        return spec
    return _zatom_design_specification_class_for(design_specification_cls).safe_init(
        **spec
    )


def _coerce_zatom_design_specification(spec):
    """Validate dict specs with Zatom-specific fields."""
    return _coerce_zatom_design_specification_for(spec, DesignInputSpecification)
