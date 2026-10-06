"""Compare two CIF files with periodic, translation, and rotation awareness.

The comparison is best-effort and self-contained:

- CIFs are parsed with Biotite, which handles the richer CIF flavor used in
  this repository (compared to that preferred by ASE).
- Atomic ordering is matched by chemical symbol using the Hungarian algorithm.
- Periodic wrap-around is handled by unwrapping each structure around a stable
  anchor atom before canonicalization.
- Translation and rotation are removed with a Kabsch alignment on the matched
  atom sets.

This does not enumerate crystallographic space-group symmetries. If you need a
full symmetry-equivalent search, use a crystallography library such as pymatgen
or spglib. For same-structure comparisons and debugging model outputs, this
utility is likely enough.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import typer
from biotite.structure.io.pdbx import CIFFile, get_structure
from scipy.optimize import linear_sum_assignment

from nucleus.utils.materials import cartesian_to_fractional, fractional_to_cartesian


@dataclass(frozen=True)
class CIFComparisonReport:
    """Structured comparison output."""

    reference_cif: str
    candidate_cif: str
    n_atoms_reference: int
    n_atoms_candidate: int
    formula_reference: str
    formula_candidate: str
    cell_lengths_reference: list[float]
    cell_lengths_candidate: list[float]
    cell_angles_reference: list[float]
    cell_angles_candidate: list[float]
    matched_rmsd_pre_align: float
    matched_rmsd_post_align: float
    max_atom_displacement_post_align: float
    mean_atom_displacement_post_align: float


def _cell_lengths_angles(atoms) -> tuple[np.ndarray, np.ndarray]:
    """Extract cell lengths and angles from the box matrix."""
    box = _box_matrix(atoms)
    lengths = np.linalg.norm(box, axis=1)
    if np.any(lengths == 0):
        raise ValueError("Encountered a zero-length lattice vector.")

    cos_alpha = np.dot(box[1], box[2]) / (lengths[1] * lengths[2])
    cos_beta = np.dot(box[0], box[2]) / (lengths[0] * lengths[2])
    cos_gamma = np.dot(box[0], box[1]) / (lengths[0] * lengths[1])
    angles = np.degrees(np.arccos(np.clip([cos_alpha, cos_beta, cos_gamma], -1.0, 1.0)))
    return lengths, angles


def _formula(atoms) -> str:
    """Compute the chemical formula from the atom symbols."""
    counts: dict[str, int] = {}
    for symbol in _normalized_symbols(atoms.element).tolist():
        counts[symbol] = counts.get(symbol, 0) + 1
    parts = []
    for symbol in sorted(counts):
        count = counts[symbol]
        parts.append(symbol if count == 1 else f"{symbol}{count}")
    return "".join(parts)


def _normalize_symbol(symbol: str) -> str:
    """Canonicalize element labels like `AG` or `ag` to `Ag`."""
    symbol = str(symbol).strip()
    if not symbol:
        return symbol
    return symbol[:1].upper() + symbol[1:].lower()


def _normalized_symbols(symbols: Iterable[str]) -> np.ndarray:
    return np.asarray([_normalize_symbol(symbol) for symbol in symbols], dtype=object)


def _choose_anchor_index(
    frac_coords: np.ndarray,
    symbols: Iterable[str],
    box: np.ndarray,
) -> int:
    """Pick a stable anchor atom independent of input ordering and translation."""
    wrapped = np.asarray(frac_coords, dtype=float) % 1.0
    deltas = wrapped[:, None, :] - wrapped[None, :, :]
    deltas -= np.round(deltas)
    cart_deltas = fractional_to_cartesian(deltas, box)
    scores = np.linalg.norm(cart_deltas, axis=-1).sum(axis=1)
    keys = [(scores[i], str(symbols[i]), i) for i in range(len(wrapped))]
    return min(keys)[-1]


def _box_matrix(atoms) -> np.ndarray:
    """Extract the 3x3 box matrix from the atoms object."""
    box = np.asarray(atoms.box, dtype=float)
    if box.ndim == 3:
        return box[0]
    return box


def _unwrap_around_anchor(frac_coords: np.ndarray, anchor_index: int) -> np.ndarray:
    """Unwrap fractional coordinates around the anchor atom."""
    anchor = np.asarray(frac_coords[anchor_index], dtype=float)
    delta = np.asarray(frac_coords, dtype=float) - anchor
    delta -= np.round(delta)
    return anchor + delta


def _canonical_pca_frame(coords: np.ndarray) -> np.ndarray:
    """Rotate coordinates into a deterministic PCA frame."""
    centered = np.asarray(coords, dtype=float) - np.asarray(coords, dtype=float).mean(
        axis=0
    )
    covariance = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]

    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0

    # Fix the sign of each axis in a deterministic way. This is not perfect for
    # degenerate point clouds, but it is stable for the material examples used in
    # this repository.
    for axis in range(3):
        pivot = int(np.argmax(np.abs(axes[:, axis])))
        if axes[pivot, axis] < 0:
            axes[:, axis] *= -1.0

    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0

    return centered @ axes


def _canonicalize_structure(atoms) -> tuple[np.ndarray, np.ndarray]:
    """Return canonicalized Cartesian coordinates and symbols."""
    symbols = _normalized_symbols(atoms.element)
    frac = np.asarray(
        cartesian_to_fractional(atoms.coord, _box_matrix(atoms)),
        dtype=float,
    )
    anchor_index = _choose_anchor_index(frac, symbols, _box_matrix(atoms))
    unwrapped_frac = _unwrap_around_anchor(frac, anchor_index)
    coords = fractional_to_cartesian(unwrapped_frac, _box_matrix(atoms))
    coords = _canonical_pca_frame(coords)
    return coords, symbols


def _match_by_symbol(
    reference_coords: np.ndarray,
    reference_symbols: np.ndarray,
    candidate_coords: np.ndarray,
    candidate_symbols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match atoms one-to-one within each element class."""
    if len(reference_coords) != len(candidate_coords):
        raise ValueError(
            "The CIF files have different atom counts: "
            f"{len(reference_coords)} vs {len(candidate_coords)}."
        )

    reference_symbols = _normalized_symbols(reference_symbols)
    candidate_symbols = _normalized_symbols(candidate_symbols)

    matched_reference = []
    matched_candidate = []
    for symbol in sorted(
        set(reference_symbols.tolist()) | set(candidate_symbols.tolist())
    ):
        ref_indices = np.flatnonzero(reference_symbols == symbol)
        cand_indices = np.flatnonzero(candidate_symbols == symbol)
        if len(ref_indices) != len(cand_indices):
            raise ValueError(
                f"Atom counts for element {symbol!r} differ: "
                f"{len(ref_indices)} vs {len(cand_indices)}."
            )

        cost = np.linalg.norm(
            reference_coords[ref_indices][:, None, :]
            - candidate_coords[cand_indices][None, :, :],
            axis=-1,
        )
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_reference.append(reference_coords[ref_indices][row_ind])
        matched_candidate.append(candidate_coords[cand_indices][col_ind])

    return np.concatenate(matched_reference, axis=0), np.concatenate(
        matched_candidate, axis=0
    )


def _kabsch(reference_coords: np.ndarray, candidate_coords: np.ndarray) -> np.ndarray:
    """Align candidate coordinates to reference coordinates with Kabsch."""
    reference_centered = reference_coords - reference_coords.mean(axis=0)
    candidate_centered = candidate_coords - candidate_coords.mean(axis=0)
    covariance = candidate_centered.T @ reference_centered
    v, _, wt = np.linalg.svd(covariance)
    d = np.sign(np.linalg.det(v @ wt))
    rotation = v @ np.diag([1.0, 1.0, d]) @ wt
    return candidate_centered @ rotation


def _rmsd(reference_coords: np.ndarray, candidate_coords: np.ndarray) -> float:
    """Compute RMSD between two sets of coordinates."""
    delta = np.asarray(reference_coords, dtype=float) - np.asarray(
        candidate_coords, dtype=float
    )
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=-1))))


def compare_cifs(
    reference_cif: Path | str, candidate_cif: Path | str
) -> CIFComparisonReport:
    """Compare two CIF files and return a structured report."""
    reference_atoms = get_structure(CIFFile.read(str(reference_cif)))[0]
    candidate_atoms = get_structure(CIFFile.read(str(candidate_cif)))[0]

    ref_lengths, ref_angles = _cell_lengths_angles(reference_atoms)
    cand_lengths, cand_angles = _cell_lengths_angles(candidate_atoms)

    ref_coords, ref_symbols = _canonicalize_structure(reference_atoms)
    cand_coords, cand_symbols = _canonicalize_structure(candidate_atoms)

    matched_ref, matched_cand = _match_by_symbol(
        ref_coords, ref_symbols, cand_coords, cand_symbols
    )
    matched_rmsd_pre_align = _rmsd(matched_ref, matched_cand)

    aligned_cand = _kabsch(matched_ref, matched_cand)
    aligned_displacements = matched_ref - aligned_cand
    displacement_norms = np.linalg.norm(aligned_displacements, axis=-1)

    return CIFComparisonReport(
        reference_cif=str(reference_cif),
        candidate_cif=str(candidate_cif),
        n_atoms_reference=int(len(reference_atoms)),
        n_atoms_candidate=int(len(candidate_atoms)),
        formula_reference=_formula(reference_atoms),
        formula_candidate=_formula(candidate_atoms),
        cell_lengths_reference=ref_lengths.tolist(),
        cell_lengths_candidate=cand_lengths.tolist(),
        cell_angles_reference=ref_angles.tolist(),
        cell_angles_candidate=cand_angles.tolist(),
        matched_rmsd_pre_align=matched_rmsd_pre_align,
        matched_rmsd_post_align=_rmsd(matched_ref, aligned_cand),
        max_atom_displacement_post_align=float(np.max(displacement_norms)),
        mean_atom_displacement_post_align=float(np.mean(displacement_norms)),
    )


def _format_report(report: CIFComparisonReport) -> str:
    """Format the comparison report for human-readable output."""
    lines = [
        f"reference: {report.reference_cif}",
        f"candidate: {report.candidate_cif}",
        f"atoms: {report.n_atoms_reference} vs {report.n_atoms_candidate}",
        f"formula: {report.formula_reference} vs {report.formula_candidate}",
        "cell lengths (Å): "
        f"{np.array(report.cell_lengths_reference)} vs {np.array(report.cell_lengths_candidate)}",
        "cell angles (deg): "
        f"{np.array(report.cell_angles_reference)} vs {np.array(report.cell_angles_candidate)}",
        f"matched RMSD before Kabsch (Å): {report.matched_rmsd_pre_align:.6f}",
        f"matched RMSD after Kabsch  (Å): {report.matched_rmsd_post_align:.6f}",
        f"max atom displacement after Kabsch (Å): {report.max_atom_displacement_post_align:.6f}",
        f"mean atom displacement after Kabsch (Å): {report.mean_atom_displacement_post_align:.6f}",
    ]
    return "\n".join(lines)


def main(
    reference_cif: Path = typer.Argument(..., exists=True, readable=True),
    candidate_cif: Path = typer.Argument(..., exists=True, readable=True),
    json_output: Path | None = typer.Option(
        None,
        "--json-output",
        help="Optional path for machine-readable JSON output.",
    ),
) -> None:
    """Compare two CIF files and print a periodic, rigid-alignment-aware report."""
    report = compare_cifs(reference_cif, candidate_cif)
    typer.echo(_format_report(report))
    if json_output is not None:
        json_output.write_text(json.dumps(asdict(report), indent=2) + "\n")


if __name__ == "__main__":
    typer.run(main)
