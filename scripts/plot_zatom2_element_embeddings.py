#!/usr/bin/env python3
"""Element-resolved companion to the Appendix C.4 structure-pooled PCA.

Extract with plot_zatom2_subset_embeddings.py extract --atomwise for each variant.
This CPU-only analysis verifies matched inputs and plots the most widely sampled
four elements per domain (or by minimum domain coverage for the joint view).
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from plot_zatom2_subset_embeddings import CHECKPOINTS, DEST, ROOT, sha256, write_json


def structure_means(values, structures):
    """Average atom diagnostics within each structure before averaging systems."""
    unique, inverse = np.unique(structures, return_inverse=True)
    return unique, np.bincount(inverse, weights=values) / np.bincount(inverse)


def neighbor_scores(values, labels, structures, k=10):
    """Same-label 10-NN fractions, excluding all atoms of the query structure.

    Use exact Euclidean distances in the original 128-dimensional space. The
    random-neighbor baseline respects the same exclusion and label imbalance.
    """
    from sklearn.neighbors import NearestNeighbors

    _, sizes = np.unique(structures, return_counts=True)
    if len(values) - sizes.max() < k:
        raise ValueError("Insufficient atoms from other structures")
    candidates = NearestNeighbors(
        n_neighbors=min(len(values), k + int(sizes.max())), algorithm="brute"
    ).fit(values)
    purity = np.empty(len(values))
    chance = np.empty(len(values))
    for start in range(0, len(values), 256):
        indices = candidates.kneighbors(
            values[start : start + 256], return_distance=False
        )
        for offset, row in enumerate(indices):
            i = start + offset
            neighbors = row[structures[row] != structures[i]][:k]
            if len(neighbors) != k:
                raise ValueError("Incomplete cross-structure neighborhood")
            purity[i] = np.mean(labels[neighbors] == labels[i])
    for structure in np.unique(structures):
        query = structures == structure
        if len(np.unique(labels[query])) != 1:
            raise ValueError("A structure must have one domain/subset label")
        chance[query] = np.mean(labels[~query] == labels[query][0])
    return structure_means(purity, structures)[1], structure_means(chance, structures)[
        1
    ]


def paired_interval(differences, seed=20260925):
    """Descriptive paired structure bootstrap; training-seed uncertainty excluded."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(2000, len(differences)), replace=True)
    return np.quantile(draws.mean(axis=1), [0.025, 0.975]).tolist()


def load_matched(work, samples):
    """Check archive integrity, atom correspondence, and Figure 12 provenance."""
    data, provenance = {}, {}
    for variant in CHECKPOINTS:
        path = work / f"element_embeddings_{variant}.npz"
        provenance[variant] = json.loads(
            path.with_name(path.stem + "_provenance.json").read_text()
        )
        if sha256(path) != provenance[variant]["archive_sha256"]:
            raise ValueError(f"Archive checksum mismatch: {variant}")
        if sha256(samples) != provenance[variant]["sample_manifest_sha256"]:
            raise ValueError("Sample manifest mismatch")
        with np.load(path, allow_pickle=False) as archive:
            data[variant] = dict(archive)
        values = data[variant]
        if values["atom_embeddings"].shape != (len(values["atomic_numbers"]), 128):
            raise ValueError("Expected one 128-dimensional embedding per atom")
        if not np.isfinite(values["atom_embeddings"]).all():
            raise ValueError("Non-finite atom embedding")
        np.testing.assert_array_equal(
            np.bincount(values["structure_index"]), values["n_atoms"]
        )
        # A rerun must still reproduce the original figure's processed inputs.
        with np.load(DEST / f"subset_embeddings_{variant}.npz") as original:
            for key in ("input_sha256", "domain", "subset", "n_atoms", "composition"):
                np.testing.assert_array_equal(values[key], original[key])
    left, right = (data[v] for v in CHECKPOINTS)
    for key in (
        "input_sha256",
        "domain",
        "subset",
        "n_atoms",
        "atomic_numbers",
        "structure_index",
    ):
        np.testing.assert_array_equal(left[key], right[key])
    for key in (
        "common_preprocessing_checkpoint_sha256",
        "precision",
        "seed",
        "backbone_parameters",
    ):
        if len({p[key] for p in provenance.values()}) != 1:
            raise ValueError(f"Unmatched extraction setting: {key}")
    return data, provenance


def analyze(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ase.data import chemical_symbols
    from sklearn.decomposition import PCA

    data, provenance = load_matched(args.work, args.samples)
    reference = data["without_properties"]
    structures = reference["structure_index"]
    domains = reference["domain"][structures]
    elements = reference["atomic_numbers"]
    subsets = reference["subset"][structures]
    coverage = {
        d: {
            int(z): len(np.unique(structures[(elements == z) & (domains == d)]))
            for z in np.unique(elements)
        }
        for d in ("omol25", "omat24")
    }
    results = {
        "protocol": {
            "max_atoms_per_element_domain": args.max_atoms,
            "min_structures_per_element_domain": args.min_structures,
            "seed": args.seed,
            "pca": "independent centered full-SVD PCA, equal atom weight, no scaling",
            "neighbors": "10 Euclidean neighbors in 128D, exclude every atom of query structure",
            "aggregation": "mean over query atoms within structure, then mean over structures",
            "interval": "2000 paired structure bootstrap draws; fixed checkpoints and atom sample",
            "selection": "top four by structure coverage; joint ranks minimum coverage across domains",
            "extraction_provenance": provenance,
            "analysis_script_sha256": sha256(__file__),
            "software": {
                n: importlib.metadata.version(n)
                for n in ("numpy", "scikit-learn", "matplotlib")
            },
        }
    }
    coordinate_rows, metric_rows = [], []
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    for view_index, view in enumerate(("joint", "omol25", "omat24")):
        score = {
            z: min(coverage[d][z] for d in coverage)
            if view == "joint"
            else coverage[view][z]
            for z in np.unique(elements)
        }
        eligible = sorted(
            (int(z) for z in score if score[z] >= args.min_structures),
            key=lambda z: (-score[z], z),
        )
        shown = eligible[:4]
        if len(shown) != 4:
            raise ValueError(f"Fewer than four sufficiently sampled elements: {view}")
        results[view] = {
            "shown_elements": [chemical_symbols[z] for z in shown],
            "elements": {},
        }
        groups = sorted(set(domains if view == "joint" else subsets[domains == view]))
        palette = plt.get_cmap("tab10" if len(groups) <= 10 else "tab20")
        colors = {g: palette(i) for i, g in enumerate(groups)}
        fig, axes = plt.subplots(4, 2, figsize=(7.4, 9.0))
        for z in eligible:
            rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, int(z), view_index])
            )
            selected = []
            for domain in ("omol25", "omat24") if view == "joint" else (view,):
                indices = np.flatnonzero((elements == z) & (domains == domain))
                selected.extend(
                    rng.choice(
                        indices, min(args.max_atoms, len(indices)), replace=False
                    )
                )
            selected = np.sort(selected)
            labels = (domains if view == "joint" else subsets)[selected]
            selected_structures = structures[selected]
            element_result = {
                "n_atoms": len(selected),
                "n_structures": len(np.unique(selected_structures)),
                "label_atom_counts": {g: int(np.sum(labels == g)) for g in groups},
                "structure_coverage": {d: coverage[d][z] for d in coverage},
            }
            scores = {}
            for column, variant in enumerate(CHECKPOINTS):
                values = data[variant]["atom_embeddings"][selected].astype(np.float64)
                scores[variant], chance = neighbor_scores(
                    values, labels, selected_structures
                )
                pca = PCA(n_components=2, svd_solver="full")
                xy = pca.fit_transform(values)
                element_result[variant] = {
                    "purity": float(scores[variant].mean()),
                    "chance": float(chance.mean()),
                    "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                }
                if z in shown:
                    row = shown.index(z)
                    ax = axes[row, column]
                    for group in groups:
                        keep = labels == group
                        ax.scatter(
                            xy[keep, 0],
                            xy[keep, 1],
                            s=4,
                            alpha=0.5,
                            color=colors[group],
                            linewidths=0,
                            rasterized=True,
                        )
                    ax.set_title(
                        f"{chr(97 + 2 * row + column)}  {chemical_symbols[z]}: "
                        f"{len(selected):,} atoms / {len(np.unique(selected_structures))} structures",
                        fontsize=8,
                        loc="left",
                        weight="bold",
                    )
                    ax.set_xlabel(
                        f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=8
                    )
                    ax.set_ylabel(
                        f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=8
                    )
                    ax.tick_params(labelsize=7)
                    ax.spines[["top", "right"]].set_visible(False)
                    for index, point in zip(selected, xy, strict=True):
                        coordinate_rows.append(
                            {
                                "view": view,
                                "element": chemical_symbols[z],
                                "model": variant,
                                "atom_archive_index": int(index),
                                "structure_index": int(structures[index]),
                                "label": str(
                                    (domains if view == "joint" else subsets)[index]
                                ),
                                "pc1": point[0],
                                "pc2": point[1],
                            }
                        )
            delta = scores["with_properties"] - scores["without_properties"]
            element_result["purity_difference"] = float(delta.mean())
            element_result["paired_structure_bootstrap_95ci"] = paired_interval(
                delta, args.seed
            )
            results[view]["elements"][chemical_symbols[z]] = element_result
            metric_rows.append(
                {
                    "view": view,
                    "element": chemical_symbols[z],
                    "shown": z in shown,
                    "n_atoms": len(selected),
                    "n_structures": element_result["n_structures"],
                    "without_mlip": float(scores["without_properties"].mean()),
                    "with_mlip": float(scores["with_properties"].mean()),
                    "chance": float(chance.mean()),
                    "difference": float(delta.mean()),
                    "ci_low": element_result["paired_structure_bootstrap_95ci"][0],
                    "ci_high": element_result["paired_structure_bootstrap_95ci"][1],
                }
            )
            print(
                view,
                chemical_symbols[z],
                element_result["purity_difference"],
                flush=True,
            )
        handles = [
            plt.Line2D([], [], marker="o", linestyle="", color=colors[g], markersize=4)
            for g in groups
        ]
        names = [
            g.replace("aimd-from-PBE-", "AIMD ")
            .replace("-subsampled", " (sub)")
            .replace("omol25", "OMol25")
            .replace("omat24", "OMat24")
            for g in groups
        ]
        fig.legend(
            handles,
            names,
            loc="lower center",
            ncol=2 if view == "joint" else 3,
            frameon=False,
            fontsize=7,
            columnspacing=1.4,
            handletextpad=0.3,
        )
        fig.subplots_adjust(
            left=0.10,
            right=0.98,
            top=0.92,
            bottom=0.09 if view != "omat24" else 0.12,
            hspace=0.65,
            wspace=0.32,
        )
        fig.text(0.29, 0.97, "Zatom-2 (IV): without MLIP", ha="center", weight="bold")
        fig.text(0.78, 0.97, "Zatom-2 (V): with MLIP", ha="center", weight="bold")
        for suffix in ("pdf", "png"):
            fig.savefig(
                args.output / f"element_embeddings_{view}_pca.{suffix}",
                dpi=220,
                bbox_inches="tight",
            )
        plt.close(fig)
    for name, rows in (("pca", coordinate_rows), ("metrics", metric_rows)):
        with (args.output / f"element_embeddings_{name}.csv").open("w") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    write_json(args.output / "element_embeddings_results.json", results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT / "logs/element_embeddings")
    parser.add_argument("--output", type=Path, default=DEST)
    parser.add_argument(
        "--samples", type=Path, default=DEST / "subset_embeddings_samples.json"
    )
    parser.add_argument("--max-atoms", type=int, default=4000)
    parser.add_argument("--min-structures", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    analyze(args)


if __name__ == "__main__":
    main()
