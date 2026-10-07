#!/usr/bin/env python3
"""Matched-protein neighborhood and local intervention analysis for Section 4."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from plot_zatom2_subset_embeddings import DEST, ROOT, sha256, write_json
from probe_protein_embeddings import SEED, VARIANTS


def protein_means(values, structures):
    """Give every protein equal weight, irrespective of residue count."""
    _, inverse = np.unique(structures, return_inverse=True)
    return np.bincount(inverse, weights=values) / np.bincount(inverse)


def interval(values):
    """Conditional protein bootstrap, holding checkpoint and neighbor pool fixed."""
    draws = np.random.default_rng(SEED).choice(values, (2000, len(values)))
    return np.quantile(draws.mean(axis=1), [0.025, 0.975]).tolist()


def neighbor_scores(values, labels, structures, k=10):
    """Exact Euclidean k-NN purity; exclude every residue of the query protein."""
    from sklearn.neighbors import NearestNeighbors

    _, sizes = np.unique(structures, return_counts=True)
    if len(values) - sizes.max() < k:
        raise ValueError("Insufficient external-protein neighbors")
    neighbors = NearestNeighbors(n_neighbors=k + int(sizes.max()), algorithm="brute")
    neighbors.fit(values)
    scores = np.zeros(len(labels))
    chance = np.zeros(len(labels))
    for start in range(0, len(values), 256):
        nearest = neighbors.kneighbors(
            values[start : start + 256], return_distance=False
        )
        for offset, row in enumerate(nearest):
            i = start + offset
            selected = row[structures[row] != structures[i]][:k]
            scores[i] = np.mean(labels[selected] == labels[i])
    for structure in np.unique(structures):
        query = structures == structure
        for label in np.unique(labels[query]):
            chance[query & (labels == label)] = np.mean(labels[~query] == label)
    return protein_means(scores, structures), protein_means(chance, structures)


def summarize_responses(rows):
    """Aggregate directions and target sites within each protein first."""
    summaries = {}
    for magnitude in sorted({r["magnitude"] for r in rows}):
        values = []
        for structure in sorted({r["structure"] for r in rows}):
            samples = []
            for row in rows:
                if row["magnitude"] != magnitude or row["structure"] != structure:
                    continue
                response = np.asarray(row["response"])
                target = response[row["residue"]]
                others = np.delete(response, row["residue"])
                samples.append(
                    [
                        np.sqrt(target),
                        np.sqrt(others.mean()),
                        target / response.sum() if response.sum() else 0.0,
                    ]
                )
            values.append(np.mean(samples, axis=0))
        array = np.asarray(values)
        summaries[str(magnitude)] = {
            name: dict(
                mean=float(array[:, i].mean()),
                ci=interval(array[:, i]),
                per_protein=array[:, i].tolist(),
            )
            for i, name in enumerate(("target_rms", "other_rms", "target_fraction"))
        }
    return summaries


def summarize_losses(rows):
    """Average paired loss changes over trials, then equally over proteins."""
    summaries = {}
    for magnitude in sorted({row["magnitude"] for row in rows}):
        summaries[str(magnitude)] = {}
        for metric in ("structure_loss", "sequence_loss"):
            for region in ("target", "other"):
                values, baseline_values = [], []
                for structure in sorted({row["structure"] for row in rows}):
                    changes, baselines = [], []
                    for row in rows:
                        if (
                            row["magnitude"] != magnitude
                            or row["structure"] != structure
                        ):
                            continue
                        loss = np.asarray(row[metric])
                        baseline = np.asarray(row[f"baseline_{metric}"])
                        selection = np.arange(len(loss)) == row["residue"]
                        if region == "other":
                            selection = ~selection
                        changes.append((loss - baseline)[selection].mean())
                        baselines.append(baseline[selection].mean())
                    values.append(np.mean(changes))
                    baseline_values.append(np.mean(baselines))
                values = np.asarray(values)
                summaries[str(magnitude)][f"{region}_{metric}"] = dict(
                    mean=float(values.mean()),
                    ci=interval(values),
                    per_protein=values.tolist(),
                    baseline_mean=float(np.mean(baseline_values)),
                )
    return summaries


def validate_paired_trials(rows_a, rows_b):
    """Require a complete, unique and matched protein/site/direction grid."""
    from probe_protein_embeddings import MAGNITUDES

    def keys(rows):
        result = [
            (row["structure"], row["residue"], row["direction"], row["magnitude"])
            for row in rows
        ]
        if len(result) != len(set(result)):
            raise ValueError("Duplicate perturbation trials")
        trials = {key[:3] for key in result}
        if set(result) != {
            (*trial, magnitude) for trial in trials for magnitude in MAGNITUDES
        }:
            raise ValueError("Incomplete perturbation magnitude grid")
        return result

    if keys(rows_a) != keys(rows_b):
        raise ValueError("Unpaired perturbation trials")
    for a, b in zip(rows_a, rows_b):
        if a["direction_sha256"] != b["direction_sha256"]:
            raise ValueError("Unpaired perturbation directions")
        for row in (a, b):
            for metric in ("structure_loss", "sequence_loss"):
                loss, baseline = (
                    np.asarray(row[key]) for key in (metric, f"baseline_{metric}")
                )
                if not np.isfinite(loss).all() or not np.isfinite(baseline).all():
                    raise ValueError("Nonfinite reconstruction loss")
                if row["magnitude"] == 0 and not np.array_equal(loss, baseline):
                    raise ValueError("Zero perturbation changed reconstruction loss")


def analyze(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    args.output.mkdir(parents=True, exist_ok=True)
    data, provenance = {}, {}
    results = dict(
        protocol=dict(
            seed=SEED,
            neighbors=10,
            distance="Euclidean, unscaled original representation",
            weighting="mean over residues within each protein, then over proteins",
            uncertainty="2000 paired protein bootstrap draws, fixed neighbor pools and checkpoints",
            analysis_sha256=sha256(__file__),
            software={
                name: importlib.metadata.version(name)
                for name in ("numpy", "scikit-learn", "matplotlib")
            },
        )
    )
    for variant in VARIANTS:
        path = args.work / f"embeddings_{variant}.npz"
        provenance[variant] = json.loads(
            (args.work / f"provenance_{variant}.json").read_text()
        )
        if sha256(path) != provenance[variant]["archive_sha256"]:
            raise ValueError("Embedding archive checksum mismatch")
        if (
            sha256(args.work / "manifest.json")
            != provenance[variant]["manifest_sha256"]
        ):
            raise ValueError("Selection manifest mismatch")
        data[variant] = dict(np.load(path, allow_pickle=False))
        if any(
            not np.isfinite(data[variant][key]).all()
            for key in ("encoder", "final", "geometry")
        ):
            raise ValueError("Nonfinite representation")
    a, b = (data[v] for v in VARIANTS)
    for key in (
        "labels",
        "structure",
        "geometry",
        "input_sha256",
        "perturbation_input_sha256",
    ):
        np.testing.assert_array_equal(a[key], b[key])
    labels, structures = a["labels"], a["structure"]
    results["n_residues"] = len(labels)
    results["n_proteins"] = len(np.unique(structures))
    results["embedding_dimension"] = a["encoder"].shape[1]
    results["provenance"] = provenance
    results["neighbors"] = {}
    per_protein = {}
    for name, values in [("geometry", a["geometry"])] + [
        (f"{variant}_{layer}", data[variant][layer])
        for variant in VARIANTS
        for layer in ("encoder", "final")
    ]:
        score, chance = neighbor_scores(values, labels, structures)
        per_protein[name] = score
        results["neighbors"][name] = dict(
            mean=float(score.mean()),
            ci=interval(score),
            chance=float(chance.mean()),
            per_protein=score.tolist(),
        )
        print(name, results["neighbors"][name]["mean"], flush=True)
    difference = (
        per_protein["joint_properties_fc_encoder"] - per_protein["scratch_encoder"]
    )
    results["encoder_difference"] = dict(
        mean=float(difference.mean()), ci=interval(difference)
    )
    difference = per_protein["joint_properties_fc_final"] - per_protein["scratch_final"]
    results["final_difference"] = dict(
        mean=float(difference.mean()), ci=interval(difference)
    )
    results["responses"] = {}
    results["reconstruction_loss_changes"] = {}
    response_rows = {}
    for variant in VARIANTS:
        path = args.work / f"responses_{variant}.json"
        if sha256(path) != provenance[variant]["responses_sha256"]:
            raise ValueError("Response archive checksum mismatch")
        response_rows[variant] = json.loads(path.read_text())
        results["responses"][variant] = summarize_responses(response_rows[variant])
        results["reconstruction_loss_changes"][variant] = summarize_losses(
            response_rows[variant]
        )
    validate_paired_trials(*(response_rows[variant] for variant in VARIANTS))
    if (
        provenance[VARIANTS[0]]["perturbation_protocol"]
        != provenance[VARIANTS[1]]["perturbation_protocol"]
    ):
        raise ValueError("Different perturbation protocols")
    results["perturbation_protocol"] = provenance[VARIANTS[0]]["perturbation_protocol"]

    # Label-balanced subsampling affects only the display, never the metric.
    rng = np.random.default_rng(SEED)
    selected = np.sort(
        np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(labels == label),
                    min(200, np.sum(labels == label)),
                    replace=False,
                )
                for label in sorted(set(labels))
            ]
        )
    )
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 4, figsize=(8.4, 2.7))
    palette = plt.get_cmap("tab20")
    coordinates = []
    for axis, variant, title in zip(
        axes[:2], VARIANTS, ("SCOPe-only", "Zatom-2 (V)\npretraining + SCOPe")
    ):
        values = data[variant]["final"][selected]
        projected = PCA(n_components=50, svd_solver="full").fit_transform(values)
        xy = TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            random_state=SEED,
            max_iter=1500,
        ).fit_transform(projected)
        for i, label in enumerate(sorted(set(labels))):
            mask = labels[selected] == label
            axis.scatter(
                *xy[mask].T,
                s=3,
                alpha=0.65,
                color=palette(i),
                label=label,
                rasterized=True,
            )
        axis.set(xticks=[], yticks=[], title=title)
        for index, (x, y) in zip(selected, xy):
            coordinates.append(
                dict(
                    variant=variant,
                    residue_index=int(index),
                    structure=int(structures[index]),
                    label=labels[index],
                    x=float(x),
                    y=float(y),
                )
            )
    for axis, metric, title, units in zip(
        axes[2:],
        ("sequence_loss", "structure_loss"),
        ("Sequence", "Structure"),
        (r"$\Delta$ cross-entropy (nats)", r"$\Delta$ MSE ($\mathrm{\AA}^2$)"),
    ):
        axis.axhline(0, color="0.8", linewidth=0.6)
        for variant, color, label in zip(
            VARIANTS,
            ("#67758c", "#c24739"),
            ("SCOPe-only", "Zatom-2 (V)\npretraining + SCOPe"),
        ):
            values = results["reconstruction_loss_changes"][variant]
            xs = sorted(map(float, values))
            for region, style in (("target", "-"), ("other", "--")):
                name = f"{region}_{metric}"
                means = [values[str(x)][name]["mean"] for x in xs]
                bounds = np.array([values[str(x)][name]["ci"] for x in xs])
                axis.plot(
                    xs,
                    means,
                    linestyle=style,
                    marker="o",
                    markersize=2.5,
                    color=color,
                    label=f"{label}, {region}",
                )
                axis.fill_between(
                    xs, bounds[:, 0], bounds[:, 1], color=color, alpha=0.12
                )
        axis.set(
            xlabel=r"Perturbation $\alpha$",
            ylabel=units,
            title=title,
            xticks=[0, 0.5, 1],
        )
    handles, _ = axes[3].get_legend_handles_labels()
    axes[3].legend(
        handles[::2],
        ("SCOPe-only", "Zatom-2 (V)\npretraining + SCOPe"),
        frameon=False,
        fontsize=6.5,
        handlelength=0.8,
        handletextpad=0.2,
        labelspacing=0.18,
        borderaxespad=0.08,
        loc="upper left",
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=10,
        fontsize=9,
        frameon=False,
        markerscale=2,
        columnspacing=0.8,
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1), w_pad=1.1)
    for extension in ("pdf", "png"):
        fig.savefig(
            args.output / f"protein_embeddings.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)
    with (args.output / "protein_embeddings_tsne.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coordinates[0]))
        writer.writeheader()
        writer.writerows(coordinates)
    results["tsne"] = dict(
        representation="final token trunk output; both stages quantified above",
        n_points=len(selected),
        max_per_residue_type=200,
        perplexity=30,
        pca_dimensions=50,
        iterations=1500,
        independent_fits=True,
    )
    write_json(args.output / "protein_embeddings_results.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT / "logs/protein_embeddings")
    parser.add_argument("--output", type=Path, default=DEST)
    analyze(parser.parse_args())
