#!/usr/bin/env python3
"""Matched MP-20 embedding analysis using the PlatonicRep v1 archive.

Run prepare, extract (on a GPU), then plot. See the manuscript reproduction note.
External pickle inputs are restricted to NumPy arrays/scalars and builtins.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import pickle
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = "zatom2-table1-zatom2-l-joint-properties-ablation"
REFERENCE_FILES = {
    "MACE-small": "small_all_atom_embeddings.pkl",
    "MACE-large": "large_all_atom_embeddings.pkl",
    "Orb-v3-direct": "orb_all_atoms_embeddings_flat_direct.pkl",
    "Untrained MACE": "dummy_all_atom_embeddings.pkl",
}
UPSTREAM_COMMIT = "56cba2354f0879c536dac68d70d509fc82a2c6ae"


def download(args):
    """Download the versioned public archive with resumable file-level caching."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    source = args.work / "source"
    source.mkdir(parents=True, exist_ok=True)
    for filename, url in {
        "zenodo.json": "https://zenodo.org/api/records/17721681",
        "commit.json": f"https://api.github.com/repos/WMD-group/PlatonicRep/commits/{UPSTREAM_COMMIT}",
        "PlatonicRep.ipynb": f"https://raw.githubusercontent.com/WMD-group/PlatonicRep/{UPSTREAM_COMMIT}/PlatonicRep.ipynb",
    }.items():
        if not (source / filename).exists():
            urllib.request.urlretrieve(url, source / filename)
    archive = json.loads((source / "zenodo.json").read_text())
    wanted = {"train.csv", "small_all_atom_types.pkl", *REFERENCE_FILES.values()}

    def fetch(item):
        path = source / item["key"]
        if not path.exists() or path.stat().st_size != item["size"]:
            partial = path.with_suffix(path.suffix + ".partial")
            urllib.request.urlretrieve(item["links"]["self"], partial)
            partial.replace(path)
        print(f"Downloaded {item['key']}", flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, [x for x in archive["files"] if x["key"] in wanted]))


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class NumpyUnpickler(pickle.Unpickler):
    """Read archived arrays without permitting arbitrary pickle globals."""

    def find_class(self, module, name):
        allowed = {
            ("numpy", "ndarray"),
            ("numpy", "dtype"),
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "scalar"),
            ("numpy._core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "scalar"),
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported pickle global: {module}.{name}")
        return super().find_class(module, name)


def read_array(path):
    with Path(path).open("rb") as handle:
        return np.asarray(NumpyUnpickler(handle).load())


def prepare(args):
    """Pin the archive, select structures, and validate every selected atom."""
    import ase_db_backends.aselmdb  # noqa: F401
    import pandas as pd
    from ase.db import connect
    from ase.io import read

    source = args.work / "source"
    archive = json.loads((source / "zenodo.json").read_text())
    records = {}
    for item in archive["files"]:
        path = source / item["key"]
        if path.exists() and item["key"] in {
            "train.csv",
            "small_all_atom_types.pkl",
            *REFERENCE_FILES.values(),
        }:
            assert path.stat().st_size == item["size"], path
            with path.open("rb") as handle:
                checksum = hashlib.file_digest(handle, "md5").hexdigest()
            assert "md5:" + checksum == item["checksum"], path
            records[item["key"]] = {
                "sha256": sha256(path),
                "url": item["links"]["self"],
            }
    assert len(records) == 6
    frame = pd.read_csv(source / "train.csv")
    # MP-20 CIFs explicitly list P1 sites. Validate against ASE for all selected
    # structures and against the archive's complete atomic-number sequence.
    counts = np.array(
        [
            len(re.findall(r"^\s+[A-Z][a-z]?\s+[A-Z][a-z]?\d+\s+\d+\s", cif, re.M))
            for cif in frame.cif
        ]
    )
    offsets = np.r_[0, counts.cumsum()]
    elements = read_array(source / "small_all_atom_types.pkl").astype(int)
    assert offsets[-1] == len(elements) == 282847
    notebook = json.loads((source / "PlatonicRep.ipynb").read_text())
    code = "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    )
    anchors = None
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "anchor_indices" for t in node.targets
        ):
            anchors = np.asarray(ast.literal_eval(node.value), dtype=int)
    assert anchors is not None and len(np.unique(anchors)) == 100
    rng = np.random.default_rng(args.seed)
    random_anchors = rng.choice(len(elements), 100, replace=False)
    anchor_structures = np.unique(
        np.searchsorted(offsets[1:], np.r_[anchors, random_anchors], side="right")
    )
    candidates = np.setdiff1d(np.arange(len(frame)), anchor_structures)
    evaluation = np.sort(rng.choice(candidates, args.structures, replace=False))
    selected = np.union1d(anchor_structures, evaluation)
    indices = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in selected])
    db_path = args.work / "mp20.aselmdb"
    if db_path.exists():
        raise FileExistsError(f"Refusing to overwrite {db_path}")
    with connect(db_path, type="aselmdb", use_lock_file=False, serial=True) as database:
        for i in selected:
            atoms = read(io.StringIO(frame.iloc[i].cif), format="cif")
            np.testing.assert_array_equal(
                atoms.numbers, elements[offsets[i] : offsets[i + 1]]
            )
            database.write(
                atoms,
                data={"source_row": int(i)},
                material_id=frame.iloc[i].material_id,
            )
    np.savez(
        args.work / "selection.npz",
        indices=indices,
        elements=elements[indices],
        structure_rows=np.repeat(selected, counts[selected]),
        selected=selected,
        evaluation=evaluation,
        anchors=anchors,
        random_anchors=random_anchors,
    )
    for name, filename in REFERENCE_FILES.items():
        values = read_array(source / filename)
        assert values.ndim == 2 and len(values) == len(elements), (name, values.shape)
        assert np.isfinite(values).all()
        np.save(args.work / (name + ".npy"), values[indices].astype(np.float32))
    write_json(
        args.work / "provenance.json",
        {
            "seed": args.seed,
            "evaluation_structures": len(evaluation),
            "extracted_structures": len(selected),
            "extracted_atoms": len(indices),
            "checkpoint_alias": CHECKPOINT,
            "archive": "10.5281/zenodo.17721681",
            "source_files": records,
            "notebook_sha256": sha256(source / "PlatonicRep.ipynb"),
            "upstream_commit": json.loads((source / "commit.json").read_text())["sha"],
            "selection_sha256": sha256(args.work / "selection.npz"),
            "material_ids": frame.iloc[selected].material_id.tolist(),
        },
    )
    print(f"Prepared {len(selected)} structures / {len(indices)} atoms", flush=True)


def extract(args):
    """Evaluate the installed EMA checkpoint once per clean crystal at t=1."""
    import os
    import random

    import torch
    from dotenv import load_dotenv
    from hydra import compose, initialize_config_dir
    from zatom2.mlip_evaluation import _build_dataset
    from zatom2.run_inference import build_inference_engine

    from nucleus.utils.batching import SparseCollate
    from nucleus.utils.inference import find_ema_wrapper

    load_dotenv(ROOT / ".env", override=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    with initialize_config_dir(
        config_dir=str(ROOT / "models/zatom2/configs"), version_base="1.3"
    ):
        cfg = compose(
            config_name="mlip_evaluation",
            overrides=[
                f"ckpt_path={CHECKPOINT}",
                f"out_dir={args.work}",
                f"seed={args.seed}",
                "evaluation.num_workers=0",
            ],
        )
    engine = build_inference_engine(cfg)
    train_cfg = engine.initialize()
    trainer = engine.trainer
    ema = find_ema_wrapper(trainer.state["model"])
    assert ema is not None, "Expected checkpoint EMA weights"
    model = ema.shadow.eval()
    dataset = _build_dataset(train_cfg, "omat24", str(args.work / "mp20.aselmdb"), 64)
    # Preserve a source-site identifier through tokenization and any sorting.
    original_loader = dataset.loader

    def indexed_loader(raw):
        loaded = original_loader(raw)
        loaded["atom_array"].set_annotation(
            "alignment_source_site", np.arange(len(loaded["atom_array"]))
        )
        return loaded

    dataset.loader = indexed_loader
    selection = np.load(args.work / "selection.npz")
    parts = args.work / "parts"
    parts.mkdir(exist_ok=True)
    for i in range(len(dataset)):
        path = parts / f"{i:05d}.npz"
        if path.exists():
            continue
        example = dataset[i]
        atom_array = example["atom_array"]
        order = np.argsort(atom_array.alignment_source_site)
        n = len(order)
        assert sorted(atom_array.alignment_source_site.tolist()) == list(range(n))
        collated = trainer.fabric.to_device(SparseCollate()([example]))
        network = trainer._assemble_network_inputs(collated)
        inputs = trainer._single_view_property_input(network)
        with torch.inference_mode(), trainer.fabric.autocast():
            outputs = model.predict_properties(
                inputs, return_embeddings=True, allow_missing_elemental_references=True
            )
        embedding = outputs["Q_L"].float().cpu().numpy().reshape(n, -1)[order]
        assert np.isfinite(embedding).all()
        np.savez(
            path, embeddings=embedding, elements=np.asarray(atom_array.element)[order]
        )
        if i % 50 == 0:
            print(
                f"Extracted {i+1}/{len(dataset)} structures; Q_L={embedding.shape}",
                flush=True,
            )
    from ase.data import atomic_numbers

    values, elements = [], []
    for i in range(len(dataset)):
        part = np.load(parts / f"{i:05d}.npz")
        values.append(part["embeddings"])
        elements.extend(atomic_numbers[x.title()] for x in part["elements"])
    np.testing.assert_array_equal(elements, selection["elements"])
    np.save(args.work / "Zatom-2-L.npy", np.concatenate(values))
    metadata = json.loads((args.work / "provenance.json").read_text())
    metadata.update(
        checkpoint_sha256=sha256(engine.ckpt_path),
        embedding="Q_L",
        endpoint=1,
        atom_types="known",
        weights="EMA",
        torch_version=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        job_id=os.environ.get("SLURM_JOB_ID"),
        precision=trainer.fabric._precision.precision,
        embedding_shape=list(np.concatenate(values).shape),
    )
    write_json(args.work / "provenance.json", metadata)


def relative_embeddings(values, anchors):
    """Cosine coordinates, matching PlatonicRep's uncentered projection."""
    values = np.asarray(values, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    anchor_norms = np.linalg.norm(anchors, axis=1, keepdims=True)
    if np.any(norms == 0) or np.any(anchor_norms == 0):
        raise ValueError("Cosine coordinates require nonzero embeddings")
    return (values / norms) @ (anchors / anchor_norms).T


def alignment_scores(first, second, k=10):
    """Centered unit-Frobenius Procrustes, linear CKA, and kNN Jaccard.

    Procrustes uses 1 - residual / 2 after separately centering and normalizing
    both matrices. Neighborhoods use Euclidean distance without self-neighbors;
    squared distances are rounded to 12 decimals before stable sorting to make
    numerical ties reproducible for the archived element-only dummy features.
    """
    from scipy.spatial.distance import cdist

    if first.shape[0] != second.shape[0] or not 0 < k < len(first):
        raise ValueError("Expected paired samples and 0 < k < sample count")
    a = np.asarray(first, dtype=np.float64) - np.mean(first, axis=0)
    b = np.asarray(second, dtype=np.float64) - np.mean(second, axis=0)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Embeddings must be finite")
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        raise ValueError("Alignment requires nonconstant embeddings")
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    cross = a.T @ b
    procrustes = np.linalg.svd(cross, compute_uv=False).sum()
    cka = np.sum(cross**2) / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b))
    neighbors = []
    for values in (first, second):
        distances = np.round(cdist(values, values, metric="sqeuclidean"), 12)
        np.fill_diagonal(distances, np.inf)
        neighbors.append(np.argsort(distances, axis=1, kind="stable")[:, :k])
    intersection = (neighbors[0][:, :, None] == neighbors[1][:, None, :]).sum(
        axis=(1, 2)
    )
    jaccard = np.mean(intersection / (2 * k - intersection))
    return np.array([procrustes, cka, jaccard])


def element_families(numbers):
    """Stable chemical categories for the shared scatterplot legend."""
    from pymatgen.core import Element

    labels = []
    for number in numbers:
        element = Element.from_Z(int(number))
        if element.is_lanthanoid or element.is_actinoid:
            family = "Lanthanides / actinides"
        elif element.is_transition_metal:
            family = "Transition metals"
        elif element.is_alkali:
            family = "Alkali metals"
        elif element.is_alkaline:
            family = "Alkaline earths"
        elif element.is_halogen:
            family = "Halogens"
        elif element.is_chalcogen:
            family = "Chalcogens"
        elif element.is_metalloid:
            family = "Metalloids"
        elif element.is_metal:
            family = "Other metals"
        else:
            family = "Other nonmetals"
        labels.append(family)
    return np.array(labels)


def plot(args):
    """Export vector figures and all plotted numbers from the saved embeddings."""
    import importlib.metadata

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.decomposition import PCA

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )
    destination = ROOT / "models/zatom2/docs/manuscript/figs"
    provenance = json.loads((args.work / "provenance.json").read_text())
    provenance["analysis_script_sha256"] = sha256(__file__)
    provenance["analysis_software_versions"] = {
        name: importlib.metadata.version(name)
        for name in (
            "numpy",
            "scipy",
            "scikit-learn",
            "matplotlib",
            "pandas",
            "ase",
            "pymatgen",
        )
    }
    if args.seed != provenance["seed"]:
        raise ValueError("Plot seed must match the prepared selection")
    selection = np.load(args.work / "selection.npz")
    names = ["Zatom-2-L", *REFERENCE_FILES]
    raw = {
        name: np.load(args.work / (name + ".npy")).astype(np.float64) for name in names
    }
    for name, values in raw.items():
        assert (
            len(values) == len(selection["indices"]) and np.isfinite(values).all()
        ), name
    anchor_positions = np.searchsorted(selection["indices"], selection["anchors"])
    random_positions = np.searchsorted(
        selection["indices"], selection["random_anchors"]
    )
    np.testing.assert_array_equal(
        selection["indices"][anchor_positions], selection["anchors"]
    )
    np.testing.assert_array_equal(
        selection["indices"][random_positions], selection["random_anchors"]
    )
    eligible = np.flatnonzero(
        np.isin(selection["structure_rows"], selection["evaluation"])
    )
    rng = np.random.default_rng(args.seed)
    display = rng.choice(eligible, min(5000, len(eligible)), replace=False)
    # Exactly one atom per sampled structure avoids trivial same-crystal
    # symmetry copies in neighborhood scoring and gives equal structure weight.
    structure_sample = rng.choice(selection["evaluation"], 1000, replace=False)
    sample = np.array(
        [
            rng.choice(np.flatnonzero(selection["structure_rows"] == row))
            for row in structure_sample
        ]
    )
    projected = {
        name: relative_embeddings(values, values[anchor_positions])
        for name, values in raw.items()
    }
    matrices = np.zeros((3, len(names), len(names)))
    for i, name in enumerate(names):
        for j in range(i, len(names)):
            scores = alignment_scores(
                projected[name][sample], projected[names[j]][sample]
            )
            matrices[:, i, j] = matrices[:, j, i] = scores
    metric_names = ["Procrustes", "Linear CKA", "10-NN Jaccard"]
    families = element_families(selection["elements"][display])
    categories = [
        "Alkali metals",
        "Alkaline earths",
        "Transition metals",
        "Lanthanides / actinides",
        "Other metals",
        "Metalloids",
        "Chalcogens",
        "Halogens",
        "Other nonmetals",
    ]
    colors = [
        "#0072B2",
        "#56B4E9",
        "#777777",
        "#CC79A7",
        "#882255",
        "#AA4499",
        "#E69F00",
        "#009E73",
        "#332288",
    ]
    figure_names = []

    def save(fig, stem):
        fig.savefig(
            destination / f"{stem}.pdf",
            bbox_inches="tight",
            metadata={"CreationDate": None, "ModDate": None},
        )
        fig.savefig(destination / f"{stem}.png", bbox_inches="tight")
        figure_names.append(stem)
        plt.close(fig)

    shown = names[:-1]
    common_pca = PCA(n_components=2, svd_solver="full").fit(
        np.concatenate([projected[n][display] for n in shown])
    )
    joint = {name: common_pca.transform(projected[name][display]) for name in shown}
    limits = np.concatenate(list(joint.values()))
    bounds = [
        (limits[:, dim].min() - 0.15, limits[:, dim].max() + 0.15) for dim in range(2)
    ]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.6))
    for column, name in enumerate(shown):
        native = PCA(n_components=2, svd_solver="full").fit_transform(
            raw[name][display]
        )
        for row, xy in enumerate((native, joint[name])):
            ax = axes[row, column]
            for category, color in zip(categories, colors, strict=True):
                mask = families == category
                ax.scatter(
                    xy[mask, 0],
                    xy[mask, 1],
                    s=2.2,
                    c=color,
                    alpha=0.55,
                    linewidths=0,
                    rasterized=True,
                )
            ax.set_xlabel("Native PC 1" if row == 0 else "Shared PC 1")
            if column == 0:
                ax.set_ylabel("Native PC 2" if row == 0 else "Shared PC 2")
            if row == 0:
                ax.set_title(name, fontweight="bold")
            else:
                ax.set_xlim(bounds[0])
                ax.set_ylim(bounds[1])
            ax.tick_params(length=2)
    fig.text(
        0.015, 0.97, "a   Original representations · separate PCA fits", weight="bold"
    )
    fig.text(
        0.015,
        0.54,
        "b   Relative representations · 100 published anchors · one shared PCA fit",
        weight="bold",
    )
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=4, color=c, label=n)
        for n, c in zip(categories, colors, strict=True)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(top=0.90, bottom=0.24, hspace=0.85, wspace=0.48)
    save(fig, "representation_alignment_geometry")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    short = ["Zatom-2-L", "MACE-S", "MACE-L", "Orb-v3", "Untrained"]
    for index, (ax, title, matrix) in enumerate(
        zip(axes, metric_names, matrices, strict=True)
    ):
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="cividis")
        ax.set_title(f"{'abc'[index]}   {title}", loc="left", weight="bold")
        ax.set_xticks(range(5), short, rotation=50, ha="right")
        ax.set_yticks(range(5), short if index == 0 else [""] * 5)
        for i in range(5):
            for j in range(5):
                ax.text(
                    j,
                    i,
                    f"{matrix[i,j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if matrix[i, j] < 0.45 else "black",
                )
        ax.tick_params(length=0)
    fig.subplots_adjust(wspace=0.10, right=0.92, bottom=0.28, top=0.90)
    color_ax = fig.add_axes([0.94, 0.29, 0.013, 0.59])
    fig.colorbar(im, cax=color_ax, ticks=[0, 0.5, 1])
    save(fig, "representation_alignment_metrics")

    # Match the paper's anchor-count ablation up to the 100 published anchors.
    sizes = [3, 8, 20, 50, 100]
    convergence = {}
    controls = {}
    element_sample = selection["elements"][sample]
    permutations = []
    for _ in range(20):
        permutation = np.arange(len(sample))
        for element in np.unique(element_sample):
            positions = np.flatnonzero(element_sample == element)
            permutation[positions] = rng.permutation(positions)
        permutations.append(permutation)
    for name in names[1:]:
        convergence[name] = {}
        for strategy, positions in [
            ("Published", anchor_positions),
            ("Random", random_positions),
        ]:
            convergence[name][strategy] = [
                alignment_scores(
                    relative_embeddings(
                        raw[names[0]][sample], raw[names[0]][positions[:size]]
                    ),
                    relative_embeddings(raw[name][sample], raw[name][positions[:size]]),
                ).tolist()
                for size in sizes
            ]
        controls[name] = [
            alignment_scores(
                projected[names[0]][sample], projected[name][sample][permutation]
            ).tolist()
            for permutation in permutations
        ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    model_colors = ["#0072B2", "#009E73", "#D55E00", "#888888"]
    for metric, ax in enumerate(axes):
        for name, color in zip(names[1:], model_colors, strict=True):
            for strategy, style in [("Published", "-"), ("Random", "--")]:
                ax.plot(
                    sizes,
                    np.array(convergence[name][strategy])[:, metric],
                    style,
                    marker="o",
                    markersize=2.5,
                    linewidth=1.1,
                    color=color,
                )
        ax.set_xscale("log")
        ax.set_xticks(sizes, sizes)
        ax.set_xlabel("Number of anchors")
        ax.set_title(
            f"{'abc'[metric]}   {metric_names[metric]}", loc="left", weight="bold"
        )
        ax.set_ylim(0, 1 if metric < 2 else 0.15)
        ax.grid(axis="y", alpha=0.2)
    handles = [
        Line2D([], [], color=c, label=n, linewidth=1.5)
        for n, c in zip(names[1:], model_colors, strict=True)
    ]
    handles += [
        Line2D([], [], color="black", linestyle=s, label=n)
        for n, s in [("Published anchors", "-"), ("Random anchors", "--")]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=0.33, wspace=0.35, top=0.86)
    save(fig, "representation_alignment_anchors")
    result = {
        "models": names,
        "metrics": metric_names,
        "matrices": matrices.tolist(),
        "metric_sample_global_atom_indices": selection["indices"][sample].tolist(),
        "display_global_atom_indices": selection["indices"][display].tolist(),
        "metric_sample_structures": structure_sample.tolist(),
        "metric_sample_size": len(sample),
        "eligible_atoms": len(eligible),
        "shared_pca_explained_variance_ratio": common_pca.explained_variance_ratio_.tolist(),
        "anchor_sizes": sizes,
        "convergence": convergence,
        "within_element_permutations": controls,
        "within_element_permutation_mean": {
            n: np.mean(v, axis=0).tolist() for n, v in controls.items()
        },
        "within_element_permutation_sd": {
            n: np.std(v, axis=0, ddof=1).tolist() for n, v in controls.items()
        },
        "embedding_sha256": {n: sha256(args.work / (n + ".npy")) for n in names},
        "figures": {n: sha256(destination / (n + ".pdf")) for n in figure_names},
    }
    pair_rng = np.random.default_rng(args.seed)
    first_pairs = pair_rng.choice(eligible, 10000)
    second_pairs = pair_rng.choice(eligible, 10000)
    result["random_pair_cosine_quantiles_05_50_95"] = {}
    for name, values in raw.items():
        normalized = values / np.linalg.norm(values, axis=1, keepdims=True)
        cosine = np.sum(normalized[first_pairs] * normalized[second_pairs], axis=1)
        result["random_pair_cosine_quantiles_05_50_95"][name] = np.quantile(
            cosine, [0.05, 0.5, 0.95]
        ).tolist()
    write_json(args.work / "results.json", result)
    write_json(destination / "representation_alignment_results.json", result)
    write_json(destination / "representation_alignment_provenance.json", provenance)
    print(
        json.dumps(
            {
                "zatom_scores": matrices[:, 0, :].tolist(),
                "control_means": result["within_element_permutation_mean"],
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["download", "prepare", "extract", "plot"])
    parser.add_argument(
        "--work", type=Path, default=ROOT / "logs/representation_alignment"
    )
    parser.add_argument("--structures", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.work = args.work.resolve()
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
