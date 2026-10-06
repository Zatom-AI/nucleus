#!/usr/bin/env python3
"""Compare base Zatom-2 embeddings with and without property pretraining.

Extraction replays the saved Appendix C validation manifest through a common
preprocessing pipeline. Plotting requires only the two compact embedding archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models/zatom2/docs/manuscript/figs"
CHECKPOINTS = {
    "without_properties": "zatom2-table1-joint-ablation",
    "with_properties": "zatom2-table1-joint-properties-ablation",
}
TITLES = {
    "without_properties": "Zatom-2 (IV): without E/F prediction",
    "with_properties": "Zatom-2 (V): with E/F prediction",
}


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def resolve_checkpoint(value):
    """Resolve an existing local checkpoint without downloading anything."""
    from nucleus.inference_engines.checkpoint_registry import REGISTERED_CHECKPOINTS

    path = (
        REGISTERED_CHECKPOINTS[value].get_default_path()
        if value in REGISTERED_CHECKPOINTS
        else Path(value).expanduser()
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Install the checkpoint or provide its local path: {path}"
        )
    return path.resolve()


def input_digest(features, coordinates):
    """Hash actual backbone inputs to verify matching across independent runs."""
    import torch

    digest = hashlib.sha256()

    def visit(value):
        if torch.is_tensor(value):
            tensor = value.detach().contiguous().cpu()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                digest.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode())

    visit(features)
    visit(coordinates)
    return digest.hexdigest()


def backbone_embedding(model, inputs):
    """Capture the final atom decoder output without requiring property heads."""
    import torch

    features = model._copy_property_features(inputs)
    model._neutralize_mean_force_conditioning(features)
    model._expose_property_electronic_metadata(features)
    model._validate_known_property_atom_types(features)
    coordinates = inputs["X_clean_L"]
    if coordinates.ndim == 2:
        coordinates = coordinates.unsqueeze(0)
    features["diffusion_batch_size"] = coordinates.shape[0]
    batch_size = int(features["batch_size"])
    signature = input_digest(features, coordinates)
    captured = []

    def capture(_module, _args, outputs):
        captured.append(outputs["Q_L"].detach())

    hook = model.diffusion_module.backbone.register_forward_hook(capture)
    try:
        model.diffusion_module(
            X_noisy_L=coordinates,
            t=torch.ones(
                (coordinates.shape[0], batch_size),
                device=coordinates.device,
                dtype=coordinates.dtype,
            ),
            f=features,
            n_recycle=1,
            **model.token_initializer(features),
        )
    finally:
        hook.remove()
    if len(captured) != 1:
        raise ValueError(f"Expected exactly one backbone pass, got {len(captured)}")
    return captured[0], signature


def extract(args):
    """Replay identical structures and preprocessing for either base checkpoint."""
    import os
    import random

    import torch
    from dotenv import load_dotenv
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from zatom2.mlip_evaluation import _build_dataset
    from zatom2.run_inference import build_inference_engine

    from nucleus.inference_engines.checkpoint_registry import REGISTERED_CHECKPOINTS
    from nucleus.utils.batching import SparseCollate
    from nucleus.utils.inference import find_ema_wrapper

    load_dotenv(ROOT / ".env", override=False)
    atomwise = getattr(args, "atomwise", False)
    prefix = "element_embeddings" if atomwise else "subset_embeddings"
    archive = args.output / f"{prefix}_{args.variant}.npz"
    if archive.exists():
        raise FileExistsError(archive)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    checkpoint = resolve_checkpoint(args.checkpoint or CHECKPOINTS[args.variant])
    reference = resolve_checkpoint(args.preprocessing_checkpoint)
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != REGISTERED_CHECKPOINTS[CHECKPOINTS[args.variant]].sha256:
        raise ValueError(
            "Checkpoint does not match the registered base-model comparison"
        )
    reference_hash = sha256(reference)
    if reference_hash != REGISTERED_CHECKPOINTS[CHECKPOINTS["with_properties"]].sha256:
        raise ValueError("Common preprocessing must come from the base E/F checkpoint")
    reference_state = torch.load(reference, map_location="cpu", weights_only=False)
    common_cfg = reference_state["train_cfg"]
    del reference_state
    with initialize_config_dir(
        config_dir=str(ROOT / "models/zatom2/configs"), version_base="1.3"
    ):
        cfg = compose(
            config_name="mlip_evaluation",
            overrides=[
                f"ckpt_path={checkpoint}",
                f"out_dir={args.work / args.variant}",
                f"seed={args.seed}",
                "evaluation.num_workers=0",
            ],
        )
    engine = build_inference_engine(cfg)
    train_cfg = engine.initialize()
    trainer = engine.trainer
    ema = find_ema_wrapper(trainer.state["model"])
    if ema is None:
        raise RuntimeError("Expected EMA weights")
    model = ema.shadow.eval()
    enabled = model.diffusion_module.atomistic_property_prediction_enabled
    if enabled != (args.variant == "with_properties"):
        raise ValueError("Unexpected property-head configuration")
    records = json.loads(args.samples.read_text())
    pooled, compositions, signatures = [], [], []
    atom_embeddings, atom_numbers = [], []
    checked_domains = []
    roots = {}
    for domain in ("omol25", "omat24"):
        root = Path(str(cfg.evaluation.datasets[domain].path)).expanduser().resolve()
        roots[domain] = str(root)
        dataset = _build_dataset(common_cfg, domain, str(root), args.max_atoms)
        for record in (r for r in records if r["domain"] == domain):
            index = int(record["global_index"])
            shard, local = dataset._resolve_index(index)
            row = dataset._get_row_by_local_index(shard, local)
            if (
                str(Path(dataset.paths[shard]).relative_to(root)) != record["shard"]
                or int(row.id) != record["row_id"]
                or dataset._get_example_id(row, shard) != record["example_id"]
            ):
                raise ValueError("Validation manifest no longer matches source rows")
            atoms = row.toatoms()
            example = dataset[index]
            gt = example["ground_truth"]
            if not (
                int(gt["source_atom_count_B"].item())
                == int(gt["retained_atom_count_B"].item())
                == len(example["atom_array"])
                == len(atoms)
                == record["n_atoms"]
            ):
                raise ValueError("Embedding extraction must preserve every source atom")
            collated = trainer.fabric.to_device(SparseCollate()([example]))
            inputs = trainer._single_view_property_input(
                trainer._assemble_network_inputs(collated)
            )
            with torch.inference_mode(), trainer.fabric.autocast():
                embedding, signature = backbone_embedding(model, inputs)
                if enabled and domain not in checked_domains:
                    public = model.predict_properties(
                        inputs,
                        return_embeddings=True,
                        allow_missing_elemental_references=True,
                    )["Q_L"]
                    torch.testing.assert_close(
                        embedding, public.reshape_as(embedding), rtol=0, atol=0
                    )
                    checked_domains.append(domain)
            values = embedding.float().cpu().numpy().reshape(len(atoms), -1)
            if not np.isfinite(values).all():
                raise ValueError("Non-finite embeddings")
            pooled.append(values.mean(axis=0))
            if atomwise:
                from ase.data import atomic_numbers

                # Q_L follows the transformed AtomArray, not necessarily ASE order.
                numbers = np.array(
                    [atomic_numbers[e.title()] for e in example["atom_array"].element],
                    dtype=np.int16,
                )
                np.testing.assert_array_equal(np.sort(numbers), np.sort(atoms.numbers))
                atom_embeddings.append(values)
                atom_numbers.append(numbers)
            compositions.append(np.bincount(atoms.numbers, minlength=119) / len(atoms))
            signatures.append(signature)
            if len(pooled) % 64 == 0:
                print(f"{args.variant}: {len(pooled)}/{len(records)}", flush=True)
    if [r["domain"] for r in records] != sorted(
        (r["domain"] for r in records), reverse=True
    ):
        # The published manifest is OMol25 followed by OMat24.
        raise ValueError("Manifest must group OMol25 before OMat24")
    atom_data = (
        {
            "atom_embeddings": np.concatenate(atom_embeddings),
            "atomic_numbers": np.concatenate(atom_numbers),
            "structure_index": np.repeat(
                np.arange(len(records)), [r["n_atoms"] for r in records]
            ),
        }
        if atomwise
        else {}
    )
    np.savez_compressed(
        archive,
        embeddings=np.stack(pooled),
        composition=np.stack(compositions),
        input_sha256=np.array(signatures),
        domain=np.array([r["domain"] for r in records]),
        subset=np.array([r["subset"] for r in records]),
        n_atoms=np.array([r["n_atoms"] for r in records]),
        **atom_data,
    )
    provenance = {
        "checkpoint_alias": CHECKPOINTS[args.variant],
        "checkpoint_sha256": checkpoint_hash,
        "common_preprocessing_checkpoint_sha256": reference_hash,
        "sample_manifest_sha256": sha256(args.samples),
        "archive_sha256": sha256(archive),
        "extraction_script_sha256": sha256(__file__),
        "variant": args.variant,
        "weights": "EMA",
        "endpoint": 1,
        "recycles": 1,
        "embedding": (
            "per-atom backbone output Q_L in transformed AtomArray order, before task heads"
            if atomwise
            else "mean over atoms of backbone output Q_L, before task heads"
        ),
        "seed": args.seed,
        "max_atoms": args.max_atoms,
        "force_conditioning": False,
        "random_augmentation": False,
        "atom_types": "known",
        "property_heads_enabled": enabled,
        "public_api_exact_parity_domains": checked_domains,
        "backbone_parameters": sum(
            p.numel() for p in model.diffusion_module.backbone.parameters()
        ),
        "training_task_frequencies": {
            k: float(v.frequency)
            for k, v in train_cfg.datasets.global_transform_args.train_conditions.items()
        },
        "diffusion_config": OmegaConf.to_container(
            train_cfg.model.net.diffusion_module, resolve=True
        ),
        "precision": trainer.fabric._precision.precision,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "validation_roots": roots,
    }
    write_json(args.output / f"{prefix}_{args.variant}_provenance.json", provenance)


def neighborhood_purity(values, labels, k=10):
    """Fraction of same-label neighbors, excluding each point itself."""
    from sklearn.neighbors import NearestNeighbors

    indices = (
        NearestNeighbors(n_neighbors=k + 1)
        .fit(values)
        .kneighbors(values, return_distance=False)
    )
    neighbors = np.array([row[row != i][:k] for i, row in enumerate(indices)])
    return float(np.mean(labels[neighbors] == labels[:, None]))


def plot(args):
    """Plot checkpoint columns and dataset rows; fit each PCA independently."""
    import csv
    import importlib.metadata

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    variants = list(CHECKPOINTS)
    archives = {v: args.output / f"subset_embeddings_{v}.npz" for v in variants}
    provenance = {
        v: json.loads(
            (args.output / f"subset_embeddings_{v}_provenance.json").read_text()
        )
        for v in variants
    }
    for variant in variants:
        if sha256(archives[variant]) != provenance[variant]["archive_sha256"]:
            raise ValueError(f"Archive checksum mismatch: {variant}")
    for key in (
        "common_preprocessing_checkpoint_sha256",
        "sample_manifest_sha256",
        "backbone_parameters",
        "precision",
    ):
        if provenance[variants[0]][key] != provenance[variants[1]][key]:
            raise ValueError(f"Unmatched comparison setting: {key}")
    data = {v: np.load(path, allow_pickle=False) for v, path in archives.items()}
    left, right = (data[v] for v in variants)
    for key in ("domain", "subset", "n_atoms", "composition", "input_sha256"):
        np.testing.assert_array_equal(
            left[key], right[key], err_msg=f"Mismatched {key}"
        )
    for values in data.values():
        if values["embeddings"].shape != (len(left["domain"]), 128):
            raise ValueError("Expected matching 128-dimensional base-model embeddings")
        if not np.isfinite(values["embeddings"]).all():
            raise ValueError("Non-finite embedding archive")
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.4, 9.0))
    grid = fig.add_gridspec(
        6, 2, height_ratios=[1, 0.15, 1, 0.23, 1, 0.40], hspace=0.45, wspace=0.30
    )
    results, coordinates = {}, []
    for row, name in enumerate(("joint", "omol25", "omat24")):
        mask = (
            np.ones(len(left["domain"]), bool)
            if name == "joint"
            else left["domain"] == name
        )
        labels = left["domain"][mask] if name == "joint" else left["subset"][mask]
        groups = sorted(set(labels))
        counts = np.array([np.sum(labels == group) for group in groups])
        chance = float(
            np.sum(counts * (counts - 1)) / (len(labels) * (len(labels) - 1))
        )
        results[name] = {
            "n": int(mask.sum()),
            "counts": dict(zip(groups, counts.tolist(), strict=True)),
            "random_label_expectation": chance,
            "10nn_purity_element_fractions": neighborhood_purity(
                left["composition"][mask], labels
            ),
        }
        for col, variant in enumerate(variants):
            ax = fig.add_subplot(grid[2 * row, col])
            selected = data[variant]["embeddings"][mask].astype(np.float64)
            pca = PCA(n_components=2, svd_solver="full")
            xy = pca.fit_transform(selected)
            palette = plt.get_cmap("tab10" if len(groups) <= 10 else "tab20")
            for i, group in enumerate(groups):
                keep = labels == group
                label = group.replace("aimd-from-PBE-", "AIMD ").replace(
                    "-subsampled", " (sub)"
                )
                label = {"omol25": "OMol25", "omat24": "OMat24"}.get(label, label)
                ax.scatter(
                    xy[keep, 0],
                    xy[keep, 1],
                    s=6,
                    alpha=0.65,
                    color=palette(i),
                    edgecolors="none",
                    label=label,
                )
            title = {"joint": "Joint", "omol25": "OMol25", "omat24": "OMat24"}[name]
            ax.set_title(
                f"{chr(97+row*2+col)}  {title} ($n={mask.sum():,}$)",
                loc="left",
                fontsize=9,
                weight="bold",
            )
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=8)
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            results[name][variant] = {
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "10nn_purity_embedding": neighborhood_purity(selected, labels),
            }
            for index, point, label in zip(
                np.flatnonzero(mask), xy, labels, strict=True
            ):
                coordinates.append(
                    {
                        "sample_index": int(index),
                        "domain_panel": name,
                        "model": variant,
                        "label": str(label),
                        "pc1": point[0],
                        "pc2": point[1],
                    }
                )
        legend = fig.add_subplot(grid[2 * row + 1, :])
        legend.axis("off")
        legend.legend(
            *ax.get_legend_handles_labels(),
            loc="center",
            ncol=2 if name == "joint" else 3,
            frameon=False,
            fontsize=7,
            markerscale=1.5,
            columnspacing=1.5,
        )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.93, bottom=0.015)
    for x, v in zip((0.29, 0.79), variants, strict=True):
        fig.text(x, 0.975, TITLES[v], ha="center", va="top", fontsize=10, weight="bold")
    for suffix in ("pdf", "png"):
        fig.savefig(
            args.output / f"subset_embeddings_pca.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)
    results["archive_sha256"] = {v: sha256(p) for v, p in archives.items()}
    results["analysis_script_sha256"] = sha256(__file__)
    results["matched_input_sha256_verified"] = True
    results["software"] = {
        n: importlib.metadata.version(n)
        for n in ("numpy", "scikit-learn", "matplotlib")
    }
    write_json(args.output / "subset_embeddings_results.json", results)
    with (args.output / "subset_embeddings_pca.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(coordinates[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(coordinates)
    print(json.dumps(results, indent=2))


def plot_main(args):
    """Export the saved joint PCA coordinates as a compact main-text figure."""
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (args.output / "subset_embeddings_pca.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["domain_panel"] == "joint"]
    results = json.loads((args.output / "subset_embeddings_results.json").read_text())
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 1.65))
    reference = None
    for col, (variant, ax) in enumerate(zip(CHECKPOINTS, axes, strict=True)):
        selected = [r for r in rows if r["model"] == variant]
        samples = [(int(r["sample_index"]), r["label"]) for r in selected]
        if len(samples) != results["joint"]["n"] or len(set(samples)) != len(samples):
            raise ValueError(f"Incomplete or duplicate joint coordinates: {variant}")
        if reference is not None and samples != reference:
            raise ValueError("Joint panels must contain identical ordered examples")
        reference = samples
        xy = np.array([[float(r["pc1"]), float(r["pc2"])] for r in selected])
        labels = np.array([r["label"] for r in selected])
        for i, label in enumerate(sorted(set(labels))):
            keep = labels == label
            ax.scatter(
                xy[keep, 0],
                xy[keep, 1],
                s=6,
                alpha=0.65,
                color=plt.get_cmap("tab10")(i),
                edgecolors="none",
                label={"omol25": "OMol25", "omat24": "OMat24"}[label],
            )
        variance = results["joint"][variant]["explained_variance_ratio"]
        ax.set_title(f"({chr(97 + col)}) {TITLES[variant]}", fontsize=10, loc="left")
        ax.set_xlabel(f"PC1 ({variance[0]:.1%})", labelpad=1)
        ax.set_ylabel(f"PC2 ({variance[1]:.1%})", labelpad=1)
        ax.tick_params(labelsize=8, pad=1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(
        loc="upper left",
        frameon=False,
        fontsize=8,
        markerscale=1.5,
        handletextpad=0.3,
        borderaxespad=0.2,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.83, bottom=0.25, wspace=0.30)
    for suffix in ("pdf", "png"):
        fig.savefig(
            args.output / f"subset_embeddings_joint_pca.{suffix}",
            dpi=220,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("extract", "plot", "plot-main"))
    parser.add_argument("--output", type=Path, default=DEST)
    parser.add_argument(
        "--work", type=Path, default=ROOT / "logs/subset_embeddings_comparison"
    )
    parser.add_argument(
        "--samples", type=Path, default=DEST / "subset_embeddings_samples.json"
    )
    parser.add_argument(
        "--variant", choices=tuple(CHECKPOINTS), default="with_properties"
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--preprocessing-checkpoint", default=CHECKPOINTS["with_properties"]
    )
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--max-atoms", type=int, default=512)
    parser.add_argument(
        "--atomwise", action="store_true", help="Also retain unpooled atom embeddings."
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    {"extract": extract, "plot": plot, "plot-main": plot_main}[args.mode](args)


if __name__ == "__main__":
    main()
