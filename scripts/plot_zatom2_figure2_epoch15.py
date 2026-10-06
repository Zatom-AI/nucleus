#!/usr/bin/env python3
"""Reproduce the four-panel scaling figure directly from saved epoch-15 outputs.

Uses the intersection of saved target IDs across conditions for each task;
--require-complete instead requires every fixture target. No interpolation,
training-history scalars, Zatom inference, or external oracle is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path

from omegaconf import OmegaConf
from recompute_zatom2_figure2_metrics import sha256, tensor_sha256
from rescore_zatom2_figure2_omol25 import MODEL_NAME, MODEL_REVISION

REPO = Path(__file__).resolve().parents[1]
EPOCH = 15
RUN_PATHS = {name: name for name in ("base_500k", "large_500k", "base_full", "large_full")}

TASKS = {
    "omol25_generation": ("omol_afd", 1000),
    "omat24_structure_prediction": ("omat_match", 500),
    "omol25_mlip": ("omol_force_mae", 500),
    "omat24_mlip": ("omat_force_mae", 500),
}
MATCHER = {"ltol": 0.3, "stol": 0.5, "angle_tol": 10.0, "scale": True}


def digest(value: object) -> str:
    """Hash a JSON value using stable ordering and strict finite numbers."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Publish an output only after it has been completely serialized."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def load_fixture(path: Path) -> dict:
    """Apply the inference JSON's global overrides and relative input paths."""
    data = json.loads(path.read_text())
    global_args = data.pop("global_args", {})
    specifications = {}
    for target, value in data.items():
        specification = value | global_args
        if specification.get("input"):
            specification["input"] = str(
                (path.parent / specification["input"]).resolve()
            )
        specifications[target] = specification
    return specifications


def saved_pairs(directory: Path, expected: set[str]) -> dict:
    """Reject orphan files, unexpected targets, and multiple candidates."""
    files = {}
    for path in sorted(directory.iterdir()):
        if not path.name.endswith((".cif.gz", ".json")):
            continue
        suffix = ".cif.gz" if path.name.endswith(".cif.gz") else ".json"
        stem = path.name.removesuffix(suffix)
        target, model = stem.rsplit("_model_", 1)
        if model != "0" or target not in expected:
            raise ValueError(f"Unexpected target or candidate: {path}")
        files.setdefault(target, {})[suffix] = file_record(path)
    for target, pair in files.items():
        if set(pair) != {".cif.gz", ".json"}:
            raise ValueError(f"Incomplete CIF/JSON pair: {directory}/{target}")
    return files


def common_targets(pools: dict[str, dict], expected: set[str], strict: bool) -> list:
    """Choose identical targets before looking at any model's scores."""
    if strict and any(set(pool) != expected for pool in pools.values()):
        raise ValueError("Incomplete target coverage; see inventory.json")
    common = sorted(set.intersection(expected, *(set(pool) for pool in pools.values())))
    if len(common) < 2:
        raise ValueError("Fewer than two common targets")
    return common


def inventory(train_dir: Path) -> tuple[dict, dict]:
    """Pin the backfill's exact jobs and fingerprint inputs and saved outputs."""
    configs = {}
    runs = {}
    for condition, relative in RUN_PATHS.items():
        run = train_dir / relative
        config_path = run / ".hydra/config.yaml"
        cfg = OmegaConf.load(config_path)
        cfg.paths.root_dir = str(REPO)
        configs[condition] = cfg
        runs[condition] = {
            "run": str(run),
            "config": file_record(config_path),
            "sample_origin": "checkpoint backfill (generation/structure)"
            if condition.endswith("500k")
            else "original training validation",
        }
    datasets, specifications = {}, {}
    for task, (_, expected_count) in TASKS.items():
        pools, fixture_records = {}, {}
        canonical = None
        for condition, relative in RUN_PATHS.items():
            # The backfill deliberately uses its paired full-run validation
            # fixture, because the old 500k generation fixture was different.
            fixture_condition = condition.replace("500k", "full")
            cfg = configs[fixture_condition]
            dataset = cfg.datasets.val[task].dataset
            if dataset.get("subset_to_keys") or int(dataset.eval_every_n) != 1:
                raise ValueError(f"Unsupported validation subset: {condition}/{task}")
            path = Path(str(dataset.data))
            specs = load_fixture(path)
            if len(specs) != expected_count or (canonical and specs != canonical):
                raise ValueError(f"Validation fixtures disagree for {task}")
            canonical = specs
            fixture_records[condition] = file_record(path)
            if task.endswith("mlip"):
                actual = configs[condition].datasets.val[task].dataset
                if load_fixture(Path(str(actual.data))) != specs:
                    raise ValueError(f"MLIP fixture differs: {condition}")
                transform = actual.transform
                if bool(transform.randomly_augment) or bool(transform.remove_hydrogens):
                    raise ValueError("Saved MLIP requires unaugmented, all-atom order")
            directory = (
                train_dir
                / relative
                / f"val_structures/predictions/epoch_{EPOCH}"
                / task
            )
            pools[condition] = saved_pairs(directory, set(specs))
        selected = common_targets(pools, set(canonical), strict=False)
        references = {
            target: file_record(Path(spec["input"]))
            for target, spec in canonical.items()
            if spec.get("input")
        }
        datasets[task] = {
            "expected": expected_count,
            "selected_targets": selected,
            "fixtures": fixture_records,
            "references": references,
            "pools": pools,
            "missing_targets": {
                condition: sorted(set(canonical) - set(pool))
                for condition, pool in pools.items()
            },
            "excluded_from_comparison": sorted(set(canonical) - set(selected)),
        }
        specifications[task] = canonical
        print(f"{task}: {len(selected)}/{expected_count} common targets", flush=True)
    return {"epoch": EPOCH, "runs": runs, "datasets": datasets}, specifications


def score_mlip(pool: dict, specs: dict, targets: list[str]) -> dict:
    """Pool Cartesian errors over atoms, and energy errors over systems."""
    import numpy as np
    from compare_atomistic_property_predictions import compare_prediction_metadata
    from recompute_zatom2_figure2_metrics import load_prediction

    rows = []
    for target in targets:
        metadata = json.loads(Path(pool[target][".json"]["path"]).read_text())
        result = compare_prediction_metadata(
            metadata | {"specification": specs[target]}
        )
        if result["n_atoms"] <= 0:
            raise ValueError(f"Empty force prediction: {target}")
        prediction = load_prediction(Path(pool[target][".cif.gz"]["path"]))
        reference = load_prediction(Path(specs[target]["input"]))
        # Molecular validation preserves Cartesian axes up to translation.
        # Periodic CIFs can use different Cartesian lattice orientations, so
        # their authoritative fractional annotations establish atom order.
        # 1e-4 Å accommodates float32 centering and CIF serialization only.
        periodic = hasattr(reference, "material_frac_coords")
        if periodic and hasattr(prediction, "material_frac_coords"):
            delta = prediction.material_frac_coords - reference.material_frac_coords
            delta -= delta[0]
            delta -= np.rint(delta)
            coordinates_match = np.allclose(
                prediction.box, reference.box, rtol=0, atol=1e-5
            ) and np.allclose(delta @ reference.box, 0, rtol=0, atol=1e-4)
        else:
            coordinates_match = not periodic and np.allclose(
                prediction.coord - prediction.coord.mean(axis=0),
                reference.coord - reference.coord.mean(axis=0),
                rtol=0,
                atol=1e-4,
            )
        if (
            len(prediction) != result["n_atoms"]
            or not np.array_equal(prediction.element, reference.element)
            or not coordinates_match
        ):
            raise ValueError(f"MLIP atom order/coordinate frame differs: {target}")
        rows.append({"target_id": target, **result})
    components = sum(3 * row["n_atoms"] for row in rows)
    error_sum = math.fsum(
        row["force_component_mae_eV_per_angstrom"] * 3 * row["n_atoms"] for row in rows
    )
    return {
        "value": error_sum / components,
        "num_force_components": components,
        "force_component_absolute_error_sum": error_sum,
        "energy_mae_eV": math.fsum(row["energy_absolute_error_eV"] for row in rows)
        / len(rows),
        "energy_per_atom_mae_eV": math.fsum(
            row["energy_per_atom_absolute_error_eV"] for row in rows
        )
        / len(rows),
        "per_target": rows,
    }


def realizable_cell(lengths, angles) -> bool:
    """Check that declared lengths/angles define a nondegenerate 3-D cell.

    The shared lattice converter clamps a negative squared height to 1e-12.
    Reject impossible angle triples before that artificial cell reaches Niggli
    reduction. No physical volume/density cutoff is imposed on valid cells.
    """
    import numpy as np

    lengths, angles = np.asarray(lengths), np.asarray(angles)
    if (
        lengths.shape != (3,)
        or angles.shape != (3,)
        or not np.isfinite(lengths).all()
        or not np.isfinite(angles).all()
        or np.any(lengths <= 0)
        or np.any(angles <= 0)
        or np.any(angles >= 180)
        or angles.sum() >= 360
        or 2 * angles.max() >= angles.sum()
    ):
        return False
    a, b, c = np.cos(np.deg2rad(angles))
    return bool(1 + 2 * a * b * c - a * a - b * b - c * c > 0)


def score_structure(pool: dict, specs: dict, targets: list[str]) -> dict:
    """Use the repository's top-1 periodic matcher with the manuscript settings."""
    from recompute_zatom2_figure2_metrics import load_prediction
    from zevals.materials.structure_prediction import (
        compute_material_structure_prediction_metrics,
    )

    from nucleus.utils.material_fingerprints import atom_array_to_structure
    from nucleus.utils.materials import get_material_lattice

    generated, references = {}, {}
    invalid_cells = []
    for target in targets:
        atoms = load_prediction(Path(pool[target][".cif.gz"]["path"]))
        lattice = get_material_lattice(atoms)
        if lattice is None or not realizable_cell(lattice[0], lattice[1]):
            invalid_cells.append(target)
            generated[target] = [None]
        else:
            generated[target] = [atom_array_to_structure(atoms)]
        reference = load_prediction(Path(specs[target]["input"]))
        lattice = get_material_lattice(reference)
        if lattice is None or not realizable_cell(lattice[0], lattice[1]):
            raise ValueError(f"Invalid reference cell: {target}")
        references[target] = [atom_array_to_structure(reference)]
    metrics = compute_material_structure_prediction_metrics(
        generated, references, **MATCHER, top_k=[1]
    )
    return {
        "value": 100 * metrics["material_match_rate_top1"],
        "metrics": metrics,
        "invalid_cell_targets": invalid_cells,
    }


def reference_identity(path: Path) -> dict:
    """Require the audited OMol25 encoder/reference pair, without rebuilding it."""
    metadata_path = path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text())
    signature = metadata["signature"]
    if (signature["model_name"], signature["model_revision"]) != (
        MODEL_NAME,
        MODEL_REVISION,
    ):
        raise ValueError("Reference cache uses the wrong AFD encoder")
    if (
        metadata["shape"] != [188762, 256]
        or signature["batch_size"] != 256
        or signature["matmul_precision"] != "medium"
        or signature["deterministic_algorithms"] is not True
        or signature["cublas_workspace_config"] != ":4096:8"
    ):
        raise ValueError("Reference cache does not match the deterministic protocol")
    if sha256(path) != metadata["cache_sha256"]:
        raise ValueError("AFD reference cache hash mismatch")
    return {
        "cache": file_record(path),
        "metadata": file_record(metadata_path),
        "details": metadata,
    }


def make_afd_scorer(reference: dict, device: str):
    """Load only the locally cached, pinned encoder and verified reference tensor."""
    import torch
    from huggingface_hub import try_to_load_from_cache
    from zevals.atomistic.afd.features import _default_model_cache_dir

    from nucleus.callbacks.metrics_logging import _CachedReferenceAFDScorer

    checkpoint = try_to_load_from_cache(
        MODEL_NAME, "last.ckpt", cache_dir=str(_default_model_cache_dir())
    )
    signature = reference["details"]["signature"]
    if (
        not isinstance(checkpoint, str)
        or sha256(Path(checkpoint)) != signature["model_checkpoint_sha256"]
    ):
        raise ValueError("Locally cached AFD encoder does not match the reference pin")
    features = torch.load(
        reference["cache"]["path"], map_location="cpu", weights_only=True
    )
    if tensor_sha256(features) != reference["details"]["features_sha256"]:
        raise ValueError("Reference tensor hash mismatch")
    return _CachedReferenceAFDScorer(
        features,
        data_modality="molecules",
        model_name=MODEL_NAME,
        batch_size=256,
        device=device,
    )


def score_afd(pool: dict, targets: list[str], scorer, feature_path: Path) -> dict:
    """Embed every saved molecule using CIF species and score matched-N AFD once."""
    import torch
    from recompute_zatom2_figure2_metrics import load_prediction
    from zevals.atomistic.afd.score import afd_score

    from nucleus.utils.atom_array import _atom_array_to_afd_record

    records = []
    for target in targets:
        record = _atom_array_to_afd_record(
            load_prediction(Path(pool[target][".cif.gz"]["path"]))
        )
        if record is None:
            raise ValueError(f"Cannot embed generation target: {target}")
        records.append(record)
    features = scorer.embeddings(records, show_progress=True)
    if len(features) != len(targets) or not torch.isfinite(features).all():
        raise ValueError("Incomplete or nonfinite generated features")
    reference = scorer.sample_ref(len(targets), seed=0)
    value = float(afd_score(reference, features))
    torch.save({"target_ids": targets, "features": features}, feature_path)
    return {
        "value": value,
        "n_reference": len(reference),
        "reference_seed": 0,
        "features": file_record(feature_path),
        "selected_reference_features_sha256": tensor_sha256(reference),
    }


def plot(rows: list[dict], output: Path) -> None:
    """Render a compact figure with one shared legend and honest denominators."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panels = [
        ("omol_afd", "OMol25 generation", "AFD ↓"),
        ("omat_match", "OMat24 structure prediction", "Top-1 match (%) ↑"),
        ("omol_force_mae", "OMol25 MLIP", "Force MAE (eV/Å) ↓"),
        ("omat_force_mae", "OMat24 MLIP", "Force MAE (eV/Å) ↓"),
    ]
    values = {(row["condition"], row["metric"]): row for row in rows}
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "svg.hashsalt": "zatom2-epoch15",
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
        for letter, ax, (metric, title, ylabel) in zip(
            "abcd", axes.flat, panels, strict=True
        ):
            for support, color, marker in (
                ("500k", "#c77723", "o"),
                ("full", "#286a9b", "s"),
            ):
                y = [
                    values[(f"{size}_{support}", metric)]["value"]
                    for size in ("base", "large")
                ]
                ax.plot(
                    [0, 1], y, color=color, marker=marker, linewidth=1.6, markersize=6
                )
            count = values[("base_full", metric)]["n_targets"]
            ax.set_title(f"({letter}) {title}", loc="left", fontsize=11)
            ax.text(
                0.97,
                0.94,
                f"n = {count:,}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color="0.4",
            )
            ax.set_xticks([0, 1], ["Base\n142.4M", "Large\n276.0M"])
            ax.set_xlim(-0.25, 1.25)
            ax.margins(y=0.3)
            ax.set_ylabel(ylabel)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
        handles = [
            Line2D([], [], color=color, marker=marker, label=label)
            for color, marker, label in [
                ("#c77723", "o", "500k / domain"),
                ("#286a9b", "s", "Full pools"),
            ]
        ]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
        fig.tight_layout(rect=(0, 0.035, 1, 0.94), h_pad=1.5, w_pad=1.5)
        fig.text(
            0.5,
            0.015,
            "Epoch 15 · 16 completed epochs · common targets per panel",
            ha="center",
            fontsize=9,
        )
        for suffix in ("pdf", "png", "svg"):
            metadata = (
                {"CreationDate": None, "ModDate": None}
                if suffix == "pdf"
                else ({"Date": None} if suffix == "svg" else {})
            )
            fig.savefig(
                output / f"figure2_epoch15.{suffix}", dpi=300, metadata=metadata
            )
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=REPO / "logs/train")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO / "logs/analysis/zatom2_figure2_epoch15"
    )
    parser.add_argument(
        "--reference-features",
        type=Path,
        default=REPO
        / "data/omol25/afd_feature_cache/Ty-Perez__ct-scd-omol25_reference_features.pt",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inventory-only", action="store_true")
    mode.add_argument(
        "--plot-only",
        action="store_true",
        help="Verify source hashes and plot cached scores without loading models",
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    os.environ.update(
        DEBUG="false",
        WANDB_MODE="disabled",
        HF_HUB_OFFLINE="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    os.environ.setdefault("PROJECT_ROOT", str(REPO))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs, specs = inventory(args.train_dir.resolve())
    write_json(output / "inventory.json", inputs)
    if args.require_complete:
        for dataset in inputs["datasets"].values():
            if dataset["excluded_from_comparison"]:
                raise ValueError("Incomplete target coverage; see inventory.json")
    if args.inventory_only:
        return
    reference = reference_identity(args.reference_features.resolve())
    source_paths = [
        Path(__file__),
        REPO / "uv.lock",
        REPO / "scripts/compare_atomistic_property_predictions.py",
        REPO / "scripts/recompute_zatom2_figure2_metrics.py",
        REPO / "src/nucleus/utils/atom_array.py",
        REPO / "src/nucleus/callbacks/metrics_logging.py",
        REPO / "src/nucleus/utils/material_fingerprints.py",
        REPO / "src/nucleus/utils/materials.py",
        REPO / "src/nucleus/inference_input_parsing.py",
    ]
    source_paths += sorted(
        (REPO / "models/zevals/src/zevals/atomistic/afd").rglob("*.py")
    )
    source_paths += [
        REPO / "models/zevals/src/zevals/materials/structure_prediction.py"
    ]
    protocol = {
        "sources": [file_record(path) for path in source_paths],
        "reference": reference,
        "matcher": MATCHER,
        "cell_validation": "positive lengths; realizable nondegenerate cell angles",
        "afd_model": MODEL_NAME,
        "afd_revision": MODEL_REVISION,
        "afd_reference_seed": 0,
        "afd_batch_size": 256,
        "selection": "sorted intersection per task",
        "epoch": EPOCH,
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "numpy",
                "scipy",
                "pymatgen",
                "biotite",
                "torch-geometric",
                "matplotlib",
            )
        },
    }
    runtime = None
    if not args.plot_only:
        import torch

        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("medium")
        torch.use_deterministic_algorithms(True)
        runtime = {
            "device": args.device,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "cuda": torch.version.cuda,
            "matmul_precision": "medium",
            "deterministic_algorithms": True,
        }
    rows, scorer, results = [], None, []
    for task, (metric, _) in TASKS.items():
        dataset = inputs["datasets"][task]
        targets = dataset["selected_targets"]
        for condition in RUN_PATHS:
            identity = digest(
                {
                    "protocol": protocol,
                    "inputs": dataset,
                    "run": inputs["runs"][condition],
                    "condition": condition,
                }
            )
            result_path = output / f"{condition}_{task}.json"
            cached = json.loads(result_path.read_text()) if result_path.exists() else {}
            reusable = cached.get("identity") == identity and (
                args.plot_only or cached.get("runtime") == runtime
            )
            if not reusable:
                if args.plot_only:
                    raise ValueError(
                        f"Missing/stale scores; rerun scoring: {result_path}"
                    )
                print(f"Scoring {condition}/{task}", flush=True)
                pool = dataset["pools"][condition]
                if task.endswith("generation"):
                    if scorer is None:
                        scorer = make_afd_scorer(reference, args.device)
                    result = score_afd(
                        pool, targets, scorer, output / f"{condition}_features.pt"
                    )
                elif task.endswith("mlip"):
                    result = score_mlip(pool, specs[task], targets)
                else:
                    result = score_structure(pool, specs[task], targets)
                if not math.isfinite(result["value"]) or result["value"] < 0:
                    raise ValueError(f"Invalid metric: {condition}/{task}")
                cached = {
                    "identity": identity,
                    "runtime": runtime,
                    "condition": condition,
                    "task": task,
                    "metric": metric,
                    "epoch": EPOCH,
                    "n_targets": len(targets),
                    **result,
                }
                write_json(result_path, cached)
            if (
                "features" in cached
                and file_record(Path(cached["features"]["path"])) != cached["features"]
            ):
                raise ValueError(f"Generated feature cache changed: {condition}")
            rows.append(
                {
                    key: cached.get(key, "")
                    for key in (
                        "condition",
                        "epoch",
                        "metric",
                        "value",
                        "n_targets",
                        "num_force_components",
                        "energy_mae_eV",
                        "energy_per_atom_mae_eV",
                    )
                }
            )
            results.append(file_record(result_path))
    with (output / "figure2_epoch15.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, output)
    write_json(
        output / "provenance.json",
        {
            "protocol": protocol,
            "inventory": file_record(output / "inventory.json"),
            "results": results,
            "artifacts": [
                file_record(output / f"figure2_epoch15.{suffix}")
                for suffix in ("csv", "pdf", "png", "svg")
            ],
        },
    )
    print(f"Figure and source data: {output}", flush=True)


if __name__ == "__main__":
    main()
