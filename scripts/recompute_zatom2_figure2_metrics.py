#!/usr/bin/env python3
"""Recompute Figure 2 generation/structure metrics from saved validation CIFs.

This performs no Zatom inference and no UMA scoring. It reuses the original
validation evaluators, fixed AFD reference sampling, and saved Hydra settings.
Missing predictions remain missing. AFD uses saved predicted species, correcting
the original stale atomic_number annotation. Historical AFD is therefore not a
valid species-aware control; isolated all-carbon replay establishes the bug.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
from recover_zatom2_figure2_validation import EXPERIMENTS

DATASETS = {
    "omol25_generation": "omol_afd",
    "omat24_generation": "omat_afd",
    "omol25_structure_prediction": "omol_rmsd",
    "omat24_structure_prediction": "omat_match",
}
AFD_PROTOCOL = "species_consistent_afd_v1"
STRUCTURE_PROTOCOL = "original_conditioned_structure_v1"


def validate_legacy_afd_control(domain: str, directory: Path) -> dict:
    """Require the isolated historical-bug replay before accepting corrected AFD."""
    path = directory / f"base_full_epoch15_{domain}_stale_carbon_medium.json"
    control = json.loads(path.read_text())
    difference = abs(control["value"] - control["historical_logged_value"])
    # AFD involves GPU scatter reductions/TF32 and matrix eigendecomposition.
    # This absolute bound resolves the replay residual (3.3/5.3e-6), while the
    # species correction changes scores by 0.004/0.168, orders of magnitude more.
    tolerance = 1e-5
    passed = (
        control["status"] == "diagnostic_stale_atomic_number"
        and control["n_samples"] == 1000
        and difference <= tolerance
    )
    if not passed:
        raise ValueError(f"Historical stale-carbon replay failed: {path}")
    return {
        "passed": True,
        "method": "historical_stale_carbon_replay_identifies_species_bug",
        "source_file": str(path.resolve()),
        "source_sha256": sha256(path),
        "historical_value": control["historical_logged_value"],
        "replay_value": control["value"],
        "absolute_difference": difference,
        "absolute_tolerance": tolerance,
        "notes": "Control verifies the historical all-carbon bug; corrected species-aware AFD must differ from historical scores. CIF coordinates preserve float32 values; remaining replay residual includes GPU reduction/batching numerics.",
    }


def validate_structure_control(condition, epoch, metric, value, output: Path):
    """Check the original complete epoch, or require its successful replay."""
    original = {"omol_rmsd": 1.7375221674016377, "omat_match": 0.072}
    dataset = next(name for name, key in DATASETS.items() if key == metric)
    if condition == "base_full" and epoch == 15:
        tolerance = 1e-6 if metric == "omol_rmsd" else 1e-12
        difference = abs(value - original[metric])
        return {
            "passed": difference <= tolerance,
            "method": "complete_base_full_epoch15_saved_cif_roundtrip",
            "historical_value": original[metric],
            "recomputed_value": value,
            "absolute_difference": difference,
            "absolute_tolerance": tolerance,
            "notes": "CIF saves float32 Cartesian positions. RMSD allows 1e-6 angstrom for coordinate/alignment roundoff; categorical material match count must agree exactly.",
        }
    path = output / f"base_full_epoch15_{dataset}.json"
    if not path.is_file():
        return {"passed": False, "notes": f"Required control absent: {path}"}
    result = json.loads(path.read_text())
    passed = result.get("control_validation", {}).get("passed") is True
    return {
        "passed": passed,
        "method": "validated_original_structure_protocol",
        "source_file": str(path),
        "source_sha256": sha256(path),
    }


def sha256(path: Path) -> str:
    """Hash a source artifact without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value) -> str:
    """Hash a JSON-serializable manifest or configuration."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    """Atomically save one dataset's result or manifest."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_config(path: Path, repo: Path):
    """Resolve relevant saved configuration nodes without Hydra runtime state."""
    config = OmegaConf.load(path)
    config.paths.root_dir = str(repo)
    return config


def discover(train_dir: Path, condition: str, epoch: int, dataset: str, repo: Path):
    """Union complementary job artifacts, rejecting conflicting duplicate CIFs."""
    from zatom2.generation_evaluation import _load_input_specifications

    files, configurations = {}, []
    signature = None
    specifications = None
    selected_config = None
    for job in sorted((train_dir / EXPERIMENTS[condition]).iterdir()):
        directory = job / "val_structures" / "predictions" / f"epoch_{epoch}" / dataset
        if not directory.is_dir():
            continue
        config_path = job / ".hydra" / "config.yaml"
        config = load_config(config_path, repo)
        dataset_config = config.datasets.val[dataset]
        fixture = Path(str(dataset_config.dataset.data))
        current_specifications = _load_input_specifications(fixture)
        if dataset_config.get("key_to_balance"):
            raise ValueError(
                "Balanced validation samplers need an explicit order manifest"
            )
        if dataset_config.dataset.get("subset_to_keys"):
            raise ValueError(
                "Subset validation fixtures need an explicit order manifest"
            )
        if int(dataset_config.dataset.get("eval_every_n", 1)) != 1:
            raise ValueError(
                "Expected eval_every_n=1 for the original Figure 2 fixtures"
            )
        callback_key = (
            "log_molecule_afd_validation_metrics_callback"
            if dataset == "omol25_generation"
            else "log_materials_afd_validation_metrics_callback"
            if dataset == "omat24_generation"
            else "log_grouped_zevals_structure_prediction_metrics_callback"
        )
        relevant = {
            "fixture_sha256": sha256(fixture),
            "callback": OmegaConf.to_container(
                config.callbacks[callback_key], resolve=True
            ),
            "bond_sidecar_path": dataset_config.dataset.get("bond_sidecar_path"),
            "world_size": int(config.trainer.num_nodes)
            * int(config.trainer.devices_per_node),
        }
        if signature is not None and signature != relevant:
            raise ValueError(
                f"Inconsistent evaluation settings across {condition} epoch {epoch}"
            )
        signature, selected_config = relevant, config
        specifications = current_specifications
        configurations.append({"path": str(config_path), "sha256": sha256(config_path)})
        for path in sorted(directory.glob("*_model_*.cif.gz")):
            target_id, model_id = path.name.removesuffix(".cif.gz").rsplit("_model_", 1)
            if model_id != "0":
                raise ValueError(f"Expected one candidate per target; found {path}")
            if target_id not in specifications:
                raise ValueError(
                    f"Prediction absent from validation fixture: {target_id}"
                )
            digest = sha256(path)
            if target_id in files:
                if files[target_id]["sha256"] != digest:
                    raise ValueError(
                        f"Conflicting duplicate prediction for {target_id}"
                    )
                files[target_id]["duplicate_paths"].append(str(path))
                continue
            files[target_id] = {
                "target_id": target_id,
                "path": str(path),
                "sha256": digest,
                "duplicate_paths": [],
            }
    if not files:
        return None
    # DistributedSampler(shuffle=False), followed by rank-ordered gather_object.
    # Padding targets are excluded by the training callback and are not repeated.
    ids = list(specifications)
    world_size = signature["world_size"]
    ordered = [target for rank in range(world_size) for target in ids[rank::world_size]]
    manifest = [files[target] for target in ordered if target in files]
    return {
        "manifest": manifest,
        "specifications": specifications,
        "config": selected_config,
        "settings": signature,
        "configurations": configurations,
        "missing_targets": [target for target in ids if target not in files],
        "expected_targets": len(ids),
        "ordering": "fixture insertion order; rank-major stride over original world size; no padding",
    }


def load_prediction(path: Path):
    """Read saved atoms verbatim; do not perceive bonds or sanitize generation."""
    import numpy as np
    from atomworks.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER
    from atomworks.io.transforms.atom_array import add_atomic_number_annotation
    from atomworks.io.utils.io_utils import get_structure, read_any

    from nucleus.inference_input_parsing import _add_material_annotations_from_cif

    atom_array = get_structure(read_any(path), model=1, include_bonds=False)
    atom_array = add_atomic_number_annotation(atom_array)
    expected = np.array(
        [
            ELEMENT_NAME_TO_ATOMIC_NUMBER[str(element).upper()]
            for element in atom_array.element
        ]
    )
    if not np.array_equal(atom_array.atomic_number, expected) or np.any(expected <= 0):
        raise ValueError(f"Saved CIF species/atomic numbers do not agree: {path}")
    return _add_material_annotations_from_cif(path, atom_array)


def tensor_sha256(value) -> str:
    """Hash exact tensor values for reproducible feature-cache provenance."""
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(str((array.dtype, array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def afd_scorer(settings: dict, domain: str, device: str, batch_size: int | None):
    """Construct the original callback with only the scoring device overridden."""
    from nucleus.callbacks.metrics_logging import (
        LogMaterialsAFDValidationMetricsCallback,
        LogMoleculeAFDValidationMetricsCallback,
    )

    kwargs = dict(settings["callback"])
    kwargs.pop("_target_")
    kwargs.update(afd_device=device, show_progress=True)
    if batch_size is not None:
        kwargs["afd_batch_size"] = batch_size
    callback_type = (
        LogMoleculeAFDValidationMetricsCallback
        if domain == "omol25"
        else LogMaterialsAFDValidationMetricsCallback
    )
    callback = callback_type(**kwargs)
    reference_path = callback._reference_feature_cache_path()
    if domain == "omol25" and (reference_path is None or not reference_path.is_file()):
        raise FileNotFoundError(
            "Original OMol AFD reference feature cache is required; refusing to regenerate it"
        )
    scorer = callback._get_afd_scorer()
    reference_provenance = {
        "reference_features_sha256": tensor_sha256(scorer.ref_features),
        "reference_feature_shape": list(scorer.ref_features.shape),
        "reference_cache": str(reference_path)
        if reference_path
        else "canonical mp20_features.pt",
        "model_name": callback.afd_model_name,
        "batch_size": callback.afd_batch_size,
        "reference_sampling_seed": callback.seed,
    }
    from huggingface_hub import try_to_load_from_cache
    from zevals.atomistic.afd.features import _default_model_cache_dir

    checkpoint = try_to_load_from_cache(
        callback.afd_model_name, "last.ckpt", cache_dir=str(_default_model_cache_dir())
    )
    if isinstance(checkpoint, str):
        reference_provenance["model_checkpoint_path"] = checkpoint
        reference_provenance["model_checkpoint_sha256"] = sha256(Path(checkpoint))
    return callback, scorer, reference_provenance


def compute_afd(bundle: dict, callbacks: dict, domain: str, output: Path, args):
    """Reuse a scorer and cache generated features for the exact CIF manifest."""
    import torch
    from zevals.atomistic.afd.score import afd_score

    control_validation = validate_legacy_afd_control(domain, args.afd_control_dir)
    scorer_key = json_digest(
        {"domain": domain, "settings": bundle["settings"]["callback"]}
    )
    if scorer_key not in callbacks:
        callbacks[scorer_key] = afd_scorer(
            bundle["settings"], domain, args.device, args.afd_batch_size
        )
    callback, scorer, provenance = callbacks[scorer_key]
    fingerprint = json_digest(
        {
            "manifest": bundle["manifest"],
            "scorer": provenance,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "script_sha256": sha256(Path(__file__)),
        }
    )
    cache_path = output / "features" / f"{fingerprint}.pt"
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        features, successful_ids = cached["features"], cached["target_ids"]
        species_checks = cached["species_checks"]
    else:
        records, successful_ids, species_checks = [], [], []
        for item in bundle["manifest"]:
            atom_array = load_prediction(Path(item["path"]))
            record = callback._atom_array_to_afd_record(atom_array)
            if record is not None:
                records.append(record)
                successful_ids.append(item["target_id"])
                species_checks.append(
                    {
                        "target_id": item["target_id"],
                        "n_atoms": len(record["z"]),
                        "atomic_numbers_sha256": json_digest(record["z"].tolist()),
                    }
                )
        if len(records) < 2:
            raise ValueError("AFD requires at least two usable generation records")
        features = scorer.embeddings(records, show_progress=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        torch.save(
            {
                "features": features,
                "target_ids": successful_ids,
                "species_checks": species_checks,
            },
            temporary,
        )
        temporary.replace(cache_path)
    reference_size = callback.afd_ref_sample_size
    if reference_size is None and callback.afd_match_reference_size:
        reference_size = len(features)
    reference_features = (
        scorer.sample_ref(reference_size, seed=callback.seed)
        if reference_size is not None
        else scorer.ref_features
    )
    metrics = {
        "afd_score": float(afd_score(reference_features, features)),
        "afd_num_samples": len(features),
        "afd_ref_sample_size": reference_size or len(scorer.ref_features),
        "afd_input_success_rate": len(features) / len(bundle["manifest"]),
    }
    return metrics, {
        **provenance,
        "generated_features_path": str(cache_path),
        "generated_features_sha256": tensor_sha256(features),
        "selected_reference_features_sha256": tensor_sha256(reference_features),
        "successful_target_ids": successful_ids,
        "metric_protocol": AFD_PROTOCOL,
        "control_validation": control_validation,
        "species_consistency": {
            "passed": True,
            "checked_cifs": len(species_checks),
            "source": "CIF element symbols; atomic_number rebuilt and checked for every CIF",
            "per_target": species_checks,
        },
    }


def compute_structure(bundle: dict, domain: str):
    """Use the original conditioned molecular graph or periodic matcher."""
    from zatom2.generation_evaluation import (
        GeneratedSample,
        _compute_structure_prediction_metrics,
    )

    from nucleus.utils.material_fingerprints import atom_array_to_structure

    callback = bundle["settings"]["callback"]
    top_k = callback["molecule_top_k" if domain == "omol25" else "material_top_k"]
    if list(top_k) != [1] or callback["material_matcher_scales"] != [True]:
        raise ValueError("Figure 2 requires top1 and material matcher scale=True")
    samples, converted = [], []
    for index, item in enumerate(bundle["manifest"]):
        target_id = item["target_id"]
        atom_array = load_prediction(Path(item["path"]))
        samples.append(
            GeneratedSample(
                sample_index=index,
                sample_id=target_id,
                target_id=target_id,
                batch_index=index,
                model_index=0,
                specification=bundle["specifications"][target_id],
                output=SimpleNamespace(atom_array=atom_array),
            )
        )
        converted.append(
            atom_array_to_structure(atom_array) if domain == "omat24" else None
        )
    evaluation_config = OmegaConf.create(
        {
            "domain": domain,
            "bond_sidecar_path": bundle["settings"]["bond_sidecar_path"],
            "structure_prediction": {
                "top_k": top_k,
                "ltol": callback["structure_ltol"],
                "stol": callback["structure_stol"],
                "angle_tol": callback["structure_angle_tol"],
                "scale": True,
            },
        }
    )
    if domain == "omol25" and evaluation_config.bond_sidecar_path is None:
        raise ValueError("Original OMol conditioned-graph sidecar is required")
    metrics = _compute_structure_prediction_metrics(
        samples, converted, evaluation_config
    )
    reference_files = [Path(sample.specification["input"]) for sample in samples]
    references = [
        {"path": str(path), "sha256": sha256(path)} for path in reference_files
    ]
    return metrics, {
        "reference_cifs": references,
        "evaluation_config": OmegaConf.to_container(evaluation_config, resolve=True),
        "num_targets": len(samples),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("logs/train"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/analysis/zatom2_figure2_20260906/recomputed"),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=EXPERIMENTS,
        default=["base_full", "large_full"],
    )
    epochs = parser.add_mutually_exclusive_group(required=True)
    epochs.add_argument("--epochs", nargs="+", type=int)
    epochs.add_argument("--all-saved-epochs", action="store_true")
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--afd-control-dir",
        type=Path,
        default=Path("logs/analysis/zatom2_figure2_20260906/recomputed/diagnostics"),
    )
    parser.add_argument(
        "--afd-batch-size",
        type=int,
        default=None,
        help="Default: original 256; overrides are recorded",
    )
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    from dotenv import load_dotenv

    load_dotenv(repo / ".env", override=True)
    os.environ["DEBUG"] = "false"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ.setdefault("PROJECT_ROOT", str(repo))
    args.train_dir = args.train_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    import torch

    torch.set_float32_matmul_precision("medium")  # Original zatom2.train setting.
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": args.device,
        "gpu": torch.cuda.get_device_name()
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else None,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "script_sha256": sha256(Path(__file__)),
    }
    callbacks = {}
    for condition in args.conditions:
        selected_epochs = args.epochs
        if args.all_saved_epochs:
            selected_epochs = sorted(
                {
                    int(path.name.removeprefix("epoch_"))
                    for path in (args.train_dir / EXPERIMENTS[condition]).glob(
                        "*/val_structures/predictions/epoch_*"
                    )
                    if path.name.removeprefix("epoch_").isdigit()
                }
            )
        for epoch in selected_epochs:
            for dataset, metric in DATASETS.items():
                if dataset not in args.datasets:
                    continue
                name = f"{condition}_epoch{epoch}_{dataset}"
                print(f"Inventory: {name}", flush=True)
                bundle = discover(args.train_dir, condition, epoch, dataset, repo)
                if bundle is None:
                    print(f"No saved predictions: {name}", flush=True)
                    continue
                path = output / f"{name}.json"
                identity = json_digest(
                    {
                        "manifest": bundle["manifest"],
                        "settings": bundle["settings"],
                        "runtime": runtime,
                        "afd_batch_size": args.afd_batch_size,
                    }
                )
                if path.exists() and not args.overwrite:
                    previous = json.loads(path.read_text())
                    if previous.get("identity") == identity:
                        print(f"Reusing verified result: {name}", flush=True)
                        continue
                    raise ValueError(
                        f"Existing result differs: {path}; use --overwrite"
                    )
                provenance = {
                    key: bundle[key]
                    for key in (
                        "manifest",
                        "settings",
                        "configurations",
                        "missing_targets",
                        "expected_targets",
                        "ordering",
                    )
                }
                if args.inventory_only:
                    write_json(output / f"{name}.inventory.json", provenance)
                    print(
                        f"Saved {len(bundle['manifest'])}/{bundle['expected_targets']} targets",
                        flush=True,
                    )
                    continue
                domain = dataset.split("_")[0]
                if dataset.endswith("generation"):
                    metrics, details = compute_afd(
                        bundle, callbacks, domain, output, args
                    )
                    value, n_samples = metrics["afd_score"], metrics["afd_num_samples"]
                    control_validation = details["control_validation"]
                    metric_protocol = AFD_PROTOCOL
                else:
                    metrics, details = compute_structure(bundle, domain)
                    value = metrics[
                        "molecule_best_rmsd_mean_top1"
                        if domain == "omol25"
                        else "material_match_rate_top1"
                    ]
                    n_samples = len(bundle["manifest"])
                    control_validation = validate_structure_control(
                        condition, epoch, metric, value, output
                    )
                    metric_protocol = STRUCTURE_PROTOCOL
                status = (
                    "recomputed_saved_complete"
                    if n_samples == bundle["expected_targets"]
                    else "recomputed_saved_partial"
                )
                if not control_validation["passed"]:
                    status = "recomputed_control_failed"
                result = {
                    "condition": condition,
                    "epoch": epoch,
                    "dataset": dataset,
                    "metric": metric,
                    "value": value,
                    "n_samples": n_samples,
                    "status": status,
                    "metric_protocol": metric_protocol,
                    "control_validation": control_validation,
                    "supersedes_historical_afd": metric_protocol == AFD_PROTOCOL,
                    "source_file": str(path),
                    "identity": identity,
                    "runtime": runtime,
                    "provenance": provenance,
                    "scoring": details,
                    "metrics": {
                        key: val
                        if val is not None and float("-inf") < float(val) < float("inf")
                        else None
                        for key, val in metrics.items()
                    },
                    "computed_at": datetime.now(timezone.utc).isoformat(),
                    "notes": (
                        "Recomputed from saved CIF predicted species; corrects historical stale carbon atomic_number AFD bug. Species-consistent protocol supersedes all historical AFD."
                        if metric_protocol == AFD_PROTOCOL
                        else "Recomputed from serialized validation CIFs with original conditioned-graph/matching settings; complete epoch15 control validates the protocol."
                    ),
                }
                write_json(path, result)
                print(f"{name}: {value:.10g}, {status}, n={n_samples}", flush=True)
                update_csv(output)
    update_csv(output)


def update_csv(output: Path) -> None:
    """Keep a flat scalar table of completed results, including earlier invocations."""
    fields = [
        "condition",
        "epoch",
        "dataset",
        "metric",
        "value",
        "status",
        "n_samples",
        "source_file",
        "notes",
        "metric_protocol",
    ]
    records = []
    for path in sorted(output.glob("*_epoch*_*.json")):
        if path.name.endswith(".inventory.json"):
            continue
        result = json.loads(path.read_text())
        records.append({key: result.get(key, "") for key in fields})
    temporary = output / "recomputed_metrics.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(output / "recomputed_metrics.csv")


if __name__ == "__main__":
    main()
