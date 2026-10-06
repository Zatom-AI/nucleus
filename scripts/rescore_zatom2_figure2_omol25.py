#!/usr/bin/env python3
"""Rescore preserved Figure 2 molecules with the corrected OMol25 AFD encoder.

No Zatom inference is performed. The original CIFs, reference records, reference
sampling seed, and sample counts are retained; both sides are embedded anew.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

MODEL_NAME = "Ty-Perez/ct-scd-omol25"
MODEL_REVISION = "1296d85333f5177fa143b087b2858c4bdee7e6e7"
PROTOCOL = "species_consistent_afd_v1"


def sha256(path: Path) -> str:
    """Hash an input without loading its entire contents into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_results(root: Path) -> list[Path]:
    """Require the complete previous inventory, including its partial pool."""
    paths = sorted(
        (root / "recomputed/corrected_saved").glob("*_epoch*_omol25_generation.json")
    )
    assert len(paths) == 30
    paths += [
        root
        / f"recomputed/checkpoint_{condition}/pooled/omol25_generation/pooled_metrics.json"
        for condition in ("base_500k", "large_500k")
    ]
    return paths


def result_path(output: Path, source: dict) -> Path:
    if source["condition"].endswith("500k"):
        return (
            output
            / f"checkpoint_{source['condition']}/pooled/omol25_generation/pooled_metrics.json"
        )
    return (
        output
        / "saved"
        / f"{source['condition']}_epoch{source['epoch']}_omol25_generation.json"
    )


def update_csv(output: Path, results: list[dict]) -> None:
    fields = [
        "condition",
        "epoch",
        "dataset",
        "metric",
        "value",
        "n_samples",
        "status",
        "source_file",
        "notes",
        "metric_protocol",
        "afd_model_name",
    ]
    path = output / "recomputed_metrics.csv"
    temporary = path.with_suffix(".csv.partial")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in results)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("logs/analysis/zatom2_figure2_20260906"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    from dotenv import load_dotenv

    repo = Path(__file__).resolve().parents[1]
    load_dotenv(repo / ".env", override=True)
    os.environ.update(
        DEBUG="false",
        WANDB_MODE="disabled",
        PROJECT_ROOT=str(repo),
        HF_HUB_OFFLINE="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    import torch
    from huggingface_hub import try_to_load_from_cache
    from omegaconf import OmegaConf
    from recompute_zatom2_figure2_metrics import load_prediction, tensor_sha256
    from zevals.atomistic.afd.data import resolve_records
    from zevals.atomistic.afd.features import _default_model_cache_dir
    from zevals.atomistic.afd.score import afd_score
    from zevals.molecules.afd import compute_afd_metrics

    from nucleus.callbacks.metrics_logging import _CachedReferenceAFDScorer
    from nucleus.utils.atom_array import _atom_array_to_afd_record

    torch.set_float32_matmul_precision("medium")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    root = args.root.resolve()
    output = root / "recomputed/omol25_encoder_update"
    output.mkdir(parents=True, exist_ok=True)
    config_path = repo / "models/zatom2/configs/evaluation/omol25_unconditional.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)["afd"]
    assert config["model_name"] == MODEL_NAME
    structure_config_path = (
        repo / "models/zatom2/configs/evaluation/omol25_structure_prediction.yaml"
    )
    structure_config = OmegaConf.to_container(
        OmegaConf.load(structure_config_path), resolve=True
    )["afd"]
    assert structure_config == config
    reference_path = Path(config["reference_records_path"])
    cache_dir = Path(config["feature_cache_dir"])
    reference_cache = cache_dir / "Ty-Perez__ct-scd-omol25_reference_features.pt"
    cache_metadata_path = reference_cache.with_suffix(".metadata.json")
    checkpoint = try_to_load_from_cache(
        MODEL_NAME,
        "last.ckpt",
        revision=MODEL_REVISION,
        cache_dir=str(_default_model_cache_dir()),
    )
    assert isinstance(checkpoint, str), "Download the pinned encoder first"
    current_checkpoint = try_to_load_from_cache(
        MODEL_NAME, "last.ckpt", cache_dir=str(_default_model_cache_dir())
    )
    assert current_checkpoint == checkpoint, "Cached main no longer matches the pin"
    runtime = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "script_sha256": sha256(Path(__file__)),
    }
    reference_signature = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "model_checkpoint_path": checkpoint,
        "model_checkpoint_sha256": sha256(Path(checkpoint)),
        "reference_records_path": str(reference_path),
        "reference_records_sha256": sha256(reference_path),
        "batch_size": args.batch_size,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "evaluation_config_path": str(config_path),
        "evaluation_config_sha256": sha256(config_path),
    }
    sources = [(path, json.loads(path.read_text())) for path in source_results(root)]
    assert sum(source["n_samples"] for _, source in sources) == 31769
    assert all(
        source["scoring"]["model_name"] == "Ty-Perez/ct-scd-geom10"
        for _, source in sources
    )
    write_json(
        output / "run_manifest.json",
        {
            "runtime": runtime,
            "reference_signature": reference_signature,
            "sources": [
                {
                    "path": str(path),
                    "sha256": sha256(path),
                    "condition": source["condition"],
                    "epoch": source["epoch"],
                    "n_samples": source["n_samples"],
                }
                for path, source in sources
            ],
            "new_zatom_samples": 0,
        },
    )
    print("Loading the unchanged raw reference records", flush=True)
    records = resolve_records(str(reference_path))
    assert len(records) == 188762
    # This scorer always builds the requested encoder; no geom10 tensor is reused.
    scorer = _CachedReferenceAFDScorer(
        torch.empty((0, 0)),
        data_modality="molecules",
        model_name=MODEL_NAME,
        batch_size=args.batch_size,
        device=args.device,
    )
    if reference_cache.exists():
        metadata = json.loads(cache_metadata_path.read_text())
        assert metadata["signature"] == reference_signature
        assert metadata["cache_sha256"] == sha256(reference_cache)
        reference_features = torch.load(
            reference_cache, map_location="cpu", weights_only=False
        )
        assert metadata["features_sha256"] == tensor_sha256(reference_features)
    else:
        print("Embedding all 188762 reference records with ct-scd-omol25", flush=True)
        reference_features = scorer.embeddings(records, show_progress=True)
        assert len(reference_features) == len(records)
        assert torch.isfinite(reference_features).all()
        temporary = reference_cache.with_suffix(".pt.partial")
        torch.save(reference_features, temporary)
        temporary.replace(reference_cache)
        write_json(
            cache_metadata_path,
            {
                "signature": reference_signature,
                "features_sha256": tensor_sha256(reference_features),
                "cache_sha256": sha256(reference_cache),
                "shape": list(reference_features.shape),
            },
        )
    scorer.ref_features = reference_features
    reference_indices = torch.randperm(
        len(records), generator=torch.Generator().manual_seed(0)
    )[:1000]
    old_cache = cache_dir / "Ty-Perez__ct-scd-geom10_reference_features.pt"
    old_features = torch.load(old_cache, map_location="cpu", weights_only=False)
    assert old_features.shape[0] == len(records)
    old_features_hash = tensor_sha256(old_features)
    old_selected_hash = tensor_sha256(old_features[reference_indices])
    assert all(
        source["scoring"]["reference_features_sha256"] == old_features_hash
        for _, source in sources
    )
    assert all(
        source["scoring"]["selected_reference_features_sha256"] == old_selected_hash
        for _, source in sources
        if source["n_samples"] == 1000
    )
    del old_features
    independent_scorer = _CachedReferenceAFDScorer(
        reference_features,
        data_modality="molecules",
        model_name=MODEL_NAME,
        batch_size=args.batch_size,
        device=args.device,
    )
    replay_checks = []
    for start in (
        0,
        (len(records) // 2 // args.batch_size) * args.batch_size,
        ((len(records) - 1) // args.batch_size) * args.batch_size,
    ):
        stop = min(start + args.batch_size, len(records))
        replay_features = independent_scorer.embeddings(
            records[start:stop], show_progress=False
        )
        expected_features = reference_features[start:stop]
        torch.testing.assert_close(
            replay_features, expected_features, rtol=1.3e-6, atol=1e-5
        )
        replay_checks.append(
            {
                "start": start,
                "stop": stop,
                "bitwise_equal": torch.equal(replay_features, expected_features),
                "max_abs_error": float(
                    (replay_features - expected_features).abs().max()
                ),
            }
        )
    del independent_scorer, replay_features, expected_features
    assert torch.equal(
        scorer.sample_ref(1000, seed=0), reference_features[reference_indices]
    )
    control_source_path, control_source = next(
        (path, source) for path, source in sources if source["n_samples"] == 1000
    )
    control_manifest = (
        control_source.get("manifest") or control_source["provenance"]["manifest"]
    )
    control_records = []
    for item in control_manifest:
        cif_path = Path(item["path"])
        assert sha256(cif_path) == item["sha256"]
        record = _atom_array_to_afd_record(load_prediction(cif_path))
        assert record is not None and len(record["z"]) > 0
        control_records.append(record)
    captured_features = []
    original_embeddings = scorer.embeddings

    def capture_embeddings(dataset, **kwargs):
        features = original_embeddings(dataset, **kwargs)
        captured_features.append(features)
        return features

    scorer.embeddings = capture_embeddings
    try:
        public = compute_afd_metrics(
            control_records,
            scorer=scorer,
            match_reference_size=True,
            seed=0,
            show_progress=False,
        )
    finally:
        scorer.embeddings = original_embeddings
    assert len(captured_features) == 1
    assert public["afd_num_samples"] == public["afd_ref_sample_size"] == 1000
    control_value = float(
        afd_score(reference_features[reference_indices], captured_features[0])
    )
    assert abs(public["afd_score"] - control_value) <= 1e-10
    analytic_x = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=torch.float64)
    analytic_expected = 5 - 4e-6
    analytic_value = float(
        afd_score(analytic_x, analytic_x + torch.tensor([1, 2], dtype=torch.float64))
    )
    assert abs(analytic_value - analytic_expected) <= 1e-12
    self_distance = float(
        afd_score(reference_features[:1000], reference_features[:1000])
    )
    assert self_distance <= 1e-6
    control_path = output / "encoder_control.json"
    control = {
        "passed": True,
        "method": "new_encoder_independent_reference_replay_public_api_and_analytic_afd",
        "reference_signature": reference_signature,
        "reference_features_sha256": tensor_sha256(reference_features),
        "reference_feature_shape": list(reference_features.shape),
        "reference_batch_replays": replay_checks,
        "feature_rtol": 1.3e-6,
        "feature_atol": 1e-5,
        "public_api_control_source": str(control_source_path),
        "public_api_control_sample_count": len(control_records),
        "public_api_score": public["afd_score"],
        "cached_feature_score": control_value,
        "api_absolute_difference": abs(public["afd_score"] - control_value),
        "api_absolute_tolerance": 1e-10,
        "analytic_afd_expected": analytic_expected,
        "analytic_afd_actual": analytic_value,
        "analytic_afd_absolute_tolerance": 1e-12,
        "self_afd": self_distance,
        "reference_sampling_seed": 0,
        "selected_reference_indices_sha256": tensor_sha256(reference_indices),
        "old_selected_reference_features_sha256": old_selected_hash,
        "notes": "Validates the new encoder/cache and score path; no cross-encoder score equality is expected. Historical carbon replay remains a separate old-encoder diagnostic.",
    }
    write_json(control_path, control)
    proof = {
        "passed": True,
        "method": control["method"],
        "source_file": str(control_path),
        "source_sha256": sha256(control_path),
    }
    del records, control_records, captured_features
    print("New encoder reference and public API controls passed", flush=True)
    results = []
    for source_path, previous in sources:
        manifest = previous.get("manifest") or previous["provenance"]["manifest"]
        n_samples = previous["n_samples"]
        assert len(manifest) == len({row["target_id"] for row in manifest}) == n_samples
        generated_records, ids, species_checks = [], [], []
        for item in manifest:
            path = Path(item["path"])
            assert sha256(path) == item["sha256"], f"Saved CIF changed: {path}"
            atoms = load_prediction(path)
            record = _atom_array_to_afd_record(atoms)
            assert record is not None and len(record["z"]) > 0
            generated_records.append(record)
            ids.append(item["target_id"])
            species_checks.append(
                {
                    "target_id": item["target_id"],
                    "n_atoms": len(record["z"]),
                    "atomic_numbers_sha256": hashlib.sha256(
                        json.dumps(record["z"].tolist()).encode()
                    ).hexdigest(),
                }
            )
        print(
            f"Scoring {previous['condition']} epoch{previous['epoch']} N={n_samples}",
            flush=True,
        )
        features = scorer.embeddings(generated_records, show_progress=True)
        assert len(features) == n_samples and torch.isfinite(features).all()
        reference = scorer.sample_ref(n_samples, seed=0)
        value = float(afd_score(reference, features))
        assert 0 <= value < float("inf")
        path = result_path(output, previous)
        feature_path = (
            output / "features" / f"{previous['condition']}_epoch{previous['epoch']}.pt"
        )
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": features, "target_ids": ids}, feature_path)
        species = {
            "passed": True,
            "checked_cifs": n_samples,
            "per_target": species_checks,
            "source": "Original saved CIF element symbols; atomic numbers rebuilt and verified",
        }
        scoring = {
            **reference_signature,
            "reference_cache": str(reference_cache),
            "reference_features_sha256": tensor_sha256(reference_features),
            "reference_feature_shape": list(reference_features.shape),
            "reference_sampling_seed": 0,
            "selected_reference_features_sha256": tensor_sha256(reference),
            "selected_reference_indices_sha256": tensor_sha256(
                reference_indices[:n_samples]
            ),
            "generated_features_path": str(feature_path),
            "generated_features_sha256": tensor_sha256(features),
            "successful_target_ids": ids,
            "metric_protocol": PROTOCOL,
            "control_validation": proof,
            "species_consistency": species,
        }
        result = copy.deepcopy(previous)
        result.update(
            value=value,
            n_samples=n_samples,
            source_file=str(path),
            scoring=scoring,
            control_validation=proof,
            species_consistency=species,
            afd_model_name=MODEL_NAME,
            runtime=runtime,
            computed_at=datetime.now(UTC).isoformat(),
            supersedes_result={"path": str(source_path), "sha256": sha256(source_path)},
            encoder_update="ct-scd-geom10 to ct-scd-omol25; same saved samples",
            historical_species_diagnostic=previous["control_validation"],
            notes="Same saved CIFs rescored with Ty-Perez/ct-scd-omol25 and freshly matched reference features; no new Zatom inference. Original sample origin and count retained.",
        )
        metrics = {
            "afd_score": value,
            "afd_num_samples": n_samples,
            "afd_ref_sample_size": n_samples,
            "afd_input_success_rate": 1.0,
        }
        result["afd_metrics" if "afd_metrics" in previous else "metrics"] = metrics
        write_json(path, result)
        results.append(result)
        update_csv(output, results)
        print(f"AFD={value:.12g}; {len(results)}/32 pools complete", flush=True)
    assert len(results) == 32 and sum(row["n_samples"] for row in results) == 31769
    write_json(
        output / "scoring_complete.json",
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "runtime": runtime,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "pools": len(results),
            "sample_count": 31769,
            "new_zatom_samples": 0,
            "reference_count": len(reference_features),
            "control_source_sha256": sha256(control_path),
            "recomputed_csv_sha256": sha256(output / "recomputed_metrics.csv"),
            "results": [
                {
                    "source_file": row["source_file"],
                    "sha256": sha256(Path(row["source_file"])),
                }
                for row in results
            ],
        },
    )
    print("All molecular AFD pools rescored successfully", flush=True)


if __name__ == "__main__":
    main()
