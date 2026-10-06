"""Reproducible QM9 and MP20 evaluation for the Zatom-2 Table-2 model."""

from __future__ import annotations

import csv
import gc
import gzip
import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from zatom2.generation_evaluation import (
    _dump_samples,
    _flatten_outputs,
    _load_input_specifications,
    _sanitize_json,
    _write_json,
)
from zatom2.table2_resume import (
    EvaluationPaused,
    atomic_json,
    bind_contract,
    commit_group,
    evaluation_allocation,
    group_seed,
    remaining_seconds,
    sampling_source_hashes,
    verify_group,
)

LOGGER = logging.getLogger(__name__)

PROTOCOL_NAME = "zatom2-paper-table2-qm9-mp20"
PROTOCOL_VERSION = 3
# Scoring revisions must not change the frozen version-3 sampling contracts.
QM9_SCORING_REVISION = 1
DOMAIN_ORDER = ("qm9", "mp20")
ALL_STAGES = ("prepare", "sample", "score_qm9", "score_mp20", "report")

# These identify the exact source artifacts and deterministic selections used to
# train the supplied Table-2 checkpoint. Artifact paths are deliberately not
# included because they are machine-local; the content and selected-index
# digests are portable.
CANONICAL_TRAINING_DATASETS = {
    "qm9": {
        "num_examples": 100_000,
        "source_num_examples": 130_831,
        "source_artifact_sha256": (
            "ed0df15e32aaf1fa72527285cea9bc1e8f6ae60fc3fcf66357bd432c6138e9cd"
        ),
        "selection": "seeded_torch_permutation",
        "selection_seed": 42,
        "selected_indices_sha256": (
            "d78b660c598f6da58bb54cdc12a7327971700f62a0a45eee82753e51babe4043"
        ),
        "shards": [f"data{index:04d}.aselmdb" for index in range(10)],
        "shard_sha256": {
            "data0000.aselmdb": "eafecf0ebae2cbb14b000447ef056bc02d008648df85e08e7c8d3caafe12f3ed",
            "data0001.aselmdb": "12320ae8b96320fc8b6bcb76a47ee774f21cfe498b6f88e4b06fdb1342f831e6",
            "data0002.aselmdb": "b306d682239f00cb1ad29c31d7670da127a1dd5511e4b671628653a034a8a372",
            "data0003.aselmdb": "40a7a290fd94a1d7418949c240d925c8c5368912d8cbbb4295ad80e70d415b2e",
            "data0004.aselmdb": "41823b9c97a703389ce115b03e0f42f763e36f522f20b7a1e5770e67b41c7d0a",
            "data0005.aselmdb": "8cc99e4d8cb38f9001928074622ee8b2bddfb0bd7025b85058403ac930ec89ef",
            "data0006.aselmdb": "d96b4e05666eba4c5152f9d9a2dc8b3598c8a0d65e274e4fcbbad25d3ef44f2c",
            "data0007.aselmdb": "ebf293d8861e23503682cc839a7d9f937778f35bd6b6f954321012f619726943",
            "data0008.aselmdb": "925d67b24ac9a022e8c48211f05313618833ae85db16e18213aad3baa56e2e97",
            "data0009.aselmdb": "2ad710213e71a49e070174fe88fd646bec1f9ec013135d61dd80d582b4b7b824",
        },
    },
    "mp20": {
        "num_examples": 27_138,
        "source_num_examples": 45_229,
        "source_artifact_sha256": (
            "2cdccedce5d3b119d0b18a73ba3568f4ce6c5f9197f62f59f287c9a793207850"
        ),
        "selection": "source_order_prefix",
        "selection_seed": None,
        "selected_indices_sha256": (
            "256948d603a3ac935f3a56445c91ba1d9bcbd677a86afbb17b754e7db6506544"
        ),
        "shards": [f"data{index:04d}.aselmdb" for index in range(3)],
        "shard_sha256": {
            "data0000.aselmdb": "3fdd0f8904983e39ab32df78dd3352778d5a57b37244e5726855877f7ecf0fa2",
            "data0001.aselmdb": "032370392f74bff8776332129561ffdf000ee3aa25df0d3d387eccbfeb7541be",
            "data0002.aselmdb": "5466a2fc46d6a95ed0339939c86c595544d22999cef0db541bef3f99978d9270",
        },
    },
}

FROZEN_INFERENCE_SAMPLER = {
    "kind": "default",
    "cfg_feature_mode": "configured",
    "cfg_features": [
        "active_donor",
        "active_acceptor",
        "ref_atomwise_rasa",
        "mean_force_norm",
        "has_mean_force_norm",
        "force_rms_p95",
        "has_force_rms_p95",
    ],
    "use_classifier_free_guidance": False,
    "cfg_t_max": None,
    "cfg_scale": 1.5,
    "center_option": "all",
    "s_trans": 1.0,
    "inference_noise_scaling_factor": 1.0,
    "allow_realignment": False,
    "n_recycle": 2,
    "num_timesteps": 100,
    "step_scale": 1.5,
    "noise_scale": 1.003,
    "p": 7,
    "gamma_0": 0.8,
    "gamma_min": 1.0,
    "s_jitter_origin": 0.0,
    "solver": "af3",
    "min_t": 0,
    "max_t": 1,
    "sigma_data": 16,
    "s_min": 4e-4,
    "s_max": 160,
    "fraction_of_steps_to_fix_motif": 0.0,
    "skip_few_diffusion_steps": False,
    "insert_motif_at_end": True,
}

FROZEN_TABLE2_RUNTIME = {
    "seed": 42,
    "evaluation.seed": 42,
    "evaluation.qm9.num_samples": 10_000,
    "evaluation.qm9.training_num_examples": 100_000,
    "evaluation.mp20.num_samples": 2_500,
    "evaluation.mp20.training_num_examples": 27_138,
    "diffusion_batch_size": 8,
    "n_batches": 1,
    "num_nodes": 1,
    "devices_per_node": 1,
    "use_ema_weights": True,
    "skip_existing": False,
    "save_cifs": True,
    "save_molecules": False,
    "dump_prediction_metadata_json": True,
    "dump_trajectories": False,
    "output_full_json": True,
    "prevalidate_inputs": True,
    "low_memory_mode": False,
    "cleanup_guideposts": True,
    "cleanup_virtual_atoms": True,
    "read_sequence_from_sequence_head": True,
    "align_trajectory_structures": False,
    "bond_sidecar_path": None,
    "global_prefix": "",
    # Freeze these mappings as whole objects, not just their known leaves.
    # Hydra permits arbitrary additions to both, and the inference engine
    # consumes them when constructing every sample.
    "specification": {},
    "transform_overrides": {"remove_hydrogens": None},
    "material_coordinate_output_source": "fractional",
    "wrap_material_euclidean_coordinates": True,
    # Compare the whole mapping so future/unknown Hydra additions cannot alter
    # sampling while escaping the canonical-runtime marker.
    "inference_sampler": FROZEN_INFERENCE_SAMPLER,
    "evaluation.mp20.lematgenbench.run_name": "zatom2-table2",
    "evaluation.mp20.lematgenbench.config_name": "comprehensive_multi_mlip_hull",
    "evaluation.mp20.lematgenbench.families": [
        "distribution",
        "diversity",
        "novelty",
        "uniqueness",
        "stability",
        "sun",
    ],
    "evaluation.mp20.lematgenbench.fingerprint_method": "structure-matcher",
    "evaluation.mp20.lematgenbench.runner_command": [
        "uv",
        "run",
        "--frozen",
        "python",
    ],
    "evaluation.mp20.lematgenbench.expected_commit": (
        "58e6eae3e4a6c87c22171cf069123ecc4e2fa7e6"
    ),
    "evaluation.mp20.lematgenbench.expected_lock_sha256": (
        "74970323f6697d50e60251de9182e7a559031bce67ab2f693986742e7c286850"
    ),
    "evaluation.mp20.lematgenbench.verify_mace_checkpoint": True,
    "evaluation.mp20.lematgenbench.expected_mace_checkpoint_sha256": (
        "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638"
    ),
    "evaluation.mp20.lematgenbench.verify_orb_checkpoint": True,
    "evaluation.mp20.lematgenbench.expected_orb_checkpoint_sha256": (
        "0a41ef1132ad9c41ee0c7b9d855fccabfefb97abe04625c97ec0d07930b6c0f0"
    ),
    "evaluation.mp20.lematgenbench.verify_uma_checkpoint": True,
    "evaluation.mp20.lematgenbench.expected_uma_checkpoint_sha256": (
        "02f83e20547294d3d47dd9dfbe4a4ee51eebca21b2e1febe63210e55b764cc78"
    ),
    "evaluation.mp20.lematgenbench.enforce_reference_revisions": True,
    "evaluation.mp20.lematgenbench.runtime_device": "cuda",
    "evaluation.mp20.lematgenbench.cuda_visible_devices": None,
    "evaluation.mp20.lematgenbench.parse_failures_as_invalid": True,
    "evaluation.mp20.lematgenbench.require_complete_oracle_coverage": True,
    "evaluation.mp20.lematgenbench.require_clean_repo": True,
    "evaluation.mp20.lematgenbench.fail_on_family_error": True,
    "evaluation.mp20.lematgenbench.distribution_key": "distribution/JSDistance",
    "evaluation.mp20.lematgenbench.diversity_key": ("diversity/element_diversity"),
}

_QM9_SCORING_MODULES = (
    "zatom2.table2_evaluation",
    "zatom2.generation_evaluation",
    "zevals.molecules.qm9",
    "zevals.molecules.qm9_zatom1",
    "nucleus.utils.atom_array",
    "nucleus.inference_input_parsing",
    "zevals.utils.packages",
    "pymatgen.core.structure",
    "pymatgen.io.babel",
    "openbabel.openbabel",
    "openbabel.pybel",
    "openbabel._openbabel",
)
_MP20_SCORING_MODULES = (
    "zatom2.table2_evaluation",
    "zevals.materials.lematgenbench",
    "zevals.materials._lemat_resume",
    "zevals.materials._lemat_resume_runtime",
    "zevals.materials.chgnet_prerelax",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _write_json_gz(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    payload = (json.dumps(_sanitize_json(value), separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    temporary_path.write_bytes(gzip.compress(payload, mtime=0))
    temporary_path.replace(path)


def _read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_stages(value: str | Sequence[str]) -> tuple[str, ...]:
    stages = [value] if isinstance(value, str) else [str(stage) for stage in value]
    if "all" in stages:
        if len(stages) != 1:
            raise ValueError(
                "evaluation.stages='all' cannot be combined with other stages."
            )
        return ALL_STAGES
    unknown = sorted(set(stages).difference(ALL_STAGES))
    if unknown:
        raise ValueError(f"Unknown Table-2 evaluation stages: {unknown}")
    if not stages:
        raise ValueError("At least one Table-2 evaluation stage is required.")
    requested = set(stages)
    return tuple(stage for stage in ALL_STAGES if stage in requested)


def _protocol_conformance_mismatches(cfg: DictConfig) -> dict[str, dict[str, Any]]:
    """Return resolved runtime values that differ from the manuscript protocol."""
    mismatches = {}
    for path, expected in FROZEN_TABLE2_RUNTIME.items():
        actual = OmegaConf.select(cfg, path, default=None)
        if actual != expected:
            mismatches[path] = {"configured": actual, "expected": expected}
    if bool(cfg.evaluation.get("hash_checkpoint", True)) is not True:
        mismatches["evaluation.hash_checkpoint"] = {
            "configured": bool(cfg.evaluation.get("hash_checkpoint", True)),
            "expected": True,
        }
    if OmegaConf.select(cfg, "evaluation.mp20.prerelax.enabled", default=False):
        for key, expected in {
            "fmax": 0.1,
            "steps": 1500,
            "relax_cell": True,
            "device": "cuda",
            "seed": int(cfg.evaluation.seed),
        }.items():
            path = f"evaluation.mp20.prerelax.{key}"
            actual = OmegaConf.select(cfg, path)
            if actual != expected:
                mismatches[path] = {"configured": actual, "expected": expected}
    return mismatches


def _validate_protocol_conformance(cfg: DictConfig) -> dict[str, dict[str, Any]]:
    """Require frozen defaults unless a diagnostic run explicitly opts out."""
    mismatches = _protocol_conformance_mismatches(cfg)
    if not mismatches:
        return mismatches
    if not bool(cfg.evaluation.get("allow_noncanonical_protocol", False)):
        raise ValueError(
            "Table-2 runtime overrides would change the frozen manuscript "
            f"protocol: {mismatches}. Set "
            "evaluation.allow_noncanonical_protocol=true only for a diagnostic "
            "run that will not populate Table 2."
        )
    LOGGER.warning("Running a noncanonical diagnostic Table-2 protocol: %s", mismatches)
    return mismatches


def _protocol_paths(protocol_dir: Path) -> dict[str, Path]:
    return {
        "manifest": protocol_dir / "protocol_manifest.json",
        "qm9_counts": protocol_dir / "qm9_train_node_counts.json.gz",
        "mp20_counts": protocol_dir / "mp20_train_node_counts.json.gz",
        "qm9_reference": protocol_dir / "qm9_train_canonical_smiles.json.gz",
        "qm9_inputs": protocol_dir / "qm9_inputs.json",
        "mp20_inputs": protocol_dir / "mp20_inputs.json",
    }


def sample_node_counts(
    node_counts: Sequence[int], *, num_samples: int, seed: int
) -> list[int]:
    """Sample a deterministic empirical atom-count distribution with replacement."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    count_tensor = torch.as_tensor(node_counts, dtype=torch.long, device="cpu")
    if count_tensor.ndim != 1 or count_tensor.numel() == 0:
        raise ValueError("node_counts must be a non-empty one-dimensional sequence.")
    if bool((count_tensor <= 0).any()):
        raise ValueError("node_counts must contain only positive values.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    histogram = torch.bincount(count_tensor).float()
    sampled = torch.multinomial(
        histogram,
        int(num_samples),
        replacement=True,
        generator=generator,
    )
    return sampled.tolist()


def _sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value with stable key and whitespace handling."""
    payload = json.dumps(
        _sanitize_json(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_input_manifest(
    domain: str,
    node_counts: Sequence[int],
    *,
    num_samples: int,
    seed: int,
    diffusion_batch_size: int = 8,
) -> dict[str, Any]:
    """Build a grouped, immutable inputless-design manifest for one domain.

    Zatom-1 draws one node count per requested structure. RFD3 instead produces
    ``diffusion_batch_size`` replicas of one specification, all of which share
    that specification's node count. To retain Zatom-1's exact sampled count
    multiset without running one under-filled model call per structure, requests
    with the same sampled count are grouped into diffusion replicas. The final
    group for a count may contain padding replicas, which are omitted according
    to the recorded ``table2_sample_indices`` plan after inference.
    """
    if domain not in DOMAIN_ORDER:
        raise ValueError(f"Unsupported Table-2 domain: {domain!r}")
    if diffusion_batch_size <= 0:
        raise ValueError("diffusion_batch_size must be positive.")
    sampled_counts = sample_node_counts(node_counts, num_samples=num_samples, seed=seed)
    manifest: dict[str, Any] = {}
    if domain == "qm9":
        manifest["global_args"] = {
            "total_charge": 0,
            "spin_multiplicity": 1,
        }
    sample_indices_by_count: dict[int, list[int]] = defaultdict(list)
    for sample_index, atom_count in enumerate(sampled_counts):
        sample_indices_by_count[int(atom_count)].append(sample_index)

    suffix = "m" if domain == "qm9" else "M"
    group_index = 0
    for atom_count in sorted(sample_indices_by_count):
        sample_indices = sample_indices_by_count[atom_count]
        for start in range(0, len(sample_indices), diffusion_batch_size):
            retained_sample_indices = sample_indices[
                start : start + diffusion_batch_size
            ]
            group_id = f"{domain}-n{atom_count:03d}-group-{group_index:06d}"
            manifest[group_id] = {
                "contig": f"{atom_count}{suffix}",
                "extra": {
                    "table2_domain": domain,
                    "table2_group_index": group_index,
                    "table2_node_count": atom_count,
                    "table2_node_count_seed": int(seed),
                    "table2_diffusion_batch_size": int(diffusion_batch_size),
                    # Replica i maps to table2_sample_indices[i]. Any remaining
                    # replica indices in this group are deterministic padding.
                    "table2_sample_indices": retained_sample_indices,
                    "table2_retained_model_indices": list(
                        range(len(retained_sample_indices))
                    ),
                },
            }
            group_index += 1
    return manifest


def _sampling_plan_metadata(
    domain: str,
    specifications: Mapping[str, Mapping[str, Any]],
    *,
    diffusion_batch_size: int,
    expected_num_samples: int | None = None,
) -> dict[str, Any]:
    """Validate grouped specifications and summarize their retention plan."""
    if domain not in DOMAIN_ORDER:
        raise ValueError(f"Unsupported Table-2 domain: {domain!r}")
    if diffusion_batch_size <= 0:
        raise ValueError("diffusion_batch_size must be positive.")

    groups: list[dict[str, Any]] = []
    all_sample_indices: list[int] = []
    node_count_histogram: Counter[int] = Counter()
    node_count_seeds: set[int] = set()
    for expected_group_index, (group_id, specification) in enumerate(
        specifications.items()
    ):
        extra = specification.get("extra")
        if not isinstance(extra, Mapping):
            raise ValueError(f"{group_id}: missing Table-2 sampling-plan metadata.")
        try:
            plan_domain = str(extra["table2_domain"])
            group_index = int(extra["table2_group_index"])
            node_count = int(extra["table2_node_count"])
            node_count_seed = int(extra["table2_node_count_seed"])
            plan_batch_size = int(extra["table2_diffusion_batch_size"])
            sample_indices = [int(index) for index in extra["table2_sample_indices"]]
            retained_model_indices = [
                int(index) for index in extra["table2_retained_model_indices"]
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{group_id}: incomplete Table-2 sampling-plan metadata."
            ) from error
        if plan_domain != domain:
            raise ValueError(
                f"{group_id}: plan domain {plan_domain!r} does not match {domain!r}."
            )
        if group_index != expected_group_index:
            raise ValueError(
                f"{group_id}: expected table2_group_index={expected_group_index}, "
                f"found {group_index}."
            )
        if node_count <= 0:
            raise ValueError(f"{group_id}: table2_node_count must be positive.")
        expected_group_id = (
            f"{domain}-n{node_count:03d}-group-{expected_group_index:06d}"
        )
        if group_id != expected_group_id:
            raise ValueError(
                f"Expected canonical group ID {expected_group_id!r}, found "
                f"{group_id!r}."
            )
        suffix = "m" if domain == "qm9" else "M"
        if specification.get("contig") != f"{node_count}{suffix}":
            raise ValueError(
                f"{group_id}: contig does not match planned node count {node_count}."
            )
        if plan_batch_size != diffusion_batch_size:
            raise ValueError(
                f"{group_id}: plan diffusion batch size {plan_batch_size} does not "
                f"match configured size {diffusion_batch_size}."
            )
        if not sample_indices or len(sample_indices) > diffusion_batch_size:
            raise ValueError(
                f"{group_id}: retained sample count must be in [1, "
                f"{diffusion_batch_size}]."
            )
        if retained_model_indices != list(range(len(sample_indices))):
            raise ValueError(
                f"{group_id}: retained model indices must be a contiguous prefix."
            )
        all_sample_indices.extend(sample_indices)
        node_count_histogram[node_count] += len(sample_indices)
        node_count_seeds.add(node_count_seed)
        groups.append(
            {
                "group_id": str(group_id),
                "group_index": group_index,
                "node_count": node_count,
                "sample_indices": sample_indices,
                "retained_model_indices": retained_model_indices,
            }
        )

    ordered_sample_indices = sorted(all_sample_indices)
    if len(set(all_sample_indices)) != len(all_sample_indices):
        raise ValueError("Table-2 sampling plan contains duplicate sample indices.")
    if ordered_sample_indices != list(range(len(all_sample_indices))):
        raise ValueError(
            "Table-2 sampling plan must cover contiguous sample indices starting at 0."
        )
    if (
        expected_num_samples is not None
        and len(all_sample_indices) != expected_num_samples
    ):
        raise ValueError(
            f"Expected a {domain.upper()} plan for {expected_num_samples} samples, "
            f"found {len(all_sample_indices)}."
        )
    if len(node_count_seeds) != 1:
        raise ValueError(
            "All Table-2 sampling groups must use the same node-count seed."
        )

    plan_payload = {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "domain": domain,
        "node_count_seed": next(iter(node_count_seeds)),
        "diffusion_batch_size": diffusion_batch_size,
        "groups": groups,
    }
    generated_outputs = len(groups) * diffusion_batch_size
    return {
        "sha256": _sha256_json(plan_payload),
        "node_count_seed": next(iter(node_count_seeds)),
        "diffusion_batch_size": diffusion_batch_size,
        "num_input_specifications": len(groups),
        "num_generated_outputs": generated_outputs,
        "num_retained_outputs": len(all_sample_indices),
        "num_discarded_padding_outputs": generated_outputs - len(all_sample_indices),
        "node_count_histogram": {
            str(node_count): node_count_histogram[node_count]
            for node_count in sorted(node_count_histogram)
        },
    }


def _dataset_manifest(
    data_dir: Path,
    domain: str,
    expected_count: int,
    *,
    require_canonical: bool = True,
) -> dict[str, Any]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{domain.upper()} training data manifest does not exist: {manifest_path}"
        )
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise TypeError(f"Dataset manifest must be a JSON object: {manifest_path}")
    if str(manifest.get("dataset")) != domain:
        raise ValueError(
            f"Expected a {domain!r} dataset at {data_dir}, found "
            f"{manifest.get('dataset')!r}."
        )
    if int(manifest.get("num_examples", -1)) != expected_count:
        raise ValueError(
            f"Expected {expected_count:,} {domain.upper()} training records, "
            f"but {manifest_path} declares {manifest.get('num_examples')!r}."
        )
    canonical = CANONICAL_TRAINING_DATASETS[domain]
    enforce_canonical = (
        require_canonical and expected_count == canonical["num_examples"]
    )
    if enforce_canonical:
        contract_keys = (
            "source_num_examples",
            "source_artifact_sha256",
            "selection",
            "selection_seed",
            "selected_indices_sha256",
            "shards",
        )
        mismatches = {
            key: {"found": manifest.get(key), "expected": canonical[key]}
            for key in contract_keys
            if manifest.get(key) != canonical[key]
        }
        if mismatches:
            raise ValueError(
                f"{domain.upper()} training data does not match the exact "
                f"Table-2 checkpoint selection: {mismatches}"
            )
    shard_metadata = []
    for shard_name in manifest.get("shards", []):
        shard_path = data_dir / str(shard_name)
        if not shard_path.is_file():
            raise FileNotFoundError(f"Dataset shard is missing: {shard_path}")
        shard_sha256 = _sha256_file(shard_path)
        if enforce_canonical:
            expected_shard_sha256 = canonical["shard_sha256"].get(str(shard_name))
            if shard_sha256 != expected_shard_sha256:
                raise ValueError(
                    f"{domain.upper()} shard does not match the exact Table-2 "
                    f"checkpoint partition: {shard_path} has {shard_sha256}, "
                    f"expected {expected_shard_sha256}."
                )
        shard_metadata.append(
            {
                "path": str(shard_path),
                "size_bytes": shard_path.stat().st_size,
                "sha256": shard_sha256,
            }
        )
    return {
        "path": str(data_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest": dict(manifest),
        "shards": shard_metadata,
    }


def _scan_qm9_training_data(data_dir: Path) -> tuple[list[int], list[str], int]:
    """Preserve version-3 sampling assets, including the legacy EDM reference.

    The corrected Table 2 scorer does not use this novelty reference. Keeping
    the prepared assets unchanged lets completed samples be rescored.
    """
    from atomworks.ml.datasets import ASELMDBDataset
    from atomworks.ml.datasets.loaders import create_ase_atoms_loader
    from rdkit import rdBase
    from zevals.molecules.qm9 import evaluate_qm9_sample

    node_counts: list[int] = []
    canonical_smiles: list[str] = []
    invalid_references = 0
    with rdBase.BlockLogs():
        with ASELMDBDataset(
            paths=str(data_dir),
            name="table2_qm9_reference",
            loader=create_ase_atoms_loader(res_name="LIG"),
            readonly=True,
            readahead=False,
        ) as dataset:
            for record in dataset:
                atoms = record["atoms"]
                node_counts.append(len(atoms))
                result = evaluate_qm9_sample(
                    atoms,
                    smiles_remove_hydrogens=False,
                    isomeric_smiles=True,
                    # EDM builds its training novelty set from the full
                    # geometry-inferred graph; largest-fragment selection is
                    # applied only to generated-sample validity.
                    smiles_largest_fragment=False,
                )
                if result["valid"]:
                    canonical_smiles.append(str(result["canonical_smiles"]))
                else:
                    invalid_references += 1
    return node_counts, canonical_smiles, invalid_references


def _scan_mp20_training_data(data_dir: Path) -> list[int]:
    from atomworks.ml.datasets import ASELMDBDataset
    from atomworks.ml.datasets.loaders import create_ase_materials_loader

    node_counts: list[int] = []
    with ASELMDBDataset(
        paths=str(data_dir),
        name="table2_mp20_node_counts",
        loader=create_ase_materials_loader(
            res_name="LIG",
            wrap_fractional_coordinates=True,
            include_atom_array=True,
        ),
        readonly=True,
        readahead=False,
    ) as dataset:
        for record in dataset:
            node_counts.append(len(record["atoms"]))
    return node_counts


def _protocol_request(cfg: DictConfig) -> dict[str, Any]:
    qm9_data_dir = Path(cfg.evaluation.qm9.training_data_dir).expanduser().resolve()
    mp20_data_dir = Path(cfg.evaluation.mp20.training_data_dir).expanduser().resolve()
    qm9_count = int(cfg.evaluation.qm9.training_num_examples)
    mp20_count = int(cfg.evaluation.mp20.training_num_examples)
    diffusion_batch_size = int(cfg.diffusion_batch_size)
    n_batches = int(cfg.n_batches)
    if diffusion_batch_size <= 0:
        raise ValueError("diffusion_batch_size must be positive.")
    if n_batches != 1:
        raise ValueError(
            "Table-2 grouped sampling requires n_batches=1; groups already encode "
            "the exact number of model calls."
        )
    return {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(cfg.evaluation.seed),
        "qm9": {
            "num_samples": int(cfg.evaluation.qm9.num_samples),
            "training_num_examples": qm9_count,
            "training_data": _dataset_manifest(
                qm9_data_dir,
                "qm9",
                qm9_count,
            ),
        },
        "mp20": {
            "num_samples": int(cfg.evaluation.mp20.num_samples),
            "training_num_examples": mp20_count,
            "training_data": _dataset_manifest(
                mp20_data_dir,
                "mp20",
                mp20_count,
            ),
        },
        "node_count_sampling": {
            "source": "exact_training_partition",
            "algorithm": "torch.multinomial(torch.bincount(counts).float(), replacement=True)",
            "generator": "dedicated_cpu_torch_generator_per_domain",
        },
        "inference_batching": {
            "diffusion_batch_size": diffusion_batch_size,
            "n_batches": n_batches,
            "strategy": "group_equal_node_counts_and_discard_tail_padding",
        },
        "qm9_metrics": {
            # Historical preparation metadata, retained byte-for-byte for
            # version-3 sample reuse. Current scoring is bound separately in
            # metrics.json and has no training-reference-dependent columns.
            "implementation": "zevals.molecules.qm9",
            "bond_perception": "EDM distance thresholds",
            "smiles_remove_hydrogens": False,
            "isomeric_smiles": True,
            "novelty_reference": "EDM full-graph exact 100k training geometries",
        },
    }


def _verify_protocol_artifacts(protocol_dir: Path, manifest: Mapping[str, Any]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"Protocol manifest has no artifact map: {protocol_dir}")
    for name, metadata in artifacts.items():
        if not isinstance(metadata, Mapping):
            raise TypeError(f"Invalid protocol artifact metadata for {name!r}.")
        relative_path = Path(str(metadata["path"]))
        if relative_path.is_absolute():
            raise ValueError(
                f"Protocol artifact path must be relative: {relative_path}"
            )
        protocol_root = protocol_dir.resolve()
        path = (protocol_root / relative_path).resolve()
        if not path.is_relative_to(protocol_root):
            raise ValueError(
                f"Protocol artifact escapes its protocol directory: {relative_path}"
            )
        if not path.is_file():
            raise FileNotFoundError(f"Protocol artifact is missing: {path}")
        actual_hash = _sha256_file(path)
        if actual_hash != metadata["sha256"]:
            raise ValueError(
                f"Protocol artifact hash mismatch for {path}: expected "
                f"{metadata['sha256']}, got {actual_hash}."
            )


def _verify_sampling_plans(
    paths: Mapping[str, Path], manifest: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Verify that frozen grouped inputs match their recorded sampling plans."""
    request = manifest.get("request")
    stored_plans = manifest.get("sampling_plans")
    if not isinstance(request, Mapping) or not isinstance(stored_plans, Mapping):
        raise ValueError("Protocol manifest has no grouped sampling-plan metadata.")
    try:
        diffusion_batch_size = int(
            request["inference_batching"]["diffusion_batch_size"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Protocol inference-batching metadata is incomplete."
        ) from error

    verified: dict[str, dict[str, Any]] = {}
    for domain in DOMAIN_ORDER:
        specifications = _load_input_specifications(paths[f"{domain}_inputs"])
        try:
            expected_num_samples = int(request[domain]["num_samples"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Protocol request has no {domain.upper()} sample count."
            ) from error
        actual = _sampling_plan_metadata(
            domain,
            specifications,
            diffusion_batch_size=diffusion_batch_size,
            expected_num_samples=expected_num_samples,
        )
        if stored_plans.get(domain) != actual:
            raise ValueError(
                f"Frozen {domain.upper()} input manifest does not match its "
                "recorded grouped sampling plan."
            )
        verified[domain] = actual
    return verified


def _verify_protocol_request(cfg: DictConfig, manifest: Mapping[str, Any]) -> None:
    """Reject staged runs whose config no longer matches the frozen protocol."""
    request = manifest.get("request")
    if not isinstance(request, Mapping):
        raise ValueError("Protocol manifest has no request metadata.")
    expected = {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(cfg.evaluation.seed),
        "diffusion_batch_size": int(cfg.diffusion_batch_size),
        "n_batches": int(cfg.n_batches),
        "batching_strategy": "group_equal_node_counts_and_discard_tail_padding",
        "node_count_source": "exact_training_partition",
        "qm9_novelty_reference": ("EDM full-graph exact 100k training geometries"),
        "qm9_num_samples": int(cfg.evaluation.qm9.num_samples),
        "qm9_training_num_examples": int(cfg.evaluation.qm9.training_num_examples),
        "mp20_num_samples": int(cfg.evaluation.mp20.num_samples),
        "mp20_training_num_examples": int(cfg.evaluation.mp20.training_num_examples),
    }
    try:
        actual = {
            "protocol": request["protocol"],
            "protocol_version": request["protocol_version"],
            "seed": int(request["seed"]),
            "diffusion_batch_size": int(
                request["inference_batching"]["diffusion_batch_size"]
            ),
            "n_batches": int(request["inference_batching"]["n_batches"]),
            "batching_strategy": request["inference_batching"]["strategy"],
            "node_count_source": request["node_count_sampling"]["source"],
            "qm9_novelty_reference": request["qm9_metrics"]["novelty_reference"],
            "qm9_num_samples": int(request["qm9"]["num_samples"]),
            "qm9_training_num_examples": int(request["qm9"]["training_num_examples"]),
            "mp20_num_samples": int(request["mp20"]["num_samples"]),
            "mp20_training_num_examples": int(request["mp20"]["training_num_examples"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Protocol manifest request metadata is incomplete.") from error
    mismatches = {
        key: {"protocol": actual[key], "config": value}
        for key, value in expected.items()
        if actual[key] != value
    }
    if mismatches:
        raise ValueError(
            "Current Table-2 config does not match the frozen protocol manifest: "
            f"{mismatches}"
        )


def prepare_protocol(cfg: DictConfig) -> dict[str, Any]:
    """Build or verify benchmark references and immutable sampling manifests."""
    import fcntl

    protocol_dir = Path(cfg.evaluation.protocol_dir).expanduser().resolve()
    protocol_dir.mkdir(parents=True, exist_ok=True)
    # Different checkpoint outputs may share a protocol directory. This lock
    # protects that directory without replacing the active allocation deadline.
    with (protocol_dir / ".preparation.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another evaluation is preparing {protocol_dir}."
            ) from error
        try:
            return _prepare_protocol_locked(cfg, protocol_dir)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _prepare_protocol_locked(cfg: DictConfig, protocol_dir: Path) -> dict[str, Any]:
    """Publish protocol assets with a request marker that survives interruption."""
    paths = _protocol_paths(protocol_dir)
    request = _protocol_request(cfg)
    resume = bool(cfg.evaluation.get("resume", False))
    overwrite = bool(cfg.evaluation.overwrite)
    if resume and overwrite:
        raise ValueError("resume and overwrite cannot both be enabled.")
    if paths["manifest"].is_file() and not overwrite:
        manifest = _read_json(paths["manifest"])
        if manifest.get("request") != request:
            raise ValueError(
                f"Existing protocol at {protocol_dir} does not match this request. "
                "Choose another evaluation.protocol_dir or set evaluation.overwrite=true."
            )
        _verify_protocol_artifacts(protocol_dir, manifest)
        _verify_sampling_plans(paths, manifest)
        LOGGER.info("Verified existing Table-2 protocol assets at %s", protocol_dir)
        return manifest

    request_path = protocol_dir / ".preparation_request.json"
    if request_path.is_file() and not overwrite:
        if _read_json(request_path) != request:
            raise ValueError(
                "Interrupted preparation request changed; choose another "
                "evaluation.protocol_dir."
            )
        if not resume:
            raise FileExistsError(
                "Protocol preparation is incomplete; use evaluation.resume=true "
                f"to recompute its uncommitted assets: {protocol_dir}"
            )
        LOGGER.info("Resuming interrupted Table-2 preparation at %s", protocol_dir)
    else:
        # Atomic marker writes can themselves be interrupted before publishing.
        # Such temporary marker files contain no committed protocol assets.
        existing_assets = [
            path
            for path in protocol_dir.iterdir()
            if path.name != ".preparation.lock"
            and not (
                path.name.startswith(f".{request_path.name}.")
                and path.name.endswith(".tmp")
            )
        ]
        if existing_assets and not overwrite:
            raise FileExistsError(
                f"Protocol directory is non-empty but incomplete: {protocol_dir}; "
                "no matching preparation request marker is available."
            )
        # Bind the complete request, including dataset hashes, before creating
        # any asset that might need recomputation after a four-hour allocation.
        atomic_json(request_path, _sanitize_json(request))

    qm9_data_dir = Path(cfg.evaluation.qm9.training_data_dir).expanduser().resolve()
    mp20_data_dir = Path(cfg.evaluation.mp20.training_data_dir).expanduser().resolve()
    LOGGER.info("Scanning the exact QM9 training split for node counts and novelty.")
    qm9_counts, qm9_reference, invalid_qm9_reference = _scan_qm9_training_data(
        qm9_data_dir
    )
    LOGGER.info("Scanning the exact MP20 training split for node counts.")
    mp20_counts = _scan_mp20_training_data(mp20_data_dir)
    expected_qm9 = int(cfg.evaluation.qm9.training_num_examples)
    expected_mp20 = int(cfg.evaluation.mp20.training_num_examples)
    if len(qm9_counts) != expected_qm9 or len(mp20_counts) != expected_mp20:
        raise ValueError(
            "Training dataset cardinality changed while scanning: "
            f"QM9={len(qm9_counts)} (expected {expected_qm9}), "
            f"MP20={len(mp20_counts)} (expected {expected_mp20})."
        )

    _write_json_gz(paths["qm9_counts"], qm9_counts)
    _write_json_gz(paths["mp20_counts"], mp20_counts)
    _write_json_gz(paths["qm9_reference"], qm9_reference)
    seed = int(cfg.evaluation.seed)
    diffusion_batch_size = int(cfg.diffusion_batch_size)
    input_manifests = {
        "qm9": build_input_manifest(
            "qm9",
            qm9_counts,
            num_samples=int(cfg.evaluation.qm9.num_samples),
            seed=seed,
            diffusion_batch_size=diffusion_batch_size,
        ),
        "mp20": build_input_manifest(
            "mp20",
            mp20_counts,
            num_samples=int(cfg.evaluation.mp20.num_samples),
            seed=seed,
            diffusion_batch_size=diffusion_batch_size,
        ),
    }
    for domain, input_manifest in input_manifests.items():
        _write_json(paths[f"{domain}_inputs"], input_manifest)
    sampling_plans = {
        domain: _sampling_plan_metadata(
            domain,
            _load_input_specifications(paths[f"{domain}_inputs"]),
            diffusion_batch_size=diffusion_batch_size,
            expected_num_samples=int(cfg.evaluation[domain].num_samples),
        )
        for domain in DOMAIN_ORDER
    }
    artifacts = {}
    for name, path in paths.items():
        if name == "manifest":
            continue
        artifacts[name] = {
            "path": path.relative_to(protocol_dir).as_posix(),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "request": request,
        "reference": {
            "qm9_valid_training_geometries": len(qm9_reference),
            "qm9_invalid_training_geometries": invalid_qm9_reference,
            "qm9_unique_canonical_smiles": len(set(qm9_reference)),
        },
        "sampling_plans": sampling_plans,
        "artifacts": artifacts,
    }
    # Publish the marker last; a killed preparation can regenerate uncommitted
    # assets without mistaking a truncated manifest for a completed protocol.
    atomic_json(paths["manifest"], _sanitize_json(manifest))
    LOGGER.info("Wrote frozen Table-2 protocol assets to %s", protocol_dir)
    return manifest


def load_protocol(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Path]]:
    """Load and integrity-check an already prepared protocol directory."""
    protocol_dir = Path(cfg.evaluation.protocol_dir).expanduser().resolve()
    paths = _protocol_paths(protocol_dir)
    if not paths["manifest"].is_file():
        raise FileNotFoundError(
            f"Table-2 protocol is not prepared: {paths['manifest']}. Include the "
            "prepare stage first."
        )
    manifest = _read_json(paths["manifest"])
    _verify_protocol_request(cfg, manifest)
    _verify_protocol_artifacts(protocol_dir, manifest)
    _verify_sampling_plans(paths, manifest)
    return manifest, paths


def _sampling_output_dir(cfg: DictConfig, domain: str) -> Path:
    return Path(cfg.out_dir).expanduser().resolve() / domain


def _sample_artifact_path(domain_dir: Path, relative_path: Any) -> Path:
    """Resolve one manifest artifact while rejecting absolute/path-traversal input."""
    raw_path = Path(str(relative_path))
    if raw_path.is_absolute():
        raise ValueError(f"Sample artifact path must be relative: {raw_path}")
    root = domain_dir.resolve()
    artifact_path = (root / raw_path).resolve()
    if not artifact_path.is_relative_to(root):
        raise ValueError(f"Sample artifact escapes its output directory: {raw_path}")
    return artifact_path


def _sample_artifacts_sha256(domain_dir: Path) -> str:
    """Hash every saved CIF and metadata file in manifest order."""
    manifest_path = domain_dir / "sample_manifest.json"
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, list):
        raise TypeError(f"Sample manifest must contain a JSON list: {manifest_path}")
    digest = hashlib.sha256()
    for row in manifest:
        if not isinstance(row, Mapping):
            raise TypeError(f"Invalid sample-manifest row in {manifest_path}")
        digest.update(str(row["sample_id"]).encode("utf-8"))
        for key in ("cif_path", "metadata_path"):
            artifact_path = _sample_artifact_path(domain_dir, row[key])
            if not artifact_path.is_file():
                raise FileNotFoundError(
                    f"Saved sample artifact is missing: {artifact_path}"
                )
            digest.update(_sha256_file(artifact_path).encode("ascii"))
    return digest.hexdigest()


def _validate_saved_sample_manifest(
    domain_dir: Path,
    *,
    domain: str,
    expected_num_samples: int,
    expected_checkpoint: str | Path,
) -> list[dict[str, Any]]:
    """Validate saved Table-2 artifacts without requiring a CIF parser."""
    manifest_path = domain_dir / "sample_manifest.json"
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, list) or len(manifest) != expected_num_samples:
        actual = (
            len(manifest) if isinstance(manifest, list) else type(manifest).__name__
        )
        raise ValueError(
            f"Expected {expected_num_samples} saved {domain.upper()} samples, "
            f"found {actual} in {manifest_path}."
        )

    checkpoint_path = Path(expected_checkpoint).expanduser().resolve()
    validated: list[dict[str, Any]] = []
    expected_cif_paths: set[Path] = set()
    expected_metadata_paths: set[Path] = set()
    for sample_index, raw_row in enumerate(manifest):
        if not isinstance(raw_row, Mapping):
            raise TypeError(f"Sample-manifest row {sample_index} is not an object.")
        row = dict(raw_row)
        expected_sample_id = f"{domain}-{sample_index:06d}"
        if int(row.get("sample_index", -1)) != sample_index:
            raise ValueError(
                f"{domain.upper()} sample-manifest row {sample_index} has a "
                "noncanonical sample_index."
            )
        if str(row.get("sample_id")) != expected_sample_id:
            raise ValueError(
                f"{domain.upper()} sample-manifest row {sample_index} must have "
                f"sample_id={expected_sample_id!r}."
            )
        cif_path = _sample_artifact_path(domain_dir, row.get("cif_path"))
        metadata_path = _sample_artifact_path(domain_dir, row.get("metadata_path"))
        if not cif_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Saved artifacts are missing for {expected_sample_id}: "
                f"{cif_path}, {metadata_path}"
            )
        if cif_path in expected_cif_paths or metadata_path in expected_metadata_paths:
            raise ValueError(
                f"Duplicate saved-artifact path in {manifest_path}: "
                f"{cif_path}, {metadata_path}"
            )
        expected_cif_paths.add(cif_path)
        expected_metadata_paths.add(metadata_path)
        metadata = _read_json(metadata_path)
        if not isinstance(metadata, Mapping):
            raise TypeError(f"Saved metadata must be an object: {metadata_path}")
        saved_checkpoint = metadata.get("ckpt_path")
        if saved_checkpoint in (None, ""):
            raise ValueError(f"{metadata_path} does not identify its checkpoint.")
        if Path(str(saved_checkpoint)).expanduser().resolve() != checkpoint_path:
            raise ValueError(
                f"{expected_sample_id} was not generated from {checkpoint_path}."
            )
        sampling = metadata.get("table2_sampling")
        if not isinstance(sampling, Mapping):
            raise ValueError(f"{metadata_path} has no Table-2 sampling provenance.")
        provenance_pairs = {
            "canonical_sample_id": expected_sample_id,
            "canonical_sample_index": sample_index,
            "sampling_group_id": row.get("sampling_group_id"),
            "sampling_group_model_index": row.get("sampling_group_model_index"),
            "sampling_plan_sha256": row.get("sampling_plan_sha256"),
        }
        mismatches = {
            key: {"metadata": sampling.get(key), "manifest": expected_value}
            for key, expected_value in provenance_pairs.items()
            if sampling.get(key) != expected_value
        }
        if mismatches:
            raise ValueError(
                f"{expected_sample_id} sampling provenance does not match its "
                f"manifest row: {mismatches}"
            )
        row["resolved_cif_path"] = str(cif_path)
        row["resolved_metadata_path"] = str(metadata_path)
        validated.append(row)

    discovered_cifs = {
        path.resolve()
        for pattern in ("*.cif", "*.cif.gz")
        for path in (domain_dir / "samples").rglob(pattern)
        if path.is_file()
    }
    if discovered_cifs != expected_cif_paths:
        raise ValueError(
            f"{domain.upper()} samples directory does not exactly match the "
            "sample manifest; refusing to score added or omitted CIFs."
        )
    return validated


def _check_sample_destination(cfg: DictConfig, domain: str) -> None:
    domain_dir = _sampling_output_dir(cfg, domain)
    samples_dir = domain_dir / "samples"
    if (
        samples_dir.exists()
        and any(samples_dir.iterdir())
        and not bool(cfg.evaluation.overwrite)
    ):
        raise FileExistsError(
            f"Sample directory is not empty: {samples_dir}. Omit the sample stage "
            "to rescore it, set evaluation.overwrite=true, or choose a new out_dir."
        )


def _checkpoint_provenance(cfg: DictConfig) -> dict[str, Any]:
    checkpoint_path = Path(cfg.ckpt_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    metadata: dict[str, Any] = {}
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        metadata = {
            "checkpoint_kind": checkpoint.get("checkpoint_kind"),
            "current_epoch": checkpoint.get("current_epoch"),
            "global_step": checkpoint.get("global_step"),
        }
        train_cfg = checkpoint.get("train_cfg")
        if train_cfg is not None:
            metadata["training_seed"] = OmegaConf.select(train_cfg, "seed")
            sampler = OmegaConf.select(train_cfg, "model.net.inference_sampler")
            if sampler is not None:
                metadata["checkpoint_sampler"] = OmegaConf.to_container(
                    sampler, resolve=True
                )
        del checkpoint
    except Exception as error:
        metadata["metadata_read_error"] = repr(error)

    result = {
        "path": str(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        "metadata": metadata,
    }
    if bool(cfg.evaluation.hash_checkpoint):
        result["sha256"] = _sha256_file(checkpoint_path)
    return result


def _repository_provenance() -> dict[str, Any]:
    repository_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()

    def git(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=all")
    patch = subprocess.run(
        ["git", "-C", str(repository_root), "diff", "--binary", "HEAD", "--"],
        check=False,
        capture_output=True,
    ).stdout
    untracked_process = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=False,
        capture_output=True,
    )
    untracked_files = []
    if untracked_process.returncode == 0:
        for encoded_path in untracked_process.stdout.split(b"\0"):
            if not encoded_path:
                continue
            relative_path = os.fsdecode(encoded_path)
            path = (repository_root / relative_path).resolve()
            if path.is_file() and path.is_relative_to(repository_root):
                untracked_files.append(
                    {
                        "path": relative_path,
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                )
    return {
        "root": str(repository_root),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
        "tracked_patch_size_bytes": len(patch),
        "tracked_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "untracked_files": untracked_files,
    }


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for distribution in (
        "torch",
        "numpy",
        "scipy",
        "rdkit",
        "pymatgen",
        "ase",
        "biotite",
        "hydra-core",
        "omegaconf",
        "lightning",
        "atomworks",
        "rc-foundry",
        "zatom-nucleus",
        "zatom2",
        "zevals",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _software_provenance() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "packages": _package_versions(),
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "numerical_settings": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_config": torch.__config__.show(),
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }


def _scoring_provenance(module_names: Sequence[str]) -> dict[str, Any]:
    """Fingerprint the implementation and package versions used for scoring."""
    modules = {}
    for module_name in module_names:
        module = importlib.import_module(module_name)
        source = getattr(module, "__file__", None)
        if source is None:
            raise RuntimeError(f"Cannot locate scoring module source: {module_name}")
        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Scoring module source does not exist: {module_name} -> {source_path}"
            )
        modules[module_name] = {
            "sha256": _sha256_file(source_path),
            "size_bytes": source_path.stat().st_size,
        }
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": _package_versions(),
        "modules": modules,
    }


def _qm9_scoring_provenance() -> dict[str, Any]:
    """Bind the Table 2 molecule scorer and its isolated chemistry runtime."""
    import posebusters
    from openbabel import openbabel

    provenance = _scoring_provenance(_QM9_SCORING_MODULES)
    for distribution in ("posebusters", "openbabel-wheel"):
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        provenance["packages"][distribution] = version
    provenance["openbabel_version"] = openbabel.OBReleaseVersion()
    # PoseBusters delegates to multiple modules and molecule/InChI helpers.
    # Hash the whole implementation and configs rather than only __init__.py.
    root = Path(posebusters.__file__).resolve().parent
    provenance["posebusters_files"] = {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".yml", ".yaml"}
    }
    return provenance


def _retain_planned_samples(
    domain: str,
    samples: Sequence[Any],
    specifications: Mapping[str, Mapping[str, Any]],
    *,
    diffusion_batch_size: int,
    expected_num_samples: int,
    sampling_plan: Mapping[str, Any] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Apply a grouped input's retention plan and assign canonical sample IDs."""
    plan = sampling_plan or _sampling_plan_metadata(
        domain,
        specifications,
        diffusion_batch_size=diffusion_batch_size,
        expected_num_samples=expected_num_samples,
    )
    expected_generated = len(specifications) * diffusion_batch_size
    if len(samples) != expected_generated:
        raise ValueError(
            f"Expected {expected_generated} generated {domain.upper()} "
            f"replicas before padding removal, got {len(samples)}."
        )

    samples_by_group: dict[str, dict[int, Any]] = defaultdict(dict)
    for sample in samples:
        group_id = str(sample.target_id)
        model_index = int(sample.model_index)
        if group_id not in specifications:
            raise ValueError(f"Unexpected {domain.upper()} sampling group: {group_id}")
        if int(sample.batch_index) != 0:
            raise ValueError(
                f"{group_id}: grouped Table-2 sampling requires batch_index=0."
            )
        if model_index in samples_by_group[group_id]:
            raise ValueError(f"{group_id}: duplicate model output index {model_index}.")
        samples_by_group[group_id][model_index] = sample

    retained: list[Any] = []
    for group_id, specification in specifications.items():
        model_outputs = samples_by_group.get(group_id, {})
        expected_model_indices = set(range(diffusion_batch_size))
        if set(model_outputs) != expected_model_indices:
            raise ValueError(
                f"{group_id}: expected model indices "
                f"{sorted(expected_model_indices)}, found {sorted(model_outputs)}."
            )
        extra = specification["extra"]
        sample_indices = [int(index) for index in extra["table2_sample_indices"]]
        retained_model_indices = [
            int(index) for index in extra["table2_retained_model_indices"]
        ]
        for sample_index, model_index in zip(
            sample_indices, retained_model_indices, strict=True
        ):
            sample = model_outputs[model_index]
            source_output_id = str(sample.sample_id)
            canonical_sample_id = f"{domain}-{sample_index:06d}"
            canonical_specification = dict(specification)
            canonical_extra = dict(extra)
            canonical_extra.update(
                {
                    "table2_sample_index": sample_index,
                    "table2_sampling_group_id": group_id,
                    "table2_sampling_group_model_index": model_index,
                }
            )
            canonical_specification["extra"] = canonical_extra
            sample.sample_index = sample_index
            sample.sample_id = canonical_sample_id
            sample.specification = canonical_specification
            sample.output.example_id = canonical_sample_id
            metadata = getattr(sample.output, "metadata", None)
            sample.output.metadata = (
                dict(metadata) if isinstance(metadata, Mapping) else {}
            )
            # Reloading saved samples treats metadata.json as authoritative for
            # the conditioning specification, so replace RFD3's collated form
            # with this plain per-output canonical/group specification.
            sample.output.metadata["specification"] = canonical_specification
            sample.output.metadata["table2_sampling"] = {
                "sampling_group_id": group_id,
                "sampling_group_model_index": model_index,
                "source_output_id": source_output_id,
                "canonical_sample_index": sample_index,
                "canonical_sample_id": canonical_sample_id,
                "planned_node_count": int(extra["table2_node_count"]),
                "sampling_plan_sha256": plan["sha256"],
            }
            retained.append(sample)

    retained.sort(key=lambda sample: sample.sample_index)
    actual_indices = [int(sample.sample_index) for sample in retained]
    expected_indices = sorted(
        int(index)
        for spec in specifications.values()
        for index in spec["extra"]["table2_sample_indices"]
    )
    if actual_indices != expected_indices:
        raise ValueError(
            f"Retained {domain.upper()} outputs do not cover the canonical sample "
            f"indices 0..{expected_num_samples - 1}."
        )
    canonical_ids = [sample.sample_id for sample in retained]
    if len(set(canonical_ids)) != len(canonical_ids):
        raise ValueError(f"Retained {domain.upper()} outputs have duplicate IDs.")

    stats = dict(plan)
    stats["num_generated_outputs"] = len(samples)
    stats["num_retained_outputs"] = len(retained)
    stats["num_discarded_padding_outputs"] = len(samples) - len(retained)
    return retained, stats


def sample_checkpoint(
    cfg: DictConfig, protocol_manifest: Mapping[str, Any], paths: Mapping[str, Path]
) -> tuple[dict[str, Any], bool]:
    """Commit each independently seeded group, retaining progress across jobs."""
    if int(cfg.get("num_nodes", 1)) != 1 or int(cfg.get("devices_per_node", 1)) != 1:
        raise ValueError("Resumable Table-2 sampling requires one node and one GPU.")
    resume = bool(cfg.evaluation.get("resume", False))
    if resume and bool(cfg.evaluation.overwrite):
        raise ValueError("resume and overwrite cannot both be enabled.")
    output_dir = Path(cfg.out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "sampling_progress.json"
    if not progress_path.exists():
        for domain in DOMAIN_ORDER:
            _check_sample_destination(cfg, domain)
    checkpoint = _checkpoint_provenance(cfg)
    software = _software_provenance()
    sampling_config = OmegaConf.to_container(cfg, resolve=True)
    # Paths, selected stages, walltime and scoring options do not alter sampling.
    for key in ("evaluation", "inputs", "json_keys_subset", "out_dir", "ckpt_path"):
        sampling_config.pop(key, None)
    contract = {
        "protocol_version": PROTOCOL_VERSION,
        "diffusion_seed": int(cfg.evaluation.seed),
        "protocol_sha256": _sha256_file(paths["manifest"]),
        "input_sha256": {
            domain: _sha256_file(paths[f"{domain}_inputs"]) for domain in DOMAIN_ORDER
        },
        "checkpoint": checkpoint,
        "sampling_config": sampling_config,
        "implementation": _scoring_provenance(
            ("zatom2.table2_evaluation", "zatom2.table2_resume", "zatom2.run_inference")
        ),
        "source_packages": sampling_source_hashes(),
        "runtime": {
            key: value
            for key, value in software.items()
            if key not in {"hostname", "platform"}
        },
        "rng": "sha256(zatom2-table2-v3:seed:domain:group_id)[:4], big-endian",
    }
    contract = _sanitize_json(contract)
    contract_hash = bind_contract(progress_path, contract, resume=resume)
    engine = None
    summaries: dict[str, Any] = {}
    diffusion_batch_size = int(cfg.diffusion_batch_size)
    try:
        for domain in DOMAIN_ORDER:
            input_path = paths[f"{domain}_inputs"]
            specifications = _load_input_specifications(input_path)
            plan = protocol_manifest["sampling_plans"][domain]
            expected = int(protocol_manifest["request"][domain]["num_samples"])
            domain_dir = _sampling_output_dir(cfg, domain)
            domain_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for group_id, specification in specifications.items():
                marker = domain_dir / ".sampling_progress" / f"{group_id}.json"
                if marker.exists():
                    rows.extend(
                        verify_group(
                            marker,
                            domain_dir,
                            contract_sha256=contract_hash,
                            group_id=group_id,
                        )
                    )
                    continue
                remaining_seconds()
                if engine is None:
                    from zatom2.run_inference import build_inference_engine

                    cfg.inputs = str(input_path)
                    cfg.json_keys_subset = [group_id]
                    engine = build_inference_engine(cfg)
                    # Model construction consumes RNG; reset only after initialize.
                    engine.initialize()
                    if int(engine.trainer.fabric.world_size) != 1:
                        raise ValueError("Table-2 sampling requires a single rank.")
                engine.json_keys_subset = [group_id]
                engine.global_prefix = ""
                seed = group_seed(int(cfg.evaluation.seed), domain, group_id)
                engine.seed = seed
                from lightning.fabric import seed_everything

                seed_everything(seed, workers=True, verbose=False)
                outputs = engine.run(inputs=str(input_path), n_batches=1, out_dir=None)
                if not outputs:
                    raise RuntimeError(f"{group_id}: inference produced no outputs.")
                generated = _flatten_outputs(
                    outputs,
                    {group_id: specification},
                    input_path=input_path,
                    global_prefix="",
                    n_batches=1,
                )
                samples, _ = _retain_planned_samples(
                    domain,
                    generated,
                    {group_id: specification},
                    diffusion_batch_size=diffusion_batch_size,
                    expected_num_samples=expected,
                    sampling_plan=plan,
                )
                for sample in samples:
                    sample.output.metadata["table2_sampling"]["group_seed"] = seed
                group_rows = _dump_samples(samples, domain_dir)
                for row, sample in zip(group_rows, samples, strict=True):
                    row.update(
                        {
                            "sampling_group_id": group_id,
                            "sampling_group_model_index": sample.model_index,
                            "planned_node_count": int(
                                specification["extra"]["table2_node_count"]
                            ),
                            "sampling_plan_sha256": plan["sha256"],
                        }
                    )
                commit_group(
                    marker,
                    domain_dir,
                    group_rows,
                    contract_sha256=contract_hash,
                    group_id=group_id,
                    seed=seed,
                )
                rows.extend(group_rows)
                del outputs, generated, samples
                LOGGER.info(
                    "%s: committed %d/%d attempted samples", domain, len(rows), expected
                )
            rows.sort(key=lambda row: row["sample_index"])
            atomic_json(domain_dir / "sample_manifest.json", rows)
            _validate_saved_sample_manifest(
                domain_dir,
                domain=domain,
                expected_num_samples=expected,
                expected_checkpoint=cfg.ckpt_path,
            )
            summaries[domain] = {
                "num_samples": len(rows),
                "sampling": plan,
                "sampling_plan_sha256": plan["sha256"],
                "sample_manifest": str(domain_dir / "sample_manifest.json"),
                "sample_manifest_sha256": _sha256_file(
                    domain_dir / "sample_manifest.json"
                ),
                "sample_artifacts_sha256": _sample_artifacts_sha256(domain_dir),
                "input_manifest": str(input_path),
                "input_manifest_sha256": _sha256_file(input_path),
                "world_size": 1,
                "domain_seed": int(cfg.evaluation.seed),
                "group_seed_formula": contract["rng"],
                "sampling_progress_sha256": contract_hash,
                "reproducibility_scope": "fixed software, GPU type and per-group seed",
            }
        run_manifest_path = output_dir / "run_manifest.json"
        if resume and run_manifest_path.exists():
            saved = _read_json(run_manifest_path)
            if (
                saved.get("domains") != summaries
                or saved.get("checkpoint") != checkpoint
            ):
                raise ValueError(
                    "Completed sampling manifest differs from verified progress."
                )
            return saved, True
        OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)
        run_manifest = {
            "protocol": {
                "name": PROTOCOL_NAME,
                "version": PROTOCOL_VERSION,
                "manifest": str(paths["manifest"]),
                "manifest_sha256": _sha256_file(paths["manifest"]),
                "canonical_runtime": not _protocol_conformance_mismatches(cfg),
                "runtime_mismatches": _protocol_conformance_mismatches(cfg),
                "noncanonical_override_allowed": bool(
                    cfg.evaluation.get("allow_noncanonical_protocol", False)
                ),
            },
            "checkpoint": checkpoint,
            "sampler": OmegaConf.to_container(cfg.inference_sampler, resolve=True),
            "ema_weights": bool(cfg.use_ema_weights),
            "seed": int(cfg.seed),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "domains": summaries,
            "repository": _repository_provenance(),
            "software": software,
            "command": sys.argv,
        }
        run_manifest = _sanitize_json(run_manifest)
        atomic_json(run_manifest_path, run_manifest)
        return run_manifest, True
    finally:
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _compute_qm9_saved_metrics(
    cfg: DictConfig,
    *,
    expected_num_samples: int,
) -> dict[str, Any]:
    """Recompute Zatom-1 Table 2 metrics from every saved QM9 attempt."""
    from biotite import InvalidFileError
    from nucleus.inference_input_parsing import inference_load_preserving_hydrogens
    from nucleus.utils.atom_array import _atomic_numbers_from_atom_array
    from zevals.molecules.qm9_zatom1 import compute_qm9_zatom1_generation_metrics

    domain_dir = _sampling_output_dir(cfg, "qm9")
    rows = _validate_saved_sample_manifest(
        domain_dir,
        domain="qm9",
        expected_num_samples=expected_num_samples,
        expected_checkpoint=cfg.ckpt_path,
    )
    molecule_arrays = []
    for row in rows:
        try:
            atom_array = inference_load_preserving_hydrogens(
                Path(row["resolved_cif_path"])
            )["atom_array"]
        except (
            InvalidFileError,
            ValueError,
            KeyError,
            IndexError,
            EOFError,
            gzip.BadGzipFile,
        ) as error:
            # A malformed generated CIF is an attempted molecule. Missing
            # files, permission failures and unavailable dependencies abort.
            LOGGER.warning(
                "Could not parse QM9 attempt %s: %s", row["sample_id"], error
            )
            molecule_arrays.append(None)
            continue
        coordinates = np.asarray(atom_array.coord, dtype=np.float64)
        atomic_numbers = _atomic_numbers_from_atom_array(atom_array, len(coordinates))
        if atomic_numbers is None:
            atomic_numbers = np.zeros(len(coordinates), dtype=np.int64)
        molecule_arrays.append((atomic_numbers, coordinates))
    return compute_qm9_zatom1_generation_metrics(molecule_arrays)


def score_qm9(
    cfg: DictConfig, protocol_manifest: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    """Score saved QM9 outputs with the nine Zatom-1 Table 2 metrics."""
    domain_dir = _sampling_output_dir(cfg, "qm9")
    expected = int(protocol_manifest["request"]["qm9"]["num_samples"])
    metrics = _compute_qm9_saved_metrics(
        cfg,
        expected_num_samples=expected,
    )
    metrics["evaluation"] = {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "qm9_scoring_revision": QM9_SCORING_REVISION,
        "num_attempted": expected,
        "sample_manifest_sha256": _sha256_file(domain_dir / "sample_manifest.json"),
        "sample_artifacts_sha256": _sample_artifacts_sha256(domain_dir),
        "scoring_provenance": _qm9_scoring_provenance(),
    }
    atomic_json(domain_dir / "metrics.json", _sanitize_json(metrics))
    LOGGER.info("QM9 metrics written to %s", domain_dir / "metrics.json")
    return metrics


def _native_lemat_rates(metrics: Mapping[str, float]) -> dict[str, float | None]:
    key_options = {
        "overall_valid": (
            "validity/overall_validity_ratio",
            "validity_filtering/validity_rate",
        ),
        "unique": ("uniqueness/uniqueness_ratio", "uniqueness/uniqueness_score"),
        "novel": ("novelty/novelty_ratio", "novelty/novelty_score"),
        "stable": ("stability/stable_ratio",),
        "sun": ("sun/sun_rate",),
        "metasun": ("sun/msun_rate",),
    }
    return {
        name: next((float(metrics[key]) for key in keys if key in metrics), None)
        for name, keys in key_options.items()
    }


def _lemat_component_group(
    metrics: Mapping[str, float], prefix: str
) -> dict[str, float]:
    return {
        key.removeprefix(f"{prefix}/"): float(value)
        for key, value in metrics.items()
        if key.startswith(f"{prefix}/")
    }


_LEMAT_ARTIFACT_RELATIVE_PATHS = {
    "raw": "lematgenbench_raw.json",
    "metrics": "lematgenbench_metrics.json",
    "manifest": "lematgenbench_manifest.json",
    "stdout": "lematgenbench.stdout.log",
    "stderr": "lematgenbench.stderr.log",
    "lock": "lematgenbench.uv.lock",
    "cif_manifest": "cif_paths.txt",
    "mace_stdout": "lematgenbench_mace_preflight.stdout.log",
    "mace_stderr": "lematgenbench_mace_preflight.stderr.log",
    "runtime": "lematgenbench_runtime.json",
    "runtime_stdout": "lematgenbench_runtime_preflight.stdout.log",
    "runtime_stderr": "lematgenbench_runtime_preflight.stderr.log",
    "reference_audit": "lematgenbench_reference_pin_audit.jsonl",
    "reference_stdout": "lematgenbench_reference_pin_preflight.stdout.log",
    "reference_stderr": "lematgenbench_reference_pin_preflight.stderr.log",
    "reference_shim": "lematgenbench_reference_pin_shim/sitecustomize.py",
}


def _effective_lemat_run_name(
    base_name: str,
    *,
    domain_dir: Path,
    sample_manifest_sha256: str,
    sample_artifacts_sha256: str,
) -> str:
    """Create a stable upstream name that cannot collide across output trees."""
    identity = _sha256_json(
        {
            "domain_dir": str(domain_dir.resolve()),
            "sample_manifest_sha256": sample_manifest_sha256,
            "sample_artifacts_sha256": sample_artifacts_sha256,
        }
    )[:16]
    return f"{base_name}-{identity}"


def _expected_lemat_source_inputs(
    validated_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    records = [
        {
            "source_path": str(Path(str(row["resolved_cif_path"])).resolve()),
            "sha256": _sha256_file(Path(str(row["resolved_cif_path"]))),
        }
        for row in validated_rows
    ]
    return sorted(records, key=lambda record: record["source_path"])


def _validate_lemat_artifact_hashes(
    lemat_dir: Path, artifact_sha256: Mapping[str, Any]
) -> dict[str, str]:
    """Verify every canonical adapter artifact against its archived digest."""
    if not isinstance(artifact_sha256, Mapping):
        raise ValueError("LeMat-GenBench provenance has no artifact digest map.")
    relative_paths = dict(_LEMAT_ARTIFACT_RELATIVE_PATHS)
    if "resume_runtime" in artifact_sha256:
        relative_paths["resume_runtime"] = (
            "lematgenbench_reference_pin_shim/_lemat_resume_runtime.py"
        )
    required_keys = set(relative_paths)
    actual_keys = set(artifact_sha256)
    if actual_keys != required_keys:
        raise ValueError(
            "LeMat-GenBench artifact digest map is incomplete or unexpected: "
            f"missing={sorted(required_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - required_keys)}."
        )
    verified: dict[str, str] = {}
    for key, relative_path in relative_paths.items():
        artifact_path = lemat_dir / relative_path
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"LeMat-GenBench provenance artifact is missing: {artifact_path}"
            )
        actual_sha256 = _sha256_file(artifact_path)
        expected_sha256 = str(artifact_sha256[key])
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"LeMat-GenBench {key} artifact changed after evaluation: "
                f"expected {expected_sha256}, got {actual_sha256}."
            )
        verified[key] = actual_sha256
    return verified


def _validate_lemat_manifest_contract(
    adapter_manifest: Mapping[str, Any],
    *,
    lemat_dir: Path,
    lemat_cfg: DictConfig,
    effective_run_name: str,
    expected_inputs: Sequence[Mapping[str, str]],
    reference_revisions: Mapping[str, str],
    oracle_identities: Mapping[str, Any],
) -> dict[str, int]:
    """Require the complete manuscript LeMat contract for fresh and reused runs."""
    expected_count = len(expected_inputs)
    expected_orb_sha256 = str(lemat_cfg.expected_orb_checkpoint_sha256)
    expected_mace_sha256 = str(lemat_cfg.expected_mace_checkpoint_sha256)
    expected_uma_sha256 = str(lemat_cfg.expected_uma_checkpoint_sha256)
    expected_top_level = {
        "adapter": "zevals.materials.lematgenbench",
        "upstream_commit": str(lemat_cfg.expected_commit),
        "expected_commit": str(lemat_cfg.expected_commit),
        "upstream_uv_lock_sha256": str(lemat_cfg.expected_lock_sha256),
        "expected_uv_lock_sha256": str(lemat_cfg.expected_lock_sha256),
        "config_name": str(lemat_cfg.config_name),
        "families": list(lemat_cfg.families),
        "fingerprint_method": str(lemat_cfg.fingerprint_method),
        "parse_failures_as_invalid": True,
        "require_clean_repo": True,
        "run_name": effective_run_name,
        "family_errors": {},
        "input_integrity_errors": {},
        "local_reference_changes_during_run": {},
        "local_reference_integrity_errors": {},
    }
    mismatches = {
        key: {"found": adapter_manifest.get(key), "expected": expected}
        for key, expected in expected_top_level.items()
        if adapter_manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "LeMat-GenBench manifest does not match the manuscript protocol: "
            f"{mismatches}"
        )

    input_count = adapter_manifest.get("input_count")
    if not isinstance(input_count, Mapping):
        raise ValueError("LeMat-GenBench manifest has no input accounting.")
    loaded = input_count.get("upstream_loaded_structures")
    if not isinstance(loaded, int) or isinstance(loaded, bool):
        raise ValueError("LeMat-GenBench loaded-structure count is not an integer.")
    if not 0 <= loaded <= expected_count:
        raise ValueError(
            "LeMat-GenBench loaded-structure count is outside the staged "
            f"denominator: {loaded} versus {expected_count}."
        )
    dropped = expected_count - loaded
    expected_input_accounting = {
        "expected_cifs": expected_count,
        "upstream_loaded_structures": loaded,
        "dropped_during_cif_parsing": dropped,
        "exact_match": dropped == 0,
        "accepted_as_invalid_attempts": dropped > 0,
    }
    accounting_mismatches = {
        key: {"found": input_count.get(key), "expected": expected}
        for key, expected in expected_input_accounting.items()
        if input_count.get(key) != expected
    }
    error = input_count.get("error")
    if (dropped == 0 and error is not None) or (
        dropped > 0 and not isinstance(error, str)
    ):
        accounting_mismatches["error"] = {
            "found": error,
            "expected": None if dropped == 0 else "CIF parse-drop diagnostic",
        }
    if accounting_mismatches:
        raise ValueError(
            "LeMat-GenBench input accounting is inconsistent with the frozen "
            f"attempt denominator: {accounting_mismatches}"
        )

    input_cifs = adapter_manifest.get("input_cifs")
    if not isinstance(input_cifs, list) or len(input_cifs) != expected_count:
        raise ValueError(
            "LeMat-GenBench manifest does not contain exactly the expected CIF "
            f"inputs ({expected_count})."
        )
    staged_root = (lemat_dir / "input_cifs").resolve()
    for index, (record, expected_source) in enumerate(
        zip(input_cifs, expected_inputs, strict=True)
    ):
        if not isinstance(record, Mapping):
            raise TypeError(f"LeMat-GenBench input record {index} is not an object.")
        staged_path = Path(str(record.get("path"))).resolve()
        if not staged_path.is_relative_to(staged_root) or not staged_path.is_file():
            raise ValueError(
                f"LeMat-GenBench staged CIF {index} is missing or outside "
                f"{staged_root}: {staged_path}"
            )
        expected_record = {
            "source_path": expected_source["source_path"],
            "sha256": expected_source["sha256"],
            "post_run_sha256": expected_source["sha256"],
        }
        record_mismatches = {
            key: {"found": record.get(key), "expected": expected}
            for key, expected in expected_record.items()
            if record.get(key) != expected
        }
        staged_sha256 = _sha256_file(staged_path)
        if staged_sha256 != expected_source["sha256"]:
            record_mismatches["current_staged_sha256"] = {
                "found": staged_sha256,
                "expected": expected_source["sha256"],
            }
        if record_mismatches:
            raise ValueError(
                f"LeMat-GenBench input {index} is not bound to the current saved "
                f"sample: {record_mismatches}"
            )

    runtime = adapter_manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("LeMat-GenBench manifest has no runtime provenance.")
    requested_visibility = lemat_cfg.cuda_visible_devices
    if requested_visibility is not None:
        requested_visibility = str(requested_visibility)
    expected_runtime = {
        "device_policy": str(lemat_cfg.runtime_device),
        "requested_cuda_visible_devices": requested_visibility,
        "inherited_pythonpath_sanitized": True,
    }
    runtime_mismatches = {
        key: {"found": runtime.get(key), "expected": expected}
        for key, expected in expected_runtime.items()
        if runtime.get(key) != expected
    }
    runtime_provenance = runtime.get("provenance")
    if not isinstance(runtime_provenance, Mapping):
        runtime_mismatches["provenance"] = {
            "found": runtime_provenance,
            "expected": "mapping",
        }
    else:
        if runtime_provenance.get("requested_device_policy") != str(
            lemat_cfg.runtime_device
        ):
            runtime_mismatches["provenance.requested_device_policy"] = {
                "found": runtime_provenance.get("requested_device_policy"),
                "expected": str(lemat_cfg.runtime_device),
            }
        if str(lemat_cfg.runtime_device) == "cuda" and (
            runtime_provenance.get("cuda_available") is not True
            or not isinstance(runtime_provenance.get("cuda_device_count"), int)
            or runtime_provenance["cuda_device_count"] < 1
        ):
            runtime_mismatches["provenance.cuda"] = {
                "found": {
                    "available": runtime_provenance.get("cuda_available"),
                    "device_count": runtime_provenance.get("cuda_device_count"),
                },
                "expected": "at least one visible CUDA device",
            }
    if runtime_mismatches:
        raise ValueError(
            "LeMat-GenBench runtime does not match the frozen device policy: "
            f"{runtime_mismatches}"
        )

    mace = adapter_manifest.get("mace_checkpoint")
    if not isinstance(mace, Mapping):
        raise ValueError("LeMat-GenBench manifest has no MACE checkpoint audit.")
    expected_mace = {
        "verification_enabled": True,
        "expected_sha256": expected_mace_sha256,
        "sha256": expected_mace_sha256,
        "post_run_sha256": expected_mace_sha256,
        "post_run_error": None,
        "fallback_to_small_detected": False,
    }
    mace_mismatches = {
        key: {"found": mace.get(key), "expected": expected}
        for key, expected in expected_mace.items()
        if mace.get(key) != expected
    }
    if mace_mismatches:
        raise ValueError(
            "LeMat-GenBench MACE checkpoint audit is not canonical: "
            f"{mace_mismatches}"
        )

    oracle_protocol = adapter_manifest.get("oracle_protocol")
    if not isinstance(oracle_protocol, Mapping):
        raise ValueError("LeMat-GenBench manifest has no oracle protocol audit.")
    expected_oracle = {
        "identities": dict(oracle_identities),
        "verify_orb_checkpoint": True,
        "expected_orb_checkpoint_sha256": expected_orb_sha256,
        "verify_uma_checkpoint": True,
        "expected_uma_checkpoint_sha256": expected_uma_sha256,
        "runtime_checkpoint_verification_effective": {
            "orb": True,
            "mace": True,
            "uma": True,
        },
        "require_complete_coverage": True,
        "coverage_error": None,
    }
    oracle_mismatches = {
        key: {"found": oracle_protocol.get(key), "expected": expected}
        for key, expected in expected_oracle.items()
        if oracle_protocol.get(key) != expected
    }
    coverage = oracle_protocol.get("coverage")
    if not isinstance(coverage, Mapping):
        oracle_mismatches["coverage"] = {
            "found": coverage,
            "expected": "mapping",
        }
    else:
        valid_structures = coverage.get("valid_structures")
        if (
            not isinstance(valid_structures, int)
            or isinstance(valid_structures, bool)
            or valid_structures <= 0
            or coverage.get("status") != "complete"
            or coverage.get("all_three_complete") is not True
        ):
            oracle_mismatches["coverage"] = {
                "found": coverage,
                "expected": "complete ORB/MACE/UMA coverage for a nonempty valid set",
            }
    if oracle_mismatches:
        raise ValueError(
            "LeMat-GenBench oracle protocol is not canonical: " f"{oracle_mismatches}"
        )

    references = adapter_manifest.get("reference_revision_pins")
    if not isinstance(references, Mapping):
        raise ValueError("LeMat-GenBench manifest has no reference-pin audit.")
    expected_references = {
        "enforced": True,
        "revisions": dict(reference_revisions),
        "expected_oracle_digests": {
            "mace": expected_mace_sha256,
            "orb": expected_orb_sha256,
            "uma": expected_uma_sha256,
        },
        "error": None,
    }
    reference_mismatches = {
        key: {"found": references.get(key), "expected": expected}
        for key, expected in expected_references.items()
        if references.get(key) != expected
    }
    runtime_access = references.get("runtime_access")
    if not isinstance(runtime_access, Mapping) or (
        runtime_access.get("verification") != "enabled"
        or runtime_access.get("missing_expected_repositories") != []
        or runtime_access.get("missing_oracle_artifacts") != []
        or runtime_access.get("mismatched_oracle_artifacts") != []
    ):
        reference_mismatches["runtime_access"] = {
            "found": runtime_access,
            "expected": "complete pinned-reference and oracle-artifact audit",
        }
    if reference_mismatches:
        raise ValueError(
            "LeMat-GenBench reference revisions are not fully enforced: "
            f"{reference_mismatches}"
        )

    local_references = adapter_manifest.get("local_reference_artifacts")
    if not isinstance(local_references, Mapping):
        raise ValueError("LeMat-GenBench manifest has no local reference inventory.")
    if adapter_manifest.get("local_reference_artifacts_post_run") != local_references:
        raise ValueError(
            "LeMat-GenBench local reference inventory changed during evaluation."
        )
    local_hull_overrides = sorted(
        path
        for path, record in local_references.items()
        if str(path).startswith("data/convex_hulls/")
        and isinstance(record, Mapping)
        and bool(record.get("exists"))
    )
    if local_hull_overrides:
        raise ValueError(
            "LeMat-GenBench used unaudited local hull overrides: "
            f"{local_hull_overrides}"
        )
    return {"attempted": expected_count, "loaded": loaded, "dropped": dropped}


def _material_scoring_input(
    cfg: DictConfig, rows: Sequence[Mapping[str, Any]], *, run: bool
) -> tuple[Path, list[dict[str, str]], dict[str, Any] | None]:
    """Resolve raw or audited CHGNet-relaxed CIFs without changing attempts."""
    domain_dir = _sampling_output_dir(cfg, "mp20")
    input_dir = domain_dir / "samples"
    if not OmegaConf.select(cfg, "evaluation.mp20.prerelax.enabled", default=False):
        return input_dir, _expected_lemat_source_inputs(rows), None
    from zevals.materials.chgnet_prerelax import (
        run_chgnet_prerelaxation,
        validate_chgnet_prerelaxation,
    )

    options = cfg.evaluation.mp20.prerelax
    kwargs = {
        "output_dir": domain_dir / "chgnet_prerelax",
        "fmax": float(options.fmax),
        "steps": int(options.steps),
        "relax_cell": bool(options.relax_cell),
        "device": str(options.device),
        "seed": int(options.seed),
    }
    if run:
        result = run_chgnet_prerelaxation(
            input_dir,
            **kwargs,
            runner_command=list(options.runner_command),
            resume=bool(cfg.evaluation.get("resume", False)),
            timeout_seconds=remaining_seconds(),
        )
        if not result.complete:
            raise EvaluationPaused("CHGNet pre-relaxation saved partial progress.")
    else:
        result = validate_chgnet_prerelaxation(input_dir, **kwargs)
    expected_inputs = [
        {
            "source_path": str(
                (result.cif_dir / Path(row["resolved_cif_path"]).name).resolve()
            ),
            "sha256": _sha256_file(
                result.cif_dir / Path(row["resolved_cif_path"]).name
            ),
        }
        for row in rows
    ]
    expected_inputs.sort(key=lambda item: item["source_path"])
    if len(result.manifest["records"]) != len(rows):
        raise ValueError("CHGNet pre-relaxation changed the attempted sample count.")
    provenance = {
        "enabled": True,
        "variant": "chgnet-flowmm",
        "manifest_path": str(result.manifest_path.resolve()),
        "manifest_sha256": _sha256_file(result.manifest_path),
        "contract_sha256": result.manifest["contract_sha256"],
        "runtime_sha256": result.manifest["runtime_sha256"],
        "records_sha256": result.manifest["records_sha256"],
        "status_counts": result.manifest["status_counts"],
        "scoring_inputs": expected_inputs,
    }
    return result.cif_dir, expected_inputs, provenance


def _material_run_hash(raw_hash: str, prerelax: Mapping[str, Any] | None) -> str:
    """Separate raw and pre-relaxed upstream run namespaces."""
    return (
        raw_hash
        if prerelax is None
        else _sha256_json({"raw_samples": raw_hash, "prerelaxation": prerelax})
    )


def score_mp20(cfg: DictConfig, protocol_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Run or reload pinned LeMat-GenBench and form the Table-2 material row."""
    from zevals.materials.lematgenbench import (
        LEMAT_GENBENCH_ORACLE_IDENTITIES,
        LEMAT_GENBENCH_REFERENCE_REVISIONS,
        derive_attempt_normalized_rates,
        extract_lematgenbench_metrics,
        run_lematgenbench,
        select_table2_metrics,
    )

    domain_dir = _sampling_output_dir(cfg, "mp20")
    expected = int(protocol_manifest["request"]["mp20"]["num_samples"])
    validated_rows = _validate_saved_sample_manifest(
        domain_dir,
        domain="mp20",
        expected_num_samples=expected,
        expected_checkpoint=cfg.ckpt_path,
    )
    lemat_dir = domain_dir / "lematgenbench"
    sample_manifest_sha256 = _sha256_file(domain_dir / "sample_manifest.json")
    sample_artifacts_sha256 = _sample_artifacts_sha256(domain_dir)
    scoring_dir, expected_inputs, prerelax = _material_scoring_input(
        cfg,
        validated_rows,
        run=not bool(cfg.evaluation.mp20.lematgenbench.reuse_raw_results),
    )
    lemat_cfg = cfg.evaluation.mp20.lematgenbench
    effective_run_name = _effective_lemat_run_name(
        str(lemat_cfg.run_name),
        domain_dir=domain_dir,
        sample_manifest_sha256=sample_manifest_sha256,
        sample_artifacts_sha256=_material_run_hash(sample_artifacts_sha256, prerelax),
    )
    reuse_raw = bool(lemat_cfg.reuse_raw_results)
    if reuse_raw:
        raw_path = lemat_dir / "lematgenbench_raw.json"
        if not raw_path.is_file():
            raise FileNotFoundError(
                f"Cannot reuse LeMat-GenBench because raw output is missing: {raw_path}"
            )
        previous_metrics_path = domain_dir / "metrics.json"
        if not previous_metrics_path.is_file():
            raise FileNotFoundError(
                "Safe LeMat-GenBench reuse requires the prior MP20 metrics file "
                f"that binds raw results to sample hashes: {previous_metrics_path}"
            )
        previous_metrics = _read_json(previous_metrics_path)
        previous_evaluation = previous_metrics.get("evaluation", {})
        if previous_evaluation.get("prerelaxation") != prerelax:
            raise ValueError("Pre-relaxation differs from the saved LeMat result.")
        if (
            previous_evaluation.get("sample_manifest_sha256") != sample_manifest_sha256
            or previous_evaluation.get("sample_artifacts_sha256")
            != sample_artifacts_sha256
        ):
            raise ValueError(
                "Saved MP20 samples differ from those bound to the existing "
                "LeMat-GenBench result; refusing stale-result reuse."
            )
        adapter_manifest_path = lemat_dir / "lematgenbench_manifest.json"
        if not adapter_manifest_path.is_file():
            raise FileNotFoundError(
                f"LeMat-GenBench provenance is missing: {adapter_manifest_path}"
            )
        adapter_manifest = _read_json(adapter_manifest_path)
        input_accounting = _validate_lemat_manifest_contract(
            adapter_manifest,
            lemat_dir=lemat_dir,
            lemat_cfg=lemat_cfg,
            effective_run_name=effective_run_name,
            expected_inputs=expected_inputs,
            reference_revisions=LEMAT_GENBENCH_REFERENCE_REVISIONS,
            oracle_identities=LEMAT_GENBENCH_ORACLE_IDENTITIES,
        )
        raw_result_sha256 = _sha256_file(raw_path)
        previous_provenance = previous_metrics.get("lematgenbench")
        if not isinstance(previous_provenance, Mapping):
            raise ValueError(
                "Prior MP20 metrics have no LeMat-GenBench provenance; refusing "
                "raw-result reuse."
            )
        artifact_sha256 = previous_provenance.get("artifact_sha256")
        verified_artifacts = _validate_lemat_artifact_hashes(lemat_dir, artifact_sha256)
        if verified_artifacts["raw"] != raw_result_sha256:
            raise ValueError(
                "Existing LeMat-GenBench raw output differs from the digest bound "
                "to the prior MP20 metrics file."
            )
        if (
            previous_provenance.get("adapter_manifest_sha256")
            != verified_artifacts["manifest"]
        ):
            raise ValueError(
                "Existing LeMat-GenBench manifest differs from the digest bound "
                "to the prior MP20 metrics file."
            )
        flat_metrics = extract_lematgenbench_metrics(_read_json(raw_path))
        flat_metrics.update(
            derive_attempt_normalized_rates(flat_metrics, total_attempts=expected)
        )
        flat_metrics.update(
            {
                "input_accounting/attempted_structures": float(expected),
                "input_accounting/upstream_loaded_structures": float(
                    input_accounting["loaded"]
                ),
                "input_accounting/cif_parse_failure_count": float(
                    input_accounting["dropped"]
                ),
            }
        )
        persisted_flat_metrics = _read_json(lemat_dir / "lematgenbench_metrics.json")
        if not isinstance(persisted_flat_metrics, Mapping):
            raise TypeError("Archived LeMat-GenBench metrics must be a JSON object.")
        for key, value in persisted_flat_metrics.items():
            flat_metrics.setdefault(str(key), float(value))
        flat_metrics = dict(sorted(flat_metrics.items()))
        provenance = {
            **dict(previous_provenance),
            "reused_raw_results": True,
            "raw_result_path": str(raw_path),
            "raw_result_sha256": raw_result_sha256,
            "adapter_manifest_sha256": verified_artifacts["manifest"],
            "artifact_sha256": verified_artifacts,
            "effective_run_name": effective_run_name,
        }
    else:
        if lemat_cfg.repo_path is None:
            raise ValueError(
                "evaluation.mp20.lematgenbench.repo_path is required for the "
                "score_mp20 stage. Set ZEVALS_LEMAT_GENBENCH_REPO or override it."
            )
        result = run_lematgenbench(
            scoring_dir,
            repo_path=lemat_cfg.repo_path,
            run_name=effective_run_name,
            output_dir=lemat_dir,
            config_name=str(lemat_cfg.config_name),
            families=list(lemat_cfg.families),
            fingerprint_method=str(lemat_cfg.fingerprint_method),
            runner_command=list(lemat_cfg.runner_command),
            expected_commit=str(lemat_cfg.expected_commit),
            expected_lock_sha256=str(lemat_cfg.expected_lock_sha256),
            verify_mace_checkpoint=bool(lemat_cfg.verify_mace_checkpoint),
            expected_mace_checkpoint_sha256=str(
                lemat_cfg.expected_mace_checkpoint_sha256
            ),
            verify_orb_checkpoint=bool(lemat_cfg.verify_orb_checkpoint),
            expected_orb_checkpoint_sha256=str(
                lemat_cfg.expected_orb_checkpoint_sha256
            ),
            verify_uma_checkpoint=bool(lemat_cfg.verify_uma_checkpoint),
            expected_uma_checkpoint_sha256=str(
                lemat_cfg.expected_uma_checkpoint_sha256
            ),
            enforce_reference_revisions=bool(lemat_cfg.enforce_reference_revisions),
            runtime_device=str(lemat_cfg.runtime_device),
            cuda_visible_devices=(
                None
                if lemat_cfg.cuda_visible_devices is None
                else str(lemat_cfg.cuda_visible_devices)
            ),
            parse_failures_as_invalid=bool(lemat_cfg.parse_failures_as_invalid),
            require_complete_oracle_coverage=bool(
                lemat_cfg.require_complete_oracle_coverage
            ),
            require_clean_repo=bool(lemat_cfg.require_clean_repo),
            timeout_seconds=remaining_seconds(lemat_cfg.timeout_seconds),
            resume=bool(cfg.evaluation.get("resume", False)),
            fail_on_family_error=bool(lemat_cfg.fail_on_family_error),
        )
        adapter_manifest = _read_json(result.manifest_path)
        input_accounting = _validate_lemat_manifest_contract(
            adapter_manifest,
            lemat_dir=lemat_dir,
            lemat_cfg=lemat_cfg,
            effective_run_name=effective_run_name,
            expected_inputs=expected_inputs,
            reference_revisions=LEMAT_GENBENCH_REFERENCE_REVISIONS,
            oracle_identities=LEMAT_GENBENCH_ORACLE_IDENTITIES,
        )
        verified_artifacts = _validate_lemat_artifact_hashes(
            lemat_dir, result.artifact_sha256
        )
        flat_metrics = dict(result.metrics)
        flat_metrics.update(
            derive_attempt_normalized_rates(flat_metrics, total_attempts=expected)
        )
        flat_metrics = dict(sorted(flat_metrics.items()))
        provenance = result.provenance | {
            "reused_raw_results": False,
            "raw_result_sha256": verified_artifacts["raw"],
            "adapter_manifest_sha256": verified_artifacts["manifest"],
            "artifact_sha256": verified_artifacts,
            "effective_run_name": effective_run_name,
        }

    if _sample_artifacts_sha256(domain_dir) != sample_artifacts_sha256:
        raise ValueError(
            "Saved MP20 artifacts changed while LeMat-GenBench was being scored."
        )
    table_projection = select_table2_metrics(
        flat_metrics,
        distribution_key=str(lemat_cfg.distribution_key),
        diversity_key=str(lemat_cfg.diversity_key),
    )
    report = {
        "evaluation": {
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "num_attempted": expected,
            "num_loaded_by_lematgenbench": input_accounting["loaded"],
            "num_cif_parse_failures": input_accounting["dropped"],
            "sample_manifest_sha256": sample_manifest_sha256,
            "sample_artifacts_sha256": sample_artifacts_sha256,
            "table_projection": {
                "rate_denominator": "all attempted structures (Zatom-1 Table convention)",
                "distribution_key": str(lemat_cfg.distribution_key),
                "diversity_key": str(lemat_cfg.diversity_key),
            },
            "scoring_provenance": _scoring_provenance(_MP20_SCORING_MODULES),
        },
        "table2": table_projection,
        "table2_percent": {
            key: 100.0 * value
            for key, value in table_projection.items()
            if key in {"overall_valid", "unique", "novel", "stable", "sun", "metasun"}
        },
        "native_lemat_rates": _native_lemat_rates(flat_metrics),
        "distribution": _lemat_component_group(flat_metrics, "distribution"),
        "diversity": _lemat_component_group(flat_metrics, "diversity"),
        "flat_lemat_metrics": flat_metrics,
        "lematgenbench": provenance,
    }
    if prerelax is not None:
        _, _, revalidated = _material_scoring_input(cfg, validated_rows, run=False)
        if revalidated != prerelax:
            raise ValueError("CHGNet artifacts changed during LeMat scoring.")
        report["evaluation"]["prerelaxation"] = prerelax
    atomic_json(domain_dir / "metrics.json", _sanitize_json(report))
    LOGGER.info("MP20 metrics written to %s", domain_dir / "metrics.json")
    return report


def _write_table_csv(
    path: Path, qm9: Mapping[str, Any], mp20: Mapping[str, Any]
) -> None:
    row = {
        "mp20_variant": mp20.get("evaluation", {})
        .get("prerelaxation", {})
        .get("variant", "unrelaxed"),
        **{f"qm9_{key}_percent": value for key, value in qm9["percent"].items()},
        **{
            f"mp20_{key}": value
            for key, value in mp20["table2"].items()
            if key in {"distribution", "diversity"}
        },
        **{
            f"mp20_{key}_percent": value
            for key, value in mp20["table2_percent"].items()
        },
    }
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    temporary_path.replace(path)


def _verify_qm9_report_binding(cfg: DictConfig, metrics: Mapping[str, Any]) -> None:
    """Recompute QM9 metrics so a stale or edited table row cannot be emitted."""
    rescore_hint = (
        " Recompute from the saved samples with evaluation.stages=score_qm9 "
        "evaluation.resume=false using the pinned Table 2 QM9 environment."
    )
    evaluation = metrics.get("evaluation")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("qm9_scoring_revision") != QM9_SCORING_REVISION
    ):
        raise ValueError("QM9 scoring protocol is outdated." + rescore_hint)
    domain_dir = _sampling_output_dir(cfg, "qm9")
    expected = int(cfg.evaluation.qm9.num_samples)
    recomputed = _compute_qm9_saved_metrics(
        cfg,
        expected_num_samples=expected,
    )
    expected_evaluation = {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "qm9_scoring_revision": QM9_SCORING_REVISION,
        "num_attempted": expected,
        "sample_manifest_sha256": _sha256_file(domain_dir / "sample_manifest.json"),
        "sample_artifacts_sha256": _sample_artifacts_sha256(domain_dir),
        "scoring_provenance": _qm9_scoring_provenance(),
    }
    expected_keys = set(recomputed) | {"evaluation"}
    if set(metrics) != expected_keys or any(
        metrics.get(key) != value for key, value in recomputed.items()
    ):
        raise ValueError(
            "QM9 metrics do not match a fresh recomputation from the hash-bound "
            "samples." + rescore_hint
        )
    if metrics.get("evaluation") != expected_evaluation:
        raise ValueError(
            "QM9 metric provenance does not match the current samples, protocol, "
            "and chemistry runtime." + rescore_hint
        )


def _verify_report_bindings(
    cfg: DictConfig,
    *,
    run_manifest: Mapping[str, Any],
    domain_metrics: Mapping[str, Mapping[str, Any]],
) -> None:
    """Verify that report inputs still belong to this checkpoint and protocol."""
    checkpoint = run_manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Run manifest has no checkpoint provenance.")
    expected_checkpoint = str(Path(cfg.ckpt_path).expanduser().resolve())
    if checkpoint.get("path") != expected_checkpoint:
        raise ValueError(
            "Run manifest checkpoint does not match this report request: "
            f"{checkpoint.get('path')!r} != {expected_checkpoint!r}."
        )
    checkpoint_path = Path(expected_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Report checkpoint does not exist: {checkpoint_path}")
    recorded_checkpoint_size = checkpoint.get("size_bytes")
    if (
        recorded_checkpoint_size is not None
        and int(recorded_checkpoint_size) != checkpoint_path.stat().st_size
    ):
        raise ValueError("Checkpoint size changed after sample generation.")
    recorded_checkpoint_sha256 = checkpoint.get("sha256")
    if recorded_checkpoint_sha256 is not None:
        if _sha256_file(checkpoint_path) != recorded_checkpoint_sha256:
            raise ValueError("Checkpoint contents changed after sample generation.")
    elif bool(cfg.evaluation.get("hash_checkpoint", True)):
        raise ValueError(
            "Run manifest has no checkpoint SHA-256, but hash_checkpoint is enabled."
        )

    protocol_path = (
        Path(cfg.evaluation.protocol_dir).expanduser().resolve()
        / "protocol_manifest.json"
    )
    expected_protocol_hash = _sha256_file(protocol_path)
    run_protocol = run_manifest.get("protocol")
    expected_run_protocol = {
        "name": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "manifest_sha256": expected_protocol_hash,
        "canonical_runtime": True,
        "runtime_mismatches": {},
    }
    if not isinstance(run_protocol, Mapping):
        raise ValueError(
            "Run manifest is not bound to the current frozen protocol manifest."
        )
    protocol_mismatches = {
        key: {"found": run_protocol.get(key), "expected": expected}
        for key, expected in expected_run_protocol.items()
        if run_protocol.get(key) != expected
    }
    if protocol_mismatches:
        raise ValueError(
            "Run manifest is noncanonical or is not bound to the current frozen "
            f"protocol manifest: {protocol_mismatches}"
        )

    run_domains = run_manifest.get("domains")
    if not isinstance(run_domains, Mapping):
        raise ValueError("Run manifest has no domain sampling provenance.")
    for domain, metrics in domain_metrics.items():
        domain_dir = _sampling_output_dir(cfg, domain)
        manifest_hash = _sha256_file(domain_dir / "sample_manifest.json")
        artifacts_hash = _sample_artifacts_sha256(domain_dir)
        evaluation = metrics.get("evaluation")
        run_domain = run_domains.get(domain)
        if not isinstance(evaluation, Mapping) or not isinstance(run_domain, Mapping):
            raise ValueError(f"Missing {domain.upper()} report provenance.")
        if (
            evaluation.get("sample_manifest_sha256") != manifest_hash
            or evaluation.get("sample_artifacts_sha256") != artifacts_hash
            or run_domain.get("sample_manifest_sha256") != manifest_hash
            or run_domain.get("sample_artifacts_sha256") != artifacts_hash
        ):
            raise ValueError(
                f"{domain.upper()} metrics, saved samples, and run manifest are "
                "not hash-bound to the same generation run."
            )
        if domain == "qm9":
            _verify_qm9_report_binding(cfg, metrics)
        elif domain == "mp20":
            _verify_mp20_lemat_report_binding(cfg, metrics)


def _verify_mp20_lemat_report_binding(
    cfg: DictConfig, metrics: Mapping[str, Any]
) -> None:
    """Revalidate LeMat artifacts and recompute its publication projection."""
    from zevals.materials.lematgenbench import (
        LEMAT_GENBENCH_ORACLE_IDENTITIES,
        LEMAT_GENBENCH_REFERENCE_REVISIONS,
        derive_attempt_normalized_rates,
        extract_lematgenbench_metrics,
        select_table2_metrics,
    )

    domain_dir = _sampling_output_dir(cfg, "mp20")
    expected = int(cfg.evaluation.mp20.num_samples)
    validated_rows = _validate_saved_sample_manifest(
        domain_dir,
        domain="mp20",
        expected_num_samples=expected,
        expected_checkpoint=cfg.ckpt_path,
    )
    sample_manifest_sha256 = _sha256_file(domain_dir / "sample_manifest.json")
    sample_artifacts_sha256 = _sample_artifacts_sha256(domain_dir)
    _, expected_inputs, prerelax = _material_scoring_input(
        cfg, validated_rows, run=False
    )
    lemat_cfg = cfg.evaluation.mp20.lematgenbench
    effective_run_name = _effective_lemat_run_name(
        str(lemat_cfg.run_name),
        domain_dir=domain_dir,
        sample_manifest_sha256=sample_manifest_sha256,
        sample_artifacts_sha256=_material_run_hash(sample_artifacts_sha256, prerelax),
    )
    lemat_dir = domain_dir / "lematgenbench"
    adapter_manifest = _read_json(lemat_dir / "lematgenbench_manifest.json")
    if not isinstance(adapter_manifest, Mapping):
        raise TypeError("LeMat-GenBench adapter manifest must be a JSON object.")
    input_accounting = _validate_lemat_manifest_contract(
        adapter_manifest,
        lemat_dir=lemat_dir,
        lemat_cfg=lemat_cfg,
        effective_run_name=effective_run_name,
        expected_inputs=expected_inputs,
        reference_revisions=LEMAT_GENBENCH_REFERENCE_REVISIONS,
        oracle_identities=LEMAT_GENBENCH_ORACLE_IDENTITIES,
    )

    provenance = metrics.get("lematgenbench")
    if not isinstance(provenance, Mapping):
        raise ValueError("MP20 metrics have no LeMat-GenBench provenance.")
    verified_artifacts = _validate_lemat_artifact_hashes(
        lemat_dir, provenance.get("artifact_sha256")
    )
    runtime = adapter_manifest["runtime"]
    mace = adapter_manifest["mace_checkpoint"]
    oracle_protocol = adapter_manifest["oracle_protocol"]
    reference_pins = adapter_manifest["reference_revision_pins"]
    reused_raw_results = provenance.get("reused_raw_results")
    if not isinstance(reused_raw_results, bool):
        raise ValueError(
            "MP20 metrics do not record whether raw LeMat-GenBench results "
            "were reused."
        )
    expected_provenance = {
        "upstream_commit": adapter_manifest["upstream_commit"],
        "upstream_uv_lock_sha256": adapter_manifest["upstream_uv_lock_sha256"],
        "raw_result_path": str(lemat_dir / "lematgenbench_raw.json"),
        "metrics_path": str(lemat_dir / "lematgenbench_metrics.json"),
        "manifest_path": str(lemat_dir / "lematgenbench_manifest.json"),
        "stdout_path": str(lemat_dir / "lematgenbench.stdout.log"),
        "stderr_path": str(lemat_dir / "lematgenbench.stderr.log"),
        "uv_lock_path": adapter_manifest["archived_uv_lock_path"],
        "upstream_result_path": adapter_manifest["upstream_result_path"],
        "command": adapter_manifest["command"],
        "family_errors": adapter_manifest["family_errors"],
        "expected_input_count": expected,
        "upstream_loaded_count": input_accounting["loaded"],
        "parse_failure_count": input_accounting["dropped"],
        "mace_model": mace["model"],
        "mace_checkpoint_path": mace["path"],
        "mace_checkpoint_sha256": mace["sha256"],
        "reference_revisions_enforced": reference_pins["enforced"],
        "reference_revisions": reference_pins["revisions"],
        "reference_pin_shim_path": reference_pins["shim_path"],
        "reference_pin_shim_sha256": reference_pins["shim_sha256"],
        "reference_pin_audit_path": reference_pins["audit_path"],
        "runtime_environment": runtime["provenance"],
        "runtime_environment_path": runtime["provenance_path"],
        "runtime_environment_sha256": runtime["provenance_sha256"],
        "artifact_sha256": verified_artifacts,
        "oracle_identities": oracle_protocol["identities"],
        "oracle_coverage": oracle_protocol["coverage"],
        "reused_raw_results": reused_raw_results,
        "raw_result_sha256": verified_artifacts["raw"],
        "adapter_manifest_sha256": verified_artifacts["manifest"],
        "effective_run_name": effective_run_name,
    }
    if provenance != expected_provenance:
        raise ValueError(
            "MP20 metrics are not bound to the current LeMat-GenBench artifacts: "
            f"found {dict(provenance)}, expected {expected_provenance}"
        )

    evaluation = metrics.get("evaluation")
    expected_evaluation = {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "num_attempted": expected,
        "num_loaded_by_lematgenbench": input_accounting["loaded"],
        "num_cif_parse_failures": input_accounting["dropped"],
        "sample_manifest_sha256": sample_manifest_sha256,
        "sample_artifacts_sha256": sample_artifacts_sha256,
        "table_projection": {
            "rate_denominator": ("all attempted structures (Zatom-1 Table convention)"),
            "distribution_key": str(lemat_cfg.distribution_key),
            "diversity_key": str(lemat_cfg.diversity_key),
        },
        "scoring_provenance": _scoring_provenance(_MP20_SCORING_MODULES),
    }
    if prerelax is not None:
        expected_evaluation["prerelaxation"] = prerelax
    if evaluation != expected_evaluation:
        raise ValueError(
            "MP20 metric provenance and denominators do not match the audited "
            f"LeMat-GenBench run: found {evaluation}, expected {expected_evaluation}"
        )

    recomputed = extract_lematgenbench_metrics(
        _read_json(lemat_dir / "lematgenbench_raw.json")
    )
    recomputed.update(
        derive_attempt_normalized_rates(recomputed, total_attempts=expected)
    )
    recomputed.update(
        {
            "input_accounting/attempted_structures": float(expected),
            "input_accounting/upstream_loaded_structures": float(
                input_accounting["loaded"]
            ),
            "input_accounting/cif_parse_failure_count": float(
                input_accounting["dropped"]
            ),
        }
    )
    persisted_flat_metrics = _read_json(lemat_dir / "lematgenbench_metrics.json")
    if not isinstance(persisted_flat_metrics, Mapping):
        raise TypeError("Archived LeMat-GenBench metrics must be a JSON object.")
    for key, value in persisted_flat_metrics.items():
        recomputed.setdefault(str(key), float(value))
    recomputed = dict(sorted(recomputed.items()))
    table_projection = select_table2_metrics(
        recomputed,
        distribution_key=str(lemat_cfg.distribution_key),
        diversity_key=str(lemat_cfg.diversity_key),
    )
    if metrics.get("table2") != table_projection:
        raise ValueError(
            "MP20 Table-2 projection does not match the hash-bound raw "
            "LeMat-GenBench result."
        )
    expected_percent = {
        key: 100.0 * value
        for key, value in table_projection.items()
        if key in {"overall_valid", "unique", "novel", "stable", "sun", "metasun"}
    }
    if metrics.get("table2_percent") != expected_percent:
        raise ValueError(
            "MP20 percentage columns do not match the hash-bound raw "
            "LeMat-GenBench result."
        )
    auxiliary_fields = {
        "native_lemat_rates": _native_lemat_rates(recomputed),
        "distribution": _lemat_component_group(recomputed, "distribution"),
        "diversity": _lemat_component_group(recomputed, "diversity"),
        "flat_lemat_metrics": recomputed,
    }
    auxiliary_mismatches = {
        key: {"found": metrics.get(key), "expected": expected_value}
        for key, expected_value in auxiliary_fields.items()
        if metrics.get(key) != expected_value
    }
    if auxiliary_mismatches:
        raise ValueError(
            "MP20 auxiliary metrics do not match the hash-bound raw and "
            f"adapter metric artifacts: {auxiliary_mismatches}"
        )


def write_report(cfg: DictConfig) -> dict[str, Any]:
    """Merge domain metrics into one machine-readable report and table row."""
    runtime_mismatches = _protocol_conformance_mismatches(cfg)
    if runtime_mismatches:
        raise ValueError(
            "Refusing to publish a Table-2 report from a noncanonical runtime, "
            "even when evaluation.allow_noncanonical_protocol=true: "
            f"{runtime_mismatches}"
        )
    output_dir = Path(cfg.out_dir).expanduser().resolve()
    qm9_path = output_dir / "qm9" / "metrics.json"
    mp20_path = output_dir / "mp20" / "metrics.json"
    missing = [path for path in (qm9_path, mp20_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The report stage requires both domain metric files; missing "
            + ", ".join(str(path) for path in missing)
        )
    qm9 = _read_json(qm9_path)
    mp20 = _read_json(mp20_path)
    run_manifest_path = output_dir / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(
            f"The report stage requires sampling provenance: {run_manifest_path}"
        )
    run_manifest = _read_json(run_manifest_path)
    if not isinstance(run_manifest, Mapping):
        raise TypeError(f"Run manifest must be a JSON object: {run_manifest_path}")
    _verify_report_bindings(
        cfg,
        run_manifest=run_manifest,
        domain_metrics={"qm9": qm9, "mp20": mp20},
    )
    report = {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "material_variant": (
            "chgnet-flowmm"
            if OmegaConf.select(cfg, "evaluation.mp20.prerelax.enabled", default=False)
            else "unrelaxed"
        ),
        "qm9": qm9,
        "mp20": mp20,
        "run_manifest": run_manifest,
    }
    atomic_json(output_dir / "table2_metrics.json", _sanitize_json(report))
    _write_table_csv(output_dir / "table2_metrics.csv", qm9, mp20)
    LOGGER.info("Combined Table-2 report written to %s", output_dir)
    return report


def _run_table2_stages(cfg: DictConfig) -> dict[str, Any]:
    """Execute selected preparation, sampling, scoring, and reporting stages."""
    if int(cfg.seed) != int(cfg.evaluation.seed):
        raise ValueError(
            "seed and evaluation.seed must match so node-count and diffusion "
            "sampling use one recorded Table-2 seed. Override seed, not both."
        )
    _validate_protocol_conformance(cfg)
    stages = _normalize_stages(cfg.evaluation.stages)
    resume = bool(cfg.evaluation.get("resume", False))
    protocol_path = (
        Path(cfg.evaluation.protocol_dir).expanduser() / "protocol_manifest.json"
    )
    if "prepare" in stages and not (resume and protocol_path.exists()):
        remaining_seconds()
        prepare_protocol(cfg)
    protocol_manifest, paths = load_protocol(cfg)
    results: dict[str, Any] = {"stages": list(stages)}
    if "sample" in stages:
        run_manifest, is_rank_zero = sample_checkpoint(cfg, protocol_manifest, paths)
        if not is_rank_zero:
            return {}
        results["sampling"] = run_manifest
    if "score_qm9" in stages:
        remaining_seconds()
        saved_path = _sampling_output_dir(cfg, "qm9") / "metrics.json"
        if resume and saved_path.is_file():
            results["qm9"] = _read_json(saved_path)
            _verify_qm9_report_binding(cfg, results["qm9"])
        else:
            results["qm9"] = score_qm9(cfg, protocol_manifest, paths)
    if "score_mp20" in stages:
        remaining_seconds()
        saved_path = _sampling_output_dir(cfg, "mp20") / "metrics.json"
        if resume and saved_path.is_file():
            results["mp20"] = _read_json(saved_path)
            _verify_mp20_lemat_report_binding(cfg, results["mp20"])
        else:
            results["mp20"] = score_mp20(cfg, protocol_manifest)
    if "report" in stages:
        remaining_seconds()
        results["report"] = write_report(cfg)
    return results


def run_table2_evaluation(cfg: DictConfig) -> dict[str, Any]:
    """Run stages with a single-writer lock and an optional allocation budget."""
    # Validate this invariant before touching output paths (also useful to callers
    # validating an incomplete config).
    if int(cfg.seed) != int(cfg.evaluation.seed):
        raise ValueError("seed and evaluation.seed must match.")
    with evaluation_allocation(
        Path(cfg.out_dir).expanduser().resolve(),
        cfg.evaluation.get("walltime_seconds"),
    ):
        return _run_table2_stages(cfg)


__all__ = [
    "ALL_STAGES",
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "build_input_manifest",
    "prepare_protocol",
    "run_table2_evaluation",
    "sample_node_counts",
    "score_mp20",
    "score_qm9",
    "write_report",
]
