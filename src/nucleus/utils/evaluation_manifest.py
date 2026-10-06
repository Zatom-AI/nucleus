"""Stable identities for restartable scientific evaluations."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf

_NON_PREDICTION_ROOT_CONFIG_KEYS = {
    "dump_prediction_metadata_json",
    "dump_trajectories",
    "inputs",
    "n_batches",
    "out_dir",
    "prevalidate_inputs",
    "save_cifs",
    "save_molecules",
    "skip_existing",
    "verbose",
}
_NON_PREDICTION_EVALUATION_CONFIG_KEYS = {
    "num_workers",
    "prefetch_factor",
    "progress_every",
    "resume",
}


def _config_container(config: Any, *, resolve: bool) -> Any:
    """Convert an OmegaConf value into deterministic JSON-compatible data."""
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(
            config,
            resolve=resolve,
            enum_to_str=True,
        )
    return config


def stable_json_sha256(value: Any) -> str:
    """Hash nested configuration data independent of mapping insertion order."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def source_tree_fingerprint(package_names: Iterable[str]) -> dict[str, Any]:
    """Hash all Python sources in the packages that can affect predictions.

    Package-relative paths are used so equivalent editable and wheel installs have
    the same identity. Hashing source contents also detects uncommitted changes,
    unlike a version or Git revision alone.
    """
    package_digests: dict[str, dict[str, Any]] = {}
    combined_digest = hashlib.sha256()
    for package_name in sorted(set(package_names)):
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            raise ModuleNotFoundError(
                f"Cannot fingerprint prediction package {package_name!r}."
            )

        roots = [Path(path).resolve() for path in spec.submodule_search_locations or ()]
        if roots:
            source_entries = [
                (f"{root_index}/{path.relative_to(root).as_posix()}", path)
                for root_index, root in enumerate(sorted(roots))
                for path in sorted(root.rglob("*.py"))
                if path.is_file()
            ]
        elif spec.origin is not None:
            origin = Path(spec.origin).resolve()
            source_entries = [(origin.name, origin)]
        else:
            source_entries = []

        if not source_entries:
            raise FileNotFoundError(
                f"No source files found for prediction package {package_name!r}."
            )

        package_digest = hashlib.sha256()
        for relative_path, source_path in source_entries:
            package_digest.update(relative_path.encode())
            package_digest.update(b"\0")
            package_digest.update(source_path.read_bytes())
            package_digest.update(b"\0")
        digest = package_digest.hexdigest()
        package_digests[package_name] = {
            "sha256": digest,
            "python_file_count": len(source_entries),
        }
        combined_digest.update(package_name.encode())
        combined_digest.update(b"\0")
        combined_digest.update(digest.encode())
        combined_digest.update(b"\0")

    return {
        "algorithm": "sha256",
        "sha256": combined_digest.hexdigest(),
        "packages": package_digests,
    }


def _runtime_versions() -> dict[str, str]:
    """Return numerical/runtime package versions relevant to prediction output."""
    versions = {"python": platform.python_version()}
    for distribution_name in (
        "atomworks",
        "lightning",
        "pyarrow",
        "rc-foundry",
        "torch",
        "zatom-nucleus",
    ):
        try:
            versions[distribution_name] = metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            versions[distribution_name] = "not-installed"
    return versions


def _prediction_runtime_config(config: Any) -> dict[str, Any]:
    """Remove output-only and resume-throughput options from a run config."""
    container = _config_container(config, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("MLIP evaluation config must resolve to a mapping.")
    prediction_config = dict(container)
    for key in _NON_PREDICTION_ROOT_CONFIG_KEYS:
        prediction_config.pop(key, None)
    evaluation = prediction_config.get("evaluation")
    if isinstance(evaluation, dict):
        evaluation = dict(evaluation)
        for key in _NON_PREDICTION_EVALUATION_CONFIG_KEYS:
            evaluation.pop(key, None)
        prediction_config["evaluation"] = evaluation
    return prediction_config


def _checkpoint_config_fingerprint(checkpoint_config: Any) -> dict[str, Any]:
    """Hash both serialized config and resolved prediction-critical sections.

    The unresolved hash detects any checkpoint-config edit. Resolving the model
    and trainer sections additionally captures environment-backed interpolations
    that can change precision, device behavior, or the forward path without
    changing the serialized interpolation expression. Dataset preprocessing is
    resolved separately from the exact configs constructed by the evaluator.
    """
    unresolved_config = _config_container(checkpoint_config, resolve=False)
    fingerprint: dict[str, Any] = {
        "unresolved_sha256": stable_json_sha256(unresolved_config),
        "resolved_sections": {},
    }
    if not OmegaConf.is_config(checkpoint_config):
        fingerprint["resolved_sections"]["all"] = stable_json_sha256(checkpoint_config)
        return fingerprint

    for section_name in ("model", "trainer"):
        try:
            section = OmegaConf.select(
                checkpoint_config,
                section_name,
                default=None,
            )
            resolved_section = _config_container(section, resolve=True)
        except Exception:
            # An unused section can contain an unavailable environment resolver.
            # Its unresolved form remains covered by the full-config hash. Any
            # section needed for inference must already resolve during engine
            # initialization, so successful evaluations take the resolved path.
            resolved_section = (
                unresolved_config.get(section_name)
                if isinstance(unresolved_config, dict)
                else None
            )
        fingerprint["resolved_sections"][section_name] = stable_json_sha256(
            resolved_section
        )
    return fingerprint


def build_mlip_prediction_identity(
    *,
    config: Any,
    checkpoint_config: Any,
    preprocessing_configs: dict[str, Any],
    model_package: str,
    prediction_schema: str,
) -> dict[str, Any]:
    """Build the complete identity required to reuse MLIP prediction parts."""
    runtime_config = _prediction_runtime_config(config)
    preprocessing_container = {
        str(domain): _config_container(domain_config, resolve=True)
        for domain, domain_config in sorted(preprocessing_configs.items())
    }
    source_fingerprint = source_tree_fingerprint(
        (model_package, "atomworks", "foundry", "nucleus", "rfd3")
    )
    return {
        "version": 1,
        "runtime_config_sha256": stable_json_sha256(runtime_config),
        "checkpoint_config": _checkpoint_config_fingerprint(checkpoint_config),
        "preprocessing_config_sha256": stable_json_sha256(preprocessing_container),
        "prediction_schema_sha256": stable_json_sha256(prediction_schema),
        "source": source_fingerprint,
        "runtime_versions": _runtime_versions(),
    }


__all__ = [
    "build_mlip_prediction_identity",
    "source_tree_fingerprint",
    "stable_json_sha256",
]
