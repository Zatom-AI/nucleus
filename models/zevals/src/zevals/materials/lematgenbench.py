"""Reproducible subprocess integration for LeMat-GenBench.

LeMat-GenBench is intentionally executed from an isolated source checkout. Its
runtime depends on specific PyTorch, PyG, MACE, ORB, and UMA versions that are
not compatible with every Nucleus training environment, and the leaderboard
entry point is currently a repository script rather than a stable Python API.

The adapter targets upstream commit :data:`LEMAT_GENBENCH_COMMIT`. It verifies
that commit (and, by default, a clean worktree), invokes the upstream benchmark
runner without a shell, preserves the complete upstream JSON, and exposes a
stable, flattened map of numeric ``final_scores`` for downstream reporting.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

from pymatgen.core import Structure  # type: ignore[import]

from zevals.materials._lemat_resume import (
    LeMatResumeTimeout,
    archive_attempt,
    load_completed,
    locked_resume,
    remaining_timeout,
    restore_audit,
    run_streamed,
    runtime_contract,
    save_completed,
    validate_contract,
)

LEMAT_GENBENCH_REPOSITORY = "https://github.com/LeMaterial/lemat-genbench"
"""Canonical upstream repository."""

LEMAT_GENBENCH_COMMIT = "58e6eae3e4a6c87c22171cf069123ecc4e2fa7e6"
"""Upstream revision supported by this adapter (``main`` on 2026-09-04)."""

LEMAT_GENBENCH_UV_LOCK_SHA256 = (
    "74970323f6697d50e60251de9182e7a559031bce67ab2f693986742e7c286850"
)
"""SHA-256 of the ``uv.lock`` committed at :data:`LEMAT_GENBENCH_COMMIT`."""

LEMAT_GENBENCH_MACE_MODEL = "medium-mpa-0"
"""Explicit MACE model implied by the pinned ``mace-torch`` default."""

LEMAT_GENBENCH_MACE_CHECKPOINT_SHA256 = (
    "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638"
)
"""Expected SHA-256 of the ``medium-mpa-0`` checkpoint."""

LEMAT_GENBENCH_ORB_MODEL = "orb_v3_conservative_inf_omat"
"""ORB model selected by the canonical multi-MLIP configuration."""

LEMAT_GENBENCH_ORB_CHECKPOINT_URL = (
    "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/"
    "forcefields/orb-v3/orb-v3-conservative-inf-omat-20250404.ckpt"
)
"""Checkpoint URL embedded in the locked ``orb-models==0.5.4`` package."""

LEMAT_GENBENCH_ORB_CHECKPOINT_SHA256 = (
    "0a41ef1132ad9c41ee0c7b9d855fccabfefb97abe04625c97ec0d07930b6c0f0"
)
"""Expected SHA-256 of the canonical ORB checkpoint."""

LEMAT_GENBENCH_UMA_MODEL = "uma-s-1"
"""UMA model selected by the canonical multi-MLIP configuration."""

LEMAT_GENBENCH_UMA_TASK = "omat"
"""UMA prediction head selected by the canonical configuration."""

LEMAT_GENBENCH_UMA_REPOSITORY = "facebook/UMA"
"""Hugging Face repository containing the canonical UMA checkpoint."""

LEMAT_GENBENCH_UMA_REVISION = "f611b917d9c68566bbbeccbb0aa0f7cad1696cb2"
"""Audited Hugging Face revision for the canonical UMA checkpoint."""

LEMAT_GENBENCH_UMA_CHECKPOINT_FILE = "checkpoints/uma-s-1.pt"
"""Repository-relative path of the canonical UMA checkpoint."""

LEMAT_GENBENCH_UMA_CHECKPOINT_SHA256 = (
    "02f83e20547294d3d47dd9dfbe4a4ee51eebca21b2e1febe63210e55b764cc78"
)
"""Expected SHA-256 of the canonical UMA checkpoint."""

LEMAT_GENBENCH_REFERENCE_REVISIONS = {
    "LeMaterial/LeMat-Bulk": "0dc17eea904b860ad7288141e9870f67f8e6bb2c",
    "LeMaterial/LeMat-Bulk-MLIP-Hull": ("70d505bb294c16658a9551e2b81ace69a28d9790"),
    "LeMaterial/LeMat-GenBench-embeddings": (
        "0d5895780c53fa54347ffc076cc72c7748e692e4"
    ),
    LEMAT_GENBENCH_UMA_REPOSITORY: LEMAT_GENBENCH_UMA_REVISION,
}
"""Audited Hugging Face dataset and model revisions used by the protocol."""

LEMAT_GENBENCH_LOCAL_REFERENCE_SHA256 = {
    "data/all_compositions.npz": (
        "28b362f9a558e25d6ba3b846060b56da2ec5caada299bc0737e7b7256325f38f"
    ),
    "data/lematbulk_jsdistance_distributions.json": (
        "9603f06e5f86d11877e81735e89de3e39f2f6c5c0395d61d69cd20ea88108748"
    ),
    "data/lematbulk_mmd_values_15k.pkl": (
        "1361fa4c1e7ac402cac16f9428ab77e32c753262a771374eda7e166cb9f71fd0"
    ),
    "data/mace_mu.npy": (
        "a39c081db7c8db67bce2a23c28add8edc7dccbcf9c7bc80230fecddca3f1befd"
    ),
    "data/mace_sigma.npy": (
        "644801230ae4bf86982e57394fddbd82414cd5db23ac644040722c3008644aea"
    ),
    "data/orb_mu.npy": (
        "72c6c25763e91065d79068b1018a624ed0350c97963fef58e02490da226d1b7a"
    ),
    "data/orb_sigma.npy": (
        "f6625a5548218992875db07a55093a7a1de5d415e296cd2885097dbfd7eaca80"
    ),
    "data/uma_mu.npy": (
        "1685dad90fa2a5c670731ee7e07e63a926ae595745f50e01b337b030427ea28a"
    ),
    "data/uma_sigma.npy": (
        "adec18113f0e44b06fed966387d2800f9d49464318fdb745534efdab4eba431b"
    ),
}
"""Digests of ordinary Git blobs used as canonical local references.

These files are tracked directly (no Git LFS filter) at
:data:`LEMAT_GENBENCH_COMMIT`. Pinning the source commit is therefore enough to
identify them, while the explicit digest check also detects incomplete LFS-like
checkouts and local overrides before a manuscript run.
"""

LEMAT_GENBENCH_ORACLE_IDENTITIES = {
    "orb": {
        "model": LEMAT_GENBENCH_ORB_MODEL,
        "checkpoint_url": LEMAT_GENBENCH_ORB_CHECKPOINT_URL,
        "checkpoint_sha256": LEMAT_GENBENCH_ORB_CHECKPOINT_SHA256,
        "fetch_api": "cached_path.cached_path",
    },
    "mace": {
        "model": LEMAT_GENBENCH_MACE_MODEL,
        "checkpoint_sha256": LEMAT_GENBENCH_MACE_CHECKPOINT_SHA256,
        "fetch_api": "mace.calculators.mace_mp",
    },
    "uma": {
        "model": LEMAT_GENBENCH_UMA_MODEL,
        "task": LEMAT_GENBENCH_UMA_TASK,
        "repository": LEMAT_GENBENCH_UMA_REPOSITORY,
        "revision": LEMAT_GENBENCH_UMA_REVISION,
        "checkpoint_file": LEMAT_GENBENCH_UMA_CHECKPOINT_FILE,
        "checkpoint_sha256": LEMAT_GENBENCH_UMA_CHECKPOINT_SHA256,
        "fetch_api": "huggingface_hub.hf_hub_download",
    },
}
"""Exact oracle identities audited for the canonical Table 2 protocol."""

LEMAT_GENBENCH_CONFIG = "comprehensive_multi_mlip_hull"
"""Leaderboard-compatible upstream configuration."""

LEMAT_GENBENCH_TABLE2_FAMILIES = (
    "distribution",
    "diversity",
    "novelty",
    "uniqueness",
    "stability",
    "sun",
)
"""Families needed for the material columns in Zatom-2 Table 2.

Validity is mandatory in the upstream runner and is therefore not listed.
"""

TABLE2_DISTRIBUTION_KEYS = (
    "distribution/JSDistance",
    "distribution/MMD",
    "distribution/FrechetDistance",
)
"""Available upstream distribution scalars; the manuscript must freeze one."""

TABLE2_DIVERSITY_KEYS = (
    "diversity/element_diversity",
    "diversity/space_group_diversity",
    "diversity/site_number_diversity",
    "diversity/physical_size_diversity",
)
"""Available upstream diversity scalars; the manuscript must freeze one."""

_DEFAULT_RUNNER_COMMAND = ("uv", "run", "--frozen", "python")
_UPSTREAM_DEFAULT_FAMILIES = (
    "distribution",
    "diversity",
    "novelty",
    "uniqueness",
    "hhi",
    "sun",
    "stability",
)
_PATH_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RESULT_PATH_PATTERN = re.compile(r"Results saved to:\s*(.+?\.json)\s*$")
_MACE_PROBE_RESULT_PREFIX = "ZEVALS_MACE_CHECKPOINT="
_MACE_FALLBACK_LOG_MARKER = "Successfully loaded MACE model using alternative approach"
_RUNTIME_PROBE_RESULT_PREFIX = "ZEVALS_LEMAT_RUNTIME="
_REFERENCE_PIN_AUDIT_ENV = "ZEVALS_LEMAT_REFERENCE_PIN_AUDIT"
_REFERENCE_PIN_ROLE_ENV = "ZEVALS_LEMAT_INTERPRETER_ROLE"
_REFERENCE_PIN_PROBE_PREFIX = "ZEVALS_REFERENCE_PIN_SHIM="
_REFERENCE_PIN_PATCHED_APIS = (
    "cached_path.cached_path",
    "datasets.load_dataset",
    "huggingface_hub.hf_hub_download",
    "huggingface_hub.snapshot_download",
    "mace.calculators.mace_mp",
)
_ALLOWED_UNTRACKED_PREFIXES = (
    "data/.cache/",
    "data/convex_hulls/",
    "data/reference_fingerprints/",
    "hydra-outputs/",
    "multirun/",
    "results/",
    "results_final/",
    "results_new/",
)
_IGNORED_CODE_SUFFIXES = (".json", ".py", ".pyi", ".so", ".toml", ".yaml", ".yml")


class LeMatGenBenchError(RuntimeError):
    """Raised when an upstream LeMat-GenBench run cannot be completed or read."""


class LeMatGenBenchTimeout(LeMatGenBenchError, LeMatResumeTimeout):
    """The current allocation ended; completed work can be resumed."""


@dataclass(frozen=True)
class LeMatGenBenchResult:
    """Artifacts and metrics produced by one LeMat-GenBench invocation.

    Attributes:
        raw_result: Complete JSON document emitted by the upstream runner.
        metrics: Deterministically ordered ``family/key`` numeric map extracted
            from each benchmark family's upstream ``final_scores``.
        output_dir: Directory containing the archived result and provenance.
        raw_result_path: Archived copy of the complete upstream JSON.
        metrics_path: JSON file containing ``metrics``.
        manifest_path: JSON file containing invocation and input provenance.
        stdout_path: Captured standard output from the upstream process.
        stderr_path: Captured standard error from the upstream process.
        uv_lock_path: Archived copy of the upstream lockfile, when present.
        upstream_result_path: Result file originally written by upstream.
        upstream_commit: Verified Git revision of the upstream checkout.
        upstream_lock_sha256: Verified digest of the upstream lockfile.
        command: Exact argument vector used to launch the benchmark.
        family_errors: Upstream family errors (empty for a complete run).
        expected_input_count: Number of CIF paths supplied to upstream.
        upstream_loaded_count: Number of CIFs upstream successfully parsed.
        parse_failure_count: Number of staged CIFs dropped by upstream parsing.
        mace_model: Explicitly verified MACE model name, when MLIPs ran.
        mace_checkpoint_path: Path reported by the isolated MACE installation.
        mace_checkpoint_sha256: Verified MACE checkpoint digest.
        reference_pin_shim_path: Archived Python startup shim, when enabled.
        reference_pin_shim_sha256: Digest of the exact runtime shim.
        reference_pin_audit_path: JSONL audit written by the runtime shim.
        reference_revisions_enforced: Whether the runtime pin layer was enabled.
        runtime_environment: Isolated Python/Torch/CUDA/GPU provenance.
        runtime_environment_path: Archived runtime provenance JSON.
        runtime_environment_sha256: Digest of the runtime provenance JSON.
        artifact_sha256: Frozen digest map for manifests, locks, audits, and logs.
        oracle_coverage: Per-oracle identity and structure-coverage audit.
    """

    raw_result: dict[str, Any]
    metrics: dict[str, float]
    output_dir: Path
    raw_result_path: Path
    metrics_path: Path
    manifest_path: Path
    stdout_path: Path
    stderr_path: Path
    uv_lock_path: Path | None
    upstream_result_path: Path
    upstream_commit: str
    upstream_lock_sha256: str | None
    command: tuple[str, ...]
    family_errors: dict[str, str]
    expected_input_count: int
    upstream_loaded_count: int
    parse_failure_count: int
    mace_model: str | None
    mace_checkpoint_path: Path | None
    mace_checkpoint_sha256: str | None
    reference_pin_shim_path: Path | None
    reference_pin_shim_sha256: str | None
    reference_pin_audit_path: Path | None
    reference_revisions_enforced: bool
    runtime_environment: dict[str, Any]
    runtime_environment_path: Path
    runtime_environment_sha256: str
    artifact_sha256: dict[str, str]
    oracle_coverage: dict[str, Any]

    @property
    def provenance(self) -> dict[str, Any]:
        """Return JSON-serializable paths and revision metadata."""
        return {
            "upstream_commit": self.upstream_commit,
            "upstream_uv_lock_sha256": self.upstream_lock_sha256,
            "raw_result_path": str(self.raw_result_path),
            "metrics_path": str(self.metrics_path),
            "manifest_path": str(self.manifest_path),
            "stdout_path": str(self.stdout_path),
            "stderr_path": str(self.stderr_path),
            "uv_lock_path": (
                str(self.uv_lock_path) if self.uv_lock_path is not None else None
            ),
            "upstream_result_path": str(self.upstream_result_path),
            "command": list(self.command),
            "family_errors": self.family_errors,
            "expected_input_count": self.expected_input_count,
            "upstream_loaded_count": self.upstream_loaded_count,
            "parse_failure_count": self.parse_failure_count,
            "mace_model": self.mace_model,
            "mace_checkpoint_path": (
                str(self.mace_checkpoint_path)
                if self.mace_checkpoint_path is not None
                else None
            ),
            "mace_checkpoint_sha256": self.mace_checkpoint_sha256,
            "reference_revisions_enforced": self.reference_revisions_enforced,
            "reference_revisions": dict(LEMAT_GENBENCH_REFERENCE_REVISIONS),
            "reference_pin_shim_path": (
                str(self.reference_pin_shim_path)
                if self.reference_pin_shim_path is not None
                else None
            ),
            "reference_pin_shim_sha256": self.reference_pin_shim_sha256,
            "reference_pin_audit_path": (
                str(self.reference_pin_audit_path)
                if self.reference_pin_audit_path is not None
                else None
            ),
            "runtime_environment": self.runtime_environment,
            "runtime_environment_path": str(self.runtime_environment_path),
            "runtime_environment_sha256": self.runtime_environment_sha256,
            "artifact_sha256": self.artifact_sha256,
            "oracle_identities": json.loads(
                json.dumps(LEMAT_GENBENCH_ORACLE_IDENTITIES)
            ),
            "oracle_coverage": self.oracle_coverage,
        }


@dataclass(frozen=True)
class _CheckoutInfo:
    commit: str
    lock_sha256: str | None
    pyproject_sha256: str


@dataclass(frozen=True)
class _MaceCheckpointInfo:
    model: str
    path: Path
    sha256: str
    command: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(repo_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_path), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _git_path_list(
    result: subprocess.CompletedProcess[str], operation: str
) -> list[str]:
    if result.returncode != 0:
        raise LeMatGenBenchError(f"Could not {operation}: {result.stderr.strip()}")
    separator = "\0" if "\0" in result.stdout else "\n"
    return sorted(path for path in result.stdout.split(separator) if path)


def _inspect_checkout(
    repo_path: Path,
    *,
    expected_commit: str | None,
    expected_lock_sha256: str | None,
    require_clean_repo: bool,
) -> _CheckoutInfo:
    script_path = repo_path / "scripts" / "run_benchmarks.py"
    pyproject_path = repo_path / "pyproject.toml"
    if not script_path.is_file() or not pyproject_path.is_file():
        raise LeMatGenBenchError(
            f"{repo_path} is not a LeMat-GenBench source checkout: expected "
            "scripts/run_benchmarks.py and pyproject.toml"
        )

    revision = _run_git(repo_path, "rev-parse", "HEAD")
    if revision.returncode != 0:
        raise LeMatGenBenchError(
            f"Could not determine the LeMat-GenBench revision at {repo_path}: "
            f"{revision.stderr.strip()}"
        )
    actual_commit = revision.stdout.strip()
    if expected_commit is not None and actual_commit != expected_commit:
        raise LeMatGenBenchError(
            "LeMat-GenBench revision mismatch: "
            f"expected {expected_commit}, found {actual_commit} at {repo_path}"
        )

    if require_clean_repo:
        status = _run_git(
            repo_path,
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
        if status.returncode != 0:
            raise LeMatGenBenchError(
                f"Could not inspect the LeMat-GenBench worktree at {repo_path}: "
                f"{status.stderr.strip()}"
            )
        if status.stdout.strip():
            raise LeMatGenBenchError(
                "LeMat-GenBench has tracked worktree changes. Use a clean checkout "
                "or explicitly set require_clean_repo=False and record the diff."
            )

        untracked = _git_path_list(
            _run_git(
                repo_path,
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            ),
            f"inspect untracked files at {repo_path}",
        )
        unexpected_untracked = [
            path
            for path in untracked
            if not path.startswith(_ALLOWED_UNTRACKED_PREFIXES)
        ]
        ignored_under_code_roots = _git_path_list(
            _run_git(
                repo_path,
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "-z",
                "--",
                "scripts",
                "src",
            ),
            f"inspect ignored code/config files at {repo_path}",
        )
        ignored_code = [
            path
            for path in ignored_under_code_roots
            if not path.endswith((".pyc", ".pyo"))
            and "__pycache__/" not in path
            and path.endswith(_IGNORED_CODE_SUFFIXES)
        ]
        if unexpected_untracked or ignored_code:
            raise LeMatGenBenchError(
                "LeMat-GenBench has untracked files that can change code, config, "
                "or protocol inputs. Remove them or explicitly set "
                "require_clean_repo=False and record the checkout: "
                f"nonignored={unexpected_untracked}, ignored_code={ignored_code}"
            )

    lock_path = repo_path / "uv.lock"
    lock_sha256 = _sha256_file(lock_path) if lock_path.is_file() else None
    if expected_lock_sha256 is not None and lock_sha256 != expected_lock_sha256:
        raise LeMatGenBenchError(
            "LeMat-GenBench uv.lock mismatch: "
            f"expected {expected_lock_sha256}, found {lock_sha256!r} at {lock_path}"
        )
    return _CheckoutInfo(
        commit=actual_commit,
        lock_sha256=lock_sha256,
        pyproject_sha256=_sha256_file(pyproject_path),
    )


def _validate_path_component(value: str, label: str) -> None:
    if not _PATH_COMPONENT_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} must be a single path-safe name containing only letters, "
            f"numbers, '.', '_' and '-'; got {value!r}"
        )


def _normalize_runner_command(command: str | Sequence[str]) -> tuple[str, ...]:
    normalized = (command,) if isinstance(command, str) else tuple(command)
    if not normalized or any(
        not isinstance(part, str) or not part for part in normalized
    ):
        raise ValueError("runner_command must contain one or more non-empty strings")
    return normalized


def _reference_pin_shim_source(
    revisions: Mapping[str, str],
    *,
    expected_mace_checkpoint_sha256: str | None,
    mace_checkpoint_path: Path | None,
    expected_orb_checkpoint_sha256: str | None,
    expected_uma_checkpoint_sha256: str | None,
) -> str:
    revisions_json = json.dumps(dict(sorted(revisions.items())), separators=(",", ":"))
    expected_digests_json = json.dumps(
        {
            "mace": expected_mace_checkpoint_sha256,
            "orb": expected_orb_checkpoint_sha256,
            "uma": expected_uma_checkpoint_sha256,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    patched_apis_json = json.dumps(_REFERENCE_PIN_PATCHED_APIS, separators=(",", ":"))
    return f'''"""Generated by zevals; force audited Hugging Face revisions."""
import hashlib as _hashlib
import json as _json
import os as _os

import cached_path as _cached_path_module
import datasets as _datasets
import huggingface_hub as _huggingface_hub
import mace.calculators as _mace_calculators

ZEVALS_REFERENCE_REVISIONS = _json.loads({revisions_json!r})
ZEVALS_EXPECTED_ORACLE_DIGESTS = _json.loads({expected_digests_json!r})
ZEVALS_PATCHED_APIS = tuple(_json.loads({patched_apis_json!r}))
_REVISIONS_CASEFOLD = {{
    key.casefold(): (key, revision)
    for key, revision in ZEVALS_REFERENCE_REVISIONS.items()
}}
_AUDIT_PATH = _os.environ.get({_REFERENCE_PIN_AUDIT_ENV!r})
_ORIGINAL_LOAD_DATASET = _datasets.load_dataset
_ORIGINAL_HF_HUB_DOWNLOAD = _huggingface_hub.hf_hub_download
_ORIGINAL_SNAPSHOT_DOWNLOAD = _huggingface_hub.snapshot_download
_ORIGINAL_CACHED_PATH = _cached_path_module.cached_path
_ORIGINAL_MACE_MP = _mace_calculators.mace_mp
_ORB_CHECKPOINT_URL = {LEMAT_GENBENCH_ORB_CHECKPOINT_URL!r}
_MACE_MODEL = {LEMAT_GENBENCH_MACE_MODEL!r}
_MACE_CHECKPOINT_PATH = {str(mace_checkpoint_path) if mace_checkpoint_path is not None else None!r}
_UMA_REPOSITORY = {LEMAT_GENBENCH_UMA_REPOSITORY!r}
_UMA_CHECKPOINT_FILE = {LEMAT_GENBENCH_UMA_CHECKPOINT_FILE!r}


def _audit(event, **payload):
    if not _AUDIT_PATH:
        return
    record = {{"event": event, **payload}}
    with open(_AUDIT_PATH, "a", encoding="utf-8") as handle:
        handle.write(_json.dumps(record, sort_keys=True) + "\\n")


def _pinned_kwargs(api, repository, kwargs):
    if not isinstance(repository, str):
        return kwargs
    target = _REVISIONS_CASEFOLD.get(repository.casefold())
    if target is None:
        return kwargs
    canonical_repository, revision = target
    requested_revision = kwargs.get("revision")
    if requested_revision not in (None, revision):
        raise RuntimeError(
            f"zevals reference pin conflict for {{canonical_repository}}: "
            f"requested {{requested_revision}}, required {{revision}}"
        )
    pinned = dict(kwargs)
    pinned["revision"] = revision
    _audit(
        "pin",
        api=api,
        repository=canonical_repository,
        requested_revision=requested_revision,
        enforced_revision=revision,
    )
    return pinned


def _sha256_file(path):
    digest = _hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_artifact(oracle, identifier, resolved_path):
    path = _os.path.realpath(_os.fspath(resolved_path))
    if not _os.path.isfile(path):
        raise RuntimeError(
            f"zevals could not verify {{oracle}} artifact {{identifier}}: "
            f"download API returned non-file {{path}}"
        )
    actual_sha256 = _sha256_file(path)
    expected_sha256 = ZEVALS_EXPECTED_ORACLE_DIGESTS.get(oracle)
    matches_expected = (
        actual_sha256 == expected_sha256 if expected_sha256 is not None else None
    )
    _audit(
        "artifact",
        oracle=oracle,
        identifier=identifier,
        path=path,
        sha256=actual_sha256,
        expected_sha256=expected_sha256,
        matches_expected=matches_expected,
    )
    if matches_expected is False:
        raise RuntimeError(
            f"zevals {{oracle}} checkpoint digest mismatch for {{identifier}}: "
            f"expected {{expected_sha256}}, found {{actual_sha256}} at {{path}}"
        )
    return resolved_path


def _load_dataset(*args, **kwargs):
    repository = args[0] if args else kwargs.get("path")
    return _ORIGINAL_LOAD_DATASET(
        *args, **_pinned_kwargs("datasets.load_dataset", repository, kwargs)
    )


def _hf_hub_download(*args, **kwargs):
    repository = args[0] if args else kwargs.get("repo_id")
    pinned_kwargs = _pinned_kwargs(
        "huggingface_hub.hf_hub_download", repository, kwargs
    )
    resolved_path = _ORIGINAL_HF_HUB_DOWNLOAD(*args, **pinned_kwargs)
    target = (
        _REVISIONS_CASEFOLD.get(repository.casefold())
        if isinstance(repository, str)
        else None
    )
    if target is not None:
        filename = args[1] if len(args) > 1 else kwargs.get("filename")
        subfolder = kwargs.get("subfolder")
        relative_file = (
            f"{{subfolder.strip('/')}}/{{filename}}" if subfolder else filename
        )
        _audit(
            "resolved",
            api="huggingface_hub.hf_hub_download",
            repository=target[0],
            revision=target[1],
            file=relative_file,
            path=_os.path.realpath(_os.fspath(resolved_path)),
        )
        if (
            target[0].casefold() == _UMA_REPOSITORY.casefold()
            and relative_file == _UMA_CHECKPOINT_FILE
        ):
            return _verify_artifact("uma", _UMA_CHECKPOINT_FILE, resolved_path)
    return resolved_path


def _snapshot_download(*args, **kwargs):
    repository = args[0] if args else kwargs.get("repo_id")
    pinned_kwargs = _pinned_kwargs(
        "huggingface_hub.snapshot_download", repository, kwargs
    )
    resolved_path = _ORIGINAL_SNAPSHOT_DOWNLOAD(*args, **pinned_kwargs)
    target = (
        _REVISIONS_CASEFOLD.get(repository.casefold())
        if isinstance(repository, str)
        else None
    )
    if target is not None:
        _audit(
            "resolved",
            api="huggingface_hub.snapshot_download",
            repository=target[0],
            revision=target[1],
            path=_os.path.realpath(_os.fspath(resolved_path)),
        )
    return resolved_path


def _cached_path(*args, **kwargs):
    resource = args[0] if args else kwargs.get("url_or_filename")
    resolved_path = _ORIGINAL_CACHED_PATH(*args, **kwargs)
    if isinstance(resource, str) and resource == _ORB_CHECKPOINT_URL:
        return _verify_artifact("orb", _ORB_CHECKPOINT_URL, resolved_path)
    return resolved_path


def _mace_mp(*args, **kwargs):
    requested_model = args[0] if args else kwargs.get("model")
    if requested_model not in (None, _MACE_MODEL):
        raise RuntimeError(
            f"zevals MACE model conflict: requested {{requested_model}}, "
            f"required {{_MACE_MODEL}}"
        )
    pinned_args = args
    pinned_kwargs = dict(kwargs)
    if args:
        pinned_args = (_MACE_MODEL, *args[1:])
    else:
        pinned_kwargs["model"] = _MACE_MODEL
    _audit(
        "oracle_model",
        oracle="mace",
        requested_model=requested_model,
        enforced_model=_MACE_MODEL,
    )
    if _MACE_CHECKPOINT_PATH is not None:
        _verify_artifact("mace", _MACE_MODEL, _MACE_CHECKPOINT_PATH)
    return _ORIGINAL_MACE_MP(*pinned_args, **pinned_kwargs)


_datasets.load_dataset = _load_dataset
_huggingface_hub.hf_hub_download = _hf_hub_download
_huggingface_hub.snapshot_download = _snapshot_download
_cached_path_module.cached_path = _cached_path
_mace_calculators.mace_mp = _mace_mp
_audit(
    "loaded",
    interpreter_role=_os.environ.get({_REFERENCE_PIN_ROLE_ENV!r}),
    revisions=ZEVALS_REFERENCE_REVISIONS,
    expected_oracle_digests=ZEVALS_EXPECTED_ORACLE_DIGESTS,
    patched_apis=ZEVALS_PATCHED_APIS,
)
'''


def _write_reference_pin_shim(
    shim_dir: Path,
    revisions: Mapping[str, str],
    *,
    expected_mace_checkpoint_sha256: str | None,
    mace_checkpoint_path: Path | None,
    expected_orb_checkpoint_sha256: str | None,
    expected_uma_checkpoint_sha256: str | None,
) -> tuple[Path, str]:
    shim_dir.mkdir()
    shim_path = shim_dir / "sitecustomize.py"
    shim_path.write_text(
        _reference_pin_shim_source(
            revisions,
            expected_mace_checkpoint_sha256=expected_mace_checkpoint_sha256,
            mace_checkpoint_path=mace_checkpoint_path,
            expected_orb_checkpoint_sha256=expected_orb_checkpoint_sha256,
            expected_uma_checkpoint_sha256=expected_uma_checkpoint_sha256,
        ),
        encoding="utf-8",
    )
    return shim_path, _sha256_file(shim_path)


def _runtime_environment(
    device: str,
    *,
    cuda_visible_devices: str | None,
) -> dict[str, str]:
    """Build an explicit CPU/CUDA policy without leaking it to the parent."""
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"runtime_device must be 'cpu' or 'cuda', got {device!r}")
    if device == "cpu" and cuda_visible_devices is not None:
        raise ValueError("cuda_visible_devices cannot be set when runtime_device='cpu'")
    if cuda_visible_devices is not None and "\x00" in cuda_visible_devices:
        raise ValueError("cuda_visible_devices cannot contain a NUL character")

    environment = os.environ.copy()
    # An inherited import path can shadow the locked checkout's packages or
    # execute an unrelated sitecustomize before our audited hook.
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    if device == "cpu":
        # The pinned runner ignores its YAML device and branches exclusively on
        # torch.cuda.is_available(), so hiding CUDA is the only non-dirty way to
        # force its CPU path.
        environment["CUDA_VISIBLE_DEVICES"] = ""
    elif cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    # Make CUDA's logical ordering stable while respecting any scheduler-provided
    # visibility mask (or the explicit override above).
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return environment


def _reference_pin_environment(
    shim_dir: Path,
    audit_path: Path,
    *,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = (
        dict(base_environment) if base_environment is not None else os.environ.copy()
    )
    environment["PYTHONPATH"] = str(shim_dir)
    environment[_REFERENCE_PIN_AUDIT_ENV] = str(audit_path)
    return environment


def _environment_with_role(environment: Mapping[str, str], role: str) -> dict[str, str]:
    child_environment = dict(environment)
    child_environment[_REFERENCE_PIN_ROLE_ENV] = role
    return child_environment


def _run_preflight(
    command: Sequence[str],
    *,
    repo_path: Path,
    environment: Mapping[str, str] | None,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float | None,
) -> subprocess.CompletedProcess[str]:
    if environment and environment.get("ZEVALS_LEMAT_STREAM_LOGS") == "1":
        try:
            return run_streamed(
                command,
                cwd=repo_path,
                environment=environment,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                timeout_seconds=timeout_seconds,
            )
        except LeMatResumeTimeout as exc:
            raise LeMatGenBenchTimeout(str(exc)) from exc
    return subprocess.run(
        command,
        cwd=repo_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=dict(environment) if environment is not None else None,
    )


def _probe_runtime_environment(
    *,
    repo_path: Path,
    command_prefix: Sequence[str],
    environment: Mapping[str, str],
    expected_device: str,
    requested_cuda_visible_devices: str | None,
    output_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float | None,
) -> tuple[dict[str, Any], tuple[str, ...], str]:
    """Capture and validate the exact isolated Torch/CUDA execution context."""
    probe_source = f"""import json
import os
import platform
import sys
import torch

devices = []
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    devices.append({{
        "index": index,
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "multi_processor_count": properties.multi_processor_count,
        "uuid": str(getattr(properties, "uuid", "")) or None,
    }})
payload = {{
    "requested_device_policy": {expected_device!r},
    "requested_cuda_visible_devices": {requested_cuda_visible_devices!r},
    "effective_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
    "python_executable": sys.executable,
    "python_version": sys.version,
    "python_implementation": platform.python_implementation(),
    "platform": platform.platform(),
    "torch_version": torch.__version__,
    "torch_git_version": getattr(torch.version, "git_version", None),
    "torch_debug_build": torch.version.debug,
    "torch_cuda_build": torch.version.cuda,
    "torch_hip_build": torch.version.hip,
    "cudnn_version": torch.backends.cudnn.version(),
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_current_device": (
        torch.cuda.current_device() if torch.cuda.is_available() else None
    ),
    "cuda_devices": devices,
    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    "cudnn_deterministic": torch.backends.cudnn.deterministic,
    "cudnn_benchmark": torch.backends.cudnn.benchmark,
    "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    "torch_config": torch.__config__.show(),
}}
print({_RUNTIME_PROBE_RESULT_PREFIX!r} + json.dumps(payload, sort_keys=True))
"""
    command = [*command_prefix, "-c", probe_source]
    try:
        completed = _run_preflight(
            command,
            repo_path=repo_path,
            environment=environment,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=timeout_seconds,
        )
    except LeMatGenBenchTimeout:
        raise
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LeMatGenBenchError(
            f"Could not inspect the LeMat-GenBench runtime environment: {exc}"
        ) from exc
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise LeMatGenBenchError(
            "LeMat-GenBench runtime preflight failed with status "
            f"{completed.returncode}. See {stdout_path} and {stderr_path}."
        )

    payload: dict[str, Any] | None = None
    for line in reversed(completed.stdout.splitlines()):
        if not line.startswith(_RUNTIME_PROBE_RESULT_PREFIX):
            continue
        try:
            candidate = json.loads(line.removeprefix(_RUNTIME_PROBE_RESULT_PREFIX))
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if payload is None:
        raise LeMatGenBenchError(
            "Runtime preflight did not report structured Python/Torch provenance. "
            f"See {stdout_path} and {stderr_path}."
        )

    cuda_available = payload.get("cuda_available") is True
    cuda_device_count = payload.get("cuda_device_count")
    if expected_device == "cuda" and (
        not cuda_available
        or not isinstance(cuda_device_count, int)
        or cuda_device_count < 1
    ):
        raise LeMatGenBenchError(
            "runtime_device='cuda' was requested, but the isolated pinned "
            f"environment reported no usable CUDA device: {payload}"
        )
    if expected_device == "cpu" and cuda_available:
        raise LeMatGenBenchError(
            "runtime_device='cpu' was requested, but CUDA remained visible in the "
            f"isolated pinned environment: {payload}"
        )

    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload, tuple(command), _sha256_file(output_path)


def _verify_reference_pin_shim(
    *,
    repo_path: Path,
    command_prefix: Sequence[str],
    environment: Mapping[str, str],
    expected_shim_path: Path,
    expected_revisions: Mapping[str, str],
    expected_oracle_digests: Mapping[str, str | None],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float | None,
) -> tuple[str, ...]:
    probe_source = (
        "import json,sitecustomize\n"
        f"print({_REFERENCE_PIN_PROBE_PREFIX!r}+json.dumps({{"
        "'path':sitecustomize.__file__,"
        "'revisions':sitecustomize.ZEVALS_REFERENCE_REVISIONS,"
        "'expected_oracle_digests':"
        "sitecustomize.ZEVALS_EXPECTED_ORACLE_DIGESTS,"
        "'patched_apis':sitecustomize.ZEVALS_PATCHED_APIS"
        "},sort_keys=True))"
    )
    command = [*command_prefix, "-c", probe_source]
    try:
        completed = _run_preflight(
            command,
            repo_path=repo_path,
            environment=environment,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=timeout_seconds,
        )
    except LeMatGenBenchTimeout:
        raise
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LeMatGenBenchError(
            f"Could not verify the Hugging Face reference-pin shim: {exc}"
        ) from exc
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise LeMatGenBenchError(
            "Reference-pin shim preflight failed with status "
            f"{completed.returncode}. See {stdout_path} and {stderr_path}."
        )

    payload: Mapping[str, Any] | None = None
    for line in reversed(completed.stdout.splitlines()):
        if not line.startswith(_REFERENCE_PIN_PROBE_PREFIX):
            continue
        try:
            candidate = json.loads(line.removeprefix(_REFERENCE_PIN_PROBE_PREFIX))
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, Mapping):
            payload = candidate
            break
    expected_path = expected_shim_path.resolve()
    if payload is None or Path(str(payload.get("path", ""))).resolve() != expected_path:
        raise LeMatGenBenchError(
            "Python did not load the archived zevals reference-pin sitecustomize. "
            f"See {stdout_path} and {stderr_path}."
        )
    if payload.get("revisions") != dict(expected_revisions):
        raise LeMatGenBenchError(
            "The loaded reference-pin shim reported an unexpected revision mapping: "
            f"{payload.get('revisions')!r}"
        )
    if payload.get("expected_oracle_digests") != dict(expected_oracle_digests):
        raise LeMatGenBenchError(
            "The loaded reference-pin shim reported unexpected oracle digests: "
            f"{payload.get('expected_oracle_digests')!r}"
        )
    if payload.get("patched_apis") != list(_REFERENCE_PIN_PATCHED_APIS):
        raise LeMatGenBenchError(
            "The loaded reference-pin shim did not install every expected API hook: "
            f"{payload.get('patched_apis')!r}"
        )
    return tuple(command)


def _read_reference_pin_audit(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LeMatGenBenchError(
                f"Invalid reference-pin audit JSON on line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise LeMatGenBenchError(
                f"Reference-pin audit line {line_number} is not a JSON object"
            )
        records.append(value)
    return records


def _active_families(families: Sequence[str] | None) -> tuple[str, ...]:
    return _UPSTREAM_DEFAULT_FAMILIES if families is None else tuple(families)


def _needs_mace(families: Sequence[str] | None) -> bool:
    return bool({"distribution", "stability", "sun"} & set(_active_families(families)))


_CANONICAL_DISTRIBUTION_REFERENCE_FILES = (
    "data/lematbulk_jsdistance_distributions.json",
    "data/lematbulk_mmd_values_15k.pkl",
    "data/orb_mu.npy",
    "data/orb_sigma.npy",
    "data/mace_mu.npy",
    "data/mace_sigma.npy",
    "data/uma_mu.npy",
    "data/uma_sigma.npy",
)
_CANONICAL_NOVELTY_REFERENCE_FILES = ("data/all_compositions.npz",)
_CANONICAL_HULL_TYPES = ("orb_conserv_inf", "mace_mp", "uma")


def _canonical_reference_inventory(
    repo_path: Path, config_name: str
) -> dict[str, dict[str, Any]]:
    """Hash local protocol inputs that can replace a remote reference fetch."""
    relative_paths = [f"src/config/{config_name}.yaml"]
    if config_name == LEMAT_GENBENCH_CONFIG:
        relative_paths.extend(_CANONICAL_DISTRIBUTION_REFERENCE_FILES)
        relative_paths.extend(_CANONICAL_NOVELTY_REFERENCE_FILES)
        for hull_type in _CANONICAL_HULL_TYPES:
            relative_paths.extend(
                (
                    f"data/convex_hulls/{hull_type}_above_hull_dataset.parquet",
                    "data/convex_hulls/"
                    f"{hull_type}_above_hull_composition_matrix.npz",
                )
            )

    inventory: dict[str, dict[str, Any]] = {}
    for relative_path in relative_paths:
        path = repo_path / relative_path
        expected_sha256 = LEMAT_GENBENCH_LOCAL_REFERENCE_SHA256.get(relative_path)
        record: dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
            "is_file": path.is_file(),
        }
        if expected_sha256 is not None:
            record["expected_sha256"] = expected_sha256
        if record["is_file"]:
            actual_sha256 = _sha256_file(path)
            record.update(
                {
                    "size_bytes": path.stat().st_size,
                    "sha256": actual_sha256,
                }
            )
            if expected_sha256 is not None:
                record["matches_expected"] = actual_sha256 == expected_sha256
        elif expected_sha256 is not None:
            record["matches_expected"] = False
        inventory[relative_path] = record
    return dict(sorted(inventory.items()))


def _local_hull_overrides(
    inventory: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    return sorted(
        relative_path
        for relative_path, record in inventory.items()
        if relative_path.startswith("data/convex_hulls/") and bool(record.get("exists"))
    )


def _required_local_reference_paths(
    families: Sequence[str] | None,
) -> tuple[str, ...]:
    """Return canonical commit-bound files needed by selected families."""
    active = set(_active_families(families))
    required: set[str] = set()
    if "distribution" in active:
        required.update(_CANONICAL_DISTRIBUTION_REFERENCE_FILES)
    if {"novelty", "sun"} & active:
        required.update(_CANONICAL_NOVELTY_REFERENCE_FILES)
    return tuple(sorted(required))


def _local_reference_integrity_errors(
    inventory: Mapping[str, Mapping[str, Any]],
    *,
    families: Sequence[str] | None,
) -> dict[str, str]:
    """Find missing or digest-mismatched commit-bound protocol inputs."""
    errors: dict[str, str] = {}
    for relative_path in _required_local_reference_paths(families):
        record = inventory.get(relative_path, {})
        if not bool(record.get("exists")):
            errors[relative_path] = "required ordinary Git blob is missing"
        elif record.get("matches_expected") is not True:
            errors[relative_path] = (
                "SHA-256 mismatch: expected "
                f"{record.get('expected_sha256')}, found {record.get('sha256')}"
            )
    return dict(sorted(errors.items()))


def _local_reference_changes(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Report protocol-reference filesystem changes during a benchmark run."""
    changes: dict[str, dict[str, Any]] = {}
    for relative_path in sorted(set(before) | set(after)):
        before_identity = {
            key: before.get(relative_path, {}).get(key)
            for key in ("exists", "is_file", "sha256")
        }
        after_identity = {
            key: after.get(relative_path, {}).get(key)
            for key in ("exists", "is_file", "sha256")
        }
        if before_identity != after_identity:
            changes[relative_path] = {
                "before": before_identity,
                "after": after_identity,
            }
    return changes


def _canonical_reference_access_plan(
    *,
    config_name: str,
    families: Sequence[str] | None,
    valid_structure_count: int | None,
    local_inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe which pinned repositories must be observed at runtime."""
    active = set(_active_families(families))
    expected: dict[str, str] = {}
    not_expected: dict[str, str] = {}

    if config_name != LEMAT_GENBENCH_CONFIG:
        reason = (
            "custom configuration: the shim pins matching calls, but zevals cannot "
            "infer which repositories that configuration must access"
        )
        return {
            "assertions_supported": False,
            "expected_runtime_repositories": expected,
            "not_expected_at_runtime": {
                repository: reason for repository in LEMAT_GENBENCH_REFERENCE_REVISIONS
            },
        }

    if valid_structure_count == 0:
        no_valid_reason = "no valid structures; upstream skips remaining benchmarks"
    else:
        no_valid_reason = ""
    may_have_valid_structures = (
        valid_structure_count is None or valid_structure_count > 0
    )

    bulk_repository = "LeMaterial/LeMat-Bulk"
    if may_have_valid_structures and {"novelty", "sun"} & active:
        expected[bulk_repository] = (
            "structure-matcher novelty/SUN loads the LeMat-Bulk compatible_pbe "
            "dataset at runtime"
        )
    else:
        not_expected[bulk_repository] = no_valid_reason or (
            "neither novelty nor SUN was selected"
        )

    hull_repository = "LeMaterial/LeMat-Bulk-MLIP-Hull"
    hull_relative_paths = [
        relative_path
        for relative_path in local_inventory
        if relative_path.startswith("data/convex_hulls/")
    ]
    local_hulls_complete = bool(hull_relative_paths) and all(
        bool(local_inventory[path].get("exists")) for path in hull_relative_paths
    )
    if may_have_valid_structures and {"stability", "sun"} & active:
        if local_hulls_complete:
            not_expected[hull_repository] = (
                "all six ORB/MACE/UMA hull dataset and composition-matrix files "
                "were present locally; this unverified override is permitted only "
                "because reference-revision enforcement was disabled"
            )
        else:
            expected[hull_repository] = (
                "stability/SUN needs at least one missing MLIP-specific hull file"
            )
    else:
        not_expected[hull_repository] = no_valid_reason or (
            "neither stability nor SUN was selected"
        )

    embeddings_repository = "LeMaterial/LeMat-GenBench-embeddings"
    not_expected[embeddings_repository] = (
        "the canonical runner reads committed data/{orb,mace,uma}_{mu,sigma}.npy "
        "statistics; the embeddings dataset is only used by the separate reference-"
        "statistics generation utility"
    )

    if may_have_valid_structures and _needs_mace(families):
        expected[LEMAT_GENBENCH_UMA_REPOSITORY] = (
            "the multi-MLIP preprocessor loads uma-s-1 through fairchem"
        )
    else:
        not_expected[LEMAT_GENBENCH_UMA_REPOSITORY] = no_valid_reason or (
            "no distribution, stability, or SUN family was selected"
        )

    return {
        "assertions_supported": True,
        "expected_runtime_repositories": dict(sorted(expected.items())),
        "not_expected_at_runtime": dict(sorted(not_expected.items())),
    }


def _reference_pin_audit_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    access_plan: Mapping[str, Any],
    require_oracle_artifacts: bool,
    expected_oracle_digests: Mapping[str, str | None],
) -> tuple[dict[str, Any], str | None]:
    """Validate interpreter hooks, repository pins, and oracle artifacts."""
    loaded_events = [
        dict(record)
        for record in records
        if record.get("event") == "loaded"
        and record.get("revisions") == dict(LEMAT_GENBENCH_REFERENCE_REVISIONS)
        and record.get("patched_apis") == list(_REFERENCE_PIN_PATCHED_APIS)
        and record.get("expected_oracle_digests") == dict(expected_oracle_digests)
    ]
    loaded_roles = sorted(
        {
            str(record.get("interpreter_role"))
            for record in loaded_events
            if record.get("interpreter_role") is not None
        }
    )
    observed_pin_events: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("event") != "pin":
            continue
        repository = record.get("repository")
        if not isinstance(repository, str):
            continue
        expected_revision = LEMAT_GENBENCH_REFERENCE_REVISIONS.get(repository)
        if record.get("enforced_revision") != expected_revision:
            continue
        observed_pin_events.setdefault(repository, []).append(dict(record))

    expected_repositories = access_plan.get("expected_runtime_repositories", {})
    missing_repositories = sorted(
        repository
        for repository in expected_repositories
        if repository not in observed_pin_events
    )

    oracle_artifacts: dict[str, list[dict[str, Any]]] = {
        "mace": [],
        "orb": [],
        "uma": [],
    }
    mismatched_artifacts: list[str] = []
    for record in records:
        if record.get("event") != "artifact":
            continue
        oracle = record.get("oracle")
        if oracle not in oracle_artifacts:
            continue
        oracle_artifacts[oracle].append(dict(record))
        if record.get("matches_expected") is False:
            mismatched_artifacts.append(str(oracle))

    required_oracle_artifacts = (
        sorted(
            oracle
            for oracle, expected_digest in expected_oracle_digests.items()
            if expected_digest is not None
        )
        if require_oracle_artifacts
        else []
    )
    missing_oracle_artifacts = sorted(
        oracle
        for oracle in required_oracle_artifacts
        if not oracle_artifacts.get(oracle)
    )
    mace_model_events = [
        dict(record)
        for record in records
        if record.get("event") == "oracle_model"
        and record.get("oracle") == "mace"
        and record.get("enforced_model") == LEMAT_GENBENCH_MACE_MODEL
    ]
    summary = {
        **dict(access_plan),
        "loaded_interpreter_events": len(loaded_events),
        "loaded_interpreter_roles": loaded_roles,
        "observed_pinned_repositories": dict(sorted(observed_pin_events.items())),
        "missing_expected_repositories": missing_repositories,
        "oracle_artifacts": oracle_artifacts,
        "required_oracle_artifacts": required_oracle_artifacts,
        "missing_oracle_artifacts": missing_oracle_artifacts,
        "mismatched_oracle_artifacts": sorted(set(mismatched_artifacts)),
        "mace_model_invocations": mace_model_events,
    }

    errors: list[str] = []
    missing_interpreter_roles = sorted(
        {"reference-pin-preflight", "benchmark"} - set(loaded_roles)
    )
    if missing_interpreter_roles:
        errors.append(
            "audit does not prove that required interpreters loaded every reference-"
            f"pin hook; missing roles {missing_interpreter_roles}"
        )
    if missing_repositories:
        errors.append(
            "no pinned API call was observed for required repositories "
            f"{missing_repositories}"
        )
    if missing_oracle_artifacts:
        errors.append(
            "no checkpoint artifact was observed for active oracles "
            f"{missing_oracle_artifacts}"
        )
    if mismatched_artifacts:
        errors.append(
            "checkpoint digest mismatch was reported for oracles "
            f"{sorted(set(mismatched_artifacts))}"
        )
    if require_oracle_artifacts and not mace_model_events:
        errors.append(
            "no benchmark-time invocation of the pinned MACE model was observed"
        )
    return summary, "; ".join(errors) if errors else None


def _oracle_coverage_summary(
    metrics: Mapping[str, float],
    *,
    config_name: str,
    families: Sequence[str] | None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Read per-MLIP stability coverage from public upstream final scores."""
    active = set(_active_families(families))
    valid_count_value = metrics.get("validity/overall_validity_count")
    if valid_count_value is None:
        valid_count_value = metrics.get("validity_filtering/valid_structures")
    valid_count = (
        int(valid_count_value)
        if isinstance(valid_count_value, Real)
        and not isinstance(valid_count_value, bool)
        and valid_count_value >= 0
        and float(valid_count_value).is_integer()
        else None
    )

    summary: dict[str, Any] = {
        "configured_oracles": json.loads(json.dumps(LEMAT_GENBENCH_ORACLE_IDENTITIES)),
        "upstream_min_mlips_required": 2,
        "upstream_require_all_mlips": False,
        "valid_structures": valid_count,
        "per_mlip_e_above_hull_counts": {},
        "status": "not_applicable",
        "all_three_complete": None,
    }
    coverage_metrics: dict[str, float] = {}
    if valid_count is not None:
        coverage_metrics["oracle_coverage/expected_valid_structures"] = float(
            valid_count
        )

    if config_name != LEMAT_GENBENCH_CONFIG:
        summary["status"] = "unsupported_custom_config"
        return summary, coverage_metrics
    if not {"distribution", "stability", "sun"} & active or valid_count == 0:
        return summary, coverage_metrics
    if "stability" not in active:
        summary["status"] = "not_reported_without_stability_family"
        return summary, coverage_metrics

    counts: dict[str, int] = {}
    for oracle in ("orb", "mace", "uma"):
        suffix = f"n_valid_structures_{oracle}"
        candidates = sorted(
            (key, value)
            for key, value in metrics.items()
            if key.startswith("stability/") and key.endswith(suffix)
        )
        candidate_value = candidates[0][1] if len(candidates) == 1 else None
        if (
            isinstance(candidate_value, Real)
            and not isinstance(candidate_value, bool)
            and candidate_value >= 0
            and float(candidate_value).is_integer()
        ):
            count = int(candidate_value)
            counts[oracle] = count
            coverage_metrics[f"oracle_coverage/{oracle}_structures"] = float(count)

    summary["per_mlip_e_above_hull_counts"] = counts
    if len(counts) != 3 or valid_count is None:
        summary["status"] = "coverage_counts_missing"
        return summary, coverage_metrics

    all_three_complete = all(count == valid_count for count in counts.values())
    summary["all_three_complete"] = all_three_complete
    summary["status"] = "complete" if all_three_complete else "incomplete"
    coverage_metrics["oracle_coverage/all_three_complete"] = float(all_three_complete)
    return summary, coverage_metrics


def _probe_mace_checkpoint(
    *,
    repo_path: Path,
    command_prefix: Sequence[str],
    model: str,
    expected_sha256: str | None,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float | None,
    environment: Mapping[str, str] | None = None,
) -> _MaceCheckpointInfo:
    """Resolve and hash an explicit MACE checkpoint in the upstream env."""
    probe_source = (
        "import hashlib,json\n"
        "from pathlib import Path\n"
        "from mace.calculators.foundations_models import "
        "download_mace_mp_checkpoint\n"
        f"p=Path(download_mace_mp_checkpoint({model!r})).resolve()\n"
        "with p.open('rb') as f:\n"
        " h=hashlib.file_digest(f,'sha256').hexdigest()\n"
        f"print({_MACE_PROBE_RESULT_PREFIX!r}+json.dumps({{'path':str(p),'sha256':h}}))"
    )
    command = [*command_prefix, "-c", probe_source]
    try:
        completed = _run_preflight(
            command,
            repo_path=repo_path,
            environment=environment,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=timeout_seconds,
        )
    except LeMatGenBenchTimeout:
        raise
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LeMatGenBenchError(
            f"Could not resolve the pinned MACE checkpoint: {exc}"
        ) from exc

    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise LeMatGenBenchError(
            f"MACE checkpoint preflight exited with status {completed.returncode}. "
            f"See {stdout_path} and {stderr_path}."
        )

    payload: Mapping[str, Any] | None = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(_MACE_PROBE_RESULT_PREFIX):
            try:
                candidate = json.loads(line.removeprefix(_MACE_PROBE_RESULT_PREFIX))
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, Mapping):
                payload = candidate
                break
    if payload is None:
        raise LeMatGenBenchError(
            "MACE checkpoint preflight did not report a parseable checkpoint. "
            f"See {stdout_path} and {stderr_path}."
        )

    checkpoint_path = Path(str(payload.get("path", ""))).expanduser().resolve()
    checkpoint_sha256 = payload.get("sha256")
    if not checkpoint_path.is_file() or not isinstance(checkpoint_sha256, str):
        raise LeMatGenBenchError(
            "MACE checkpoint preflight returned an invalid path or digest: "
            f"{dict(payload)!r}"
        )
    if _sha256_file(checkpoint_path) != checkpoint_sha256:
        raise LeMatGenBenchError(
            f"MACE checkpoint changed during preflight: {checkpoint_path}"
        )
    if expected_sha256 is not None and checkpoint_sha256 != expected_sha256:
        raise LeMatGenBenchError(
            "MACE checkpoint digest mismatch for "
            f"{model}: expected {expected_sha256}, found {checkpoint_sha256} at "
            f"{checkpoint_path}"
        )
    return _MaceCheckpointInfo(
        model=model,
        path=checkpoint_path,
        sha256=checkpoint_sha256,
        command=tuple(command),
    )


def _write_cif_list(cif_paths: Sequence[Path], destination: Path) -> Path:
    destination.write_text(
        "".join(f"{path.resolve()}\n" for path in cif_paths),
        encoding="utf-8",
    )
    return destination


def _material_cif_source() -> str:
    """Read the standalone converter that will run in the oracle interpreter."""
    return Path(__file__).with_name("_material_cif.py").read_text(encoding="utf-8")


def _cif_loader_bootstrap_source(normalizer_source: str) -> str:
    """Normalize staged mmCIFs in memory, retaining upstream parsing semantics.

    The bootstrap is archived verbatim in the adapter's hash-bound command.
    Only paths in the staged input manifest are eligible; reference structures,
    relaxation POSCARs, and ordinary materials CIFs use the original reader.
    """
    return f"""import gzip
import runpy
import sys
from pathlib import Path
from pymatgen.core import Structure

_normalizer = {{"__name__": "_zevals_material_cif"}}
exec(compile({normalizer_source!r}, "<zevals-material-cif>", "exec"), _normalizer)
_normalize = _normalizer["normalize_material_cif"]
sys.argv = sys.argv[1:]
_manifest = Path(sys.argv[sys.argv.index("--cifs") + 1])
_staged_paths = {{Path(line).resolve() for line in _manifest.read_text().splitlines()}}
_original_from_file = Structure.from_file.__func__

def _from_file(cls, filename, *args, **kwargs):
    path = Path(filename)
    if path.resolve() not in _staged_paths:
        return _original_from_file(cls, filename, *args, **kwargs)
    original = path.read_bytes()
    normalized = _normalize(path)
    if normalized == original:
        return _original_from_file(cls, filename, *args, **kwargs)
    if path.name.endswith(".gz"):
        normalized = gzip.decompress(normalized)
    return cls.from_str(normalized.decode("utf-8"), "cif", *args, **kwargs)

Structure.from_file = classmethod(_from_file)
sys.path[0] = str(Path(sys.argv[0]).resolve().parent)
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    Structure.from_file = classmethod(_original_from_file)
"""


def _prepare_cif_input(
    generated_structures: Sequence[Structure] | str | Path,
    output_dir: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    """Stage immutable CIF bytes and return their deterministic manifest."""
    cif_dir = output_dir / "input_cifs"
    if cif_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite staged LeMat-GenBench inputs: {cif_dir}"
        )

    source_paths: list[Path | None]
    if isinstance(generated_structures, (str, Path)):
        input_path = Path(generated_structures).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(
                f"LeMat-GenBench input does not exist: {input_path}"
            )

        if input_path.is_dir():
            discovered_paths = sorted(
                {
                    path.resolve()
                    for pattern in ("*.cif", "*.cif.gz")
                    for path in input_path.rglob(pattern)
                    if path.is_file()
                }
            )
        elif input_path.name.lower().endswith((".cif", ".cif.gz")):
            discovered_paths = [input_path]
        else:
            discovered_paths = []
            for raw_line in input_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                candidate = Path(line).expanduser()
                if not candidate.is_absolute():
                    candidate = input_path.parent / candidate
                candidate = candidate.resolve()
                if not candidate.is_file():
                    raise FileNotFoundError(
                        f"CIF listed by {input_path} does not exist: {candidate}"
                    )
                discovered_paths.append(candidate)

        # A repeated path is one saved generation artifact, not multiple
        # independent attempts. Resolve, deduplicate, and sort before copying.
        source_paths = [
            *sorted(
                {path.resolve() for path in discovered_paths},
                key=lambda path: str(path),
            )
        ]
        if not source_paths:
            raise ValueError("LeMat-GenBench requires at least one CIF structure")
        cif_dir.mkdir()
        cif_paths = []
        for index, source_path in enumerate(source_paths):
            extension = (
                ".cif.gz" if source_path.name.lower().endswith(".cif.gz") else ".cif"
            )
            staged_path = cif_dir / f"{index:06d}{extension}"
            shutil.copyfile(source_path, staged_path)
            cif_paths.append(staged_path)
    else:
        structures = list(generated_structures)
        cif_dir.mkdir()
        cif_paths = []
        source_paths = []
        for index, structure in enumerate(structures):
            cif_path = cif_dir / f"{index:06d}.cif"
            structure.to(filename=str(cif_path), fmt="cif")
            cif_paths.append(cif_path)
            source_paths.append(None)

    if not cif_paths:
        raise ValueError("LeMat-GenBench requires at least one CIF structure")

    # Always give upstream an explicitly sorted manifest. Its directory loader
    # uses filesystem traversal order, which is not a reproducible ordering.
    cif_paths = [path.resolve() for path in cif_paths]
    upstream_input = _write_cif_list(cif_paths, output_dir / "cif_paths.txt")

    input_hashes = [
        {
            "path": str(path),
            "source_path": str(source) if source is not None else None,
            "sha256": _sha256_file(path),
        }
        for path, source in zip(cif_paths, source_paths, strict=True)
    ]
    return upstream_input, input_hashes


def _rehash_staged_inputs(input_records: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Record post-run hashes and report any CIF mutation during evaluation."""
    errors: dict[str, str] = {}
    for record in input_records:
        path = Path(str(record["path"]))
        if not path.is_file():
            record["post_run_sha256"] = None
            errors[str(path)] = "staged CIF disappeared during evaluation"
            continue
        actual_sha256 = _sha256_file(path)
        record["post_run_sha256"] = actual_sha256
        if actual_sha256 != record["sha256"]:
            errors[str(path)] = (
                f"staged CIF changed from {record['sha256']} to {actual_sha256}"
            )
    return dict(sorted(errors.items()))


class _NumpyLiteralNormalizer(ast.NodeTransformer):
    """Reduce NumPy scalar reprs in upstream dataclass strings to literals."""

    _SCALAR_NAMES = {
        "bool_",
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"np", "numpy"}
            and node.func.attr in self._SCALAR_NAMES
            and len(node.args) == 1
            and not node.keywords
        ):
            return node.args[0]
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        if node.id == "nan":
            return ast.copy_location(ast.Constant(float("nan")), node)
        if node.id == "inf":
            return ast.copy_location(ast.Constant(float("inf")), node)
        return node


def _literal_eval_upstream_node(node: ast.AST) -> Any:
    expression = ast.Expression(body=node)
    normalized = _NumpyLiteralNormalizer().visit(expression)
    ast.fix_missing_locations(normalized)
    return ast.literal_eval(normalized)


def _balanced_mapping_after(text: str, marker: str) -> str | None:
    """Extract a brace-balanced mapping repr after ``marker``.

    Some upstream evaluator fields contain object reprs that are not valid
    Python syntax, so parsing the complete ``BenchmarkResult(...)`` can fail.
    The final-score mapping itself is regular Python/NumPy scalar syntax.
    """
    marker_index = text.find(marker)
    if marker_index < 0:
        return None
    start = text.find("{", marker_index + len(marker))
    if start < 0:
        return None

    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _scores_from_result_repr(value: str) -> Mapping[str, Any]:
    """Safely extract ``final_scores`` from upstream's dataclass repr."""
    try:
        expression = ast.parse(value, mode="eval").body
    except SyntaxError:
        expression = None
    if isinstance(expression, ast.Call):
        for keyword in expression.keywords:
            if keyword.arg == "final_scores":
                try:
                    scores = _literal_eval_upstream_node(keyword.value)
                except (ValueError, TypeError, SyntaxError):
                    break
                return scores if isinstance(scores, Mapping) else {}

    mapping_repr = _balanced_mapping_after(value, "final_scores=")
    if mapping_repr is not None:
        try:
            mapping_node = ast.parse(mapping_repr, mode="eval").body
            scores = _literal_eval_upstream_node(mapping_node)
        except (ValueError, TypeError, SyntaxError):
            return {}
        return scores if isinstance(scores, Mapping) else {}
    return {}


def _final_scores(value: Any) -> Mapping[str, Any]:
    if isinstance(value, str):
        return _scores_from_result_repr(value)
    if not isinstance(value, Mapping):
        return {}
    scores = value.get("final_scores")
    if isinstance(scores, Mapping):
        return scores
    if "error" in value:
        return {}
    return value


def _flatten_numeric(destination: dict[str, float], prefix: str, value: Any) -> None:
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            _flatten_numeric(destination, child_prefix, value[key])
    elif isinstance(value, Real) and not isinstance(value, bool):
        destination[prefix] = float(value)


def extract_lematgenbench_metrics(raw_result: Mapping[str, Any]) -> dict[str, float]:
    """Extract a deterministic numeric map from an upstream result document.

    Current LeMat-GenBench JSON serializes each ``BenchmarkResult`` as its
    dataclass string representation. This function safely reads only the
    ``final_scores`` AST node. It also accepts a future structured
    ``{"final_scores": ...}`` representation without changing output keys.

    Args:
        raw_result: Parsed complete JSON document from LeMat-GenBench.

    Returns:
        Sorted ``family/metric`` scalar values, numeric validity-filtering
        metadata under ``validity_filtering/...``, and clearly labelled
        ``attempt_normalized/...`` rates derived from counts over the original
        input size. Native upstream ratios are retained unchanged.
    """
    flattened: dict[str, float] = {}
    validity_filtering = raw_result.get("validity_filtering")
    if isinstance(validity_filtering, Mapping):
        _flatten_numeric(flattened, "validity_filtering", validity_filtering)

    results = raw_result.get("results")
    if isinstance(results, Mapping):
        for family in sorted(results, key=lambda item: str(item)):
            _flatten_numeric(flattened, str(family), _final_scores(results[family]))
    flattened.update(derive_attempt_normalized_rates(flattened))
    return dict(sorted(flattened.items()))


def _family_errors(
    raw_result: Mapping[str, Any], required_families: Sequence[str] = ()
) -> dict[str, str]:
    errors: dict[str, str] = {}
    results = raw_result.get("results")
    if not isinstance(results, Mapping):
        return {"results": "upstream JSON has no results mapping"}
    for family in required_families:
        if family not in results:
            errors[family] = "upstream JSON omitted the requested family"
    for family, value in results.items():
        family_name = str(family)
        if isinstance(value, Mapping) and "error" in value:
            errors[family_name] = str(value["error"])
        elif not _final_scores(value):
            errors[family_name] = "could not parse upstream final_scores"
    return dict(sorted(errors.items()))


def _upstream_loaded_count(raw_result: Mapping[str, Any]) -> int:
    """Read and validate upstream's post-CIF-parsing structure count."""
    validity_filtering = raw_result.get("validity_filtering")
    if not isinstance(validity_filtering, Mapping):
        raise LeMatGenBenchError(
            "Upstream result has no validity_filtering mapping, so the number "
            "of successfully parsed CIFs cannot be verified"
        )
    value = validity_filtering.get("total_input_structures")
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not float(value).is_integer()
        or value < 0
    ):
        raise LeMatGenBenchError(
            "Upstream validity_filtering.total_input_structures must be a "
            f"non-negative integer, got {value!r}"
        )
    loaded_count = int(value)

    run_info = raw_result.get("run_info")
    if isinstance(run_info, Mapping) and "n_structures" in run_info:
        run_info_count = run_info["n_structures"]
        if (
            not isinstance(run_info_count, Real)
            or isinstance(run_info_count, bool)
            or not float(run_info_count).is_integer()
            or int(run_info_count) != loaded_count
        ):
            raise LeMatGenBenchError(
                "Upstream result reports inconsistent loaded structure counts: "
                "validity_filtering.total_input_structures="
                f"{loaded_count}, run_info.n_structures={run_info_count!r}"
            )

    results = raw_result.get("results")
    if isinstance(results, Mapping) and "validity" in results:
        validity_total = _final_scores(results["validity"]).get("total_structures")
        if validity_total is not None and (
            not isinstance(validity_total, Real)
            or isinstance(validity_total, bool)
            or not float(validity_total).is_integer()
            or int(validity_total) != loaded_count
        ):
            raise LeMatGenBenchError(
                "Upstream result reports inconsistent loaded structure counts: "
                "validity_filtering.total_input_structures="
                f"{loaded_count}, validity.total_structures={validity_total!r}"
            )
    return loaded_count


def _first_metric(metrics: Mapping[str, float], *keys: str) -> float:
    for key in keys:
        if key in metrics:
            return float(metrics[key])
    raise KeyError(f"None of the required LeMat-GenBench keys are present: {keys}")


def _count_rate(
    metrics: Mapping[str, float], count_keys: Sequence[str], total: float
) -> float:
    if total <= 0:
        raise ValueError(
            f"LeMat-GenBench total structure count must be positive, got {total}"
        )
    return _first_metric(metrics, *count_keys) / total


def derive_attempt_normalized_rates(
    metrics: Mapping[str, float],
    *,
    total_attempts: int | None = None,
) -> dict[str, float]:
    """Derive publication rates from counts divided by all generation attempts.

    The upstream runner filters invalid samples before downstream families, so
    native family ratios may use the valid subset. This helper retains those
    native values and adds separately named rates with the original number of
    input CIFs as the denominator.

    Args:
        metrics: Flattened native metrics. Partial-family results are accepted.
        total_attempts: Explicit attempted-sample denominator. When omitted,
            use the input count reported by upstream.

    Returns:
        Available attempt-normalized rates under an explicit namespace.
    """
    if total_attempts is None:
        try:
            total = _first_metric(
                metrics,
                "validity_filtering/total_input_structures",
                "validity/total_structures",
            )
        except KeyError:
            return {}
    else:
        total = float(total_attempts)
    if total <= 0:
        return {}

    count_keys = {
        "attempt_normalized/overall_valid_rate": (
            "validity/overall_validity_count",
            "validity_filtering/valid_structures",
        ),
        "attempt_normalized/unique_rate": ("uniqueness/unique_structures_count",),
        "attempt_normalized/novel_rate": ("novelty/novel_structures_count",),
        "attempt_normalized/stable_rate": (
            "stability/stable_count",
            "sun/stable_count",
        ),
        "attempt_normalized/metastable_rate": (
            "sun/metastable_count",
            "stability/metastable_count",
        ),
        "attempt_normalized/sun_rate": ("sun/sun_count",),
        "attempt_normalized/metasun_rate": ("sun/msun_count",),
    }
    rates: dict[str, float] = {}
    for output_key, candidates in count_keys.items():
        try:
            rates[output_key] = _count_rate(metrics, candidates, total)
        except KeyError:
            continue
    return rates


def select_table2_metrics(
    metrics: Mapping[str, float],
    *,
    distribution_key: str,
    diversity_key: str,
) -> dict[str, float]:
    """Select the eight Zatom-2 Table 2 material columns.

    Upstream exposes three distribution scores and four diversity scores, not
    one canonical aggregate for either family. Callers must explicitly freeze
    one key from :data:`TABLE2_DISTRIBUTION_KEYS` and one key from
    :data:`TABLE2_DIVERSITY_KEYS`; the choices can then be recorded alongside
    the table.

    Args:
        metrics: Flattened output from :func:`extract_lematgenbench_metrics`.
        distribution_key: One full key from ``TABLE2_DISTRIBUTION_KEYS``.
        diversity_key: One full key from ``TABLE2_DIVERSITY_KEYS``.

    Returns:
        Table-column name to scalar mapping using upstream definitions. Rates
        are fractions in ``[0, 1]``. As in the published LeMat-GenBench tables,
        all count-based rates use the original input count as denominator, not
        the valid-only subset passed to downstream benchmark families.
    """
    if distribution_key not in TABLE2_DISTRIBUTION_KEYS:
        raise ValueError(
            f"distribution_key must be one of {TABLE2_DISTRIBUTION_KEYS}; "
            f"got {distribution_key!r}"
        )
    if diversity_key not in TABLE2_DIVERSITY_KEYS:
        raise ValueError(
            f"diversity_key must be one of {TABLE2_DIVERSITY_KEYS}; "
            f"got {diversity_key!r}"
        )

    return {
        "overall_valid": _first_metric(
            metrics, "attempt_normalized/overall_valid_rate"
        ),
        "distribution": _first_metric(metrics, distribution_key),
        "diversity": _first_metric(metrics, diversity_key),
        "unique": _first_metric(metrics, "attempt_normalized/unique_rate"),
        "novel": _first_metric(metrics, "attempt_normalized/novel_rate"),
        "stable": _first_metric(metrics, "attempt_normalized/stable_rate"),
        "sun": _first_metric(metrics, "attempt_normalized/sun_rate"),
        "metasun": _first_metric(metrics, "attempt_normalized/metasun_rate"),
    }


def _result_inventory(results_dir: Path, prefix: str) -> dict[Path, tuple[int, int]]:
    return {
        path.resolve(): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in results_dir.glob(f"{prefix}_*.json")
        if path.is_file()
    }


def _path_reported_by_upstream(output: str, repo_path: Path) -> Path | None:
    for raw_line in reversed(output.splitlines()):
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw_line).strip()
        match = _RESULT_PATH_PATTERN.search(line)
        if match is None:
            continue
        candidate = Path(match.group(1).strip()).expanduser()
        if not candidate.is_absolute():
            candidate = repo_path / candidate
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    return None


def _locate_upstream_result(
    *,
    repo_path: Path,
    result_prefix: str,
    inventory_before: Mapping[Path, tuple[int, int]],
    stdout: str,
    stderr: str,
) -> Path:
    reported = _path_reported_by_upstream(f"{stdout}\n{stderr}", repo_path)
    if reported is not None:
        return reported

    inventory_after = _result_inventory(repo_path / "results_final", result_prefix)
    changed = [
        path
        for path, metadata in inventory_after.items()
        if inventory_before.get(path) != metadata
    ]
    if len(changed) == 1:
        return changed[0]
    if not changed:
        raise LeMatGenBenchError(
            "LeMat-GenBench exited successfully but did not report or create a "
            "result JSON file"
        )
    raise LeMatGenBenchError(
        "LeMat-GenBench created multiple matching result files; use a unique "
        f"run_name. Candidates: {sorted(str(path) for path in changed)}"
    )


def _ensure_new_artifacts(
    output_dir: Path, *, allow_existing: bool = False
) -> dict[str, Path]:
    artifacts = {
        "raw": output_dir / "lematgenbench_raw.json",
        "metrics": output_dir / "lematgenbench_metrics.json",
        "manifest": output_dir / "lematgenbench_manifest.json",
        "stdout": output_dir / "lematgenbench.stdout.log",
        "stderr": output_dir / "lematgenbench.stderr.log",
        "lock": output_dir / "lematgenbench.uv.lock",
        "cif_manifest": output_dir / "cif_paths.txt",
        "mace_stdout": output_dir / "lematgenbench_mace_preflight.stdout.log",
        "mace_stderr": output_dir / "lematgenbench_mace_preflight.stderr.log",
        "runtime": output_dir / "lematgenbench_runtime.json",
        "runtime_stdout": output_dir / "lematgenbench_runtime_preflight.stdout.log",
        "runtime_stderr": output_dir / "lematgenbench_runtime_preflight.stderr.log",
        "reference_shim_dir": output_dir / "lematgenbench_reference_pin_shim",
        "reference_audit": output_dir / "lematgenbench_reference_pin_audit.jsonl",
        "reference_stdout": (
            output_dir / "lematgenbench_reference_pin_preflight.stdout.log"
        ),
        "reference_stderr": (
            output_dir / "lematgenbench_reference_pin_preflight.stderr.log"
        ),
    }
    existing = [path for path in artifacts.values() if path.exists()]
    if existing and not allow_existing:
        raise FileExistsError(
            "Refusing to overwrite existing LeMat-GenBench artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    return artifacts


def _artifact_digest_map(
    artifacts: Mapping[str, Path],
    *,
    reference_shim_path: Path | None,
) -> dict[str, str]:
    """Hash every persisted top-level protocol artifact that was produced."""
    digests = {
        key: _sha256_file(path) for key, path in artifacts.items() if path.is_file()
    }
    if reference_shim_path is not None and reference_shim_path.is_file():
        digests["reference_shim"] = _sha256_file(reference_shim_path)
    return dict(sorted(digests.items()))


@locked_resume
def run_lematgenbench(
    generated_structures: Sequence[Structure] | str | Path,
    *,
    repo_path: str | Path,
    run_name: str,
    output_dir: str | Path | None,
    config_name: str = LEMAT_GENBENCH_CONFIG,
    families: Sequence[str] | None = None,
    fingerprint_method: str = "structure-matcher",
    runner_command: str | Sequence[str] = _DEFAULT_RUNNER_COMMAND,
    expected_commit: str | None = LEMAT_GENBENCH_COMMIT,
    expected_lock_sha256: str | None = LEMAT_GENBENCH_UV_LOCK_SHA256,
    verify_mace_checkpoint: bool = True,
    expected_mace_checkpoint_sha256: str | None = (
        LEMAT_GENBENCH_MACE_CHECKPOINT_SHA256
    ),
    verify_orb_checkpoint: bool = True,
    expected_orb_checkpoint_sha256: str | None = (LEMAT_GENBENCH_ORB_CHECKPOINT_SHA256),
    verify_uma_checkpoint: bool = True,
    expected_uma_checkpoint_sha256: str | None = (LEMAT_GENBENCH_UMA_CHECKPOINT_SHA256),
    enforce_reference_revisions: bool = True,
    runtime_device: str = "cuda",
    cuda_visible_devices: str | None = None,
    parse_failures_as_invalid: bool = False,
    require_complete_oracle_coverage: bool = True,
    require_clean_repo: bool = True,
    timeout_seconds: float | None = None,
    fail_on_family_error: bool = True,
    resume: bool = False,
) -> LeMatGenBenchResult:
    """Run the pinned LeMat-GenBench repository benchmark on CIF structures.

    The upstream environment must already exist. The recommended command is
    the default, ``uv run --frozen python``, which uses the checkout's locked
    environment without modifying its lock file. See the zevals
    LeMat-GenBench guide for one-time setup, model access, and data caveats.

    Args:
        generated_structures: Pymatgen structures, a CIF directory, a single
            CIF, or a text file listing CIF paths. Relative paths in a list file
            are resolved relative to that file before invoking upstream.
        repo_path: LeMat-GenBench source checkout containing the runner script.
        run_name: Path-safe, explicit upstream run identifier.
        output_dir: Dedicated directory for inputs, raw output, logs, metrics,
            and provenance. If ``None``, a persistent temporary directory is
            created; published evaluations should always provide this value.
        config_name: Upstream config name. The leaderboard-compatible default
            is ``comprehensive_multi_mlip_hull``.
        families: Optional upstream benchmark families. Validity always runs.
        fingerprint_method: Upstream fingerprint implementation.
        runner_command: Command prefix used before ``run_benchmarks.py``.
            Pass the checkout's Python executable directly if not using uv.
        expected_commit: Exact revision to require, or ``None`` to opt out.
        expected_lock_sha256: Exact upstream ``uv.lock`` digest to require, or
            ``None`` to opt out. The default matches the pinned revision.
        verify_mace_checkpoint: Resolve the exact MACE model implied by the
            pinned dependency before any MLIP-dependent family runs.
        expected_mace_checkpoint_sha256: MACE model digest to require, or
            ``None`` to resolve and record without verifying it.
        verify_orb_checkpoint: Hash and require the canonical ORB checkpoint
            when an MLIP-dependent family actually loads it.
        expected_orb_checkpoint_sha256: ORB checkpoint digest to require, or
            ``None`` to resolve and record without comparing a known digest.
        verify_uma_checkpoint: Hash and require the canonical UMA checkpoint
            when an MLIP-dependent family actually loads it.
        expected_uma_checkpoint_sha256: UMA checkpoint digest to require, or
            ``None`` to resolve and record without comparing a known digest.
        enforce_reference_revisions: Inject and verify a runtime startup shim
            that forces audited Hugging Face reference revisions. Disable only
            for diagnostic, non-manuscript runs.
        runtime_device: Explicit ``"cuda"`` or ``"cpu"`` policy. The pinned
            runner otherwise silently selects CUDA based on host visibility.
        cuda_visible_devices: Optional exact CUDA visibility mask for the child
            processes. When omitted, preserve the scheduler-provided mask; CPU
            mode always hides CUDA. The effective mask and GPU properties are
            archived.
        parse_failures_as_invalid: If true, treat CIFs dropped by upstream's
            parser as invalid attempts and retain the staged-CIF denominator.
            If false, any staged/loaded count mismatch is a hard failure.
        require_complete_oracle_coverage: Require the public stability final
            scores to show ORB, MACE, and UMA results for every valid structure
            whenever the stability family is selected.
        require_clean_repo: Reject tracked upstream worktree modifications.
        timeout_seconds: Optional wall-clock timeout for the subprocess.
        fail_on_family_error: Raise after preserving artifacts if upstream
            reports a family error or emits an unreadable result schema.
        resume: Persist per-oracle and complete-set family results atomically,
            and reuse them across matching invocations in the same output
            directory. Requires an explicit output directory and reference pins.

    Returns:
        :class:`LeMatGenBenchResult` with both raw and flattened results.

    Raises:
        LeMatGenBenchError: If checkout validation, execution, or result loading
            fails.
    """
    deadline = (
        time.monotonic() + timeout_seconds if timeout_seconds is not None else None
    )
    if resume and not enforce_reference_revisions:
        raise ValueError(
            "Resumable LeMat runs require enforce_reference_revisions=True"
        )
    _validate_path_component(run_name, "run_name")
    _validate_path_component(config_name, "config_name")
    if families is not None:
        families = tuple(families)
        if not families:
            raise ValueError(
                "families must be non-empty when provided; omit it for upstream "
                "defaults or pass ('validity',) for validity-only evaluation"
            )
        for family in families:
            _validate_path_component(family, "family")

    resolved_repo = Path(repo_path).expanduser().resolve()
    checkout = _inspect_checkout(
        resolved_repo,
        expected_commit=expected_commit,
        expected_lock_sha256=expected_lock_sha256,
        require_clean_repo=require_clean_repo,
    )
    command_prefix = _normalize_runner_command(runner_command)
    normalizer_source = _material_cif_source()
    normalizer_sha256 = hashlib.sha256(normalizer_source.encode("utf-8")).hexdigest()
    benchmark_environment = _runtime_environment(
        runtime_device,
        cuda_visible_devices=cuda_visible_devices,
    )
    if "--frozen" in command_prefix and checkout.lock_sha256 is None:
        raise LeMatGenBenchError(
            f"{resolved_repo}/uv.lock is missing. Restore the lockfile committed "
            "at the pinned revision before using the default frozen runner."
        )
    local_reference_inventory = _canonical_reference_inventory(
        resolved_repo, config_name
    )
    local_reference_errors = _local_reference_integrity_errors(
        local_reference_inventory,
        families=families,
    )
    if (
        enforce_reference_revisions
        and config_name == LEMAT_GENBENCH_CONFIG
        and local_reference_errors
    ):
        raise LeMatGenBenchError(
            "Canonical local reference files do not match the ordinary Git blobs "
            f"audited at {LEMAT_GENBENCH_COMMIT}: {local_reference_errors}. Restore "
            "the pinned checkout, or explicitly disable reference enforcement for "
            "a diagnostic run."
        )
    local_hull_overrides = _local_hull_overrides(local_reference_inventory)
    if (
        enforce_reference_revisions
        and config_name == LEMAT_GENBENCH_CONFIG
        and {"stability", "sun"} & set(_active_families(families))
        and local_hull_overrides
    ):
        raise LeMatGenBenchError(
            "Canonical reference-revision enforcement cannot verify local hull "
            "overrides. Remove these files so LeMat-GenBench fetches the pinned "
            "LeMaterial/LeMat-Bulk-MLIP-Hull revision, or explicitly disable "
            f"enforcement for a diagnostic run: {local_hull_overrides}"
        )

    if output_dir is None:
        resolved_output = Path(tempfile.mkdtemp(prefix="zevals-lematgenbench-"))
    else:
        resolved_output = Path(output_dir).expanduser().resolve()
        resolved_output.mkdir(parents=True, exist_ok=True)
    artifacts = _ensure_new_artifacts(resolved_output, allow_existing=resume)
    resume_history: list[dict] = []
    resume_request_sha256: str | None = None
    resume_contract_sha256: str | None = None
    if resume:
        # Stage once into temporary storage to validate exact supplied bytes
        # before moving any earlier attempt or loading any completed result.
        with tempfile.TemporaryDirectory(
            prefix=".resume-input-", dir=resolved_output
        ) as staging:
            _, candidate_inputs = _prepare_cif_input(
                generated_structures, Path(staging)
            )
        resume_request = {
            "upstream_commit": checkout.commit,
            "upstream_lock_sha256": checkout.lock_sha256,
            "upstream_pyproject_sha256": checkout.pyproject_sha256,
            "runner_command": list(command_prefix),
            "run_name": run_name,
            "config_name": config_name,
            "families": list(families) if families is not None else None,
            "fingerprint_method": fingerprint_method,
            "runtime_device": runtime_device,
            "expected_commit": expected_commit,
            "expected_lock_sha256": expected_lock_sha256,
            "verify_mace_checkpoint": verify_mace_checkpoint,
            "expected_mace_checkpoint_sha256": expected_mace_checkpoint_sha256,
            "verify_orb_checkpoint": verify_orb_checkpoint,
            "expected_orb_checkpoint_sha256": expected_orb_checkpoint_sha256,
            "verify_uma_checkpoint": verify_uma_checkpoint,
            "expected_uma_checkpoint_sha256": expected_uma_checkpoint_sha256,
            "reference_revisions": LEMAT_GENBENCH_REFERENCE_REVISIONS,
            "local_references": local_reference_inventory,
            "parse_failures_as_invalid": parse_failures_as_invalid,
            "require_complete_oracle_coverage": require_complete_oracle_coverage,
            "require_clean_repo": require_clean_repo,
            "fail_on_family_error": fail_on_family_error,
            "inputs": [
                {"source_path": entry["source_path"], "sha256": entry["sha256"]}
                for entry in candidate_inputs
            ],
            "adapter_sha256": _sha256_file(Path(__file__)),
            "material_cif_normalizer_sha256": normalizer_sha256,
            "resume_host_sha256": _sha256_file(
                Path(__file__).with_name("_lemat_resume.py")
            ),
            "resume_runtime_sha256": _sha256_file(
                Path(__file__).with_name("_lemat_resume_runtime.py")
            ),
        }
        state_dir = resolved_output / "resume_state"
        resume_request_sha256 = validate_contract(
            state_dir / "request.json", resume_request
        )
        completed_artifacts = {
            **artifacts,
            "resume_runtime": artifacts["reference_shim_dir"]
            / "_lemat_resume_runtime.py",
        }
        previous_result = load_completed(
            state_dir / "completed.json", LeMatGenBenchResult, completed_artifacts
        )
        if previous_result is not None:
            return previous_result
        resume_history = archive_attempt(resolved_output, artifacts)
        benchmark_environment["ZEVALS_LEMAT_STREAM_LOGS"] = "1"
    upstream_input, input_hashes = _prepare_cif_input(
        generated_structures, resolved_output
    )
    if (
        resume
        and [
            {"source_path": entry["source_path"], "sha256": entry["sha256"]}
            for entry in input_hashes
        ]
        != resume_request["inputs"]
    ):
        raise LeMatGenBenchError(
            "Generated CIF bytes changed while staging resume inputs"
        )
    mace_checkpoint: _MaceCheckpointInfo | None = None
    if verify_mace_checkpoint and _needs_mace(families):
        mace_checkpoint = _probe_mace_checkpoint(
            repo_path=resolved_repo,
            command_prefix=command_prefix,
            model=LEMAT_GENBENCH_MACE_MODEL,
            expected_sha256=expected_mace_checkpoint_sha256,
            stdout_path=artifacts["mace_stdout"],
            stderr_path=artifacts["mace_stderr"],
            timeout_seconds=remaining_timeout(deadline),
            environment=_environment_with_role(benchmark_environment, "mace-preflight"),
        )
    expected_oracle_digests = {
        "mace": (expected_mace_checkpoint_sha256 if verify_mace_checkpoint else None),
        "orb": (expected_orb_checkpoint_sha256 if verify_orb_checkpoint else None),
        "uma": (expected_uma_checkpoint_sha256 if verify_uma_checkpoint else None),
    }

    reference_shim_path: Path | None = None
    reference_shim_sha256: str | None = None
    reference_probe_command: tuple[str, ...] | None = None
    if enforce_reference_revisions:
        reference_shim_path, reference_shim_sha256 = _write_reference_pin_shim(
            artifacts["reference_shim_dir"],
            LEMAT_GENBENCH_REFERENCE_REVISIONS,
            expected_mace_checkpoint_sha256=expected_oracle_digests["mace"],
            mace_checkpoint_path=(
                mace_checkpoint.path if mace_checkpoint is not None else None
            ),
            expected_orb_checkpoint_sha256=expected_oracle_digests["orb"],
            expected_uma_checkpoint_sha256=expected_oracle_digests["uma"],
        )
        benchmark_environment = _reference_pin_environment(
            artifacts["reference_shim_dir"],
            artifacts["reference_audit"],
            base_environment=benchmark_environment,
        )
        if resume:
            restore_audit(resume_history, artifacts["reference_audit"])
        reference_probe_command = _verify_reference_pin_shim(
            repo_path=resolved_repo,
            command_prefix=command_prefix,
            environment=_environment_with_role(
                benchmark_environment, "reference-pin-preflight"
            ),
            expected_shim_path=reference_shim_path,
            expected_revisions=LEMAT_GENBENCH_REFERENCE_REVISIONS,
            expected_oracle_digests=expected_oracle_digests,
            stdout_path=artifacts["reference_stdout"],
            stderr_path=artifacts["reference_stderr"],
            timeout_seconds=remaining_timeout(deadline),
        )

    runtime_provenance, runtime_probe_command, runtime_provenance_sha256 = (
        _probe_runtime_environment(
            repo_path=resolved_repo,
            command_prefix=command_prefix,
            environment=_environment_with_role(
                benchmark_environment, "runtime-preflight"
            ),
            expected_device=runtime_device,
            requested_cuda_visible_devices=cuda_visible_devices,
            output_path=artifacts["runtime"],
            stdout_path=artifacts["runtime_stdout"],
            stderr_path=artifacts["runtime_stderr"],
            timeout_seconds=remaining_timeout(deadline),
        )
    )

    if resume:
        state_dir = resolved_output / "resume_state"
        resume_contract_sha256 = validate_contract(
            state_dir / "cache_contract.json",
            {
                "request_sha256": resume_request_sha256,
                "runtime": runtime_contract(runtime_provenance),
            },
        )
        artifacts["resume_runtime"] = (
            artifacts["reference_shim_dir"] / "_lemat_resume_runtime.py"
        )
        shutil.copyfile(
            Path(__file__).with_name("_lemat_resume_runtime.py"),
            artifacts["resume_runtime"],
        )
        assert reference_shim_path is not None
        with reference_shim_path.open("a") as stream:
            stream.write(
                "\ntry:\n import _lemat_resume_runtime\n"
                " _lemat_resume_runtime.install()\n"
                "except BaseException:\n import traceback\n"
                " traceback.print_exc()\n os._exit(86)\n"
            )
        reference_shim_sha256 = _sha256_file(reference_shim_path)
        benchmark_environment["ZEVALS_LEMAT_RESUME_CACHE"] = str(state_dir / "cache")
        benchmark_environment["ZEVALS_LEMAT_RESUME_CONTRACT"] = resume_contract_sha256

    script_path = resolved_repo / "scripts" / "run_benchmarks.py"
    command = [
        *command_prefix,
        "-c",
        _cif_loader_bootstrap_source(normalizer_source),
        str(script_path),
        "--cifs",
        str(upstream_input),
        "--config",
        config_name,
        "--name",
        run_name,
        "--fingerprint-method",
        fingerprint_method,
    ]
    if families:
        command.extend(["--families", *families])

    result_prefix = f"{run_name}_{config_name}"
    inventory_before = _result_inventory(resolved_repo / "results_final", result_prefix)
    try:
        if resume:
            completed = run_streamed(
                command,
                cwd=resolved_repo,
                environment=_environment_with_role(benchmark_environment, "benchmark"),
                stdout_path=artifacts["stdout"],
                stderr_path=artifacts["stderr"],
                timeout_seconds=remaining_timeout(deadline),
            )
        else:
            completed = subprocess.run(
                command,
                cwd=resolved_repo,
                check=False,
                capture_output=True,
                text=True,
                timeout=remaining_timeout(deadline),
                env=_environment_with_role(benchmark_environment, "benchmark"),
            )
    except LeMatResumeTimeout as exc:
        raise LeMatGenBenchTimeout(str(exc)) from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LeMatGenBenchError(
            f"Could not run LeMat-GenBench with command {command!r}: {exc}"
        ) from exc

    artifacts["stdout"].write_text(completed.stdout, encoding="utf-8")
    artifacts["stderr"].write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise LeMatGenBenchError(
            f"LeMat-GenBench exited with status {completed.returncode}. "
            f"See {artifacts['stdout']} and {artifacts['stderr']}."
        )

    upstream_result_path = _locate_upstream_result(
        repo_path=resolved_repo,
        result_prefix=result_prefix,
        inventory_before=inventory_before,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    try:
        raw_result = json.loads(upstream_result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LeMatGenBenchError(
            f"Could not read upstream result JSON {upstream_result_path}: {exc}"
        ) from exc
    if not isinstance(raw_result, dict):
        raise LeMatGenBenchError(
            f"Expected a JSON object in {upstream_result_path}, got "
            f"{type(raw_result).__name__}"
        )

    shutil.copy2(upstream_result_path, artifacts["raw"])
    upstream_lock_path = resolved_repo / "uv.lock"
    archived_lock_path: Path | None = None
    if upstream_lock_path.is_file():
        shutil.copy2(upstream_lock_path, artifacts["lock"])
        archived_lock_path = artifacts["lock"]

    input_integrity_errors = _rehash_staged_inputs(input_hashes)
    local_reference_inventory_post_run = _canonical_reference_inventory(
        resolved_repo, config_name
    )
    local_reference_changes = _local_reference_changes(
        local_reference_inventory,
        local_reference_inventory_post_run,
    )
    mace_post_run_sha256: str | None = None
    mace_checkpoint_error: str | None = None
    if mace_checkpoint is not None:
        if not mace_checkpoint.path.is_file():
            mace_checkpoint_error = (
                f"MACE checkpoint disappeared during evaluation: {mace_checkpoint.path}"
            )
        else:
            mace_post_run_sha256 = _sha256_file(mace_checkpoint.path)
            if mace_post_run_sha256 != mace_checkpoint.sha256:
                mace_checkpoint_error = (
                    "MACE checkpoint changed during evaluation: expected "
                    f"{mace_checkpoint.sha256}, found {mace_post_run_sha256} at "
                    f"{mace_checkpoint.path}"
                )

    expected_input_count = len(input_hashes)
    input_count_error: str | None = None
    upstream_loaded_count: int | None
    try:
        upstream_loaded_count = _upstream_loaded_count(raw_result)
    except LeMatGenBenchError as exc:
        upstream_loaded_count = None
        input_count_error = str(exc)
    else:
        if upstream_loaded_count != expected_input_count:
            if upstream_loaded_count < expected_input_count:
                input_count_error = (
                    "LeMat-GenBench silently dropped CIFs while parsing: staged "
                    f"{expected_input_count}, loaded {upstream_loaded_count}"
                )
            else:
                input_count_error = (
                    "LeMat-GenBench reported more loaded structures than staged "
                    f"CIFs: staged {expected_input_count}, loaded "
                    f"{upstream_loaded_count}"
                )

    metrics = extract_lematgenbench_metrics(raw_result)
    metrics.update(
        derive_attempt_normalized_rates(metrics, total_attempts=expected_input_count)
    )
    metrics.update(
        {
            "input_accounting/attempted_structures": float(expected_input_count),
            "input_accounting/upstream_loaded_structures": float(
                upstream_loaded_count or 0
            ),
            "input_accounting/cif_parse_failure_count": float(
                max(expected_input_count - (upstream_loaded_count or 0), 0)
            ),
        }
    )
    oracle_coverage, oracle_coverage_metrics = _oracle_coverage_summary(
        metrics,
        config_name=config_name,
        families=families,
    )
    metrics.update(oracle_coverage_metrics)
    metrics = dict(sorted(metrics.items()))
    required_families = (
        "validity",
        *(families if families is not None else _UPSTREAM_DEFAULT_FAMILIES),
    )
    family_errors = _family_errors(raw_result, required_families)
    mace_fallback_detected = _MACE_FALLBACK_LOG_MARKER in (
        f"{completed.stdout}\n{completed.stderr}"
    )
    valid_structure_count = oracle_coverage.get("valid_structures")
    reference_access_plan = _canonical_reference_access_plan(
        config_name=config_name,
        families=families,
        valid_structure_count=(
            valid_structure_count if isinstance(valid_structure_count, int) else None
        ),
        local_inventory=local_reference_inventory,
    )
    require_oracle_artifacts = bool(
        config_name == LEMAT_GENBENCH_CONFIG
        and (valid_structure_count is None or valid_structure_count > 0)
        and _needs_mace(families)
    )
    reference_pin_audit: list[dict[str, Any]] = []
    reference_pin_summary: dict[str, Any] = {
        **reference_access_plan,
        "verification": "disabled",
    }
    reference_pin_error: str | None = None
    if enforce_reference_revisions:
        reference_pin_summary["verification"] = "enabled"
        try:
            reference_pin_audit = _read_reference_pin_audit(
                artifacts["reference_audit"]
            )
        except LeMatGenBenchError as exc:
            reference_pin_error = str(exc)
        else:
            reference_pin_summary, reference_pin_error = _reference_pin_audit_summary(
                reference_pin_audit,
                access_plan=reference_access_plan,
                require_oracle_artifacts=require_oracle_artifacts,
                expected_oracle_digests=expected_oracle_digests,
            )
            reference_pin_summary["verification"] = "enabled"
    oracle_coverage_error: str | None = None
    if (
        require_complete_oracle_coverage
        and config_name == LEMAT_GENBENCH_CONFIG
        and "stability" in set(_active_families(families))
        and (
            valid_structure_count is None
            or (
                valid_structure_count > 0
                and oracle_coverage.get("status") != "complete"
            )
        )
    ):
        oracle_coverage_error = (
            "ORB/MACE/UMA coverage is not complete for every valid structure: "
            f"{oracle_coverage}"
        )
    artifacts["metrics"].write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "adapter": "zevals.materials.lematgenbench",
        "upstream_repository": LEMAT_GENBENCH_REPOSITORY,
        "upstream_commit": checkout.commit,
        "expected_commit": expected_commit,
        "expected_uv_lock_sha256": expected_lock_sha256,
        "upstream_pyproject_sha256": checkout.pyproject_sha256,
        "upstream_uv_lock_sha256": checkout.lock_sha256,
        "archived_uv_lock_path": (
            str(archived_lock_path) if archived_lock_path is not None else None
        ),
        "config_name": config_name,
        "families": list(families) if families is not None else None,
        "fingerprint_method": fingerprint_method,
        "parse_failures_as_invalid": parse_failures_as_invalid,
        "require_clean_repo": require_clean_repo,
        "run_name": run_name,
        "resume": {
            "enabled": resume,
            "request_sha256": resume_request_sha256,
            "cache_contract_sha256": resume_contract_sha256,
            "attempt_history": resume_history,
            "cache_semantics": "oracle-and-novelty-structure-comparison-row-full-set-family",
            "runtime_hook_sha256": (
                _sha256_file(artifacts["resume_runtime"]) if resume else None
            ),
        },
        "command": command,
        "cif_loading": {
            "normalizer_sha256": normalizer_sha256,
            "scope": "staged input manifest paths only",
            "method": "mmCIF dictionary aliases; upstream Structure.from_str",
            "staged_source_bytes_preserved": True,
            "normalizer_source_archived_in_command": True,
        },
        "runtime": {
            "device_policy": runtime_device,
            "requested_cuda_visible_devices": cuda_visible_devices,
            "provenance": runtime_provenance,
            "provenance_path": str(artifacts["runtime"]),
            "provenance_sha256": runtime_provenance_sha256,
            "probe_command": list(runtime_probe_command),
            "inherited_pythonpath_sanitized": True,
        },
        "mace_checkpoint": {
            "verification_enabled": verify_mace_checkpoint,
            "model": mace_checkpoint.model if mace_checkpoint is not None else None,
            "path": (
                str(mace_checkpoint.path) if mace_checkpoint is not None else None
            ),
            "sha256": (mace_checkpoint.sha256 if mace_checkpoint is not None else None),
            "expected_sha256": expected_mace_checkpoint_sha256,
            "post_run_sha256": mace_post_run_sha256,
            "post_run_error": mace_checkpoint_error,
            "probe_command": (
                list(mace_checkpoint.command) if mace_checkpoint is not None else None
            ),
            "fallback_to_small_detected": mace_fallback_detected,
        },
        "oracle_protocol": {
            "identities": json.loads(json.dumps(LEMAT_GENBENCH_ORACLE_IDENTITIES)),
            "verify_orb_checkpoint": verify_orb_checkpoint,
            "expected_orb_checkpoint_sha256": expected_oracle_digests["orb"],
            "verify_uma_checkpoint": verify_uma_checkpoint,
            "expected_uma_checkpoint_sha256": expected_oracle_digests["uma"],
            "runtime_checkpoint_verification_effective": {
                "orb": bool(
                    enforce_reference_revisions
                    and require_oracle_artifacts
                    and verify_orb_checkpoint
                    and expected_oracle_digests["orb"] is not None
                ),
                "mace": bool(
                    mace_checkpoint is not None
                    and expected_mace_checkpoint_sha256 is not None
                ),
                "uma": bool(
                    enforce_reference_revisions
                    and require_oracle_artifacts
                    and verify_uma_checkpoint
                    and expected_oracle_digests["uma"] is not None
                ),
            },
            "require_complete_coverage": require_complete_oracle_coverage,
            "coverage": oracle_coverage,
            "coverage_error": oracle_coverage_error,
            "model_load_log_markers": {
                "orb": (
                    f"Successfully loaded ORB model: {LEMAT_GENBENCH_ORB_MODEL}"
                    in f"{completed.stdout}\n{completed.stderr}"
                ),
                "mace": (
                    "Successfully loaded MACE model: mp"
                    in f"{completed.stdout}\n{completed.stderr}"
                ),
                "uma": (
                    f"Successfully loaded UMA model: {LEMAT_GENBENCH_UMA_MODEL}"
                    in f"{completed.stdout}\n{completed.stderr}"
                ),
            },
        },
        "reference_revision_pins": {
            "enforced": enforce_reference_revisions,
            "revisions": dict(LEMAT_GENBENCH_REFERENCE_REVISIONS),
            "patched_apis": list(_REFERENCE_PIN_PATCHED_APIS),
            "expected_oracle_digests": expected_oracle_digests,
            "shim_path": (
                str(reference_shim_path) if reference_shim_path is not None else None
            ),
            "shim_sha256": reference_shim_sha256,
            "audit_path": (
                str(artifacts["reference_audit"])
                if enforce_reference_revisions
                else None
            ),
            "probe_command": (
                list(reference_probe_command)
                if reference_probe_command is not None
                else None
            ),
            "pythonpath_prepend": (
                str(artifacts["reference_shim_dir"])
                if enforce_reference_revisions
                else None
            ),
            "audit": reference_pin_audit,
            "runtime_access": reference_pin_summary,
            "error": reference_pin_error,
        },
        "local_reference_artifacts": local_reference_inventory,
        "local_reference_artifacts_post_run": local_reference_inventory_post_run,
        "local_reference_changes_during_run": local_reference_changes,
        "local_reference_integrity_errors": local_reference_errors,
        "input": str(upstream_input),
        "input_cifs": input_hashes,
        "input_integrity_errors": input_integrity_errors,
        "input_count": {
            "expected_cifs": expected_input_count,
            "upstream_loaded_structures": upstream_loaded_count,
            "dropped_during_cif_parsing": (
                expected_input_count - upstream_loaded_count
                if upstream_loaded_count is not None
                else None
            ),
            "exact_match": input_count_error is None,
            "accepted_as_invalid_attempts": bool(
                input_count_error is not None
                and parse_failures_as_invalid
                and upstream_loaded_count is not None
                and upstream_loaded_count < expected_input_count
            ),
            "error": input_count_error,
        },
        "upstream_result_path": str(upstream_result_path),
        "raw_result_path": str(artifacts["raw"]),
        "metrics_path": str(artifacts["metrics"]),
        "family_errors": family_errors,
    }
    artifacts["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    accepted_parse_failures = bool(
        input_count_error is not None
        and parse_failures_as_invalid
        and upstream_loaded_count is not None
        and upstream_loaded_count < expected_input_count
    )
    if input_count_error is not None and not accepted_parse_failures:
        raise LeMatGenBenchError(
            f"{input_count_error}. Raw results, the input manifest, and logs are "
            f"preserved in {resolved_output}."
        )
    if input_integrity_errors:
        raise LeMatGenBenchError(
            "Staged CIF bytes changed while LeMat-GenBench was running: "
            f"{input_integrity_errors}. Raw results and logs are preserved in "
            f"{resolved_output}."
        )
    if local_reference_changes:
        raise LeMatGenBenchError(
            "Canonical local reference files changed while LeMat-GenBench was "
            f"running: {local_reference_changes}. Raw results and logs are preserved "
            f"in {resolved_output}."
        )
    assert upstream_loaded_count is not None
    parse_failure_count = expected_input_count - upstream_loaded_count
    if reference_pin_error is not None:
        raise LeMatGenBenchError(
            f"{reference_pin_error}. Raw results and logs are preserved in "
            f"{resolved_output}."
        )
    if mace_checkpoint_error is not None:
        raise LeMatGenBenchError(
            f"{mace_checkpoint_error}. Raw results and logs are preserved in "
            f"{resolved_output}."
        )
    if mace_fallback_detected:
        raise LeMatGenBenchError(
            "LeMat-GenBench silently replaced the pinned MACE medium-mpa-0 model "
            "with its alternative small model. Raw results and logs are preserved "
            f"in {resolved_output}."
        )
    if oracle_coverage_error is not None:
        raise LeMatGenBenchError(
            f"{oracle_coverage_error}. Raw results and logs are preserved in "
            f"{resolved_output}."
        )
    if family_errors and fail_on_family_error:
        raise LeMatGenBenchError(
            "LeMat-GenBench reported incomplete benchmark families: "
            f"{family_errors}. Raw results and logs are preserved in {resolved_output}."
        )

    artifact_sha256 = _artifact_digest_map(
        artifacts,
        reference_shim_path=reference_shim_path,
    )

    result = LeMatGenBenchResult(
        raw_result=raw_result,
        metrics=metrics,
        output_dir=resolved_output,
        raw_result_path=artifacts["raw"],
        metrics_path=artifacts["metrics"],
        manifest_path=artifacts["manifest"],
        stdout_path=artifacts["stdout"],
        stderr_path=artifacts["stderr"],
        uv_lock_path=archived_lock_path,
        upstream_result_path=upstream_result_path,
        upstream_commit=checkout.commit,
        upstream_lock_sha256=checkout.lock_sha256,
        command=tuple(command),
        family_errors=family_errors,
        expected_input_count=expected_input_count,
        upstream_loaded_count=upstream_loaded_count,
        parse_failure_count=parse_failure_count,
        mace_model=(mace_checkpoint.model if mace_checkpoint is not None else None),
        mace_checkpoint_path=(
            mace_checkpoint.path if mace_checkpoint is not None else None
        ),
        mace_checkpoint_sha256=(
            mace_checkpoint.sha256 if mace_checkpoint is not None else None
        ),
        reference_pin_shim_path=reference_shim_path,
        reference_pin_shim_sha256=reference_shim_sha256,
        reference_pin_audit_path=(
            artifacts["reference_audit"] if enforce_reference_revisions else None
        ),
        reference_revisions_enforced=enforce_reference_revisions,
        runtime_environment=runtime_provenance,
        runtime_environment_path=artifacts["runtime"],
        runtime_environment_sha256=runtime_provenance_sha256,
        artifact_sha256=artifact_sha256,
        oracle_coverage=oracle_coverage,
    )
    if resume:
        save_completed(resolved_output / "resume_state" / "completed.json", result)
    return result
