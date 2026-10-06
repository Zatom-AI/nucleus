"""Atomic progress and allocation budgets for the single-GPU Table-2 protocol."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import math
import os
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

_DEADLINE: ContextVar[float | None] = ContextVar("table2_deadline", default=None)


class EvaluationPaused(TimeoutError):
    """The allocation budget expired; committed work can be resumed (exit 75)."""


def remaining_seconds(limit: float | None = None) -> float | None:
    """Return the remaining allocation budget, optionally capped by a stage limit."""
    deadline = _DEADLINE.get()
    remaining = None if deadline is None else deadline - time.monotonic()
    if remaining is not None and remaining <= 0:
        raise EvaluationPaused("Allocation budget exhausted; resume this out_dir.")
    if limit is not None:
        if not math.isfinite(float(limit)) or float(limit) <= 0:
            raise ValueError("Timeout must be positive.")
        remaining = float(limit) if remaining is None else min(remaining, float(limit))
    return remaining


@contextmanager
def evaluation_allocation(output_dir: Path, seconds: float | None) -> Iterator[None]:
    """Prevent concurrent writers and scope the wall clock to this invocation."""
    if seconds is not None and (
        not math.isfinite(float(seconds)) or float(seconds) <= 0
    ):
        raise ValueError("evaluation.walltime_seconds must be positive or null.")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".table2_evaluation.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another evaluation is writing {output_dir}."
            ) from error
        token = _DEADLINE.set(
            None if seconds is None else time.monotonic() + float(seconds)
        )
        try:
            yield
        finally:
            _DEADLINE.reset(token)
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_json(path: Path, value: Any) -> None:
    """Commit a JSON marker only after all of its bytes reach the filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def group_seed(seed: int, domain: str, group_id: str) -> int:
    """Derive a stable independent diffusion stream for one D-replica group."""
    payload = f"zatom2-table2-v3:{seed}:{domain}:{group_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def sampling_source_hashes() -> dict[str, str]:
    """Bind editable inference dependencies as well as installed version numbers."""
    from zatom2.table2_evaluation import _sha256_file, _sha256_json

    fingerprints = {}
    for package in ("zatom2", "nucleus", "rfd3", "foundry", "atomworks"):
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError(f"Cannot fingerprint inference package: {package}")
        files = {}
        for index, location in enumerate(spec.submodule_search_locations):
            root = Path(location)
            for path in sorted(root.rglob("*.py")):
                files[f"{index}/{path.relative_to(root)}"] = _sha256_file(path)
        fingerprints[package] = _sha256_json(files)
    return fingerprints


def verify_group(
    marker: Path, domain_dir: Path, *, contract_sha256: str, group_id: str
) -> list[dict[str, Any]]:
    """Verify committed CIF and metadata bytes before skipping a sampling group."""
    from zatom2.table2_evaluation import _read_json, _sample_artifact_path, _sha256_file

    saved = _read_json(marker)
    if (
        saved.get("contract_sha256") != contract_sha256
        or saved.get("group_id") != group_id
    ):
        raise ValueError(f"Sampling progress contract changed: {marker}")
    rows = saved["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Invalid sampling progress rows: {marker}")
    expected_hashes = saved["artifact_sha256"]
    actual_hashes = {
        row[key]: _sha256_file(_sample_artifact_path(domain_dir, row[key]))
        for row in rows
        for key in ("cif_path", "metadata_path")
    }
    if actual_hashes != expected_hashes:
        raise ValueError(f"Committed sample artifacts changed: {marker}")
    return rows


def commit_group(
    marker: Path,
    domain_dir: Path,
    rows: list[dict[str, Any]],
    *,
    contract_sha256: str,
    group_id: str,
    seed: int,
) -> None:
    """Commit a completed group after persisting its immutable sample artifacts."""
    from zatom2.table2_evaluation import _sample_artifact_path, _sha256_file

    hashes = {}
    parents = set()
    for row in rows:
        for key in ("cif_path", "metadata_path"):
            path = _sample_artifact_path(domain_dir, row[key])
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            parents.add(path.parent)
            hashes[row[key]] = _sha256_file(path)
    for parent in parents:
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    atomic_json(
        marker,
        {
            "contract_sha256": contract_sha256,
            "group_id": group_id,
            "seed": seed,
            "rows": rows,
            "artifact_sha256": hashes,
        },
    )


def bind_contract(path: Path, contract: Mapping[str, Any], *, resume: bool) -> str:
    """Refuse to mix progress from different checkpoints, inputs, or runtimes."""
    from zatom2.table2_evaluation import _read_json, _sha256_json

    if path.exists():
        if not resume:
            raise FileExistsError(
                f"Progress exists: {path}; use evaluation.resume=true."
            )
        if _read_json(path) != contract:
            raise ValueError(
                "Sampling resume contract changed (checkpoint, protocol, sampler, "
                "software, or GPU type). Use a new out_dir."
            )
    else:
        atomic_json(path, contract)
    return _sha256_json(contract)
