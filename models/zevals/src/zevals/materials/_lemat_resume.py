"""Host-side persistence and child-process lifecycle for LeMat restarts."""

from __future__ import annotations

import dataclasses
import fcntl
import functools
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


class LeMatResumeTimeout(TimeoutError):
    """An allocation deadline interrupted a restartable LeMat invocation."""


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def locked_resume(function):
    """Reject concurrent writers to one resumable output directory."""

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        if not kwargs.get("resume", False):
            return function(*args, **kwargs)
        if kwargs.get("output_dir") is None:
            raise ValueError("resume=True requires an explicit persistent output_dir")
        directory = Path(kwargs["output_dir"]).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".resume.lock").open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"LeMat output already in use: {directory}") from exc
            return function(*args, **kwargs)

    return wrapped


def validate_contract(path: Path, contract: dict) -> str:
    """Freeze a request or numerical runtime contract before reusing any work."""
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != contract:
            raise ValueError(
                f"LeMat resume contract changed: {path}. Use a new output directory."
            )
    else:
        atomic_json(path, contract)
    return digest


def runtime_contract(runtime: dict) -> dict:
    """Allow a different allocated GPU of the same type, retaining math settings."""
    payload = json.loads(json.dumps(runtime))
    for key in (
        "requested_cuda_visible_devices",
        "effective_cuda_visible_devices",
    ):
        payload.pop(key, None)
    for device in payload.get("cuda_devices", []):
        device.pop("uuid", None)
    return payload


def archive_attempt(output: Path, artifacts: dict[str, Path]) -> list[dict]:
    """Move an interrupted attempt intact and retain its reference-access audit."""
    state = output / "resume_state"
    history_path = state / "attempts.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    if history and history[-1].get("status") == "moving":
        entry = history[-1]
        attempt = Path(entry["directory"])
        existing = [output / name for name in entry["paths"]]
    else:
        paths = {*artifacts.values(), output / "input_cifs"}
        existing = sorted((path for path in paths if path.exists()), key=str)
        if not existing:
            return history
        attempt = state / "attempts" / f"{len(history):06d}"
        inventory = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in existing
            if path.is_file()
        }
        entry = {
            "directory": str(attempt),
            "file_sha256": inventory,
            "paths": [path.name for path in existing],
            "status": "moving",
        }
        history.append(entry)
        atomic_json(history_path, history)
    attempt.mkdir(parents=True, exist_ok=True)
    for path in existing:
        if path.exists():
            if (attempt / path.name).exists():
                raise ValueError(f"Ambiguous interrupted LeMat archive: {path}")
            path.rename(attempt / path.name)
    _sync_directory(attempt)
    _sync_directory(output)
    entry["status"] = "archived"
    atomic_json(history_path, history)
    return history


def restore_audit(history: list[dict], target: Path) -> None:
    """Carry verified pin evidence from cached calculations into the final audit."""
    with target.open("w") as destination:
        for attempt in history:
            source = Path(attempt["directory"]) / target.name
            if not source.exists():
                continue
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != attempt["file_sha256"][target.name]:
                raise ValueError(f"Archived reference audit changed: {source}")
            # Each attempt already includes its predecessors, so copy only the
            # most recent complete audit instead of duplicating history.
            destination.seek(0)
            destination.truncate()
            lines = raw.decode().splitlines()
            for index, line in enumerate(lines):
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    if index == len(lines) - 1 and not raw.endswith(b"\n"):
                        break  # A scheduler kill may interrupt the final append.
                    raise ValueError(f"Malformed archived reference audit: {source}")
                destination.write(line + "\n")


def save_completed(path: Path, result: Any) -> None:
    payload = dataclasses.asdict(result)
    for field in dataclasses.fields(result):
        if isinstance(payload[field.name], Path):
            payload[field.name] = str(payload[field.name])
    atomic_json(path, payload)


def load_completed(path: Path, result_class: Any, artifacts: dict[str, Path]) -> Any:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    manifest = json.loads(artifacts["manifest"].read_text())
    for record in manifest["input_cifs"]:
        staged = Path(record["path"])
        if (
            not staged.is_file()
            or hashlib.sha256(staged.read_bytes()).hexdigest() != record["sha256"]
        ):
            raise ValueError(f"Completed LeMat staged input changed: {staged}")
    for key, expected in payload["artifact_sha256"].items():
        artifact = (
            artifacts["reference_shim_dir"] / "sitecustomize.py"
            if key == "reference_shim"
            else artifacts[key]
        )
        if (
            not artifact.is_file()
            or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"Completed LeMat artifact changed: {artifact}")
    for field, artifact_name in (("raw_result", "raw"), ("metrics", "metrics")):
        archived = json.loads(artifacts[artifact_name].read_text())
        if json.dumps(payload[field], sort_keys=True) != json.dumps(
            archived, sort_keys=True
        ):
            raise ValueError(f"Completed LeMat result changed: {path} ({field})")
    for key, value in payload.items():
        if value is not None and (key.endswith("_path") or key == "output_dir"):
            payload[key] = Path(value)
    payload["command"] = tuple(payload["command"])
    return result_class(**payload)


def run_streamed(
    command,
    *,
    cwd,
    environment,
    stdout_path,
    stderr_path,
    timeout_seconds,
):
    """Persist logs during execution and kill the entire worker tree on timeout."""
    process = None
    old_handlers = {}

    def interrupted(signum, _frame):
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise LeMatResumeTimeout(f"LeMat interrupted by signal {signum}")

    def stop():
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # Workers may outlive the interpreter that spawned them.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGTERM, signal.SIGINT):
                old_handlers[signum] = signal.signal(signum, interrupted)
        with (
            Path(stdout_path).open("w") as stdout,
            Path(stderr_path).open("w") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env={**environment, "PYTHONUNBUFFERED": "1"},
                stdout=stdout,
                stderr=stderr,
                text=True,
                start_new_session=True,
            )
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                raise LeMatResumeTimeout(
                    "LeMat allocation time budget exhausted"
                ) from exc
    except BaseException:
        stop()
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout=Path(stdout_path).read_text(),
        stderr=Path(stderr_path).read_text(),
    )


def remaining_timeout(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise LeMatResumeTimeout("LeMat allocation time budget exhausted")
    return remaining
