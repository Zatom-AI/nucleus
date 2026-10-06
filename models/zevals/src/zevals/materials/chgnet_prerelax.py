"""Resumable CHGNet pre-relaxation before materials benchmark scoring.

The numerical protocol follows FlowMM's released environment: CHGNet 0.3.1,
the bundled 0.3.0 checkpoint, FIRE, ExpCellFilter, 0.1 eV/Angstrom, and 1,500
steps. CHGNet runs in an isolated interpreter; importing this module does not
import CHGNet, PyTorch, ASE, or pymatgen. Every attempted CIF retains its name,
including parse/relaxation failures, which explicitly fall back to the original.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from ._material_cif import normalize_material_cif
else:
    # The pinned Python 3.11 worker executes this file directly, without zevals
    # installed. The sibling helper is lazy and needs only stdlib at import.
    from _material_cif import normalize_material_cif

CHGNET_VERSION = "0.3.1"
CHGNET_MODEL = "0.3.0"
CHGNET_ASE_VERSION = "3.22.1"
CHGNET_CHECKPOINT_FILE = "pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar"
CHGNET_CHECKPOINT_SHA256 = (
    "d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1"
)
"""Digest verified against the official CHGNet 0.3.1 Linux CPython 3.11 wheel."""
CHGNET_PROTOCOL_VERSION = 2
_MANIFEST = "chgnet_prerelax.manifest.json"
_STATUSES = {"relaxed", "fallback_parse_error", "fallback_relaxation_error"}


@dataclass(frozen=True)
class CHGNetPrerelaxationResult:
    """Validated pre-relaxation outputs, with completion separate from convergence."""

    cif_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    complete: bool


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> None:
    """Commit one file on the same filesystem, preserving the previous version."""
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".pending-", delete=False
    ) as out:
        temporary = Path(out.name)
        try:
            out.write(content)
            out.flush()
            os.fsync(out.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
    )


def _input_inventory(input_dir: Path) -> list[dict[str, str]]:
    if not input_dir.is_dir():
        raise ValueError(f"CIF input directory does not exist: {input_dir}")
    paths = sorted([*input_dir.glob("*.cif"), *input_dir.glob("*.cif.gz")])
    if not paths:
        raise ValueError(f"No .cif or .cif.gz files in {input_dir}")
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("CHGNet inputs must be ordinary CIF files, not symlinks")
    return [{"name": path.name, "sha256": _sha256(path)} for path in paths]


def _configuration(
    *, fmax: float, steps: int, relax_cell: bool, device: str, seed: int
) -> dict[str, Any]:
    if not math.isfinite(fmax) or fmax <= 0:
        raise ValueError("fmax must be finite and positive")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if not isinstance(relax_cell, bool):
        raise ValueError("relax_cell must be boolean")
    if (
        device != "cpu"
        and device != "cuda"
        and not (device.startswith("cuda:") and device[5:].isdigit())
    ):
        raise ValueError("device must be cpu, cuda, or cuda:<index>")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    return {
        "chgnet_version": CHGNET_VERSION,
        "model": CHGNET_MODEL,
        "checkpoint_sha256": CHGNET_CHECKPOINT_SHA256,
        "ase_version": CHGNET_ASE_VERSION,
        "optimizer": "FIRE",
        "cell_filter": "ExpCellFilter",
        "fmax_eV_per_A": float(fmax),
        "steps": steps,
        "relax_cell": relax_cell,
        "stress_weight": 1 / 160.21766208,
        "on_isolated_atoms": "warn",
        "device": device,
        "seed": seed,
        "failure_policy": "fallback_original",
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
        "torch_num_threads": 1,
    }


def _contract(input_dir: Path, configuration: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": CHGNET_PROTOCOL_VERSION,
        "input_dir": str(input_dir),
        "inputs": _input_inventory(input_dir),
        "configuration": configuration,
        "worker_sha256": _sha256(Path(__file__)),
        "cif_conversion_sha256": _sha256(Path(__file__).with_name("_material_cif.py")),
    }


def _records(
    output_dir: Path, contract: dict[str, Any], contract_sha256: str
) -> list[dict[str, Any]]:
    records = []
    expected_names = {item["name"] for item in contract["inputs"]}
    record_dir = output_dir / "records"
    if record_dir.exists():
        unexpected = {
            path.name.removesuffix(".json") for path in record_dir.glob("*.json")
        } - expected_names
        if unexpected:
            raise ValueError(
                f"Unexpected CHGNet completion records: {sorted(unexpected)}"
            )
    for item in contract["inputs"]:
        record_path = record_dir / f"{item['name']}.json"
        if not record_path.exists():
            continue
        record = json.loads(record_path.read_text())
        output_path = output_dir / "cifs" / item["name"]
        if (
            record.get("name") != item["name"]
            or record.get("input_sha256") != item["sha256"]
            or record.get("contract_sha256") != contract_sha256
            or record.get("status") not in _STATUSES
            or not output_path.is_file()
            or output_path.is_symlink()
            or record_path.is_symlink()
            or not isinstance(record.get("converged"), bool)
            or record.get("output_sha256") != _sha256(output_path)
        ):
            raise ValueError(
                f"CHGNet completion record/output failed validation: {record_path}"
            )
        if record["status"].startswith("fallback_") and (
            record["output_sha256"] != item["sha256"]
            or record.get("converged") is not False
        ):
            raise ValueError(f"CHGNet fallback differs from original: {item['name']}")
        records.append(record)
    return records


def _load_result(
    output_dir: Path, contract: dict[str, Any], require_complete: bool
) -> CHGNetPrerelaxationResult:
    manifest_path = output_dir / _MANIFEST
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("contract") != contract:
        raise ValueError(
            "CHGNet inputs, configuration, or worker changed; use a new output directory"
        )
    contract_sha256 = _json_digest(contract)
    if manifest.get("contract_sha256") != contract_sha256:
        raise ValueError("CHGNet manifest contract digest does not match")
    runtime = manifest.get("runtime", {})
    if manifest.get("runtime_sha256") != _json_digest(runtime):
        raise ValueError("CHGNet runtime manifest digest does not match")
    checkpoint = runtime.get("checkpoint", {})
    packages = runtime.get("packages", {})
    requested_device = contract["configuration"]["device"]
    actual_device = runtime.get("actual_device", "")
    if (
        packages.get("chgnet") != CHGNET_VERSION
        or packages.get("ase") != CHGNET_ASE_VERSION
        or checkpoint.get("model") != CHGNET_MODEL
        or not isinstance(checkpoint.get("sha256"), str)
        or len(checkpoint["sha256"]) != 64
        or checkpoint["sha256"] != checkpoint.get("wheel_record_sha256")
        or checkpoint["sha256"] != CHGNET_CHECKPOINT_SHA256
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("cublas_workspace_config") != ":4096:8"
        or (requested_device == "cpu" and actual_device != "cpu")
        or (
            requested_device.startswith("cuda") and not actual_device.startswith("cuda")
        )
    ):
        raise ValueError(
            "CHGNet runtime does not satisfy the pinned numerical protocol"
        )
    records = _records(output_dir, contract, contract_sha256)
    for record in records:
        if record.get("runtime_sha256") != manifest["runtime_sha256"]:
            raise ValueError("CHGNet completion records mix different runtimes")
    complete = len(records) == len(contract["inputs"])
    # The manifest may lag one atomic record if a job ended during aggregation.
    if manifest.get("complete"):
        if not complete or manifest.get("records_sha256") != _json_digest(records):
            raise ValueError("Completed CHGNet manifest does not match its records")
        if (
            manifest.get("attempted_count") != len(contract["inputs"])
            or manifest.get("completed_count") != len(records)
            or manifest.get("converged_count")
            != sum(record["converged"] for record in records)
            or manifest.get("status_counts")
            != {
                status: sum(record["status"] == status for record in records)
                for status in sorted(_STATUSES)
            }
        ):
            raise ValueError("CHGNet aggregate counts do not match completion records")
    if require_complete and (not complete or not manifest.get("complete")):
        raise ValueError("CHGNet pre-relaxation is incomplete; resume before scoring")
    expected_names = {item["name"] for item in contract["inputs"]}
    actual_names = {
        path.name
        for pattern in ("*.cif", "*.cif.gz")
        for path in (output_dir / "cifs").glob(pattern)
    }
    if actual_names - expected_names:
        raise ValueError("Unexpected CIFs in CHGNet output directory")
    manifest = dict(manifest, completed_count=len(records), records=records)
    return CHGNetPrerelaxationResult(
        output_dir / "cifs",
        manifest_path,
        manifest,
        bool(complete and manifest.get("complete")),
    )


def validate_chgnet_prerelaxation(
    input_dir: str | Path,
    *,
    output_dir: str | Path,
    fmax: float = 0.1,
    steps: int = 1500,
    relax_cell: bool = True,
    device: str = "cuda",
    seed: int = 42,
    require_complete: bool = True,
) -> CHGNetPrerelaxationResult:
    """Read and verify persisted inputs, protocol, and per-CIF output hashes.

    This read-only operation does not require the CHGNet environment or GPU.
    Resuming numerical work additionally checks the active environment against
    the runtime recorded at the first invocation.
    """
    configuration = _configuration(
        fmax=fmax, steps=steps, relax_cell=relax_cell, device=device, seed=seed
    )
    return _load_result(
        Path(output_dir).resolve(),
        _contract(Path(input_dir).resolve(), configuration),
        require_complete,
    )


def run_chgnet_prerelaxation(
    input_dir: str | Path,
    *,
    output_dir: str | Path,
    runner_command: Sequence[str] = ("python",),
    fmax: float = 0.1,
    steps: int = 1500,
    relax_cell: bool = True,
    device: str = "cuda",
    seed: int = 42,
    resume: bool = True,
    timeout_seconds: float | None = None,
) -> CHGNetPrerelaxationResult:
    """Pre-relax CIFs in an isolated interpreter with atomic per-CIF resumption.

    Args:
        input_dir: Persisted original CIFs; never modified.
        output_dir: Private directory for outputs, records, manifest, and logs.
        runner_command: Interpreter command, normally ``[env / 'bin/python']``.
        fmax: ASE force tolerance in eV/Angstrom, including the cell filter.
        steps: Maximum FIRE steps per CIF.
        relax_cell: Also optimize the lattice using the historical ExpCellFilter.
        device: Explicit CPU or CUDA device.
        seed: Base seed; reset deterministically per filename before relaxation.
        resume: Verify and skip existing completed CIFs.
        timeout_seconds: Process time budget. Expiry returns ``complete=False``
            if the worker has initialized its manifest; committed CIFs survive.

    Returns:
        Validated outputs and provenance. Only score when ``complete`` is true.
    """
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if (
        output_dir == input_dir
        or output_dir in input_dir.parents
        or input_dir in output_dir.parents
    ):
        raise ValueError("CHGNet input and output directories must be disjoint")
    if (
        isinstance(runner_command, str)
        or not runner_command
        or not all(isinstance(part, str) and part for part in runner_command)
    ):
        raise ValueError("runner_command must be a nonempty list of command arguments")
    if timeout_seconds is not None and (
        not math.isfinite(timeout_seconds) or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be positive")
    configuration = _configuration(
        fmax=fmax, steps=steps, relax_cell=relax_cell, device=device, seed=seed
    )
    contract = _contract(input_dir, configuration)
    if (output_dir / _MANIFEST).exists():
        if not resume:
            raise FileExistsError(f"CHGNet output already exists: {output_dir}")
        result = _load_result(output_dir, contract, require_complete=False)
        if result.complete:
            return result
    output_dir.mkdir(parents=True, exist_ok=True)
    request = {"contract": contract, "output_dir": str(output_dir), "resume": resume}
    # 2,500 CIF hashes exceed Linux's per-argument length limit. Use an isolated
    # request file rather than passing the inventory as one command argument.
    with tempfile.NamedTemporaryFile(
        dir=output_dir, prefix=".request-", suffix=".json", delete=False
    ) as handle:
        request_path = Path(handle.name)
        handle.write(json.dumps(request).encode())
    command = [
        *runner_command,
        str(Path(__file__).resolve()),
        "--request",
        str(request_path),
    ]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(
        PYTHONHASHSEED=str(seed),
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    timed_out = False
    with (
        (output_dir / "worker.stdout.log").open("ab") as stdout,
        (output_dir / "worker.stderr.log").open("ab") as stderr,
    ):
        try:
            process = subprocess.run(
                command,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
        else:
            if process.returncode not in (0, 75):
                raise RuntimeError(
                    f"CHGNet worker exited {process.returncode}; see {output_dir / 'worker.stderr.log'}. "
                    "Completed CIFs are preserved for resumption."
                )
        finally:
            request_path.unlink(missing_ok=True)
    if not (output_dir / _MANIFEST).is_file():
        if timed_out:
            raise TimeoutError(
                "CHGNet timeout occurred before runtime initialization; retry with a longer budget"
            )
        raise RuntimeError("CHGNet worker did not initialize its manifest")
    return _load_result(output_dir, contract, require_complete=False)


class _Interrupted(BaseException):
    """Stop current work without recording a scientific relaxation failure."""


class _CHGNetRuntime:
    """Optional scientific imports are confined to the worker interpreter."""

    def __init__(self, configuration: Mapping[str, Any]) -> None:
        import numpy as np
        import torch
        from ase.optimize.fire import FIRE
        from chgnet.model import CHGNet, StructOptimizer
        from chgnet.model import model as model_module
        from pymatgen.core import Structure

        for package, expected in (
            ("chgnet", CHGNET_VERSION),
            ("ase", CHGNET_ASE_VERSION),
        ):
            if importlib.metadata.version(package) != expected:
                raise RuntimeError(
                    f"Pre-relaxation requires {package}=={expected}; run setup_chgnet_prerelax.sh"
                )
        if configuration["device"].startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for CHGNet but is unavailable")
        torch.set_num_threads(configuration["torch_num_threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.np, self.torch, self.Structure = np, torch, Structure
        self.configuration = configuration
        checkpoint = (
            Path(model_module.__file__).resolve().parent.parent / CHGNET_CHECKPOINT_FILE
        )
        checkpoint_digest = _sha256(checkpoint)
        if checkpoint_digest != CHGNET_CHECKPOINT_SHA256:
            raise RuntimeError(
                "CHGNet bundled checkpoint differs from the pinned 0.3.0 weights"
            )
        distribution = importlib.metadata.distribution("chgnet")
        entry = next(
            (
                entry
                for entry in distribution.files or []
                if Path(distribution.locate_file(entry)).resolve() == checkpoint
            ),
            None,
        )
        if entry is None or entry.hash is None or entry.hash.mode != "sha256":
            raise RuntimeError(
                "Bundled CHGNet checkpoint must have a wheel RECORD SHA-256"
            )
        record_digest = base64.urlsafe_b64decode(
            entry.hash.value + "=" * (-len(entry.hash.value) % 4)
        ).hex()
        if checkpoint_digest != record_digest:
            raise RuntimeError("CHGNet checkpoint differs from installed wheel RECORD")
        self.model = CHGNet.from_file(
            str(checkpoint), mlp_out_bias=False, version=CHGNET_MODEL
        )
        self.model.eval()
        runtime = self

        class ReportingFIRE(FIRE):
            def run(self, *args: Any, **kwargs: Any) -> bool:
                converged = super().run(*args, **kwargs)
                runtime.last_optimizer = {
                    "converged": bool(converged),
                    "optimizer_steps": int(self.nsteps),
                    "final_max_filter_force_eV_per_A": float(
                        np.linalg.norm(self.atoms.get_forces(), axis=1).max()
                    ),
                }
                return converged

        self.last_optimizer: dict[str, Any] = {}
        self.optimizer = StructOptimizer(
            model=self.model,
            optimizer_class=ReportingFIRE,
            use_device=configuration["device"],
            stress_weight=configuration["stress_weight"],
            on_isolated_atoms=configuration["on_isolated_atoms"],
        )
        actual_device = str(self.optimizer.calculator.device)
        device_properties = None
        if actual_device.startswith("cuda"):
            properties = torch.cuda.get_device_properties(actual_device)
            device_properties = {
                "name": properties.name,
                "total_memory": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
            }
        package_sources = {}
        for package in ("chgnet", "ase", "pymatgen"):
            dist = importlib.metadata.distribution(package)
            files = {}
            for file in dist.files or []:
                if file.suffix in (".py", ".so"):
                    path = Path(dist.locate_file(file))
                    if path.is_file():
                        files[str(file)] = _sha256(path)
            package_sources[package] = _json_digest(files)
        self.provenance = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "packages": dict(
                sorted(
                    (dist.metadata["Name"].lower(), dist.version)
                    for dist in importlib.metadata.distributions()
                    if dist.metadata["Name"]
                )
            ),
            "package_sources_sha256": package_sources,
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_digest,
                "wheel_record_sha256": record_digest,
                "model": CHGNET_MODEL,
            },
            "actual_device": actual_device,
            "device_properties": device_properties,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }

    def read_structure(self, source: Path) -> Any:
        """Read materials CIF or generated mmCIF without changing source bytes."""
        original = source.read_bytes()
        normalized = normalize_material_cif(source)
        if normalized == original:
            return self.Structure.from_file(source)
        if source.name.lower().endswith(".gz"):
            normalized = gzip.decompress(normalized)
        return self.Structure.from_str(normalized.decode("utf-8"), fmt="cif")

    def relax(self, source: Path) -> tuple[bytes, dict[str, Any]]:
        """Return one final CIF and ASE endpoint/convergence metadata."""
        structure = self.read_structure(source)
        per_sample_seed = (
            self.configuration["seed"]
            + int(hashlib.sha256(source.name.encode()).hexdigest()[:8], 16)
        ) % 2**32
        random.seed(per_sample_seed)
        self.np.random.seed(per_sample_seed)
        self.torch.manual_seed(per_sample_seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(per_sample_seed)
        self.last_optimizer = {}
        result = self.optimizer.relax(
            structure,
            fmax=self.configuration["fmax_eV_per_A"],
            steps=self.configuration["steps"],
            relax_cell=self.configuration["relax_cell"],
            loginterval=1,
            verbose=False,
        )
        final_structure, trajectory = result["final_structure"], result["trajectory"]
        energies = [float(energy) for energy in trajectory.energies]
        if not energies or not all(math.isfinite(energy) for energy in energies):
            raise ValueError("CHGNet returned nonfinite or empty energy trajectory")
        if (
            not self.np.isfinite(final_structure.cart_coords).all()
            or not self.np.isfinite(final_structure.lattice.matrix).all()
            or final_structure.volume <= 0
        ):
            raise ValueError("CHGNet returned invalid positions or lattice")
        if final_structure.composition != structure.composition or len(
            final_structure
        ) != len(structure):
            raise ValueError("CHGNet changed composition or number of sites")
        if not self.last_optimizer:
            raise RuntimeError("CHGNet did not report optimizer convergence")
        metadata = {
            **self.last_optimizer,
            "num_sites": len(structure),
            "sample_seed": per_sample_seed,
            "initial_energy_eV": energies[0],
            "final_energy_eV": energies[-1],
            "initial_energy_eV_per_atom": energies[0] / len(structure),
            "final_energy_eV_per_atom": energies[-1] / len(structure),
            "trajectory_frames": len(energies),
            "final_max_atomic_force_eV_per_A": float(
                self.np.linalg.norm(trajectory.forces[-1], axis=1).max()
            ),
        }
        if not all(math.isfinite(value) for value in metadata.values()):
            raise ValueError("CHGNet returned nonfinite endpoint statistics")
        return final_structure.to(fmt="cif").encode(), metadata


def _fatal_exception(error: Exception) -> bool:
    filesystem_or_memory_error = isinstance(
        error, (MemoryError, OSError)
    ) and not isinstance(error, gzip.BadGzipFile)
    return filesystem_or_memory_error or any(
        marker in str(error).lower()
        for marker in ("cuda", "cublas", "cudnn", "out of memory", "deterministic")
    )


def _run_worker(request: dict[str, Any]) -> bool:
    contract = request["contract"]
    output_dir = Path(request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another CHGNet worker owns this output directory"
            ) from error
        if contract != _contract(
            Path(contract["input_dir"]), contract["configuration"]
        ):
            raise ValueError("CHGNet inputs or worker changed before subprocess start")
        runtime = _CHGNetRuntime(contract["configuration"])
        contract_digest, runtime_digest = (
            _json_digest(contract),
            _json_digest(runtime.provenance),
        )
        manifest_path = output_dir / _MANIFEST
        if manifest_path.exists():
            if not request["resume"]:
                raise FileExistsError(
                    f"CHGNet manifest already exists: {manifest_path}"
                )
            prior = _load_result(output_dir, contract, require_complete=False)
            if prior.manifest["runtime"] != runtime.provenance:
                raise ValueError(
                    "CHGNet environment, checkpoint, or device changed during resume"
                )
            if prior.complete:
                return True
        elif any((output_dir / name).exists() for name in ("records", "cifs")):
            raise ValueError("CHGNet outputs exist without their protocol manifest")
        manifest = {
            "contract": contract,
            "contract_sha256": contract_digest,
            "runtime": runtime.provenance,
            "runtime_sha256": runtime_digest,
            "complete": False,
            "attempted_count": len(contract["inputs"]),
        }
        _atomic_json(manifest_path, manifest)
        for name in ("records", "cifs"):
            (output_dir / name).mkdir(exist_ok=True)
        existing = {
            record["name"] for record in _records(output_dir, contract, contract_digest)
        }
        for item in contract["inputs"]:
            if item["name"] in existing:
                continue
            source = Path(contract["input_dir"]) / item["name"]
            if _sha256(source) != item["sha256"]:
                raise ValueError(f"CHGNet source changed during execution: {source}")
            start = time.monotonic()
            record = {
                "name": item["name"],
                "input_sha256": item["sha256"],
                "contract_sha256": contract_digest,
                "runtime_sha256": runtime_digest,
                "converged": False,
            }
            try:
                runtime.read_structure(source)
            except Exception as error:
                if _fatal_exception(error):
                    raise
                content = source.read_bytes()
                record.update(
                    status="fallback_parse_error",
                    error=f"{type(error).__name__}: {error}",
                )
            else:
                try:
                    content, metadata = runtime.relax(source)
                except Exception as error:
                    if _fatal_exception(error):
                        raise
                    content = source.read_bytes()
                    record.update(
                        status="fallback_relaxation_error",
                        error=f"{type(error).__name__}: {error}",
                    )
                else:
                    record.update(status="relaxed", **metadata)
                    if source.name.endswith(".cif.gz"):
                        content = gzip.compress(content, mtime=0)
            if _sha256(source) != item["sha256"]:
                raise ValueError(f"CHGNet source changed during relaxation: {source}")
            record.update(
                output_sha256=hashlib.sha256(content).hexdigest(),
                elapsed_seconds=time.monotonic() - start,
            )
            _atomic_bytes(output_dir / "cifs" / item["name"], content)
            # The record is the commit marker. An orphan output is rerun safely.
            _atomic_json(output_dir / "records" / f"{item['name']}.json", record)
        if contract != _contract(
            Path(contract["input_dir"]), contract["configuration"]
        ):
            raise ValueError("CHGNet inputs changed before finalization")
        if (
            _sha256(Path(runtime.provenance["checkpoint"]["path"]))
            != runtime.provenance["checkpoint"]["sha256"]
        ):
            raise ValueError("CHGNet checkpoint changed during execution")
        records = _records(output_dir, contract, contract_digest)
        manifest.update(
            complete=True,
            completed_count=len(records),
            records_sha256=_json_digest(records),
            status_counts={
                status: sum(record["status"] == status for record in records)
                for status in sorted(_STATUSES)
            },
            converged_count=sum(record["converged"] for record in records),
        )
        _atomic_json(manifest_path, manifest)
        return True


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()

    def interrupt(signum: int, frame: Any) -> None:
        raise _Interrupted()

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        return 0 if _run_worker(json.loads(args.request.read_text())) else 75
    except _Interrupted:
        return 75


if __name__ == "__main__":
    raise SystemExit(_main())
