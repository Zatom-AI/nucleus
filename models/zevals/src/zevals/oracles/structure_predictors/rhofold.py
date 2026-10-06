"""RhoFold+ RNA structure-prediction oracle.

RhoFold+ has an isolated, legacy dependency stack, so Zevals invokes its
official model implementation in a pinned external checkout rather than
importing it into the Python 3.12 Zevals process. Single-sequence inference is
the reproducible benchmark default, while supplied/generated MSAs and upstream
Amber relaxation remain available.

Hydra config: configs/oracle/structure_predictor/rhofold.yaml
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from zevals.oracles.base import (
    BaseStructurePredictor,
    StructurePrediction,
    run_subprocess,
)
from zevals.utils.devices import resolve_device

_VALID_RNA = frozenset("ACGU")


class RhoFoldPredictor(BaseStructurePredictor):
    """Run pinned RhoFold+ inference in its isolated environment."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str | None = None,
        relax_steps: int = 0,
        single_sequence: bool = True,
        msa_paths: dict[str, str] | None = None,
        database_path: str | None = None,
        binary_path: str | None = None,
        expected_repo_revision: str | None = None,
        expected_checkpoint_sha256: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.checkpoint_path = (
            Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
        )
        self.device = resolve_device(device)
        self.relax_steps = relax_steps
        self.single_sequence = single_sequence
        self.msa_paths = {
            name: str(Path(path).expanduser().resolve())
            for name, path in (msa_paths or {}).items()
        }
        self.database_path = (
            Path(database_path).expanduser().resolve() if database_path else None
        )
        self.binary_path = (
            Path(binary_path).expanduser().resolve() if binary_path else None
        )
        if self.single_sequence and self.msa_paths:
            raise ValueError("msa_paths cannot be used with single_sequence=True")
        self.expected_repo_revision = expected_repo_revision
        self.expected_checkpoint_sha256 = expected_checkpoint_sha256
        self._checkpoint_validated = False
        self._repo_validated = False

    def predict(self, fasta_path: Path, output_dir: Path) -> list[StructurePrediction]:
        """Fold every standard A/C/G/U FASTA record with RhoFold+."""
        repo_path = self._validated_repo_path()
        checkpoint_path = self._validated_checkpoint_path()
        records = _read_fasta(fasta_path)
        if not records:
            raise ValueError(f"No FASTA records found in {fasta_path}")

        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        normalized_fasta = output_dir / "input.fasta"
        fasta_lines: list[str] = []
        for sequence_id, sequence in records:
            invalid = sorted(set(sequence) - _VALID_RNA)
            if invalid:
                raise ValueError(
                    f"RhoFold+ accepts only A/C/G/U; {sequence_id!r} contains "
                    f"{invalid}. Translate the inverse-folder alphabet first."
                )

            fasta_lines.extend((f">{sequence_id}", sequence))
        normalized_fasta.write_text("\n".join(fasta_lines) + "\n")

        msa_manifest: Path | None = None
        if self.msa_paths:
            missing_ids = sorted(
                sequence_id
                for sequence_id, _ in records
                if sequence_id not in self.msa_paths
            )
            missing_files = sorted(
                path for path in self.msa_paths.values() if not Path(path).is_file()
            )
            if missing_ids or missing_files:
                raise FileNotFoundError(
                    "RhoFold+ MSA inputs are incomplete; missing IDs: "
                    f"{missing_ids}; missing files: {missing_files}"
                )
            msa_manifest = output_dir / "msa_manifest.json"
            msa_manifest.write_text(json.dumps(self.msa_paths, indent=2))

        python_command, subprocess_conda_env = self._python_command()
        cmd = [
            python_command,
            str(_resolve_runner_path()),
            "--repo-path",
            str(repo_path),
            "--input-fas",
            str(normalized_fasta),
            "--output-dir",
            str(output_dir),
            "--ckpt",
            str(checkpoint_path),
            "--device",
            self.device,
            "--relax-steps",
            str(self.relax_steps),
        ]
        if self.single_sequence:
            cmd.append("--single-sequence")
        elif msa_manifest is not None:
            cmd.extend(["--msa-manifest", str(msa_manifest)])
        else:
            if self.database_path is None or self.binary_path is None:
                raise ValueError(
                    "RhoFold+ generated-MSA mode requires database_path and "
                    "binary_path"
                )
            cmd.extend(
                [
                    "--database-path",
                    str(self.database_path),
                    "--binary-path",
                    str(self.binary_path),
                ]
            )
        bind_paths = [
            str(repo_path),
            str(checkpoint_path.parent),
            str(output_dir),
        ]
        bind_paths.extend(str(Path(path).parent) for path in self.msa_paths.values())
        if self.database_path is not None:
            bind_paths.append(str(self.database_path))
        if self.binary_path is not None:
            bind_paths.append(str(self.binary_path))
        run_subprocess(
            cmd,
            conda_env=subprocess_conda_env,
            singularity_image=self.singularity_image,
            bind_paths=bind_paths,
            cwd=repo_path,
        )

        predictions: list[StructurePrediction] = []
        for index, (sequence_id, sequence) in enumerate(records):
            record_dir = output_dir / f"{index:04d}"
            unrelaxed_path = record_dir / "unrelaxed_model.pdb"
            relaxed_path = record_dir / f"relaxed_{self.relax_steps}_model.pdb"
            pdb_path = relaxed_path if self.relax_steps > 0 else unrelaxed_path
            if not pdb_path.is_file():
                raise FileNotFoundError(
                    f"RhoFold+ did not produce its expected output: {pdb_path}"
                )
            predictions.append(
                StructurePrediction(
                    sequence=sequence,
                    pdb_path=pdb_path,
                    # Relaxation does not preserve RhoFold confidence in the
                    # B-factor field, so always read it from the raw prediction.
                    plddt=_mean_pdb_confidence(unrelaxed_path),
                    ptm=math.nan,
                    extra={
                        "backend": "RhoFold+",
                        "mode": _inference_mode(
                            self.single_sequence, bool(self.msa_paths)
                        ),
                        "relaxed": self.relax_steps > 0,
                        "sequence_id": sequence_id,
                        "unrelaxed_path": unrelaxed_path,
                        "secondary_structure_path": record_dir / "ss.ct",
                        "distribution_path": record_dir / "results.npz",
                    },
                )
            )
        return predictions

    def _python_command(self) -> tuple[str, str | None]:
        """Use an environment's interpreter directly when given an env path.

        This avoids requiring ``conda`` to be loaded on every compute node.
        Named environments still use the generic ``conda run -n`` path.
        """
        conda_env = self.conda_env or str(_default_rhofold_environment())
        if Path(conda_env).expanduser().is_dir():
            python_path = Path(conda_env).expanduser() / "bin" / "python"
            if not python_path.is_file():
                raise FileNotFoundError(
                    f"RhoFold+ environment has no Python interpreter: {python_path}"
                )
            return str(python_path), None
        return "python", conda_env

    def _validated_repo_path(self) -> Path:
        if self.repo_path is None:
            self.repo_path = _default_rhofold_repo()
        self.repo_path = self.repo_path.expanduser().resolve()
        inference_path = self.repo_path / "inference.py"
        if not inference_path.is_file():
            raise FileNotFoundError(
                f"Missing RhoFold+ inference script: {inference_path}"
            )
        if self.expected_repo_revision and not self._repo_validated:
            result = run_subprocess(
                ["git", "-C", str(self.repo_path), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            )
            revision = result.stdout.strip()
            if revision != self.expected_repo_revision:
                raise RuntimeError(
                    "RhoFold+ checkout revision mismatch: expected "
                    f"{self.expected_repo_revision}, found {revision}"
                )
            self._repo_validated = True
        return self.repo_path

    def _validated_checkpoint_path(self) -> Path:
        checkpoint_path = self.checkpoint_path
        if checkpoint_path is None:
            from zevals.setup import default_checkpoint_dir

            local_checkpoint = (
                _source_repository_root()
                / "data"
                / "zevals"
                / "checkpoints"
                / "rhofold"
                / "RhoFold_pretrained.pt"
            )
            checkpoint_path = (
                local_checkpoint
                if local_checkpoint.is_file()
                else default_checkpoint_dir() / "rhofold" / "RhoFold_pretrained.pt"
            )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"No RhoFold+ checkpoint at {checkpoint_path}. Run `zevals setup "
                "--checkpoints rhofold` or set ZEVALS_RHOFOLD_CHECKPOINT."
            )
        if self.expected_checkpoint_sha256 and not self._checkpoint_validated:
            digest = hashlib.sha256()
            with checkpoint_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != self.expected_checkpoint_sha256:
                raise RuntimeError(
                    "RhoFold+ checkpoint SHA-256 mismatch: expected "
                    f"{self.expected_checkpoint_sha256}, found {actual_sha256}"
                )
            self._checkpoint_validated = True
        self.checkpoint_path = checkpoint_path
        return checkpoint_path


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    sequence_id: str | None = None
    sequence_parts: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if sequence_id is not None:
                records.append((sequence_id, "".join(sequence_parts).upper()))
            sequence_id = line[1:].split()[0]
            sequence_parts = []
        elif line.strip():
            if sequence_id is None:
                raise ValueError(
                    f"Sequence data precedes the first FASTA header in {path}"
                )
            sequence_parts.append(line.strip())
    if sequence_id is not None:
        records.append((sequence_id, "".join(sequence_parts).upper()))
    return records


def _mean_pdb_confidence(pdb_path: Path) -> float:
    """Read mean residue confidence from RhoFold+'s PDB B-factor column."""
    c1_confidences: list[float] = []
    all_confidences: list[float] = []
    for line in pdb_path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 66:
            continue
        try:
            confidence = float(line[60:66])
        except ValueError:
            continue
        all_confidences.append(confidence)
        if line[12:16].strip() == "C1'":
            c1_confidences.append(confidence)
    values = c1_confidences or all_confidences
    return sum(values) / len(values) if values else math.nan


def _inference_mode(single_sequence: bool, supplied_msa: bool) -> str:
    if single_sequence:
        return "single_sequence"
    return "supplied_msa" if supplied_msa else "generated_msa"


def _resolve_runner_path() -> Path:
    """Locate the packaged Python-3.7-compatible RhoFold+ runner."""
    runner_path = Path(__file__).with_name("_rhofold_runner.py")
    if not runner_path.is_file():
        raise FileNotFoundError(f"Missing packaged RhoFold+ runner: {runner_path}")
    return runner_path


def _source_repository_root() -> Path:
    """Return the monorepo root for a source checkout, or a harmless candidate."""
    return Path(__file__).resolve().parents[6]


def _default_rhofold_repo() -> Path:
    local_path = _source_repository_root() / "data" / "zevals" / "oracles" / "rhofold"
    cache_root = _zevals_cache_root() / "oracles" / "rhofold"
    return local_path if local_path.is_dir() else cache_root


def _default_rhofold_environment() -> Path:
    local_path = _source_repository_root() / "data" / "zevals" / "envs" / "rhofold"
    cache_root = _zevals_cache_root() / "envs" / "rhofold"
    return local_path if local_path.is_dir() else cache_root


def _zevals_cache_root() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "zevals"
