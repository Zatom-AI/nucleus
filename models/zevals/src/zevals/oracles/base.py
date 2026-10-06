"""Abstract base classes for swappable oracle backends.

## Execution model

`zevals` deps are installed collectively via the `zevals[...]`
extras (uv-managed, not conda), so most oracles run **in-process**: either
by importing the tool as a normal Python dependency (e.g. `fair-esm`), or —
for tools with no installable package, like ProteinMPNN — by vendoring the
minimal inference code under a private `_<tool>/` subpackage next to the
wrapper (see `inverse_folders/_protein_mpnn/` for the reference layout) and
calling it directly. Frozen weights are never vendored; only `weights_dir`
/ `checkpoint_path`-style config fields point at them.

A `subprocess` escape hatch still exists for tools that are impractical to
vendor (heavy/conflicting native deps, HPC-only binaries):

  in-process  (conda_env=None): caller manages the environment themselves.
              Useful when running inside the oracle's own conda env already.

  subprocess  (conda_env set):  the wrapper serialises inputs to temp files,
              shells out to the tool's own inference script inside its env,
              then deserialises the outputs. Zero dependency conflicts.

Config fields shared by all subprocess-capable oracles:
  conda_env         — path to the oracle's conda env (or env name). null = in-process.
  repo_path         — path to the cloned oracle repo (needed by most Python tools).
  singularity_image — alternative to conda_env on HPC clusters.
  tmp_dir           — scratch directory for intermediate files (default /tmp).

Four oracle roles:
  BaseStructurePredictor — forward-fold sequences into 3-D structures.
  BaseInverseFolder      — inverse-fold a backbone into candidate sequences.
  BasePoseValidator      — assess physical validity of a ligand binding pose.
  BaseStructureSearcher  — search / cluster structures for novelty + diversity.

## Adding a new oracle

Swapping in a different tool (or an entirely new one) in place of an
existing oracle doesn't require touching any pipeline code — implement the
matching `Base*` ABC below and point a Hydra config's `_target_` at it. The
in-process, torch-based oracles (`ESMFoldPredictor`, `ProteinMPNNFolder`,
`LigandMPNNFolder`) all follow the same shape; match it for consistency:

  1. Subclass the `Base*` ABC matching your oracle's role and implement its
     one abstract method (`predict`/`design`/`validate`/`search`+`cluster`).
  2. Resolve the `device` constructor param via
     `zevals.utils.devices.resolve_device(device)` rather than
     hand-rolling device selection — this gives you automatic GPU
     round-robin placement across oracles for free (and respects an
     explicit `device: "cuda:N"` override unchanged).
  3. Lazily load the model on first use and cache it on `self._model` (see
     any existing oracle's `_get_model()`) — never load in `__init__`.
     Construction should be cheap; the actual checkpoint load happens once,
     on first `predict()`/`design()`/etc. call.
  4. After loading, call `model.eval()` and `model.requires_grad_(False)` —
     oracles are always frozen, never a training target. Wrap every forward
     pass in `torch.no_grad()`.
  5. Construct your oracle once (e.g. via `ZevalSuite`, which does this for
     every pipeline it builds) and reuse it across an entire run — this is
     what makes both the lazy-load cache and the device round-robin
     assignment behave correctly; re-instantiating per call defeats both.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Shared result types
# ---------------------------------------------------------------------------


@dataclass
class StructurePrediction:
    """One predicted structure for a single input sequence."""

    sequence: str
    pdb_path: Path  # path to output PDB written by the oracle
    plddt: float  # mean pLDDT across all residues [0, 100]
    ptm: float  # predicted TM-score [0, 1]; nan if unavailable
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplexPrediction:
    """One predicted multi-entity complex.

    Unlike :class:`StructurePrediction`, this result does not pretend that a
    complex has one representative sequence.  ``extra`` retains backend-native
    outputs such as affinity predictions and per-chain confidence matrices.
    """

    structure_path: Path
    plddt: float
    ptm: float
    iptm: float = float("nan")
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class InverseFoldResult:
    """Designed sequences for a single input backbone."""

    sequences: list[str]
    scores: list[float]  # log-likelihood or similar per sequence
    extra: dict[str, Any] = field(default_factory=dict)


def write_fasta(
    path: Path, name: str, sequences: list[str], scores: list[float]
) -> None:
    """Write designed sequences + scores to a debug FASTA — shared by every
    in-process inverse-folder oracle's `design()` (ProteinMPNN/LigandMPNN/
    NA-MPNN all call this identically)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, (seq, score) in enumerate(zip(sequences, scores)):
        lines += [f">{name}_{i}, score={score:.4f}", seq]
    path.write_text("\n".join(lines) + "\n")


@dataclass
class PoseValidationResult:
    """Validation outcome for a single protein–ligand complex."""

    passes: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Output from a structure search / cluster operation."""

    query_id: str
    hits: list[dict[str, Any]]  # list of {target, tm_score, ...}
    cluster_id: str | None = None  # set after clustering


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _build_conda_prefix(conda_env: str) -> list[str]:
    """Return the command prefix to run inside a conda env."""
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-p" if os.sep in conda_env else "-n",
        conda_env,
    ]


def _build_singularity_prefix(
    image: str, bind_paths: list[str] | None = None
) -> list[str]:
    binds = []
    for p in bind_paths or []:
        binds += ["--bind", p]
    return ["singularity", "exec", *binds, image]


def run_subprocess(
    cmd: list[str],
    conda_env: str | None = None,
    singularity_image: str | None = None,
    bind_paths: list[str] | None = None,
    check: bool = True,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """Run `cmd`, optionally inside a conda env or Singularity container."""
    if singularity_image:
        cmd = _build_singularity_prefix(singularity_image, bind_paths) + cmd
    elif conda_env:
        cmd = _build_conda_prefix(conda_env) + cmd
    return subprocess.run(cmd, check=check, **kwargs)


def resolve_binary(name: str, configured_path: str | None = None) -> str:
    """Resolve a Zevals tool from an explicit path, staged source data, or PATH."""
    if configured_path:
        return configured_path
    env_bin_dir = os.environ.get("ZEVALS_BIN_DIR")
    if env_bin_dir:
        candidate = Path(env_bin_dir).expanduser() / name
        if candidate.is_file():
            return str(candidate)
    repository_root = Path(__file__).resolve().parents[5]
    local_candidate = repository_root / "data" / "zevals" / "bin" / name
    if local_candidate.is_file():
        return str(local_candidate)
    return shutil.which(name) or name


# ---------------------------------------------------------------------------
# BaseStructurePredictor
# ---------------------------------------------------------------------------


class BaseStructurePredictor(ABC):
    """Forward-fold sequences into 3-D structures.

    Implementations include ESMFoldPredictor, BoltzPredictor, RhoFoldPredictor,
    and RhoFoldPredictor.

    Config parameters (Hydra YAML):
      conda_env         — oracle's conda env; null = in-process.
      repo_path         — path to the cloned oracle repo.
      singularity_image — alternative to conda_env on HPC.
      tmp_dir           — scratch directory for I/O files.
      num_samples       — structure samples per sequence.
      num_recycles      — recycling iterations.
    """

    def __init__(
        self,
        num_samples: int = 1,
        num_recycles: int = 4,
        conda_env: str | None = None,
        repo_path: str | None = None,
        singularity_image: str | None = None,
        tmp_dir: str = "/tmp/zevals",
        **kwargs: Any,
    ) -> None:
        self.num_samples = num_samples
        self.num_recycles = num_recycles
        self.conda_env = conda_env
        self.repo_path = Path(repo_path) if repo_path else None
        self.singularity_image = singularity_image
        self.tmp_dir = Path(tmp_dir)

    @abstractmethod
    def predict(self, fasta_path: Path, output_dir: Path) -> list[StructurePrediction]:
        """Run structure prediction for all sequences in `fasta_path`.

        Args:
            fasta_path:  FASTA file with one or more sequences.
            output_dir:  Directory where PDB outputs will be written.

        Returns:
            One StructurePrediction per input sequence (× num_samples).
        """

    def predict_one(self, sequence: str, output_dir: Path) -> list[StructurePrediction]:
        """Convenience wrapper: predict a single sequence."""
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        fasta = self.tmp_dir / "single.fasta"
        fasta.write_text(f">seq\n{sequence}\n")
        return self.predict(fasta, output_dir)


# ---------------------------------------------------------------------------
# BaseInverseFolder
# ---------------------------------------------------------------------------


class BaseInverseFolder(ABC):
    """Inverse-fold a backbone into candidate sequences.

    Concrete implementations: ProteinMPNNFolder, NAMPNNFolder.

    Config parameters (Hydra YAML):
      conda_env       — oracle's conda env; null = in-process.
      repo_path       — path to the cloned oracle repo.
      num_sequences   — sequences designed per backbone.
      temperature     — sampling temperature.
      backbone_noise  — Gaussian noise (Å) added to coords before design.
    """

    def __init__(
        self,
        num_sequences: int = 8,
        temperature: float = 0.1,
        backbone_noise: float = 0.0,
        conda_env: str | None = None,
        repo_path: str | None = None,
        singularity_image: str | None = None,
        tmp_dir: str = "/tmp/zevals",
        **kwargs: Any,
    ) -> None:
        self.num_sequences = num_sequences
        self.temperature = temperature
        self.backbone_noise = backbone_noise
        self.conda_env = conda_env
        self.repo_path = Path(repo_path) if repo_path else None
        self.singularity_image = singularity_image
        self.tmp_dir = Path(tmp_dir)

    @abstractmethod
    def design(self, pdb_path: Path, output_dir: Path) -> InverseFoldResult:
        """Design sequences for the backbone in `pdb_path`.

        Args:
            pdb_path:   Input backbone structure file (`.cif` or `.pdb`) —
                        concrete implementations dispatch on the file's own
                        suffix (see e.g. `ProteinMPNNFolder.design`).
            output_dir: Directory where output FASTA / score files are written.

        Returns:
            InverseFoldResult with `num_sequences` candidates and their scores.
        """


# ---------------------------------------------------------------------------
# BasePoseValidator
# ---------------------------------------------------------------------------


class BasePoseValidator(ABC):
    """Assess the physical validity of a protein–ligand binding pose."""

    @abstractmethod
    def validate(
        self,
        ligand_path: Path,
        protein_path: Path | None = None,
    ) -> PoseValidationResult:
        """Validate a ligand pose, optionally with the receptor context."""


# ---------------------------------------------------------------------------
# BaseStructureSearcher
# ---------------------------------------------------------------------------


class BaseStructureSearcher(ABC):
    """Search and cluster protein/RNA structures for novelty and diversity.

    Concrete implementations: FoldSeekSearcher.

    Config parameters (Hydra YAML):
      binary_path — path to the tool binary (e.g. /usr/local/bin/foldseek).
                    Falls back to PATH if null.
      tmp_dir     — scratch directory for intermediate databases and results.
    """

    def __init__(
        self,
        binary_path: str | None = None,
        tmp_dir: str = "/tmp/zevals",
        **kwargs: Any,
    ) -> None:
        self.binary_path = binary_path
        self.tmp_dir = Path(tmp_dir)

    @abstractmethod
    def search(
        self,
        query_dir: Path,
        target_db: Path,
        output_tsv: Path,
    ) -> list[SearchResult]:
        """Search structure files (`.cif` or `.pdb`) in `query_dir` against
        `target_db`, write hits to `output_tsv`."""

    @abstractmethod
    def cluster(
        self,
        structure_dir: Path,
        output_tsv: Path,
        min_tm: float = 0.5,
    ) -> list[SearchResult]:
        """All-vs-all cluster structure files (`.cif` or `.pdb`) in
        `structure_dir`, write cluster TSV."""
