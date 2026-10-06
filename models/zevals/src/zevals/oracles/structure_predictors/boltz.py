"""Pinned subprocess adapter for the full Boltz-2 prediction interface.

Boltz has a large dependency stack that conflicts with Zevals' lightweight
environment.  This adapter therefore invokes a separately installed, pinned
Boltz executable and exchanges only YAML/mmCIF/JSON files with it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import yaml

from zevals.oracles.base import (
    BaseStructurePredictor,
    ComplexPrediction,
    StructurePrediction,
    run_subprocess,
)

BOLTZ_VERSION = "2.2.1"
BOLTZ_REVISION = "cb04aeccdd480fd4db707f0bbafde538397fa2ac"
_POLYMER_ALPHABETS = {
    "protein": frozenset("ACDEFGHIKLMNPQRSTVWY"),
    "dna": frozenset("ACGT"),
    "rna": frozenset("ACGU"),
}


class BoltzPredictor(BaseStructurePredictor):
    """Run Boltz-2 for FASTA polymers or unrestricted native Boltz inputs.

    ``predict`` is the high-throughput single-polymer convenience API used by
    the RNA evaluation pipeline.  ``predict_complex`` passes native Boltz YAML,
    FASTA, or input directories through unchanged, retaining ligands,
    modifications, MSAs, constraints, templates, affinity properties, and
    future schema additions supported by the pinned Boltz executable.
    """

    def __init__(
        self,
        entity_type: str = "rna",
        cache_dir: str | None = None,
        executable: str | None = None,
        model: str = "boltz2",
        sampling_steps: int = 200,
        seed: int = 42,
        no_kernels: bool = True,
        output_format: str = "mmcif",
        devices: int = 1,
        num_workers: int = 0,
        cli_options: dict[str, Any] | None = None,
        expected_repo_revision: str = BOLTZ_REVISION,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if entity_type not in _POLYMER_ALPHABETS:
            raise ValueError(
                "Boltz entity_type must be protein, dna, or rna; "
                f"received {entity_type!r}"
            )
        self.entity_type = entity_type
        self.cache_dir = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir
            else _default_boltz_cache()
        )
        self.executable = (
            Path(executable).expanduser().resolve()
            if executable
            else _default_boltz_environment() / "bin" / "boltz"
        )
        self.model = model
        self.sampling_steps = sampling_steps
        self.seed = seed
        self.no_kernels = no_kernels
        if output_format not in {"mmcif", "pdb"}:
            raise ValueError("Boltz output_format must be 'mmcif' or 'pdb'")
        self.output_format = output_format
        self.devices = devices
        self.num_workers = num_workers
        self.cli_options = {
            name.removeprefix("--"): value
            for name, value in dict(cli_options or {}).items()
        }
        collisions = sorted(_CORE_CLI_OPTIONS & self.cli_options.keys())
        if collisions:
            raise ValueError(
                "Boltz cli_options cannot override managed options: "
                + ", ".join(collisions)
            )
        self.expected_repo_revision = expected_repo_revision
        self._installation_validated = False

    def predict(self, fasta_path: Path, output_dir: Path) -> list[StructurePrediction]:
        """Predict every FASTA record as the configured polymer type."""
        records = _read_fasta(fasta_path)
        if not records:
            raise ValueError(f"No FASTA records found in {fasta_path}")
        inputs_dir = Path(output_dir) / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        record_ids: list[str] = []
        for index, (sequence_id, sequence) in enumerate(records):
            _validate_sequence(sequence, self.entity_type, sequence_id)
            record_id = _record_id(index, sequence_id, sequence)
            entity: dict[str, Any] = {
                "id": "A",
                "sequence": sequence,
            }
            if self.entity_type == "protein":
                entity["msa"] = "empty"
            _write_yaml(
                inputs_dir / f"{record_id}.yaml",
                {"version": 1, "sequences": [{self.entity_type: entity}]},
            )
            record_ids.append(record_id)

        output_dir = Path(output_dir).expanduser().resolve()
        self._run(inputs_dir, output_dir)
        predictions: list[StructurePrediction] = []
        for (_, sequence), record_id in zip(records, record_ids):
            for structure_path, confidence, artifacts in self._outputs(
                output_dir, record_id
            ):
                predictions.append(
                    StructurePrediction(
                        sequence=sequence,
                        pdb_path=structure_path,
                        plddt=100.0 * float(confidence["complex_plddt"]),
                        ptm=float(confidence.get("ptm", math.nan)),
                        extra={
                            "backend": f"Boltz-{BOLTZ_VERSION}",
                            "entity_type": self.entity_type,
                            **confidence,
                            **artifacts,
                        },
                    )
                )
        return predictions

    def predict_complex(
        self, input_path: Path, output_dir: Path
    ) -> list[ComplexPrediction]:
        """Predict native Boltz input without narrowing its schema.

        Args:
            input_path: A Boltz YAML/YML file, FASTA file, or directory of
                inputs. Relative MSA/template paths continue to resolve exactly
                as they do in the native Boltz CLI because the source is passed
                directly to the executable.
            output_dir: Destination root supplied to ``boltz predict``.

        Returns:
            One result per generated diffusion sample across every input record.
        """
        input_path = Path(input_path).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Boltz input does not exist: {input_path}")
        if input_path.is_file() and input_path.suffix.lower() not in {
            ".yaml",
            ".yml",
            ".fasta",
            ".fa",
        }:
            raise ValueError(
                "Full Boltz complex prediction accepts native YAML or FASTA. "
                "Use native YAML for ligands, constraints, templates, and "
                "affinity properties."
            )

        output_dir = Path(output_dir).expanduser().resolve()
        self._run(input_path, output_dir)
        native_outputs = self._discover_outputs(output_dir)
        if not native_outputs:
            raise FileNotFoundError(
                f"Boltz produced no structure/confidence pairs under {output_dir}"
            )
        return [
            ComplexPrediction(
                structure_path=structure_path,
                plddt=100.0 * float(confidence["complex_plddt"]),
                ptm=float(confidence.get("ptm", math.nan)),
                iptm=float(confidence.get("iptm", math.nan)),
                extra={
                    "backend": f"Boltz-{BOLTZ_VERSION}",
                    **confidence,
                    **artifacts,
                },
            )
            for structure_path, confidence, artifacts in native_outputs
        ]

    def _run(self, input_path: Path, output_dir: Path) -> None:
        self._validate_installation()
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(self.executable),
            "predict",
            str(input_path),
            "--out_dir",
            str(output_dir),
            "--cache",
            str(self.cache_dir),
            "--model",
            self.model,
            "--output_format",
            self.output_format,
            "--devices",
            str(self.devices),
            "--recycling_steps",
            str(self.num_recycles),
            "--sampling_steps",
            str(self.sampling_steps),
            "--diffusion_samples",
            str(self.num_samples),
            "--num_workers",
            str(self.num_workers),
            "--seed",
            str(self.seed),
        ]
        if self.no_kernels:
            cmd.append("--no_kernels")
        cmd.extend(_serialize_cli_options(self.cli_options))
        run_subprocess(cmd)

    def _outputs(
        self, output_dir: Path, record_id: str
    ) -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
        prediction_roots = [
            output_dir / "predictions",
            output_dir / "boltz_results_inputs" / "predictions",
        ]
        prediction_root = next(
            (root for root in prediction_roots if (root / record_id).is_dir()),
            prediction_roots[0],
        )
        prediction_dir = prediction_root / record_id
        outputs: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
        for sample_index in range(self.num_samples):
            extension = "cif" if self.output_format == "mmcif" else "pdb"
            structure_path = (
                prediction_dir / f"{record_id}_model_{sample_index}.{extension}"
            )
            confidence_path = (
                prediction_dir / f"confidence_{record_id}_model_{sample_index}.json"
            )
            if not structure_path.is_file() or not confidence_path.is_file():
                raise FileNotFoundError(
                    "Boltz did not produce the expected structure/confidence pair: "
                    f"{structure_path}, {confidence_path}"
                )
            outputs.append(
                (
                    structure_path,
                    json.loads(confidence_path.read_text()),
                    _optional_artifacts(prediction_dir, record_id),
                )
            )
        return outputs

    def _discover_outputs(
        self, output_dir: Path
    ) -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
        """Discover outputs for native single-file and directory inputs."""
        extension = "cif" if self.output_format == "mmcif" else "pdb"
        outputs: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
        for confidence_path in sorted(output_dir.rglob("confidence_*_model_*.json")):
            name = confidence_path.name.removeprefix("confidence_").removesuffix(
                ".json"
            )
            structure_path = confidence_path.parent / f"{name}.{extension}"
            if not structure_path.is_file():
                raise FileNotFoundError(
                    "Boltz emitted confidence without its structure: "
                    f"{confidence_path}, {structure_path}"
                )
            record_id = name.rsplit("_model_", 1)[0]
            outputs.append(
                (
                    structure_path,
                    json.loads(confidence_path.read_text()),
                    _optional_artifacts(confidence_path.parent, record_id),
                )
            )
        return outputs

    def _validate_installation(self) -> None:
        if self._installation_validated:
            return
        if not self.executable.is_file():
            raise FileNotFoundError(
                f"Boltz executable is missing: {self.executable}. "
                "Install Boltz-2 in an isolated environment and set "
                "ZEVALS_BOLTZ_EXECUTABLE."
            )
        required_cache = [self.cache_dir / "mols"]
        # Boltz-2 checks both default model files at startup, even without an
        # affinity property. Explicit checkpoint options and other supported
        # model variants must remain usable, so only enforce defaults when the
        # adapter will actually ask Boltz to use them.
        if self.model == "boltz2" and "checkpoint" not in self.cli_options:
            required_cache.append(self.cache_dir / "boltz2_conf.ckpt")
        if self.model == "boltz2" and "affinity_checkpoint" not in self.cli_options:
            required_cache.append(self.cache_dir / "boltz2_aff.ckpt")
        missing = [str(path) for path in required_cache if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Boltz cache is incomplete; missing: " + ", ".join(missing)
            )
        if self.repo_path is not None and self.expected_repo_revision:
            result = run_subprocess(
                ["git", "-C", str(self.repo_path), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            )
            revision = result.stdout.strip()
            if revision != self.expected_repo_revision:
                raise RuntimeError(
                    "Boltz checkout revision mismatch: expected "
                    f"{self.expected_repo_revision}, found {revision}"
                )
        self._installation_validated = True


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    sequence_id: str | None = None
    parts: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if sequence_id is not None:
                records.append((sequence_id, "".join(parts).upper()))
            sequence_id = line[1:].split()[0]
            parts = []
        elif line.strip():
            if sequence_id is None:
                raise ValueError(f"Sequence precedes first FASTA header in {path}")
            parts.append(line.strip())
    if sequence_id is not None:
        records.append((sequence_id, "".join(parts).upper()))
    return records


def _validate_sequence(sequence: str, entity_type: str, sequence_id: str) -> None:
    invalid = sorted(set(sequence) - _POLYMER_ALPHABETS[entity_type])
    if invalid:
        raise ValueError(
            f"Boltz {entity_type} input {sequence_id!r} contains unsupported "
            f"symbols: {invalid}"
        )


def _record_id(index: int, name: str, content: str) -> str:
    safe_name = "".join(character if character.isalnum() else "_" for character in name)
    digest = hashlib.sha256(content.encode()).hexdigest()[:10]
    return f"{index:04d}_{safe_name[:48]}_{digest}"


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


_CORE_CLI_OPTIONS = frozenset(
    {
        "out_dir",
        "cache",
        "model",
        "output_format",
        "devices",
        "recycling_steps",
        "sampling_steps",
        "diffusion_samples",
        "num_workers",
        "seed",
        "no_kernels",
    }
)


def _serialize_cli_options(options: dict[str, Any]) -> list[str]:
    """Serialize optional Boltz Click parameters without limiting its CLI."""
    arguments: list[str] = []
    for name, value in options.items():
        option = "--" + name.removeprefix("--")
        if value is True:
            arguments.append(option)
        elif value is False or value is None:
            continue
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                arguments.extend([option, str(item)])
        else:
            arguments.extend([option, str(value)])
    return arguments


def _optional_artifacts(prediction_dir: Path, record_id: str) -> dict[str, Any]:
    """Retain optional affinity and uncertainty outputs when requested."""
    artifacts: dict[str, Any] = {}
    affinity_path = prediction_dir / f"affinity_{record_id}.json"
    if affinity_path.is_file():
        artifacts["affinity"] = json.loads(affinity_path.read_text())
        artifacts["affinity_path"] = affinity_path
    for metric in ("pae", "pde", "plddt"):
        path = prediction_dir / f"{metric}_{record_id}.npz"
        if path.is_file():
            artifacts[f"{metric}_path"] = path
    return artifacts


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _default_boltz_environment() -> Path:
    return _repository_root() / "data" / "zevals" / "envs" / "boltz-v2.2.1"


def _default_boltz_cache() -> Path:
    return _repository_root() / "data" / "zevals" / "checkpoints" / "boltz-v2.2.1"
