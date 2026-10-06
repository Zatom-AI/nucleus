"""Tests for the isolated Boltz-2 predictor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from zevals.oracles.structure_predictors import boltz


def _predictor(tmp_path: Path, **kwargs) -> boltz.BoltzPredictor:
    executable = tmp_path / "env" / "bin" / "boltz"
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.write_text("")
    cache = tmp_path / "cache"
    (cache / "mols").mkdir(parents=True)
    (cache / "boltz2_conf.ckpt").write_bytes(b"weights")
    (cache / "boltz2_aff.ckpt").write_bytes(b"weights")
    return boltz.BoltzPredictor(
        executable=str(executable), cache_dir=str(cache), **kwargs
    )


def _fake_boltz_run(cmd: list[str], **kwargs) -> None:
    inputs_dir = Path(cmd[2])
    output_dir = Path(cmd[cmd.index("--out_dir") + 1])
    num_samples = int(cmd[cmd.index("--diffusion_samples") + 1])
    for input_path in inputs_dir.glob("*.yaml"):
        record_id = input_path.stem
        prediction_dir = output_dir / "boltz_results_inputs" / "predictions" / record_id
        prediction_dir.mkdir(parents=True)
        for index in range(num_samples):
            (prediction_dir / f"{record_id}_model_{index}.cif").write_text("data_x\n")
            (prediction_dir / f"confidence_{record_id}_model_{index}.json").write_text(
                json.dumps(
                    {
                        "complex_plddt": 0.81,
                        "ptm": 0.72,
                        "iptm": 0.63,
                        "confidence_score": 0.79,
                    }
                )
            )


def test_predict_writes_dna_yaml_and_parses_confidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">dna one\nACGT\n")
    monkeypatch.setattr(boltz, "run_subprocess", _fake_boltz_run)
    predictor = _predictor(tmp_path, entity_type="dna", num_samples=2)

    predictions = predictor.predict(fasta, tmp_path / "outputs")

    assert len(predictions) == 2
    assert predictions[0].plddt == pytest.approx(81.0)
    assert predictions[0].ptm == pytest.approx(0.72)
    payload = yaml.safe_load(
        next((tmp_path / "outputs" / "inputs").glob("*.yaml")).read_text()
    )
    assert payload["sequences"] == [{"dna": {"id": "A", "sequence": "ACGT"}}]


def test_predict_rejects_wrong_polymer_alphabet_before_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">bad\nACGU\n")
    monkeypatch.setattr(
        boltz,
        "run_subprocess",
        lambda *args, **kwargs: pytest.fail("subprocess should not run"),
    )
    predictor = _predictor(tmp_path, entity_type="dna")

    with pytest.raises(ValueError, match="unsupported symbols"):
        predictor.predict(fasta, tmp_path / "outputs")


def test_native_yaml_retains_full_schema_and_cli_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    native_input = tmp_path / "complex.yaml"
    payload = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": "MKT", "msa": "empty"}},
            {"ligand": {"id": "L", "smiles": "CCO"}},
        ],
        "constraints": [
            {
                "pocket": {
                    "binder": "L",
                    "contacts": [["A", 1], ["A", 2]],
                    "max_distance": 6.0,
                }
            }
        ],
        "templates": [{"cif": "template.cif", "chain_id": "A", "template_id": "B"}],
        "properties": [{"affinity": {"binder": "L"}}],
    }
    native_input.write_text(yaml.safe_dump(payload, sort_keys=False))
    original = native_input.read_bytes()
    commands: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs) -> None:
        commands.append(cmd)
        output_dir = Path(cmd[cmd.index("--out_dir") + 1])
        prediction_dir = (
            output_dir / "boltz_results_complex" / "predictions" / "complex"
        )
        prediction_dir.mkdir(parents=True)
        (prediction_dir / "complex_model_0.cif").write_text("data_x\n")
        (prediction_dir / "confidence_complex_model_0.json").write_text(
            json.dumps({"complex_plddt": 0.91, "ptm": 0.8, "iptm": 0.74})
        )
        (prediction_dir / "affinity_complex.json").write_text(
            json.dumps({"affinity_pred_value": -1.25})
        )

    monkeypatch.setattr(boltz, "run_subprocess", fake_run)
    predictor = _predictor(
        tmp_path,
        cli_options={
            "use_msa_server": True,
            "write_full_pae": True,
            "sampling_steps_affinity": 100,
        },
    )

    predictions = predictor.predict_complex(native_input, tmp_path / "outputs")

    assert native_input.read_bytes() == original
    assert commands[0][2] == str(native_input.resolve())
    assert "--use_msa_server" in commands[0]
    assert "--write_full_pae" in commands[0]
    affinity_option = commands[0].index("--sampling_steps_affinity")
    assert commands[0][affinity_option + 1] == "100"
    assert predictions[0].plddt == pytest.approx(91.0)
    assert predictions[0].iptm == pytest.approx(0.74)
    assert predictions[0].extra["affinity"]["affinity_pred_value"] == -1.25


def test_native_input_rejects_non_boltz_structure_file(tmp_path: Path) -> None:
    structure = tmp_path / "complex.cif"
    structure.write_text("data_x\n")
    predictor = _predictor(tmp_path)

    with pytest.raises(ValueError, match="Use native YAML"):
        predictor.predict_complex(structure, tmp_path / "outputs")


def test_cli_options_cannot_override_managed_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="managed options: devices"):
        _predictor(tmp_path, cli_options={"devices": 8})
