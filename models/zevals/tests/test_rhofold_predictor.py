"""Tests for the subprocess-backed RhoFold+ structure predictor."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zevals.oracles.structure_predictors import rhofold  # noqa: E402


def _write_confidence_pdb(path: Path) -> None:
    path.write_text(
        "ATOM      1  C1'   A A   1       0.000   0.000   0.000  1.00 80.00\n"
        "ATOM      2  C1'   U A   2       1.000   0.000   0.000  1.00 60.00\n"
        "END\n"
    )


def test_predict_batches_fasta_and_parses_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "inference.py").write_text("")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    fasta = tmp_path / "designed.fasta"
    fasta.write_text(">first\nACGU\n>second\nUGCA\n")
    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        for index in range(2):
            record_dir = output_dir / f"{index:04d}"
            record_dir.mkdir(parents=True)
            _write_confidence_pdb(record_dir / "unrelaxed_model.pdb")

    monkeypatch.setattr(rhofold, "run_subprocess", fake_run)
    predictor = rhofold.RhoFoldPredictor(
        repo_path=str(repo),
        checkpoint_path=str(checkpoint),
        conda_env="rhofold",
        device="cpu",
    )

    predictions = predictor.predict(fasta, tmp_path / "outputs")

    assert [prediction.sequence for prediction in predictions] == ["ACGU", "UGCA"]
    assert [prediction.plddt for prediction in predictions] == [70.0, 70.0]
    assert len(captured) == 1
    assert all("--repo-path" in cmd for cmd in captured)
    assert all(cmd[cmd.index("--relax-steps") + 1] == "0" for cmd in captured)
    assert "--single-sequence" in captured[0]


def test_predict_rejects_nonstandard_rna_before_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "inference.py").write_text("")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    fasta = tmp_path / "bad.fasta"
    fasta.write_text(">bad\nACGN\n")
    monkeypatch.setattr(
        rhofold,
        "run_subprocess",
        lambda *args, **kwargs: pytest.fail("subprocess should not run"),
    )
    predictor = rhofold.RhoFoldPredictor(
        repo_path=str(repo), checkpoint_path=str(checkpoint), device="cpu"
    )

    with pytest.raises(ValueError, match="only A/C/G/U"):
        predictor.predict(fasta, tmp_path / "outputs")


def test_predict_requires_expected_output_pdb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "inference.py").write_text("")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">rna\nACGU\n")
    monkeypatch.setattr(rhofold, "run_subprocess", lambda *args, **kwargs: None)
    predictor = rhofold.RhoFoldPredictor(
        repo_path=str(repo), checkpoint_path=str(checkpoint), device="cpu"
    )

    with pytest.raises(FileNotFoundError, match="expected output"):
        predictor.predict(fasta, tmp_path / "outputs")


def test_environment_path_uses_its_python_without_conda(tmp_path: Path):
    env = tmp_path / "env"
    python = env / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    predictor = rhofold.RhoFoldPredictor(conda_env=str(env), device="cpu")

    assert predictor._python_command() == (str(python), None)


def test_packaged_runner_is_available() -> None:
    assert rhofold._resolve_runner_path().name == "_rhofold_runner.py"


def test_predict_preserves_supplied_msa_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "inference.py").write_text("")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    msa = tmp_path / "rna.a3m"
    msa.write_text(">rna\nACGU\n")
    fasta = tmp_path / "rna.fasta"
    fasta.write_text(">rna\nACGU\n")
    captured: list[str] = []

    def fake_run(cmd, **kwargs):
        captured.extend(cmd)
        output_dir = Path(cmd[cmd.index("--output-dir") + 1]) / "0000"
        output_dir.mkdir(parents=True)
        _write_confidence_pdb(output_dir / "unrelaxed_model.pdb")

    monkeypatch.setattr(rhofold, "run_subprocess", fake_run)
    predictor = rhofold.RhoFoldPredictor(
        repo_path=str(repo),
        checkpoint_path=str(checkpoint),
        device="cpu",
        single_sequence=False,
        msa_paths={"rna": str(msa)},
    )

    predictions = predictor.predict(fasta, tmp_path / "outputs")

    assert "--msa-manifest" in captured
    assert "--single-sequence" not in captured
    assert predictions[0].extra["mode"] == "supplied_msa"


def test_predict_uses_relaxed_output_when_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "inference.py").write_text("")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    fasta = tmp_path / "rna.fasta"
    fasta.write_text(">rna\nACGU\n")

    def fake_run(cmd, **kwargs):
        output_dir = Path(cmd[cmd.index("--output-dir") + 1]) / "0000"
        output_dir.mkdir(parents=True)
        _write_confidence_pdb(output_dir / "unrelaxed_model.pdb")
        _write_confidence_pdb(output_dir / "relaxed_25_model.pdb")

    monkeypatch.setattr(rhofold, "run_subprocess", fake_run)
    predictor = rhofold.RhoFoldPredictor(
        repo_path=str(repo),
        checkpoint_path=str(checkpoint),
        device="cpu",
        relax_steps=25,
    )

    prediction = predictor.predict(fasta, tmp_path / "outputs")[0]

    assert prediction.pdb_path.name == "relaxed_25_model.pdb"
    assert prediction.extra["relaxed"] is True
