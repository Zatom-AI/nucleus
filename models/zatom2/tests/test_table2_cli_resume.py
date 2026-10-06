"""Allocation retry status is reserved for resumable evaluation interruptions."""

import subprocess

import pytest
from typer.testing import CliRunner

import zatom2.cli as cli
import zatom2.table2_evaluation as table2


@pytest.mark.parametrize(
    "error,expected_status",
    [
        (table2.EvaluationPaused("allocation budget"), 75),
        (subprocess.TimeoutExpired("oracle", 1), 75),
        (RuntimeError("invalid checkpoint"), 1),
        (None, 0),
    ],
)
def test_cli_only_retries_timeout_failures(
    monkeypatch, tmp_path, error, expected_status
):
    seen = []

    def evaluate(cfg):
        seen.append(cfg)
        if error is not None:
            raise error
        return {}

    monkeypatch.setattr(table2, "run_table2_evaluation", evaluate)
    result = CliRunner().invoke(
        cli.app,
        [
            "evaluate-table2",
            f"ckpt_path={tmp_path / 'checkpoint.ckpt'}",
            f"out_dir={tmp_path / 'evaluation'}",
            "evaluation.resume=true",
            "evaluation.walltime_seconds=13800",
        ],
    )
    assert result.exit_code == expected_status, result.output
    assert len(seen) == 1
    assert seen[0].evaluation.resume is True
    assert seen[0].evaluation.walltime_seconds == 13800
    if expected_status == 75:
        assert "resume the same out_dir" in result.output
