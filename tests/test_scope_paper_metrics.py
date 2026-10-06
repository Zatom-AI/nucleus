"""Tests for the model-agnostic SCOPe paper-metrics workflow."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compute_scope_advanced_novelty_diversity.py"
)
SPEC = importlib.util.spec_from_file_location("scope_paper_metrics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _cohort_root(tmp_path: Path, rmsds: list[float]) -> Path:
    root = tmp_path / "self_consistency"
    backbone_dir = root / "shards" / "shard_00" / "backbones"
    backbone_dir.mkdir(parents=True)
    backbone = backbone_dir / "scope_len050_rep00_0_model_0.pdb"
    backbone.write_text("END\n")
    sample = backbone.stem
    details = {
        "per_backbone": [
            {
                "sample": sample,
                "backbone_pdb": str(backbone),
                "shard": "shard_00",
            }
        ],
        "per_refold": [
            {
                "sample": sample,
                "sequence_index": index,
                "scRMSD": rmsd,
                "sc_atom": "CA",
            }
            for index, rmsd in enumerate(rmsds)
        ],
    }
    (root / "self_consistency_details.json").write_text(json.dumps(details))
    return root


def test_designability_and_per_sequence_success_use_strict_threshold(tmp_path):
    root = _cohort_root(
        tmp_path,
        [1.0, 1.49, 1.5, 1.6, 2.0, 2.1, 2.2, 2.3],
    )
    cohort = MODULE._load_cohort({"output_root": str(root)}, 1.5, 8)
    metrics = MODULE._designability_metrics(cohort["entries"], 1.5, 8)

    assert metrics["best_of_8_count"] == 1
    assert metrics["best_of_8_rate"] == 1.0
    assert metrics["per_sequence_success_count"] == 2
    assert metrics["per_sequence_success_total"] == 8
    assert metrics["per_sequence_success_rate"] == pytest.approx(0.25)
    assert metrics["all_refolds_count"] == 0


def test_incomplete_refolds_are_rejected(tmp_path):
    root = _cohort_root(tmp_path, [1.0] * 7)
    with pytest.raises(ValueError, match="Expected 8 refolds"):
        MODULE._load_cohort({"output_root": str(root)}, 1.5, 8)


def test_sweep_manifest_uses_only_complete_rows(tmp_path):
    summary = tmp_path / "summary.csv"
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "complete", "output_root", "ckpt_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "step": 2500,
                "complete": 1,
                "output_root": tmp_path / "step2500",
                "ckpt_path": tmp_path / "step-00002500.ckpt",
            }
        )
        writer.writerow(
            {
                "step": 5000,
                "complete": 0,
                "output_root": tmp_path / "step5000",
                "ckpt_path": tmp_path / "step-00005000.ckpt",
            }
        )

    rows = MODULE.build_input_manifest(
        self_consistency_root=None,
        summary_csv=summary,
        model="test_model",
        label="Test model",
        cohort="checkpoint_selection",
        row_id=None,
        length_min=None,
        length_max=None,
    )

    assert len(rows) == 1
    assert rows[0]["row_id"] == "test_model_step_00002500"
    assert rows[0]["checkpoint"].endswith("step-00002500.ckpt")


def test_focused_paper_row_has_current_manuscript_metrics():
    result = {
        "row": {"row_id": "test", "model": "m", "label": "M"},
        "designability": {
            "n_backbones": 1,
            "threshold_A": 1.5,
            "expected_refolds_per_backbone": 8,
            "best_of_8_count": 1,
            "best_of_8_rate": 1.0,
            "per_sequence_success_count": 2,
            "per_sequence_success_total": 8,
            "per_sequence_success_rate": 0.25,
            "all_refolds_count": 0,
            "all_refolds_rate": 0.0,
            "all_refolds_definition": "test",
        },
        "novelty": {
            "best_of_8": {
                "novel_count_max_tm_lt_0_5": 1,
                "novel_rate_max_tm_lt_0_5": 1.0,
                "mean_max_tmalign_tm_query": 0.4,
            }
        },
        "diversity": {"best_of_8": {"n_clusters_at_0_5": 1}},
        "sequence": {"best_of_8": {"n_clusters": 1, "joint_n_clusters": 1}},
    }

    row = MODULE._paper_metric_row(result)
    assert row["designability_rate"] == 1.0
    assert row["per_sequence_success_rate"] == 0.25
    assert row["novelty_rate"] == 1.0
    assert row["distinct_tm_clusters"] == 1
    assert row["sequence_clusters"] == 1
    assert row["joint_sequence_structure_clusters"] == 1


def test_single_cohort_writes_focused_csv(tmp_path, monkeypatch):
    root = _cohort_root(tmp_path, [1.0, 1.49, 1.5, 1.6, 2.0, 2.1, 2.2, 2.3])
    targets = tmp_path / "targets"
    targets.mkdir()
    (targets / "train.pdb").write_text("END\n")
    output = tmp_path / "metrics"
    row = {
        "row_id": "dummy",
        "model": "dummy_model",
        "label": "Dummy model",
        "cohort": "paper_evaluation",
        "output_root": str(root),
    }

    monkeypatch.setattr(
        MODULE,
        "_exact_novelty",
        lambda *args, **kwargs: (
            {
                "best_of_8": {
                    "n_designable": 1,
                    "n_rescored": 1,
                    "mean_max_tmalign_tm_query": 0.4,
                    "novel_count_max_tm_lt_0_5": 1,
                    "novel_rate_max_tm_lt_0_5": 1.0,
                    "criterion": "test",
                },
                "all_refolds": {
                    "n_designable": 0,
                    "n_rescored": 0,
                    "mean_max_tmalign_tm_query": float("nan"),
                    "novel_count_max_tm_lt_0_5": 0,
                    "novel_rate_max_tm_lt_0_5": float("nan"),
                    "criterion": "test",
                },
            },
            {},
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "compute_tmalign_complete_linkage_diversity",
        lambda *args, **kwargs: {
            "tmalign_diversity_auc": 1.0,
            "tmalign_cluster_fraction_at_0.5": 1.0,
            "tmalign_n_clusters_at_0.5": 1.0,
            "tmalign_mean_pairwise_tm": float("nan"),
            "tmalign_median_pairwise_tm": float("nan"),
            "tmalign_n_structures": 1.0,
            "tmalign_n_missing_pair_scores": 0.0,
        },
    )
    monkeypatch.setattr(
        MODULE,
        "_read_tm_matrix",
        lambda *args, **kwargs: MODULE.np.ones((1, 1)),
    )
    monkeypatch.setattr(
        MODULE,
        "_sequence_metrics",
        lambda *args, **kwargs: {
            "best_of_8": {
                "n_sequences": 1,
                "n_clusters": 1,
                "cluster_fraction": 1.0,
                "joint_n_clusters": 1,
                "joint_cluster_fraction": 1.0,
            },
            "all_refolds": {
                "n_sequences": 0,
                "n_clusters": 0,
                "cluster_fraction": 0.0,
                "joint_n_clusters": 0,
                "joint_cluster_fraction": 0.0,
            },
        },
    )

    MODULE.evaluate_row(
        row,
        output,
        targets,
        "TMalign",
        "mmseqs",
        1.5,
        1,
        8,
        1,
        8,
    )
    manifest = output / "manifest.json"
    MODULE.write_manifest(manifest, [row])
    MODULE.merge_results(manifest, output, output / "plots")

    with (output / "paper_metrics.csv").open(newline="") as handle:
        focused = next(csv.DictReader(handle))
    assert focused["row_id"] == "dummy"
    assert float(focused["designability_rate"]) == 1.0
    assert float(focused["per_sequence_success_rate"]) == 0.25
    assert float(focused["novelty_rate"]) == 1.0
    assert float(focused["distinct_tm_clusters"]) == 1.0
