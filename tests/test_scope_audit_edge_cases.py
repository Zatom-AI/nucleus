"""Regression tests for SCOPe audit review edge cases."""

from __future__ import annotations

import csv
import gzip
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GENIE_METRICS = _load(
    "compute_scope_sweep_genie_metrics",
    "compute_scope_sweep_genie_metrics.py",
)
ADVANCED = _load(
    "scope_advanced_review_test",
    "compute_scope_advanced_novelty_diversity.py",
)
DIRECTIONAL = _load(
    "scope_directional_review_test",
    "compute_scope_genie_short_directional_foldseek.py",
)
EXACT = _load(
    "scope_exact_review_test",
    "compute_scope_genie_short_exact_novelty.py",
)
TM_AUDIT = _load(
    "scope_tm_audit_review_test",
    "compute_scope_genie_short_exact_tm_audit.py",
)


def test_generated_model_cifs_are_indexed_for_direct_sequence_metrics(tmp_path):
    generated = tmp_path / "shards" / "shard_00" / "generated"
    backbones = tmp_path / "shards" / "shard_00" / "backbones"
    generated.mkdir(parents=True)
    backbones.mkdir(parents=True)
    (backbones / "sample.pdb").write_text("END\n")
    with gzip.open(generated / "sample.cif.gz", "wt") as handle:
        handle.write("data_sample\n")

    indexed_backbones, indexed_generated = ADVANCED._root_local_index(tmp_path)

    assert indexed_backbones["sample.pdb"] == backbones / "sample.pdb"
    assert indexed_generated["sample"] == generated / "sample.cif.gz"


def test_all_backbone_plot_keys_match_flattened_metric_schema():
    result = {
        "row": {},
        "designability": {},
        "novelty": {},
        "diversity": {
            "all_backbones": {
                "n_clusters_at_0_5": 7,
                "cluster_fraction_at_0_5": 0.7,
            }
        },
        "sequence": {},
    }

    flat = ADVANCED._flatten(result)

    assert flat[ADVANCED.ALL_BACKBONE_CLUSTER_COUNT_KEY] == 7
    assert flat[ADVANCED.ALL_BACKBONE_CLUSTER_FRACTION_KEY] == pytest.approx(0.7)


def test_plddt_many_uses_schema_aware_mmcif_and_pdb_parsing(tmp_path):
    mmcif = tmp_path / "prediction.cif"
    mmcif.write_text(
        "\n".join(
            [
                "data_prediction",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.B_iso_or_equiv",
                "_atom_site.label_atom_id",
                "ATOM 81.25 CA",
                "ATOM 10.00 N",
                "#",
            ]
        )
        + "\n"
    )
    pdb = tmp_path / "prediction.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1      11.104  13.207  14.120  1.00 77.50           C\n"
    )

    values = GENIE_METRICS._read_plddt_many([mmcif, pdb])

    assert values[str(mmcif)] == pytest.approx(81.25)
    assert values[str(pdb)] == pytest.approx(77.5)


@pytest.mark.parametrize(
    ("module", "writer_name", "fields_name"),
    [
        (DIRECTIONAL, "_write_csv", "DETAIL_FIELDS"),
        (TM_AUDIT, "_write_csv", "DETAIL_FIELDS"),
    ],
)
def test_empty_detail_outputs_retain_schema(
    tmp_path, module, writer_name: str, fields_name: str
):
    path = tmp_path / f"{module.__name__}.csv"
    fields = getattr(module, fields_name)
    getattr(module, writer_name)(path, [], fields)

    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [fields]


def test_empty_exhaustive_detail_output_retains_schema(tmp_path):
    path = tmp_path / "exact.csv"
    EXACT.write_csv(path, [], fields=EXACT.DETAIL_FIELDS)

    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [EXACT.DETAIL_FIELDS]


def test_relative_heldout_sources_create_resolvable_links(tmp_path):
    all_pdbs = tmp_path / "all_pdbs"
    output = tmp_path / "prepared" / "heldout"
    all_pdbs.mkdir()
    (all_pdbs / "d1.pdb").write_text("END\n")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("domain_id\n")
    all_csv = tmp_path / "all.csv"
    all_csv.write_text("domain_id\nd1\n")

    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "prepare_scope2k_heldout_foldseek_pdbs.py"),
            "--train-csv",
            "train.csv",
            "--all-csv",
            "all.csv",
            "--all-pdb-dir",
            "all_pdbs",
            "--output-dir",
            "prepared/heldout",
        ],
        cwd=tmp_path,
        check=True,
    )

    link = output / "d1.pdb"
    assert link.is_symlink()
    assert link.resolve() == (all_pdbs / "d1.pdb").resolve()
    assert link.read_text() == "END\n"


