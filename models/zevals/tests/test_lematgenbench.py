from __future__ import annotations

import gzip
import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zevals.materials import lematgenbench as lemat  # noqa: E402


@pytest.fixture
def fake_runtime_probe(monkeypatch: pytest.MonkeyPatch):
    """Avoid requiring the heavyweight pinned Torch env in adapter unit tests."""

    def probe(**kwargs):
        payload = {
            "requested_device_policy": kwargs["expected_device"],
            "requested_cuda_visible_devices": kwargs["requested_cuda_visible_devices"],
            "effective_cuda_visible_devices": kwargs["environment"].get(
                "CUDA_VISIBLE_DEVICES"
            ),
            "cuda_device_order": kwargs["environment"].get("CUDA_DEVICE_ORDER"),
            "cuda_available": kwargs["expected_device"] == "cuda",
            "cuda_device_count": int(kwargs["expected_device"] == "cuda"),
            "cuda_devices": [],
            "python_version": "test",
            "torch_version": "test",
        }
        kwargs["stdout_path"].write_text("runtime probe\n")
        kwargs["stderr_path"].write_text("")
        kwargs["output_path"].write_text(json.dumps(payload, sort_keys=True) + "\n")
        return (
            payload,
            ("/fake/python", "-c", "runtime-probe"),
            lemat._sha256_file(kwargs["output_path"]),
        )

    monkeypatch.setattr(lemat, "_probe_runtime_environment", probe)
    return probe


def _representative_raw_result() -> dict:
    return {
        "run_info": {"n_structures": 2500},
        "validity_filtering": {
            "total_input_structures": 2500,
            "valid_structures": 2213,
            "validity_rate": 0.8852,
            "validity_mandatory": True,
        },
        "results": {
            "validity": (
                "BenchmarkResult(evaluator_results={}, "
                "final_scores={'overall_validity_ratio': np.float64(0.8852), "
                "'overall_validity_count': 2213, 'total_structures': 2500, "
                "'avg_charge_deviation': np.float64(nan)}, metadata={})"
            ),
            "distribution": {
                "final_scores": {
                    "JSDistance": 0.11,
                    "MMD": 0.22,
                    "FrechetDistance": 0.33,
                }
            },
            "diversity": (
                "BenchmarkResult(evaluator_results={}, "
                "final_scores={'element_diversity': np.float64(0.4), "
                "'space_group_diversity': np.float32(0.5), "
                "'site_number_diversity': 0.6, "
                "'physical_size_diversity': 0.7}, metadata={})"
            ),
            "uniqueness": (
                "BenchmarkResult(evaluator_results={}, "
                "final_scores={'unique_structures_count': np.int64(2110), "
                "'uniqueness_ratio': np.float64(0.9535)}, metadata={})"
            ),
            "novelty": (
                "BenchmarkResult(evaluator_results={}, "
                "final_scores={'novel_structures_count': 203, "
                "'novelty_ratio': 0.0962}, metadata={})"
            ),
            "stability": (
                "BenchmarkResult(evaluator_results={}, "
                "final_scores={'stable_count': 68, 'stable_ratio': 0.0307, "
                "'metastable_count': 1362, 'metastable_ratio': 0.6154, "
                "'n_valid_structures_orb': 2213, "
                "'n_valid_structures_mace': 2213, "
                "'n_valid_structures_uma': 2213}, "
                "metadata={})"
            ),
            "sun": (
                "BenchmarkResult(evaluator_results={}, "
                "final_scores={'stable_count': 68, 'metastable_count': 1294, "
                "'sun_count': 1, "
                "'msun_count': 9, 'sun_rate': 0.0005, "
                "'msun_rate': 0.0041, 'sun_indices': [3]}, metadata={})"
            ),
        },
    }


def test_extract_lematgenbench_metrics_supports_current_and_structured_json():
    metrics = lemat.extract_lematgenbench_metrics(_representative_raw_result())

    assert list(metrics) == sorted(metrics)
    assert metrics["validity/overall_validity_ratio"] == pytest.approx(0.8852)
    assert metrics["validity/overall_validity_count"] == 2213.0
    assert (
        metrics["validity/avg_charge_deviation"]
        != metrics["validity/avg_charge_deviation"]
    )
    assert metrics["distribution/FrechetDistance"] == pytest.approx(0.33)
    assert metrics["diversity/space_group_diversity"] == pytest.approx(0.5)
    assert metrics["uniqueness/unique_structures_count"] == 2110.0
    assert metrics["uniqueness/uniqueness_ratio"] == pytest.approx(0.9535)
    assert metrics["attempt_normalized/unique_rate"] == pytest.approx(2110 / 2500)
    assert metrics["attempt_normalized/metastable_rate"] == pytest.approx(1294 / 2500)
    assert "sun/sun_indices" not in metrics
    assert "validity_filtering/validity_mandatory" not in metrics


def test_result_repr_parser_never_evaluates_calls(tmp_path: Path):
    marker = tmp_path / "executed"
    malicious = (
        "BenchmarkResult(final_scores=__import__('pathlib').Path("
        f"{str(marker)!r}).touch())"
    )

    metrics = lemat.extract_lematgenbench_metrics({"results": {"validity": malicious}})

    assert metrics == {}
    assert not marker.exists()


def test_result_repr_parser_tolerates_non_python_evaluator_repr():
    raw_result = {
        "results": {
            "diversity": (
                "BenchmarkResult(evaluator_results={'metric': <MetricResult>}, "
                "final_scores={'element_diversity': np.float64(0.75), "
                "'nested': {'count': np.int64(4)}}, metadata={})"
            )
        }
    }

    assert lemat.extract_lematgenbench_metrics(raw_result) == {
        "diversity/element_diversity": 0.75,
        "diversity/nested/count": 4.0,
    }


def test_select_table2_metrics_uses_original_input_denominator():
    metrics = lemat.extract_lematgenbench_metrics(_representative_raw_result())

    selected = lemat.select_table2_metrics(
        metrics,
        distribution_key="distribution/FrechetDistance",
        diversity_key="diversity/space_group_diversity",
    )

    assert selected == {
        "overall_valid": pytest.approx(2213 / 2500),
        "distribution": pytest.approx(0.33),
        "diversity": pytest.approx(0.5),
        "unique": pytest.approx(2110 / 2500),
        "novel": pytest.approx(203 / 2500),
        "stable": pytest.approx(68 / 2500),
        "sun": pytest.approx(1 / 2500),
        "metasun": pytest.approx(9 / 2500),
    }

    with pytest.raises(ValueError, match="distribution_key"):
        lemat.select_table2_metrics(
            metrics,
            distribution_key="distribution/not-a-real-metric",
            diversity_key="diversity/space_group_diversity",
        )


def test_prepare_cif_directory_includes_compressed_and_deduplicates_symlinks(
    tmp_path: Path,
):
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    plain = cif_dir / "a.cif"
    compressed = cif_dir / "b.cif.gz"
    plain.write_bytes(b"plain")
    compressed.write_bytes(b"compressed")
    (cif_dir / "alias.cif").symlink_to(plain)
    (cif_dir / "ignored.txt").write_text("ignored")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    manifest_path, hashes = lemat._prepare_cif_input(cif_dir, output_dir)

    staged_plain = (output_dir / "input_cifs" / "000000.cif").resolve()
    staged_compressed = (output_dir / "input_cifs" / "000001.cif.gz").resolve()
    assert manifest_path.read_text().splitlines() == [
        str(staged_plain),
        str(staged_compressed),
    ]
    assert [item["path"] for item in hashes] == [
        str(staged_plain),
        str(staged_compressed),
    ]
    assert [item["source_path"] for item in hashes] == [
        str(plain.resolve()),
        str(compressed.resolve()),
    ]
    assert staged_plain.read_bytes() == b"plain"
    assert staged_compressed.read_bytes() == b"compressed"


@pytest.mark.parametrize("compressed", [False, True])
def test_material_cif_bootstrap_normalizes_only_staged_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, compressed: bool
):
    """Exercise the actual pymatgen boundary without launching any oracles."""
    from zevals.materials._material_cif import normalize_material_cif

    mmcif = b"""data_generated
_cell.length_a 4.5
_cell.length_b 5.5
_cell.length_c 6.5
_cell.angle_alpha 90
_cell.angle_beta 90
_cell.angle_gamma 90
_space_group.name_H-M_alt 'P 1'
loop_
_atom_site.label_atom_id
_atom_site.type_symbol
_atom_site.fract_x
_atom_site.fract_y
_atom_site.fract_z
_atom_site.occupancy
F1 F 0.8 0.7 0.6 1
Li1 Li 0.125 0.25 0.375 1
"""
    input_dir = tmp_path / "generated"
    input_dir.mkdir()
    suffix = ".cif.gz" if compressed else ".cif"
    raw = input_dir / f"a-mmcif{suffix}"
    raw.write_bytes(gzip.compress(mmcif, mtime=123) if compressed else mmcif)
    core = input_dir / f"b-core{suffix}"
    core.write_bytes(normalize_material_cif(raw))
    invalid = input_dir / "c-invalid.cif"
    invalid.write_text("not a CIF structure")
    outside = tmp_path / f"not-staged{suffix}"
    outside.write_bytes(raw.read_bytes())
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    manifest, records = lemat._prepare_cif_input(input_dir, output_dir)
    staged = [Path(record["path"]) for record in records]
    before = {path: path.read_bytes() for path in [raw, core, invalid, *staged]}
    original_reader = lemat.Structure.from_file.__func__
    delegated = []

    def observed_reader(cls, filename, *args, **kwargs):
        delegated.append(Path(filename))
        return original_reader(cls, filename, *args, **kwargs)

    monkeypatch.setattr(lemat.Structure, "from_file", classmethod(observed_reader))
    runner_path = tmp_path / "run_benchmarks.py"
    monkeypatch.setattr(sys, "argv", ["-c", str(runner_path), "--cifs", str(manifest)])
    monkeypatch.setattr(sys, "path", list(sys.path))
    calls = []

    def runner(path, *, run_name):
        calls.append((path, run_name))

        class SubStructure(lemat.Structure):
            pass

        material = SubStructure.from_file(staged[0], sort=True)
        assert isinstance(material, SubStructure)
        assert [str(specie) for specie in material.species] == ["Li", "F"]
        assert material.lattice.abc == pytest.approx([4.5, 5.5, 6.5])
        assert material.frac_coords[0] == pytest.approx([0.125, 0.25, 0.375])
        lemat.Structure.from_file(staged[1])
        with pytest.raises(ValueError):
            lemat.Structure.from_file(staged[2])
        with pytest.raises(ValueError):
            lemat.Structure.from_file(outside)

    monkeypatch.setattr(runpy, "run_path", runner)
    source = lemat._cif_loader_bootstrap_source(lemat._material_cif_source())
    exec(compile(source, "<cif-loader-bootstrap>", "exec"), {})

    assert calls == [(str(runner_path), "__main__")]
    assert delegated == [staged[1], staged[2], outside]
    assert lemat.Structure.from_file.__func__ is observed_reader
    assert {path: path.read_bytes() for path in before} == before
    assert not lemat._rehash_staged_inputs(records)
    assert all(record["sha256"] == record["post_run_sha256"] for record in records)


def test_canonical_local_reference_inventory_checks_expected_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    reference_path = "data/lematbulk_jsdistance_distributions.json"
    local_file = tmp_path / reference_path
    local_file.parent.mkdir(parents=True)
    local_file.write_bytes(b"commit-bound reference")
    expected_sha256 = lemat._sha256_file(local_file)
    monkeypatch.setitem(
        lemat.LEMAT_GENBENCH_LOCAL_REFERENCE_SHA256,
        reference_path,
        expected_sha256,
    )

    inventory = lemat._canonical_reference_inventory(
        tmp_path, lemat.LEMAT_GENBENCH_CONFIG
    )

    assert inventory[reference_path] == {
        "path": str(local_file),
        "exists": True,
        "is_file": True,
        "expected_sha256": expected_sha256,
        "size_bytes": len(b"commit-bound reference"),
        "sha256": expected_sha256,
        "matches_expected": True,
    }
    inventory[reference_path]["matches_expected"] = False
    errors = lemat._local_reference_integrity_errors(
        inventory,
        families=("distribution",),
    )
    assert reference_path in errors


def test_run_rejects_empty_explicit_family_list():
    with pytest.raises(ValueError, match="families must be non-empty"):
        lemat.run_lematgenbench(
            "unused",
            repo_path="unused",
            run_name="empty-families",
            output_dir=None,
            families=(),
        )


def test_run_lematgenbench_preserves_raw_result_and_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_runtime_probe
):
    repo_path = tmp_path / "upstream"
    (repo_path / "scripts").mkdir(parents=True)
    (repo_path / "scripts" / "run_benchmarks.py").write_text("# stub\n")
    (repo_path / "pyproject.toml").write_text("[project]\nname='stub'\n")
    (repo_path / "uv.lock").write_text("version = 1\n")
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    (cif_dir / "sample.cif").write_text("data_sample\n")
    output_dir = tmp_path / "artifacts"
    raw_result = _representative_raw_result()
    raw_result["run_info"]["n_structures"] = 1
    raw_result["validity_filtering"]["total_input_structures"] = 1
    raw_result["results"]["validity"] = raw_result["results"]["validity"].replace(
        "'total_structures': 2500", "'total_structures': 1"
    )

    monkeypatch.setattr(
        lemat,
        "_inspect_checkout",
        lambda *args, **kwargs: lemat._CheckoutInfo(
            commit=lemat.LEMAT_GENBENCH_COMMIT,
            lock_sha256="lock-digest",
            pyproject_sha256="project-digest",
        ),
    )

    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        result_dir = repo_path / "results_final"
        result_dir.mkdir(exist_ok=True)
        result_path = result_dir / (
            "checkpoint-100_comprehensive_multi_mlip_hull_20260904_120000.json"
        )
        result_path.write_text(json.dumps(raw_result))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"Results saved to: {result_path}\n",
            stderr="upstream log\n",
        )

    monkeypatch.setattr(lemat.subprocess, "run", fake_run)

    result = lemat.run_lematgenbench(
        cif_dir,
        repo_path=repo_path,
        run_name="checkpoint-100",
        output_dir=output_dir,
        families=lemat.LEMAT_GENBENCH_TABLE2_FAMILIES,
        runner_command=("/fake/python",),
        verify_mace_checkpoint=False,
        enforce_reference_revisions=False,
    )

    assert result.raw_result == raw_result
    assert result.upstream_commit == lemat.LEMAT_GENBENCH_COMMIT
    assert result.family_errors == {}
    assert result.expected_input_count == 1
    assert result.upstream_loaded_count == 1
    assert result.uv_lock_path is not None
    assert result.uv_lock_path.read_text() == "version = 1\n"
    assert result.raw_result_path.read_text() == json.dumps(raw_result)
    persisted_metrics = json.loads(result.metrics_path.read_text())
    assert persisted_metrics.keys() == result.metrics.keys()
    assert persisted_metrics["validity/overall_validity_count"] == 2213.0
    assert (
        persisted_metrics["validity/avg_charge_deviation"]
        != persisted_metrics["validity/avg_charge_deviation"]
    )
    assert result.stdout_path.read_text().startswith("Results saved to:")
    assert result.stderr_path.read_text() == "upstream log\n"
    assert result.provenance["reference_revisions_enforced"] is False
    assert result.provenance["runtime_environment"]["cuda_available"] is True
    assert result.provenance["runtime_environment_sha256"] == (
        lemat._sha256_file(result.runtime_environment_path)
    )
    assert result.provenance["artifact_sha256"]["manifest"] == (
        lemat._sha256_file(result.manifest_path)
    )
    assert result.provenance["artifact_sha256"]["lock"] == (
        lemat._sha256_file(result.uv_lock_path)
    )
    assert result.provenance["artifact_sha256"]["stdout"] == (
        lemat._sha256_file(result.stdout_path)
    )
    assert commands == [list(result.command)]
    assert result.command[-len(lemat.LEMAT_GENBENCH_TABLE2_FAMILIES) - 1] == (
        "--families"
    )

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["upstream_commit"] == lemat.LEMAT_GENBENCH_COMMIT
    assert manifest["upstream_uv_lock_sha256"] == "lock-digest"
    assert manifest["require_clean_repo"] is True
    assert manifest["expected_uv_lock_sha256"] == (lemat.LEMAT_GENBENCH_UV_LOCK_SHA256)
    assert manifest["input_count"] == {
        "expected_cifs": 1,
        "upstream_loaded_structures": 1,
        "dropped_during_cif_parsing": 0,
        "exact_match": True,
        "accepted_as_invalid_attempts": False,
        "error": None,
    }
    staged_cif = (output_dir / "input_cifs" / "000000.cif").resolve()
    staged_sha256 = lemat._sha256_file(staged_cif)
    assert manifest["input_cifs"] == [
        {
            "path": str(staged_cif),
            "source_path": str((cif_dir / "sample.cif").resolve()),
            "sha256": staged_sha256,
            "post_run_sha256": staged_sha256,
        }
    ]
    assert manifest["input_integrity_errors"] == {}
    assert manifest["cif_loading"]["staged_source_bytes_preserved"] is True
    assert manifest["cif_loading"]["normalizer_sha256"] == (
        lemat.hashlib.sha256(lemat._material_cif_source().encode()).hexdigest()
    )
    assert result.command[1] == "-c"
    assert result.command[2] == lemat._cif_loader_bootstrap_source(
        lemat._material_cif_source()
    )
    assert manifest["local_reference_integrity_errors"]
    cif_list_argument = result.command[result.command.index("--cifs") + 1]
    assert Path(cif_list_argument) == output_dir / "cif_paths.txt"
    assert (output_dir / "cif_paths.txt").read_text() == (f"{staged_cif}\n")


def test_run_lematgenbench_rejects_silently_dropped_cif(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_runtime_probe
):
    repo_path = tmp_path / "upstream"
    (repo_path / "scripts").mkdir(parents=True)
    (repo_path / "scripts" / "run_benchmarks.py").write_text("# stub\n")
    (repo_path / "pyproject.toml").write_text("[project]\nname='stub'\n")
    (repo_path / "uv.lock").write_text("version = 1\n")
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    (cif_dir / "a.cif").write_text("data_a\n")
    (cif_dir / "b.cif").write_text("not parseable upstream\n")
    output_dir = tmp_path / "artifacts"

    raw_result = _representative_raw_result()
    raw_result["run_info"]["n_structures"] = 1
    raw_result["validity_filtering"]["total_input_structures"] = 1
    raw_result["validity_filtering"]["valid_structures"] = 1
    raw_result["results"]["validity"] = raw_result["results"]["validity"].replace(
        "'total_structures': 2500", "'total_structures': 1"
    )
    raw_result["results"]["validity"] = raw_result["results"]["validity"].replace(
        "'overall_validity_count': 2213", "'overall_validity_count': 1"
    )
    monkeypatch.setattr(
        lemat,
        "_inspect_checkout",
        lambda *args, **kwargs: lemat._CheckoutInfo(
            commit=lemat.LEMAT_GENBENCH_COMMIT,
            lock_sha256="lock-digest",
            pyproject_sha256="project-digest",
        ),
    )

    def fake_run(command, **kwargs):
        result_dir = repo_path / "results_final"
        result_dir.mkdir(exist_ok=True)
        result_path = result_dir / (
            "dropped_comprehensive_multi_mlip_hull_20260904_120000.json"
        )
        result_path.write_text(json.dumps(raw_result))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"Results saved to: {result_path}\n",
            stderr="Failed to load b.cif\n",
        )

    monkeypatch.setattr(lemat.subprocess, "run", fake_run)

    with pytest.raises(
        lemat.LeMatGenBenchError,
        match="silently dropped CIFs while parsing: staged 2, loaded 1",
    ):
        lemat.run_lematgenbench(
            cif_dir,
            repo_path=repo_path,
            run_name="dropped",
            output_dir=output_dir,
            runner_command=("/fake/python",),
            verify_mace_checkpoint=False,
            enforce_reference_revisions=False,
        )

    manifest = json.loads((output_dir / "lematgenbench_manifest.json").read_text())
    assert manifest["input_count"] == {
        "expected_cifs": 2,
        "upstream_loaded_structures": 1,
        "dropped_during_cif_parsing": 1,
        "exact_match": False,
        "accepted_as_invalid_attempts": False,
        "error": "LeMat-GenBench silently dropped CIFs while parsing: staged 2, loaded 1",
    }
    assert (output_dir / "lematgenbench_raw.json").is_file()
    assert (output_dir / "lematgenbench.stderr.log").is_file()

    accepted = lemat.run_lematgenbench(
        cif_dir,
        repo_path=repo_path,
        run_name="accepted",
        output_dir=tmp_path / "accepted-artifacts",
        families=("validity",),
        runner_command=("/fake/python",),
        verify_mace_checkpoint=False,
        enforce_reference_revisions=False,
        parse_failures_as_invalid=True,
    )
    assert accepted.expected_input_count == 2
    assert accepted.upstream_loaded_count == 1
    assert accepted.parse_failure_count == 1
    assert accepted.metrics["input_accounting/cif_parse_failure_count"] == 1
    assert accepted.metrics["attempt_normalized/overall_valid_rate"] == 0.5
    accepted_manifest = json.loads(accepted.manifest_path.read_text())
    assert accepted_manifest["input_count"]["accepted_as_invalid_attempts"] is True


def test_checkout_rejects_lockfile_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run_benchmarks.py").write_text("# stub\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='stub'\n")
    (tmp_path / "uv.lock").write_text("modified\n")

    def fake_git(repo_path, *args):
        if args[0] == "rev-parse":
            return subprocess.CompletedProcess(
                args, 0, stdout=f"{lemat.LEMAT_GENBENCH_COMMIT}\n", stderr=""
            )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(lemat, "_run_git", fake_git)

    with pytest.raises(lemat.LeMatGenBenchError, match="uv.lock mismatch"):
        lemat._inspect_checkout(
            tmp_path,
            expected_commit=lemat.LEMAT_GENBENCH_COMMIT,
            expected_lock_sha256=lemat.LEMAT_GENBENCH_UV_LOCK_SHA256,
            require_clean_repo=True,
        )


def test_checkout_rejects_untracked_code_even_when_gitignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run_benchmarks.py").write_text("# tracked stub\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='stub'\n")

    def fake_git(repo_path, *args):
        if args[0] == "rev-parse":
            return subprocess.CompletedProcess(
                args, 0, stdout=f"{lemat.LEMAT_GENBENCH_COMMIT}\n", stderr=""
            )
        if args[0] == "status":
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if "--ignored" in args:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout="scripts/calculate_fid_after_relax.py\0",
                stderr="",
            )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(lemat, "_run_git", fake_git)
    with pytest.raises(lemat.LeMatGenBenchError, match="untracked files"):
        lemat._inspect_checkout(
            tmp_path,
            expected_commit=lemat.LEMAT_GENBENCH_COMMIT,
            expected_lock_sha256=None,
            require_clean_repo=True,
        )


@pytest.mark.parametrize("directory_backed_parquet", [False, True])
def test_enforced_run_rejects_unverified_local_hull_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    directory_backed_parquet: bool,
):
    repo_path = tmp_path / "upstream"
    (repo_path / "scripts").mkdir(parents=True)
    (repo_path / "scripts" / "run_benchmarks.py").write_text("# stub\n")
    (repo_path / "src" / "config").mkdir(parents=True)
    (repo_path / "src" / "config" / "comprehensive_multi_mlip_hull.yaml").write_text(
        "type: comprehensive_multi_mlip_hull\n"
    )
    (repo_path / "pyproject.toml").write_text("[project]\nname='stub'\n")
    (repo_path / "uv.lock").write_text("version = 1\n")
    local_hull = (
        repo_path
        / "data"
        / "convex_hulls"
        / "orb_conserv_inf_above_hull_dataset.parquet"
    )
    local_hull.parent.mkdir(parents=True)
    if directory_backed_parquet:
        local_hull.mkdir()
    else:
        local_hull.write_bytes(b"unverified local override")
    monkeypatch.setattr(
        lemat,
        "_inspect_checkout",
        lambda *args, **kwargs: lemat._CheckoutInfo(
            commit=lemat.LEMAT_GENBENCH_COMMIT,
            lock_sha256="lock-digest",
            pyproject_sha256="project-digest",
        ),
    )

    with pytest.raises(
        lemat.LeMatGenBenchError,
        match="cannot verify local hull overrides",
    ):
        lemat.run_lematgenbench(
            tmp_path / "unused-input",
            repo_path=repo_path,
            run_name="local-hull",
            output_dir=tmp_path / "artifacts",
            families=("stability",),
            runner_command=("/fake/python",),
            verify_mace_checkpoint=False,
        )

    assert not (tmp_path / "artifacts").exists()


def test_reference_pin_sitecustomize_forces_audited_revisions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    orb_checkpoint = tmp_path / "orb.ckpt"
    mace_checkpoint = tmp_path / "mace.model"
    uma_checkpoint = tmp_path / "uma.pt"
    hull_file = tmp_path / "hull.parquet"
    snapshot_dir = tmp_path / "uma-snapshot"
    orb_checkpoint.write_bytes(b"orb checkpoint")
    mace_checkpoint.write_bytes(b"mace checkpoint")
    uma_checkpoint.write_bytes(b"uma checkpoint")
    hull_file.write_bytes(b"hull")
    snapshot_dir.mkdir()
    expected_digests = {
        "mace": lemat._sha256_file(mace_checkpoint),
        "orb": lemat._sha256_file(orb_checkpoint),
        "uma": lemat._sha256_file(uma_checkpoint),
    }
    shim_dir = tmp_path / "shim"
    audit_path = tmp_path / "audit.jsonl"
    shim_path, shim_sha256 = lemat._write_reference_pin_shim(
        shim_dir,
        lemat.LEMAT_GENBENCH_REFERENCE_REVISIONS,
        expected_mace_checkpoint_sha256=expected_digests["mace"],
        mace_checkpoint_path=mace_checkpoint,
        expected_orb_checkpoint_sha256=expected_digests["orb"],
        expected_uma_checkpoint_sha256=expected_digests["uma"],
    )
    (shim_dir / "datasets.py").write_text(
        "def load_dataset(*args, **kwargs):\n    return kwargs\n"
    )
    (shim_dir / "huggingface_hub.py").write_text(
        "def hf_hub_download(*args, **kwargs):\n    return kwargs\n\n"
        "def snapshot_download(*args, **kwargs):\n    return kwargs\n"
    )
    (shim_dir / "cached_path.py").write_text(
        "def cached_path(*args, **kwargs):\n    return args[0]\n"
    )
    (shim_dir / "mace").mkdir()
    (shim_dir / "mace" / "__init__.py").write_text("")
    (shim_dir / "mace" / "calculators.py").write_text(
        "def mace_mp(*args, **kwargs):\n"
        "    return {'args': args, 'kwargs': kwargs}\n"
    )
    compile(shim_path.read_text(), str(shim_path), "exec")
    environment = lemat._reference_pin_environment(shim_dir, audit_path)
    assert environment["PYTHONPATH"] == str(shim_dir)

    probe_source = f"""
import json
import datasets
import huggingface_hub
import cached_path
import mace.calculators
import sitecustomize
sitecustomize._ORIGINAL_LOAD_DATASET = lambda *args, **kwargs: kwargs
def fake_hf_download(*args, **kwargs):
    repository = args[0] if args else kwargs.get("repo_id")
    if repository.casefold() == "facebook/uma":
        return {str(uma_checkpoint)!r}
    return {str(hull_file)!r}
sitecustomize._ORIGINAL_HF_HUB_DOWNLOAD = fake_hf_download
sitecustomize._ORIGINAL_SNAPSHOT_DOWNLOAD = lambda *args, **kwargs: {str(snapshot_dir)!r}
sitecustomize._ORIGINAL_CACHED_PATH = lambda *args, **kwargs: {str(orb_checkpoint)!r}
dataset = datasets.load_dataset("Lematerial/LeMat-Bulk", split="train")
hull = huggingface_hub.hf_hub_download(
    repo_id="LeMaterial/LeMat-Bulk-MLIP-Hull", filename="hull.parquet"
)
snapshot = huggingface_hub.snapshot_download(repo_id="facebook/UMA")
uma = huggingface_hub.hf_hub_download(
    repo_id="facebook/UMA", subfolder="checkpoints", filename="uma-s-1.pt"
)
orb = cached_path.cached_path({lemat.LEMAT_GENBENCH_ORB_CHECKPOINT_URL!r})
mace = mace.calculators.mace_mp()
try:
    datasets.load_dataset("LeMaterial/LeMat-Bulk", revision="wrong")
except RuntimeError as exc:
    conflict = str(exc)
print(json.dumps({{
    "dataset": dataset,
    "hull": hull,
    "snapshot": snapshot,
    "uma": uma,
    "orb": orb,
    "mace": mace,
    "conflict": conflict,
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe_source],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.splitlines()[-1])
    assert (
        payload["dataset"]["revision"]
        == (lemat.LEMAT_GENBENCH_REFERENCE_REVISIONS["LeMaterial/LeMat-Bulk"])
    )
    assert payload["hull"] == str(hull_file)
    assert payload["snapshot"] == str(snapshot_dir)
    assert payload["uma"] == str(uma_checkpoint)
    assert payload["orb"] == str(orb_checkpoint)
    assert payload["mace"]["kwargs"]["model"] == lemat.LEMAT_GENBENCH_MACE_MODEL
    assert "reference pin conflict" in payload["conflict"]
    records = lemat._read_reference_pin_audit(audit_path)
    assert records[0] == {
        "event": "loaded",
        "interpreter_role": None,
        "expected_oracle_digests": expected_digests,
        "patched_apis": list(lemat._REFERENCE_PIN_PATCHED_APIS),
        "revisions": lemat.LEMAT_GENBENCH_REFERENCE_REVISIONS,
    }
    pins = [record for record in records if record["event"] == "pin"]
    assert {
        (record["api"], record["repository"], record["enforced_revision"])
        for record in pins
    } >= {
        (
            "datasets.load_dataset",
            "LeMaterial/LeMat-Bulk",
            lemat.LEMAT_GENBENCH_REFERENCE_REVISIONS["LeMaterial/LeMat-Bulk"],
        ),
        (
            "huggingface_hub.hf_hub_download",
            "LeMaterial/LeMat-Bulk-MLIP-Hull",
            lemat.LEMAT_GENBENCH_REFERENCE_REVISIONS["LeMaterial/LeMat-Bulk-MLIP-Hull"],
        ),
        (
            "huggingface_hub.snapshot_download",
            "facebook/UMA",
            lemat.LEMAT_GENBENCH_UMA_REVISION,
        ),
    }
    artifacts = {
        record["oracle"]: record for record in records if record["event"] == "artifact"
    }
    assert artifacts["orb"]["sha256"] == expected_digests["orb"]
    assert artifacts["orb"]["matches_expected"] is True
    assert artifacts["mace"]["sha256"] == expected_digests["mace"]
    assert artifacts["mace"]["matches_expected"] is True
    assert artifacts["uma"]["sha256"] == expected_digests["uma"]
    assert artifacts["uma"]["matches_expected"] is True
    assert shim_sha256 == lemat._sha256_file(shim_path)


def test_runtime_environment_sanitizes_pythonpath_and_freezes_device(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("PYTHONPATH", "/uncontrolled/imports")
    monkeypatch.setenv("PYTHONHOME", "/uncontrolled/python")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-scheduler-assignment")

    cuda_environment = lemat._runtime_environment("cuda", cuda_visible_devices=None)
    assert "PYTHONPATH" not in cuda_environment
    assert "PYTHONHOME" not in cuda_environment
    assert cuda_environment["CUDA_VISIBLE_DEVICES"] == "GPU-scheduler-assignment"
    assert cuda_environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"

    cpu_environment = lemat._runtime_environment("cpu", cuda_visible_devices=None)
    assert cpu_environment["CUDA_VISIBLE_DEVICES"] == ""
    with pytest.raises(ValueError, match="cannot be set"):
        lemat._runtime_environment("cpu", cuda_visible_devices="0")


def test_runtime_probe_archives_torch_cuda_and_gpu_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    payload = {
        "requested_device_policy": "cuda",
        "requested_cuda_visible_devices": "0",
        "effective_cuda_visible_devices": "0",
        "cuda_device_order": "PCI_BUS_ID",
        "python_executable": "/pinned/.venv/bin/python",
        "python_version": "3.11.test",
        "torch_version": "2.test",
        "torch_cuda_build": "12.test",
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_current_device": 0,
        "cuda_devices": [{"index": 0, "name": "test GPU"}],
    }

    def fake_run(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "torch startup log\n"
                f"{lemat._RUNTIME_PROBE_RESULT_PREFIX}{json.dumps(payload)}\n"
            ),
            stderr="driver log\n",
        )

    monkeypatch.setattr(lemat.subprocess, "run", fake_run)
    output_path = tmp_path / "runtime.json"
    actual, command, digest = lemat._probe_runtime_environment(
        repo_path=tmp_path,
        command_prefix=("/pinned/python",),
        environment={"CUDA_VISIBLE_DEVICES": "0"},
        expected_device="cuda",
        requested_cuda_visible_devices="0",
        output_path=output_path,
        stdout_path=tmp_path / "runtime.stdout.log",
        stderr_path=tmp_path / "runtime.stderr.log",
        timeout_seconds=30,
    )

    assert actual == payload
    assert command[:2] == ("/pinned/python", "-c")
    assert json.loads(output_path.read_text()) == payload
    assert digest == lemat._sha256_file(output_path)


def test_reference_pin_audit_requires_conditional_pins_and_oracle_artifacts():
    inventory = {
        f"data/convex_hulls/{hull_type}_{suffix}": {"exists": False}
        for hull_type in lemat._CANONICAL_HULL_TYPES
        for suffix in (
            "above_hull_dataset.parquet",
            "above_hull_composition_matrix.npz",
        )
    }
    access_plan = lemat._canonical_reference_access_plan(
        config_name=lemat.LEMAT_GENBENCH_CONFIG,
        families=lemat.LEMAT_GENBENCH_TABLE2_FAMILIES,
        valid_structure_count=1,
        local_inventory=inventory,
    )
    assert set(access_plan["expected_runtime_repositories"]) == {
        "LeMaterial/LeMat-Bulk",
        "LeMaterial/LeMat-Bulk-MLIP-Hull",
        "facebook/UMA",
    }
    assert (
        "LeMaterial/LeMat-GenBench-embeddings"
        in (access_plan["not_expected_at_runtime"])
    )

    expected_digests = {
        "mace": lemat.LEMAT_GENBENCH_MACE_CHECKPOINT_SHA256,
        "orb": lemat.LEMAT_GENBENCH_ORB_CHECKPOINT_SHA256,
        "uma": lemat.LEMAT_GENBENCH_UMA_CHECKPOINT_SHA256,
    }
    loaded = {
        "event": "loaded",
        "revisions": lemat.LEMAT_GENBENCH_REFERENCE_REVISIONS,
        "expected_oracle_digests": expected_digests,
        "patched_apis": list(lemat._REFERENCE_PIN_PATCHED_APIS),
    }
    records = [
        {**loaded, "interpreter_role": "reference-pin-preflight"},
        {**loaded, "interpreter_role": "benchmark"},
    ]
    summary, error = lemat._reference_pin_audit_summary(
        records,
        access_plan=access_plan,
        require_oracle_artifacts=True,
        expected_oracle_digests=expected_digests,
    )
    assert error is not None
    assert summary["missing_expected_repositories"] == [
        "LeMaterial/LeMat-Bulk",
        "LeMaterial/LeMat-Bulk-MLIP-Hull",
        "facebook/UMA",
    ]
    assert summary["missing_oracle_artifacts"] == ["mace", "orb", "uma"]

    for repository in access_plan["expected_runtime_repositories"]:
        records.append(
            {
                "event": "pin",
                "repository": repository,
                "enforced_revision": lemat.LEMAT_GENBENCH_REFERENCE_REVISIONS[
                    repository
                ],
            }
        )
    records.extend(
        {
            "event": "artifact",
            "oracle": oracle,
            "matches_expected": True,
            "sha256": digest,
        }
        for oracle, digest in expected_digests.items()
    )
    records.append(
        {
            "event": "oracle_model",
            "oracle": "mace",
            "enforced_model": lemat.LEMAT_GENBENCH_MACE_MODEL,
        }
    )
    summary, error = lemat._reference_pin_audit_summary(
        records,
        access_plan=access_plan,
        require_oracle_artifacts=True,
        expected_oracle_digests=expected_digests,
    )
    assert error is None
    assert summary["missing_expected_repositories"] == []
    assert summary["missing_oracle_artifacts"] == []


def test_oracle_coverage_uses_all_three_public_stability_counts():
    metrics = lemat.extract_lematgenbench_metrics(_representative_raw_result())

    summary, coverage_metrics = lemat._oracle_coverage_summary(
        metrics,
        config_name=lemat.LEMAT_GENBENCH_CONFIG,
        families=lemat.LEMAT_GENBENCH_TABLE2_FAMILIES,
    )

    assert summary["status"] == "complete"
    assert summary["all_three_complete"] is True
    assert summary["per_mlip_e_above_hull_counts"] == {
        "orb": 2213,
        "mace": 2213,
        "uma": 2213,
    }
    assert coverage_metrics["oracle_coverage/all_three_complete"] == 1.0

    metrics["stability/n_valid_structures_uma"] = 2200
    summary, coverage_metrics = lemat._oracle_coverage_summary(
        metrics,
        config_name=lemat.LEMAT_GENBENCH_CONFIG,
        families=lemat.LEMAT_GENBENCH_TABLE2_FAMILIES,
    )
    assert summary["status"] == "incomplete"
    assert summary["all_three_complete"] is False
    assert coverage_metrics["oracle_coverage/all_three_complete"] == 0.0


def test_unknown_valid_count_cannot_bypass_stability_coverage():
    metrics = {
        "validity/total_structures": 1.0,
        "stability/n_valid_structures_orb": 1.0,
        "stability/n_valid_structures_mace": 1.0,
        "stability/n_valid_structures_uma": 1.0,
    }
    summary, _ = lemat._oracle_coverage_summary(
        metrics,
        config_name=lemat.LEMAT_GENBENCH_CONFIG,
        families=("stability",),
    )
    assert summary["valid_structures"] is None
    assert summary["status"] == "coverage_counts_missing"

    access_plan = lemat._canonical_reference_access_plan(
        config_name=lemat.LEMAT_GENBENCH_CONFIG,
        families=("stability",),
        valid_structure_count=None,
        local_inventory={},
    )
    assert set(access_plan["expected_runtime_repositories"]) == {
        "LeMaterial/LeMat-Bulk-MLIP-Hull",
        "facebook/UMA",
    }


def test_run_injects_and_records_reference_pin_shim_without_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_runtime_probe
):
    repo_path = tmp_path / "upstream"
    (repo_path / "scripts").mkdir(parents=True)
    (repo_path / "scripts" / "run_benchmarks.py").write_text("# stub\n")
    (repo_path / "pyproject.toml").write_text("[project]\nname='stub'\n")
    (repo_path / "uv.lock").write_text("version = 1\n")
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    (cif_dir / "sample.cif").write_text("data_sample\n")
    raw_result = _representative_raw_result()
    raw_result["run_info"]["n_structures"] = 1
    raw_result["validity_filtering"]["total_input_structures"] = 1
    raw_result["results"]["validity"] = raw_result["results"]["validity"].replace(
        "'total_structures': 2500", "'total_structures': 1"
    )
    monkeypatch.setattr(
        lemat,
        "_inspect_checkout",
        lambda *args, **kwargs: lemat._CheckoutInfo(
            commit=lemat.LEMAT_GENBENCH_COMMIT,
            lock_sha256="lock-digest",
            pyproject_sha256="project-digest",
        ),
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        assert isinstance(command, list)
        assert "shell" not in kwargs
        environment = kwargs["env"]
        shim_dir = Path(environment["PYTHONPATH"].split(lemat.os.pathsep)[0])
        shim_path = shim_dir / "sitecustomize.py"
        audit_path = Path(environment[lemat._REFERENCE_PIN_AUDIT_ENV])
        with audit_path.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "loaded",
                        "interpreter_role": environment[lemat._REFERENCE_PIN_ROLE_ENV],
                        "expected_oracle_digests": {
                            "mace": None,
                            "orb": lemat.LEMAT_GENBENCH_ORB_CHECKPOINT_SHA256,
                            "uma": lemat.LEMAT_GENBENCH_UMA_CHECKPOINT_SHA256,
                        },
                        "patched_apis": list(lemat._REFERENCE_PIN_PATCHED_APIS),
                        "revisions": lemat.LEMAT_GENBENCH_REFERENCE_REVISIONS,
                    }
                )
                + "\n"
            )
        if command[-2] == "-c":
            payload = {
                "path": str(shim_path),
                "revisions": lemat.LEMAT_GENBENCH_REFERENCE_REVISIONS,
                "expected_oracle_digests": {
                    "mace": None,
                    "orb": lemat.LEMAT_GENBENCH_ORB_CHECKPOINT_SHA256,
                    "uma": lemat.LEMAT_GENBENCH_UMA_CHECKPOINT_SHA256,
                },
                "patched_apis": list(lemat._REFERENCE_PIN_PATCHED_APIS),
            }
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=f"ZEVALS_REFERENCE_PIN_SHIM={json.dumps(payload)}\n",
                stderr="",
            )

        with audit_path.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "pin",
                        "api": "datasets.load_dataset",
                        "repository": "LeMaterial/LeMat-Bulk",
                        "enforced_revision": lemat.LEMAT_GENBENCH_REFERENCE_REVISIONS[
                            "LeMaterial/LeMat-Bulk"
                        ],
                        "requested_revision": None,
                    }
                )
                + "\n"
            )
        result_dir = repo_path / "results_final"
        result_dir.mkdir()
        result_path = result_dir / (
            "pinned_comprehensive_multi_mlip_hull_20260904_120000.json"
        )
        result_path.write_text(json.dumps(raw_result))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"Results saved to: {result_path}\n",
            stderr="",
        )

    monkeypatch.setattr(lemat.subprocess, "run", fake_run)
    result = lemat.run_lematgenbench(
        cif_dir,
        repo_path=repo_path,
        run_name="pinned",
        output_dir=tmp_path / "artifacts",
        families=("validity",),
        runner_command=("/fake/python",),
        verify_mace_checkpoint=False,
    )

    assert len(calls) == 2
    assert result.reference_pin_shim_path is not None
    assert result.reference_pin_shim_sha256 == lemat._sha256_file(
        result.reference_pin_shim_path
    )
    assert result.reference_pin_audit_path is not None
    manifest = json.loads(result.manifest_path.read_text())
    pins = manifest["reference_revision_pins"]
    assert pins["enforced"] is True
    assert pins["revisions"] == lemat.LEMAT_GENBENCH_REFERENCE_REVISIONS
    assert pins["shim_sha256"] == result.reference_pin_shim_sha256
    assert pins["error"] is None
    assert pins["patched_apis"] == list(lemat._REFERENCE_PIN_PATCHED_APIS)
    assert pins["expected_oracle_digests"] == {
        "mace": None,
        "orb": lemat.LEMAT_GENBENCH_ORB_CHECKPOINT_SHA256,
        "uma": lemat.LEMAT_GENBENCH_UMA_CHECKPOINT_SHA256,
    }
    assert pins["runtime_access"]["expected_runtime_repositories"] == {}
    assert (
        "LeMaterial/LeMat-GenBench-embeddings"
        in (pins["runtime_access"]["not_expected_at_runtime"])
    )
    assert [record["event"] for record in pins["audit"]] == [
        "loaded",
        "loaded",
        "pin",
    ]


def test_mace_preflight_verifies_explicit_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    checkpoint = tmp_path / "mace.model"
    checkpoint.write_bytes(b"pinned checkpoint")
    checkpoint_sha256 = lemat._sha256_file(checkpoint)

    def fake_run(command, **kwargs):
        payload = {"path": str(checkpoint), "sha256": checkpoint_sha256}
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"download log\nZEVALS_MACE_CHECKPOINT={json.dumps(payload)}\n",
            stderr="",
        )

    monkeypatch.setattr(lemat.subprocess, "run", fake_run)
    result = lemat._probe_mace_checkpoint(
        repo_path=tmp_path,
        command_prefix=("python",),
        model=lemat.LEMAT_GENBENCH_MACE_MODEL,
        expected_sha256=checkpoint_sha256,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        timeout_seconds=30,
    )

    assert result.model == "medium-mpa-0"
    assert result.path == checkpoint
    assert result.sha256 == checkpoint_sha256
    assert result.command[:2] == ("python", "-c")
    compile(result.command[-1], "<mace-checkpoint-probe>", "exec")

    with pytest.raises(lemat.LeMatGenBenchError, match="digest mismatch"):
        lemat._probe_mace_checkpoint(
            repo_path=tmp_path,
            command_prefix=("python",),
            model=lemat.LEMAT_GENBENCH_MACE_MODEL,
            expected_sha256="0" * 64,
            stdout_path=tmp_path / "mismatch.stdout.log",
            stderr_path=tmp_path / "mismatch.stderr.log",
            timeout_seconds=30,
        )


def test_run_lematgenbench_rejects_mace_small_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_runtime_probe
):
    repo_path = tmp_path / "upstream"
    (repo_path / "scripts").mkdir(parents=True)
    (repo_path / "scripts" / "run_benchmarks.py").write_text("# stub\n")
    (repo_path / "pyproject.toml").write_text("[project]\nname='stub'\n")
    (repo_path / "uv.lock").write_text("version = 1\n")
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    (cif_dir / "sample.cif").write_text("data_sample\n")
    output_dir = tmp_path / "artifacts"
    raw_result = _representative_raw_result()
    raw_result["run_info"]["n_structures"] = 1
    raw_result["validity_filtering"]["total_input_structures"] = 1
    raw_result["results"]["validity"] = raw_result["results"]["validity"].replace(
        "'total_structures': 2500", "'total_structures': 1"
    )
    monkeypatch.setattr(
        lemat,
        "_inspect_checkout",
        lambda *args, **kwargs: lemat._CheckoutInfo(
            commit=lemat.LEMAT_GENBENCH_COMMIT,
            lock_sha256="lock-digest",
            pyproject_sha256="project-digest",
        ),
    )

    def fake_run(command, **kwargs):
        result_dir = repo_path / "results_final"
        result_dir.mkdir()
        result_path = result_dir / (
            "fallback_comprehensive_multi_mlip_hull_20260904_120000.json"
        )
        result_path.write_text(json.dumps(raw_result))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "Successfully loaded MACE model using alternative approach\n"
                f"Results saved to: {result_path}\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(lemat.subprocess, "run", fake_run)

    with pytest.raises(lemat.LeMatGenBenchError, match="alternative small model"):
        lemat.run_lematgenbench(
            cif_dir,
            repo_path=repo_path,
            run_name="fallback",
            output_dir=output_dir,
            families=("stability",),
            runner_command=("/fake/python",),
            verify_mace_checkpoint=False,
            enforce_reference_revisions=False,
        )

    manifest = json.loads((output_dir / "lematgenbench_manifest.json").read_text())
    assert manifest["mace_checkpoint"]["fallback_to_small_detected"] is True


@pytest.mark.parametrize("resume_enabled", [False, True])
def test_material_pipeline_merges_lemat_metrics_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
    resume_enabled: bool,
):
    from zevals.pipeline import materials as pipeline

    monkeypatch.setattr(
        pipeline,
        "crystal_from_structure",
        lambda structure, idx: SimpleNamespace(valid=True, structure=structure),
    )
    monkeypatch.setattr(
        pipeline,
        "compute_validity",
        lambda crystals: {"valid_rate": 1.0},
    )
    monkeypatch.setattr(pipeline, "compute_diversity", lambda structures: {})
    monkeypatch.setattr(
        pipeline,
        "compute_uniqueness",
        lambda structures, return_unique_structures: {
            "unique_rate": 1.0,
            "unique_structures": structures,
        },
    )
    calls = {}

    def fake_lemat(structures, **kwargs):
        calls["structures"] = structures
        calls["kwargs"] = kwargs
        return SimpleNamespace(
            metrics={"validity/overall_validity_ratio": 0.9},
            provenance={"upstream_commit": lemat.LEMAT_GENBENCH_COMMIT},
        )

    monkeypatch.setattr(pipeline, "run_lematgenbench", fake_lemat)

    results = pipeline.MaterialEvalPipeline(
        run_lematgenbench=True,
        lematgenbench_repo_path="/tmp/lemat",
        lematgenbench_output_dir="/tmp/results",
        lematgenbench_run_name="checkpoint-100",
        lematgenbench_families=("validity",),
        lematgenbench_resume=resume_enabled,
    ).run(["structure"])

    assert results.aggregate["validity/overall_validity_ratio"] == 0.9
    assert results.metadata["lematgenbench"] == {
        "upstream_commit": lemat.LEMAT_GENBENCH_COMMIT
    }
    assert calls["structures"] == ["structure"]
    assert calls["kwargs"]["run_name"] == "checkpoint-100"
    assert calls["kwargs"]["families"] == ("validity",)
    assert calls["kwargs"]["fail_on_family_error"] is True
    assert calls["kwargs"]["resume"] is resume_enabled
    assert calls["kwargs"]["expected_lock_sha256"] == (
        lemat.LEMAT_GENBENCH_UV_LOCK_SHA256
    )
    assert calls["kwargs"]["verify_mace_checkpoint"] is True
    assert calls["kwargs"]["expected_mace_checkpoint_sha256"] == (
        lemat.LEMAT_GENBENCH_MACE_CHECKPOINT_SHA256
    )
    assert calls["kwargs"]["verify_orb_checkpoint"] is True
    assert calls["kwargs"]["expected_orb_checkpoint_sha256"] == (
        lemat.LEMAT_GENBENCH_ORB_CHECKPOINT_SHA256
    )
    assert calls["kwargs"]["verify_uma_checkpoint"] is True
    assert calls["kwargs"]["expected_uma_checkpoint_sha256"] == (
        lemat.LEMAT_GENBENCH_UMA_CHECKPOINT_SHA256
    )
    assert calls["kwargs"]["enforce_reference_revisions"] is True
    assert calls["kwargs"]["runtime_device"] == "cuda"
    assert calls["kwargs"]["cuda_visible_devices"] is None
    assert calls["kwargs"]["parse_failures_as_invalid"] is False
    assert calls["kwargs"]["require_complete_oracle_coverage"] is True


def test_material_pipeline_propagates_lemat_protocol_errors_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    from zevals.pipeline import materials as pipeline

    monkeypatch.setattr(
        pipeline,
        "crystal_from_structure",
        lambda structure, idx: SimpleNamespace(valid=True, structure=structure),
    )
    monkeypatch.setattr(pipeline, "compute_validity", lambda crystals: {})
    monkeypatch.setattr(pipeline, "compute_diversity", lambda structures: {})
    monkeypatch.setattr(
        pipeline,
        "compute_uniqueness",
        lambda structures, return_unique_structures: {
            "unique_rate": 1.0,
            "unique_structures": structures,
        },
    )
    monkeypatch.setattr(
        pipeline,
        "run_lematgenbench",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            lemat.LeMatGenBenchError("protocol drift")
        ),
    )

    evaluator = pipeline.MaterialEvalPipeline(
        run_lematgenbench=True,
        lematgenbench_repo_path="/tmp/lemat",
    )
    with pytest.raises(lemat.LeMatGenBenchError, match="protocol drift"):
        evaluator.run(["structure"])
