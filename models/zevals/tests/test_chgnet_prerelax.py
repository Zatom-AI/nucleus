"""Scientific attempt accounting and interruption/resume contract tests."""

from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from zevals.materials import chgnet_prerelax as chgnet

_REAL_RUNTIME = chgnet._CHGNetRuntime


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    checkpoint = tmp_path / "fake-checkpoint"
    checkpoint.write_bytes(b"fixed weights")
    monkeypatch.setattr(chgnet, "CHGNET_CHECKPOINT_SHA256", chgnet._sha256(checkpoint))

    def source_text(path):
        content = path.read_bytes()
        if path.name.endswith(".gz"):
            content = gzip.decompress(content)
        return content.decode()

    class Structure:
        @staticmethod
        def from_file(path):
            if source_text(path) == "parse-error":
                raise ValueError("invalid CIF")
            return object()

    class Runtime:
        calls = []
        interrupt_on = None
        runtime_version = "fixed"

        def __init__(self, configuration):
            self.Structure = Structure
            self.provenance = {
                "runtime_version": self.runtime_version,
                "packages": {
                    "chgnet": chgnet.CHGNET_VERSION,
                    "ase": chgnet.CHGNET_ASE_VERSION,
                },
                "actual_device": configuration["device"],
                "deterministic_algorithms": True,
                "cublas_workspace_config": ":4096:8",
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": chgnet._sha256(checkpoint),
                    "wheel_record_sha256": chgnet._sha256(checkpoint),
                    "model": chgnet.CHGNET_MODEL,
                },
            }

        def read_structure(self, source):
            return self.Structure.from_file(source)

        def relax(self, source):
            self.calls.append(source.name)
            if self.interrupt_on == source.name:
                raise chgnet._Interrupted()
            if source_text(source) == "relax-error":
                raise ValueError("unphysical geometry")
            if source_text(source) == "cuda-error":
                raise RuntimeError("CUDA out of memory")
            return ("relaxed " + source_text(source)).encode(), {
                "converged": source_text(source) != "unconverged",
                "optimizer_steps": 1500 if source_text(source) == "unconverged" else 3,
                "initial_energy_eV": -1.0,
                "final_energy_eV": -2.0,
            }

    monkeypatch.setattr(chgnet, "_CHGNetRuntime", Runtime)
    return Runtime


def _request(tmp_path, contents=("valid",)):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for index, content in enumerate(contents):
        (source_dir / f"mp20-{index:06d}.cif").write_text(content)
    configuration = chgnet._configuration(
        fmax=0.1, steps=1500, relax_cell=True, device="cpu", seed=42
    )
    return {
        "contract": chgnet._contract(source_dir, configuration),
        "output_dir": str(tmp_path / "output"),
        "resume": True,
    }


def _validate(request, **kwargs):
    return chgnet.validate_chgnet_prerelaxation(
        request["contract"]["input_dir"],
        output_dir=request["output_dir"],
        device="cpu",
        **kwargs,
    )


def test_relaxation_preserves_all_attempts_and_originals(tmp_path, fake_runtime):
    request = _request(tmp_path, ("valid", "parse-error", "relax-error", "unconverged"))
    assert chgnet._run_worker(request)
    result = _validate(request)
    assert result.complete
    assert result.manifest["attempted_count"] == 4
    assert result.manifest["status_counts"] == {
        "relaxed": 2,
        "fallback_parse_error": 1,
        "fallback_relaxation_error": 1,
    }
    assert result.manifest["converged_count"] == 1
    assert len(list(result.cif_dir.glob("*.cif"))) == 4
    assert (result.cif_dir / "mp20-000001.cif").read_text() == "parse-error"
    assert (result.cif_dir / "mp20-000002.cif").read_text() == "relax-error"
    assert (result.cif_dir / "mp20-000003.cif").read_text() == "relaxed unconverged"
    assert (tmp_path / "source/mp20-000000.cif").read_text() == "valid"
    assert result.manifest["records"][3]["optimizer_steps"] == 1500


def test_resume_skips_completed_after_interruption(tmp_path, fake_runtime):
    request = _request(tmp_path, ("valid", "valid", "valid"))
    fake_runtime.interrupt_on = "mp20-000001.cif"
    with pytest.raises(chgnet._Interrupted):
        chgnet._run_worker(request)
    incomplete = _validate(request, require_complete=False)
    assert not incomplete.complete
    assert incomplete.manifest["completed_count"] == 1
    with pytest.raises(ValueError, match="incomplete"):
        _validate(request)
    fake_runtime.interrupt_on = None
    assert chgnet._run_worker(request)
    assert _validate(request).complete
    assert fake_runtime.calls == [
        "mp20-000000.cif",
        "mp20-000001.cif",
        "mp20-000001.cif",
        "mp20-000002.cif",
    ]


def test_orphan_output_without_record_is_recomputed(
    tmp_path, fake_runtime, monkeypatch
):
    request = _request(tmp_path)
    original_write = chgnet._atomic_json

    def interrupt_record(path, value):
        if path.parent.name == "records":
            raise chgnet._Interrupted()
        original_write(path, value)

    monkeypatch.setattr(chgnet, "_atomic_json", interrupt_record)
    with pytest.raises(chgnet._Interrupted):
        chgnet._run_worker(request)
    assert (Path(request["output_dir"]) / "cifs/mp20-000000.cif").exists()
    assert _validate(request, require_complete=False).manifest["completed_count"] == 0
    monkeypatch.setattr(chgnet, "_atomic_json", original_write)
    chgnet._run_worker(request)
    assert fake_runtime.calls == ["mp20-000000.cif", "mp20-000000.cif"]


def test_completed_output_reuse_does_not_start_worker(
    tmp_path, fake_runtime, monkeypatch
):
    request = _request(tmp_path)
    chgnet._run_worker(request)

    def forbidden(*args, **kwargs):
        pytest.fail("Completed outputs must be read-only reusable")

    monkeypatch.setattr(subprocess, "run", forbidden)
    result = chgnet.run_chgnet_prerelaxation(
        request["contract"]["input_dir"],
        output_dir=request["output_dir"],
        device="cpu",
    )
    assert result.complete


@pytest.mark.parametrize(
    "mutation", ["input", "output", "record", "configuration", "worker", "converter"]
)
def test_reuse_rejects_changed_artifacts(tmp_path, fake_runtime, monkeypatch, mutation):
    request = _request(tmp_path)
    chgnet._run_worker(request)
    if mutation == "input":
        (tmp_path / "source/mp20-000000.cif").write_text("changed")
    elif mutation == "output":
        (tmp_path / "output/cifs/mp20-000000.cif").write_text("changed")
    elif mutation == "record":
        path = tmp_path / "output/records/mp20-000000.cif.json"
        record = json.loads(path.read_text())
        record["final_energy_eV"] = -100.0
        path.write_text(json.dumps(record))
    elif mutation == "configuration":
        with pytest.raises(ValueError, match="configuration"):
            _validate(request, steps=100)
        return
    else:
        original = chgnet._sha256
        changed_path = Path(chgnet.__file__)
        if mutation == "converter":
            changed_path = changed_path.with_name("_material_cif.py")
        monkeypatch.setattr(
            chgnet,
            "_sha256",
            lambda path: "changed" if path == changed_path else original(path),
        )
    with pytest.raises(ValueError):
        _validate(request)


def test_resume_rejects_changed_numerical_environment(tmp_path, fake_runtime):
    request = _request(tmp_path, ("valid", "valid"))
    fake_runtime.interrupt_on = "mp20-000001.cif"
    with pytest.raises(chgnet._Interrupted):
        chgnet._run_worker(request)
    fake_runtime.interrupt_on = None
    fake_runtime.runtime_version = "changed"
    with pytest.raises(ValueError, match="environment"):
        chgnet._run_worker(request)
    assert fake_runtime.calls == ["mp20-000000.cif", "mp20-000001.cif"]


def test_cuda_error_is_resumable_not_recorded_as_failed_material(
    tmp_path, fake_runtime
):
    request = _request(tmp_path, ("valid", "cuda-error"))
    with pytest.raises(RuntimeError, match="CUDA"):
        chgnet._run_worker(request)
    result = _validate(request, require_complete=False)
    assert not result.complete
    assert result.manifest["completed_count"] == 1
    assert not (tmp_path / "output/records/mp20-000001.cif.json").exists()


def test_timeout_returns_verified_partial_progress(tmp_path, fake_runtime, monkeypatch):
    request = _request(tmp_path, ("valid", "valid"))

    def timeout(command, **kwargs):
        assert "--request" in command
        assert len(command[-1]) < 4096
        worker_request = json.loads(Path(command[-1]).read_text())
        fake_runtime.interrupt_on = "mp20-000001.cif"
        with pytest.raises(chgnet._Interrupted):
            chgnet._run_worker(worker_request)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    result = chgnet.run_chgnet_prerelaxation(
        request["contract"]["input_dir"],
        output_dir=request["output_dir"],
        device="cpu",
        timeout_seconds=2,
    )
    assert not result.complete
    assert result.manifest["completed_count"] == 1
    assert not list((tmp_path / "output").glob(".request-*"))


def test_configuration_and_path_validation(tmp_path):
    request = _request(tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        chgnet.run_chgnet_prerelaxation(
            tmp_path / "source",
            output_dir=tmp_path / "source/relaxed",
        )
    for invalid in (float("nan"), 0.0, -1.0):
        with pytest.raises(ValueError, match="fmax"):
            chgnet.validate_chgnet_prerelaxation(
                request["contract"]["input_dir"],
                output_dir=request["output_dir"],
                fmax=invalid,
            )


def test_gzip_filenames_and_fallback_bytes_are_preserved(tmp_path, fake_runtime):
    request = _request(tmp_path)
    source_dir = Path(request["contract"]["input_dir"])
    for index, content in enumerate((b"valid", b"parse-error"), start=1):
        (source_dir / f"mp20-{index:06d}.cif.gz").write_bytes(
            gzip.compress(content, mtime=123)
        )
    request["contract"] = chgnet._contract(
        source_dir, request["contract"]["configuration"]
    )
    chgnet._run_worker(request)
    result = _validate(request)
    assert result.complete
    assert sorted(path.name for path in result.cif_dir.iterdir()) == [
        "mp20-000000.cif",
        "mp20-000001.cif.gz",
        "mp20-000002.cif.gz",
    ]
    assert (
        gzip.decompress((result.cif_dir / "mp20-000001.cif.gz").read_bytes())
        == b"relaxed valid"
    )
    assert (result.cif_dir / "mp20-000001.cif.gz").read_bytes()[4:8] == bytes(4)
    assert (result.cif_dir / "mp20-000002.cif.gz").read_bytes() == (
        source_dir / "mp20-000002.cif.gz"
    ).read_bytes()
    (result.cif_dir / "unexpected.cif.gz").write_bytes(gzip.compress(b"extra"))
    with pytest.raises(ValueError, match="Unexpected CIFs"):
        _validate(request)


def test_corrupt_gzip_is_a_failed_parse_attempt(tmp_path, fake_runtime):
    request = _request(tmp_path)
    source_dir = Path(request["contract"]["input_dir"])
    (source_dir / "mp20-000001.cif.gz").write_bytes(b"not a gzip stream")
    request["contract"] = chgnet._contract(
        source_dir, request["contract"]["configuration"]
    )
    chgnet._run_worker(request)
    result = _validate(request)
    assert result.manifest["status_counts"]["fallback_parse_error"] == 1
    assert (result.cif_dir / "mp20-000001.cif.gz").read_bytes() == b"not a gzip stream"


@pytest.mark.parametrize("compressed", [False, True])
def test_generated_mmcif_reaches_relaxation_without_changing_raw_bytes(
    tmp_path, fake_runtime, monkeypatch, compressed
):
    structure_class = pytest.importorskip("pymatgen.core").Structure
    mmcif = b"""data_generated
_cell.length_a 4.0
_cell.length_b 5.0
_cell.length_c 6.0
_cell.angle_alpha 90
_cell.angle_beta 90
_cell.angle_gamma 90
_space_group.name_H-M_alt 'P 1'
_space_group.IT_number 1
loop_
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.occupancy
_atom_site.fract_x
_atom_site.fract_y
_atom_site.fract_z
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
Na Na1 1.0 0.25 0.50 0.75 99.0 99.0 99.0
Cl Cl2 1.0 0.75 0.50 0.25 99.0 99.0 99.0
"""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    name = "mp20-000000.cif.gz" if compressed else "mp20-000000.cif"
    source = source_dir / name
    original = gzip.compress(mmcif, mtime=123) if compressed else mmcif
    source.write_bytes(original)
    request = {
        "contract": chgnet._contract(
            source_dir,
            chgnet._configuration(
                fmax=0.1, steps=1500, relax_cell=True, device="cpu", seed=42
            ),
        ),
        "output_dir": str(tmp_path / "output"),
        "resume": True,
    }

    def read_structure(self, path):
        return _REAL_RUNTIME.read_structure(
            SimpleNamespace(Structure=structure_class), path
        )

    def relax(self, path):
        # Exercise the real parser, but no model initialization or oracle call.
        structure = self.read_structure(path)
        assert structure.lattice.abc == pytest.approx((4.0, 5.0, 6.0))
        sites = {site.specie.symbol: site.frac_coords.tolist() for site in structure}
        assert sites["Na"] == pytest.approx([0.25, 0.50, 0.75], abs=1e-12)
        assert sites["Cl"] == pytest.approx([0.75, 0.50, 0.25], abs=1e-12)
        self.calls.append(path.name)
        return structure.to(fmt="cif").encode(), {"converged": True}

    monkeypatch.setattr(fake_runtime, "read_structure", read_structure)
    monkeypatch.setattr(fake_runtime, "relax", relax)
    assert chgnet._run_worker(request)
    result = _validate(request)
    assert result.manifest["status_counts"]["relaxed"] == 1
    assert result.manifest["status_counts"]["fallback_parse_error"] == 0
    assert fake_runtime.calls == [name]
    assert source.read_bytes() == original
    assert len(structure_class.from_file(result.cif_dir / name)) == 2
