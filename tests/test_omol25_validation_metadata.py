"""Regression tests for committed OMol25 validation electronic states."""

import hashlib
import json
import math
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "models" / "zatom2" / "docs" / "examples"

# These manifests were derived from their configured canonical ASE-LMDB sources
# using each entry's exact ``example_id`` or ``source_row_index``. Keeping the
# source-derived digest here catches metadata drift without requiring the private
# OMol25 datasets in a fresh clone.
METADATA_MANIFESTS = {
    "omol25_val_structure_prediction_500.json": {
        "count": 500,
        "digest": "cbeac38ccb32560f6e7e5c41b53ea3d7575866207e95544d3ba9cddf8bfa11bc",
        "nonzero_charge_count": 108,
        "non_singlet_count": 0,
        "metadata_in_extra": False,
    },
    "omol25_val_energy_force_prediction_500.json": {
        "count": 500,
        "digest": "cbeac38ccb32560f6e7e5c41b53ea3d7575866207e95544d3ba9cddf8bfa11bc",
        "nonzero_charge_count": 108,
        "non_singlet_count": 0,
        "metadata_in_extra": True,
    },
    "omol25_geom_orca6_structure_prediction_500.json": {
        "count": 500,
        "digest": "fce04e905de9b015ae4fa1026f2a7ec50786fb9f3054f16865ee51a087d1fa9d",
        "nonzero_charge_count": 31,
        "non_singlet_count": 0,
        "metadata_in_extra": False,
    },
    "omol25_geom_orca6_energy_force_prediction_500.json": {
        "count": 500,
        "digest": "fce04e905de9b015ae4fa1026f2a7ec50786fb9f3054f16865ee51a087d1fa9d",
        "nonzero_charge_count": 31,
        "non_singlet_count": 0,
        "metadata_in_extra": True,
    },
    "omol25_geom_orca6_val_low_force_representative_1000_with_hydrogens_with_forces.json": {
        "count": 1000,
        "digest": "8cb57b193ce66b536c96488c8d41be06889136815d628d40bcd9113362aedcb9",
        "nonzero_charge_count": 46,
        "non_singlet_count": 0,
        "metadata_in_extra": False,
    },
    "omol25_geom_orca6_val_representative_1000_with_hydrogens_with_forces.json": {
        "count": 1000,
        "digest": "7466d076092976db95d96149d4306ae99e399891dfdd9cc9034443d5b55cae0b",
        "nonzero_charge_count": 433,
        "non_singlet_count": 156,
        "metadata_in_extra": False,
    },
}


def _metadata_records(
    entries: dict[str, dict], *, metadata_in_extra: bool
) -> list[tuple[str, str | int, int, int]]:
    records = []
    for entry_id, specification in sorted(entries.items()):
        metadata = specification["extra"] if metadata_in_extra else specification
        source_id = specification.get("example_id")
        if source_id is None:
            source_id = specification["extra"]["source_row_index"]
        records.append(
            (
                entry_id,
                source_id,
                metadata["total_charge"],
                metadata["spin_multiplicity"],
            )
        )
    return records


@pytest.mark.parametrize(
    ("filename", "manifest"),
    METADATA_MANIFESTS.items(),
)
def test_omol25_validation_metadata_matches_source_manifest(filename, manifest):
    """Committed charge/spin labels match their canonical source manifest."""
    entries = json.loads((EXAMPLES_DIR / filename).read_text())
    records = _metadata_records(
        entries, metadata_in_extra=manifest["metadata_in_extra"]
    )

    assert len(records) == manifest["count"]
    assert all(isinstance(charge, int) for _, _, charge, _ in records)
    assert all(isinstance(spin, int) and spin >= 1 for _, _, _, spin in records)
    assert (
        sum(charge != 0 for _, _, charge, _ in records)
        == manifest["nonzero_charge_count"]
    )
    assert sum(spin != 1 for _, _, _, spin in records) == manifest["non_singlet_count"]

    serialized = "".join(
        f"{entry_id}\0{source_id}\0{charge}\0{spin}\n"
        for entry_id, source_id, charge, spin in records
    )
    assert hashlib.sha256(serialized.encode()).hexdigest() == manifest["digest"]


@pytest.mark.parametrize(
    ("structure_filename", "mlip_filename"),
    [
        (
            "omol25_val_structure_prediction_500.json",
            "omol25_val_energy_force_prediction_500.json",
        ),
        (
            "omol25_geom_orca6_structure_prediction_500.json",
            "omol25_geom_orca6_energy_force_prediction_500.json",
        ),
    ],
)
def test_structure_and_mlip_validation_use_the_same_electronic_states(
    structure_filename,
    mlip_filename,
):
    """The two task views of the same OMol subset stay synchronized."""
    structure_entries = json.loads((EXAMPLES_DIR / structure_filename).read_text())
    mlip_entries = json.loads((EXAMPLES_DIR / mlip_filename).read_text())

    structure_records = _metadata_records(structure_entries, metadata_in_extra=False)
    mlip_records = _metadata_records(mlip_entries, metadata_in_extra=True)
    assert structure_records == mlip_records


def test_geom_orca6_structure_force_sampling_provenance_manifest():
    """Lock the force-distribution sample used by structure prediction."""
    path = EXAMPLES_DIR / "omol25_geom_orca6_structure_prediction_500.json"
    entries = json.loads(path.read_text())
    val_indices_path = (
        PROJECT_ROOT / "data/omol25/subsets/train_4M_geom_orca6_val_indices.txt"
    )
    val_indices = {
        int(line)
        for raw_line in val_indices_path.read_text().splitlines()
        if (line := raw_line.split("#", maxsplit=1)[0].strip())
    }
    assert len(val_indices) == 19_980
    digest_rows = []

    for sample_index, (entry_id, specification) in enumerate(entries.items()):
        metadata = specification["extra"]
        example_filename_id = specification["example_id"].replace(":", "_")
        assert entry_id == (f"omol25_geom_orca6_val_force_sample_{sample_index:03d}")
        assert specification["input"].endswith(
            f"/{sample_index:03d}_{example_filename_id}.cif"
        )
        assert specification["contig"] == "A1"
        assert metadata["source_distribution"] == (
            "omol25_geom_orca6_val_mean_force_norm_multinomial_over_all_rows"
        )
        assert metadata["sample_seed"] == 42
        assert metadata["sample_index"] == sample_index
        assert metadata["sample_size"] == 500
        assert metadata["sample_with_replacement"] is True
        assert metadata["source_split"] == "val"
        assert metadata["source_row_index"] >= 0
        assert metadata["source_row_index"] in val_indices
        assert metadata["num_reference_rows"] == 19980
        assert metadata["num_mean_force_norm_bins"] == 150
        assert 0 <= metadata["mean_force_norm_bin_index"] < 150
        assert (
            metadata["mean_force_norm_bin_lower_eV_per_angstrom"]
            <= specification["mean_force_norm"]
            <= metadata["mean_force_norm_bin_upper_eV_per_angstrom"]
        )
        assert metadata["num_atoms_with_hydrogens"] > 0
        assert 0 <= metadata["num_hydrogens"] <= metadata["num_atoms_with_hydrogens"]
        assert specification["has_mean_force_norm"] is True
        assert specification["has_force_rms_p95"] is True
        assert specification["force_rms"] >= specification["mean_force_norm"] > 0.0
        assert specification["force_p95"] > 0.0
        digest_rows.append(
            {
                "entry_id": entry_id,
                "example_id": specification["example_id"],
                "input": specification["input"],
                "source_row_index": metadata["source_row_index"],
                "num_atoms_with_hydrogens": metadata["num_atoms_with_hydrogens"],
                "num_hydrogens": metadata["num_hydrogens"],
                "mean_force_norm_bin_index": metadata["mean_force_norm_bin_index"],
                "mean_force_norm_bin_lower_eV_per_angstrom": metadata[
                    "mean_force_norm_bin_lower_eV_per_angstrom"
                ],
                "mean_force_norm_bin_upper_eV_per_angstrom": metadata[
                    "mean_force_norm_bin_upper_eV_per_angstrom"
                ],
                "total_charge": specification["total_charge"],
                "spin_multiplicity": specification["spin_multiplicity"],
                "mean_force_norm": specification["mean_force_norm"],
                "force_rms": specification["force_rms"],
                "force_p95": specification["force_p95"],
            }
        )

    digest = hashlib.sha256(
        json.dumps(digest_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert len(digest_rows) == 500
    assert digest == "c807574cc1e088665ead3604847f157ec4f69f9ea7d51b6c1f2225617e38aa69"


def test_geom_orca6_generation_force_provenance_manifest():
    """Lock the source rows and force statistics in the generation fixture."""
    path = (
        EXAMPLES_DIR
        / "omol25_geom_orca6_val_low_force_representative_1000_with_hydrogens_with_forces.json"
    )
    entries = json.loads(path.read_text())
    digest_rows = []

    for sample_index, (entry_id, specification) in enumerate(entries.items()):
        metadata = specification["extra"]
        n_atoms = metadata["num_atoms_with_hydrogens"]
        n_hydrogens = metadata["num_hydrogens"]
        assert entry_id == (
            f"omol25_geom_orca6_val_low_force_repr_{sample_index:04d}"
            f"_atoms{n_atoms}_hydrogens{n_hydrogens}"
        )
        assert specification["contig"] == f"{n_atoms}m"
        assert metadata["source_distribution"] == (
            "omol25_geom_orca6_mean_force_norm_multinomial_over_all_rows"
        )
        assert metadata["mean_force_norm_filter_operator"] is None
        assert metadata["mean_force_norm_filter_threshold"] is None
        assert metadata["sample_seed"] == 42
        assert metadata["sample_index"] == sample_index
        assert metadata["sample_size"] == 1000
        assert metadata["sample_with_replacement"] is True
        assert metadata["source_split"] == "all"
        assert metadata["num_reference_rows"] == 199805
        assert metadata["num_mean_force_norm_bins"] == 150
        assert 0 <= metadata["mean_force_norm_bin_index"] < 150
        assert (
            metadata["mean_force_norm_bin_lower_eV_per_angstrom"]
            <= metadata["mean_force_norm"]
            <= metadata["mean_force_norm_bin_upper_eV_per_angstrom"]
        )
        assert metadata["has_mean_force_norm"] is True
        assert metadata["has_force_rms_p95"] is True
        assert math.isfinite(metadata["mean_force_norm"])
        assert math.isfinite(metadata["force_rms"])
        assert math.isfinite(metadata["force_p95"])
        assert metadata["force_rms"] >= metadata["mean_force_norm"] > 0.0
        assert metadata["force_p95"] > 0.0
        digest_rows.append(
            {
                "entry_id": entry_id,
                "contig": specification["contig"],
                "source_row_index": metadata["source_row_index"],
                "total_charge": specification["total_charge"],
                "spin_multiplicity": specification["spin_multiplicity"],
                "mean_force_norm_bin_index": metadata["mean_force_norm_bin_index"],
                "mean_force_norm_bin_lower_eV_per_angstrom": metadata[
                    "mean_force_norm_bin_lower_eV_per_angstrom"
                ],
                "mean_force_norm_bin_upper_eV_per_angstrom": metadata[
                    "mean_force_norm_bin_upper_eV_per_angstrom"
                ],
                "mean_force_norm": metadata["mean_force_norm"],
                "force_rms": metadata["force_rms"],
                "force_p95": metadata["force_p95"],
            }
        )

    digest = hashlib.sha256(
        json.dumps(digest_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert len(digest_rows) == 1000
    assert digest == "9e3e92290729ecb89258ce1fcb0fc1ea580352a159cae9e0f22161cd2793b82a"
