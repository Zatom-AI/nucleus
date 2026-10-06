import json
from pathlib import Path

import numpy as np
import pytest

from nucleus.utils.atomistic_properties import AtomisticPropertyStatistics
from scripts import fit_atomistic_property_statistics as fitter
from scripts.fit_atomistic_property_statistics import StreamingPropertyMoments


def _artifact_dict():
    return {
        "schema_version": 1,
        "domains": {
            "omol25": {
                "domain_id": 0,
                "source_split_ids": ["omol25-train", "indices_sha256:abc"],
                "energy_unit": "eV",
                "force_unit": "eV/angstrom",
                "elemental_references_eV": {"1": -1.0, "8": -4.0},
                "residual_energy_per_atom_mean_eV": 0.5,
                "residual_energy_per_atom_scale_eV": 2.0,
                "force_rms_eV_per_angstrom": 3.0,
            },
            "omat24": {
                "domain_id": 1,
                "source_split_ids": ["omat24-train", "indices_sha256:def"],
                "energy_unit": "eV",
                "force_unit": "eV/angstrom",
                "elemental_references_eV": {"8": -5.0},
                "residual_energy_per_atom_mean_eV": -0.5,
                "residual_energy_per_atom_scale_eV": 4.0,
                "force_rms_eV_per_angstrom": 6.0,
            },
        },
    }


def test_statistics_artifact_round_trip_and_dense_tensors(tmp_path):
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(_artifact_dict()), encoding="utf-8")

    statistics = AtomisticPropertyStatistics.load(input_path)
    output_path = tmp_path / "output.json"
    statistics.save(output_path)
    reloaded = AtomisticPropertyStatistics.load(output_path)
    tensors = reloaded.tensors(max_atomic_number=8)

    assert reloaded.domain_names == ("omol25", "omat24")
    assert tensors["elemental_references"].shape == (2, 9)
    assert tensors["elemental_references"][0, 1].item() == -1.0
    assert tensors["elemental_references"][1, 8].item() == -5.0
    assert tensors["elemental_reference_valid"][0, 1]
    assert not tensors["elemental_reference_valid"][1, 1]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", 2, "schema_version"),
        ("energy_unit", "hartree", "energy_unit"),
        ("force_unit", "hartree/bohr", "force_unit"),
        ("residual_energy_per_atom_scale_eV", 0.0, "scales must be positive"),
    ],
)
def test_statistics_artifact_rejects_incompatible_contract(field, value, match):
    artifact = _artifact_dict()
    if field == "schema_version":
        artifact[field] = value
    else:
        artifact["domains"]["omol25"][field] = value

    with pytest.raises(ValueError, match=match):
        AtomisticPropertyStatistics.from_dict(artifact)


def test_streaming_fitter_matches_brute_force_residual_and_force_moments():
    rows = [
        (np.array([1]), 2.5, np.array([[1.0, 0.0, 0.0]])),
        (np.array([8]), 5.5, np.array([[0.0, 2.0, 0.0]])),
        (np.array([1, 8]), 8.0, np.array([[0.0, 0.0, 3.0], [1.0, 2.0, 2.0]])),
        (np.array([1, 1, 8]), 10.0, np.ones((3, 3))),
    ]
    moments = StreamingPropertyMoments()
    for atomic_numbers, energy, forces in rows:
        moments.update(atomic_numbers, energy, forces)

    fitted = moments.finalize(
        name="omol25",
        domain_id=0,
        source_split_ids=("train",),
    )
    references = fitted.elemental_references_eV
    brute_residuals = []
    all_forces = []
    for atomic_numbers, energy, forces in rows:
        reference_energy = sum(references[int(z)] for z in atomic_numbers)
        brute_residuals.append((energy - reference_energy) / len(atomic_numbers))
        all_forces.append(forces.reshape(-1))

    assert fitted.residual_energy_per_atom_mean_eV == pytest.approx(
        np.mean(brute_residuals), abs=1e-12
    )
    assert fitted.residual_energy_per_atom_scale_eV == pytest.approx(
        np.std(brute_residuals), abs=1e-12
    )
    assert fitted.force_rms_eV_per_angstrom == pytest.approx(
        np.sqrt(np.mean(np.concatenate(all_forces) ** 2)), abs=1e-12
    )
    assert fitted.n_systems == len(rows)
    assert fitted.n_atoms == sum(len(row[0]) for row in rows)


class _StoredAtoms:
    def __init__(self, numbers, energy, forces):
        self.numbers = np.asarray(numbers)
        self.calc = type(
            "StoredCalculator",
            (),
            {"results": {"energy": energy, "forces": np.asarray(forces)}},
        )()


class _InMemoryDataset:
    def __init__(self):
        self.rows = [
            _StoredAtoms([1], 1.0, [[1.0, 0.0, 0.0]]),
            _StoredAtoms([1], 3.0, [[0.0, 2.0, 0.0]]),
            _StoredAtoms([1, 1], 5.0, [[0.0, 0.0, 3.0], [1.0, 1.0, 1.0]]),
        ]
        self.requested_indices = []
        self.closed = False

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        self.requested_indices.append(index)
        return self.rows[index]

    def close(self):
        self.closed = True


def _make_data_dir(root: Path, *, shard_sizes=(3, 5)) -> Path:
    root.mkdir(parents=True)
    for index, shard_size in enumerate(shard_sizes):
        (root / f"data{index:04d}.aselmdb").write_bytes(b"x" * shard_size)
    return root


def _required_cli_args(output_path: Path) -> list[str]:
    return [
        "--omol25-split-id",
        "omol25/train_4M-all",
        "--omat24-split-id",
        "omat24/train-all",
        "--output",
        str(output_path),
    ]


def test_cli_all_records_uses_environment_backed_data_directories(tmp_path):
    omol_dir = _make_data_dir(tmp_path / "omol")
    omat_dir = _make_data_dir(tmp_path / "omat")

    args = fitter._parse_args(
        ["--all-records", *_required_cli_args(tmp_path / "stats.json")],
        environ={
            "OMOL25_DATA_DIR": str(omol_dir),
            "OMAT24_DATA_DIR": str(omat_dir),
        },
    )

    assert args.all_records
    assert args.omol25_data_dir == omol_dir
    assert args.omat24_data_dir == omat_dir
    assert args.omol25_indices is None
    assert args.omat24_indices is None


def test_cli_data_directory_option_overrides_environment(tmp_path):
    env_omol_dir = _make_data_dir(tmp_path / "env_omol")
    cli_omol_dir = _make_data_dir(tmp_path / "cli_omol")
    omat_dir = _make_data_dir(tmp_path / "omat")

    args = fitter._parse_args(
        [
            "--all-records",
            "--omol25-data-dir",
            str(cli_omol_dir),
            *_required_cli_args(tmp_path / "stats.json"),
        ],
        environ={
            "OMOL25_DATA_DIR": str(env_omol_dir),
            "OMAT24_DATA_DIR": str(omat_dir),
        },
    )

    assert args.omol25_data_dir == cli_omol_dir


@pytest.mark.parametrize(
    "selection_args",
    [
        [],
        ["--omol25-indices", "omol.txt"],
        ["--all-records", "--omol25-indices", "omol.txt"],
    ],
)
def test_cli_rejects_incomplete_or_ambiguous_record_selection(
    tmp_path,
    selection_args,
):
    omol_dir = _make_data_dir(tmp_path / "omol")
    omat_dir = _make_data_dir(tmp_path / "omat")
    (tmp_path / "omol.txt").write_text("0\n", encoding="utf-8")
    normalized_selection_args = [
        str(tmp_path / value) if value == "omol.txt" else value
        for value in selection_args
    ]

    with pytest.raises(SystemExit):
        fitter._parse_args(
            [
                *normalized_selection_args,
                *_required_cli_args(tmp_path / "stats.json"),
            ],
            environ={
                "OMOL25_DATA_DIR": str(omol_dir),
                "OMAT24_DATA_DIR": str(omat_dir),
            },
        )


def test_cli_accepts_two_explicit_index_manifests(tmp_path):
    omol_dir = _make_data_dir(tmp_path / "omol")
    omat_dir = _make_data_dir(tmp_path / "omat")
    omol_indices = tmp_path / "omol.txt"
    omat_indices = tmp_path / "omat.txt"
    omol_indices.write_text("0\n", encoding="utf-8")
    omat_indices.write_text("0\n", encoding="utf-8")

    args = fitter._parse_args(
        [
            "--omol25-indices",
            str(omol_indices),
            "--omat24-indices",
            str(omat_indices),
            *_required_cli_args(tmp_path / "stats.json"),
        ],
        environ={
            "OMOL25_DATA_DIR": str(omol_dir),
            "OMAT24_DATA_DIR": str(omat_dir),
        },
    )

    assert not args.all_records
    assert args.omol25_indices == omol_indices
    assert args.omat24_indices == omat_indices


def test_all_records_iteration_and_provenance_are_reproducible(
    tmp_path,
    monkeypatch,
):
    data_dir = _make_data_dir(tmp_path / "train")
    dataset = _InMemoryDataset()
    monkeypatch.setattr(fitter, "_build_dataset", lambda *_args, **_kwargs: dataset)

    result = fitter._fit_domain(
        name="omol25",
        domain_id=0,
        data_dir=data_dir,
        indices_path=None,
        all_records=True,
        split_id="omol25/train_4M-all",
        max_rows=2,
        progress_every=0,
    )

    assert dataset.requested_indices == [0, 1]
    assert dataset.closed
    assert result.n_systems == 2
    assert "selection:all_records" in result.source_split_ids
    assert "dataset_record_count:3" in result.source_split_ids
    assert "selected_record_count:2" in result.source_split_ids
    assert "shard_count:2" in result.source_split_ids
    assert "row_limit:2" in result.source_split_ids
    assert any(
        value.startswith("shard_manifest_v1_sha256:")
        for value in result.source_split_ids
    )
    assert not any(
        value.startswith("indices_sha256:") for value in result.source_split_ids
    )


def test_indexed_mode_validates_bounds_and_closes_dataset(tmp_path, monkeypatch):
    data_dir = _make_data_dir(tmp_path / "train")
    indices_path = tmp_path / "indices.txt"
    indices_path.write_text("-1\n", encoding="utf-8")
    dataset = _InMemoryDataset()
    monkeypatch.setattr(fitter, "_build_dataset", lambda *_args, **_kwargs: dataset)

    with pytest.raises(IndexError, match="outside"):
        fitter._fit_domain(
            name="omol25",
            domain_id=0,
            data_dir=data_dir,
            indices_path=indices_path,
            all_records=False,
            split_id="omol25/train-subset",
            max_rows=None,
            progress_every=0,
        )

    assert dataset.closed


def test_indexed_mode_records_index_and_shard_manifests(tmp_path, monkeypatch):
    data_dir = _make_data_dir(tmp_path / "train")
    indices_path = tmp_path / "indices.txt"
    indices_path.write_text("0\n1\n", encoding="utf-8")
    dataset = _InMemoryDataset()
    monkeypatch.setattr(fitter, "_build_dataset", lambda *_args, **_kwargs: dataset)

    result = fitter._fit_domain(
        name="omol25",
        domain_id=0,
        data_dir=data_dir,
        indices_path=indices_path,
        all_records=False,
        split_id="omol25/train-subset",
        max_rows=None,
        progress_every=0,
    )

    assert "selection:index_manifest" in result.source_split_ids
    assert any(value.startswith("indices_sha256:") for value in result.source_split_ids)
    assert any(
        value.startswith("shard_manifest_v1_sha256:")
        for value in result.source_split_ids
    )


def test_shard_manifest_hash_is_root_independent_and_size_sensitive(tmp_path):
    root_a = _make_data_dir(tmp_path / "a")
    root_b = _make_data_dir(tmp_path / "b")
    paths_a = fitter._shard_paths(root_a)
    paths_b = fitter._shard_paths(root_b)

    digest_a = fitter._shard_manifest_sha256(root_a, paths_a)
    assert digest_a == fitter._shard_manifest_sha256(root_b, paths_b)

    paths_b[0].write_bytes(b"different-size")
    assert digest_a != fitter._shard_manifest_sha256(root_b, paths_b)
