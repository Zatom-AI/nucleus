"""Check training-only size sampling and the shipped GEOM request provenance."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from scripts.make_geom_unconditional_json import write_inference_json


@pytest.fixture
def training_data(tmp_path):
    data_dir = tmp_path / "geom"
    data_dir.mkdir()
    torch.save(
        torch.tensor([3, 3, 3, 5], dtype=torch.int64), data_dir / "train_num_nodes.pt"
    )
    (data_dir / "source.json").write_text(
        json.dumps(
            {
                "repo_id": "test/geom",
                "revision": "pinned-test-release",
                "splits": {"train": {"conformers": 4}},
            }
        )
    )
    return data_dir


def test_requests_only_supply_sizes_and_record_training_histogram(
    training_data, tmp_path
):
    output = tmp_path / "requests.json"
    write_inference_json(training_data, output, num_samples=20, seed=42)
    requests = json.loads(output.read_text())
    report = json.loads(output.with_suffix(".sampling.json").read_text())

    assert len(requests) == 20
    assert all(set(spec) == {"length"} for spec in requests.values())
    observed = Counter(int(spec["length"][:-1]) for spec in requests.values())
    assert set(observed) <= {3, 5}
    assert report["source_split"] == "train"
    assert report["source_revision"] == "pinned-test-release"
    assert report["training_conformers"] == 4
    assert report["includes_hydrogens"] is True
    assert (
        report["source_file_sha256"]
        == hashlib.sha256(
            (training_data / "train_num_nodes.pt").read_bytes()
        ).hexdigest()
    )
    assert report["atom_count_distribution"] == [
        {
            "num_atoms": num_atoms,
            "training_frequency": frequency,
            "probability": frequency / 4,
            "sampled_frequency": observed[num_atoms],
        }
        for num_atoms, frequency in ((3, 3), (5, 1))
    ]

    second_output = tmp_path / "repeat.json"
    write_inference_json(training_data, second_output, num_samples=20, seed=42)
    assert second_output.read_bytes() == output.read_bytes()


@pytest.mark.parametrize(
    "counts", [torch.tensor([]), torch.tensor([0, 3]), torch.tensor([[3]])]
)
def test_invalid_atom_counts_are_rejected(training_data, tmp_path, counts):
    torch.save(counts, training_data / "train_num_nodes.pt")
    with pytest.raises(ValueError, match="positive integer vector"):
        write_inference_json(training_data, tmp_path / "requests.json", 5, 42)


def test_mismatched_split_counts_are_rejected(training_data, tmp_path):
    torch.save(torch.tensor([3, 3]), training_data / "train_num_nodes.pt")
    with pytest.raises(ValueError, match="training split size"):
        write_inference_json(training_data, tmp_path / "requests.json", 5, 42)


def test_existing_provenance_is_not_overwritten(training_data, tmp_path):
    output = tmp_path / "requests.json"
    report = output.with_suffix(".sampling.json")
    report.write_text("existing provenance")
    with pytest.raises(FileExistsError, match="Output already exists"):
        write_inference_json(training_data, output, 5, 42)
    assert not output.exists()
    assert report.read_text() == "existing provenance"


@pytest.mark.parametrize("num_requests", [1000, 10000])
def test_shipped_requests_match_their_sampling_provenance(num_requests):
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "models/zatom2/docs/examples"
        / f"geom_drugs_unconditional_{num_requests}.json"
    )
    requests = json.loads(path.read_text())
    report = json.loads(path.with_suffix(".sampling.json").read_text())
    observed = Counter(int(spec["length"][:-1]) for spec in requests.values())

    assert len(requests) == report["num_requests"] == num_requests
    assert all(set(spec) == {"length"} for spec in requests.values())
    assert report["source_split"] == "train"
    assert (
        sum(row["training_frequency"] for row in report["atom_count_distribution"])
        == report["training_conformers"]
    )
    assert observed == Counter(
        {
            row["num_atoms"]: row["sampled_frequency"]
            for row in report["atom_count_distribution"]
        }
    )
