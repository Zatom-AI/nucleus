#!/usr/bin/env python3
"""Sample unconditional molecular sizes from GEOM's training conformer histogram."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_inference_json(data_dir: Path, output: Path, num_samples: int, seed: int):
    """Draw atom counts with replacement and write input specs plus provenance.

    Each training conformer contributes one observation, including explicit H.
    No molecular identities, structures, or properties are supplied to inference.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")
    counts_path = data_dir / "train_num_nodes.pt"
    counts = torch.load(counts_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(counts, torch.Tensor)
        or counts.ndim != 1
        or counts.numel() == 0
        or counts.dtype != torch.int64
        or bool((counts <= 0).any())
    ):
        raise ValueError(
            f"Expected a nonempty positive integer vector in {counts_path}."
        )
    provenance = json.loads((data_dir / "source.json").read_text())
    if counts.numel() != provenance["splits"]["train"]["conformers"]:
        raise ValueError("Atom-count sidecar does not match the training split size.")

    frequencies = torch.bincount(counts)
    probabilities = frequencies.double() / counts.numel()
    sampled_counts = torch.multinomial(
        probabilities,
        num_samples,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    entries = {
        f"geom_drugs_unconditional_{index:04d}_atoms{num_atoms}": {
            "length": f"{num_atoms}m"
        }
        for index, num_atoms in enumerate(sampled_counts.tolist())
    }
    with counts_path.open("rb") as handle:
        source_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    sampled_frequencies = torch.bincount(sampled_counts, minlength=len(frequencies))
    report = {
        "source_repo": provenance["repo_id"],
        "source_revision": provenance["revision"],
        "source_split": "train",
        "source_file": counts_path.name,
        "source_file_sha256": source_sha256,
        "training_conformers": counts.numel(),
        "includes_hydrogens": True,
        "weighting": "one_observation_per_training_conformer",
        "sampling": "torch.multinomial",
        "replacement": True,
        "seed": seed,
        "torch_version": str(torch.__version__),
        "num_requests": num_samples,
        "atom_count_distribution": [
            {
                "num_atoms": n,
                "training_frequency": int(frequencies[n]),
                "probability": float(probabilities[n]),
                "sampled_frequency": int(sampled_frequencies[n]),
            }
            for n in frequencies.nonzero().flatten().tolist()
        ],
    }
    report_path = output.with_suffix(".sampling.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    for path in (output, report_path):
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")
    output.write_text(json.dumps(entries, indent=2) + "\n")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {len(entries)} requests to {output}")
    print(f"Distribution and sampling provenance: {report_path}")
    print(
        f"Sampled atom counts: {int(sampled_counts.min())}–{int(sampled_counts.max())}; "
        f"mean {float(sampled_counts.double().mean()):.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/geom")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (
        REPO_ROOT
        / "models/zatom2/docs/examples"
        / f"geom_drugs_unconditional_{args.num_samples}.json"
    )
    write_inference_json(args.data_dir, output, args.num_samples, args.seed)
