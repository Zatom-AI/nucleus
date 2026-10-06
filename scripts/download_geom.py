#!/usr/bin/env python3
"""Download the pinned GEOM release used by Zatom and record its split counts.

Run from the repository root with ``uv run python scripts/download_geom.py``.
The upstream download_data.py uses Zatom-AI/geom's processed archive, not raw
RDKit pickles. The extracted tensors can be used directly by Nucleus.
"""

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path

REPO_ID = "Zatom-AI/geom"
REVISION = "855cd36632b30787b913743ec21903283dba41c8"
ARCHIVE_SHA256 = "3183cae6e8eaaa63500cc67fd1e949d62d4c97b27148c922486ef008359856d5"


def prepare_geom(data_dir: Path) -> dict:
    """Download, verify and extract the official processed conformer splits."""
    from huggingface_hub import hf_hub_download

    data_dir = data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "processed.tar.gz"
    if not archive_path.is_file():
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            filename=archive_path.name,
            local_dir=data_dir,
        )
    with archive_path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError(
            f"Checksum mismatch for {archive_path}; expected {ARCHIVE_SHA256}."
        )
    print(f"Verified {REPO_ID}@{REVISION}: {archive_path}", flush=True)

    # Extract only into this dataset directory; do not overwrite existing files.
    with tarfile.open(archive_path) as archive:
        for member in archive:
            if member.isdir():
                continue
            if (
                not member.isfile()
                or Path(member.name).is_absolute()
                or ".." in Path(member.name).parts
            ):
                raise ValueError(f"Unexpected GEOM archive member: {member.name}")
            target = data_dir / member.name
            if target.is_file():
                if target.stat().st_size != member.size:
                    raise FileExistsError(
                        f"Existing file differs from archive: {target}"
                    )
                with (
                    target.open("rb") as existing,
                    archive.extractfile(member) as source,
                ):
                    if (
                        hashlib.file_digest(existing, "sha256").digest()
                        != hashlib.file_digest(source, "sha256").digest()
                    ):
                        raise FileExistsError(
                            f"Existing file differs from archive: {target}"
                        )
                continue
            archive.extract(member, data_dir, filter="data")

    import torch

    report = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "archive_sha256": digest,
        "selection": "all_released_conformers_in_source_order",
        "splits": {},
    }
    for split in ("train", "val", "test"):
        data, slices, _ = torch.load(
            data_dir / f"{split}.pt", map_location="cpu", weights_only=False, mmap=True
        )
        counts = Counter(source_id.rsplit("_", 1)[0] for source_id in data["id"])
        num_atoms = slices["pos"].diff()
        if len(data["id"]) != len(num_atoms) or not counts or max(counts.values()) > 5:
            raise ValueError(f"Unexpected conformer counts in {split}.pt")
        stats = {
            "molecules": len(counts),
            "conformers": len(data["id"]),
            "conformers_per_molecule": dict(sorted(Counter(counts.values()).items())),
            "min_atoms": int(num_atoms.min()),
            "max_atoms": int(num_atoms.max()),
        }
        report["splits"][split] = stats
        print(f"{split}: {stats}", flush=True)
        del data, slices, counts
    (data_dir / "source.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/geom",
    )
    prepare_geom(parser.parse_args().data_dir)
