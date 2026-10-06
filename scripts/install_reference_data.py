#!/usr/bin/env python3
"""Install the exact paper reference data from a downloaded release attachment."""

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def install(archive_path: Path, destination: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    with archive_path.open("rb") as handle:
        if (
            hashlib.file_digest(handle, "sha256").hexdigest()
            != manifest["archive_sha256"]
        ):
            raise ValueError(
                "Reference archive SHA-256 does not match the release manifest"
            )
    expected = manifest["files"]
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        staging = Path(temporary)
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            if len(members) != len(expected) or {m.name for m in members} != set(
                expected
            ):
                raise ValueError(
                    "Reference archive has missing, duplicate, or unexpected entries"
                )
            for member in members:
                name = Path(member.name)
                if not member.isfile() or name.is_absolute() or ".." in name.parts:
                    raise ValueError(f"Unsafe archive member: {member.name}")
                target = (destination / name).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError(f"Destination escapes data root: {target}")
                if target.exists():
                    with target.open("rb") as handle:
                        if (
                            hashlib.file_digest(handle, "sha256").hexdigest()
                            != expected[member.name]["sha256"]
                        ):
                            raise FileExistsError(
                                f"Existing reference differs: {target}"
                            )
                archive.extract(member, staging, filter="data")
                path = staging / name
                with path.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                if (
                    digest != expected[member.name]["sha256"]
                    or path.stat().st_size != expected[member.name]["size"]
                ):
                    raise ValueError(f"Reference file mismatch: {member.name}")
        for name in expected:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.move(staging / name, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "docs/reference_data_manifest.json"
    )
    args = parser.parse_args()
    install(args.archive, args.destination, args.manifest)
    print(f"Installed verified paper references under {args.destination}")
