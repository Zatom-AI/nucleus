#!/usr/bin/env python3
"""Build a reference-data release attachment from the original paper caches.

GitHub Releases enforces a 2 GB per-asset file size limit. Because the full
packaged archive is ~2.26 GB (2,255,986,830 bytes), it is split into equal
parts for upload to GitHub release "1.0.0" (tag: 1.0.0):
    zatom2-reference-data.tar.gz.part00
    zatom2-reference-data.tar.gz.part01
along with dist/checksums.sha256.

Users download the parts and reassemble them via:
    cat zatom2-reference-data.tar.gz.part* > zatom2-reference-data.tar.gz
and install them with:
    python scripts/install_reference_data.py --archive zatom2-reference-data.tar.gz
"""

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path

FILES = (
    "omol25/geom_orca6_rdkit_conformers.pt",
    "omol25/afd_feature_cache/Ty-Perez__ct-scd-geom10_reference_features.pt",
    "omol25/afd_feature_cache/Ty-Perez__ct-scd-omol25_reference_features.pt",
    "omol25/afd_feature_cache/Ty-Perez__ct-scd-omol25_reference_features.metadata.json",
    "omol25/sidecars/train_4M_all_bonds.lmdb",
    "omol25/sidecars/val_all_bonds.lmdb",
    "omol25/subsets/omol25_train_4M_smiles_obabel_fpsim2.h5",
    "omol25/subsets/train_4M_geom_orca6_train_smiles.txt",
    "omol25/subsets/train_4M_geom_orca6_train_smiles_fpsim2.h5",
    "omat24/subsets/train_1M_fingerprints.npy",
)


def _digest(handle):
    if hasattr(hashlib, "file_digest"):
        return hashlib.file_digest(handle, "sha256").hexdigest()
    hasher = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        hasher.update(chunk)
    return hasher.hexdigest()


def split_archive(output: Path, num_parts: int = 2):
    total_size = output.stat().st_size
    part_size = (total_size + num_parts - 1) // num_parts
    parts = {}
    with output.open("rb") as src:
        for i in range(num_parts):
            part_name = f"{output.name}.part{i:02d}"
            part_path = output.parent / part_name
            bytes_left = min(part_size, total_size - i * part_size)
            hasher = hashlib.sha256()
            with part_path.open("wb") as dst:
                while bytes_left > 0:
                    chunk = src.read(min(1024 * 1024, bytes_left))
                    if not chunk:
                        break
                    dst.write(chunk)
                    hasher.update(chunk)
                    bytes_left -= len(chunk)
            parts[part_name] = {
                "size": part_path.stat().st_size,
                "sha256": hasher.hexdigest(),
            }
            print(
                f"Created part {part_name}: {part_path.stat().st_size} bytes",
                flush=True,
            )
    return parts


def package(data_root, output, manifest_path, split_parts=2):
    for name in FILES:
        if not (data_root / name).is_file():
            raise FileNotFoundError(data_root / name)
    if output.exists():
        raise FileExistsError(output)
    manifest = {
        "description": "Original Zatom-2 OMol25 bond sidecars, AFD references, and OMol25/OMat24 novelty caches",
        "archive": output.name,
        "files": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", mtime=0
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                for name in FILES:
                    source = data_root / name
                    with source.open("rb") as handle:
                        digest = _digest(handle)
                    manifest["files"][name] = {
                        "size": source.stat().st_size,
                        "sha256": digest,
                    }
                    info = archive.gettarinfo(source, arcname=name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    with source.open("rb") as handle:
                        archive.addfile(info, handle)
                    print(f"Packed {name}", flush=True)
    with output.open("rb") as handle:
        manifest["archive_sha256"] = _digest(handle)
    manifest["archive_size"] = output.stat().st_size
    if split_parts and split_parts > 1:
        manifest["archive_parts"] = split_archive(output, split_parts)
        checksums_path = output.parent / "checksums.sha256"
        with checksums_path.open("w") as chk:
            chk.write(f"{manifest['archive_sha256']}  {output.name}\n")
            for part_name, meta in manifest["archive_parts"].items():
                chk.write(f"{meta['sha256']}  {part_name}\n")
        print(f"Wrote checksums to {checksums_path}", flush=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{output}: {output.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("dist/zatom2-reference-data.tar.gz")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("docs/reference_data_manifest.json")
    )
    parser.add_argument(
        "--split-parts",
        type=int,
        default=2,
        help="Number of equal parts to split the archive into (default: 2, for GitHub 2GB limit)",
    )
    args = parser.parse_args()
    package(args.data_root, args.output, args.manifest, split_parts=args.split_parts)
