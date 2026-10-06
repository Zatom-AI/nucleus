#!/usr/bin/env python3
"""Build a reference-data release attachment from the original paper caches."""

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path

FILES = (
    "omol25/geom_orca6_rdkit_conformers.pt",
    "omol25/subsets/omol25_train_4M_smiles_obabel_fpsim2.h5",
    "omat24/subsets/train_1M_fingerprints.npy",
)


def package(data_root, output, manifest_path):
    for name in FILES:
        if not (data_root / name).is_file():
            raise FileNotFoundError(data_root / name)
    if output.exists():
        raise FileExistsError(output)
    manifest = {
        "description": "Original Zatom-2 AFD and novelty reference caches",
        "archive": output.name,
        "files": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as archive,
    ):
        for name in FILES:
            source = data_root / name
            with source.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            manifest["files"][name] = {"size": source.stat().st_size, "sha256": digest}
            info = archive.gettarinfo(source, arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            with source.open("rb") as handle:
                archive.addfile(info, handle)
            print(f"Packed {name}", flush=True)
    with output.open("rb") as handle:
        manifest["archive_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest["archive_size"] = output.stat().st_size
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
    args = parser.parse_args()
    package(args.data_root, args.output, args.manifest)
