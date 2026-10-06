#!/usr/bin/env python3
"""Safely archive the known Table-2 mmCIF/CHGNet parse-fallback failure.

This is intentionally not a general reset command. It requires completed,
unchanged sampling and QM9 scoring, CHGNet v1 fallbacks for every MP20 input,
and a LeMat run which never advanced beyond preflight. Without --apply it only
validates and prints a plan. It never submits jobs or changes source samples.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gzip
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

PROTOCOL = "zatom2-paper-table2-qm9-mp20"
PARSE_ERROR = "ValueError: Invalid CIF file with no structures!"
STAGE_PATHS = ("mp20/chgnet_prerelax", "mp20/lematgenbench")
LOCK_PATHS = (
    ".table2-launcher.lock",
    ".table2_evaluation.lock",
    "mp20/chgnet_prerelax/.worker.lock",
    "mp20/lematgenbench/.resume.lock",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    _require(not candidate.is_absolute(), f"Expected relative path: {relative}")
    _require(".." not in candidate.parts, f"Unsafe relative path: {relative}")
    path = root / candidate
    _require(path.is_relative_to(root), f"Path escapes output directory: {path}")
    for part in (path, *path.parents):
        if part == root:
            break
        _require(not part.is_symlink(), f"Refusing symbolic link: {part}")
    _require(path.exists(), f"Required artifact is missing: {path}")
    return path


def _json(root: Path, relative: str) -> Any:
    return json.loads(_path(root, relative).read_text())


def _sample_inventory(root: Path, domain: str, summary: dict) -> dict[str, str]:
    manifest_path = _path(root, f"{domain}/sample_manifest.json")
    _require(
        _sha256(manifest_path) == summary["sample_manifest_sha256"],
        f"{domain} sample manifest changed",
    )
    rows = json.loads(manifest_path.read_text())
    _require(
        isinstance(rows, list) and len(rows) == summary["num_samples"] and bool(rows),
        f"{domain} sampling is incomplete",
    )
    aggregate = hashlib.sha256()
    files = {}
    for index, row in enumerate(rows):
        sample_id = f"{domain}-{index:06d}"
        _require(
            row["sample_id"] == sample_id and row["sample_index"] == index,
            f"Noncanonical sample ID in {domain} manifest",
        )
        aggregate.update(sample_id.encode())
        for key in ("cif_path", "metadata_path"):
            relative = row[key]
            _require(
                Path(relative).parent == Path("samples"),
                f"Unexpected {domain} sample path: {relative}",
            )
            path = _path(root, f"{domain}/{relative}")
            _require(path.name not in files, f"Duplicate sample artifact: {path}")
            files[path.name] = _sha256(path)
            aggregate.update(files[path.name].encode("ascii"))
    actual = {path.name for path in _path(root, f"{domain}/samples").iterdir()}
    _require(actual == set(files), f"Unexpected or missing {domain} sample artifacts")
    _require(
        aggregate.hexdigest() == summary["sample_artifacts_sha256"],
        f"{domain} sample contents changed",
    )
    return {
        name: digest
        for name, digest in files.items()
        if name.endswith((".cif", ".cif.gz"))
    }


def _validate(root: Path) -> dict:
    run = _json(root, "run_manifest.json")
    _require(
        run["protocol"]["name"] == PROTOCOL and run["protocol"]["version"] == 3,
        "Recovery only supports Table-2 protocol v3",
    )
    protocol_path = Path(run["protocol"]["manifest"])
    _require(
        protocol_path.is_file()
        and _sha256(protocol_path) == run["protocol"]["manifest_sha256"],
        "Frozen protocol manifest is missing or changed",
    )
    for relative in ("mp20/metrics.json", "table2_metrics.json", "table2_metrics.csv"):
        _require(
            not (root / relative).exists(), f"Scoring already completed: {relative}"
        )

    inventories = {
        domain: _sample_inventory(root, domain, run["domains"][domain])
        for domain in ("qm9", "mp20")
    }
    qm9 = _json(root, "qm9/metrics.json")["evaluation"]
    summary = run["domains"]["qm9"]
    _require(
        qm9["protocol"] == PROTOCOL
        and qm9["protocol_version"] == 3
        and qm9["num_attempted"] == summary["num_samples"]
        and all(
            qm9[key] == summary[key]
            for key in ("sample_manifest_sha256", "sample_artifacts_sha256")
        ),
        "QM9 metrics do not match the completed raw samples",
    )

    stage = _path(root, STAGE_PATHS[0])
    manifest = _json(stage, "chgnet_prerelax.manifest.json")
    contract = manifest["contract"]
    expected = inventories["mp20"]
    _require(
        manifest["complete"] is True
        and contract["protocol_version"] == 1
        and Path(contract["input_dir"]) == root / "mp20/samples"
        and contract["inputs"]
        == [{"name": name, "sha256": expected[name]} for name in sorted(expected)]
        and manifest["contract_sha256"] == _digest(contract)
        and manifest["runtime_sha256"] == _digest(manifest["runtime"]),
        "CHGNet manifest is not an intact completed pre-conversion run",
    )
    _require(
        manifest["attempted_count"] == len(expected)
        and manifest["completed_count"] == len(expected)
        and manifest["converged_count"] == 0
        and manifest["status_counts"]
        == {
            "fallback_parse_error": len(expected),
            "fallback_relaxation_error": 0,
            "relaxed": 0,
        },
        "Recovery requires all inputs to have failed during parsing",
    )
    records = []
    for name, digest in sorted(expected.items()):
        record = _json(stage, f"records/{name}.json")
        _require(
            record["name"] == name
            and record["status"] == "fallback_parse_error"
            and record["error"] == PARSE_ERROR
            and record["converged"] is False
            and record["input_sha256"] == digest
            and record["output_sha256"] == digest
            and record["contract_sha256"] == manifest["contract_sha256"]
            and record["runtime_sha256"] == manifest["runtime_sha256"]
            and _sha256(_path(stage, f"cifs/{name}")) == digest,
            f"CHGNet record is not a byte-preserving known parse fallback: {name}",
        )
        source = _path(root, f"mp20/samples/{name}")
        opener = gzip.open if name.endswith(".gz") else open
        with opener(source, "rt") as stream:
            content = stream.read()
        _require(
            "_atom_site.label_atom_id" in content
            and "_cell.length_a" in content
            and "_atom_site_label" not in content,
            f"Source is not the affected mmCIF dialect: {name}",
        )
        records.append(record)
    _require(
        {path.name for path in _path(stage, "records").iterdir()}
        == {f"{name}.json" for name in expected}
        and {path.name for path in _path(stage, "cifs").iterdir()} == set(expected)
        and manifest["records_sha256"] == _digest(records),
        "CHGNet completion records or CIF inventory changed",
    )

    lemat = _path(root, STAGE_PATHS[1])
    # Only the observed, pre-oracle failure is supported. Never discard useful
    # cached oracle calculations or completed results as part of this repair.
    allowed = {
        ".resume.lock",
        "cif_paths.txt",
        "input_cifs",
        "lematgenbench_mace_preflight.stdout.log",
        "lematgenbench_mace_preflight.stderr.log",
        "resume_state",
    }
    _require(
        {path.name for path in lemat.iterdir()} <= allowed,
        "LeMat contains artifacts beyond MACE preflight; manual review required",
    )
    state = _path(lemat, "resume_state")
    _require(
        {path.name for path in state.iterdir()} == {"request.json"},
        "LeMat has completed results or additional resume state",
    )
    for directory in (stage, lemat):
        _require(
            not any(path.is_symlink() for path in directory.rglob("*")),
            f"Refusing symlinks in archive target: {directory}",
        )
    preserved = (
        "run_manifest.json",
        "qm9/sample_manifest.json",
        "mp20/sample_manifest.json",
        "qm9/metrics.json",
    )
    return {
        "reason": "CHGNet v1 could not parse generated material mmCIF dialect",
        "out_dir": str(root),
        "archive_paths": list(STAGE_PATHS),
        "preserved_file_sha256": {
            relative: _sha256(_path(root, relative)) for relative in preserved
        },
        "sample_artifacts_sha256": {
            domain: run["domains"][domain]["sample_artifacts_sha256"]
            for domain in inventories
        },
        "fallback_count": len(expected),
    }


def archive_fallbacks(out_dir: Path, *, apply: bool = False) -> Path | None:
    """Validate an idle affected run and optionally archive only failed stages."""
    root = out_dir.expanduser().resolve(strict=True)
    with contextlib.ExitStack() as stack:
        # Match the normal launcher's outer-to-inner acquisition order. Open
        # existing files read-only so dry runs do not create or modify locks.
        for relative in LOCK_PATHS:
            stream = stack.enter_context(_path(root, relative).open("rb"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(f"Run is active; lock held: {relative}") from error
        plan = _validate(root)
        print(
            f"Validated {plan['fallback_count']} known, byte-preserving parse fallbacks."
        )
        print(
            "Preserve raw QM9/MP20 samples, QM9 metrics, protocol, and launcher state."
        )
        print("Archive only: " + ", ".join(STAGE_PATHS))
        if not apply:
            print(
                "DRY RUN: no files changed. Re-run with --apply to archive these stages."
            )
            return None

        archive = Path(tempfile.mkdtemp(prefix="cif-fallback-recovery-", dir=root))
        print(f"Recoverable archive: {archive}", flush=True)
        with (archive / "recovery.json").open("x") as stream:
            json.dump(plan, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        moved = []
        try:
            for relative in STAGE_PATHS:
                source = root / relative
                destination = archive / source.name
                source.rename(destination)
                moved.append((source, destination))
        except OSError:
            # On ordinary failures, restore the original layout. A hard kill
            # can leave a partial archive; recovery.json identifies both paths.
            for source, destination in reversed(moved):
                destination.rename(source)
            raise
        print("Archived failed stages. Nothing was deleted; resume the same OUT_DIR.")
        return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Archive after validation")
    args = parser.parse_args()
    try:
        archive_fallbacks(args.out_dir, apply=args.apply)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Refusing recovery: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
