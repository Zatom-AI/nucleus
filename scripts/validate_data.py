#!/usr/bin/env python3
"""Verify original paper training shard cardinalities without running training."""

import os
from pathlib import Path


def main():
    from dotenv import load_dotenv
    from atomworks.ml.datasets import ASELMDBDataset

    load_dotenv(override=False)
    for variable, expected in [
        ("OMOL25_DATA_DIR", 3986754),
        ("OMAT24_DATA_DIR", 1009850),
    ]:
        root = Path(os.environ[variable]).expanduser()
        shards = sorted(
            p for p in root.rglob("*.aselmdb") if not p.name.startswith("._")
        )
        if not shards:
            raise FileNotFoundError(f"No *.aselmdb shards under {root}")
        with ASELMDBDataset(
            paths=shards, name=variable, readonly=True, readahead=False
        ) as data:
            if len(data) != expected:
                raise ValueError(
                    f"{variable}: {len(data)} records, expected {expected}"
                )
            print(f"{variable}: {len(shards)} shards, {len(data)} records")


if __name__ == "__main__":
    main()
