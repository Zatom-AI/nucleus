#!/usr/bin/env python3
"""Build unconditional protein-generation JSONs from the SCOPe length profile."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path


def _summary(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        return {}
    values = sorted(lengths)
    mean, popvar = _population_moments(values)

    def quantile(q: float) -> int:
        idx = round(q * (len(values) - 1))
        return values[idx]

    return {
        "n": len(values),
        "mean": mean,
        "popvar": popvar,
        "std": math.sqrt(popvar),
        "min": values[0],
        "q01": quantile(0.01),
        "q05": quantile(0.05),
        "q10": quantile(0.10),
        "q25": quantile(0.25),
        "median": quantile(0.50),
        "q75": quantile(0.75),
        "q90": quantile(0.90),
        "q95": quantile(0.95),
        "q99": quantile(0.99),
        "max": values[-1],
    }


def _population_moments(lengths: list[int]) -> tuple[float, float]:
    mean = sum(lengths) / len(lengths)
    var = sum((length - mean) ** 2 for length in lengths) / len(lengths)
    return mean, var


def _load_scope_rows(
    path: Path,
    split: str | None,
    exclude_csv: Path | None,
    exclude_column: str,
) -> list[dict[str, str]]:
    if split == "" or split == "all":
        split = None
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if split is not None and rows and "split" in rows[0]:
        rows = [row for row in rows if row.get("split") == split]
    if exclude_csv is not None:
        with exclude_csv.open() as handle:
            excluded_rows = list(csv.DictReader(handle))
        if excluded_rows and exclude_column not in excluded_rows[0]:
            raise ValueError(f"Missing exclude column {exclude_column!r} in {exclude_csv}")
        excluded_ids = {row[exclude_column] for row in excluded_rows}
        rows = [row for row in rows if row.get(exclude_column) not in excluded_ids]
    for row in rows:
        if "length" not in row:
            raise ValueError(f"Missing length column in {path}")
        int(row["length"])
    return rows


def _sample_rows(
    rows: list[dict[str, str]],
    n_generations: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    if n_generations <= len(rows):
        return rng.sample(rows, n_generations)
    return [rng.choice(rows) for _ in range(n_generations)]


def _format_sampled_rows(sampled: list[dict[str, str]]) -> list[dict[str, object]]:
    return [
        {
            "length": int(row["length"]),
            "source_example_id": row.get("example_id", ""),
            "source_domain_id": row.get("domain_id", ""),
            "source_scope_class": row.get("scope_class", ""),
        }
        for row in sampled
    ]


def _sample_scope_lengths_empirical(
    rows: list[dict[str, str]],
    n_generations: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    return _format_sampled_rows(_sample_rows(rows, n_generations, rng))


def _moment_score(
    sampled_lengths: list[int],
    target_mean: float,
    target_var: float,
) -> float:
    mean, var = _population_moments(sampled_lengths)
    mean_scale = max(1.0, math.sqrt(target_var))
    var_scale = max(1.0, target_var)
    return ((mean - target_mean) / mean_scale) ** 2 + ((var - target_var) / var_scale) ** 2


def _sample_scope_lengths_moment_matched(
    rows: list[dict[str, str]],
    n_generations: int,
    seed: int,
    trials: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    target_lengths = [int(row["length"]) for row in rows]
    target_mean, target_var = _population_moments(target_lengths)

    best_rows: list[dict[str, str]] | None = None
    best_score = float("inf")
    for _ in range(max(1, trials)):
        sampled = _sample_rows(rows, n_generations, rng)
        score = _moment_score(
            [int(row["length"]) for row in sampled],
            target_mean,
            target_var,
        )
        if score < best_score:
            best_rows = sampled
            best_score = score

    if best_rows is None:
        raise RuntimeError("Moment-matched sampling failed to produce a candidate")
    return _format_sampled_rows(best_rows)


def _parse_lengths(raw: str) -> list[int]:
    lengths = [int(part) for part in raw.replace(" ", "").split(",") if part]
    if not lengths:
        raise ValueError("No lengths parsed from --lengths")
    return lengths


def _fixed_lengths(
    lengths: list[int],
    samples_per_length: int,
    scope_lengths: list[int],
    allow_ood: bool,
) -> list[dict[str, object]]:
    if not allow_ood and scope_lengths:
        lo, hi = min(scope_lengths), max(scope_lengths)
        ood = [length for length in lengths if length < lo or length > hi]
        if ood:
            raise ValueError(
                f"Fixed lengths {ood} are outside SCOPe length support [{lo}, {hi}]. "
                "Pass --allow-ood-fixed-lengths to override intentionally."
            )
    specs: list[dict[str, object]] = []
    for length in lengths:
        for _ in range(samples_per_length):
            specs.append({"length": length})
    return specs


def _length_counts(lengths: list[int]) -> dict[str, int]:
    return {str(k): v for k, v in sorted(Counter(lengths).items())}


def write_inputs(
    *,
    scope_csv: Path,
    split: str | None,
    exclude_csv: Path | None,
    exclude_column: str,
    output_json: Path,
    manifest_json: Path,
    n_generations: int,
    seed: int,
    sampling_mode: str,
    moment_trials: int,
    lengths: str | None,
    samples_per_length: int,
    allow_ood_fixed_lengths: bool,
) -> None:
    rows = _load_scope_rows(scope_csv, split, exclude_csv, exclude_column)
    if not rows:
        raise ValueError(f"No rows found in {scope_csv} for split={split!r}")
    scope_lengths = [int(row["length"]) for row in rows]

    if lengths:
        mode = "fixed_lengths"
        sampled = _fixed_lengths(
            _parse_lengths(lengths),
            samples_per_length,
            scope_lengths,
            allow_ood_fixed_lengths,
        )
    elif sampling_mode == "moment_matched":
        mode = "scope_moment_matched"
        sampled = _sample_scope_lengths_moment_matched(
            rows, n_generations, seed, moment_trials
        )
    elif sampling_mode == "empirical":
        mode = "scope_empirical"
        sampled = _sample_scope_lengths_empirical(rows, n_generations, seed)
    else:
        raise ValueError(f"Unsupported sampling mode: {sampling_mode}")

    data = {}
    for idx, item in enumerate(sampled):
        length = int(item["length"])
        key = f"scope_len{length:03d}_{idx:03d}"
        data[key] = {"length": length, "plddt_enhanced": False}

    sampled_lengths = [int(item["length"]) for item in sampled]
    manifest = {
        "mode": mode,
        "scope_csv": str(scope_csv),
        "split": split,
        "exclude_csv": str(exclude_csv) if exclude_csv is not None else None,
        "exclude_column": exclude_column if exclude_csv is not None else None,
        "seed": seed,
        "sampling_mode": sampling_mode,
        "moment_trials": moment_trials if sampling_mode == "moment_matched" else None,
        "n_generation_specs": len(sampled),
        "design_spec_format": "inputless_protein_length",
        "scope_length_summary": _summary(scope_lengths),
        "sampled_length_summary": _summary(sampled_lengths),
        "sampled_length_counts": _length_counts(sampled_lengths),
        "sampled_sources": sampled,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(data, indent=2) + "\n")
    manifest_json.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"wrote {len(data)} specs to {output_json}; "
        f"mode={mode}; sampled_length_range={min(sampled_lengths)}-{max(sampled_lengths)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope-csv", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--exclude-csv", type=Path, default=None)
    parser.add_argument("--exclude-column", default="domain_id")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--n-generations", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sampling-mode",
        choices=["moment_matched", "empirical"],
        default="moment_matched",
    )
    parser.add_argument("--moment-trials", type=int, default=5000)
    parser.add_argument("--lengths", default=None)
    parser.add_argument("--samples-per-length", type=int, default=1)
    parser.add_argument("--allow-ood-fixed-lengths", action="store_true")
    args = parser.parse_args()

    write_inputs(
        scope_csv=args.scope_csv,
        split=args.split,
        exclude_csv=args.exclude_csv,
        exclude_column=args.exclude_column,
        output_json=args.output_json,
        manifest_json=args.manifest_json,
        n_generations=args.n_generations,
        seed=args.seed,
        sampling_mode=args.sampling_mode,
        moment_trials=args.moment_trials,
        lengths=args.lengths,
        samples_per_length=args.samples_per_length,
        allow_ood_fixed_lengths=args.allow_ood_fixed_lengths,
    )


if __name__ == "__main__":
    main()
