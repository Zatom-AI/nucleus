#!/usr/bin/env python3
"""Re-aggregate cached SCOPe sweep results with Genie-style metrics.

This intentionally does not rerun ProteinMPNN, ESMFold, FoldSeek, or TM-align.
It reads the cached self-consistency records, the pLDDT values in the cached
prediction CIFs, the cached FoldSeek best-hit table, and the cached TM-align
matrix.  The resulting metrics are therefore a protocol comparison, not a
second oracle run.

Genie-style designability follows the paper's selection rules:

* short: select the refold with the highest scTM, then require scTM > 0.5
  and that refold's pLDDT > 70;
* long: select the refold with the lowest scRMSD, then require scRMSD < 2
  and that refold's pLDDT > 70.

For comparison, the output also contains the existing Zatom2 scRMSD < 1.5
designability and "any refold" variants.  Novelty uses the cached FoldSeek
max-TM values against the SCOPe-2k training database, while diversity uses
Genie's single-linkage TM >= 0.6 cluster fraction on the cached, min-directional
TM-align matrix.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def _as_float(value: object) -> float:
    return float(value) if value not in (None, "") else math.nan


def _mean(values: Iterable[float]) -> float:
    values = [x for x in values if math.isfinite(x)]
    return sum(values) / len(values) if values else math.nan


def _read_plddt(path: Path) -> float:
    """Read mean CA pLDDT from an ESMFold2 mmCIF or a PDB file."""

    atom_columns = []  # type: List[str]
    values = []  # type: List[float]
    with path.open() as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("_atom_site."):
                atom_columns.append(line.split()[0])
                continue
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            fields = line.split()
            if atom_columns:
                try:
                    atom_idx = atom_columns.index("_atom_site.label_atom_id")
                except ValueError:
                    atom_idx = atom_columns.index("_atom_site.auth_atom_id")
                try:
                    b_idx = atom_columns.index("_atom_site.B_iso_or_equiv")
                except ValueError:
                    b_idx = -1
                if b_idx >= 0 and atom_idx < len(fields) and fields[atom_idx] == "CA":
                    values.append(float(fields[b_idx]))
            elif len(line) >= 66 and line[12:16].strip() == "CA":
                values.append(float(line[60:66]))

    if not values:
        raise ValueError(f"no CA pLDDT values found in {path}")
    return _mean(values)


def _read_plddt_many(paths: Iterable[Path]) -> Dict[str, float]:
    """Read CA confidence from cached mmCIF or PDB files.

    Use the same schema-aware parser as the single-file audit path. The old
    ``rg`` fast path happened to match the current ESMFold2 mmCIF column order,
    but silently omitted fixed-width PDB records and could misread mmCIF files
    whose atom-site columns were reordered.
    """

    return {str(path): _read_plddt(path) for path in paths}


def _union_find(n: int, edges: Iterable[Tuple[int, int]]) -> int:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for left, right in edges:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left
    return len({find(i) for i in range(n)})


def _load_tm_matrix(path: Path) -> Tuple[List[str], List[List[float]]]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return [], []
    names = rows[0][1:]
    matrix = []
    for row in rows[1:]:
        if len(row) < len(names) + 1:
            continue
        matrix.append([_as_float(value) for value in row[1 : len(names) + 1]])
    if len(matrix) != len(names):
        raise ValueError(f"non-square TM matrix: {path}")
    return names, matrix


def _single_linkage_fraction(
    names: List[str],
    matrix: List[List[float]],
    selected: Optional[Set[str]],
    threshold: float,
) -> Tuple[float, int, int]:
    indices = [
        i for i, name in enumerate(names) if selected is None or name in selected
    ]
    if not indices:
        return math.nan, 0, 0
    edges = (
        (left_pos, right_pos)
        for left_pos, left in enumerate(indices)
        for right_pos, right in enumerate(indices[left_pos + 1 :], left_pos + 1)
        if math.isfinite(matrix[left][right]) and matrix[left][right] >= threshold
    )
    n_clusters = _union_find(len(indices), edges)
    return n_clusters / len(indices), n_clusters, len(indices)


def _sample_id_map(samples: List[str], matrix_names: List[str]) -> Dict[str, str]:
    """Map self-consistency sample names to the matrix's prefixed names."""
    result = {}
    for name in matrix_names:
        for sample in samples:
            if name.endswith("_" + sample):
                result[sample] = name
                break
    return result


def _read_details(
    root: Path,
) -> Tuple[Dict[str, List[Dict[str, float]]], List[Dict[str, Any]]]:
    with (root / "self_consistency_details.json").open() as handle:
        details = json.load(handle)
    refs = {}  # type: Dict[str, List[Dict[str, float]]]
    for row in details["per_refold"]:
        prediction = Path(row["prediction_pdb"])
        if not prediction.exists():
            raise FileNotFoundError(prediction)
        refold = {
            "scRMSD": _as_float(row.get("scRMSD")),
            "scTM": _as_float(row.get("scTM")),
            "pLDDT": math.nan,
            "prediction_pdb": str(prediction),
        }
        refs.setdefault(row["sample"], []).append(refold)

    # The paper's selected Genie metrics only need pLDDT for the best-scTM
    # and best-scRMSD refolds.  For the optional any-refold diagnostics, also
    # read candidates that could pass either structural threshold.  This
    # avoids parsing every cached CIF in a multi-checkpoint sweep.
    for candidates in refs.values():
        best_tm = max(candidates, key=lambda item: item["scTM"])
        best_rmsd = min(candidates, key=lambda item: item["scRMSD"])
        to_read = {
            id(item) for item in candidates if item["scTM"] > 0.5 or item["scRMSD"] < 2
        }
        to_read.update((id(best_tm), id(best_rmsd)))
        for item in candidates:
            if id(item) in to_read:
                item["_needs_plddt"] = 1

    selected_paths = [
        Path(item["prediction_pdb"])
        for candidates in refs.values()
        for item in candidates
        if item.get("_needs_plddt")
    ]
    plddt_by_path = _read_plddt_many(selected_paths)
    for candidates in refs.values():
        for item in candidates:
            if item.get("_needs_plddt"):
                item["pLDDT"] = plddt_by_path.get(item["prediction_pdb"], math.nan)
    return refs, details["per_backbone"]


def _read_foldseek(root: Path) -> Dict[str, float]:
    candidates = sorted((root / "novelty").glob("*/best_foldseek_hits.csv"))
    if not candidates:
        raise FileNotFoundError(f"best_foldseek_hits.csv under {root / 'novelty'}")
    with candidates[0].open(newline="") as handle:
        return {
            row["query"]: _as_float(row["best_tm"]) for row in csv.DictReader(handle)
        }


def _fraction(values: Iterable[bool]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def _f1(left: float, right: float) -> float:
    if not (math.isfinite(left) and math.isfinite(right) and left + right):
        return math.nan
    return 2.0 * left * right / (left + right)


def evaluate_row(summary_row: Dict[str, str]) -> Dict[str, Any]:
    root = Path(summary_row["output_root"])
    refs, backbones = _read_details(root)
    samples = [row["sample"] for row in backbones]

    per_sample = {}  # type: Dict[str, Dict[str, Any]]
    for sample in samples:
        candidates = refs.get(sample, [])
        if not candidates:
            raise ValueError(f"no refolds for {sample} in {root}")
        best_tm = max(candidates, key=lambda item: item["scTM"])
        best_rmsd = min(candidates, key=lambda item: item["scRMSD"])
        per_sample[sample] = {
            "zatom_1p5": any(item["scRMSD"] < 1.5 for item in candidates),
            "short_any": any(
                item["scTM"] > 0.5 and item["pLDDT"] > 70 for item in candidates
            ),
            "long_any": any(
                item["scRMSD"] < 2 and item["pLDDT"] > 70 for item in candidates
            ),
            "short_selected": best_tm["scTM"] > 0.5 and best_tm["pLDDT"] > 70,
            "long_selected": best_rmsd["scRMSD"] < 2 and best_rmsd["pLDDT"] > 70,
            "best_tm": best_tm,
            "best_rmsd": best_rmsd,
        }

    foldseek = _read_foldseek(root)
    novelty = {sample: foldseek.get(sample, math.nan) for sample in samples}
    valid_novelty = [value for value in novelty.values() if math.isfinite(value)]

    matrix_path = (
        root / "diversity" / "tmalign_complete_linkage_all" / "tmalign_tm_matrix.csv"
    )
    matrix_names, matrix = _load_tm_matrix(matrix_path)
    matrix_ids = _sample_id_map(samples, matrix_names)
    if len(matrix_ids) != len(samples):
        missing = sorted(set(samples) - set(matrix_ids))
        raise ValueError(f"matrix/sample mismatch in {root}; missing {missing[:3]}")

    def selected_ids(field: str) -> Set[str]:
        return {
            matrix_ids[sample] for sample, value in per_sample.items() if value[field]
        }

    all_single, all_clusters, all_n = _single_linkage_fraction(
        matrix_names, matrix, None, threshold=0.6
    )
    short_single, short_clusters, short_n = _single_linkage_fraction(
        matrix_names, matrix, selected_ids("short_selected"), threshold=0.6
    )
    long_single, long_clusters, long_n = _single_linkage_fraction(
        matrix_names, matrix, selected_ids("long_selected"), threshold=0.6
    )

    def novelty_fields(field: str) -> Dict[str, float]:
        selected = [sample for sample in samples if per_sample[sample][field]]
        values = [
            novelty[sample] for sample in selected if math.isfinite(novelty[sample])
        ]
        return {
            f"{field}_n": len(selected),
            f"{field}_novel_count": sum(value < 0.5 for value in values),
            f"{field}_novel_rate_conditional": _fraction(
                value < 0.5 for value in values
            ),
            f"{field}_novel_rate_all": sum(value < 0.5 for value in values)
            / len(samples)
            if samples
            else math.nan,
            f"{field}_mean_best_train_tm": _mean(values),
        }

    result = {  # type: Dict[str, Any]
        "step": summary_row.get("step", ""),
        "step_name": summary_row.get("step_name", ""),
        "run_tag": summary_row.get("run_tag", ""),
        "output_root": str(root),
        "n_backbones": len(samples),
        "n_refolds": sum(len(refs[sample]) for sample in samples),
        "zatom_designability_1p5": _fraction(
            per_sample[sample]["zatom_1p5"] for sample in samples
        ),
        "genie_short_designability_selected": _fraction(
            per_sample[sample]["short_selected"] for sample in samples
        ),
        "genie_short_designability_any_refold": _fraction(
            per_sample[sample]["short_any"] for sample in samples
        ),
        "genie_long_designability_selected": _fraction(
            per_sample[sample]["long_selected"] for sample in samples
        ),
        "genie_long_designability_any_refold": _fraction(
            per_sample[sample]["long_any"] for sample in samples
        ),
        "genie_short_mean_selected_pLDDT": _mean(
            per_sample[sample]["best_tm"]["pLDDT"] for sample in samples
        ),
        "genie_long_mean_selected_pLDDT": _mean(
            per_sample[sample]["best_rmsd"]["pLDDT"] for sample in samples
        ),
        "foldseek_n_queries": len(valid_novelty),
        "foldseek_mean_best_train_tm_all": _mean(valid_novelty),
        "foldseek_novel_rate_all": _fraction(value < 0.5 for value in valid_novelty),
        "genie_single_linkage_tm06_cluster_fraction_all": all_single,
        "genie_single_linkage_tm06_n_clusters_all": all_clusters,
        "genie_single_linkage_tm06_n_all": all_n,
        "genie_single_linkage_tm06_cluster_fraction_short_conf": short_single,
        "genie_single_linkage_tm06_n_clusters_short_conf": short_clusters,
        "genie_single_linkage_tm06_n_short_conf": short_n,
        "genie_single_linkage_tm06_cluster_fraction_long_conf": long_single,
        "genie_single_linkage_tm06_n_clusters_long_conf": long_clusters,
        "genie_single_linkage_tm06_n_long_conf": long_n,
        "genie_long_f1_all": _f1(
            _fraction(per_sample[sample]["long_selected"] for sample in samples),
            all_single,
        ),
        "genie_short_f1_all": _f1(
            _fraction(per_sample[sample]["short_selected"] for sample in samples),
            all_single,
        ),
    }
    result.update(novelty_fields("short_selected"))
    result.update(novelty_fields("long_selected"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", action="append", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = []  # type: List[Dict[str, Any]]
    for summary_path in args.summary_csv:
        with summary_path.open(newline="") as handle:
            summary_rows = list(csv.DictReader(handle))
        for row in summary_rows:
            if row.get("complete", "1") not in {"1", "True", "true"}:
                continue
            result = evaluate_row(row)
            prefix = "paper_table4_"
            result["sweep"] = (
                summary_path.parent.name[len(prefix) :]
                if summary_path.parent.name.startswith(prefix)
                else summary_path.parent.name
            )
            rows.append(result)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["sweep"]
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = args.output_csv.with_suffix(".json")
    manifest.write_text(
        json.dumps(
            {
                "metric_protocol": "Genie-style cached reaggregation",
                "summary_csv": [str(path) for path in args.summary_csv],
                "n_checkpoint_rows": len(rows),
                "designability": {
                    "short": "argmax scTM; scTM > 0.5 and selected mean pLDDT > 70",
                    "long": "argmin scRMSD; scRMSD < 2 and selected mean pLDDT > 70",
                    "zatom_reference": "any refold with CA scRMSD < 1.5",
                },
                "novelty": "cached FoldSeek max TM against SCOPe-2k train; threshold < 0.5",
                "diversity": "cached min-directional TM-align; Genie single-linkage at TM >= 0.6",
                "note": "pLDDT is from cached ESMFold2 CIFs, so this is protocol-faithful but not an identical Genie ESMFold-v1 oracle run.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {args.output_csv} ({len(rows)} checkpoint rows)")


if __name__ == "__main__":
    main()
