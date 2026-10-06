#!/usr/bin/env python3
"""Audit and analyze frozen-checkpoint SCOPe length extrapolation cohorts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from evaluate_scope_training_designability import write_json
from summarize_scope_best import (
    audit_cohort,
    stratified_bootstrap,
    wilson_interval,
    write_csv,
)

MODELS = ("scratch", "joint_fc", "joint_properties_fc", "joint_no_fc")
LABELS = (
    "None",
    "Generation + structure",
    "Generation + structure + MLIP",
    "Generation + structure (no FC)",
)
GROUPS = {
    "129-256": (129, 256),
    "129-160": (129, 160),
    "161-192": (161, 192),
    "193-224": (193, 224),
    "225-256": (225, 256),
    "384": (384, 384),
    "512": (512, 512),
}


def audit_lengths(rows):
    expected = Counter(dict.fromkeys(range(129, 257), 8))
    expected.update({384: 64, 512: 64})
    if len(rows) != 1152 or Counter(r["length"] for r in rows) != expected:
        raise ValueError(
            "Expected 1024 uniform 129-256 samples and 64 at each stress length"
        )


def cluster_metrics(matrix, passing):
    """Recompute Table 3 clustering metrics on any exact cohort/subset."""
    from zevals.proteins.diversity import (
        _cluster_count_at_threshold,
        _cluster_curve_from_merges,
        _complete_linkage_merge_tms,
        _trapezoid_auc,
    )

    n = len(matrix)
    if n == 0:
        return {"n": 0, "diversity_auc": 0.0, "distinct_designable_clusters": 0}
    thresholds = np.linspace(0, 1, 101)
    merges = _complete_linkage_merge_tms(matrix)
    _, fractions = _cluster_curve_from_merges(
        n_total=n, merge_tms=merges, thresholds=thresholds
    )
    subset = matrix[np.ix_(passing, passing)]
    n_pass = len(subset)
    clusters = (
        0
        if n_pass == 0
        else _cluster_count_at_threshold(
            n_pass, _complete_linkage_merge_tms(subset), threshold=0.5
        )
    )
    return {
        "n": n,
        "diversity_auc": _trapezoid_auc(fractions, thresholds),
        "distinct_designable_clusters": int(clusters),
        "designable_cluster_yield": clusters / n,
    }


def metrics(args):
    import biotite.structure as struc
    import biotite.structure.io as strucio
    from aggregate_scope_self_consistency_shards import aggregate
    from compute_scope_sweep_novelty_diversity import compute_foldseek_novelty
    from zevals.proteins.diversity import compute_tmalign_complete_linkage_diversity

    root = args.root / args.model
    aggregate(
        root / "shards",
        root / "self_consistency_details.json",
        root / "self_consistency_refolds.csv",
    )
    raw = json.loads((root / "self_consistency_details.json").read_text())
    rows = audit_cohort(raw, json.loads((root / "inputs.json").read_text()))
    audit_lengths(rows)
    candidates = defaultdict(list)
    for r in raw["per_refold"]:
        candidates[r["sample"]].append(r)
    compute_foldseek_novelty(
        sc_root=root,
        target_db=args.train_db,
        target_name="scope2k_train",
        output_subdir=Path("novelty/scope2k_train"),
        query_set="all",
        foldseek_bin=Path(args.foldseek),
        tm_threshold=0.5,
        force=False,
    )
    hits = {
        r["query"]: r
        for r in csv.DictReader(
            (root / "novelty/scope2k_train/best_foldseek_hits.csv").open()
        )
    }
    assert set(hits) == {r["sample"] for r in rows}
    for r in rows:
        r["single_sequence_designability"] = float(
            np.mean([x["scRMSD"] < 1.5 for x in candidates[r["sample"]]])
        )
        hit = hits[r["sample"]]
        r["nearest_training_tm"] = float(hit["best_tm"]) if hit["best_target"] else None
        r["nearest_training_target"] = hit["best_target"]
        atoms = strucio.load_structure(r["backbone_pdb"])
        sse = struc.annotate_sse(atoms)
        assert len(sse) == r["length"]
        r["alpha_fraction"] = float(np.mean(sse == "a"))
        r["beta_fraction"] = float(np.mean(sse == "b"))
        meta = root / "shards" / r["shard"] / "generated" / (r["sample"] + ".json")
        m = json.loads(meta.read_text())["metrics"]
        r["n_chainbreaks"] = m.get("n_chainbreaks")
        r["max_ca_deviation"] = m.get("max_ca_deviation")
    write_csv(root / "per_backbone_metrics.csv", rows)
    matrices = {}
    # Avoid unnecessary cross-length comparisons involving the much longer stress samples.
    for group in ["129-256", "384", "512"]:
        lo, hi = GROUPS[group]
        subset = [r for r in rows if lo <= r["length"] <= hi]
        dest = root / "diversity" / group
        marker = dest / "metrics.json"
        if not marker.exists():
            result = compute_tmalign_complete_linkage_diversity(
                [Path(r["backbone_pdb"]) for r in subset],
                dest,
                tmalign_bin=args.tmalign,
                symmetrize="min",
                chunk_size=8,
                n_workers=args.workers,
            )
            assert result["tmalign_n_missing_pair_scores"] == 0
            write_json(marker, result)
        else:
            assert json.loads(marker.read_text())["tmalign_n_missing_pair_scores"] == 0
        with (dest / "tmalign_tm_matrix.csv").open() as f:
            reader = csv.reader(f)
            next(reader)
            data = list(reader)
        ids = [r[0].split("_", 2)[2] for r in data]
        matrix = np.array([r[1:] for r in data], dtype=float)
        assert set(ids) == {r["sample"] for r in subset} and np.isfinite(matrix).all()
        matrices[group] = (ids, matrix)
    summaries = {}
    for name, (lo, hi) in GROUPS.items():
        subset = [r for r in rows if lo <= r["length"] <= hi]
        ids, matrix = matrices[name if name in ["384", "512"] else "129-256"]
        index = {sample: i for i, sample in enumerate(ids)}
        keep = [index[r["sample"]] for r in subset]
        tm = matrix[np.ix_(keep, keep)]
        passing = np.array([bool(r["designable"]) for r in subset])
        result = cluster_metrics(tm, passing)
        result["designable_count"] = int(passing.sum())
        result["designability_wilson95"] = wilson_interval(
            int(passing.sum()), len(subset)
        )
        lengths = np.array([r["length"] for r in subset])
        result["metrics"] = {}
        for key in [
            "designable",
            "single_sequence_designability",
            "min_scRMSD",
            "max_scTM",
            "nearest_training_tm",
            "alpha_fraction",
            "beta_fraction",
        ]:
            if any(r[key] is None for r in subset):
                result["metrics"][key] = {
                    "unresolved_missing": sum(r[key] is None for r in subset)
                }
            else:
                mean, low, high = stratified_bootstrap(
                    np.array([r[key] for r in subset], float), lengths
                )
                result["metrics"][key] = {"mean": mean, "ci95": [low, high]}
        summaries[name] = result
    write_json(
        root / "evaluation_complete.json",
        {
            "n_backbones": len(rows),
            "n_refolds": len(raw["per_refold"]),
            "length_counts": dict(Counter(r["length"] for r in rows)),
            "missing_foldseek_hits": sum(
                not r["nearest_training_target"] for r in rows
            ),
            "groups": summaries,
        },
    )


def bootstrap(values, lengths, seed, draws=10000):
    rng = np.random.default_rng(seed)
    totals = np.zeros((draws, values.shape[1]))
    for length in np.unique(lengths):
        subset = values[lengths == length]
        indices = rng.integers(0, len(subset), size=(draws, len(subset)))
        totals += subset[indices].sum(axis=1)
    return totals / len(values)


def comparison(first, second, multiplicity, point_difference):
    difference = first - second
    return {
        "mean_difference": float(point_difference),
        "ci95": np.quantile(difference, [0.025, 0.975]).tolist(),
        "adjusted_ci95": np.quantile(
            difference, [0.025 / multiplicity, 1 - 0.025 / multiplicity]
        ).tolist(),
        "multiplicity": multiplicity,
    }


def report(args):
    results, samples, draws, baselines = {}, {}, {}, {}
    points = {}
    for index, model in enumerate(MODELS):
        root = args.root / model
        results[model] = json.loads((root / "evaluation_complete.json").read_text())
        samples[model] = list(
            csv.DictReader((root / "per_backbone_metrics.csv").open())
        )
        prior = json.loads(
            (args.baseline / model / "self_consistency_details.json").read_text()
        )
        specs = json.loads((args.baseline / model / "inputs.json").read_text())
        old = audit_cohort(prior, specs)
        candidates = defaultdict(list)
        for r in prior["per_refold"]:
            candidates[r["sample"]].append(r)
        for r in old:
            r["single_sequence_designability"] = np.mean(
                [x["scRMSD"] < 1.5 for x in candidates[r["sample"]]]
            )
        groups = {
            k: [r for r in samples[model] if lo <= int(r["length"]) <= hi]
            for k, (lo, hi) in GROUPS.items()
        }
        groups["97-128"] = [r for r in old if 97 <= r["length"] <= 128]
        groups["50-128"] = old
        draws[model] = {}
        points[model] = {}
        baselines[model] = {}
        for gi, (name, rows) in enumerate(groups.items()):
            lengths = np.array([int(r["length"]) for r in rows])
            values = np.array(
                [
                    [
                        float(r["designable"] in [True, "True"]),
                        float(r["single_sequence_designability"]),
                    ]
                    for r in rows
                ]
            )
            draws[model][name] = bootstrap(values, lengths, 20260919 + index * 100 + gi)
            points[model][name] = values.mean(axis=0)
            if name in ["50-128", "97-128"]:
                baselines[model][name] = {
                    "n": len(rows),
                    "means": values.mean(axis=0).tolist(),
                    "ci95": np.quantile(
                        draws[model][name], [0.025, 0.975], axis=0
                    ).T.tolist(),
                }
    effects = {}
    for model in MODELS[1:]:
        result = {}
        for group in GROUPS:
            # Two endpoints x three comparisons in the primary group; 36 in the six length bins.
            mult = 6 if group == "129-256" else 36
            result[group] = {
                key: comparison(
                    draws[model][group][:, col],
                    draws["scratch"][group][:, col],
                    mult,
                    points[model][group][col] - points["scratch"][group][col],
                )
                for col, key in enumerate(
                    ["designability", "single_sequence_designability"]
                )
            }
        # Positive interaction indicates a smaller drop from the near-boundary reference.
        result["primary_retention_interaction"] = {
            key: comparison(
                draws[model]["129-256"][:, col] - draws[model]["97-128"][:, col],
                draws["scratch"]["129-256"][:, col]
                - draws["scratch"]["97-128"][:, col],
                6,
                (points[model]["129-256"][col] - points[model]["97-128"][col])
                - (
                    points["scratch"]["129-256"][col] - points["scratch"]["97-128"][col]
                ),
            )
            for col, key in enumerate(
                ["designability", "single_sequence_designability"]
            )
        }
        effects[model] = result
    output = {
        "models": results,
        "in_distribution_baselines": baselines,
        "pretraining_vs_none": effects,
        "uncertainty": "10,000 independent within-length backbone bootstraps per model/cohort. Selected checkpoints fixed. Six primary comparisons (three pretrained variants, two endpoints) and 36 length-specific comparisons are separate Bonferroni families.",
        "limitations": [
            "Different selected fine-tuning steps; not a controlled matched-training-budget experiment.",
            "One training seed per configuration; uncertainty excludes training and checkpoint selection.",
            "Absolute RMSD threshold can penalize long proteins more; scTM is supplementary.",
            "Nearest-training TM is confounded by the length mismatch against training domains <=128; low values alone do not establish novelty.",
            "Diversity/cluster counts depend on sample count; compare configurations within the same length group.",
        ],
    }
    write_json(args.root / "results.json", output)
    np.savez(
        args.root / "bootstrap_draws.npz",
        **{f"{m}_{g}": d for m, groups in draws.items() for g, d in groups.items()},
    )
    lines = [
        "# SCOPe length extrapolation at Table 3 checkpoints",
        "",
        "Each model has 1,024 primary samples (eight at every integer length 129–256), plus 64 samples each at lengths 384 and 512. The separate 16-sample feasibility pilot is excluded. All eight SolubleMPNN sequences per backbone are refolded with the unchanged ESMFold2 protocol.",
        "",
        "Intervals use within-length backbone bootstraps, except best-of-eight designability at stress lengths 384 and 512, which uses Wilson intervals. Single-sequence bootstrap intervals are unavailable when every backbone has zero successes; zero observed successes does not establish a zero population success rate. Single-sequence results describe a uniformly selected SolubleMPNN candidate, not a directly predicted Zatom-2 sequence.",
        "",
        "| Length | Initialization | N | Designability (95% CI) | Single-sequence designability (95% CI) | Mean best-of-8 scRMSD | Mean best-of-8 scTM | Distinct designable clusters | Diversity AUC | Nearest-training TM |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        for model, label in zip(MODELS, LABELS, strict=True):
            r = results[model]["groups"][group]
            m = r["metrics"]
            d = m["designable"]
            s = m["single_sequence_designability"]
            tm = m["nearest_training_tm"].get("mean")
            d_ci = r["designability_wilson95"] if group in ["384", "512"] else d["ci95"]
            single = (
                "0.00% (bootstrap CI unavailable)"
                if r["designable_count"] == 0
                else f"{100*s['mean']:.2f}% [{100*s['ci95'][0]:.2f}, {100*s['ci95'][1]:.2f}]"
            )
            lines.append(
                f"| {group} | {label} | {r['n']} | {100*d['mean']:.2f}% [{100*d_ci[0]:.2f}, {100*d_ci[1]:.2f}] | {single} | {m['min_scRMSD']['mean']:.3f} | {m['max_scTM']['mean']:.4f} | {r['distinct_designable_clusters']} | {r['diversity_auc']:.4f} | {f'{tm:.4f}' if tm is not None else 'unresolved'} |"
            )
    lines.extend(
        [
            "",
            "## Primary comparisons against no pretraining",
            "",
            "Differences are percentage points; adjusted intervals cover the six primary comparisons. Positive retention interactions indicate smaller degradation from lengths 97–128 relative to scratch. They do not by themselves imply higher extrapolation performance.",
            "",
            "| Initialization | Endpoint | Difference at 129–256 | Adjusted 95% CI | Retention interaction | Adjusted 95% CI |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for model, label in zip(MODELS[1:], LABELS[1:], strict=True):
        for key in ["designability", "single_sequence_designability"]:
            r = effects[model]["129-256"][key]
            it = effects[model]["primary_retention_interaction"][key]
            lines.append(
                f"| {label} | {key} | {100*r['mean_difference']:+.2f} | [{100*r['adjusted_ci95'][0]:+.2f}, {100*r['adjusted_ci95'][1]:+.2f}] | {100*it['mean_difference']:+.2f} | [{100*it['adjusted_ci95'][0]:+.2f}, {100*it['adjusted_ci95'][1]:+.2f}] |"
            )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            *["- " + s for s in output["limitations"]],
            "",
            output["uncertainty"],
            "",
            "All original generation artifacts, designed sequences, refolds and individual scores are retained. Point estimates are conditional on the fixed Table 3 checkpoints and sampling settings. No direct-sequence or Protein FID experiments are included.",
        ]
    )
    (args.root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    groups = ["97-128", "129-160", "161-192", "193-224", "225-256", "384", "512"]
    xs = np.array([112.5, 144.5, 176.5, 208.5, 240.5, 384, 512])
    for i, (model, label) in enumerate(zip(MODELS, LABELS, strict=True)):
        for col, ax in enumerate(axes):
            boot = [draws[model][g][:, col] for g in groups]
            ys = np.array([points[model][g][col] for g in groups]) * 100
            intervals = np.array([np.quantile(b, [0.025, 0.975]) for b in boot]) * 100
            if col == 0:
                for j, group in enumerate(groups):
                    if group in ["384", "512"]:
                        intervals[j] = (
                            np.array(
                                results[model]["groups"][group][
                                    "designability_wilson95"
                                ]
                            )
                            * 100
                        )
            ax.plot(xs, ys, "o-", label=label)
            ax.fill_between(xs, intervals[:, 0], intervals[:, 1], alpha=0.12)
    for ax, title in zip(
        axes,
        ["Best-of-eight designability", "Single-sequence designability"],
        strict=True,
    ):
        ax.axvline(128, color="black", ls="--", lw=1)
        ax.set_xlabel("Protein length (residues)")
        ax.set_ylabel("Success (%)")
        ax.set_title(title)
        ax.set_ylim(-2, 102)
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.root / "comparison.png", dpi=180)
    fig.savefig(args.root / "comparison.pdf")
    write_json(
        args.root / "validation.json",
        {
            "complete": True,
            "n_models": 4,
            "n_backbones": 4608,
            "n_refolds": 36864,
            "primary_backbones_per_model": 1024,
            "stress_per_model": 128,
            "pilot_excluded": True,
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["metrics", "report"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--train-db", type=Path)
    p.add_argument("--foldseek")
    p.add_argument("--tmalign", default="TMalign")
    p.add_argument("--workers", type=int, default=24)
    args = p.parse_args()
    globals()[args.action](args)


if __name__ == "__main__":
    main()
