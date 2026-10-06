"""
Command-line driver for AFD (Atomistic Fréchet Distance).

Two subcommands:
  - `score` — compute the score for one or more generation sets vs. a reference.
  - `cache` — pre-compute and cache the (mu, cov) of a reference set.

A `--ref` / `--gen` spec can be:
  - a path to a `.pt` file of dict-style records (qm9_gens-style),
  - a directory of `.pt` files (concatenated),
  - a `qm9[:root]` shortcut (loads via `torch_geometric.datasets.QM9`),
  - any custom scheme registered via
    `zevals.atomistic.afd.data.register_loader`.

The reference (mu, cov) is cached at `--ref-cache` so re-scoring new
generations skips the (expensive) reference embedding pass.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional

import torch

from . import data as data_mod
from .features import compute_features
from .score import frechet_distance, gaussian_stats


def _build_model_and_transform(model_name: str, cache_dir: str, device: str):
    # Deferred so `--help` doesn't pay the import cost.
    from .models import AddStandardKeys, load_pretrained_ct

    model = load_pretrained_ct(model_name, cache_dir=cache_dir, device=device).eval()
    return model, AddStandardKeys()


def _compute_ref_stats(args, model, transform):
    """Compute or load the reference (mu, cov)."""
    if args.ref_cache and os.path.exists(args.ref_cache):
        print(f"[ref] loading cached stats from {args.ref_cache}")
        blob = torch.load(args.ref_cache, weights_only=False)
        return blob["mu"], blob["cov"], blob.get("n_samples")

    print(f"[ref] resolving ref={args.ref!r} (limit={args.ref_n})")
    records = data_mod.resolve_records(args.ref, limit=args.ref_n)
    print(f"[ref] {len(records)} records; computing features on {args.device}")
    t0 = time.time()
    feats = compute_features(
        records,
        model,
        transform=transform,
        batch_size=args.batch_size,
        device=args.device,
        desc="ref features",
    )
    print(f"[ref] features shape={tuple(feats.shape)} ({time.time() - t0:.1f}s)")
    mu, cov = gaussian_stats(feats)

    if args.ref_cache:
        os.makedirs(os.path.dirname(args.ref_cache) or ".", exist_ok=True)
        torch.save({"mu": mu, "cov": cov, "n_samples": len(records)}, args.ref_cache)
        print(f"[ref] cached -> {args.ref_cache}")
    return mu, cov, len(records)


def cmd_cache(args):
    model, transform = _build_model_and_transform(
        args.model_name, args.hf_cache_dir, args.device
    )
    _compute_ref_stats(args, model, transform)


def cmd_score(args):
    model, transform = _build_model_and_transform(
        args.model_name, args.hf_cache_dir, args.device
    )

    mu_r, cov_r, n_ref = _compute_ref_stats(args, model, transform)

    print("\n=== Scores ===")
    print(f"reference: {args.ref!r}, N={n_ref}, feat_dim={mu_r.shape[0]}")
    rows = []
    for path in args.gen:
        print(f"\n[gen] resolving {path!r}")
        records = data_mod.resolve_records(path, limit=args.gen_n)
        print(f"[gen] {len(records)} records")
        t0 = time.time()
        feats = compute_features(
            records,
            model,
            transform=transform,
            batch_size=args.batch_size,
            device=args.device,
            desc=f"gen {os.path.basename(path)}",
        )
        mu_g, cov_g = gaussian_stats(feats)
        score_value = frechet_distance(mu_r, cov_r, mu_g, cov_g)
        dt = time.time() - t0
        n_gen = feats.shape[0]
        label = os.path.basename(path) if path.endswith(".pt") else path
        print(
            f"[gen] {label}: score = {score_value:.4f} "
            f"(N_gen={n_gen}, dt={dt:.1f}s)"
        )
        rows.append((label, n_gen, score_value))

    print("\n=== Summary ===")
    print(f"{'source':<60}{'N_gen':>8}{'score':>14}")
    for name, n, val in rows:
        print(f"{name:<60}{n:>8}{val:>14.4f}")


def _add_common_args(p: argparse.ArgumentParser):
    p.add_argument(
        "--ref",
        default="qm9:./tmp/qm9",
        help="Reference spec: path to .pt, directory of .pt files, or "
        "registered scheme like 'qm9' or 'qm9:./tmp/qm9'. "
        "Default: qm9:./tmp/qm9.",
    )
    p.add_argument(
        "--ref-n",
        type=int,
        default=None,
        help="Cap the reference set at the first N records.",
    )
    p.add_argument(
        "--ref-cache",
        default="./tmp/ref_stats.pt",
        help="Where to cache the reference (mu, cov). " "Set to '' to disable.",
    )
    p.add_argument(
        "--model-name",
        default="ct-scd-pcq",
        help="HuggingFace model id under Ty-Perez/ (default: ct-scd-pcq).",
    )
    p.add_argument(
        "--hf-cache-dir",
        default="./experiments",
        help="Where huggingface_hub caches the checkpoint.",
    )
    p.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for the CT forward pass."
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Device for the model (cpu / cuda / cuda:0 / ...).",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m zevals.atomistic.afd",
        description="Fréchet-distance scoring for atomistic generative models.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_score = sub.add_parser("score", help="Score one or more generation files.")
    _add_common_args(p_score)
    p_score.add_argument(
        "--gen",
        nargs="+",
        required=True,
        help="One or more generation specs (.pt file / dir / scheme).",
    )
    p_score.add_argument(
        "--gen-n",
        type=int,
        default=None,
        help="Cap each generation set at the first N records.",
    )
    p_score.set_defaults(func=cmd_score)

    p_cache = sub.add_parser("cache", help="Pre-compute the reference stats only.")
    _add_common_args(p_cache)
    p_cache.set_defaults(func=cmd_cache)

    return p


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
