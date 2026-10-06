#!/usr/bin/env python3
"""Evaluate frozen SCOPe checkpoints on an evenly sampled length cohort.

The study manifest fixes checkpoint selection and sample IDs before generation.
Uses the existing Zatom-2 generator and SolubleMPNN/ESMFold2 evaluation protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sample_seed(model: str, sample: str) -> int:
    """Stable seeds independent of worker assignment and interrupted runs."""
    return int.from_bytes(
        hashlib.sha256(f"scope-best-20260919/{model}/{sample}".encode()).digest()[:4]
    )


def generate(args: argparse.Namespace) -> None:
    """Generate one shard, seeding each sample after model initialization."""
    import numpy as np
    import torch
    from evaluate_scope_training_designability import write_json
    from extract_protein_backbones import extract_backbones
    from hydra import compose, initialize_config_dir
    from zatom2.run_inference import build_inference_engine

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    manifest = json.loads((args.root / "manifest.json").read_text())
    model = next(m for m in manifest["models"] if m["name"] == args.model)
    shard = args.root / args.model / "shards" / f"shard_{args.shard:02d}"
    specs = json.loads((shard / "inputs.json").read_text())
    generated = shard / "generated"
    generated.mkdir(exist_ok=True)
    with initialize_config_dir(
        config_dir=str(REPO / "models/zatom2/configs"), version_base="1.3"
    ):
        cfg = compose(
            config_name="inference",
            overrides=[
                f"ckpt_path={model['checkpoint']}",
                f"inputs={shard / 'inputs.json'}",
                f"out_dir={generated}",
                "seed=20260919",
                "global_prefix=",
                "diffusion_batch_size=1",
                "n_batches=1",
                "prevalidate_inputs=True",
                "dump_trajectories=False",
                "save_cifs=True",
                "inference_sampler.num_timesteps=200",
                "inference_sampler.step_scale=1.5",
                "inference_sampler.gamma_0=0.6",
            ],
        )
    engine = build_inference_engine(cfg)
    engine.initialize()
    for index, (sample, spec) in enumerate(specs.items()):
        marker = generated / f"{sample}.complete.json"
        if not marker.exists():
            seed = sample_seed(args.model, sample)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            engine.seed = seed
            # A separate file preserves inputless-protein coercion and naming.
            one_input = shard / "current_input.json"
            write_json(one_input, {sample: spec})
            engine.skip_existing = False
            start = time.monotonic()
            engine.run(inputs=str(one_input), n_batches=1, out_dir=str(generated))
            paths = list(generated.glob(f"{sample}_0_model_0.cif*"))
            if len(paths) != 1:
                raise RuntimeError(
                    f"Expected one generated structure: {sample}: {paths}"
                )
            write_json(marker, dict(seed=seed, seconds=time.monotonic() - start))
        print(f"generated {index + 1}/{len(specs)} {sample}", flush=True)
    extract_backbones(generated, shard / "backbones", overwrite=True)
    validate_backbones(shard)


def validate_backbones(shard: Path) -> None:
    """Reject missing, duplicate, truncated, or nonfinite generated backbones."""
    import numpy as np
    from Bio.PDB import PDBParser

    specs = json.loads((shard / "inputs.json").read_text())
    paths = sorted((shard / "backbones").glob("*.pdb"))
    expected = {f"{key}_0_model_0": val["length"] for key, val in specs.items()}
    if {p.stem for p in paths} != set(expected):
        raise ValueError("Generated IDs do not exactly match the frozen cohort")
    for path in paths:
        structure = PDBParser(QUIET=True).get_structure(path.stem, path)
        residues = list(structure[0].get_residues())
        if len(residues) != expected[path.stem]:
            raise ValueError(f"Incorrect residue count: {path}")
        if not all(all(atom in r for atom in ("N", "CA", "C")) for r in residues):
            raise ValueError(f"Incomplete backbone: {path}")
        if not np.isfinite([a.coord for a in structure.get_atoms()]).all():
            raise ValueError(f"Nonfinite coordinates: {path}")


def refold(args: argparse.Namespace) -> None:
    """Reuse the validated, resumable training-domain oracle runner."""
    from evaluate_scope_training_designability import run

    output = args.root / args.model
    validate_backbones(output / "shards" / f"shard_{args.shard:02d}")
    run(
        argparse.Namespace(
            output=output,
            shard=args.shard,
            model=args.oracle_model,
            tmalign=args.tmalign,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["generate", "refold"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--oracle-model", type=Path)
    parser.add_argument("--tmalign", default="TMalign")
    args = parser.parse_args()
    {"generate": generate, "refold": refold}[args.action](args)


if __name__ == "__main__":
    main()
