#!/usr/bin/env python3
"""Generate SCOPe samples with RNG streams independent of sharding and resume."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
from pathlib import Path


def validate_ccd_mirror(environment: dict[str, str]) -> Path:
    """Reject missing CCD paths before imports capture AtomWorks constants."""
    value = environment.get("CCD_MIRROR_PATH")
    if not value or not Path(value).is_dir() or not os.access(value, os.R_OK | os.X_OK):
        raise ValueError(f"Missing readable CCD_MIRROR_PATH: {value!r}")
    return Path(value).resolve()


def configure_environment() -> None:
    """Keep this explicitly configured sampling process independent of .env."""
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    validate_ccd_mirror(os.environ)


def sample_seed(seed: int, sample: str, stage: str = "generation") -> int:
    """Derive a uint32 RNG seed from the replicate, stage, and stable sample ID."""
    payload = json.dumps(["scope-sampling-v1", stage, seed, sample]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def seed_rng(seed: int) -> None:
    """Seed Python, NumPy, and Torch after model initialization."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, value: object) -> None:
    """Atomically publish provenance or a completed-sample marker."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def generate_samples(engine, specs: dict, output: Path, seed: int) -> None:
    """Resume complete samples; rerun interrupted samples from their own seed.

    Completion markers bind the input, RNG seed, and CIF checksum. Existing
    committed outputs that no longer match are rejected rather than silently
    mixing regenerated backbones with cached oracle results.
    """
    output.mkdir(parents=True, exist_ok=True)
    one_input = output.parent / "seeded_input.json"
    for sample, spec in sorted(specs.items()):
        if Path(sample).name != sample or sample in {".", ".."}:
            raise ValueError(f"Unsafe sample ID: {sample!r}")
        derived_seed = sample_seed(seed, sample)
        stem = f"inputs_{sample}_0_model_0"
        marker = output / f"{stem}.complete.json"
        identity = {"sampling_seed": seed, "seed": derived_seed, "input": spec}
        if marker.exists():
            saved = json.loads(marker.read_text())
            cif = output / saved["cif"]
            if (
                any(saved.get(key) != value for key, value in identity.items())
                or not cif.is_file()
                or file_hash(cif) != saved["sha256"]
            ):
                raise ValueError(f"Completed sample changed: {marker}")
            continue
        seed_rng(derived_seed)
        engine.seed = derived_seed
        engine.skip_existing = False
        write_json(one_input, {sample: spec})
        engine.run(inputs=str(one_input), n_batches=1, out_dir=str(output))
        cifs = [
            p
            for p in (output / f"{stem}.cif", output / f"{stem}.cif.gz")
            if p.is_file()
        ]
        if len(cifs) != 1 or not cifs[0].stat().st_size:
            raise RuntimeError(f"Expected one nonempty CIF for {stem}: {cifs}")
        write_json(
            marker, {**identity, "cif": cifs[0].name, "sha256": file_hash(cifs[0])}
        )
        print(f"Generated {sample}, seed={derived_seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", choices=["zatom2"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-timesteps", type=int, default=200)
    parser.add_argument("--step-scale", type=float, default=1.5)
    parser.add_argument("--gamma-0", type=float, default=0.6)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be a uint32")

    configure_environment()

    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    repo = Path(__file__).resolve().parents[1]
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    contract = {
        "version": 1,
        "package": args.package,
        "checkpoint": str(checkpoint),
        "checkpoint_size": checkpoint.stat().st_size,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "inputs_sha256": file_hash(args.inputs),
        "sampling_seed": args.seed,
        "num_timesteps": args.num_timesteps,
        "step_scale": args.step_scale,
        "gamma_0": args.gamma_0,
        "ccd_mirror_path": str(Path(os.environ["CCD_MIRROR_PATH"]).resolve()),
    }
    contract_path = args.output / "sampling_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError(f"Sampling contract changed: {contract_path}")
    else:
        if any(args.output.iterdir()):
            raise ValueError(f"Seeded generation requires a new output: {args.output}")
        write_json(contract_path, contract)

    # Build once per shard. Each draw is reseeded AFTER engine initialization.
    module = importlib.import_module(f"{args.package}.run_inference")
    with initialize_config_dir(
        config_dir=str(repo / "models" / args.package / "configs"), version_base="1.3"
    ):
        cfg = compose(config_name="inference")
    cfg = OmegaConf.merge(
        cfg,
        {
            "ckpt_path": str(checkpoint),
            "inputs": str(args.inputs),
            "out_dir": str(args.output),
            "seed": args.seed,
            "global_prefix": "inputs_",
            "diffusion_batch_size": 1,
            "n_batches": 1,
            "prevalidate_inputs": True,
            "dump_trajectories": False,
            "save_cifs": True,
            "inference_sampler": {
                "num_timesteps": args.num_timesteps,
                "step_scale": args.step_scale,
                "gamma_0": args.gamma_0,
            },
        },
    )
    engine = module.build_inference_engine(cfg)
    engine.initialize()
    generate_samples(
        engine, json.loads(args.inputs.read_text()), args.output, args.seed
    )


if __name__ == "__main__":
    main()
