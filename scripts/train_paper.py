#!/usr/bin/env python3
"""Launch a paper experiment with the manuscript's effective optimizer batch.

Does not request cluster allocations. Run inside your allocated resources. Use
--dry-run to inspect the command; additional Hydra overrides follow `--`.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command(args: argparse.Namespace, overrides: list[str]) -> list[str]:
    config = ROOT / "models/zatom2/configs/experiment" / f"{args.experiment}.yaml"
    if not config.is_file() or not (
        args.experiment.startswith("paper-") or args.experiment == "train-geom-drugs"
    ):
        raise ValueError(f"Unknown paper experiment: {args.experiment}")
    world = args.nodes * args.devices
    scope = args.experiment == "paper-table4-scope"
    reduced = any(
        k in args.experiment
        for k in ["zatom2-l", "paper-table2", "finetune-joint-properties"]
    )
    target_world = 16 if scope else (128 if reduced else 64)
    if world <= 0 or target_world % world:
        raise ValueError(f"GPU count must be a positive divisor of {target_world}")
    accumulation = target_world // world
    cmd = [
        sys.executable,
        "-m",
        "zatom2.train",
        f"experiment={args.experiment}",
        "trainer_runtime=ddp",
        f"trainer.num_nodes={args.nodes}",
        f"trainer.devices_per_node={args.devices}",
        f"trainer.grad_accum_steps={accumulation}",
        "logger=csv",
    ]
    if scope:
        cmd += [
            f"trainer.n_examples_per_epoch={50000 * 4 * target_world}",
            f"trainer.limit_train_batches={50000 * accumulation}",
            "trainer.checkpoint_every_n_steps=2500",
            "trainer.max_epochs=1",
        ]
    if args.experiment == "train-geom-drugs":
        cmd += ["trainer.max_epochs=80"]
    if args.checkpoint:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        cmd += [f"ckpt_path={args.checkpoint.resolve()}"]
    if args.bootstrap:
        if not args.checkpoint:
            raise ValueError(
                "--bootstrap requires --checkpoint with an epoch checkpoint"
            )
        if not args.checkpoint.name.startswith("epoch-"):
            raise ValueError(
                "Bootstrap from an epoch checkpoint; step checkpoints restore mid-epoch loader state"
            )
        cmd += [
            "reset_optimizer_on_resume=true",
            "trainer.allow_world_size_change_on_resume=true",
        ]
        if scope:
            epoch = int(args.checkpoint.stem.removeprefix("epoch-"))
            cmd += [f"trainer.max_epochs={epoch + 2}"]
    return cmd + overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Start a finetuning branch; use an epoch checkpoint",
    )
    parser.add_argument("--dry-run", action="store_true")
    argv = sys.argv[1:]
    separator = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:separator])
    try:
        cmd = command(args, argv[separator + 1 :])
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(shlex.join(cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
