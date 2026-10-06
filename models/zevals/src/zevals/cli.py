"""`zevals` console script.

Two subcommands:

    zevals setup [--tools foldseek,mmseqs,usalign] [--install-dir DIR] [--force]
                 [--build-db PDB --db-dir /persistent/storage/foldseek_dbs]
                 [--checkpoints protein_mpnn_ca,soluble_mpnn,ligand_mpnn [--checkpoint-dir DIR]]
    zevals evaluate --modality proteins=outputs/backbones \\
                    --modality molecules=outputs/ligands \\
                    --ground-truth molecules=references/conformers \\
                    [--override proteins:oracle/inverse_folder=soluble_mpnn] \\
                    [--results-dir outputs/eval]

`--build-db`/`--db-dir` are a separate, explicit step run *after* installing
the binaries (`--tools`) — reference databases are multi-GB+ and
location-sensitive (persistent cluster storage, not wherever the tiny
binaries land), so they're never downloaded implicitly. See
`zevals.setup.build_database` and the README's FoldSeek database section.

`--checkpoints`/`--checkpoint-dir` fetch small (~6-45MB), publicly-hosted,
no-license-needed checkpoints (ProteinMPNN/SolubleMPNN/LigandMPNN/NA-MPNN
variants, plus gRNAde's weights for a separately-managed gRNAde env — see
README §5). Unlike `--build-db`, `--checkpoint-dir` defaults to
`zevals.setup.default_checkpoint_dir()` — the same location every shipped
`oracle/inverse_folder/*.yaml` config's `weights_dir`/`checkpoint_path`
already falls back to when left unset, so running this with no
`--checkpoint-dir` override is what makes the default proteins config work
end to end with zero further overrides.

`evaluate` is a thin one-off CLI wrapper around `ZevalSuite` for command-line
use. For a training loop, use `zevals.pipeline.suite.ZevalSuite`
directly in Python instead (construct once, call `.evaluate()` every
validation epoch) — see that class's docstring for why.

Uses stdlib `argparse` rather than Hydra's `@hydra.main`, since
`ZevalSuite.evaluate()`'s input is a `{modality: samples_dir}` mapping
(possibly spanning several modalities/configs in one call), which doesn't
fit Hydra's single-primary-config-per-invocation model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from zevals.pipeline.suite import ZevalSuite


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="zevals")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser(
        "setup",
        help="Download FoldSeek/MMseqs2 static binaries (see zevals.setup)",
    )
    setup_parser.add_argument(
        "--tools",
        type=str,
        default=None,
        help="Comma-separated tools to install (default: all — foldseek,mmseqs,usalign)",
    )
    setup_parser.add_argument(
        "--install-dir",
        type=str,
        default=None,
        help="Install location (default: ~/.local/bin)",
    )
    setup_parser.add_argument(
        "--force", action="store_true", help="Reinstall even if already on PATH"
    )
    setup_parser.add_argument(
        "--build-db",
        type=str,
        default=None,
        metavar="DB_NAME",
        help="After installing, also build a FoldSeek reference database for "
        "novelty (e.g. PDB, Alphafold/UniProt50-minimal — run "
        "'foldseek databases' with no args to list presets). Requires --db-dir.",
    )
    setup_parser.add_argument(
        "--db-dir",
        type=str,
        default=None,
        help="Persistent, large-storage directory to build --build-db under "
        "(not /tmp or purged scratch). Required if --build-db is set.",
    )
    setup_parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        metavar="NAME1,NAME2,...",
        help="Comma-separated checkpoints to download — any of "
        "protein_mpnn_vanilla, protein_mpnn_ca, protein_mpnn_soluble, "
        "soluble_mpnn, ligand_mpnn, na_mpnn_design, na_mpnn_specificity, "
        "grnade, grnade_ribonanzanet, grnade_ribonanzanet_ss, rhofold.",
    )
    setup_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint cache root for --checkpoints (default: "
        "zevals.setup.default_checkpoint_dir(), the same location the "
        "shipped oracle configs already fall back to).",
    )

    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate one or more modalities' worth of generated samples"
    )
    eval_parser.add_argument(
        "--modality",
        action="append",
        required=True,
        metavar="NAME=SAMPLES_DIR",
        help="e.g. --modality proteins=outputs/backbones. Repeatable.",
    )
    eval_parser.add_argument(
        "--ground-truth",
        action="append",
        default=[],
        metavar="NAME=REFERENCES_DIR",
        help="Optional reference directory for a modality; repeatable. Molecule "
        "conformer ensembles require matching <target_id>/*.sdf layouts.",
    )
    eval_parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="MODALITY:KEY=VALUE",
        help="Hydra override scoped to one modality, e.g. "
        "--override proteins:oracle/inverse_folder=soluble_mpnn. Repeatable.",
    )
    eval_parser.add_argument(
        "--results-dir", type=str, default=None, help="If set, write results.json here"
    )

    args = parser.parse_args(argv)

    if args.command == "setup":
        from zevals.setup import build_database, download_checkpoint, install

        if args.build_db and not args.db_dir:
            parser.error("--build-db requires --db-dir")

        tools = args.tools.split(",") if args.tools else None
        install(tools=tools, install_dir=args.install_dir, force=args.force)

        if args.build_db:
            db_path = build_database(args.build_db, args.db_dir, force=args.force)
            print(f"FoldSeek database ready — set target_db to: {db_path}")

        if args.checkpoints:
            for name in args.checkpoints.split(","):
                path = download_checkpoint(
                    name, output_dir=args.checkpoint_dir, force=args.force
                )
                print(f"Checkpoint {name!r} ready at: {path}")
        return

    if args.command == "evaluate":
        samples_dirs = dict(_split_kv(m, sep="=") for m in args.modality)
        ground_truth_dirs = dict(
            _split_kv(ground_truth, sep="=") for ground_truth in args.ground_truth
        )

        modality_overrides: dict[str, list[str]] = {}
        for raw in args.override:
            modality, override = _split_kv(raw, sep=":")
            modality_overrides.setdefault(modality, []).append(override)

        suite = ZevalSuite(
            modality_overrides=modality_overrides, results_dir=args.results_dir
        )
        if ground_truth_dirs:
            suite.evaluate(samples_dirs, ground_truth_dirs=ground_truth_dirs)
        else:
            suite.evaluate(samples_dirs)
        return



def _split_kv(raw: str, sep: str) -> tuple[str, str]:
    if sep not in raw:
        raise ValueError(f"Expected '{sep}' in {raw!r}")
    key, value = raw.split(sep, 1)
    return key, value


if __name__ == "__main__":
    main(sys.argv[1:])
