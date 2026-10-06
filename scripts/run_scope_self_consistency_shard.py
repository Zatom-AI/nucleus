#!/usr/bin/env python3
"""Run resumable zevals protein self-consistency on one backbone shard."""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from pathlib import Path

from collect_zevals_self_consistency import collect
from generate_scope_seeded import sample_seed, seed_rng
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from zevals.proteins.designability import compute_designability

LOGGER = logging.getLogger("scope_self_consistency_shard")


def _complete_sample(
    zevals_output_dir: Path,
    backbone: Path,
    expected_sequences: int,
    allow_incomplete_refolds: bool = False,
) -> bool:
    sample_dir = zevals_output_dir / "designability" / backbone.stem
    fasta = sample_dir / "designed.fasta"
    structures = sample_dir / "structures"
    if not fasta.is_file() or not structures.is_dir():
        return False
    if allow_incomplete_refolds:
        # Native SCOPe PDBs can make SolubleMPNN emit X at unresolved
        # positions. ESMFold2 rejects those sequences, but the backbone still
        # needs to remain in the native-domain denominator as a failed attempt.
        return True
    return len(_prediction_paths(structures)) >= expected_sequences


def _prediction_paths(structures_dir: Path) -> list[Path]:
    if not structures_dir.exists():
        return []
    return sorted([*structures_dir.glob("*.pdb"), *structures_dir.glob("*.cif")])


def _complete_seeded_sample(output: Path, backbone: Path, expected: int) -> bool:
    """Reject truncated oracle writes when resuming a seeded evaluation."""
    import numpy as np
    from zevals.proteins.designability import _atom_coords

    if not _complete_sample(output, backbone, expected):
        return False
    predictions = _prediction_paths(
        output / "designability" / backbone.stem / "structures"
    )
    if len(predictions) != expected:
        return False
    try:
        length = len(_atom_coords(backbone))
        for path in predictions:
            coords = _atom_coords(path)
            if len(coords) != length or not np.isfinite(coords).all():
                return False
    except Exception:
        return False
    return True


def _write_progress(
    progress_json: Path,
    *,
    shard_index: int,
    status: str,
    total: int,
    completed: int,
    skipped: int,
    failed: int,
    current: str | None = None,
) -> None:
    progress_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "shard_index": shard_index,
        "status": status,
        "total_backbones": total,
        "completed_backbones": completed,
        "skipped_backbones": skipped,
        "failed_backbones": failed,
        "current": current,
        "updated_at_unix": time.time(),
    }
    progress_json.write_text(json.dumps(payload, indent=2) + "\n")


def _collect_current(
    backbone_dir: Path,
    zevals_output_dir: Path,
    sc_rmsd_threshold: float,
    sc_atom: str,
    tmalign_bin: str,
) -> None:
    collect(
        backbone_dir=backbone_dir,
        zevals_output_dir=zevals_output_dir,
        out_json=zevals_output_dir / "self_consistency_details.json",
        out_csv=zevals_output_dir / "self_consistency_refolds.csv",
        tmalign_bin=tmalign_bin,
        sc_rmsd_threshold=sc_rmsd_threshold,
        sc_atom=sc_atom,
    )


def _build_oracles(
    zevals_dir: Path,
    zevals_output_dir: Path,
    *,
    mpnn_num_sequences: int,
    mpnn_temperature: float,
    mpnn_backbone_noise: float,
    structure_predictor: str,
    esmfold_model_name: str,
    esmfold_chunk_size: int,
    esmfold2_num_loops: int,
    esmfold2_num_sampling_steps: int,
    esmfold2_num_samples: int,
    esmfold2_seed: int,
    sc_rmsd_threshold: float,
    sc_atom: str,
):
    overrides = [
        "oracle/inverse_folder=soluble_mpnn",
        f"pipeline.output_dir={zevals_output_dir}",
        f"oracle.inverse_folder.num_sequences={mpnn_num_sequences}",
        f"oracle.inverse_folder.temperature={mpnn_temperature}",
        f"oracle.inverse_folder.backbone_noise={mpnn_backbone_noise}",
        f"pipeline.sc_rmsd_threshold={sc_rmsd_threshold}",
        f"pipeline.sc_atom={sc_atom}",
    ]
    if structure_predictor == "esmfold2":
        overrides.extend(
            [
                "oracle/structure_predictor=esmfold2",
                f"oracle.structure_predictor.model_name={esmfold_model_name}",
                f"oracle.structure_predictor.num_loops={esmfold2_num_loops}",
                f"oracle.structure_predictor.num_sampling_steps={esmfold2_num_sampling_steps}",
                f"oracle.structure_predictor.num_samples={esmfold2_num_samples}",
                f"oracle.structure_predictor.seed={esmfold2_seed}",
            ]
        )
    elif structure_predictor == "esmfold":
        overrides.extend(
            [
                "oracle/structure_predictor=esmfold",
                f"oracle.structure_predictor.model_name={esmfold_model_name}",
                f"oracle.structure_predictor.chunk_size={esmfold_chunk_size}",
            ]
        )
    else:
        raise ValueError(f"Unsupported structure predictor: {structure_predictor}")
    with initialize_config_dir(
        version_base=None,
        config_dir=str(zevals_dir / "configs"),
    ):
        cfg = compose(config_name="proteins", overrides=overrides)
    return (
        instantiate(cfg.oracle.structure_predictor),
        instantiate(cfg.oracle.inverse_folder),
    )


def run_shard(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    backbone_dir = args.backbone_dir
    zevals_output_dir = args.zevals_output_dir
    backbones = sorted(backbone_dir.glob("*.pdb"))
    if not backbones:
        raise SystemExit(f"No PDB backbones found in {backbone_dir}")

    _write_progress(
        args.progress_json,
        shard_index=args.shard_index,
        status="starting",
        total=len(backbones),
        completed=0,
        skipped=0,
        failed=0,
    )

    structure_predictor, inverse_folder = _build_oracles(
        args.zevals_dir,
        zevals_output_dir,
        mpnn_num_sequences=args.mpnn_num_sequences,
        mpnn_temperature=args.mpnn_temperature,
        mpnn_backbone_noise=args.mpnn_backbone_noise,
        structure_predictor=args.structure_predictor,
        esmfold_model_name=args.esmfold_model_name,
        esmfold_chunk_size=args.esmfold_chunk_size,
        esmfold2_num_loops=args.esmfold2_num_loops,
        esmfold2_num_sampling_steps=args.esmfold2_num_sampling_steps,
        esmfold2_num_samples=args.esmfold2_num_samples,
        esmfold2_seed=args.esmfold2_seed,
        sc_rmsd_threshold=args.sc_rmsd_threshold,
        sc_atom=args.sc_atom,
    )

    mpnn_seed = getattr(args, "mpnn_seed", None)
    if mpnn_seed is not None:
        # Lazy model initialization consumes RNG. Complete it before per-sample
        # seeding, as in evaluate_scope_training_designability.py.
        inverse_folder._get_model(ca_only=False)

    completed = 0
    skipped = 0
    failed = 0
    failures_path = zevals_output_dir / "failed_backbones.jsonl"
    zevals_output_dir.mkdir(parents=True, exist_ok=True)

    for backbone in backbones:
        current = backbone.name
        sample_complete = (
            _complete_seeded_sample(
                zevals_output_dir, backbone, args.mpnn_num_sequences
            )
            if mpnn_seed is not None
            else _complete_sample(
                zevals_output_dir,
                backbone,
                args.mpnn_num_sequences,
                args.allow_incomplete_refolds,
            )
        )
        if sample_complete:
            skipped += 1
            completed += 1
            LOGGER.info("Skipping completed backbone %s", backbone)
            _write_progress(
                args.progress_json,
                shard_index=args.shard_index,
                status="running",
                total=len(backbones),
                completed=completed,
                skipped=skipped,
                failed=failed,
                current=current,
            )
            continue

        try:
            LOGGER.info("Evaluating backbone %s", backbone)
            if mpnn_seed is not None:
                seed_rng(sample_seed(mpnn_seed, backbone.stem, "mpnn"))
            compute_designability(
                [backbone],
                structure_predictor=structure_predictor,
                inverse_folder=inverse_folder,
                output_dir=zevals_output_dir / "designability",
                sc_rmsd_threshold=args.sc_rmsd_threshold,
                sc_atom=args.sc_atom,
            )
            if not _complete_sample(
                zevals_output_dir,
                backbone,
                args.mpnn_num_sequences,
                args.allow_incomplete_refolds,
            ):
                raise RuntimeError(
                    f"Designability produced no complete refold set for {backbone}"
                )
            completed += 1
        except Exception as exc:
            failed += 1
            if args.allow_incomplete_refolds and _complete_sample(
                zevals_output_dir,
                backbone,
                args.mpnn_num_sequences,
                allow_incomplete_refolds=True,
            ):
                completed += 1
            with failures_path.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "backbone": str(backbone),
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                            "time_unix": time.time(),
                        }
                    )
                    + "\n"
                )
            LOGGER.exception("Backbone failed: %s", backbone)

        # Collection runs TM-align for every completed refold. Doing that
        # inside this loop repeats the same comparisons after every backbone
        # and makes a 96-backbone shard unnecessarily quadratic in work.
        _write_progress(
            args.progress_json,
            shard_index=args.shard_index,
            status="running",
            total=len(backbones),
            completed=completed,
            skipped=skipped,
            failed=failed,
            current=current,
        )

    # Collect once after all refolds are available. Partial raw outputs remain
    # resumable through the per-backbone completion checks above.
    _collect_current(
        backbone_dir,
        zevals_output_dir,
        args.sc_rmsd_threshold,
        args.sc_atom,
        args.tmalign_bin,
    )
    _write_progress(
        args.progress_json,
        shard_index=args.shard_index,
        status="complete",
        total=len(backbones),
        completed=completed,
        skipped=skipped,
        failed=failed,
    )
    if failed and not args.allow_incomplete_refolds:
        raise SystemExit(f"{failed} backbones failed in shard {args.shard_index}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--zevals-dir", type=Path, required=True)
    parser.add_argument("--backbone-dir", type=Path, required=True)
    parser.add_argument("--zevals-output-dir", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path, required=True)
    parser.add_argument("--mpnn-num-sequences", type=int, default=8)
    parser.add_argument("--mpnn-temperature", type=float, default=0.1)
    parser.add_argument("--mpnn-backbone-noise", type=float, default=0.02)
    parser.add_argument(
        "--mpnn-seed",
        type=int,
        help="Optional fixed master seed; derive an independent seed per backbone.",
    )
    parser.add_argument(
        "--structure-predictor",
        choices=["esmfold", "esmfold2"],
        default="esmfold",
    )
    parser.add_argument("--esmfold-model-name", default="facebook/esmfold_v1")
    parser.add_argument("--esmfold-chunk-size", type=int, default=64)
    parser.add_argument("--esmfold2-num-loops", type=int, default=3)
    parser.add_argument("--esmfold2-num-sampling-steps", type=int, default=50)
    parser.add_argument("--esmfold2-num-samples", type=int, default=1)
    parser.add_argument("--esmfold2-seed", type=int, default=0)
    parser.add_argument("--sc-rmsd-threshold", type=float, default=1.5)
    parser.add_argument("--sc-atom", default="CA")
    parser.add_argument("--tmalign-bin", default="TMalign")
    parser.add_argument(
        "--allow-incomplete-refolds",
        action="store_true",
        help=(
            "Treat a started but incomplete oracle evaluation as a completed "
            "non-designable sample. Intended for native-domain controls where "
            "invalid MPNN sequences must remain in the denominator."
        ),
    )
    run_shard(parser.parse_args())


if __name__ == "__main__":
    main()
