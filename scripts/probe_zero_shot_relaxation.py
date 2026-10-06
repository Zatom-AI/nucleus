#!/usr/bin/env python3
"""Probe frozen Zatom-2 relaxation with paired, independent UMA evaluation.

Generate and score in separate invocations so the oracle never affects sampling.
All structures retain source atom identities; periodic Cartesian predictions use
the source cell. A flow endpoint prediction uses one denoiser pass, whereas the
Euler control uses 16 passes. Neither performs force-based optimization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = "zatom2-table1-zatom2-l-joint-properties-ablation"
CHECKPOINT_SHA256 = "ebaa4afa55b7ab4d51df819ad45a95182cf03435a05460a37d1e64242c79b386"


def flow_euler_step(state, velocity, t0, t1, sigma_data):
    """Advance with the model's sigma-normalized velocity, in physical units."""
    return state + (t1 - t0) * sigma_data * velocity


def generate(args):
    import hydra
    import torch
    from dotenv import load_dotenv
    from hydra import compose, initialize_config_dir
    from rfd3na.transforms.conditioning_base import InvalidSampledConditionException
    from zatom2.mlip_evaluation import _build_dataset_config
    from zatom2.run_inference import build_inference_engine

    from nucleus.utils.batching import SparseCollate
    from nucleus.utils.inference import find_ema_wrapper
    from nucleus.utils.materials import get_material_lattice

    load_dotenv(ROOT / ".env", override=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with initialize_config_dir(
        config_dir=str(ROOT / "models/zatom2/configs"), version_base="1.3"
    ):
        cfg = compose(
            config_name="mlip_evaluation",
            overrides=[
                f"ckpt_path={CHECKPOINT}",
                f"out_dir={args.output}",
                f"seed={args.seed}",
                "evaluation.num_workers=0",
            ],
        )
    engine = build_inference_engine(cfg)
    train_cfg = engine.initialize()
    with Path(engine.ckpt_path).open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint does not match the registered experiment weights")
    trainer = engine.trainer
    ema = find_ema_wrapper(trainer.state["model"])
    if ema is None:
        raise RuntimeError("Expected EMA wrapper for this checkpoint.")
    model = ema.shadow.eval()
    assert model.diffusion_module.f_pred == "flow_matching"
    sigma_data = float(model.diffusion_module.sigma_data)
    manifest = {
        "checkpoint": str(engine.ckpt_path),
        "registered_checkpoint": CHECKPOINT,
        "ema": True,
        "seed": args.seed,
        "sigma_data": sigma_data,
        "n_recycle": 1,
        "num_structures_per_domain": args.count,
        "max_atoms": args.max_atoms,
        "noise_std_angstrom": args.noise,
        "denoiser_sigma_angstrom": args.sigmas,
        "targets_eV_per_angstrom": [None, 0.0, 0.05],
        "cell_policy": "source cell, Cartesian coordinate output",
        "paths": {},
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "checkpoint_sha256": checkpoint_hash,
    }
    records_path = args.output / "structures.jsonl"
    if records_path.exists():
        raise FileExistsError(records_path)
    with records_path.open("w") as handle:
        for domain in args.domains:
            path = str(Path(os.environ[f"{domain.upper()}_DATA_DIR"]).parent / "val")
            if domain == "omat24":
                path = str(Path(path) / "rattled-300")
            manifest["paths"][domain] = path
            dc = _build_dataset_config(train_cfg, domain, path, args.max_atoms)
            dc.transform.train_conditions.mlip.frequency = 0.0
            dc.transform.train_conditions.structure_prediction.frequency = 1.0
            dataset = hydra.utils.instantiate(dc)
            # Fixed shuffled candidate order, independent of model/oracle outcomes.
            order = np.random.default_rng(args.seed).permutation(len(dataset))
            selected = 0
            for index in order:
                try:
                    example = dataset[int(index)]
                except InvalidSampledConditionException:
                    # Structure prediction excludes disconnected molecular graphs.
                    continue
                gt = example["ground_truth"]
                n_atoms = int(gt["source_atom_count_B"].item())
                if not (
                    3 <= n_atoms <= args.max_atoms
                    and n_atoms == int(gt["retained_atom_count_B"].item())
                    and bool(gt["property_valid_mask_B"].item())
                    and bool(gt["force_valid_mask_L"].all())
                ):
                    continue
                collated = trainer.fabric.to_device(SparseCollate()([example]))
                net = trainer._assemble_network_inputs(collated)
                features = net["f"]
                assert not features["is_motif_atom_with_fixed_coord"].any()
                assert features["is_motif_atom_with_fixed_seq"].all()
                clean = net["X_clean_L"].float()
                array = example["atom_array"]
                assert len(array) == n_atoms == clean.shape[1]
                lattice = get_material_lattice(array)
                cell = lattice[2].tolist() if lattice is not None else None
                base = {
                    "domain": domain,
                    "ordinal": selected,
                    "index": int(index),
                    "example_id": str(example["example_id"]),
                    "elements": array.element.tolist(),
                    "cell": cell,
                    "charge": int(gt["total_charge_B"].item()),
                    "spin": int(gt["spin_multiplicity_B"].item()),
                    "charge_valid": bool(gt["charge_valid_mask_B"].item()),
                    "spin_valid": bool(gt["spin_valid_mask_B"].item()),
                    "source_bonds": array.bonds.as_array().tolist()
                    if array.bonds is not None and domain == "omol25"
                    else [],
                }
                if domain == "omol25":
                    assert base["charge_valid"] and base["spin_valid"]
                rng = torch.Generator(device="cpu").manual_seed(args.seed + int(index))
                perturbation = torch.randn(clean.shape, generator=rng).to(clean.device)
                perturbation -= perturbation.mean(dim=1, keepdim=True)

                def save(coords, **metadata):
                    row = {
                        **base,
                        **metadata,
                        "positions": coords[0].float().cpu().tolist(),
                    }
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                    handle.flush()

                for noise in args.noise:
                    start = clean + noise * perturbation
                    save(start, noise=noise, method="input", sigma=None, target=None)
                    for target in (None, 0.0, 0.05):
                        f = dict(features)
                        f["mean_force_norm"] = torch.full_like(
                            features["mean_force_norm"],
                            0.0 if target is None else np.log10(max(target, 1e-8)),
                        )
                        f["has_mean_force_norm"] = torch.full_like(
                            features["has_mean_force_norm"], float(target is not None)
                        )
                        with torch.inference_mode(), trainer.fabric.autocast():
                            initializer = model.token_initializer(f)
                            for sigma in args.sigmas:
                                t0 = sigma_data / (sigma_data + sigma)
                                t = torch.tensor([t0], device=start.device)
                                torch.cuda.synchronize()
                                started = time.perf_counter()
                                out = model.diffusion_module(
                                    X_noisy_L=t0 * start,
                                    t=t,
                                    f=f,
                                    n_recycle=1,
                                    **initializer,
                                )
                                torch.cuda.synchronize()
                                save(
                                    out["X_L"],
                                    noise=noise,
                                    method="one_step",
                                    sigma=sigma,
                                    target=target,
                                    denoiser_seconds=time.perf_counter() - started,
                                )
                            # Deterministic partial flow from the same input;
                            # no extra random noise, rotation, CFG, or force calls.
                            sigma = 0.3
                            times = torch.linspace(
                                sigma_data / (sigma_data + sigma),
                                1.0,
                                17,
                                device=start.device,
                            )
                            state = times[0] * start
                            for t0, t1 in zip(times[:-1], times[1:], strict=True):
                                out = model.diffusion_module(
                                    X_noisy_L=state,
                                    t=t0.reshape(1),
                                    f=f,
                                    n_recycle=1,
                                    **initializer,
                                )
                                # V_L is normalized by sigma_data in this model.
                                state = flow_euler_step(
                                    state, out["V_L"], t0, t1, sigma_data
                                )
                            save(
                                state,
                                noise=noise,
                                method="euler16",
                                sigma=sigma,
                                target=target,
                            )
                selected += 1
                print(
                    f"Generated {domain} {selected}/{args.count}: {index}", flush=True
                )
                if selected == args.count:
                    break
            if selected != args.count:
                raise RuntimeError(f"Only found {selected} valid {domain} structures")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def score(args):
    from ase import Atoms
    from ase.data import covalent_radii
    from zevals.atomistic.uma.model import UMAModel

    if not (args.output / "manifest.json").exists():
        raise RuntimeError("Generation must complete before oracle scoring.")
    model = UMAModel(checkpoint="uma-s-1p2", device="cuda")
    output_path = args.output / "scores.jsonl"
    completed = 0
    if output_path.exists():
        completed = sum(1 for _ in output_path.open())
    with output_path.open("a") as handle:
        for ordinal, line in enumerate((args.output / "structures.jsonl").open()):
            if ordinal < completed:
                continue
            row = json.loads(line)
            atoms = Atoms(
                symbols=row["elements"],
                positions=row.pop("positions"),
                cell=row["cell"],
                pbc=row["cell"] is not None,
            )
            model.task = "omol" if row["domain"] == "omol25" else "omat"
            atoms.info.update(
                charge=row["charge"] if row["domain"] == "omol25" else 0,
                spin=row["spin"] if row["domain"] == "omol25" else 0,
            )
            if row["domain"] == "omol25" and row.get("source_bonds"):
                edges = np.array(row["source_bonds"], dtype=int)[:, :2]
                radii = covalent_radii[atoms.numbers]
                lengths = np.linalg.norm(
                    atoms.positions[edges[:, 0]] - atoms.positions[edges[:, 1]], axis=1
                )
                ratios = lengths / radii[edges].sum(axis=1)
                row["source_bond_length_ratio_min"] = float(ratios.min())
                row["source_bond_length_ratio_max"] = float(ratios.max())
                row["source_bonds_within_distance_bounds"] = bool(
                    ((ratios > 0.65) & (ratios < 1.5)).all()
                )
            try:
                prediction = model.predict(atoms)
                forces = prediction["forces"].float().cpu().numpy()
                energy = float(prediction["energy"].sum())
                if not np.isfinite(energy) or not np.isfinite(forces).all():
                    raise ValueError("Non-finite oracle output")
                row.update(
                    energy_eV=energy,
                    mean_force=float(np.linalg.norm(forces, axis=-1).mean()),
                    max_force=float(np.linalg.norm(forces, axis=-1).max()),
                )
            except Exception as error:
                row["error"] = repr(error)
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            if ordinal % 25 == 0:
                print(f"Scored {ordinal + 1} structures", flush=True)


def summarize(args):
    """Aggregate all paired trials; retain failed predictions in denominators."""
    rows = [json.loads(line) for line in (args.output / "scores.jsonl").open()]
    expected = sum(1 for _ in (args.output / "structures.jsonl").open())
    if len(rows) != expected:
        raise RuntimeError(f"Scoring is incomplete: {len(rows)}/{expected}")
    inputs = {
        (row["domain"], row["index"], row["noise"]): row
        for row in rows
        if row["method"] == "input"
    }
    unconditioned = {
        (row["domain"], row["index"], row["noise"], row["method"], row["sigma"]): row
        for row in rows
        if row["method"] != "input" and row["target"] is None
    }
    groups = defaultdict(list)
    for row in rows:
        if row["method"] != "input":
            groups[
                (
                    row["domain"],
                    row["noise"],
                    row["method"],
                    row["sigma"],
                    row["target"],
                )
            ].append(row)
    summaries = []
    for key, candidates in groups.items():
        ratios, deltas, maximums, jointly_better = [], [], [], []
        input_forces, output_forces, conditioning_ratios, bond_checks = [], [], [], []
        for row in candidates:
            source = inputs[(row["domain"], row["index"], row["noise"])]
            if "error" in row or "error" in source:
                continue
            assert row["elements"] == source["elements"]
            ratios.append(row["mean_force"] / source["mean_force"])
            deltas.append(
                (row["energy_eV"] - source["energy_eV"]) / len(row["elements"])
            )
            maximums.append(row["max_force"])
            jointly_better.append(ratios[-1] < 1 and deltas[-1] < 0)
            input_forces.append(source["mean_force"])
            output_forces.append(row["mean_force"])
            control = unconditioned[
                (row["domain"], row["index"], row["noise"], row["method"], row["sigma"])
            ]
            if "error" not in control:
                conditioning_ratios.append(row["mean_force"] / control["mean_force"])
            if "source_bonds_within_distance_bounds" in row:
                bond_checks.append(row["source_bonds_within_distance_bounds"])
        result = dict(
            zip(("domain", "noise", "method", "sigma", "target"), key, strict=True)
        )
        result.update(attempts=len(candidates), scored=len(ratios))
        if ratios:
            ratio_array = np.array(ratios)
            draws = np.random.default_rng(args.seed).choice(
                ratio_array,
                size=(10000, len(ratios)),
                replace=True,
            )
            control_interval = None
            if conditioning_ratios:
                control_draws = np.random.default_rng(args.seed).choice(
                    conditioning_ratios,
                    size=(10000, len(conditioning_ratios)),
                    replace=True,
                )
                control_interval = np.quantile(
                    np.median(control_draws, axis=1), [0.025, 0.975]
                ).tolist()
            result.update(
                median_force_ratio=float(np.median(ratios)),
                median_force_ratio_bootstrap95=np.quantile(
                    np.median(draws, axis=1), [0.025, 0.975]
                ).tolist(),
                median_delta_energy_eV_per_atom=float(np.median(deltas)),
                force_improved=sum(ratio_array < 1).item(),
                energy_improved=int(sum(value < 0 for value in deltas)),
                both_improved=int(sum(jointly_better)),
                max_force_below_005=int(sum(value < 0.05 for value in maximums)),
                median_input_mean_force=float(np.median(input_forces)),
                median_output_mean_force=float(np.median(output_forces)),
                median_force_ratio_to_unconditioned=(
                    float(np.median(conditioning_ratios))
                    if conditioning_ratios
                    else None
                ),
                median_force_ratio_to_unconditioned_bootstrap95=control_interval,
                source_bonds_within_distance_bounds=int(sum(bond_checks)),
                source_bonds_checked=len(bond_checks),
            )
        summaries.append(result)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    for row in summaries:
        print(json.dumps(row))


def relax(args):
    """Compute a fixed-cell UMA/BFGS reference on every selected input at one noise."""
    from ase import Atoms
    from ase.optimize import BFGS
    from zevals.atomistic.uma.model import UMAModel

    model = UMAModel(checkpoint="uma-s-1p2", device="cuda")
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    path = args.output / "relaxation_baseline.jsonl"
    completed = set()
    if path.exists():
        completed = {
            (row["domain"], row["index"], row["noise"])
            for row in map(json.loads, path.open())
        }
    with path.open("a") as handle:
        for line in (args.output / "structures.jsonl").open():
            row = json.loads(line)
            if row["method"] != "input" or row["noise"] != args.relax_noise:
                continue
            if (row["domain"], row["index"], row["noise"]) in completed:
                continue
            atoms = Atoms(
                symbols=row["elements"],
                positions=row.pop("positions"),
                cell=row["cell"],
                pbc=row["cell"] is not None,
            )
            task = "omol" if row["domain"] == "omol25" else "omat"
            atoms.info.update(
                charge=row["charge"] if task == "omol" else 0,
                spin=row["spin"] if task == "omol" else 0,
            )
            atoms.calc = FAIRChemCalculator(model.predict_unit, task_name=task)
            optimizer = BFGS(atoms, logfile=None, maxstep=0.04)
            started = time.perf_counter()
            try:
                converged = optimizer.run(fmax=0.05, steps=args.relax_steps)
                forces = np.linalg.norm(atoms.get_forces(), axis=-1)
                row.update(
                    converged=bool(converged),
                    steps=optimizer.nsteps,
                    energy_eV=float(atoms.get_potential_energy()),
                    mean_force=float(forces.mean()),
                    max_force=float(forces.max()),
                    positions=atoms.positions.tolist(),
                    seconds=time.perf_counter() - started,
                )
            except Exception as error:
                row["error"] = repr(error)
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            print(
                f"BFGS {row['domain']} {row['index']}: {row.get('converged', row.get('error'))}",
                flush=True,
            )


def select_one_step_inputs(rows):
    """Select Table 10 inputs: perturbed molecules and native rattled materials."""
    noise_by_domain = {"omol25": 0.1, "omat24": 0.0}
    selected = [
        row
        for row in rows
        if row["method"] == "input"
        and row["noise"] == noise_by_domain.get(row["domain"])
    ]
    keys = {(row["domain"], row["index"]) for row in selected}
    if len(keys) != len(selected):
        raise ValueError("Duplicate source structures in one-step baseline")
    return selected


def one_bfgs_update(atoms):
    """Take exactly one fixed-cell update with a fresh ASE BFGS optimizer.

    With H0 = 70 I eV/Angstrom^2, this is F/70, uniformly rescaled if the
    largest atomic displacement exceeds 0.04 Angstrom. There is no line search
    or Hessian history. Endpoint scoring is separate from the update.
    """
    from ase.optimize import BFGS

    optimizer = BFGS(atoms, logfile=None, alpha=70.0, maxstep=0.04)
    optimizer.step()


def one_step_uma(args):
    """Evaluate a single UMA/BFGS step on exactly the manuscript table inputs."""
    from ase import Atoms
    from zevals.atomistic.uma.model import UMAModel

    path = args.output / "uma_one_step_baseline.jsonl"
    if path.exists():
        raise FileExistsError(path)
    rows = select_one_step_inputs(
        [json.loads(line) for line in (args.output / "structures.jsonl").open()]
    )
    if not rows:
        raise ValueError("No matching table inputs")
    model = UMAModel(checkpoint=args.uma_checkpoint, device="cuda")
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    results = []
    with path.open("x") as handle:
        for source in rows:
            row = dict(source)
            positions = np.asarray(row.pop("positions"), dtype=float)
            atoms = Atoms(
                symbols=row["elements"],
                positions=positions,
                cell=row["cell"],
                pbc=row["cell"] is not None,
            )
            task = "omol" if row["domain"] == "omol25" else "omat"
            atoms.info.update(
                charge=row["charge"] if task == "omol" else 0,
                spin=row["spin"] if task == "omol" else 0,
            )
            atoms.calc = FAIRChemCalculator(model.predict_unit, task_name=task)
            row.update(method="uma_bfgs_one_step", steps=1, target=None, sigma=None)
            try:
                initial_forces = np.linalg.norm(atoms.get_forces(), axis=-1)
                row.update(
                    input_mean_force=float(initial_forces.mean()),
                    input_energy_eV=float(atoms.get_potential_energy()),
                )
                # The initial calculator results are cached: the update uses
                # this one force evaluation, then endpoint scoring needs another.
                one_bfgs_update(atoms)
                forces = np.linalg.norm(atoms.get_forces(), axis=-1)
                displacement = np.linalg.norm(atoms.positions - positions, axis=-1)
                row.update(
                    energy_eV=float(atoms.get_potential_energy()),
                    mean_force=float(forces.mean()),
                    max_force=float(forces.max()),
                    positions=atoms.positions.tolist(),
                    max_displacement_angstrom=float(displacement.max()),
                )
                assert displacement.max() <= 0.04 + 1e-12
                assert np.isfinite(forces).all() and np.isfinite(row["energy_eV"])
            except Exception as error:
                row["error"] = repr(error)
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            results.append(row)
            print(
                f"UMA one step {row['domain']} {row['index']}: {row.get('mean_force', row.get('error'))}",
                flush=True,
            )

    summaries = []
    for domain in ("omol25", "omat24"):
        attempted = [row for row in results if row["domain"] == domain]
        valid = [row for row in attempted if "error" not in row]
        summary = dict(domain=domain, attempts=len(attempted), scored=len(valid))
        if valid:
            ratios = np.array([r["mean_force"] / r["input_mean_force"] for r in valid])
            draws = np.random.default_rng(args.seed).choice(
                ratios, (10000, len(ratios))
            )
            summary.update(
                median_force_ratio=float(np.median(ratios)),
                median_force_ratio_bootstrap95=np.quantile(
                    np.median(draws, axis=1), [0.025, 0.975]
                ).tolist(),
                median_output_mean_force=float(
                    np.median([r["mean_force"] for r in valid])
                ),
                force_improved=int(
                    sum(r["mean_force"] < r["input_mean_force"] for r in valid)
                ),
                energy_improved=int(
                    sum(r["energy_eV"] < r["input_energy_eV"] for r in valid)
                ),
                both_improved=int(
                    sum(
                        r["mean_force"] < r["input_mean_force"]
                        and r["energy_eV"] < r["input_energy_eV"]
                        for r in valid
                    )
                ),
                max_force_below_005=int(sum(r["max_force"] < 0.05 for r in valid)),
                median_delta_energy_eV_per_atom=float(
                    np.median(
                        [
                            (r["energy_eV"] - r["input_energy_eV"]) / len(r["elements"])
                            for r in valid
                        ]
                    )
                ),
            )
        summaries.append(summary)
    payload = {
        "protocol": {
            "checkpoint": args.uma_checkpoint,
            "steps": 1,
            "optimizer": "ASE BFGS",
            "alpha_eV_per_angstrom2": 70.0,
            "max_step_angstrom": 0.04,
            "fixed_cell": True,
            "noise_by_domain": {"omol25": 0.1, "omat24": 0.0},
            "update_force_evaluations": 1,
            "endpoint_force_evaluations": 1,
            "bootstrap_seed": args.seed,
            "bootstrap_resamples": 10000,
        },
        "summary": summaries,
    }
    (args.output / "uma_one_step_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    if any("error" in row for row in results):
        raise RuntimeError("One-step baseline has failed attempts; inspect raw results")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["generate", "score", "summarize", "relax", "one-step-uma"]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--max-atoms", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--domains", nargs="+", default=["omol25", "omat24"])
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.1, 0.3])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    parser.add_argument("--relax-noise", type=float, default=0.1)
    parser.add_argument("--relax-steps", type=int, default=200)
    parser.add_argument("--uma-checkpoint", default="uma-s-1p2")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {
        "generate": generate,
        "score": score,
        "summarize": summarize,
        "relax": relax,
        "one-step-uma": one_step_uma,
    }[args.mode](args)


if __name__ == "__main__":
    main()
