#!/usr/bin/env python3
"""Frozen SCOPe representation probe; prepare on CPU, extract on one GPU.

The final token is perturbed before atom decoding and sequence readout. All
conditioning is masked; observed atom14 geometry still carries residue identity.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path

import numpy as np
from plot_zatom2_subset_embeddings import ROOT, input_digest, sha256, write_json

SEED = 20260929
VARIANTS = ("scratch", "joint_properties_fc")
MAGNITUDES = (0.0, 0.1, 0.25, 0.5, 1.0)
FLOW_TIME = 0.8
NOISE_SCALE = 16.0


def seed_all(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def eligible_rows(all_rows, train_rows):
    """Exclude both training domains and their parent PDB entries."""
    domains = {r["domain_id"] for r in train_rows}
    pdbs = {r["pdb_id"].lower() for r in train_rows}
    return [
        (i, r)
        for i, r in enumerate(all_rows)
        if r["domain_id"] not in domains
        and r["pdb_id"].lower() not in pdbs
        and 50 <= int(r["length"]) <= 128
    ]


def prepare(args):
    """Freeze label-blind domain selection and identical model inputs."""
    import hydra
    import torch
    from omegaconf import OmegaConf

    from nucleus.utils.batching import SparseCollate

    if (args.work / "manifest.json").exists():
        raise FileExistsError(args.work / "manifest.json")
    args.work.mkdir(parents=True, exist_ok=True)
    selection = json.loads(args.checkpoint_manifest.read_text())
    models = {m["name"]: m for m in selection["models"] if m["name"] in VARIANTS}
    checkpoint = Path(models["joint_properties_fc"]["checkpoint"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    cfg = OmegaConf.create(
        OmegaConf.to_container(
            state["train_cfg"].datasets.train.scope.dataset, resolve=True
        )
    )
    cfg.dataset.data = str(args.all_csv.resolve())
    cfg.dataset_parser.base_path = str(args.pdb_dir.resolve())
    cfg.cif_parser_args.ccd_mirror_path = None
    cfg.transform.return_atom_array = True
    cfg.transform.diffusion_batch_size = 1
    cfg.transform.randomly_augment = False
    cfg.transform.sigma_perturb = 0.0
    cfg.transform.sigma_perturb_com = 0.0
    cfg.transform.mean_force_norm_conditioning_probability = 0.0
    for key in cfg.transform.meta_conditioning_probabilities:
        cfg.transform.meta_conditioning_probabilities[key] = 0.0
    dataset = hydra.utils.instantiate(cfg)
    OmegaConf.save(cfg, args.work / "preprocessing.yaml")
    with args.all_csv.open() as handle:
        all_rows = list(csv.DictReader(handle))
    with args.train_csv.open() as handle:
        train_rows = list(csv.DictReader(handle))
    candidates = eligible_rows(all_rows, train_rows)
    candidates.sort(key=lambda x: x[1]["domain_id"])
    np.random.default_rng(SEED).shuffle(candidates)
    records, rejected, used_pdbs = [], [], set()
    standard = set(
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
    )
    for index, row in candidates:
        if row["pdb_id"] in used_pdbs:
            continue
        seed_all(SEED + index)
        try:
            example = dataset[index]
            atoms = example["atom_array"]
            feats = example["feats"]
            mapping = np.asarray(feats["atom_to_token_map"])
            n_residues = int(mapping.max()) + 1
            if n_residues != int(row["length"]):
                raise ValueError("Residue count changed during preprocessing")
            labels = np.asarray(
                [atoms.res_name[mapping == j][0] for j in range(n_residues)]
            )
            if not set(labels).issubset(standard):
                raise ValueError("Nonstandard residue")
            if not np.all(np.bincount(mapping) == 14):
                raise ValueError("Expected exactly 14 atom slots per residue")
            for key in (
                "is_motif_atom_with_fixed_coord",
                "is_motif_atom_with_fixed_seq",
            ):
                if np.asarray(feats[key]).any():
                    raise ValueError(f"Unexpected conditioning: {key}")
            for key in ("restype", "ref_element", "ref_pos"):
                values = np.asarray(feats[key])
                if len(np.unique(values, axis=0)) != 1:
                    raise ValueError(f"Residue identity leaks through {key}")
            # Generic names can be permuted by symmetric-atom augmentation.
            # Every residue must nevertheless have the same 14-name multiset.
            generic_names = np.asarray(feats["ref_atom_name_chars"]).argmax(-1)
            name_blocks = [
                sorted(map(tuple, generic_names[mapping == j]))
                for j in range(n_residues)
            ]
            if any(block != name_blocks[0] for block in name_blocks):
                raise ValueError("Residue-specific atom names")
            collated = SparseCollate()([example])
            coordinates = collated["coord_atom_lvl_to_be_noised"]
            if coordinates.ndim == 2:
                coordinates = coordinates.unsqueeze(0)
            if not torch.isfinite(coordinates).all():
                raise ValueError("Nonfinite coordinates")
            # The distances of all atom14 slots control for geometric identity
            # information, including the standard virtual-atom placement.
            local = coordinates.numpy().reshape(n_residues, 14, 3).copy()
            for j in range(n_residues):
                names = generic_names[mapping == j]
                order = sorted(range(14), key=lambda k: tuple(names[k]))
                local[j] = local[j, order]
            distances = np.linalg.norm(local[:, :, None] - local[:, None, :], axis=-1)
            upper = np.triu_indices(14, k=1)
            geometry = distances[:, upper[0], upper[1]]
            item = dict(
                f=collated["feats"],
                coordinates=coordinates,
                labels=labels,
                geometry=geometry,
                domain_id=row["domain_id"],
            )
            destination = args.work / "inputs" / f"{row['domain_id']}.pt"
            destination.parent.mkdir(exist_ok=True)
            torch.save(item, destination)
            source = args.pdb_dir / f"{row['path']}.pdb"
            records.append(
                dict(
                    **row,
                    dataset_index=index,
                    pdb_sha256=sha256(source),
                    input_sha256=sha256(destination),
                    feature_sha256=input_digest(item["f"], coordinates),
                )
            )
            used_pdbs.add(row["pdb_id"])
            print(
                f"Prepared {len(records)}/{args.count}: {row['domain_id']}", flush=True
            )
        except (ValueError, KeyError, AssertionError) as error:
            rejected.append(dict(domain_id=row["domain_id"], reason=str(error)))
        if len(records) == args.count:
            break
    if len(records) != args.count:
        write_json(args.work / "rejected.json", rejected)
        raise ValueError(f"Only {len(records)} eligible structures; {rejected[:3]}")
    for model in models.values():
        model["sha256"] = sha256(model["checkpoint"])
    write_json(
        args.work / "manifest.json",
        dict(
            seed=SEED,
            models=models,
            records=records,
            rejected=rejected,
            candidates=len(candidates),
            all_csv_sha256=sha256(args.all_csv),
            train_csv_sha256=sha256(args.train_csv),
            checkpoint_manifest_sha256=sha256(args.checkpoint_manifest),
            preprocessing_sha256=sha256(args.work / "preprocessing.yaml"),
        ),
    )


def perturb_token(values, residue, direction, magnitude):
    """Perturb one token by a prescribed fraction of its Euclidean norm."""
    modified = values.clone()
    unit = direction / direction.norm()
    modified[:, residue] += (
        magnitude * values[:, residue].norm(dim=-1, keepdim=True) * unit
    )
    return modified


def deterministic_atom14_pool(module, R_L, tok_idx, num_tok):
    """Equivalent fixed-size mean for this probe's contiguous atom14 inputs.

    CUDA index_reduce(mean), used by the normal coordinate initializer, is
    nondeterministic. This analysis-only adapter changes reduction order, not
    parameters or the mathematical operation, and requires exactly 14 slots.
    """
    import torch

    expected = torch.arange(num_tok, device=tok_idx.device).repeat_interleave(14)
    if not torch.equal(tok_idx, expected):
        raise ValueError("Deterministic pooling requires contiguous atom14 inputs")
    projected = module.linear(R_L)
    return projected.reshape(projected.shape[0], num_tok, 14, -1).mean(dim=2)


def response_by_residue(coordinates, reference, mapping, n_residues, atom_mask):
    """Mean squared displacement of selected atoms, summing xyz components."""
    delta = (coordinates.float() - reference.float()).square().sum(-1).mean(0)
    delta = delta[atom_mask]
    mapping = mapping[atom_mask]
    import torch

    result = torch.zeros(n_residues, device=delta.device)
    count = torch.zeros_like(result)
    result.scatter_add_(0, mapping, delta)
    count.scatter_add_(0, mapping, torch.ones_like(delta))
    if (count == 0).any():
        raise ValueError("Every residue must contain a selected atom")
    return (result / count).cpu().numpy()


def sequence_loss_by_residue(logits, targets):
    """Cross-entropy against the observed sequence, averaged over views."""
    import torch.nn.functional as functional

    log_probabilities = functional.log_softmax(logits.float(), dim=-1)
    indices = targets[None, :, None].expand(logits.shape[0], -1, 1)
    return -log_probabilities.gather(-1, indices).squeeze(-1).mean(0).cpu().numpy()


def extract(args):
    """Capture token features and intervene after the final global token block."""
    import torch
    from atomworks.ml.encoding_definitions import AF3SequenceEncoding
    from hydra import compose, initialize_config_dir
    from zatom2.run_inference import build_inference_engine

    from nucleus.utils.inference import find_ema_wrapper

    archive = args.work / f"embeddings_{args.variant}.npz"
    if archive.exists():
        raise FileExistsError(archive)
    manifest = json.loads((args.work / "manifest.json").read_text())
    selected = manifest["models"][args.variant]
    if sha256(selected["checkpoint"]) != selected["sha256"]:
        raise ValueError("Checkpoint changed")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    seed_all(SEED)
    with initialize_config_dir(
        config_dir=str(ROOT / "models/zatom2/configs"), version_base="1.3"
    ):
        cfg = compose(
            config_name="inference",
            overrides=[
                f"ckpt_path={selected['checkpoint']}",
                "inputs=null",
                f"out_dir={args.work / args.variant}",
                f"seed={SEED}",
            ],
        )
    engine = build_inference_engine(cfg)
    engine._assign_override("trainer.precision", "32-true")
    engine.initialize()
    trainer = engine.trainer
    ema = find_ema_wrapper(trainer.state["model"])
    if ema is None:
        raise ValueError("EMA weights required")
    model = ema.shadow.eval()
    from types import MethodType

    model.diffusion_module.process_r_token.forward = MethodType(
        deterministic_atom14_pool, model.diffusion_module.process_r_token
    )
    trunk = model.diffusion_module.backbone.token_trunk
    tokens, final_tokens, labels, structures, geometry, signatures = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    responses = []
    perturbation_inputs = []
    start = time.monotonic()
    for structure, record in enumerate(manifest["records"]):
        path = args.work / "inputs" / f"{record['domain_id']}.pt"
        if sha256(path) != record["input_sha256"]:
            raise ValueError("Prepared example changed")
        item = torch.load(path, map_location="cpu", weights_only=False)
        signatures.append(input_digest(item["f"], item["coordinates"]))
        if signatures[-1] != record["feature_sha256"]:
            raise ValueError("Input feature hash mismatch")
        f = trainer.fabric.to_device(item["f"])
        coordinates = trainer.fabric.to_device(item["coordinates"])
        n_residues = len(item["labels"])
        f["diffusion_batch_size"] = 1
        seed_all(SEED + structure)
        captured = []
        final_captured = []

        def capture(_module, _args, kwargs):
            captured.append(kwargs["Q"].detach().clone())

        def capture_final(_module, _args, output):
            final_captured.append(output.detach().clone())

        with torch.inference_mode(), trainer.fabric.autocast():
            initialized = model.token_initializer(f)

            def forward(coords, flow_time):
                return model.diffusion_module(
                    X_noisy_L=coords,
                    t=torch.full((1, 1), flow_time, device=coords.device),
                    f=f,
                    n_recycle=1,
                    **initialized,
                )

            hook = trunk.register_forward_pre_hook(capture, with_kwargs=True)
            final_hook = trunk.register_forward_hook(capture_final)
            try:
                forward(coordinates, 1.0)
            finally:
                hook.remove()
                final_hook.remove()
            if len(captured) != 1:
                raise ValueError("Expected one trunk call")
            tokens.append(captured[0][0].float().cpu().numpy())
            final_tokens.append(final_captured[0][0].float().cpu().numpy())
            labels.extend(item["labels"])
            structures.extend([structure] * n_residues)
            geometry.append(item["geometry"])
            # The endpoint X_1 prediction is an identity at t=1. Perturb at
            # t=0.8, with the actual training interpolant and fixed input noise.
            if structure < args.perturb_structures:
                generator = torch.Generator(device=coordinates.device).manual_seed(
                    SEED + structure
                )
                noisy = FLOW_TIME * coordinates + (
                    1 - FLOW_TIME
                ) * NOISE_SCALE * torch.randn(
                    coordinates.shape,
                    device=coordinates.device,
                    dtype=coordinates.dtype,
                    generator=generator,
                )
                perturbation_inputs.append(input_digest({}, noisy))
                baseline = forward(noisy, FLOW_TIME)
                mapping = f["atom_to_token_map"]
                atom_mask = ~f["is_virtual"] & ~f["has_zero_occupancy"]
                targets = torch.as_tensor(
                    AF3SequenceEncoding().encode(item["labels"]),
                    device=coordinates.device,
                    dtype=torch.long,
                )
                baseline_structure = response_by_residue(
                    baseline["X_L"], coordinates, mapping, n_residues, atom_mask
                )
                baseline_sequence = sequence_loss_by_residue(
                    baseline["sequence_logits_I"], targets
                )
                for residue in (n_residues // 4, n_residues // 2, 3 * n_residues // 4):
                    for direction_index in range(3):
                        direction = torch.randn(
                            tokens[-1].shape[-1],
                            device=coordinates.device,
                            generator=generator,
                        )
                        for magnitude in MAGNITUDES:

                            def modify(_module, _args, output):
                                return perturb_token(
                                    output, residue, direction, magnitude
                                )

                            hook = trunk.register_forward_hook(modify)
                            try:
                                changed = forward(noisy, FLOW_TIME)
                            finally:
                                hook.remove()
                            response = response_by_residue(
                                changed["X_L"],
                                baseline["X_L"],
                                mapping,
                                n_residues,
                                atom_mask,
                            )
                            structure_loss = response_by_residue(
                                changed["X_L"],
                                coordinates,
                                mapping,
                                n_residues,
                                atom_mask,
                            )
                            sequence_loss = sequence_loss_by_residue(
                                changed["sequence_logits_I"], targets
                            )
                            if magnitude == 0 and np.max(response) != 0:
                                raise ValueError(
                                    f"Zero perturbation changed output: {response.max()}"
                                )
                            if magnitude == 0 and not np.array_equal(
                                sequence_loss, baseline_sequence
                            ):
                                raise ValueError(
                                    "Zero perturbation changed sequence loss"
                                )
                            responses.append(
                                dict(
                                    structure=structure,
                                    residue=residue,
                                    direction=direction_index,
                                    magnitude=magnitude,
                                    response=response.tolist(),
                                    structure_loss=structure_loss.tolist(),
                                    sequence_loss=sequence_loss.tolist(),
                                    baseline_structure_loss=baseline_structure.tolist(),
                                    baseline_sequence_loss=baseline_sequence.tolist(),
                                    direction_sha256=input_digest({}, direction),
                                )
                            )
        print(
            f"Embedded {structure + 1}/{len(manifest['records'])}: {record['domain_id']}",
            flush=True,
        )
    np.savez_compressed(
        archive,
        encoder=np.concatenate(tokens),
        final=np.concatenate(final_tokens),
        labels=np.array(labels),
        structure=np.array(structures),
        geometry=np.concatenate(geometry),
        input_sha256=np.array(signatures),
        perturbation_input_sha256=np.array(perturbation_inputs),
    )
    write_json(args.work / f"responses_{args.variant}.json", responses)
    write_json(
        args.work / f"provenance_{args.variant}.json",
        dict(
            checkpoint=selected,
            archive_sha256=sha256(archive),
            responses_sha256=sha256(args.work / f"responses_{args.variant}.json"),
            manifest_sha256=sha256(args.work / "manifest.json"),
            script_sha256=sha256(__file__),
            perturbation_protocol=dict(
                intervention="final token trunk output, before upcast/atom decoder and sequence head",
                n_proteins=min(args.perturb_structures, len(manifest["records"])),
                sites="zero-based floor(N/4), floor(N/2), floor(3N/4)",
                directions_per_site=3,
                magnitudes=list(MAGNITUDES),
                perturbation="h_j + alpha * norm(h_j) * normalized Gaussian direction",
                flow_time=FLOW_TIME,
                noise_scale_angstrom=NOISE_SCALE,
                input="t * clean_coordinates + (1-t) * noise_scale * standard Gaussian",
                structure_loss="mean squared Euclidean error per residue over real, observed atoms; no alignment",
                sequence_loss="cross-entropy against observed amino-acid identity, natural logarithm",
                reference="clean input coordinates and observed sequence",
                n_recycle=1,
                fixed="all other tokens, atom skip features, conditioning, input noise",
                sequence_caveat="sequence readout acts independently on each token",
            ),
            seconds=time.monotonic() - start,
            job_id=os.environ.get("SLURM_JOB_ID"),
            gpu=torch.cuda.get_device_name(),
            precision=str(trainer.fabric._precision),
            software={
                name: importlib.metadata.version(name)
                for name in ("torch", "numpy", "atomworks", "biotite")
            },
        ),
    )


def main():
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    # Use the packaged CCD for these standard-amino-acid structures, including
    # AtomWorks helpers whose mirror argument is bound at import time.
    os.environ["CCD_MIRROR_PATH"] = ""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "extract"))
    parser.add_argument("--work", type=Path, default=ROOT / "logs/protein_embeddings")
    parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        default=ROOT / "logs/analysis/scope_best_20260919/manifest.json",
    )
    parser.add_argument(
        "--all-csv", type=Path, default=ROOT / "data/scope/rungs/seed0/scope_4000.csv"
    )
    parser.add_argument(
        "--train-csv", type=Path, default=ROOT / "data/scope/rungs/seed0/scope_2000.csv"
    )
    parser.add_argument(
        "--pdb-dir", type=Path, default=ROOT / "data/scope/rungs/seed0/pdb_4000"
    )
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--perturb-structures", type=int, default=16)
    args = parser.parse_args()
    if args.action == "extract" and args.variant is None:
        parser.error("extract requires --variant")
    {"prepare": prepare, "extract": extract}[args.action](args)


if __name__ == "__main__":
    main()
