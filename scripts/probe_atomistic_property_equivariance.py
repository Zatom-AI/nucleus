#!/usr/bin/env python3
"""Probe rotational equivariance of checkpoint atomistic-property predictions."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from zatom2.mlip_evaluation import _build_dataset
from zatom2.run_inference import build_inference_engine

from nucleus.utils.batching import SparseCollate

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "models" / "zatom2" / "configs"


def _random_rotations(count: int, *, seed: int, device: torch.device) -> torch.Tensor:
    """Return one identity followed by deterministic approximately Haar rotations."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrices = torch.randn(count, 3, 3, generator=generator)
    rotations, triangular = torch.linalg.qr(matrices)
    diagonal_sign = torch.diagonal(triangular, dim1=-2, dim2=-1).sign()
    diagonal_sign[diagonal_sign == 0] = 1
    rotations = rotations * diagonal_sign[:, None, :]
    determinant = torch.linalg.det(rotations)
    rotations[:, :, -1] *= determinant[:, None]
    return torch.cat((torch.eye(3).unsqueeze(0), rotations), dim=0).to(device)


def _manifest_dataset_root(manifest: dict, domain: str) -> str:
    paths = [item["path"] for item in manifest["dataset_shards"][domain]]
    return os.path.commonpath(paths)


def _candidate_indices(dataset_size: int, desired: int) -> list[int]:
    """Return deterministic, broadly spaced candidates with nearby fallbacks."""
    anchors = np.linspace(0.03, 0.97, desired)
    candidates: list[int] = []
    for offset in (0, 1, -1, 7, -7, 31, -31):
        for fraction in anchors:
            anchor = int(fraction * max(dataset_size - 1, 0))
            index = min(max(anchor + offset, 0), dataset_size - 1)
            if index not in candidates:
                candidates.append(index)
    return candidates


@dataclass
class ProbeSums:
    structures: int = 0
    atoms: int = 0
    rotations: int = 0
    equiv_difference_squared_sum: float = 0.0
    equiv_expected_squared_sum: float = 0.0
    equiv_observed_squared_sum: float = 0.0
    equiv_dot_sum: float = 0.0
    equiv_vector_cosine_sum: float = 0.0
    equiv_vector_cosine_count: int = 0
    equiv_component_count: int = 0
    identity_difference_squared_sum: float = 0.0
    identity_component_count: int = 0
    target_difference_absolute_sum: float = 0.0
    target_difference_squared_sum: float = 0.0
    target_dot_sum: float = 0.0
    target_predicted_squared_sum: float = 0.0
    target_squared_sum: float = 0.0
    target_vector_cosine_sum: float = 0.0
    target_vector_cosine_count: int = 0
    target_component_count: int = 0
    base_predicted_squared_sum: float = 0.0
    base_target_squared_sum: float = 0.0
    energy_difference_squared_sum: float = 0.0
    energy_difference_absolute_max: float = 0.0
    energy_count: int = 0
    examples: list[dict] = field(default_factory=list)

    def finalize(self) -> dict:
        expected_norm = math.sqrt(self.equiv_expected_squared_sum)
        observed_norm = math.sqrt(self.equiv_observed_squared_sum)
        difference_norm = math.sqrt(self.equiv_difference_squared_sum)
        target_predicted_norm = math.sqrt(self.target_predicted_squared_sum)
        target_norm = math.sqrt(self.target_squared_sum)

        def divide(numerator: float, denominator: float) -> float | None:
            return numerator / denominator if denominator else None

        return {
            "num_structures": self.structures,
            "num_atoms": self.atoms,
            "num_random_rotations_per_structure": self.rotations,
            "equivariance_relative_l2": divide(difference_norm, expected_norm),
            "equivariance_symmetric_relative_l2": divide(
                2 * difference_norm, expected_norm + observed_norm
            ),
            "equivariance_flat_cosine": divide(
                self.equiv_dot_sum, expected_norm * observed_norm
            ),
            "equivariance_mean_atom_cosine": divide(
                self.equiv_vector_cosine_sum, self.equiv_vector_cosine_count
            ),
            "equivariance_force_norm_ratio": divide(observed_norm, expected_norm),
            "equivariance_component_rmse_eV_per_angstrom": math.sqrt(
                self.equiv_difference_squared_sum / self.equiv_component_count
            ),
            "identity_component_rmse_eV_per_angstrom": math.sqrt(
                self.identity_difference_squared_sum / self.identity_component_count
            ),
            "rotated_target_component_mae_eV_per_angstrom": divide(
                self.target_difference_absolute_sum, self.target_component_count
            ),
            "rotated_target_component_rmse_eV_per_angstrom": math.sqrt(
                self.target_difference_squared_sum / self.target_component_count
            ),
            "rotated_target_flat_cosine": divide(
                self.target_dot_sum, target_predicted_norm * target_norm
            ),
            "rotated_target_mean_atom_cosine": divide(
                self.target_vector_cosine_sum, self.target_vector_cosine_count
            ),
            "base_predicted_to_target_rms_ratio": math.sqrt(
                divide(self.base_predicted_squared_sum, self.base_target_squared_sum)
            ),
            "normalized_energy_rotation_rmse": math.sqrt(
                self.energy_difference_squared_sum / self.energy_count
            ),
            "normalized_energy_rotation_max_abs": self.energy_difference_absolute_max,
            "examples": self.examples,
        }


def _accumulate_vector_cosines(
    sums: ProbeSums,
    predicted: torch.Tensor,
    expected: torch.Tensor,
    *,
    equivariance: bool,
) -> None:
    predicted_norm = torch.linalg.vector_norm(predicted, dim=-1)
    expected_norm = torch.linalg.vector_norm(expected, dim=-1)
    valid = (predicted_norm > 1e-10) & (expected_norm > 1e-10)
    cosine_sum = float(
        (
            (predicted * expected).sum(dim=-1)[valid]
            / (predicted_norm[valid] * expected_norm[valid])
        ).sum()
    )
    if equivariance:
        sums.equiv_vector_cosine_sum += cosine_sum
        sums.equiv_vector_cosine_count += int(valid.sum())
    else:
        sums.target_vector_cosine_sum += cosine_sum
        sums.target_vector_cosine_count += int(valid.sum())


def _select_examples(dataset, desired: int, max_atoms: int) -> list[tuple[int, dict]]:
    selected = []
    for index in _candidate_indices(len(dataset), desired):
        example = dataset[index]
        source_count = int(example["ground_truth"]["source_atom_count_B"].item())
        retained_count = int(example["ground_truth"]["retained_atom_count_B"].item())
        property_valid = bool(example["ground_truth"]["property_valid_mask_B"].item())
        forces_valid = bool(example["ground_truth"]["force_valid_mask_L"].all())
        if (
            property_valid
            and forces_valid
            and source_count == retained_count
            and source_count <= max_atoms
        ):
            selected.append((index, example))
        if len(selected) == desired:
            return selected
    raise RuntimeError(
        f"Found only {len(selected)} complete examples with at most {max_atoms} atoms."
    )


def _probe_domain(
    *,
    engine,
    train_cfg,
    manifest: dict,
    domain: str,
    num_structures: int,
    num_rotations: int,
    max_atoms: int,
    seed: int,
) -> dict:
    trainer = engine.trainer
    dataset = _build_dataset(
        train_cfg,
        domain,
        _manifest_dataset_root(manifest, domain),
        max_atoms,
    )
    selected = _select_examples(dataset, num_structures, max_atoms)
    model = trainer.state["model"]
    sums = ProbeSums(rotations=num_rotations)

    with torch.inference_mode():
        for ordinal, (global_index, example) in enumerate(selected):
            collated = SparseCollate()([example])
            collated = trainer.fabric.to_device(collated)
            network_input = trainer._assemble_network_inputs(collated)
            property_input = trainer._single_view_property_input(network_input)
            baseline = model.forward(
                input=property_input,
                property_prediction_atom_types="known",
                allow_missing_property_elemental_references=True,
            )

            coordinates = property_input["X_clean_L"][0]
            rotations = _random_rotations(
                num_rotations,
                seed=seed + 1009 * ordinal + (0 if domain == "omol25" else 100_000),
                device=coordinates.device,
            ).to(dtype=coordinates.dtype)
            rotated_coordinates = torch.einsum("rij,lj->rli", rotations, coordinates)
            rotated_forces = []
            rotated_energies = []
            for rotated_coordinate_view in rotated_coordinates:
                rotated_input = dict(property_input)
                rotated_input["f"] = dict(property_input["f"])
                rotated_input["X_clean_L"] = rotated_coordinate_view.unsqueeze(0)
                rotated_input["X_noisy_L"] = rotated_coordinate_view.unsqueeze(0)
                rotated = model.forward(
                    input=rotated_input,
                    property_prediction_atom_types="known",
                    allow_missing_property_elemental_references=True,
                )
                rotated_forces.append(torch.as_tensor(rotated["forces_L"]).float())
                rotated_energies.append(
                    torch.as_tensor(rotated["normalized_energy_B"])
                    .float()
                    .reshape(-1)[0]
                )

            baseline_force = torch.as_tensor(baseline["forces_L"]).float()
            rotated_force = torch.stack(rotated_forces)
            canonical_force = rotated_force[0]
            expected_force = torch.einsum(
                "rij,lj->rli", rotations.float(), canonical_force
            )
            identity_difference = canonical_force - baseline_force
            random_rotated_force = rotated_force[1:]
            random_expected_force = expected_force[1:]
            equivariance_difference = random_rotated_force - random_expected_force

            target_force = collated["ground_truth"]["force_gt_D_L"][0].float()
            rotated_target = torch.einsum(
                "rij,lj->rli", rotations[1:].float(), target_force
            )
            target_difference = random_rotated_force - rotated_target

            rotated_energy = torch.stack(rotated_energies)
            energy_difference = rotated_energy[1:] - rotated_energy[0]

            sums.structures += 1
            sums.atoms += int(target_force.shape[0])
            sums.identity_difference_squared_sum += float(
                torch.square(identity_difference).sum()
            )
            sums.identity_component_count += identity_difference.numel()
            sums.equiv_difference_squared_sum += float(
                torch.square(equivariance_difference).sum()
            )
            sums.equiv_expected_squared_sum += float(
                torch.square(random_expected_force).sum()
            )
            sums.equiv_observed_squared_sum += float(
                torch.square(random_rotated_force).sum()
            )
            sums.equiv_dot_sum += float(
                (random_rotated_force * random_expected_force).sum()
            )
            sums.equiv_component_count += equivariance_difference.numel()
            _accumulate_vector_cosines(
                sums,
                random_rotated_force,
                random_expected_force,
                equivariance=True,
            )

            sums.target_difference_absolute_sum += float(target_difference.abs().sum())
            sums.target_difference_squared_sum += float(
                torch.square(target_difference).sum()
            )
            sums.target_dot_sum += float((random_rotated_force * rotated_target).sum())
            sums.target_predicted_squared_sum += float(
                torch.square(random_rotated_force).sum()
            )
            sums.target_squared_sum += float(torch.square(rotated_target).sum())
            sums.target_component_count += target_difference.numel()
            _accumulate_vector_cosines(
                sums,
                random_rotated_force,
                rotated_target,
                equivariance=False,
            )

            sums.base_predicted_squared_sum += float(torch.square(baseline_force).sum())
            sums.base_target_squared_sum += float(torch.square(target_force).sum())
            sums.energy_difference_squared_sum += float(
                torch.square(energy_difference).sum()
            )
            sums.energy_difference_absolute_max = max(
                sums.energy_difference_absolute_max,
                float(energy_difference.abs().max()),
            )
            sums.energy_count += energy_difference.numel()
            sums.examples.append(
                {
                    "global_index": global_index,
                    "example_id": str(example["example_id"]),
                    "data_id": str(
                        example.get("extra_info", {}).get(
                            "data_id",
                            example.get("extra_info", {}).get("calc_id", "unknown"),
                        )
                    ),
                    "num_atoms": int(target_force.shape[0]),
                }
            )
            print(
                f"{domain}: {ordinal + 1}/{num_structures} "
                f"{example['example_id']} ({target_force.shape[0]} atoms)",
                flush=True,
            )
    return sums.finalize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-structures", type=int, default=6)
    parser.add_argument("--num-rotations", type=int, default=8)
    parser.add_argument("--max-atoms", type=int, default=192)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env", override=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    checkpoint_path = manifest["checkpoint"]["path"]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        cfg = compose(
            config_name="mlip_evaluation",
            overrides=[
                f"ckpt_path={checkpoint_path}",
                f"out_dir={args.output.parent}",
                f"seed={args.seed}",
                "evaluation.num_workers=0",
            ],
        )

    engine = build_inference_engine(cfg)
    train_cfg = engine.initialize()
    result = {
        "manifest": str(manifest_path),
        "checkpoint": manifest["checkpoint"],
        "use_ema_weights": bool(cfg.get("use_ema_weights", True)),
        "seed": args.seed,
        "num_structures_per_domain": args.num_structures,
        "num_random_rotations_per_structure": args.num_rotations,
        "max_atoms": args.max_atoms,
        "domains": {},
    }
    for domain in ("omol25", "omat24"):
        result["domains"][domain] = _probe_domain(
            engine=engine,
            train_cfg=train_cfg,
            manifest=manifest,
            domain=domain,
            num_structures=args.num_structures,
            num_rotations=args.num_rotations,
            max_atoms=args.max_atoms,
            seed=args.seed,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
