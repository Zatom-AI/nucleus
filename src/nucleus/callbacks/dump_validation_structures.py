"""Validation structure export with support for periodic materials."""

import functools
from pathlib import Path
from typing import Literal

import numpy as np
from atomworks.io.utils.io_utils import to_cif_file
from biotite.structure import AtomArray, AtomArrayStack
from foundry.utils.ddp import RankedLogger
from rfd3.trainer.dump_validation_structures import (
    DumpValidationStructuresCallback as RFD3DumpValidationStructuresCallback,
)
from rfd3.utils.io import dump_metadata, dump_trajectories

from nucleus.utils.materials import (
    add_fractional_cif_fields,
    material_cif_extra_categories,
)

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def _prepare_structure_for_cif(
    structure: AtomArray | AtomArrayStack,
) -> AtomArray | AtomArrayStack:
    """Copy a structure and expose material fractional coordinates as CIF fields."""
    structure = structure.copy()
    if isinstance(structure, AtomArray):
        return add_fractional_cif_fields(structure)

    if "material_frac_coords" in structure.get_annotation_categories():
        fractional_coordinates = (
            np.asarray(structure.material_frac_coords, dtype=float) % 1.0
        )
        structure.set_annotation("fract_x", fractional_coordinates[:, 0])
        structure.set_annotation("fract_y", fractional_coordinates[:, 1])
        structure.set_annotation("fract_z", fractional_coordinates[:, 2])
    return structure


def _extra_fields_for_structure(
    structure: AtomArray | AtomArrayStack,
    requested_fields: list[str] | Literal["all"],
) -> list[str] | Literal["all"]:
    """Add available material CIF fields without affecting molecular outputs."""
    if requested_fields == "all":
        return requested_fields

    extra_fields = list(requested_fields)
    categories = structure.get_annotation_categories()
    for field in ("fract_x", "fract_y", "fract_z"):
        if field in categories and field not in extra_fields:
            extra_fields.append(field)
    return extra_fields


def dump_validation_structures(
    atom_arrays: AtomArrayStack | list[AtomArray] | AtomArray,
    base_path: Path,
    one_model_per_file: bool,
    extra_fields: list[str] | Literal["all"] | None = None,
) -> None:
    """Write validation CIFs, including fractional coordinates and unit cells."""
    requested_fields = [] if extra_fields is None else extra_fields

    def write_structure(structure, path):
        prepared_structure = _prepare_structure_for_cif(structure)
        to_cif_file(
            prepared_structure,
            path,
            file_type="cif.gz",
            include_entity_poly=False,
            extra_fields=_extra_fields_for_structure(
                prepared_structure, requested_fields
            ),
            extra_categories=material_cif_extra_categories(prepared_structure),
        )

    base_path = Path(base_path)
    if one_model_per_file:
        if not isinstance(atom_arrays, (AtomArrayStack, list)):
            raise TypeError(
                "one_model_per_file=True requires an AtomArrayStack or list"
            )
        for model_index, atom_array in enumerate(atom_arrays):
            write_structure(atom_array, f"{base_path}_model_{model_index}")
    else:
        write_structure(atom_arrays, base_path)


class DumpValidationStructuresCallback(RFD3DumpValidationStructuresCallback):
    """Dump validation structures while preserving periodic material fields."""

    def on_validation_batch_end(
        self,
        *,
        trainer,
        outputs: dict,
        batch,
        dataset_name: str,
        is_distributed_padding: bool = False,
        **_,
    ):
        """Write predictions and metadata at the configured validation cadence."""
        if is_distributed_padding or (
            not self.dump_predictions and not self.dump_trajectories
        ):
            return

        current_epoch = trainer.state["current_epoch"]
        if current_epoch % self.dump_every_n != 0:
            ranked_logger.debug(
                "Skipping validation batch dump at epoch %s (not every %s epochs)",
                current_epoch,
                self.dump_every_n,
            )
            return

        if "network_output" not in outputs:
            raise AssertionError(
                "Validation outputs must contain `network_output` to dump structures!"
            )

        network_output = outputs["network_output"]
        atom_arrays = outputs["predicted_atom_array_stack"]
        prediction_metadata = outputs["prediction_metadata"]
        example = batch[0]

        build_path = functools.partial(
            self._build_path_from_example_id,
            example_id=example["example_id"],
            epoch=current_epoch,
            dataset_name=dataset_name,
        )

        if self.dump_predictions:
            predictions_path = build_path(dir="predictions")
            dump_validation_structures(
                atom_arrays=atom_arrays,
                base_path=predictions_path,
                one_model_per_file=self.one_model_per_file,
                extra_fields=self.extra_fields,
            )
            if self.verbose:
                ranked_logger.info(
                    "Dumped validation predictions to %s", predictions_path
                )

        if self.dump_prediction_metadata_json and prediction_metadata:
            metadata_path = build_path(dir="predictions")
            dump_metadata(
                prediction_metadata=prediction_metadata,
                base_path=metadata_path,
                one_model_per_file=self.one_model_per_file,
            )
            if self.verbose:
                ranked_logger.info("Dumped validation metadata to %s", metadata_path)

        if self.dump_trajectories:
            denoised_path = build_path(dir="trajectories", extra="_denoised")
            dump_trajectories(
                trajectory_list=network_output["X_denoised_L_traj"],
                atom_array=example["atom_array"],
                base_path=denoised_path,
                align_structures=self.align_trajectories,
            )
            if not self.dump_denoised_trajectories_only:
                dump_trajectories(
                    trajectory_list=network_output["X_noisy_L_traj"],
                    atom_array=example["atom_array"],
                    base_path=build_path(dir="trajectories", extra="_noisy"),
                    align_structures=self.align_trajectories,
                )
            if self.verbose:
                ranked_logger.info(
                    "Dumped validation trajectories to %s", denoised_path
                )
