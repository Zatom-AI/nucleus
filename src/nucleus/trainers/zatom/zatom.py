import re
from contextlib import nullcontext

import numpy as np
import torch
from atomworks.constants import ATOMIC_NUMBER_TO_ELEMENT
from atomworks.ml.utils.token import get_token_starts
from beartype.typing import Any, List, Union
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.residues import get_residue_starts
from einops import repeat
from foundry.metrics.losses import Loss
from foundry.metrics.metric import MetricManager
from foundry.utils.ddp import RankedLogger
from foundry.utils.torch import assert_no_nans, assert_same_shape
from lightning_utilities import apply_to_collection
from omegaconf import DictConfig

from nucleus import DEBUG_ANOMALIES
from nucleus.constants import (
    MLIP_PROPERTY_TASK_ID,
    STRUCTURE_PREDICTION_PROPERTY_TASK_ID,
    UNCONDITIONAL_PROPERTY_TASK_ID,
)
from nucleus.inference_input_parsing import uniquify_atom_names_for_cif
from nucleus.metrics.hbonds import get_hbond_metrics
from nucleus.metrics.losses import (
    collate_shared_backbone_gradient_metrics,
    compute_loss_output_gradient_l2_norms,
    compute_shared_backbone_gradient_cosines,
    pop_loss_output_gradient_diagnostics,
    pop_shared_backbone_gradient_diagnostics,
)
from nucleus.trainers.fabric import FabricTrainer
from nucleus.transforms.atomistic_property_transforms import (
    OMAT24_DOMAIN_ID,
    OMOL25_DOMAIN_ID,
    infer_property_task_name,
)
from nucleus.utils.batching import SparseCollate
from nucleus.utils.datasets import SANITIZED_COORDINATE_NAN_COUNT_KEY
from nucleus.utils.materials import (
    add_fractional_cif_fields,
    annotate_material_properties,
    cartesian_to_fractional,
    fit_periodic_rigid_alignment,
    fractional_to_cartesian,
    get_material_lattice,
    get_material_mask,
    lattice_vectors_from_lengths_angles,
    material_cif_extra_categories,
    unnormalize_lattice_lengths,
    wrap_cartesian_coordinates,
)
from nucleus.utils.torch import cast_float64_tensors_to_float32
from nucleus.utils.trainer import cleanup_virtual_atoms_and_assign_atom_name_elements

try:
    from rfd3na.metrics.design_metrics import get_all_backbone_metrics
    from rfd3na.trainer.recycling import get_recycle_schedule
    from rfd3na.trainer.trainer_utils import (
        _build_atom_array_stack,
        _reassign_unindexed_token_chains,
        _remap_outputs,
        _reorder_dict,
        process_unindexed_outputs,
    )
    from rfd3na.utils.io import (
        build_stack_from_atom_array_and_batched_coords,
    )
except ImportError:
    raise ImportError(
        "Failed to import RFD3NA modules. Please ensure that the `rc-foundry[all]` package is installed and up to date."
    )

global_logger = RankedLogger(__name__, rank_zero_only=False)

_SHARED_BACKBONE_BLOCK_PATTERN = re.compile(
    r"(?:^|\.)diffusion_module\.backbone\."
    r"(token_trunk|atom_decoder)\.blocks\.(\d+)(?:\.|$)"
)


def _select_last_shared_backbone_block_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[str, ...], tuple[torch.nn.Parameter, ...]]:
    """Select the last token-trunk and atom-decoder blocks."""
    candidates = []
    for name, parameter in model.named_parameters():
        match = _SHARED_BACKBONE_BLOCK_PATTERN.search(name)
        if match is not None and parameter.requires_grad:
            candidates.append((match.group(1), int(match.group(2)), name, parameter))
    if not candidates:
        raise RuntimeError(
            "Shared-backbone gradient logging could not find trainable parameters "
            "under the final token-trunk and atom-decoder blocks."
        )

    last_block_by_stage = {
        stage: max(
            block_index
            for candidate_stage, block_index, _, _ in candidates
            if candidate_stage == stage
        )
        for stage in {stage for stage, _, _, _ in candidates}
    }
    selected = sorted(
        (
            (name, parameter)
            for stage, block_index, name, parameter in candidates
            if block_index == last_block_by_stage[stage]
        ),
        key=lambda item: item[0],
    )
    names, parameters = zip(*selected, strict=True)
    return tuple(names), tuple(parameters)


class ZatomDesignTrainer(FabricTrainer):
    """A Trainer primarily for unique things like saving outputs and parsing inputs.

    Args:
        allow_sequence_outputs (bool): Whether to allow sequence outputs in the model.
        cleanup_guideposts (bool): Whether to clean up guidepost atoms in the output.
        cleanup_virtual_atoms (bool): Whether to clean up virtual atoms in the output.
        read_sequence_from_sequence_head (bool): Whether to read the predicted sequence from the sequence head
            (instead of from the token indices, which is the default). This is useful if the sequence head is performing well, and we want to use its predictions directly; however, it can be unstable, and so is disabled by default.
        output_full_json (bool): Whether to output the full JSON specification in the output metadata (in addition to the task name). This is useful for debugging and analysis, but results in larger output files, so is disabled by default.
        association_scheme (str): Association scheme to use for the sequence head. Defaults to "atom14".
        material_coordinate_output_source (str): Source for generated material atom coordinates in final outputs.
            ``"euclidean"`` derives fractional CIF fields from the diffusion rollout coordinates; ``"fractional"``
            uses the model's fractional-coordinate head and converts it back to Cartesian coordinates.
        compute_non_clash_metrics_for_diffused_region_only (bool): Whether to compute non-clash-based metrics (e.g. LDDT) for the diffused region only, as identified by the `diffused_index_map` in the output metadata dictionary. This can be useful for obtaining a more fine-grained understanding of model performance in the diffused region vs. the rest of the structure, especially during early training when the diffused region may be of particularly low quality; however, it is disabled by default.
        seed (int | None): The random seed used for this design, which will be dumped in the output JSON.
            If None, no value will be dumped. Currently, this parameter is deprecated and has no effect.
        n_recycles_train (int | None): The number of recycles to use during training. If None, no recycling will be used during training. Note that this only sets the maximum number of recycles; the actual number of recycles for each batch will be determined by the recycle schedule, which is sampled at initialization to ensure all GPUs within a distributed batch see the same number of recycles.
        loss (DictConfig | dict | None): The loss configuration dictionary. If None, no loss will be computed.
        metrics (DictConfig | dict | None): The metrics configuration dictionary. If None, no metrics will be computed.
        log_loss_output_gradient_norms (bool): Whether to log the L2 norm of each
            weighted loss component's gradient with respect to its prediction tensor.
        loss_output_gradient_norm_log_every_n_steps (int): Number of training
            batches between loss output-gradient norm measurements.
        log_shared_backbone_gradient_cosines (bool): Whether to compare
            atomistic-property gradients with flow gradients from low-, mid-, and
            high-noise bins on the last shared token-trunk and atom-decoder blocks.
            Rank-local measurements are collated into all-rank means, standard
            deviations, conflict fractions, and valid-rank fractions.
        shared_backbone_gradient_cosine_log_every_n_steps (int): Number of
            optimizer steps between shared-backbone gradient measurements.
        collate_validation_batch_before_device_transfer (bool): Whether to keep
            validation examples on CPU until after sparse collation. Disabled by
            default.
    """

    collate_train_batch_before_device_transfer = True

    def __init__(
        self,
        allow_sequence_outputs,
        cleanup_guideposts,
        cleanup_virtual_atoms,
        read_sequence_from_sequence_head,
        output_full_json,
        association_scheme,
        material_coordinate_output_source: str = "fractional",
        wrap_material_euclidean_coordinates: bool = True,
        compute_non_clash_metrics_for_diffused_region_only=False,
        seed=None,  # Deprecated
        n_recycles_train: int | None = None,
        tensor_validation_every_n_steps: int | None = None,
        coordinate_sanitization_log_every_n_steps: int = 100,
        loss: DictConfig | dict | None = None,
        metrics: DictConfig | dict | None = None,
        log_loss_output_gradient_norms: bool = False,
        loss_output_gradient_norm_log_every_n_steps: int = 1,
        log_shared_backbone_gradient_cosines: bool = False,
        shared_backbone_gradient_cosine_log_every_n_steps: int = 250,
        collate_validation_batch_before_device_transfer: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.allow_sequence_outputs = allow_sequence_outputs
        self.cleanup_guideposts = cleanup_guideposts
        self.cleanup_virtual_atoms = cleanup_virtual_atoms
        self.read_sequence_from_sequence_head = read_sequence_from_sequence_head
        self.output_full_json = output_full_json
        self.compute_non_clash_metrics_for_diffused_region_only = (
            compute_non_clash_metrics_for_diffused_region_only
        )
        self.association_scheme = association_scheme
        self.material_coordinate_output_source = material_coordinate_output_source
        self.wrap_material_euclidean_coordinates = wrap_material_euclidean_coordinates
        self.seed = None

        # Logging and validation settings
        self.log_loss_output_gradient_norms = log_loss_output_gradient_norms
        if loss_output_gradient_norm_log_every_n_steps < 1:
            raise ValueError(
                "loss_output_gradient_norm_log_every_n_steps must be positive."
            )
        self.loss_output_gradient_norm_log_every_n_steps = (
            loss_output_gradient_norm_log_every_n_steps
        )
        self.log_shared_backbone_gradient_cosines = log_shared_backbone_gradient_cosines
        if shared_backbone_gradient_cosine_log_every_n_steps < 1:
            raise ValueError(
                "shared_backbone_gradient_cosine_log_every_n_steps must be positive."
            )
        self.shared_backbone_gradient_cosine_log_every_n_steps = (
            shared_backbone_gradient_cosine_log_every_n_steps
        )
        self.collate_validation_batch_before_device_transfer = (
            collate_validation_batch_before_device_transfer
        )
        self._shared_backbone_gradient_parameter_names = None
        self._shared_backbone_gradient_parameters = None

        if (
            tensor_validation_every_n_steps is not None
            and tensor_validation_every_n_steps <= 0
        ):
            raise ValueError(
                "tensor_validation_every_n_steps must be positive or None."
            )
        if coordinate_sanitization_log_every_n_steps <= 0:
            raise ValueError(
                "coordinate_sanitization_log_every_n_steps must be positive."
            )
        self.tensor_validation_every_n_steps = tensor_validation_every_n_steps
        self.coordinate_sanitization_log_every_n_steps = (
            coordinate_sanitization_log_every_n_steps
        )
        self._sanitized_coordinate_nan_count = 0

        # (Initialize recycle schedule upfront so all GPU's can sample the same number of recycles within a batch)
        self.n_recycles_train = n_recycles_train
        self.recycle_schedule = get_recycle_schedule(
            max_cycle=n_recycles_train,
            n_epochs=self.max_epochs,  # Set by FabricTrainer
            n_train=self.n_examples_per_epoch,  # Set by FabricTrainer
            world_size=self.fabric.world_size,
        )  # [n_epochs, n_examples_per_epoch // world_size]

        # Metrics
        # (We could have instantiated loss and metrics recursively, but we prioritize being explicit)
        enabled_metrics = {
            name: metric_config
            for name, metric_config in (metrics or {}).items()
            if metric_config is not None
        }
        self.metrics = (
            MetricManager.instantiate_from_hydra(metrics_cfg=enabled_metrics)
            if enabled_metrics
            else None
        )
        # Loss (full precision)
        enabled_losses = {
            name: loss_config
            for name, loss_config in (loss or {}).items()
            if loss_config is not None
        }
        with torch.autocast(device_type=self.fabric.device.type, enabled=False):
            self.loss = Loss(**enabled_losses) if enabled_losses else None

    def _assemble_network_inputs(self, example: dict) -> dict:
        """
        Assemble and validate the network inputs.

        Args:
            example (dict): The input example containing the necessary fields for assembling the network inputs.

        Returns:
            dict: A dictionary containing the assembled network inputs, including the noisy coordinates, time step, and features, among others.
        """
        assert_same_shape(example["coord_atom_lvl_to_be_noised"], example["noise"])
        coord = example["coord_atom_lvl_to_be_noised"]
        noise = example["noise"]
        X_noisy_L = coord + noise
        X_clean_L = coord
        if X_noisy_L.ndim == 2:
            X_noisy_L = X_noisy_L.unsqueeze(0)
            X_clean_L = X_clean_L.unsqueeze(0)

        network_input = {
            "X_noisy_L": X_noisy_L,
            "X_clean_L": X_clean_L,
            "t": example["t"],
            "f": example["feats"],
        }
        if "ground_truth" in example:
            network_input["self_embedding_ground_truth"] = {
                k: example["ground_truth"][k]
                for k in (
                    "sequence_gt_I",
                    "atom_type_gt_I",
                    "atom_type_valid_mask",
                )
                if k in example["ground_truth"]
            }
        if "t_L" in example:
            network_input["t_L"] = example["t_L"]
        if "t_I" in example:
            network_input["t_I"] = example["t_I"]
        if "material_token_mask" in example["feats"]:
            network_input["f"]["is_material_token"] = example["feats"][
                "material_token_mask"
            ].unsqueeze(-1)
        if "material_atom_mask" in example["feats"]:
            network_input["f"]["is_material_atom"] = example["feats"][
                "material_atom_mask"
            ].unsqueeze(-1)

        should_validate_tensors = not self.state["model"].training or (
            self.tensor_validation_every_n_steps is not None
            and self.state["global_step"] % self.tensor_validation_every_n_steps == 0
        )
        if should_validate_tensors:
            assert_no_nans(
                network_input["X_noisy_L"],
                msg=f"network_input (X_noisy_L) for example_id: {example['example_id']}",
            )
            assert_no_nans(
                network_input["f"],
                msg=f"NaN detected in `feats` for example_id: {example['example_id']}",
            )

        return network_input

    def training_step(
        self,
        batch: Any,
        batch_idx: int,
        is_accumulating: bool,
    ) -> None:
        """Training step, running forward and backward passes.

        Args:
            batch: The current batch; can be of any form.
            batch_idx: The index of the current batch.
            is_accumulating: Whether we are accumulating gradients (i.e., not yet calling optimizer.step()).
                If this is the case, we should skip the synchronization during the backward pass.

        Returns:
            None; we call `loss.backward()` directly, and store the outputs in `self._current_train_return`.
        """
        model = self.state["model"]
        assert model.training, "Model must be training!"

        # Recycling
        # (Number of recycles for the current batch; shared across all GPUs within a distributed batch)
        n_cycle = self.recycle_schedule[self.state["current_epoch"], batch_idx].item()

        training_ctx = (
            torch.autograd.set_detect_anomaly(True)
            if DEBUG_ANOMALIES
            else nullcontext()
        )

        with self.fabric.no_backward_sync(model, enabled=is_accumulating):
            with training_ctx:
                max_num_tokens = self.state["train_cfg"].datasets.get("max_num_tokens")
                example = SparseCollate(max_num_tokens=max_num_tokens)(
                    batch if isinstance(batch, list) else [batch]
                )

                sanitized_counts = example.pop(SANITIZED_COORDINATE_NAN_COUNT_KEY, None)
                if sanitized_counts is None:
                    # Preserve safe behavior for loaders that have not enabled
                    # worker-side sanitation, without a synchronous GPU check.
                    for key in ("coord_atom_lvl_to_be_noised", "noise"):
                        if key in example:
                            example[key] = torch.nan_to_num(example[key])
                else:
                    self._sanitized_coordinate_nan_count += sum(sanitized_counts)

                if (
                    self.state["global_step"] + 1
                ) % self.coordinate_sanitization_log_every_n_steps == 0:
                    sanitized_count = torch.tensor(
                        self._sanitized_coordinate_nan_count,
                        dtype=torch.long,
                        device=self.fabric.device,
                    )
                    sanitized_count = self.fabric.all_reduce(
                        sanitized_count, reduce_op="sum"
                    )
                    if self.fabric.is_global_zero:
                        self.fabric.log_dict(
                            {"train/sanitized_coordinate_nan_count": (sanitized_count)},
                            step=self.state["global_step"],
                        )
                    self._sanitized_coordinate_nan_count = 0

                if self.fabric.device.type == "mps":
                    example = cast_float64_tensors_to_float32(example)
                example = self.fabric.to_device(example)

                if max_num_tokens is not None:
                    batch_util_rate = example["feats"]["batch_util_rate"]
                    batch_util_rate = self.fabric.all_reduce(
                        batch_util_rate, reduce_op="max"
                    )
                    if self.fabric.is_global_zero:
                        self.fabric.log_dict(
                            {"train/batch_util_rate": batch_util_rate},
                            step=self.state["global_step"],
                        )

                network_input = self._assemble_network_inputs(example)

                should_compute_shared_gradient_cosines = (
                    self.log_shared_backbone_gradient_cosines
                    and not is_accumulating
                    and self.state["global_step"]
                    % self.shared_backbone_gradient_cosine_log_every_n_steps
                    == 0
                )
                loss_extra_info = self._assemble_loss_extra_info(example)
                loss_extra_info["optimizer_step"] = self.state["global_step"]

                if should_compute_shared_gradient_cosines:
                    if self._shared_backbone_gradient_parameters is None:
                        (
                            self._shared_backbone_gradient_parameter_names,
                            self._shared_backbone_gradient_parameters,
                        ) = _select_last_shared_backbone_block_parameters(model)
                        parameter_count = sum(
                            parameter.numel()
                            for parameter in self._shared_backbone_gradient_parameters
                        )
                        if self.fabric.is_global_zero:
                            global_logger.info(
                                "Shared-backbone gradient diagnostics use %d "
                                "parameters from the final token-trunk/atom-decoder "
                                "blocks (%s ...).",
                                parameter_count,
                                self._shared_backbone_gradient_parameter_names[0],
                            )

                    loss_extra_info["collect_shared_backbone_gradient_diagnostics"] = (
                        True
                    )
                    rng_devices = []
                    if self.fabric.device.type == "cuda":
                        device_index = self.fabric.device.index
                        rng_devices = [
                            torch.cuda.current_device()
                            if device_index is None
                            else device_index
                        ]
                    # Fabric requires its own backward entry point. Use a separate
                    # no-sync graph so the four diagnostic backwards neither touch
                    # DDP reducer state nor perturb the ordinary training graph.
                    with torch.random.fork_rng(devices=rng_devices):
                        with self.fabric.no_backward_sync(model, enabled=True):
                            diagnostic_network_output = model.forward(
                                input=network_input, n_cycle=n_cycle
                            )
                            self.loss(
                                network_input=network_input,
                                network_output=diagnostic_network_output,
                                loss_input=loss_extra_info,
                            )
                            pop_loss_output_gradient_diagnostics(self.loss)
                            shared_gradient_diagnostics = (
                                pop_shared_backbone_gradient_diagnostics(self.loss)
                            )
                            shared_gradient_metrics = (
                                compute_shared_backbone_gradient_cosines(
                                    shared_gradient_diagnostics,
                                    self._shared_backbone_gradient_parameters,
                                    backward=self.fabric.backward,
                                )
                            )
                    shared_gradient_metrics = collate_shared_backbone_gradient_metrics(
                        shared_gradient_metrics,
                        lambda tensor: self.fabric.all_reduce(tensor, reduce_op="sum"),
                        world_size=self.fabric.world_size,
                    )
                    if shared_gradient_metrics and self.fabric.is_global_zero:
                        self.fabric.log_dict(
                            {
                                f"train/shared_backbone_gradient/{name}": value
                                for name, value in shared_gradient_metrics.items()
                            },
                            step=self.state["global_step"],
                        )
                    del diagnostic_network_output, shared_gradient_diagnostics

                # Forward pass (without rollout)
                network_output = model.forward(input=network_input, n_cycle=n_cycle)
                if (
                    self.tensor_validation_every_n_steps is not None
                    and self.state["global_step"] % self.tensor_validation_every_n_steps
                    == 0
                ):
                    assert_no_nans(
                        network_output,
                        msg=f"network_output for example_id: {example['example_id']}",
                    )

                loss_extra_info["collect_shared_backbone_gradient_diagnostics"] = False

                total_loss, loss_dict_batched = self.loss(
                    network_input=network_input,
                    network_output=network_output,
                    # TODO: Rename `loss_input` to `extra_info` to pattern-match metrics
                    loss_input=loss_extra_info,
                )

                gradient_diagnostics = pop_loss_output_gradient_diagnostics(self.loss)
                should_log_gradient_norms = (
                    self.log_loss_output_gradient_norms
                    and self.fabric.is_global_zero
                    and batch_idx % self.loss_output_gradient_norm_log_every_n_steps
                    == 0
                )
                if should_log_gradient_norms:
                    gradient_norms = compute_loss_output_gradient_l2_norms(
                        gradient_diagnostics
                    )
                    self.fabric.log_dict(
                        {
                            f"train/loss_output_gradient_l2_norm/{name}": value
                            for name, value in gradient_norms.items()
                        },
                        step=self.state["global_step"],
                    )

                # Backward pass
                self._backward_training_loss(total_loss)

            # ... store the outputs without gradients for use in logging, callbacks, learning rate schedulers, etc.
            training_return = {
                "total_loss": total_loss,
                "loss_dict": loss_dict_batched,
            }
            if "model_diagnostics" in network_output:
                training_return["model_diagnostics"] = network_output[
                    "model_diagnostics"
                ]
            self._current_train_return = apply_to_collection(
                training_return,
                dtype=torch.Tensor,
                function=lambda x: x.detach(),
            )

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
        compute_metrics: bool = True,
    ) -> dict:
        """Validation step, running forward pass and computing validation metrics.

        Args:
            batch: The current batch; can be of any form.
            batch_idx: The index of the current batch. Currently unused, but we include it here for potential use in logging, etc.
            compute_metrics: Whether to compute metrics. If False, we will not compute metrics, and the output will be None.
                Set to False during the inference pipeline, where we need the network output but cannot compute metrics (since we
                do not have the ground truth).

        Returns:
            dict: Output dictionary containing the validation metrics and network output.
        """
        model = self.state["model"]
        assert not model.training, "Model must be in evaluation mode during validation!"

        # Detect multiple examples in a validation batch
        multi_example_batch = isinstance(batch, list) and len(batch) > 1
        if multi_example_batch:
            # TODO: Investigate how to handle multiple validation examples in a batch.
            global_logger.warning(
                f"Validation batch contains more than one example (batch size: {len(batch)}); only the first example will be used for validation metrics and outputs. Example IDs in this batch: {[b['example_id'] for b in batch]}"
            )
            batch = batch[0]

        example = SparseCollate()(batch if isinstance(batch, list) else [batch])
        example["atom_array"] = example["atom_array"][0]
        if getattr(self, "collate_validation_batch_before_device_transfer", False):
            if self.fabric.device.type == "mps":
                example = cast_float64_tensors_to_float32(example)
            example = self.fabric.to_device(example)

        network_input = self._assemble_network_inputs(example)

        assert_no_nans(
            network_input,
            msg=f"network_input for example_id: {example['example_id']}",
        )

        is_mlip_property_inference = self._is_mlip_property_inference(example)
        if is_mlip_property_inference:
            global_logger.info(
                "Running MLIP inference through one clean-endpoint "
                "predict_properties() evaluation."
            )
            property_input = self._single_view_property_input(network_input)
            # Call through forward() so an eval-mode EMA wrapper dispatches to its
            # shadow model. Zatom2.forward then delegates to predict_properties().
            network_output = model.forward(
                input=property_input,
                property_prediction_atom_types="known",
            )
            network_output = self._format_property_inference_output(
                network_output,
                property_input=property_input,
            )
        else:
            # ... forward pass (with rollout)
            # (Note that forward() passes to the EMA/shadow model if the model is not training)
            network_output = model.forward(
                input=network_input,
                coord_atom_lvl_to_be_noised=example["coord_atom_lvl_to_be_noised"],
            )

        assert_no_nans(
            network_output,
            msg=f"network_output for example_id: {example['example_id']}",
        )

        # ... Convert output to a stack of atom arrays
        predicted_atom_array_stack, prediction_metadata = (
            self._build_predicted_atom_array_stack(network_output, example)
        )

        metrics_extra_info = self._assemble_metrics_extra_info(example, network_output)
        ground_truth_atom_array_stack = build_stack_from_atom_array_and_batched_coords(
            metrics_extra_info["X_gt_L"], example.get("atom_array", None)
        )

        metrics_output = {}
        if compute_metrics:
            assert self.metrics is not None, "Metrics are not defined!"

            metric_network_input = network_input
            metric_network_output = network_output
            if (
                is_mlip_property_inference
                and self.fabric.device.type == "mps"
                and getattr(
                    self, "collate_validation_batch_before_device_transfer", False
                )
            ):
                # MLIP energy metrics use float64 for precise absolute errors,
                # which MPS cannot represent. Keep this CPU metric copy scoped
                # to the opted-in local validation path.
                metric_network_input = apply_to_collection(
                    network_input, torch.Tensor, lambda tensor: tensor.detach().cpu()
                )
                metric_network_output = apply_to_collection(
                    network_output, torch.Tensor, lambda tensor: tensor.detach().cpu()
                )
                metrics_extra_info = apply_to_collection(
                    metrics_extra_info,
                    torch.Tensor,
                    lambda tensor: tensor.detach().cpu(),
                )

            metrics_output = self.metrics(
                network_input=metric_network_input,
                network_output=metric_network_output,
                extra_info=metrics_extra_info,
                # (Uses the permuted ground truth after symmetry resolution)
                ground_truth_atom_array_stack=ground_truth_atom_array_stack,
                predicted_atom_array_stack=predicted_atom_array_stack,
                prediction_metadata=prediction_metadata,
            )

            if "X_gt_index_to_X" in metrics_extra_info:
                # Remap outputs to minimize error with ground truth
                # TODO: Remap before computing metrics, so that we can avoid pass `extra_info` to metrics (we instead just pass the remapped prediction)
                mapping = metrics_extra_info["X_gt_index_to_X"]  # [D, L]
                network_output["X_L"] = _remap_outputs(network_output["X_L"], mapping)

            # Avoid gradients in stored values to prevent memory leaks
            if metrics_output is not None:
                metrics_output = apply_to_collection(
                    metrics_output, torch.Tensor, lambda x: x.detach()
                )

        if network_output is not None:
            network_output = apply_to_collection(
                network_output, torch.Tensor, lambda x: x.detach()
            )

        return {
            "metrics_output": metrics_output,
            "network_output": network_output,
            "predicted_atom_array_stack": predicted_atom_array_stack,
            "ground_truth_atom_array_stack": ground_truth_atom_array_stack,
            "metrics_extra_info": metrics_extra_info,
            "metric_network_input": network_input,
            "prediction_metadata": prediction_metadata,
        }

    @staticmethod
    def _is_mlip_property_inference(example: dict) -> bool:
        """Return whether a batch requests clean-endpoint MLIP prediction."""
        property_task_ids = example.get("ground_truth", {}).get("property_task_id_B")
        if property_task_ids is None:
            return False
        property_task_ids = torch.as_tensor(property_task_ids)
        return property_task_ids.numel() > 0 and bool(
            torch.all(property_task_ids == MLIP_PROPERTY_TASK_ID).item()
        )

    @staticmethod
    def _single_view_property_input(network_input: dict) -> dict:
        """Create a shallow one-view input for direct clean-endpoint prediction."""
        property_input = dict(network_input)
        for key in ("X_clean_L", "X_noisy_L"):
            coordinates = property_input.get(key)
            if torch.is_tensor(coordinates) and coordinates.ndim == 3:
                property_input[key] = coordinates[:1]
        return property_input

    @staticmethod
    def _format_property_inference_output(
        property_output: dict,
        *,
        property_input: dict,
    ) -> dict:
        """Adapt public property API tensors to the generative output contract."""
        network_output = dict(property_output)
        clean_coordinates = property_input["X_clean_L"]
        if clean_coordinates.ndim == 2:
            clean_coordinates = clean_coordinates.unsqueeze(0)
        if clean_coordinates.ndim != 3 or clean_coordinates.shape[0] != 1:
            raise ValueError(
                "Direct MLIP inference requires one clean coordinate view with "
                f"shape [1, L, 3], got {tuple(clean_coordinates.shape)}."
            )
        network_output["X_L"] = clean_coordinates

        for key in ("energy_B", "normalized_energy_B"):
            value = network_output.get(key)
            if torch.is_tensor(value) and value.ndim == 1:
                network_output[key] = value.unsqueeze(0)
        for key in ("forces_L", "normalized_forces_L"):
            value = network_output.get(key)
            if torch.is_tensor(value) and value.ndim == 2:
                network_output[key] = value.unsqueeze(0)

        # Preserve compatibility with optional trajectory dumping. These are one
        # frame rather than a sampler rollout, and both represent the clean input.
        network_output["X_noisy_L_traj"] = [clean_coordinates]
        network_output["X_denoised_L_traj"] = [clean_coordinates]
        return network_output

    def _assemble_loss_extra_info(self, example: dict) -> dict:
        """
        Assemble metadata arguments to the loss function (incremental to the network inputs and outputs).

        Args:
            example (dict): The input example containing the necessary fields for assembling the loss extra info.

        Returns:
            dict: A dictionary containing the assembled extra info for the loss function, including the ground truth coordinates, mask, sequence information, and token indices, among others.
        """
        # ... reshape
        coord = example["coord_atom_lvl_to_be_noised"]
        if coord.ndim == 2:
            diffusion_batch_size = 1
            coord_in = coord.unsqueeze(0)
        else:
            diffusion_batch_size = coord.shape[0]
            coord_in = coord

        X_gt_L = repeat(
            example["ground_truth"]["coord_atom_lvl"],
            "l c -> d l c",
            d=diffusion_batch_size,
        )  # [L, 3] -> [D, L, 3] with broadcasting

        ret = {
            "X_gt_L": X_gt_L,  # [D, L, 3]
            "X_gt_L_in_input_frame": coord_in,  # [D, L, 3] for no-align loss
            "crd_mask_L": example["ground_truth"]["mask_atom_lvl"],  # [L,]
            "is_original_unindexed_token": example["ground_truth"][
                "is_original_unindexed_token"
            ],  # [I,]
            # Sequence information:
            "seq_token_lvl": example["ground_truth"]["sequence_gt_I"],  # [I, 32]
            "sequence_valid_mask": example["ground_truth"][
                "sequence_valid_mask"
            ],  # [I,]
            "atom_type_gt_I": example["ground_truth"].get(
                "atom_type_gt_I",
                torch.zeros_like(example["ground_truth"]["sequence_gt_I"]),
            ),
            "atom_type_valid_mask": example["ground_truth"].get(
                "atom_type_valid_mask",
                torch.zeros_like(
                    example["ground_truth"]["sequence_valid_mask"],
                    dtype=torch.bool,
                ),
            ),
            "fractional_coord_gt_L": example["ground_truth"].get(
                "fractional_coord_gt_L",
                torch.zeros_like(example["ground_truth"]["coord_atom_lvl"]),
            ),
            "fractional_coord_valid_mask_L": example["ground_truth"].get(
                "fractional_coord_valid_mask_L",
                torch.zeros_like(
                    example["ground_truth"]["mask_atom_lvl"],
                    dtype=torch.bool,
                ),
            ),
        }

        # When enabled, the OMol25/OMat24 property transform always emits these
        # targets, including typed placeholders for missing labels.
        # Keep the keys explicit so mixed-domain loss inputs do not depend on
        # optional-key intersection during collation.
        atomistic_property_keys = (
            "energy_gt_B",
            "energy_valid_mask_B",
            "energy_present_mask_B",
            "force_present_mask_B",
            "force_gt_D_L",
            "force_valid_mask_L",
            "domain_id_B",
            "domain_valid_mask_B",
            "source_atom_count_B",
            "retained_atom_count_B",
            "total_charge_B",
            "spin_multiplicity_B",
            "has_total_charge_B",
            "has_spin_multiplicity_B",
            "charge_valid_mask_B",
            "spin_valid_mask_B",
            "property_valid_mask_B",
            "property_task_id_B",
            "property_task_valid_mask_B",
        )
        ret.update(
            {
                key: example["ground_truth"][key]
                for key in atomistic_property_keys
                if key in example["ground_truth"]
            }
        )

        tok_idx = example["feats"]["atom_to_token_map"]
        batch_token_lvl = example["feats"]["batch_token_lvl"]
        B = int(batch_token_lvl.max().item() + 1)

        def get_system_level(key, default_shape, dtype):
            """
            Helper function to get system-level ground truth values,
            which may be stored either at the batch level (with suffix "_B")
            or the token level (with suffix "_I"). If stored at the token level,
            we take the value from the first token of each batch element.
            If neither is present, we return a default value (e.g., zeros).
            """
            if key + "_B" in example["ground_truth"]:
                return example["ground_truth"][key + "_B"]
            elif key + "_I" in example["ground_truth"]:
                val_I = example["ground_truth"][key + "_I"]
                first_indices = torch.zeros(
                    (B,), device=tok_idx.device, dtype=tok_idx.dtype
                ).scatter_reduce(
                    0, batch_token_lvl, tok_idx, reduce="amin", include_self=False
                )  # (B,)
                return val_I[first_indices]
            else:
                return torch.zeros(default_shape, device=tok_idx.device, dtype=dtype)

        ret.update(
            {
                "lattice_lengths_scaled_gt_B": get_system_level(
                    "lattice_lengths_scaled_gt",
                    (B, 3),
                    example["ground_truth"]["coord_atom_lvl"].dtype,
                ),
                "lattice_angles_radians_gt_B": get_system_level(
                    "lattice_angles_radians_gt",
                    (B, 3),
                    example["ground_truth"]["coord_atom_lvl"].dtype,
                ),
                "lattice_valid_mask_B": get_system_level(
                    "lattice_valid_mask", (B,), torch.bool
                ),
                # Scalar offset to add to token indices to index into the correct position
                # in the concatenated `is_original_unindexed_token` tensor (only needed
                # when using multiple examples per batch, which is currently only the case
                # for training):
                "is_original_unindexed_token_batch_offset": example["ground_truth"][
                    "is_original_unindexed_token_batch_offset"
                ],
                # Token indices for each batch element.
                "atom_to_token_map_per_batch_element": example["ground_truth"][
                    "atom_to_token_map_per_batch_element"
                ],
            }
        )
        return ret

    def _assemble_metrics_extra_info(self, example: dict, network_output: dict) -> dict:
        """
        Prepare extra info for the metrics.

        Args:
            example (dict): The input example containing the necessary fields for assembling the metrics extra info.
            network_output (dict): The output from the network, which may contain additional information needed for the metrics.

        Returns:
            dict: A dictionary containing the assembled extra info for the metrics, including the ground truth coordinates, mask, sequence information, token indices, and any additional metadata from the example dictionary.
        """
        # We need the same information as for the loss...
        metrics_extra_info = self._assemble_loss_extra_info(example)

        # ... and possibly some additional metadata from the example dictionary
        # TODO: Generalize, so we always use the `extra_info` key, rather than unpacking the ground truth as well
        metrics_extra_info.update(
            {
                # TODO: Remove, instead using `extra_info` for all keys
                **{
                    k: example["ground_truth"][k]
                    for k in [
                        "interfaces_to_score",
                        "pn_units_to_score",
                        "chain_iid_token_lvl",
                    ]
                    if k in example["ground_truth"]
                },
                "example_id": example[
                    "example_id"
                ],  # We require the example ID for logging
                # Preserve inference JSON fields for metrics that compare a
                # generated structure against its requested conditioning target.
                "specification": example.get("specification", {}),
                # (From the parser)
                **example.get("extra_info", {}),
                "sampled_condition_name": (
                    example["sampled_condition_name"][0]
                    if isinstance(
                        example.get("sampled_condition_name"),
                        (list, tuple),
                    )
                    and len(example["sampled_condition_name"]) == 1
                    else example.get("sampled_condition_name")
                ),
            }
        )

        existing_tags = metrics_extra_info.get("metrics_tags", ())
        if isinstance(existing_tags, str):
            existing_tags = (existing_tags,)
        metrics_extra_info["metrics_tags"] = sorted(
            self._atomistic_metrics_tags(example) | set(existing_tags)
        )

        # (Create a shallow copy to avoid modifying the original dictionary)
        return {**metrics_extra_info}

    @staticmethod
    def _atomistic_metrics_tags(example: dict) -> set[str]:
        """Build metric-routing tags from atomistic domain and task metadata."""
        ground_truth = example.get("ground_truth", {})
        tags: set[str] = set()

        domain_ids = ground_truth.get("domain_id_B")
        if domain_ids is not None:
            unique_domains = set(torch.as_tensor(domain_ids).detach().cpu().tolist())
            if unique_domains == {OMOL25_DOMAIN_ID}:
                tags.update(("omol25", "molecule"))
            elif unique_domains == {OMAT24_DOMAIN_ID}:
                tags.update(("omat24", "material"))

        if not tags.intersection(("omol25", "omat24")):
            atom_array = example.get("atom_array")
            if atom_array is not None and np.any(get_material_mask(atom_array)):
                tags.update(("omat24", "material"))
            elif atom_array is not None and bool(
                np.all(np.asarray(getattr(atom_array, "is_ligand", False)))
            ):
                tags.update(("omol25", "molecule"))

        task_ids = ground_truth.get("property_task_id_B")
        if task_ids is not None:
            unique_tasks = set(torch.as_tensor(task_ids).detach().cpu().tolist())
            if unique_tasks == {UNCONDITIONAL_PROPERTY_TASK_ID}:
                tags.add("unconditional")
            elif unique_tasks == {STRUCTURE_PREDICTION_PROPERTY_TASK_ID}:
                tags.add("structure_prediction")
            elif unique_tasks == {MLIP_PROPERTY_TASK_ID}:
                tags.add("mlip")
        else:
            sampled_condition_name = example.get("sampled_condition_name")
            while (
                isinstance(sampled_condition_name, (list, tuple))
                and len(sampled_condition_name) == 1
            ):
                sampled_condition_name = sampled_condition_name[0]
            task_example = {
                **example,
                "sampled_condition_name": sampled_condition_name,
            }
            # Validation datasets may omit ``is_inference`` after collation,
            # while retaining the canonical task-conditioning masks. This
            # routing path is validation-only, so allow those masks to recover
            # the task tag without changing training-time task inference.
            task_tag = infer_property_task_name(
                task_example,
                allow_mask_inference=True,
            )
            if task_tag is not None:
                tags.add(task_tag)

        return tags

    def _build_predicted_atom_array_stack(
        self, network_output: dict, example: dict
    ) -> Union[AtomArrayStack, List[AtomArray]]:
        """
        Build the predicted atom array stack from the network output and example metadata.

        Args:
            network_output (dict): The output from the network, containing the predicted atom positions and other relevant information.
            example (dict): The input example containing the necessary fields for building the predicted atom array stack.

        Returns:
            Union[AtomArrayStack, List[AtomArray]]: The predicted atom array stack or a list of atom arrays, depending on the configuration.
        """
        atom_array = example["atom_array"]
        source_atom_array = atom_array.copy()
        f = example["feats"]

        # ... Cleanup atom array:
        atom_array.bonds = None
        atom_array.res_name[~atom_array.is_motif_atom_with_fixed_seq] = (
            "UNK"  # Ensure non-motif residues set to UNK
        )
        atom_array = _reassign_unindexed_token_chains(atom_array)

        # ... Build output atom array stack
        atom_array_stack = _build_atom_array_stack(
            network_output["X_L"],
            atom_array,
            sequence_logits=network_output.get("sequence_logits_I"),
            sequence_indices=network_output.get("sequence_indices_I"),
            allow_sequence_outputs=self.allow_sequence_outputs,
            read_sequence_from_sequence_head=self.read_sequence_from_sequence_head,
            association_scheme=self.association_scheme,
        )  # NB: Will be either list (when sequences are saved) or stack

        arrays = atom_array_stack
        metadata_dict = {i: {"metrics": {}} for i in range(len(arrays))}

        for sample_index in range(len(arrays)):
            property_metadata = self._atomistic_property_prediction_metadata(
                network_output,
                sample_index=sample_index,
            )
            if property_metadata:
                metadata_dict[sample_index]["atomistic_properties"] = property_metadata

        # Add the seed to the metadata dictionary if provided
        if self.seed is not None:
            for i in range(len(arrays)):
                metadata_dict[i]["seed"] = self.seed

        atom_array_stack = []
        for i, atom_array in enumerate(arrays):
            atom_array = self._apply_atom_type_predictions(
                atom_array=atom_array,
                source_atom_array=source_atom_array,
                atom_type_indices=network_output.get("atom_type_indices_I", None),
                sample_idx=i,
                generation_mask=f.get("atom_type_generation_mask", None),
            )
            atom_array, material_metadata = self._apply_material_predictions(
                atom_array=atom_array,
                source_atom_array=source_atom_array,
                network_output=network_output,
                sample_idx=i,
                material_atom_mask=f.get("material_atom_mask", None),
                material_token_mask=f.get("material_token_mask", None),
                material_coord_generation_mask=f.get(
                    "material_coord_generation_mask", None
                ),
                material_lattice_generation_mask=f.get(
                    "material_lattice_generation_mask", None
                ),
                material_coordinate_output_source=self.material_coordinate_output_source,
                wrap_material_euclidean_coordinates=self.wrap_material_euclidean_coordinates,
            )
            # ... Create essential outputs for metadata dictionary
            if "example" in example["specification"]:
                metadata_dict[i] |= {"task": example["specification"]["example"]}
            if material_metadata:
                metadata_dict[i] |= {"material": material_metadata}

            # ... Add original specification to metadata
            if self.output_full_json:
                metadata_dict[i] |= {
                    "specification": example["specification"],
                }
                if (
                    hasattr(self, "inference_sampler_overrides")
                    and self.inference_sampler_overrides
                ):
                    metadata_dict[i] |= {
                        "inference_sampler": self.inference_sampler_overrides
                    }

            if np.any(atom_array.is_motif_atom_unindexed):
                # ... insert unindexed motif to output
                atom_array_processed, metadata = process_unindexed_outputs(
                    atom_array,
                    insert_guideposts=self.cleanup_guideposts,
                )
                global_logger.info(
                    f"Inserted unindexed motif atoms for example {i} with RMSD {metadata['insertion_rmsd']:.3f} A"
                )
                if self.cleanup_guideposts:
                    atom_array = atom_array_processed

                diffused_index_map = metadata.pop("diffused_index_map", None)
                metadata_dict[i]["metrics"] |= metadata
                if diffused_index_map is not None:
                    metadata_dict[i]["diffused_index_map"] = diffused_index_map
            else:
                metadata_dict[i]["diffused_index_map"] = {}

            # Also record where indexed motifs ended up
            residue_start_atoms = atom_array[get_residue_starts(atom_array)]
            indexed_residue_starts_non_ligand = residue_start_atoms[
                ~residue_start_atoms.is_motif_atom_unindexed
                & ~residue_start_atoms.is_ligand
            ]

            # If the src_component starts with an alphabetic character, it's from an external source
            # This is not relevant for when we train on ligands only.
            if (
                "src_component"
                in indexed_residue_starts_non_ligand.get_annotation_categories()
            ):
                external_src_mask = np.array(
                    [
                        (s[0].isalpha() if len(s) > 0 else False)
                        for s in indexed_residue_starts_non_ligand.src_component
                    ],
                    dtype=bool,
                )
                indexed_residue_starts_from_external_src = (
                    indexed_residue_starts_non_ligand[external_src_mask]
                )

                for token in indexed_residue_starts_from_external_src:
                    metadata_dict[i]["diffused_index_map"][token.src_component] = (
                        f"{token.chain_id}{token.res_id}"
                    )

            # ... Delete virtual atoms and assign atom names and elements
            if self.cleanup_virtual_atoms and np.any(atom_array.element == "VX"):
                atom_array = cleanup_virtual_atoms_and_assign_atom_name_elements(
                    atom_array, association_scheme=self.association_scheme
                )

                # ... When cleaning up virtual atoms, we can also calculate native_array_metrics.

                metadata_dict[i]["metrics"] |= get_all_backbone_metrics(
                    atom_array,
                    compute_non_clash_metrics_for_diffused_region_only=self.compute_non_clash_metrics_for_diffused_region_only,
                )

            if (
                "active_donor" in atom_array.get_annotation_categories()
                or "active_acceptor" in atom_array.get_annotation_categories()
            ):
                metadata_dict[i]["metrics"] |= get_hbond_metrics(
                    atom_array,
                    association_scheme=self.association_scheme,
                )

            if "partial_t" in f:
                # Try calculate a central atom (CA) RMSD to input:
                aa_in = example["atom_array"]
                xyz_ca_input = aa_in.coord[np.isin(aa_in.atom_name, "CA")]
                xyz_ca_output = atom_array.coord[np.isin(atom_array.atom_name, "CA")]

                # Align CA and calculate RMSD:
                if xyz_ca_input.shape == xyz_ca_output.shape:
                    try:
                        from foundry.utils.alignment import weighted_rigid_align

                        xyz_ca_output_aligned = (
                            weighted_rigid_align(
                                torch.from_numpy(xyz_ca_input)[None],
                                torch.from_numpy(xyz_ca_output)[None],
                            )
                            .squeeze(0)
                            .numpy()
                        )
                        metadata_dict[i]["metrics"] |= {
                            "ca_rmsd_to_input": float(
                                np.sqrt(
                                    np.mean(
                                        np.square(
                                            xyz_ca_input - xyz_ca_output_aligned
                                        ).sum(-1)
                                    )
                                )
                            )
                        }
                    except Exception as e:
                        global_logger.warning(
                            f"Failed to calculate CA RMSD for partial diffusion output: {e}"
                        )

            atom_array = uniquify_atom_names_for_cif(atom_array)
            atom_array_stack.append(atom_array)

        # Reorder metadata dictionaries to ensure 'metrics' and 'specification' are last
        metadata_dict = {k: _reorder_dict(d) for k, d in metadata_dict.items()}
        return atom_array_stack, metadata_dict

    @staticmethod
    def _atomistic_property_prediction_metadata(
        network_output: dict,
        *,
        sample_index: int,
    ) -> dict:
        """Serialize one generated sample's property predictions in physical units."""
        metadata = {}
        energy = network_output.get("energy_B")
        if energy is not None:
            energy = torch.as_tensor(energy).detach().cpu()
            if energy.ndim != 2 or energy.shape[1] != 1:
                raise ValueError(
                    "Inference energy_B must have shape [D, 1], got "
                    f"{tuple(energy.shape)}."
                )
            metadata["energy_eV"] = float(energy[sample_index, 0])

        forces = network_output.get("forces_L")
        if forces is not None:
            forces = torch.as_tensor(forces).detach().cpu()
            if forces.ndim != 3 or forces.shape[-1] != 3:
                raise ValueError(
                    "Inference forces_L must have shape [D, L, 3], got "
                    f"{tuple(forces.shape)}."
                )
            metadata["forces_eV_per_angstrom"] = forces[sample_index].tolist()
        return metadata

    @staticmethod
    def _apply_atom_type_predictions(
        *,
        atom_array: AtomArray,
        source_atom_array: AtomArray,
        atom_type_indices: torch.Tensor | None,
        sample_idx: int,
        generation_mask,
    ) -> AtomArray:
        """Write predicted atomic numbers back onto generated atomized molecule tokens."""
        if atom_type_indices is None or generation_mask is None:
            return atom_array

        generation_mask = np.asarray(
            generation_mask.cpu()
            if hasattr(generation_mask, "cpu")
            else generation_mask,
            dtype=bool,
        )
        if not np.any(generation_mask):
            return atom_array

        predicted_atomic_numbers = (
            atom_type_indices[sample_idx].detach().cpu().numpy().astype(int)
        )
        source_token_starts = get_token_starts(source_atom_array)
        token_starts = get_token_starts(atom_array)
        if len(token_starts) != len(generation_mask):
            global_logger.warning(
                "Skipping atom-type output assignment because token counts changed "
                f"({len(token_starts)} output vs {len(generation_mask)} source)."
            )
            return atom_array

        generated_atom_counts_by_residue = {}
        for token_i, token_start in enumerate(token_starts):
            if not generation_mask[token_i]:
                continue
            atomic_number = int(predicted_atomic_numbers[token_i])
            element = ATOMIC_NUMBER_TO_ELEMENT.get(atomic_number)
            if element is None:
                continue

            token_end = (
                token_starts[token_i + 1]
                if token_i + 1 < len(token_starts)
                else atom_array.array_length()
            )
            source_res_name = source_atom_array.res_name[source_token_starts[token_i]]
            residue_key = (
                str(atom_array.chain_id[token_start]),
                int(atom_array.res_id[token_start]),
            )
            generated_atom_counts_by_residue[residue_key] = (
                generated_atom_counts_by_residue.get(residue_key, 0) + 1
            )
            atom_name = f"{element}{generated_atom_counts_by_residue[residue_key]}"[:4]
            atom_array.element[token_start:token_end] = element
            if "atomic_number" in atom_array.get_annotation_categories():
                atom_array.atomic_number[token_start:token_end] = atomic_number
            atom_array.atom_name[token_start:token_end] = atom_name
            atom_array.res_name[token_start:token_end] = source_res_name

        return atom_array

    @staticmethod
    def _as_numpy(value, dtype=None):
        """Convert optional tensors/config values to numpy arrays."""
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        else:
            value = np.asarray(value)
        return value.astype(dtype) if dtype is not None else value

    @staticmethod
    def _apply_material_predictions(
        *,
        atom_array: AtomArray,
        source_atom_array: AtomArray,
        network_output: dict,
        sample_idx: int,
        material_atom_mask,
        material_token_mask,
        material_coord_generation_mask,
        material_lattice_generation_mask,
        material_coordinate_output_source: str = "fractional",
        wrap_material_euclidean_coordinates: bool = True,
    ) -> tuple[AtomArray, dict]:
        """Write predicted material fractional coordinates and lattice to outputs."""
        atom_mask = ZatomDesignTrainer._as_numpy(material_atom_mask, bool)
        token_mask = ZatomDesignTrainer._as_numpy(material_token_mask, bool)
        if atom_mask is None or token_mask is None or not np.any(atom_mask):
            return atom_array, {}
        if len(atom_mask) != atom_array.array_length():
            global_logger.warning(
                "Skipping material output assignment because atom counts changed "
                f"({atom_array.array_length()} output vs {len(atom_mask)} source)."
            )
            return atom_array, {}

        coord_generation_mask = ZatomDesignTrainer._as_numpy(
            material_coord_generation_mask, bool
        )
        lattice_generation_mask = ZatomDesignTrainer._as_numpy(
            material_lattice_generation_mask, bool
        )
        if coord_generation_mask is None:
            coord_generation_mask = atom_mask
        if lattice_generation_mask is None:
            lattice_generation_mask = token_mask

        source_lattice = get_material_lattice(source_atom_array)
        use_predicted_lattice = (
            "lattice_lengths_B" in network_output
            and "lattice_angles_B" in network_output
            and np.any(lattice_generation_mask)
        )
        if use_predicted_lattice:
            lengths_scaled_B = (
                network_output["lattice_lengths_B"][sample_idx].detach().cpu().numpy()
            )
            angles_radians_B = (
                network_output["lattice_angles_B"][sample_idx].detach().cpu().numpy()
            )
            lengths_scaled = lengths_scaled_B[0]
            angles_radians = angles_radians_B[0]
            n_material_tokens = int(np.sum(token_mask))
            lengths = unnormalize_lattice_lengths(lengths_scaled, n_material_tokens)
            angles = np.degrees(angles_radians)
        elif source_lattice is not None:
            lengths, angles, _, _ = source_lattice
        else:
            lengths = np.array([1.0, 1.0, 1.0], dtype=float)
            angles = np.array([90.0, 90.0, 90.0], dtype=float)

        lengths = np.clip(np.asarray(lengths, dtype=float), 0.1, None)
        angles = np.clip(np.asarray(angles, dtype=float), 1.0, 179.0)
        lattice_vectors = lattice_vectors_from_lengths_angles(lengths, angles)

        alignment_metadata = {}
        if material_coordinate_output_source == "euclidean" and np.any(
            coord_generation_mask
        ):
            if not np.all(coord_generation_mask[atom_mask]):
                raise ValueError(
                    "Euclidean cell-frame alignment requires all material atom "
                    "coordinates to be generated."
                )
            if (
                "material_frame_reference_X_L" not in network_output
                or "fractional_coord_L" not in network_output
            ):
                raise ValueError(
                    "Euclidean material output requires paired conditional Cartesian "
                    "and fractional predictions from the sampler. Final fixed-motif "
                    "realignment is not supported; use allow_realignment=false."
                )
            conditional_coordinates = ZatomDesignTrainer._as_numpy(
                network_output["material_frame_reference_X_L"][sample_idx]
            )
            predicted_fractional_coordinates = ZatomDesignTrainer._as_numpy(
                network_output["fractional_coord_L"][sample_idx]
            )
            rotation, translation, rmsd = fit_periodic_rigid_alignment(
                conditional_coordinates[atom_mask],
                predicted_fractional_coordinates[atom_mask],
                lattice_vectors,
            )
            # Rotate the guided endpoint rigidly, preserving its internal
            # Cartesian geometry, before deriving fractional coordinates.
            atom_array.coord[atom_mask] = (
                atom_array.coord[atom_mask] @ rotation + translation
            )
            alignment_metadata = {"cell_frame_alignment_rmsd_angstrom": rmsd}

        if (
            "material_frac_coords" in source_atom_array.get_annotation_categories()
            and source_atom_array.array_length() == atom_array.array_length()
        ):
            frac = np.asarray(
                source_atom_array.material_frac_coords,
                dtype=float,
            ).copy()
        else:
            frac = np.zeros((atom_array.array_length(), 3), dtype=float)
            frac[atom_mask] = cartesian_to_fractional(
                atom_array.coord[atom_mask],
                lattice_vectors,
            )

        if np.any(coord_generation_mask):
            if material_coordinate_output_source == "fractional":
                if "fractional_coord_L" not in network_output:
                    raise ValueError(
                        "material_coordinate_output_source='fractional' requires "
                        "fractional_coord_L in network_output."
                    )
                pred_frac = (
                    network_output["fractional_coord_L"][sample_idx]
                    .detach()
                    .cpu()
                    .numpy()
                )
                frac[coord_generation_mask] = pred_frac[coord_generation_mask] % 1.0
            else:
                frac[coord_generation_mask] = cartesian_to_fractional(
                    atom_array.coord[coord_generation_mask],
                    lattice_vectors,
                )

        atom_array = annotate_material_properties(
            atom_array,
            fractional_coordinates=frac,
            lattice_lengths=lengths,
            lattice_angles=angles,
            lattice_vectors=lattice_vectors,
            is_material=atom_mask,
        )
        if material_coordinate_output_source == "fractional":
            # Sync cartesian coordinates to fractional predictions
            # if using predicted fractional coordinates
            atom_array.coord[atom_mask] = fractional_to_cartesian(
                frac[atom_mask],
                lattice_vectors,
            )
        elif wrap_material_euclidean_coordinates:
            # Re-express predicted Euclidean coordinates in the predicted cell
            atom_array.coord[atom_mask] = wrap_cartesian_coordinates(
                atom_array.coord[atom_mask],
                lattice_vectors,
            )
        atom_array = add_fractional_cif_fields(atom_array)

        return atom_array, {
            "lattice_lengths": lengths.tolist(),
            "lattice_angles": angles.tolist(),
            "coordinate_output_source": material_coordinate_output_source,
            **alignment_metadata,
            "cif_extra_categories": material_cif_extra_categories(atom_array),
        }
