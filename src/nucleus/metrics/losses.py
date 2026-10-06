from collections.abc import Callable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from foundry.training.checkpoint import activation_checkpointing

from nucleus.constants import (
    MLIP_PROPERTY_TASK_ID,
    UNCONDITIONAL_PROPERTY_TASK_ID,
)
from nucleus.model.flow_matching import FLOW_MATCHING, flow_time_to_edm_noise
from nucleus.utils.atomistic_properties import AtomisticPropertyStatistics
from nucleus.utils.torch import expand_t_to_level, get_device_compatible_float64_dtype

try:
    from rfd3.metrics.losses import (
        DiffusionLoss as RFD3DiffusionLoss,
    )
    from rfd3.metrics.losses import (
        SequenceLoss as RFD3SequenceLoss,
    )
except ImportError:
    raise ImportError(
        "Could not import RFD3 losses. Please ensure that the `rc-foundry[all]` package is installed and up to date."
    )

LossOutputGradientDiagnostic = tuple[torch.Tensor, torch.Tensor]
SharedBackboneGradientDiagnostic = torch.Tensor

FLOW_GRADIENT_DIAGNOSTIC_NAMES = (
    "flow_mse_low_noise",
    "flow_mse_mid_noise",
    "flow_mse_high_noise",
)
PROPERTY_GRADIENT_DIAGNOSTIC_NAME = "atomistic_property"


def pop_loss_output_gradient_diagnostics(
    loss_container: nn.Module,
) -> dict[str, LossOutputGradientDiagnostic]:
    """Collect and clear differentiable loss/output pairs after a forward pass.

    Args:
        loss_container: A single loss module or Foundry ``Loss`` container.

    Returns:
        Mapping from diagnostic name to ``(weighted_loss, prediction)``.
    """
    loss_modules = getattr(loss_container, "to_compute", (loss_container,))
    diagnostics: dict[str, LossOutputGradientDiagnostic] = {}
    for loss_module in loss_modules:
        module_diagnostics = getattr(
            loss_module, "loss_output_gradient_diagnostics", {}
        )
        duplicate_names = diagnostics.keys() & module_diagnostics.keys()
        if duplicate_names:
            raise ValueError(
                "Duplicate loss gradient diagnostic names: "
                f"{sorted(duplicate_names)}"
            )
        diagnostics.update(module_diagnostics)
        loss_module.loss_output_gradient_diagnostics = {}
    return diagnostics


def compute_loss_output_gradient_l2_norms(
    diagnostics: dict[str, LossOutputGradientDiagnostic],
) -> dict[str, torch.Tensor]:
    """Compute each weighted loss gradient's L2 norm at its prediction tensor.

    The use of ``torch.autograd.grad`` does not populate or modify parameter
    ``.grad`` buffers. Retaining the graph allows the normal total-loss backward
    pass to run afterward.

    Args:
        diagnostics: Mapping from diagnostic name to
            ``(weighted_loss, prediction)``.

    Returns:
        Scalar L2 gradient norm for each differentiable diagnostic.
    """
    gradient_norms = {}
    for name, (weighted_loss, prediction) in diagnostics.items():
        if not weighted_loss.requires_grad or not prediction.requires_grad:
            continue
        gradient = torch.autograd.grad(
            weighted_loss,
            prediction,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if gradient is None:
            gradient_norms[name] = weighted_loss.detach().new_zeros(())
        else:
            gradient_norms[name] = torch.linalg.vector_norm(gradient.detach().float())
    return gradient_norms


def pop_shared_backbone_gradient_diagnostics(
    loss_container: nn.Module,
) -> dict[str, SharedBackboneGradientDiagnostic]:
    """Collect differentiable task losses used for shared-gradient diagnostics.

    Args:
        loss_container: A single loss module or Foundry ``Loss`` container.

    Returns:
        Mapping from task/noise-bin name to its weighted scalar loss.
    """
    loss_modules = getattr(loss_container, "to_compute", (loss_container,))
    diagnostics: dict[str, SharedBackboneGradientDiagnostic] = {}
    for loss_module in loss_modules:
        module_diagnostics = getattr(
            loss_module, "shared_backbone_gradient_diagnostics", {}
        )
        duplicate_names = diagnostics.keys() & module_diagnostics.keys()
        if duplicate_names:
            raise ValueError(
                "Duplicate shared-backbone gradient diagnostic names: "
                f"{sorted(duplicate_names)}"
            )
        diagnostics.update(module_diagnostics)
        loss_module.shared_backbone_gradient_diagnostics = {}
    return diagnostics


def compute_shared_backbone_gradient_cosines(
    diagnostics: dict[str, SharedBackboneGradientDiagnostic],
    parameters: tuple[nn.Parameter, ...] | list[nn.Parameter],
    backward: Callable[..., None] | None = None,
) -> dict[str, torch.Tensor]:
    """Compare flow-bin and property gradients on shared parameters.

    By default, the function uses ``torch.autograd.grad`` and leaves parameter
    ``.grad`` buffers untouched. When a Fabric-compatible ``backward`` callback
    is supplied, temporary ``.grad`` buffers are used and restored after each
    task-gradient calculation.

    Args:
        diagnostics: Weighted scalar losses collected from the flow and
            atomistic-property loss modules.
        parameters: Representative trainable parameters from the shared
            backbone.
        backward: Optional Fabric-compatible backward function. When supplied,
            gradients are captured from temporary ``.grad`` buffers and any
            pre-existing accumulated gradients are restored afterward.

    Returns:
        Gradient norms, flow/property cosine similarities, and binary conflict
        indicators. A cosine/conflict value is NaN when either task has zero
        gradient on the selected parameters.
    """
    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    property_loss = diagnostics.get(PROPERTY_GRADIENT_DIAGNOSTIC_NAME)
    if not parameters or property_loss is None or not property_loss.requires_grad:
        return {}

    def gradients(loss: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        if not loss.requires_grad:
            return tuple(None for _ in parameters)
        if backward is not None:
            saved_gradients = tuple(
                (
                    parameter.grad,
                    None if parameter.grad is None else parameter.grad.detach().clone(),
                )
                for parameter in parameters
            )
            try:
                for parameter in parameters:
                    parameter.grad = None
                backward(loss, retain_graph=True, inputs=parameters)
                return tuple(
                    None if parameter.grad is None else parameter.grad.detach().clone()
                    for parameter in parameters
                )
            finally:
                for parameter, (original_gradient, saved_value) in zip(
                    parameters, saved_gradients, strict=True
                ):
                    if original_gradient is None:
                        parameter.grad = None
                    else:
                        original_gradient.copy_(saved_value)
                        parameter.grad = original_gradient
        return torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )

    property_gradients = gradients(property_loss)
    zero = property_loss.detach().new_zeros((), dtype=torch.float32)

    def squared_norm(gradient_tensors) -> torch.Tensor:
        value = zero
        for gradient in gradient_tensors:
            if gradient is not None:
                gradient_float = gradient.detach().float()
                value = value + torch.sum(gradient_float * gradient_float)
        return value

    property_norm_squared = squared_norm(property_gradients)
    metrics = {
        f"l2_norm/{PROPERTY_GRADIENT_DIAGNOSTIC_NAME}": (
            torch.sqrt(property_norm_squared)
        )
    }

    for flow_name in FLOW_GRADIENT_DIAGNOSTIC_NAMES:
        flow_loss = diagnostics.get(flow_name)
        if flow_loss is None:
            continue
        flow_gradients = gradients(flow_loss)
        flow_norm_squared = squared_norm(flow_gradients)
        dot_product = zero
        for flow_gradient, property_gradient in zip(
            flow_gradients, property_gradients, strict=True
        ):
            if flow_gradient is None or property_gradient is None:
                continue
            dot_product = dot_product + torch.sum(
                flow_gradient.detach().float() * property_gradient.detach().float()
            )

        denominator = torch.sqrt(flow_norm_squared * property_norm_squared)
        valid = denominator > 0
        nan = denominator.new_full((), float("nan"))
        cosine = torch.where(valid, dot_product / denominator.clamp_min(1e-30), nan)
        cosine = torch.where(valid, cosine.clamp(-1.0, 1.0), nan)
        conflict = torch.where(valid, (dot_product < 0).to(torch.float32), nan)
        comparison_name = f"{flow_name}_vs_{PROPERTY_GRADIENT_DIAGNOSTIC_NAME}"
        metrics[f"l2_norm/{flow_name}"] = torch.sqrt(flow_norm_squared)
        metrics[f"cosine/{comparison_name}"] = cosine
        metrics[f"conflict/{comparison_name}"] = conflict

    return metrics


def collate_shared_backbone_gradient_metrics(
    metrics: dict[str, torch.Tensor],
    all_reduce: Callable[[torch.Tensor], torch.Tensor],
    *,
    world_size: int,
) -> dict[str, torch.Tensor]:
    """Aggregate rank-local shared-gradient metrics across all DDP ranks.

    Existing metric names become valid-rank means. Consequently, ``conflict``
    metrics become the fraction of valid ranks reporting a negative cosine.
    The function also emits across-rank standard deviations for cosines and
    norms, plus the fraction of ranks contributing a finite cosine. All scalar
    sufficient statistics are packed into one collective.

    Args:
        metrics: Rank-local scalar metrics from
            :func:`compute_shared_backbone_gradient_cosines`.
        all_reduce: Sum-reduction callback shared by every rank.
        world_size: Total number of distributed ranks.

    Returns:
        All-rank means under the original names, together with rank standard
        deviations and valid-rank fractions.
    """
    if world_size < 1:
        raise ValueError("world_size must be positive.")
    if not metrics:
        return {}

    names = tuple(sorted(metrics))
    values = torch.stack([metrics[name].detach().float() for name in names])
    finite = torch.isfinite(values)
    finite_values = torch.where(finite, values, torch.zeros_like(values))
    sufficient_statistics = torch.stack(
        (
            finite_values,
            finite_values.square(),
            finite.to(dtype=values.dtype),
        )
    )
    reduced = all_reduce(sufficient_statistics)
    sums, sums_of_squares, valid_counts = reduced.unbind(dim=0)
    safe_counts = valid_counts.clamp_min(1)
    means = sums / safe_counts
    variances = (sums_of_squares / safe_counts - means.square()).clamp_min(0)
    rank_stds = torch.sqrt(variances)
    nan = means.new_full(means.shape, float("nan"))
    means = torch.where(valid_counts > 0, means, nan)
    rank_stds = torch.where(valid_counts > 0, rank_stds, nan)

    collated = {}
    for index, name in enumerate(names):
        collated[name] = means[index]
        if name.startswith("cosine/"):
            comparison_name = name.removeprefix("cosine/")
            collated[f"cosine_rank_std/{comparison_name}"] = rank_stds[index]
            collated[f"cosine_valid_rank_fraction/{comparison_name}"] = (
                valid_counts[index] / world_size
            )
        elif name.startswith("l2_norm/"):
            task_name = name.removeprefix("l2_norm/")
            collated[f"l2_norm_rank_std/{task_name}"] = rank_stds[index]
    return collated


class SequenceLoss(RFD3SequenceLoss):
    """
    RFD3 sequence loss for the inpainting task. Only applies to tokens with t in [min_t, max_t).
    Also supervising atom types when present (e.g., for molecules and materials). Effectively,
    this loss supervises token type prediction with support for inpainting and dataloader batch_size > 1.

    Args:
        min_t: minimum t value to consider for the loss
        max_t: maximum t value to consider for the loss
        atom_type_weight: scalar weight for the atom type loss component
        sequence_weight: scalar weight for the sequence loss component
        label_smoothing: cross-entropy label smoothing. A head that becomes
            near-deterministic pays a very large penalty when confidently
            wrong. Smoothing makes the finite-confidence optimum explicit but
            does not itself bound cross-entropy.
        token_ce_cap_mode: How to limit sequence and atom-type CE. ``mean_hard``
            preserves the historical clamp after averaging, ``per_token``
            clamps outliers before averaging, and ``none`` disables the cap.
        sequence_loss_cap: Sequence CE cap used by ``token_ce_cap_mode``.
        atom_type_loss_cap: Atom-type CE cap used by ``token_ce_cap_mode``.
    """

    def __init__(
        self,
        min_t=0,
        max_t=torch.inf,
        atom_type_weight: float = 1.0,
        sequence_weight: float = 1.0,
        objective: str = "edm",
        sigma_data: float = 1.0,
        label_smoothing: float = 0.0,
        token_ce_cap_mode: str = "mean_hard",
        sequence_loss_cap: float = 5.0,
        atom_type_loss_cap: float = 10.0,
    ):
        super().__init__(weight=1.0, min_t=min_t, max_t=max_t)
        if label_smoothing:
            self.loss_fn = nn.CrossEntropyLoss(
                reduction="none", label_smoothing=label_smoothing
            )
        if token_ce_cap_mode not in {"mean_hard", "per_token", "none"}:
            raise ValueError(f"Unknown token_ce_cap_mode: {token_ce_cap_mode!r}")
        if sequence_loss_cap <= 0 or atom_type_loss_cap <= 0:
            raise ValueError("Token CE loss caps must be positive")
        self.sequence_weight = sequence_weight
        self.atom_type_weight = atom_type_weight
        self.objective = objective
        self.sigma_data = sigma_data
        self.loss_output_gradient_diagnostics = {}
        self.token_ce_cap_mode = token_ce_cap_mode
        self.sequence_loss_cap = sequence_loss_cap
        self.atom_type_loss_cap = atom_type_loss_cap

    def _loss_time(self, time: torch.Tensor) -> torch.Tensor:
        """Keep legacy noise-threshold semantics under flow time."""
        if self.objective == FLOW_MATCHING:
            return flow_time_to_edm_noise(time, self.sigma_data)
        return time

    @staticmethod
    def _empty_token_ce_metrics(
        logits: torch.Tensor, metric_prefix: str
    ) -> dict[str, torch.Tensor]:
        """Return the complete metric schema for a token branch with no targets."""
        zero = logits.new_zeros(())
        nan = logits.new_tensor(float("nan"))
        return {
            f"{metric_prefix}_loss": zero,
            f"effective_{metric_prefix}_loss": zero,
            f"{metric_prefix}_loss_above_cap": zero,
            f"{metric_prefix}_token_loss_above_cap_fraction": zero,
            f"{metric_prefix}_recovery": nan,
            f"lowest_t_{metric_prefix}_recovery": nan,
        }

    def _token_ce_loss(
        self,
        *,
        logits,
        indices,
        gt,
        target_mask,
        valid_t_expanded,
        t_I,
        metric_prefix,
        loss_cap,
        batch_token_lvl_expanded=None,
        batch_size: int | None = None,
        post_mean_cap: float | None = None,
    ):
        """Compute an example-balanced token CE branch.

        Token and valid-view counts can differ between systems in a sparse batch.
        Reduce those dimensions within each system first, then average over all
        systems (including a zero contribution from systems without targets).
        This makes a packed batch equivalent to averaging the same examples
        evaluated independently.
        """
        if batch_token_lvl_expanded is None:
            batch_token_lvl_expanded = torch.zeros_like(
                valid_t_expanded, dtype=torch.long
            )
        if batch_token_lvl_expanded.shape != valid_t_expanded.shape:
            raise ValueError(
                "batch_token_lvl_expanded and valid_t_expanded must have the same "
                f"shape, got {batch_token_lvl_expanded.shape} and "
                f"{valid_t_expanded.shape}."
            )
        if batch_size is None:
            batch_size = 1
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        branch_logits = logits[valid_t_expanded]
        if branch_logits.numel() == 0:
            zero = logits.sum() * 0.0
            return zero, self._empty_token_ce_metrics(logits, metric_prefix)

        branch_gt = gt[None].expand(logits.shape[0], -1)[valid_t_expanded]
        branch_mask = target_mask[None].expand(logits.shape[0], -1)[valid_t_expanded]
        branch_mask = branch_mask.to(dtype=torch.bool)
        branch_batch = batch_token_lvl_expanded[valid_t_expanded].to(dtype=torch.long)

        token_loss = self.loss_fn(branch_logits, branch_gt)
        if not torch.any(branch_mask):
            zero = token_loss.sum() * 0.0 + logits.sum() * 0.0
            return zero, self._empty_token_ce_metrics(logits, metric_prefix)

        token_loss = token_loss[branch_mask]
        token_batch = branch_batch[branch_mask]
        pred = indices[valid_t_expanded][branch_mask]
        target = branch_gt[branch_mask]
        token_time = t_I[valid_t_expanded][branch_mask]
        recovery = (pred == target).to(dtype=token_loss.dtype)

        token_count_by_system = torch.zeros(
            batch_size, device=logits.device, dtype=token_loss.dtype
        ).scatter_add(
            0,
            token_batch,
            torch.ones_like(token_loss),
        )
        active_system = token_count_by_system > 0
        token_loss_sum_by_system = torch.zeros_like(token_count_by_system).scatter_add(
            0, token_batch, token_loss
        )
        raw_loss_by_system = token_loss_sum_by_system / token_count_by_system.clamp_min(
            1
        )
        raw_loss = raw_loss_by_system.mean()

        if self.token_ce_cap_mode == "mean_hard":
            effective_loss_by_system = torch.clamp(raw_loss_by_system, max=loss_cap)
        elif self.token_ce_cap_mode == "per_token":
            capped_loss_sum_by_system = torch.zeros_like(
                token_count_by_system
            ).scatter_add(0, token_batch, torch.clamp(token_loss, max=loss_cap))
            effective_loss_by_system = (
                capped_loss_sum_by_system / token_count_by_system.clamp_min(1)
            )
        else:
            effective_loss_by_system = raw_loss_by_system
        reported_effective_loss = effective_loss_by_system.mean()
        if post_mean_cap is not None:
            effective_loss_by_system = torch.clamp(
                effective_loss_by_system, max=post_mean_cap
            )
        effective_loss = effective_loss_by_system.mean()

        recovery_sum_by_system = torch.zeros_like(token_count_by_system).scatter_add(
            0, token_batch, recovery
        )
        recovery_by_system = recovery_sum_by_system / token_count_by_system.clamp_min(1)
        mean_recovery = recovery_by_system[active_system].mean()

        if batch_size == 1:
            # Retain the exact historical tie behavior for singleton batches.
            _, order = torch.sort(token_time)
            lowest_t_rec = recovery[order][0].detach()
        else:
            # Select the first token at the minimum valid time independently for
            # every active system, then mean those per-system values.
            min_time_by_system = torch.full(
                (batch_size,),
                torch.inf,
                device=token_time.device,
                dtype=token_time.dtype,
            ).scatter_reduce(
                0,
                token_batch,
                token_time,
                reduce="amin",
                include_self=True,
            )
            # MPS scatter_reduce does not support int64 source/output tensors.
            # Positions remain exact in int32 for any practically realizable batch.
            position_dtype = (
                torch.int32 if token_loss.device.type == "mps" else torch.long
            )
            token_position = torch.arange(
                token_loss.numel(),
                device=token_loss.device,
                dtype=position_dtype,
            )
            is_min_time = token_time == min_time_by_system[token_batch]
            sentinel = token_loss.numel()
            first_min_position_by_system = torch.full(
                (batch_size,),
                sentinel,
                device=token_loss.device,
                dtype=position_dtype,
            ).scatter_reduce(
                0,
                token_batch,
                torch.where(
                    is_min_time,
                    token_position,
                    torch.full_like(token_position, sentinel),
                ),
                reduce="amin",
                include_self=True,
            )
            lowest_t_recovery_by_system = recovery[
                first_min_position_by_system[active_system].to(dtype=torch.long)
            ]
            lowest_t_rec = lowest_t_recovery_by_system.mean().detach()

        above_cap_by_system = ((raw_loss_by_system >= loss_cap) & active_system).to(
            dtype=token_loss.dtype
        )
        token_above_cap_sum_by_system = torch.zeros_like(
            token_count_by_system
        ).scatter_add(0, token_batch, (token_loss >= loss_cap).to(token_loss.dtype))
        token_above_cap_fraction = (
            token_above_cap_sum_by_system / token_count_by_system.clamp_min(1)
        ).mean()

        return effective_loss, {
            f"{metric_prefix}_loss": raw_loss.detach(),
            f"effective_{metric_prefix}_loss": reported_effective_loss.detach(),
            f"{metric_prefix}_loss_above_cap": (above_cap_by_system.mean().detach()),
            f"{metric_prefix}_token_loss_above_cap_fraction": (
                token_above_cap_fraction.detach()
            ),
            f"{metric_prefix}_recovery": mean_recovery.detach(),
            f"lowest_t_{metric_prefix}_recovery": lowest_t_rec,
        }

    def forward(
        self, network_input, network_output, loss_input
    ) -> tuple[torch.Tensor, dict]:
        """
        Calculate the sequence loss for token types with support for inpainting and dataloader batch_size > 1.

        Args:
            network_input: dict containing the inputs to the network, including "t" (D, B) and "f" (dict of features)
            network_output: dict containing the outputs of the network, including "sequence_logits_I" (D, L, 32) and "sequence_indices_I" (D, L)
            loss_input: dict containing the inputs for the loss calculation, including "seq_token_lvl" (L,) and "sequence_valid_mask" (L,)

        Returns:
            loss: scalar tensor representing the sequence loss
            outs: dict containing additional metrics, including "token_lvl_sequence_loss", "seq_recovery", "lowest_t_seq_recovery", "valid_t_fraction", and "n_valid_t"
        """
        self.loss_output_gradient_diagnostics = {}
        t = self._loss_time(network_input["t"])  # (D, B)
        f = network_input["f"]
        valid_t = (self.min_t <= t) & (
            t < self.max_t
        ).contiguous()  # bool mask over batch
        n_valid_t = valid_t.sum()

        sequence_logits_I = network_output["sequence_logits_I"]  # (D, I, 32)
        sequence_indices_I = network_output["sequence_indices_I"]  # (D, I)
        atom_type_logits_I = network_output.get("atom_type_logits_I")  # (D, I, 128)
        atom_type_indices_I = network_output.get("atom_type_indices_I")  # (D, I)

        if n_valid_t == 0:
            zero = sequence_logits_I.sum() * 0.0
            sequence_outs = self._empty_token_ce_metrics(sequence_logits_I, "sequence")
            atom_type_outs = {}
            if atom_type_logits_I is not None:
                zero = zero + atom_type_logits_I.sum() * 0.0
                atom_type_outs = self._empty_token_ce_metrics(
                    atom_type_logits_I, "atom_type"
                )
            return zero, (
                {
                    "token_lvl_sequence_loss": sequence_outs["sequence_loss"],
                    "seq_recovery": sequence_outs["sequence_recovery"],
                    "lowest_t_seq_recovery": sequence_outs[
                        "lowest_t_sequence_recovery"
                    ],
                    "valid_t_fraction": sequence_logits_I.new_tensor(0.0),
                    "n_valid_t": sequence_logits_I.new_tensor(0.0),
                }
                | sequence_outs
                | atom_type_outs
                | {
                    key: value.detach()
                    for key, value in network_output.get(
                        "token_head_diagnostics", {}
                    ).items()
                }
            )

        D = f["diffusion_batch_size"]
        B = int(f.get("batch_size", t.shape[1]))
        batch_token_lvl_expanded = (
            f["batch_token_lvl"].unsqueeze(0).expand(D, -1)
        ).contiguous()
        valid_t_expanded = valid_t.gather(
            1, batch_token_lvl_expanded
        )  # (D, I) bool mask over tokens

        sequence_loss, sequence_outs = self._token_ce_loss(
            logits=sequence_logits_I,
            indices=sequence_indices_I,
            gt=loss_input["seq_token_lvl"],
            target_mask=loss_input["sequence_valid_mask"],
            valid_t_expanded=valid_t_expanded,
            t_I=self._loss_time(network_input["t_I"]),
            metric_prefix="sequence",
            loss_cap=self.sequence_loss_cap,
            batch_token_lvl_expanded=batch_token_lvl_expanded,
            batch_size=B,
        )
        atom_type_loss = sequence_logits_I.sum() * 0.0
        atom_type_outs = {}
        if atom_type_logits_I is not None and atom_type_indices_I is not None:
            atom_type_loss, atom_type_outs = self._token_ce_loss(
                logits=atom_type_logits_I,
                indices=atom_type_indices_I,
                gt=loss_input["atom_type_gt_I"],
                target_mask=loss_input["atom_type_valid_mask"],
                valid_t_expanded=valid_t_expanded,
                t_I=self._loss_time(network_input["t_I"]),
                metric_prefix="atom_type",
                loss_cap=self.atom_type_loss_cap,
                batch_token_lvl_expanded=batch_token_lvl_expanded,
                batch_size=B,
                # Keep the legacy atom-type clamp, but apply it before the
                # cross-system mean so packing cannot change the objective.
                post_mean_cap=10.0,
            )
            if torch.any(loss_input["atom_type_valid_mask"]):
                self.loss_output_gradient_diagnostics["atom_types"] = (
                    self.atom_type_weight * atom_type_loss,
                    atom_type_logits_I,
                )

        outs = (
            {
                "token_lvl_sequence_loss": sequence_outs["sequence_loss"],
                "seq_recovery": sequence_outs["sequence_recovery"],
                "lowest_t_seq_recovery": sequence_outs["lowest_t_sequence_recovery"],
                "valid_t_fraction": valid_t.float().mean().detach(),
                "n_valid_t": n_valid_t.float(),
            }
            | sequence_outs
            | atom_type_outs
            | {
                key: value.detach()
                for key, value in network_output.get(
                    "token_head_diagnostics", {}
                ).items()
            }
        )
        loss = (
            self.sequence_weight * sequence_loss
            + self.atom_type_weight * atom_type_loss
        )
        return loss, outs


class MaterialGeometryLoss(nn.Module):
    """Loss for material fractional coordinates and crystal lattice heads.

    Args:
        reduction: ``system_mean`` averages valid diffusion views within each
            system, then averages all systems, including inactive zeros.
            ``active_mean`` restores the legacy mean over valid material
            system/view pairs, so systems with more valid views have more weight.
    """

    def __init__(
        self,
        *,
        min_t=0,
        max_t=torch.inf,
        weight: float = 1.0,
        fractional_coord_weight: float = 1.0,
        lattice_lengths_weight: float = 1.0,
        lattice_angles_weight: float = 1.0,
        objective: str = "edm",
        sigma_data: float = 1.0,
        reduction: str = "system_mean",
    ):
        super().__init__()
        if reduction not in {"system_mean", "active_mean"}:
            raise ValueError(
                f"Unknown material geometry loss reduction {reduction!r}; "
                "expected 'system_mean' or 'active_mean'."
            )
        self.min_t = min_t
        self.max_t = max_t
        self.weight = weight
        self.fractional_coord_weight = fractional_coord_weight
        self.lattice_lengths_weight = lattice_lengths_weight
        self.lattice_angles_weight = lattice_angles_weight
        self.objective = objective
        self.sigma_data = sigma_data
        self.reduction = reduction
        self.loss_output_gradient_diagnostics = {}

    def _masked_mean_loss(
        self,
        pred,
        gt,
        mask,
        valid_t,
        f,
        *,
        periodic: bool = False,
        batch_lvl=None,
        per_view_loss_cap: float | None = None,
    ):
        """
        Mean MSE under target and diffusion-time masks.

        Rows are first averaged within each valid system/diffusion-view pair.
        With ``system_mean``, valid views are averaged within each system, then
        all systems are averaged, including zeros from inactive systems. This
        default is invariant to how systems are partitioned into batches.
        With ``active_mean``, only valid system/view pairs enter the final mean.
        Both modes return differentiable zero when no targets are active.

        Args:
            pred: predicted values, shape [D, ...]
            gt: ground truth values, shape [...]
            mask: boolean mask indicating valid entries, shape [...]
            valid_t: boolean mask indicating valid diffusion times, shape [D, ...]
            f: dict of features, used to reference batch-level features
            periodic: whether to apply periodic boundary conditions
            batch_lvl: optional batch-level indices for aggregating loss
            per_view_loss_cap: optional cap applied after the valid-row mean for
                each system and diffusion view

        Returns:
            mean_loss: scalar loss using the configured reduction
        """
        D = f["diffusion_batch_size"]
        B = f["batch_size"]

        valid = mask[None].expand(D, -1) & valid_t
        gt = gt if gt.shape == pred.shape else gt[None]
        delta = pred - gt
        if periodic:
            delta = delta - torch.round(delta)
        row_loss = torch.mean(delta**2, dim=-1)

        if batch_lvl is None:
            if row_loss.shape[1] != B:
                raise ValueError(
                    "System-level material loss rows must match batch size; "
                    f"got {row_loss.shape[1]} rows for batch size {B}."
                )
            batch_lvl = torch.arange(B, device=pred.device)
        else:
            batch_lvl = batch_lvl.to(device=pred.device, dtype=torch.long)
        batch_lvl_expanded = batch_lvl.unsqueeze(0).expand(D, -1)

        valid_float = valid.to(dtype=pred.dtype)
        per_view_sum = torch.zeros(
            (D, B), device=pred.device, dtype=pred.dtype
        ).scatter_add(1, batch_lvl_expanded, row_loss * valid_float)
        per_view_count = torch.zeros_like(per_view_sum).scatter_add(
            1, batch_lvl_expanded, valid_float
        )
        valid_view = per_view_count > 0
        per_view_loss = per_view_sum / per_view_count.clamp_min(1)
        if per_view_loss_cap is not None:
            per_view_loss = torch.clamp(per_view_loss, max=per_view_loss_cap)

        valid_view_float = valid_view.to(dtype=pred.dtype)
        if self.reduction == "active_mean":
            return (
                per_view_loss * valid_view_float
            ).sum() / valid_view_float.sum().clamp_min(1)

        per_system_loss = (per_view_loss * valid_view_float).sum(dim=0)
        per_system_loss = per_system_loss / valid_view_float.sum(dim=0).clamp_min(1)
        return per_system_loss.mean()

    def _valid_t_mask(self, network_input, *, level: str, n_items: int):
        """
        Build a [D, n_items] time mask for atom- or token-level outputs.

        Args:
            network_input: dict containing the inputs to the network, including "t" (D, B), "t_L" (D, L), "t_I" (D, I), and "f" (dict of features)
            level: either "atom" or "token", indicating which time input to use
            n_items: the expected number of items at the given level (e.g., number of atoms for "atom" level)

        Returns:
            valid_t_mask: a boolean tensor of shape [D, n_items] indicating which items are valid based on the time mask
        """
        if level == "system":
            t_level = network_input["t"]
        else:
            key = "t_L" if level == "atom" else "t_I"
            if key in network_input:
                t_level = network_input[key]
            else:
                batch_key = "batch_atom_lvl" if level == "atom" else "batch_token_lvl"
                batch = network_input["f"][batch_key].unsqueeze(0)
                batch = batch.expand(network_input["t"].shape[0], -1)
                t_level = network_input["t"].gather(1, batch)

        if t_level.shape[1] != n_items:
            raise ValueError(
                f"Expected {level} to contain {n_items} items, "
                f"got {t_level.shape[1]}."
            )
        if self.objective == FLOW_MATCHING:
            t_level = flow_time_to_edm_noise(t_level, self.sigma_data)
        return (self.min_t <= t_level) & (t_level < self.max_t)

    def forward(
        self, network_input, network_output, loss_input
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute optional material geometry losses.

        Args:
            network_input: dict containing the inputs to the network, including "t" (D, B), "t_L" (D, L), "t_I" (D, I), and "f" (dict of features)
            network_output: dict containing the outputs of the network, including "X_L" (D, L, 3), "fractional_coord_L" (D, L, 3), "lattice_lengths_B" (D, 3), and "lattice_angles_B" (D, 3)
            loss_input: dict containing the inputs for the loss calculation, including "fractional_coord_gt_L" (L, 3), "fractional_coord_valid_mask_L" (L,), "lattice_lengths_scaled_gt_B" (B, 3), "lattice_angles_radians_gt_B" (B, 3), "lattice_valid_mask_B" (B,), and "f" (dict of features)

        Returns:
            loss: scalar tensor representing the weighted sum of the material geometry losses
            outs: dict containing additional metrics, including "material_fractional_coord_loss", "material_lattice_lengths_loss", and "material_lattice_angles_loss"
        """
        self.loss_output_gradient_diagnostics = {}
        f = network_input["f"]
        zero = network_output["X_L"].sum() * 0.0
        loss = zero
        outs = {}

        if (
            self.fractional_coord_weight > 0
            and "fractional_coord_L" in network_output
            and "fractional_coord_gt_L" in loss_input
        ):
            pred = network_output["fractional_coord_L"]
            valid_t = self._valid_t_mask(
                network_input,
                level="atom",
                n_items=pred.shape[1],
            )
            frac_loss = self._masked_mean_loss(
                pred,
                loss_input["fractional_coord_gt_L"],
                loss_input["fractional_coord_valid_mask_L"],
                valid_t,
                f,
                periodic=True,
                batch_lvl=f["batch_atom_lvl"],
                per_view_loss_cap=2.0,
            )
            loss = loss + self.fractional_coord_weight * frac_loss
            outs["material_fractional_coord_loss"] = frac_loss.detach()
            if torch.any(loss_input["fractional_coord_valid_mask_L"]):
                self.loss_output_gradient_diagnostics["fractional_coordinates"] = (
                    self.weight * self.fractional_coord_weight * frac_loss,
                    pred,
                )
        elif "fractional_coord_L" in network_output:
            pred = network_output["fractional_coord_L"]
            frac_loss = pred.sum() * 0.0
            loss = loss + frac_loss
            outs["material_fractional_coord_loss"] = frac_loss.detach()
        else:
            raise ValueError(
                "Expected fractional_coord_L head in network output for material geometry loss."
            )

        if (
            self.lattice_lengths_weight > 0
            and "lattice_lengths_B" in network_output
            and "lattice_lengths_scaled_gt_B" in loss_input
        ):
            pred = network_output["lattice_lengths_B"]
            valid_t = self._valid_t_mask(
                network_input,
                level="system",
                n_items=pred.shape[1],
            )
            lengths_loss = self._masked_mean_loss(
                pred,
                loss_input["lattice_lengths_scaled_gt_B"],
                loss_input["lattice_valid_mask_B"],
                valid_t,
                f,
            )
            loss = loss + self.lattice_lengths_weight * lengths_loss
            outs["material_lattice_lengths_loss"] = lengths_loss.detach()
            if torch.any(loss_input["lattice_valid_mask_B"]):
                self.loss_output_gradient_diagnostics["lattice_lengths"] = (
                    self.weight * self.lattice_lengths_weight * lengths_loss,
                    pred,
                )
        elif "lattice_lengths_B" in network_output:
            pred = network_output["lattice_lengths_B"]
            lengths_loss = pred.sum() * 0.0
            loss = loss + lengths_loss
            outs["material_lattice_lengths_loss"] = lengths_loss.detach()
        else:
            raise ValueError(
                "Expected lattice_lengths_B head in network output for material geometry loss."
            )

        if (
            self.lattice_angles_weight > 0
            and "lattice_angles_B" in network_output
            and "lattice_angles_radians_gt_B" in loss_input
        ):
            pred = network_output["lattice_angles_B"]
            valid_t = self._valid_t_mask(
                network_input,
                level="system",
                n_items=pred.shape[1],
            )
            angles_loss = self._masked_mean_loss(
                pred,
                loss_input["lattice_angles_radians_gt_B"],
                loss_input["lattice_valid_mask_B"],
                valid_t,
                f,
            )
            loss = loss + self.lattice_angles_weight * angles_loss
            outs["material_lattice_angles_loss"] = angles_loss.detach()
            if torch.any(loss_input["lattice_valid_mask_B"]):
                self.loss_output_gradient_diagnostics["lattice_angles"] = (
                    self.weight * self.lattice_angles_weight * angles_loss,
                    pred,
                )
        elif "lattice_angles_B" in network_output:
            pred = network_output["lattice_angles_B"]
            angles_loss = pred.sum() * 0.0
            loss = loss + angles_loss
            outs["material_lattice_angles_loss"] = angles_loss.detach()
        else:
            raise ValueError(
                "Expected lattice_angles_B head in network output for material geometry loss."
            )

        return self.weight * loss, outs


class AtomisticPropertyLoss(nn.Module):
    """Near-clean joint energy and force loss for OMol25 and OMat24.

    Energy labels are converted to standardized residual energy per atom using
    fixed, domain-specific elemental references. Force vectors are divided by the
    domain force RMS. Both branches use Smooth-L1 with per-system balancing, and
    examples carrying ground-truth force-statistic conditioning are excluded to
    prevent target leakage. Dedicated MLIP examples are evaluated at the clean
    endpoint regardless of the globally sampled flow time; unconditional examples
    retain near-clean time gating, while structure-prediction examples do not
    contribute property loss.

    Args:
        statistics_path: Versioned training-split statistics JSON artifact.
        statistics: Embedded constructor artifact when statistics_path is null.
        domain_names: Domain ordering used with dense in-memory statistics. This
            path is primarily useful for tests; production jobs should provide an
            artifact.
        energy_mean: Per-domain residual-energy-per-atom means in eV.
        energy_scale: Per-domain residual-energy-per-atom standard deviations in
            eV.
        force_scale: Per-domain force RMS values in eV/angstrom.
        elemental_references: Dense ``[n_domain, n_element]`` elemental energy
            references in eV.
        energy_weight: Energy loss coefficient after warmup.
        force_weight: Force loss coefficient after warmup.
        latent_equivariance_weight: TransIP-style pooled-embedding consistency
            coefficient after warmup. The loss is restricted to clean OMol25
            and OMat24 MLIP examples whose diffusion views are recorded rotation
            pairs.
        energy_max_sigma: Exclusive equivalent-noise cutoff for energy loss.
        force_max_sigma: Exclusive equivalent-noise cutoff for force loss.
        sigma_data: Coordinate data scale used to convert flow time to equivalent
            EDM noise.
        objective: Coordinate objective. Flow matching uses
            ``sigma_data * (1 - t) / t``; other objectives treat ``t`` as noise.
        warmup_steps: Number of optimizer steps used for linear loss warmup. Set
            to zero to disable warmup.
        huber_delta: Smooth-L1 transition point in standardized units.
        weight: Optional outer multiplier applied to the combined property loss.
    """

    _METRIC_KEYS = (
        "atomistic_property_loss",
        "atomistic_energy_loss",
        "atomistic_force_loss",
        "atomistic_latent_equivariance_loss",
        "atomistic_property_warmup_scale",
        "atomistic_energy_valid_fraction",
        "atomistic_force_valid_fraction",
        "atomistic_energy_valid_views",
        "atomistic_force_valid_views",
        "atomistic_latent_equivariance_valid_fraction",
        "atomistic_latent_equivariance_valid_pairs",
    )

    def __init__(
        self,
        *,
        statistics_path: str | None = None,
        statistics: Mapping | None = None,
        domain_names: tuple[str, ...] | list[str] = ("omol25", "omat24"),
        energy_mean: tuple[float, ...] | list[float] | None = None,
        energy_scale: tuple[float, ...] | list[float] | None = None,
        force_scale: tuple[float, ...] | list[float] | None = None,
        elemental_references: tuple[tuple[float, ...], ...]
        | list[list[float]]
        | None = None,
        energy_weight: float = 0.1,
        force_weight: float = 0.3,
        latent_equivariance_weight: float = 0.0,
        energy_max_sigma: float = 1.0,
        force_max_sigma: float = 0.25,
        sigma_data: float = 16.0,
        objective: str = FLOW_MATCHING,
        warmup_steps: int = 10_000,
        huber_delta: float = 1.0,
        weight: float = 1.0,
    ):
        super().__init__()
        if statistics_path is not None:
            statistics = AtomisticPropertyStatistics.load(statistics_path)
        elif statistics is not None:
            statistics = AtomisticPropertyStatistics.from_dict(statistics)
        else:
            dense_values = (
                energy_mean,
                energy_scale,
                force_scale,
                elemental_references,
            )
            if any(value is None for value in dense_values):
                raise ValueError(
                    "AtomisticPropertyLoss requires statistics_path, statistics, or all dense "
                    "statistics values."
                )
            statistics = AtomisticPropertyStatistics.from_values(
                domain_names=domain_names,
                energy_mean=energy_mean,  # type: ignore[arg-type]
                energy_scale=energy_scale,  # type: ignore[arg-type]
                force_scale=force_scale,  # type: ignore[arg-type]
                elemental_references=elemental_references,  # type: ignore[arg-type]
            )

        self._property_statistics = statistics
        if statistics.domain_names != ("omol25", "omat24"):
            raise ValueError(
                "Zatom-2 property loss requires domain IDs 0=omol25 and "
                f"1=omat24; got {statistics.domain_names}."
            )

        if (
            energy_weight < 0
            or force_weight < 0
            or latent_equivariance_weight < 0
            or weight < 0
        ):
            raise ValueError("Loss weights must be non-negative.")
        if energy_max_sigma <= 0 or force_max_sigma <= 0:
            raise ValueError("Equivalent-noise cutoffs must be positive.")
        if sigma_data <= 0:
            raise ValueError("sigma_data must be positive.")
        if not isinstance(warmup_steps, int) or isinstance(warmup_steps, bool):
            raise TypeError("warmup_steps must be an integer.")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive.")

        # Energy referencing subtracts two potentially very large total energies,
        # especially for OMol25. Keep the reference path in float64 where
        # supported (float32 on MPS) and cast only the already-standardized target
        # to the prediction dtype. Force scaling does not involve cancellation and
        # remains float32.
        tensors = statistics.tensors(
            max_atomic_number=127,
            dtype=get_device_compatible_float64_dtype(),
        )
        self.register_buffer(
            "elemental_references", tensors["elemental_references"], persistent=True
        )
        self.register_buffer(
            "elemental_reference_valid",
            tensors["elemental_reference_valid"],
            persistent=True,
        )
        self.register_buffer("energy_mean", tensors["energy_mean"], persistent=True)
        self.register_buffer("energy_scale", tensors["energy_scale"], persistent=True)
        self.register_buffer(
            "force_scale", tensors["force_scale"].float(), persistent=True
        )

        self.domain_names = statistics.domain_names
        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.latent_equivariance_weight = latent_equivariance_weight
        self.energy_max_sigma = energy_max_sigma
        self.force_max_sigma = force_max_sigma
        self.sigma_data = sigma_data
        self.objective = objective
        self.warmup_steps = warmup_steps
        self.huber_delta = huber_delta
        self.weight = weight
        self.shared_backbone_gradient_diagnostics = {}

    @staticmethod
    def _as_system_vector(value: torch.Tensor, *, name: str, B: int) -> torch.Tensor:
        """Normalize a scalar system target to shape ``[B]``."""
        if value.ndim == 2 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        if value.shape != (B,):
            raise ValueError(f"Expected {name} to have shape [{B}], got {value.shape}.")
        return value

    @staticmethod
    def _system_has_mean_force_conditioning(
        f: dict, *, B: int, device: torch.device
    ) -> torch.Tensor:
        """Reduce token/atom conditioning indicators to a system mask."""
        system_mask = torch.zeros(B, dtype=torch.bool, device=device)
        for condition_key in ("has_mean_force_norm", "has_force_rms_p95"):
            condition = f.get(condition_key)
            if condition is None:
                continue
            condition = condition.to(device=device, dtype=torch.bool)
            if condition.ndim > 1:
                condition = condition.reshape(condition.shape[0], -1).any(dim=-1)
            if condition.shape == (B,):
                system_mask |= condition
                continue

            for batch_key in ("batch_token_lvl", "batch_atom_lvl"):
                batch = f.get(batch_key)
                if batch is None or batch.numel() != condition.numel():
                    continue
                counts = torch.zeros(B, dtype=torch.int64, device=device).scatter_add(
                    0,
                    batch.to(device=device, dtype=torch.long),
                    condition.to(dtype=torch.int64),
                )
                system_mask |= counts > 0
                break
            else:
                raise ValueError(
                    f"Could not align {condition_key} with systems, tokens, or atoms."
                )
        return system_mask

    def _equivalent_noise(self, time: torch.Tensor) -> torch.Tensor:
        """Convert sampled time to the thresholding convention used by losses."""
        if self.objective == FLOW_MATCHING:
            return flow_time_to_edm_noise(time, self.sigma_data)
        return time

    def _base_system_validity(
        self,
        *,
        loss_input: dict,
        domain_id: torch.Tensor,
        B: int,
    ) -> torch.Tensor:
        """Apply crop, domain, and electronic-metadata validity gates."""
        device = domain_id.device
        valid = torch.ones(B, dtype=torch.bool, device=device)
        for key in ("property_valid_mask_B", "domain_valid_mask_B"):
            if key in loss_input:
                value = self._as_system_vector(loss_input[key], name=key, B=B)
                valid &= value.to(device=device, dtype=torch.bool)

        source_count = self._as_system_vector(
            loss_input["source_atom_count_B"], name="source_atom_count_B", B=B
        ).to(device=device)
        retained_count = self._as_system_vector(
            loss_input["retained_atom_count_B"], name="retained_atom_count_B", B=B
        ).to(device=device)
        valid &= (
            torch.isfinite(source_count)
            & torch.isfinite(retained_count)
            & (source_count > 0)
            & (source_count == retained_count)
        )
        valid &= (domain_id >= 0) & (domain_id < len(self.domain_names))

        # OMol25 is domain 0 by the shared data contract and requires both
        # electronic-state metadata fields for unambiguous property prediction.
        is_omol = domain_id == 0
        charge_valid = self._as_system_vector(
            loss_input["charge_valid_mask_B"], name="charge_valid_mask_B", B=B
        ).to(device=device, dtype=torch.bool)
        spin_valid = self._as_system_vector(
            loss_input["spin_valid_mask_B"], name="spin_valid_mask_B", B=B
        ).to(device=device, dtype=torch.bool)
        charge = self._as_system_vector(
            loss_input["total_charge_B"], name="total_charge_B", B=B
        ).to(device=device)
        spin = self._as_system_vector(
            loss_input["spin_multiplicity_B"], name="spin_multiplicity_B", B=B
        ).to(device=device)
        charge_is_integer = torch.isclose(
            charge,
            charge.round(),
            rtol=0.0,
            atol=1e-6,
        )
        spin_is_integer = torch.isclose(
            spin,
            spin.round(),
            rtol=0.0,
            atol=1e-6,
        )
        valid &= ~is_omol | (
            charge_valid
            & spin_valid
            & torch.isfinite(charge)
            & torch.isfinite(spin)
            & charge_is_integer
            & spin_is_integer
            & (spin >= 1)
        )
        return valid

    def _normalized_energy_target(
        self,
        *,
        loss_input: dict,
        f: dict,
        domain_id: torch.Tensor,
        base_valid: torch.Tensor,
        B: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build standardized residual-energy targets and composition validity."""
        device = domain_id.device
        target_dtype = get_device_compatible_float64_dtype(device)
        energy = self._as_system_vector(
            loss_input["energy_gt_B"], name="energy_gt_B", B=B
        ).to(device=device, dtype=target_dtype)
        energy_valid = self._as_system_vector(
            loss_input["energy_valid_mask_B"], name="energy_valid_mask_B", B=B
        ).to(device=device, dtype=torch.bool)
        source_count = self._as_system_vector(
            loss_input["source_atom_count_B"], name="source_atom_count_B", B=B
        ).to(device=device, dtype=target_dtype)

        atomic_number = loss_input["atom_type_gt_I"].to(device=device, dtype=torch.long)
        batch_token = f["batch_token_lvl"].to(device=device, dtype=torch.long)
        if atomic_number.ndim != 1 or atomic_number.shape != batch_token.shape:
            raise ValueError(
                "Expected atom_type_gt_I and batch_token_lvl to be aligned vectors; "
                f"got {atomic_number.shape} and {batch_token.shape}."
            )

        safe_domain = domain_id.clamp(0, len(self.domain_names) - 1)
        token_domain = safe_domain[batch_token]
        elemental_references = self.elemental_references.to(
            device=device, dtype=target_dtype
        )
        elemental_reference_valid = self.elemental_reference_valid.to(device=device)
        z_in_range = (atomic_number > 0) & (
            atomic_number < elemental_references.shape[1]
        )
        safe_z = atomic_number.clamp(0, elemental_references.shape[1] - 1)
        reference_valid_I = z_in_range & elemental_reference_valid[token_domain, safe_z]
        invalid_reference_count = torch.zeros(
            B, dtype=torch.int64, device=device
        ).scatter_add(0, batch_token, (~reference_valid_I).to(dtype=torch.int64))
        composition_valid = invalid_reference_count == 0

        reference_I = elemental_references[token_domain, safe_z]
        reference_sum = torch.zeros(B, dtype=target_dtype, device=device).scatter_add(
            0, batch_token, reference_I
        )
        normalized_float64 = (
            (energy - reference_sum) / source_count.clamp_min(1)
            - self.energy_mean.to(device=device, dtype=target_dtype)[safe_domain]
        ) / self.energy_scale.to(device=device, dtype=target_dtype)[safe_domain]
        normalized = normalized_float64.to(dtype=dtype)
        valid = (
            base_valid
            & energy_valid
            & composition_valid
            & torch.isfinite(energy)
            & torch.isfinite(normalized_float64)
            & torch.isfinite(normalized)
        )
        return normalized, valid

    @staticmethod
    def _mean_system_conditional_views(
        value: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Average each system's valid views, then average all systems.

        Systems without a valid view contribute a differentiable zero to the
        outer batch mean. This makes a packed batch identical to the arithmetic
        mean of evaluating its systems independently while keeping optional-head
        parameters in the backward graph on fully inactive ranks.

        Args:
            value: Per-view losses with shape ``[D, B]``.
            valid: Validity mask with the same shape as ``value``.

        Returns:
            Scalar mean over the ``B`` per-system conditional means.
        """
        if value.ndim != 2 or valid.shape != value.shape:
            raise ValueError(
                "Expected value and valid to have matching [D, B] shapes, got "
                f"{value.shape} and {valid.shape}."
            )
        masked_value = value.masked_fill(~valid, 0.0)
        valid_count_per_system = valid.sum(dim=0, dtype=value.dtype).clamp_min(1)
        mean_per_system = masked_value.sum(dim=0) / valid_count_per_system
        return mean_per_system.mean()

    def _force_loss(
        self,
        *,
        pred: torch.Tensor,
        loss_input: dict,
        f: dict,
        domain_id: torch.Tensor,
        base_valid: torch.Tensor,
        valid_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return system-balanced force loss and valid view-system mask."""
        D, L, coordinate_dim = pred.shape
        B = domain_id.numel()
        if coordinate_dim != 3:
            raise ValueError(
                f"Expected force vectors with 3 components, got {pred.shape}."
            )
        batch_atom = f["batch_atom_lvl"].to(device=pred.device, dtype=torch.long)
        if batch_atom.shape != (L,):
            raise ValueError(
                f"Expected batch_atom_lvl shape [{L}], got {batch_atom.shape}."
            )

        target = loss_input["force_gt_D_L"].to(device=pred.device, dtype=pred.dtype)
        if target.ndim == 2:
            target = target.unsqueeze(0).expand(D, -1, -1)
        if target.shape != pred.shape:
            raise ValueError(
                f"Expected force_gt_D_L shape {pred.shape}, got {target.shape}."
            )
        force_valid_L = loss_input["force_valid_mask_L"].to(
            device=pred.device, dtype=torch.bool
        )
        if force_valid_L.shape != (L,):
            raise ValueError(
                f"Expected force_valid_mask_L shape [{L}], got {force_valid_L.shape}."
            )

        safe_domain = domain_id.clamp(0, len(self.domain_names) - 1)
        scale_L = self.force_scale.to(device=pred.device, dtype=pred.dtype)[
            safe_domain[batch_atom]
        ]
        normalized_target = target / scale_L[None, :, None]
        finite_target = torch.isfinite(normalized_target).all(dim=-1)
        valid_DL = (
            valid_time[:, batch_atom]
            & base_valid[batch_atom][None]
            & force_valid_L[None]
            & finite_target
        )

        component_loss = F.smooth_l1_loss(
            pred,
            torch.nan_to_num(normalized_target),
            reduction="none",
            beta=self.huber_delta,
        )
        atom_loss = component_loss.mean(dim=-1)
        batch_atom_expanded = batch_atom[None].expand(D, -1)
        per_system_sum = torch.zeros(
            (D, B), device=pred.device, dtype=pred.dtype
        ).scatter_add(1, batch_atom_expanded, atom_loss * valid_DL.to(pred.dtype))
        per_system_count = torch.zeros_like(per_system_sum).scatter_add(
            1, batch_atom_expanded, valid_DL.to(pred.dtype)
        )
        valid_views = per_system_count > 0
        per_system = per_system_sum / per_system_count.clamp_min(1)
        return self._mean_system_conditional_views(per_system, valid_views), valid_views

    def _warmup_scale(self, loss_input: dict) -> float:
        """Return the linear optimizer-step warmup multiplier."""
        if self.warmup_steps == 0:
            return 1.0
        optimizer_step = max(0, int(loss_input.get("optimizer_step", 0)))
        return min(1.0, optimizer_step / self.warmup_steps)

    def forward(
        self, network_input, network_output, loss_input
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute gated, standardized near-clean energy and force losses."""
        self.shared_backbone_gradient_diagnostics = {}
        energy_pred = network_output["normalized_energy_B"]
        force_pred = network_output["normalized_forces_L"]
        if energy_pred.ndim != 2:
            raise ValueError(
                "Expected normalized_energy_B with shape [D, B], got "
                f"{energy_pred.shape}."
            )
        if force_pred.ndim != 3:
            raise ValueError(
                "Expected normalized_forces_L with shape [D, L, 3], got "
                f"{force_pred.shape}."
            )
        D, B = energy_pred.shape
        if force_pred.shape[0] != D:
            raise ValueError(
                "Energy and force outputs must share diffusion dimension D."
            )
        if network_input["t"].shape != (D, B):
            raise ValueError(
                f"Expected t shape {(D, B)}, got {network_input['t'].shape}."
            )

        # Target reference subtraction is handled in float64 above. Keep the
        # prediction and Smooth-L1 reductions in at least float32 when the
        # shared backbone runs under fp16/bf16 autocast.
        if energy_pred.dtype in (torch.float16, torch.bfloat16):
            energy_pred = energy_pred.float()
        if force_pred.dtype in (torch.float16, torch.bfloat16):
            force_pred = force_pred.float()

        f = network_input["f"]
        domain_id = self._as_system_vector(
            loss_input["domain_id_B"], name="domain_id_B", B=B
        ).to(device=energy_pred.device, dtype=torch.long)
        base_valid = self._base_system_validity(
            loss_input=loss_input, domain_id=domain_id, B=B
        )
        property_task_id = self._as_system_vector(
            loss_input["property_task_id_B"], name="property_task_id_B", B=B
        ).to(device=energy_pred.device, dtype=torch.long)
        property_task_valid = self._as_system_vector(
            loss_input["property_task_valid_mask_B"],
            name="property_task_valid_mask_B",
            B=B,
        ).to(device=energy_pred.device, dtype=torch.bool)
        is_unconditional = property_task_id == UNCONDITIONAL_PROPERTY_TASK_ID
        is_mlip = property_task_id == MLIP_PROPERTY_TASK_ID
        has_mean_force_conditioning = self._system_has_mean_force_conditioning(
            f,
            B=B,
            device=energy_pred.device,
        )
        base_valid &= (
            property_task_valid
            & (is_unconditional | is_mlip)
            & ~has_mean_force_conditioning
        )
        sigma_equivalent = self._equivalent_noise(
            network_input["t"].to(device=energy_pred.device)
        )
        energy_time_valid = (sigma_equivalent < self.energy_max_sigma) | is_mlip[None]
        force_time_valid = (sigma_equivalent < self.force_max_sigma) | is_mlip[None]

        normalized_energy_target, energy_system_valid = self._normalized_energy_target(
            loss_input=loss_input,
            f=f,
            domain_id=domain_id,
            base_valid=base_valid,
            B=B,
            dtype=energy_pred.dtype,
        )
        energy_valid_views = energy_time_valid & energy_system_valid[None]
        energy_element_loss = F.smooth_l1_loss(
            energy_pred,
            torch.nan_to_num(normalized_energy_target)[None].expand(D, -1),
            reduction="none",
            beta=self.huber_delta,
        )
        energy_loss = self._mean_system_conditional_views(
            energy_element_loss,
            energy_valid_views,
        )

        force_loss, force_valid_views = self._force_loss(
            pred=force_pred,
            loss_input=loss_input,
            f=f,
            domain_id=domain_id,
            base_valid=base_valid,
            valid_time=force_time_valid,
        )

        latent_equivariance_loss = energy_pred.sum() * 0.0
        latent_valid_pairs = torch.zeros(
            (max(D - 1, 0), B),
            dtype=torch.bool,
            device=energy_pred.device,
        )
        if self.latent_equivariance_weight > 0:
            required_latent_outputs = {
                "latent_equivariance_prediction_B",
                "latent_equivariance_target_B",
            }
            missing_latent_outputs = required_latent_outputs - network_output.keys()
            if missing_latent_outputs:
                raise ValueError(
                    "Enabled latent equivariance loss requires model outputs "
                    f"{sorted(required_latent_outputs)}; missing "
                    f"{sorted(missing_latent_outputs)}."
                )
            latent_prediction = network_output["latent_equivariance_prediction_B"]
            latent_target = network_output["latent_equivariance_target_B"]
            expected_latent_prefix = (D - 1, B)
            if (
                latent_prediction.ndim != 3
                or latent_prediction.shape[:2] != expected_latent_prefix
            ):
                raise ValueError(
                    "Expected latent equivariance predictions with shape "
                    f"[D - 1, B, C] and prefix {expected_latent_prefix}, got "
                    f"{tuple(latent_prediction.shape)}."
                )
            if latent_target.shape != latent_prediction.shape:
                raise ValueError(
                    "Latent equivariance prediction and target shapes must match; "
                    f"got {tuple(latent_prediction.shape)} and "
                    f"{tuple(latent_target.shape)}."
                )
            if latent_prediction.dtype in (torch.float16, torch.bfloat16):
                latent_prediction = latent_prediction.float()
                latent_target = latent_target.float()
            else:
                latent_target = latent_target.to(dtype=latent_prediction.dtype)
            latent_valid_systems = base_valid & is_mlip
            latent_valid_pairs = latent_valid_systems.unsqueeze(0).expand(D - 1, -1)
            latent_pair_loss = (latent_prediction - latent_target).square().mean(dim=-1)
            latent_equivariance_loss = self._mean_system_conditional_views(
                latent_pair_loss,
                latent_valid_pairs,
            )

        warmup_scale = self._warmup_scale(loss_input)
        combined = warmup_scale * (
            self.energy_weight * energy_loss
            + self.force_weight * force_loss
            + self.latent_equivariance_weight * latent_equivariance_loss
        )
        total = self.weight * combined
        if loss_input.get("collect_shared_backbone_gradient_diagnostics", False):
            self.shared_backbone_gradient_diagnostics[
                PROPERTY_GRADIENT_DIAGNOSTIC_NAME
            ] = total

        n_energy_views = energy_valid_views.sum().to(dtype=energy_pred.dtype)
        n_force_views = force_valid_views.sum().to(dtype=energy_pred.dtype)
        n_latent_pairs = latent_valid_pairs.sum().to(dtype=energy_pred.dtype)
        metrics = {
            "atomistic_property_loss": total.detach(),
            "atomistic_energy_loss": energy_loss.detach(),
            "atomistic_force_loss": force_loss.detach(),
            "atomistic_latent_equivariance_loss": (latent_equivariance_loss.detach()),
            "atomistic_property_warmup_scale": energy_pred.new_tensor(warmup_scale),
            "atomistic_energy_valid_fraction": (
                energy_valid_views.float().mean().detach()
            ),
            "atomistic_force_valid_fraction": (
                force_valid_views.float().mean().detach()
            ),
            "atomistic_energy_valid_views": n_energy_views.detach(),
            "atomistic_force_valid_views": n_force_views.detach(),
            "atomistic_latent_equivariance_valid_fraction": (
                latent_valid_pairs.float().mean().detach()
                if latent_valid_pairs.numel() > 0
                else energy_pred.new_zeros(())
            ),
            "atomistic_latent_equivariance_valid_pairs": n_latent_pairs.detach(),
        }
        return total, metrics


def _split_losses_by_t(
    l_global: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split per-diffusion-example losses into low- and high-t halves."""
    if l_global.shape != t.shape:
        raise ValueError(
            f"Expected l_global and t to have same shape, got {l_global.shape} and {t.shape}"
        )

    l_global_flat = l_global.reshape(-1).contiguous()
    t_flat = t.reshape(-1).contiguous()

    _, indices = torch.sort(t_flat)
    l_global_sorted = l_global_flat.gather(0, indices)
    if l_global_sorted.numel() == 1:
        return l_global_sorted, l_global_sorted

    n_low_t = l_global_sorted.numel() // 2
    return torch.split(
        l_global_sorted,
        [n_low_t, l_global_sorted.numel() - n_low_t],
    )


class DiffusionLoss(RFD3DiffusionLoss):
    """
    RFD3 diffusion loss for the inpainting task. Applies to all tokens,
    but can be weighted differently based on t and other factors.
    Also includes support for dataloader batch_size > 1.

    Args:
        weight: scalar weight for the loss
        sigma_data: scalar representing the noise level in the data, used for calculating the lambda weighting
        lddt_weight: scalar weight for the LDDT loss component
        alpha_virtual_atom: scalar weight for virtual atoms in the MSE loss
        alpha_unindexed_diffused: scalar weight for unindexed diffused tokens in the MSE loss
        alpha_polar_residues: scalar weight for polar residues in the MSE loss
        alpha_ligand: scalar weight for ligand atoms in the MSE loss
        unindexed_t_alpha: scalar to scale t for unindexed diffused tokens when calculating lambda
        unindexed_norm_p: p value for the lp norm on unindexed diffused tokens (if lp_weight > 0)
        lp_weight: scalar weight for the lp norm component of the loss on unindexed diffused tokens
    """

    def __init__(self, *, objective: str = "edm", **kwargs):
        super().__init__(**kwargs)
        if objective not in {"edm", FLOW_MATCHING, "unconditioned", "noise_pred"}:
            raise ValueError(
                f"Unsupported coordinate objective {objective!r}; expected "
                f"'edm', {FLOW_MATCHING!r}, 'unconditioned', or 'noise_pred'."
            )
        self.objective = objective
        self.loss_output_gradient_diagnostics = {}
        self.shared_backbone_gradient_diagnostics = {}

    def _time_weight(self, time: torch.Tensor) -> torch.Tensor:
        """Return EDM loss weighting or the unweighted flow objective."""
        if self.objective == FLOW_MATCHING:
            return torch.ones_like(time)
        return self.get_lambda(time)

    def forward(
        self, network_input, network_output, loss_input
    ) -> tuple[torch.Tensor, dict]:
        """
        Calculate the diffusion loss for the inpainting task with support for dataloader batch_size > 1.

        Args:
            network_input: dict containing the inputs to the network, including "t_L" (D, L) and "f" (dict of features)
            network_output: dict containing the outputs of the network, including "X_L" (D, L, 3)
            loss_input: dict containing the inputs for the loss calculation, including "X_gt_L_in_input_frame" (D, L, 3), "crd_mask_L" (L,), and the atom-wise "is_original_unindexed_token" (L,) mask

        Returns:
            loss: scalar tensor representing the diffusion loss
            outs: dict containing additional metrics, including "mse_loss_mean", "mse_loss_low_t", "mse_loss_high_t", "lp_norm", and "lp_norm_unindexed_diffused"
        """
        self.loss_output_gradient_diagnostics = {}
        self.shared_backbone_gradient_diagnostics = {}
        X_structure_L = network_output["X_L"]  # D, L, 3
        f = network_input["f"]
        D = f["diffusion_batch_size"]
        B = f["batch_size"]
        batch_atom_lvl = f["batch_atom_lvl"]
        batch_atom_lvl_expanded = batch_atom_lvl.unsqueeze(0).expand(D, -1)

        crd_mask_L = loss_input["crd_mask_L"].unsqueeze(0).expand(D, -1)  # (D, L)
        X_gt_structure_L = torch.nan_to_num(loss_input["X_gt_L_in_input_frame"])
        if self.objective == FLOW_MATCHING:
            if "V_L" not in network_output:
                raise KeyError(
                    "Flow-matching loss requires the model output key `V_L`."
                )
            X_L = network_output["V_L"]
            raw_t_L = network_input.get("t_L")
            if raw_t_L is None:
                raw_t_L = expand_t_to_level(
                    network_input["t"],
                    f,
                    level_len=X_L.shape[1],
                    batch_key="batch_atom_lvl",
                )
            denominator = (1 - raw_t_L).clamp_min(1e-6).unsqueeze(-1)
            X_noisy_structure_L = torch.nan_to_num(network_input["X_noisy_L"])
            X_gt_L = (X_gt_structure_L - X_noisy_structure_L) / (
                # x_t = (1 - t) * x_0 + t * x_1, so
                # (x_1 - x_t) / (1 - t) = x_1 - x_0, the target flow velocity.
                # V_L predicts this velocity normalized by sigma_data.
                denominator * self.sigma_data
            )
            crd_mask_L = (
                crd_mask_L
                & ~f["is_motif_atom_with_fixed_coord"].to(
                    device=crd_mask_L.device, dtype=torch.bool
                )[None]
            )
        else:
            X_L = X_structure_L
            X_gt_L = X_gt_structure_L
        tok_idx = f["atom_to_token_map"]
        tok_idx_per_batch_element = loss_input["atom_to_token_map_per_batch_element"]
        is_original_unindexed_token_batch_offset = loss_input[
            "is_original_unindexed_token_batch_offset"
        ]
        t = network_input["t_L"]  # (D, L)
        is_original_unindexed_token = loss_input["is_original_unindexed_token"][
            tok_idx_per_batch_element + is_original_unindexed_token_batch_offset
        ]  # (L,) bool mask indicating which atoms are original unindexed tokens
        is_polar_atom = f["is_polar"][tok_idx]
        is_ligand = f["is_ligand"][tok_idx]
        is_virtual_atom = f["is_virtual"]  # L
        is_sidechain_atom = f["is_sidechain"]  # L
        is_sidechain_atom = is_sidechain_atom & ~is_virtual_atom

        w_L = torch.ones_like(tok_idx, dtype=X_L.dtype)
        w_L[is_original_unindexed_token] = (
            w_L[is_original_unindexed_token] * self.alpha_unindexed_diffused
        )
        w_L[is_virtual_atom] *= self.alpha_virtual_atom
        w_L[is_ligand] *= self.alpha_ligand

        # Upweight polar residues
        w_L[is_polar_atom] *= self.alpha_polar_residues
        w_L = w_L[None].expand(D, -1) * crd_mask_L

        valid_atoms_per_batch_element = torch.zeros(
            (D, B), device=crd_mask_L.device, dtype=X_L.dtype
        ).scatter_add(
            1,
            batch_atom_lvl_expanded,
            crd_mask_L.to(X_L.dtype),
        )

        l_mse_L = w_L * torch.sum((X_L - X_gt_L) ** 2, dim=-1)
        l_mse_L = torch.div(
            l_mse_L, 3 * valid_atoms_per_batch_element[:, batch_atom_lvl] + 1e-4
        )  # D, L

        if torch.any(is_original_unindexed_token):
            t_exp = (
                t * (~is_original_unindexed_token)
                + self.unindexed_t_alpha * t * is_original_unindexed_token
            )  # [D, L]

            l_global = self._time_weight(t_exp) * l_mse_L  # [D, L]
            l_global_per_batch = torch.zeros((D, B), device=l_global.device)
            l_global_per_batch.scatter_add_(
                1,
                batch_atom_lvl_expanded,
                l_global,
            )

            # Get renormalization factor to equalize expectation of the loss
            r = self._time_weight(t * self.unindexed_t_alpha) / self._time_weight(t)
            r = torch.zeros((D, B), device=r.device).scatter_reduce(
                1, batch_atom_lvl_expanded, r, reduce="amax"
            )  # [D, B]

            is_original_crd_mask_L_sum = torch.zeros(
                (D, B), device=crd_mask_L.device
            ).scatter_add(
                1,
                batch_atom_lvl_expanded[:, is_original_unindexed_token],
                crd_mask_L[:, is_original_unindexed_token].float(),
            )
            is_not_original_crd_mask_L_sum = torch.zeros(
                (D, B), device=crd_mask_L.device
            ).scatter_add(
                1,
                batch_atom_lvl_expanded[:, ~is_original_unindexed_token],
                crd_mask_L[:, ~is_original_unindexed_token].float(),
            )

            t_factor_denominator = (
                r * is_original_crd_mask_L_sum + is_not_original_crd_mask_L_sum
            )
            has_valid_atoms = valid_atoms_per_batch_element > 0
            # A packed batch can mix an unindexed generative system with an
            # all-fixed task such as MLIP. The global unindexed branch is then
            # active, but the fixed system has a 0 / 0 renormalization factor.
            # Give inactive systems a neutral denominator and an explicit zero
            # factor while leaving every active system's arithmetic unchanged.
            safe_denominator = torch.where(
                has_valid_atoms,
                t_factor_denominator,
                torch.ones_like(t_factor_denominator),
            )
            t_factor = valid_atoms_per_batch_element / safe_denominator
            t_factor = t_factor.masked_fill(~has_valid_atoms, 0.0)
            l_global = l_global_per_batch * t_factor
        else:
            l_global = self._time_weight(t) * l_mse_L  # [D, L]
            l_global = torch.zeros((D, B), device=l_global.device).scatter_add(
                1, batch_atom_lvl_expanded, l_global
            )

        assert l_global.shape == (D, B), l_global.shape

        if torch.any(is_original_unindexed_token):
            lp_norm_L = w_L * torch.linalg.norm(
                X_L - X_gt_L, ord=self.unindexed_norm_p, dim=-1
            )  # [D, L]
            lp_norm_unindexed_diffused = lp_norm_L * is_original_unindexed_token[None]

            # Scatter add for normalization per batch element
            tokens_per_batch = torch.zeros((D, B), device=lp_norm_L.device).scatter_add(
                1,
                batch_atom_lvl_expanded,
                is_original_unindexed_token[None].float().expand(D, -1),
            )
            norm_factor = self.alpha_unindexed_diffused * 3 * tokens_per_batch + 1e-4

            lp_norm_unindexed_diffused = torch.div(
                lp_norm_unindexed_diffused, norm_factor[:, batch_atom_lvl]
            )  # D, L

            lp_norm_unindexed_diffused_per_batch = torch.zeros(
                (D, B), device=lp_norm_L.device
            ).scatter_add(1, batch_atom_lvl_expanded, lp_norm_unindexed_diffused)

            lp_norm_unindexed_diffused = (
                lp_norm_unindexed_diffused_per_batch
                * self._time_weight(self.unindexed_t_alpha * network_input["t"])
            )

            l_total = l_global + self.lp_weight * lp_norm_unindexed_diffused
        else:
            lp_norm_unindexed_diffused = None
            lp_norm_L = None
            l_total = l_global

        # ... Aggregate
        l_mse_per_view = torch.clamp(l_total, max=10)
        assert l_mse_per_view.shape == (
            D,
            B,
        ), f"Expected l_total to be of shape (D, B), got {l_total.shape}"
        if self.objective == FLOW_MATCHING and loss_input.get(
            "collect_shared_backbone_gradient_diagnostics", False
        ):
            valid_flow_view = valid_atoms_per_batch_element > 0
            flow_time = network_input["t"]
            flow_noise_bins = {
                "flow_mse_low_noise": flow_time >= 2.0 / 3.0,
                "flow_mse_mid_noise": (flow_time >= 1.0 / 3.0)
                & (flow_time < 2.0 / 3.0),
                "flow_mse_high_noise": flow_time < 1.0 / 3.0,
            }
            for diagnostic_name, noise_bin_mask in flow_noise_bins.items():
                valid_bin = valid_flow_view & noise_bin_mask
                masked_loss = l_mse_per_view.masked_fill(~valid_bin, 0.0)
                bin_loss = masked_loss.sum() / valid_bin.sum(
                    dtype=l_mse_per_view.dtype
                ).clamp_min(1)
                self.shared_backbone_gradient_diagnostics[diagnostic_name] = (
                    self.weight * bin_loss
                )

        l_mse_total = torch.mean(l_mse_per_view)  # D, B -> scalar
        self.loss_output_gradient_diagnostics["mse_loss"] = (
            self.weight * l_mse_total,
            X_L,
        )

        # ... Return
        if self.lddt_weight > 0:
            # ... Calculate LDDT loss at the beginning
            lddt_batch_weight = None
            if self.objective == FLOW_MATCHING:
                # Emyx: alpha(t) = alpha_0 * (1 + 8 * ReLU(t - 0.5)).
                lddt_batch_weight = 1 + 8 * torch.relu(network_input["t"] - 0.5)
            smoothed_lddt_loss_, lddt_loss_dict = smoothed_lddt_loss(
                X_structure_L,
                X_gt_structure_L,
                crd_mask_L,
                f["is_dna"],
                f["is_rna"],
                f["is_ligand"],
                tok_idx,
                batch_atom_lvl,
                batch_loss_weight=lddt_batch_weight,
                return_extras=True,
            )  # D,
            l_total = l_mse_total + self.lddt_weight * smoothed_lddt_loss_.mean()
        else:
            lddt_loss_dict = {}
            l_total = l_mse_total
        # ... Return additional losses
        diagnostic_time = network_input["t"]
        if self.objective == FLOW_MATCHING:
            diagnostic_time = flow_time_to_edm_noise(diagnostic_time, self.sigma_data)
        l_mse_low, l_mse_high = _split_losses_by_t(l_global, diagnostic_time)
        # Report the unindexed-diffusion statistics on every batch, zeroed when
        # the batch holds no unindexed tokens, as the material losses already
        # do for absent targets. Emitting them conditionally gives DDP ranks
        # different metric keys, which desynchronizes epoch-end logging.
        zero = l_mse_total.new_zeros(())
        loss_dict = {
            "mse_loss_mean": l_mse_total,
            "mse_loss_low_t": l_mse_low,
            "mse_loss_high_t": l_mse_high,
            "lp_norm": zero if lp_norm_L is None else lp_norm_L,
            "lp_norm_unindexed_diffused": (
                zero
                if lp_norm_unindexed_diffused is None
                else lp_norm_unindexed_diffused
            ),
        } | lddt_loss_dict
        loss_dict = {
            k: torch.mean(v).detach() for k, v in loss_dict.items() if v is not None
        }

        return self.weight * l_total, loss_dict


def smoothed_lddt_loss(
    X_L,
    X_gt_L,
    crd_mask_L,
    is_dna,
    is_rna,
    is_ligand,
    tok_idx,
    batch_atom_lvl,
    is_virtual=None,
    alpha_virtual=1.0,
    return_extras=False,
    eps=1e-6,
    batch_loss_weight=None,
) -> tuple[torch.Tensor, dict]:
    """
    Calculate a smoothed lDDT loss between predicted and ground truth coordinates,
    optionally returning additional metrics on the lDDT scores for different subsets
    of atoms. Also includes support for dataloader batch_size > 1 by only comparing
    pairs of atoms within the same batch element.

    Args:
        X_L: (D, L, 3) tensor of predicted coordinates
        X_gt_L: (D, L, 3) tensor of ground truth coordinates
        crd_mask_L: (D, L) tensor indicating which coordinates are valid
        is_dna: (num_tokens,) bool tensor indicating which tokens are DNA
        is_rna: (num_tokens,) bool tensor indicating which tokens are RNA
        is_ligand: (num_tokens,) bool tensor indicating which tokens are ligand atoms
        tok_idx: (L,) tensor mapping each atom to its corresponding token index
        batch_atom_lvl: (L,) tensor indicating the batch element for each atom
        is_virtual: (L,) bool tensor indicating which atoms are virtual (optional)
        alpha_virtual: scalar weight to apply to pairs of virtual atoms in the loss (optional)
        return_extras: whether to return additional metrics on the lDDT scores for different subsets of atoms (optional)
        eps: small constant to avoid division by zero
        batch_loss_weight: (D, B) optional multiplicative loss weight for each
            diffusion and dataloader batch example. When omitted, every example
            receives unit weight; lDDT is still reduced within each system
            before the batch mean.

    Returns:
        lddt_loss: (D,) tensor of lDDT losses for each diffusion step
        extra_lddts: dict of additional lDDT metrics (if return_extras is True)
    """

    if batch_atom_lvl.numel() == 0:
        raise ValueError("smoothed_lddt_loss requires at least one atom.")
    batch_size = int(batch_atom_lvl.max().item()) + 1
    if batch_loss_weight is None:
        batch_loss_weight = X_L.new_ones((X_L.shape[0], batch_size))
    elif batch_loss_weight.ndim == 1:
        batch_loss_weight = batch_loss_weight.unsqueeze(-1)
    expected_weight_shape = (X_L.shape[0], batch_size)
    if batch_loss_weight.shape != expected_weight_shape:
        raise ValueError(
            "Expected batch_loss_weight to have shape (D, B) "
            f"{expected_weight_shape}, got {tuple(batch_loss_weight.shape)}."
        )
    batch_loss_weight = batch_loss_weight.to(device=X_L.device, dtype=X_L.dtype)

    @activation_checkpointing
    def _dolddt(
        X_L,
        X_gt_L,
        crd_mask_L,
        is_dna,
        is_rna,
        is_ligand,
        tok_idx,
        batch_atom_lvl,
        eps,
    ) -> tuple[torch.Tensor, dict]:
        """
        Calculate a smoothed lDDT loss between predicted and ground truth coordinates.

        Args:
            X_L: (D, L, 3) tensor of predicted coordinates
            X_gt_L: (D, L, 3) tensor of ground truth coordinates
            crd_mask_L: (D, L) tensor indicating which coordinates are valid
            is_dna: (num_tokens,) bool tensor indicating which tokens are DNA
            is_rna: (num_tokens,) bool tensor indicating which tokens are RNA
            is_ligand: (num_tokens,) bool tensor indicating which tokens are ligand atoms
            tok_idx: (L,) tensor mapping each atom to its corresponding token index
            batch_atom_lvl: (L,) tensor indicating the batch element for each atom
            eps: small constant to avoid division by zero

        Returns:
            lddt_loss: (D,) tensor of lDDT losses for each diffusion step
            extra_lddts: dict of additional lDDT metrics (if return_extras is True)
        """
        _, L = X_L.shape[:2]
        first_index, second_index = torch.triu_indices(L, L, 1, device=X_L.device)

        # compute the unique distances between all pairs of atoms
        X_gt_L = X_gt_L.nan_to_num()

        # only use native 1 (assumes dist map identical btwn all copies)
        ground_truth_distances = torch.linalg.norm(
            X_gt_L[0:1, first_index] - X_gt_L[0:1, second_index], dim=-1
        )

        # only score pairs that are close enough in the ground truth
        is_na_L = is_dna[tok_idx][first_index] | is_rna[tok_idx][first_index]
        pair_mask = torch.logical_and(
            ground_truth_distances > 0,
            ground_truth_distances < torch.where(is_na_L, 30.0, 15.0),
        )
        del is_na_L

        # only score pairs that are resolved in the ground truth
        pair_mask *= crd_mask_L[0:1, first_index] * crd_mask_L[0:1, second_index]

        # don't score pairs that are in the same token
        pair_mask *= tok_idx[None, first_index] != tok_idx[None, second_index]

        # don't score pairs that reside in different batch elements
        pair_mask *= (
            batch_atom_lvl[None, first_index] == batch_atom_lvl[None, second_index]
        )

        _, valid_pairs = pair_mask.nonzero(as_tuple=True)
        pair_mask = pair_mask[:, valid_pairs].to(X_L.dtype)
        ground_truth_distances = ground_truth_distances[:, valid_pairs]
        first_index, second_index = first_index[valid_pairs], second_index[valid_pairs]

        predicted_distances = torch.linalg.norm(
            X_L[:, first_index] - X_L[:, second_index], dim=-1
        )

        delta_distances = torch.abs(predicted_distances - ground_truth_distances + eps)
        del predicted_distances, ground_truth_distances

        if is_virtual is not None:
            pair_mask[:, (is_virtual[first_index] * is_virtual[second_index])] *= (
                alpha_virtual
            )

        # I assume gradients flow better if we sum first rather than keeping everything in D, L...
        lddt = (
            0.25
            * (
                torch.sum(torch.sigmoid(0.5 - delta_distances) * pair_mask, dim=(1))
                + torch.sum(torch.sigmoid(1.0 - delta_distances) * pair_mask, dim=(1))
                + torch.sum(torch.sigmoid(2.0 - delta_distances) * pair_mask, dim=(1))
                + torch.sum(torch.sigmoid(4.0 - delta_distances) * pair_mask, dim=(1))
            )
            / (torch.sum(pair_mask, dim=(1)) + eps)
        )
        lddt_loss = 1 - lddt

        if batch_loss_weight is not None:
            D, _ = batch_loss_weight.shape
            pair_lddt = 0.25 * (
                torch.sigmoid(0.5 - delta_distances)
                + torch.sigmoid(1.0 - delta_distances)
                + torch.sigmoid(2.0 - delta_distances)
                + torch.sigmoid(4.0 - delta_distances)
            )
            pair_batch = batch_atom_lvl[first_index][None].expand(D, -1)
            pair_mask_expanded = pair_mask.expand(D, -1)
            if pair_batch.shape[1] == 0:
                # MPS can return uninitialized values when scatter_add receives
                # an empty index tensor. Keep the zero connected to X_L so an
                # all-fixed batch still supports backward with zero gradients.
                score_sum = pair_lddt.sum(dim=1, keepdim=True).expand_as(
                    batch_loss_weight
                )
                pair_weight_sum = torch.zeros_like(batch_loss_weight)
            else:
                score_sum = torch.zeros_like(batch_loss_weight).scatter_add(
                    1, pair_batch, pair_lddt * pair_mask_expanded
                )
                pair_weight_sum = torch.zeros_like(batch_loss_weight).scatter_add(
                    1, pair_batch, pair_mask_expanded
                )
            has_valid_pairs = pair_weight_sum > 0
            per_batch_lddt_loss = torch.where(
                has_valid_pairs,
                1 - score_sum / pair_weight_sum.clamp_min(eps),
                torch.zeros_like(score_sum),
            )
            lddt_loss = (batch_loss_weight * per_batch_lddt_loss).mean(dim=1)

        if not return_extras:
            return lddt_loss, {}

        # ...Hence we recalculate the losses here and pick out the parts of interest
        with torch.no_grad():
            lddt_ = (
                0.25
                * (
                    torch.sigmoid(0.5 - delta_distances)
                    + torch.sigmoid(1.0 - delta_distances)
                    + torch.sigmoid(2.0 - delta_distances)
                    + torch.sigmoid(4.0 - delta_distances)
                )
                * pair_mask
                / (torch.sum(pair_mask, dim=(1)) + eps)
            )

            def filter_lddt(mask, scale=1.0):
                mask = mask.to(pair_mask.dtype)
                if mask.ndim > 1:
                    mask = mask[0]
                mask = (mask[first_index] * mask[second_index])[None].expand(
                    pair_mask.shape[0], -1
                )
                mask = (mask * pair_mask).to(bool)
                return (
                    (1 - torch.sum(lddt_[:, mask[0]] * scale, dim=(1))).mean().detach()
                )

            extra_lddts = {}
            extra_lddts["mean_lddt"] = filter_lddt(
                torch.full_like(crd_mask_L, 1.0, device=X_L.device)
            )
            extra_lddts["mean_lddt_dna"] = filter_lddt(is_dna[tok_idx])
            extra_lddts["mean_lddt_rna"] = filter_lddt(is_rna[tok_idx])
            extra_lddts["mean_lddt_ligand"] = filter_lddt(is_ligand[tok_idx])
            extra_lddts["mean_lddt_protein"] = filter_lddt(
                ~is_dna[tok_idx] & ~is_rna[tok_idx] & ~is_ligand[tok_idx]
            )
            # NOTE: This also seems to have issues at epoch level, as with n_valid_t
            # before. Will leave as-is for now but may want to spoof as 0 in the future.
            if is_virtual is not None:
                extra_lddts["mean_lddt_virtual"] = filter_lddt(
                    is_virtual, scale=1 / alpha_virtual
                )
                extra_lddts["mean_lddt_non_virtual"] = filter_lddt(~is_virtual)

        return lddt_loss, extra_lddts

    return _dolddt(
        X_L, X_gt_L, crd_mask_L, is_dna, is_rna, is_ligand, tok_idx, batch_atom_lvl, eps
    )
