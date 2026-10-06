import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from atomworks.ml.example_id import parse_example_id
from beartype.typing import Any
from foundry.callbacks.callback import BaseCallback
from foundry.callbacks.train_logging import (
    LogAF3TrainingLossesCallback as FoundryLogAF3TrainingLossesCallback,
)
from foundry.callbacks.train_logging import (
    LogDatasetSamplingRatiosCallback as FoundryLogDatasetSamplingRatiosCallback,
)
from foundry.callbacks.train_logging import (
    PrintExampleIDBeforeForwardPassCallback as FoundryPrintExampleIDBeforeForwardPassCallback,
)
from foundry.utils.logging import (
    safe_print,
    table_from_df,
)
from rf3.utils.loss import convert_batched_losses_to_list_of_dicts
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from torchmetrics.aggregation import MeanMetric

from nucleus.metrics.tensor_diagnostics import ModelDiagnostics


def _epoch_progress_percentage(trainer: Any, batch: Any) -> float | None:
    """Return rank-local epoch progress based on examples, not packed batches."""
    if not trainer.state.get("epoch_progress_is_exact", True):
        return None
    completed_examples = int(trainer.state.get("examples_completed_in_epoch", 0))
    current_batch_size = len(batch) if isinstance(batch, list) else 1
    total_examples = int(
        getattr(trainer, "n_examples_per_rank", trainer.n_batches_per_epoch)
    )
    if total_examples <= 0:
        return 0.0
    return min(
        100.0, 100.0 * (completed_examples + current_batch_size) / total_examples
    )


class LogModelHealthCallback(BaseCallback):
    """Log compact, pre-clipping gradient and parameter health statistics.

    Statistics are aggregated across the full model and a small number of named
    parameter groups. This keeps experiment dashboards useful without creating
    one metric for every parameter tensor.

    Args:
        log_freq: Log every ``log_freq`` optimizer steps.
        parameter_groups: Mapping from dashboard group names to substrings used
            to match parameter names. A parameter may contribute to more than
            one group when patterns overlap.
    """

    def __init__(
        self,
        log_freq: int = 100,
        parameter_groups: dict[str, list[str]] | None = None,
    ) -> None:
        if log_freq < 1:
            raise ValueError(f"log_freq must be positive, got {log_freq}")

        self.log_freq = log_freq
        self.parameter_groups = {
            group_name: tuple(patterns)
            for group_name, patterns in (parameter_groups or {}).items()
        }

    def on_before_optimizer_step(self, trainer: Any, optimizer: Any) -> None:
        """Log synchronized gradients before gradient clipping is applied."""
        step = int(trainer.state["global_step"])
        if step % self.log_freq != 0 or not trainer.fabric.is_global_zero:
            return

        model = trainer.state["model"]
        aggregate_stats: dict[str, dict[str, Any]] = {}

        for parameter_name, parameter in model.named_parameters():
            # EMA shadow parameters are detached. Restrict health statistics to
            # the live parameters that the optimizer can actually update.
            if not parameter.requires_grad or parameter.numel() == 0:
                continue

            weight_stats = self._tensor_stats(parameter)
            grad_stats = (
                self._tensor_stats(parameter.grad)
                if parameter.grad is not None
                else None
            )

            matching_groups = [
                group_name
                for group_name, patterns in self.parameter_groups.items()
                if any(pattern in parameter_name for pattern in patterns)
            ]
            for group_name in ["global", *matching_groups]:
                if group_name not in aggregate_stats:
                    aggregate_stats[group_name] = self._empty_stats(parameter.device)
                self._update_stats(
                    aggregate_stats[group_name],
                    weight_stats=weight_stats,
                    weight_numel=parameter.numel(),
                    grad_stats=grad_stats,
                    grad_numel=parameter.grad.numel()
                    if parameter.grad is not None
                    else 0,
                )

        metrics: dict[str, torch.Tensor] = {}
        for group_name, stats in aggregate_stats.items():
            prefix = (
                "train/health"
                if group_name == "global"
                else (f"train/health/groups/{group_name}")
            )
            metrics.update(
                self._finalize_stats(
                    prefix, stats, include_detailed_stats=group_name == "global"
                )
            )

        if metrics:
            trainer.fabric.log_dict(metrics, step=step)

    @staticmethod
    def _empty_stats(device: torch.device) -> dict[str, Any]:
        return {
            "weight_sum_sq": torch.zeros((), dtype=torch.float32, device=device),
            "weight_max_abs": torch.zeros((), dtype=torch.float32, device=device),
            "weight_nonfinite": torch.zeros((), dtype=torch.int64, device=device),
            "weight_numel": 0,
            "grad_sum_sq": torch.zeros((), dtype=torch.float32, device=device),
            "grad_max_abs": torch.zeros((), dtype=torch.float32, device=device),
            "grad_nonfinite": torch.zeros((), dtype=torch.int64, device=device),
            "grad_numel": 0,
        }

    @staticmethod
    def _tensor_stats(tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
        tensor_float = tensor.detach().to(dtype=torch.float32)
        finite_mask = torch.isfinite(tensor_float)
        finite_tensor = torch.where(finite_mask, tensor_float, 0.0)
        return (
            finite_tensor.square().sum(),
            finite_tensor.abs().amax(),
            (~finite_mask).sum(),
        )

    @staticmethod
    def _update_stats(
        stats: dict[str, Any],
        weight_stats: tuple[torch.Tensor, ...],
        weight_numel: int,
        grad_stats: tuple[torch.Tensor, ...] | None,
        grad_numel: int,
    ) -> None:
        weight_sum_sq, weight_max_abs, weight_nonfinite = weight_stats
        stats["weight_sum_sq"] += weight_sum_sq
        stats["weight_max_abs"] = torch.maximum(stats["weight_max_abs"], weight_max_abs)
        stats["weight_nonfinite"] += weight_nonfinite
        stats["weight_numel"] += weight_numel

        if grad_stats is None:
            return

        grad_sum_sq, grad_max_abs, grad_nonfinite = grad_stats
        stats["grad_sum_sq"] += grad_sum_sq
        stats["grad_max_abs"] = torch.maximum(stats["grad_max_abs"], grad_max_abs)
        stats["grad_nonfinite"] += grad_nonfinite
        stats["grad_numel"] += grad_numel

    @staticmethod
    def _finalize_stats(
        prefix: str,
        stats: dict[str, Any],
        include_detailed_stats: bool,
    ) -> dict[str, torch.Tensor]:
        weight_norm = stats["weight_sum_sq"].sqrt()
        grad_norm = stats["grad_sum_sq"].sqrt()
        weight_numel = max(stats["weight_numel"], 1)
        grad_numel = max(stats["grad_numel"], 1)
        denominator = weight_norm.clamp_min(torch.finfo(torch.float32).eps)

        metrics = {
            f"{prefix}/weights/rms": (stats["weight_sum_sq"] / weight_numel).sqrt(),
            f"{prefix}/grads/rms": (stats["grad_sum_sq"] / grad_numel).sqrt(),
            f"{prefix}/grad_to_weight_norm": grad_norm / denominator,
            f"{prefix}/gradient_element_coverage": torch.tensor(
                stats["grad_numel"] / weight_numel,
                dtype=torch.float32,
                device=weight_norm.device,
            ),
        }
        if not include_detailed_stats:
            return metrics

        metrics.update(
            {
                f"{prefix}/weights/l2_norm": weight_norm,
                f"{prefix}/weights/max_abs": stats["weight_max_abs"],
                f"{prefix}/weights/nonfinite_count": stats["weight_nonfinite"],
                f"{prefix}/grads/l2_norm": grad_norm,
                f"{prefix}/grads/max_abs": stats["grad_max_abs"],
                f"{prefix}/grads/nonfinite_count": stats["grad_nonfinite"],
            }
        )
        return metrics


class LogConditioningGradientNormsCallback(BaseCallback):
    """Log loss sensitivity at conditioning, token, and atom embeddings.

    Forward hooks capture the selected module outputs, while tensor hooks collect
    their gradients during the ordinary training backward pass. Statistics are
    reduced across ranks and normalized by the number of activation elements so
    feature tracks with different widths or numbers of atoms remain comparable.

    This callback is intended for occasional diagnostics. When disabled, it does
    not install hooks. When enabled, hooks record tensors only on optimizer steps
    selected by ``log_every_n``.

    Args:
        enabled: Whether to install hooks and collect diagnostics.
        log_every_n: Log every ``log_every_n`` optimizer steps.
        module_groups: Mapping from metric group names to exact module names.
            A name may omit wrapping prefixes introduced by Fabric/DDP; suffix
            matches are accepted. The defaults target Zatom's force projection,
            combined token/graph embedding, and combined atom embedding.
        prefix: Metric namespace passed to the configured Fabric loggers.
    """

    _DEFAULT_MODULE_GROUPS = {
        "force_embedding": (
            "token_initializer.token_1d_bottleneck_embedder.embedders.mean_force_norm",
            "token_initializer.token_1d_bottleneck_embedder.embedders.has_mean_force_norm",
            "token_initializer.token_1d_bottleneck_embedder.embedders.force_rms_p95",
            "token_initializer.token_1d_bottleneck_embedder.embedders.has_force_rms_p95",
            "token_initializer.token_1d_bottleneck_embedder.residual_embedder",
        ),
        "graph_features": ("token_initializer.token_1d_bottleneck_embedder",),
        "atom_features": ("token_initializer.atom_1d_embedder",),
    }

    def __init__(
        self,
        enabled: bool = False,
        log_every_n: int = 100,
        module_groups: dict[str, list[str]] | None = None,
        prefix: str = "train/conditioning_gradients",
    ) -> None:
        if log_every_n < 1:
            raise ValueError(f"log_every_n must be positive, got {log_every_n}")
        self.enabled = bool(enabled)
        self.log_every_n = log_every_n
        groups = self._DEFAULT_MODULE_GROUPS if module_groups is None else module_groups
        if not groups or any(not patterns for patterns in groups.values()):
            raise ValueError("module_groups must contain at least one name per group.")
        self.module_groups = {
            group_name: tuple(patterns) for group_name, patterns in groups.items()
        }
        self.prefix = prefix.rstrip("/")
        self._hook_handles: list[Any] = []
        self._capture = False
        self._backward_scale = 1.0
        self._stats: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _module_name_matches(module_name: str, configured_name: str) -> bool:
        """Match exact names while tolerating Fabric/DDP wrapper prefixes."""
        return module_name == configured_name or module_name.endswith(
            f".{configured_name}"
        )

    def on_fit_start(self, trainer: Any) -> None:
        """Resolve target modules and install lightweight forward hooks."""
        if not self.enabled:
            return

        # Avoid duplicate hooks if the same callback is reused for another fit.
        self._remove_hooks()
        matched_groups = set()
        for module_name, module in trainer.state["model"].named_modules():
            groups = tuple(
                group_name
                for group_name, configured_names in self.module_groups.items()
                if any(
                    self._module_name_matches(module_name, configured_name)
                    for configured_name in configured_names
                )
            )
            if not groups:
                continue
            matched_groups.update(groups)
            self._hook_handles.append(
                module.register_forward_hook(self._make_forward_hook(groups))
            )

        missing_groups = set(self.module_groups) - matched_groups
        if missing_groups:
            self._remove_hooks()
            raise ValueError(
                "Conditioning-gradient modules were not found for groups: "
                f"{sorted(missing_groups)}. Check module_groups against the model."
            )

    def on_fit_end(self, trainer: Any) -> None:
        """Remove module hooks when training finishes."""
        del trainer
        self._remove_hooks()

    def _remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def on_train_batch_start(self, trainer: Any, batch: Any, batch_idx: int) -> None:
        """Enable capture for all microbatches of a selected optimizer step."""
        del batch, batch_idx
        step = int(trainer.state["global_step"])
        self._capture = self.enabled and step % self.log_every_n == 0
        if self._capture:
            precision = getattr(
                getattr(trainer.fabric, "strategy", None), "precision", None
            )
            scaler = getattr(precision, "scaler", None)
            self._backward_scale = (
                float(scaler.get_scale()) if scaler is not None else 1.0
            )
        if self._capture and not self._stats:
            self._stats = {
                group_name: self._empty_activation_stats()
                for group_name in self.module_groups
            }

    @staticmethod
    def _empty_activation_stats() -> dict[str, Any]:
        return {
            "activation_sum_sq": None,
            "activation_numel": 0,
            "grad_sum_sq": None,
            "grad_numel": 0,
        }

    def _make_forward_hook(self, groups: tuple[str, ...]):
        def capture_output(module: Any, inputs: Any, output: Any) -> None:
            del module, inputs
            if not self._capture or not isinstance(output, torch.Tensor):
                return
            # A checkpointed module's first forward runs under no_grad and is
            # recomputed during backward. Record only the differentiable pass.
            if not output.requires_grad:
                return

            activation_sum_sq = output.detach().float().square().sum()
            for group_name in groups:
                self._accumulate(
                    group_name,
                    "activation_sum_sq",
                    activation_sum_sq,
                    "activation_numel",
                    output.numel(),
                )

            def capture_gradient(gradient: torch.Tensor) -> None:
                # Fabric scales the loss for float16 AMP. Undo that common
                # factor so absolute norms remain comparable as GradScaler's
                # dynamic scale changes; ratios would cancel it either way.
                gradient_float = gradient.detach().float() / self._backward_scale
                gradient_sum_sq = gradient_float.square().sum()
                for group_name in groups:
                    self._accumulate(
                        group_name,
                        "grad_sum_sq",
                        gradient_sum_sq,
                        "grad_numel",
                        gradient.numel(),
                    )

            output.register_hook(capture_gradient)

        return capture_output

    def _accumulate(
        self,
        group_name: str,
        sum_key: str,
        sum_value: torch.Tensor,
        count_key: str,
        count_value: int,
    ) -> None:
        stats = self._stats[group_name]
        if stats[sum_key] is None:
            stats[sum_key] = sum_value
        else:
            stats[sum_key] = stats[sum_key] + sum_value
        stats[count_key] += count_value

    def on_before_optimizer_step(self, trainer: Any, optimizer: Any) -> None:
        """Reduce and log captured gradients before clearing step statistics."""
        del optimizer
        if not self._capture:
            return

        device = next(trainer.state["model"].parameters()).device
        local_rows = []
        for group_name in self.module_groups:
            stats = self._stats[group_name]
            activation_sum_sq = stats["activation_sum_sq"]
            grad_sum_sq = stats["grad_sum_sq"]
            local_rows.append(
                torch.stack(
                    (
                        activation_sum_sq
                        if activation_sum_sq is not None
                        else torch.zeros((), device=device),
                        torch.tensor(
                            stats["activation_numel"],
                            dtype=torch.float32,
                            device=device,
                        ),
                        grad_sum_sq
                        if grad_sum_sq is not None
                        else torch.zeros((), device=device),
                        torch.tensor(
                            stats["grad_numel"],
                            dtype=torch.float32,
                            device=device,
                        ),
                    )
                )
            )

        reduced = trainer.fabric.all_reduce(torch.stack(local_rows), reduce_op="sum")
        metrics: dict[str, torch.Tensor] = {}
        gradient_rms: dict[str, torch.Tensor] = {}
        epsilon = torch.finfo(torch.float32).eps
        for group_name, row in zip(self.module_groups, reduced, strict=True):
            activation_sum_sq, activation_numel, grad_sum_sq, grad_numel = row
            activation_rms = (
                activation_sum_sq / activation_numel.clamp_min(1.0)
            ).sqrt()
            group_gradient_rms = (grad_sum_sq / grad_numel.clamp_min(1.0)).sqrt()
            gradient_rms[group_name] = group_gradient_rms
            group_prefix = f"{self.prefix}/{group_name}"
            metrics.update(
                {
                    f"{group_prefix}/activation_rms": activation_rms,
                    f"{group_prefix}/gradient_l2_norm": grad_sum_sq.sqrt(),
                    f"{group_prefix}/gradient_rms": group_gradient_rms,
                    f"{group_prefix}/gradient_to_activation_rms": (
                        group_gradient_rms / activation_rms.clamp_min(epsilon)
                    ),
                    f"{group_prefix}/gradient_element_coverage": (
                        grad_numel / activation_numel.clamp_min(1.0)
                    ),
                }
            )

        force_rms = gradient_rms.get("force_embedding")
        if force_rms is not None:
            for reference_group in ("graph_features", "atom_features"):
                reference_rms = gradient_rms.get(reference_group)
                if reference_rms is not None:
                    metrics[
                        f"{self.prefix}/force_to_{reference_group}_gradient_rms"
                    ] = force_rms / reference_rms.clamp_min(epsilon)

        if trainer.fabric.is_global_zero:
            trainer.fabric.log_dict(metrics, step=trainer.state["global_step"])

        self._capture = False
        self._stats.clear()


class PrintExampleIDBeforeForwardPassCallback(
    FoundryPrintExampleIDBeforeForwardPassCallback
):
    """Print the example IDs for each rank at the start of the forward pass for each batch.

    WARNING: Spams the console. Use only for debugging purposes.
    """

    def on_train_batch_start(self, trainer, batch: Any, batch_idx: int):
        """Prints the example IDs for this batch."""
        example_ids = [example["example_id"] for example in batch]

        # Prepare the formatted strings with colors
        rank_info = f"[grey]<Rank {trainer.fabric.global_rank}>[/grey]"
        epoch_batch_info = (
            f"[blue]Epoch {trainer.state['current_epoch']} Batch {batch_idx}[/blue]"
        )
        example_ids_info = (
            f"[bold yellow]Example IDs: {', '.join(example_ids)}[/bold yellow]"
        )

        safe_print(
            f"{rank_info} {epoch_batch_info} - {example_ids_info}",
            logger=self.logger,
        )


class RankProgressTraceCallback(FoundryPrintExampleIDBeforeForwardPassCallback):
    """Write rank-local training-loop boundaries for DDP timeout diagnosis."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)

    def _write(
        self,
        trainer: Any,
        phase: str,
        batch_idx: int | None = None,
        detail: str = "",
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rank = trainer.fabric.global_rank
        with (self.output_dir / f"rank-{rank}.log").open("a", encoding="utf-8") as file:
            file.write(
                f"{datetime.now(timezone.utc).isoformat()}\t{phase}\t"
                f"epoch={trainer.state.get('current_epoch')}\tbatch={batch_idx}\t"
                f"global_step={trainer.state.get('global_step')}\t{detail}\n"
            )
            file.flush()

    def on_before_train_loader_next(self, trainer: Any, **kwargs: Any) -> None:
        self._write(trainer, "before_loader_next")

    def on_train_batch_start(self, trainer: Any, batch: Any, batch_idx: int) -> None:
        example_ids = ",".join(str(example["example_id"]) for example in batch)
        self._write(
            trainer,
            "train_batch_start",
            batch_idx,
            detail=f"example_ids={example_ids}",
        )

    def on_train_batch_end(
        self, trainer: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        self._write(trainer, "train_batch_end", batch_idx)

    def on_before_optimizer_step(self, trainer: Any, optimizer: Any) -> None:
        self._write(trainer, "before_optimizer_step")

    def optimizer_step(self, trainer: Any, optimizer: Any) -> None:
        self._write(trainer, "optimizer_step")


class LogDatasetSamplingRatiosCallback(FoundryLogDatasetSamplingRatiosCallback):
    """Dataset-ratio logger that accepts generated ASE LMDB example IDs and multiple example IDs generally."""

    def __init__(self, fallback_dataset_name: str = "unparsed"):
        self.fallback_dataset_name = fallback_dataset_name

    def on_train_batch_start(self, trainer, batch, batch_idx):
        """Logs the dataset sampling ratios for this batch."""
        if trainer.fabric.is_global_zero:
            for example in batch:
                try:
                    dataset_string = "/".join(
                        parse_example_id(example["example_id"])["datasets"]
                    )
                except ValueError:
                    dataset_string = self.fallback_dataset_name

                self.dataset_sampling_counts[dataset_string] += 1


class LogHbondConditioningCallback(BaseCallback):
    """Audit how often HBPLUS conditioning reaches training examples."""

    _UNAVAILABLE_REASONS = (
        "res_name_too_long",
        "missing_chain_iid",
        "too_many_chain_iids",
        "nonfinite_coordinates",
        "executable_unavailable",
        "timeout",
        "runtime_error",
    )

    def on_fit_start(self, trainer: Any) -> None:
        """Initialize per-epoch sufficient statistics."""
        self._counts: defaultdict[str, float] = defaultdict(float)

    def on_train_batch_start(self, trainer: Any, batch: Any, batch_idx: int) -> None:
        """Accumulate transform counters before the batch is collated."""
        del trainer, batch_idx
        for example in batch:
            self._counts["examples"] += 1
            log_dict = example.get("log_dict", {})
            if "hbplus_available" not in log_dict:
                continue

            self._counts["attempted"] += 1
            self._counts["available"] += float(log_dict["hbplus_available"])
            detected_atoms = float(log_dict.get("hbond_total_count", 0))
            provided_atoms = float(log_dict.get("hbond_subsample_atoms", 0))
            self._counts["detected_atoms"] += detected_atoms
            self._counts["provided_atoms"] += provided_atoms
            self._counts["nonzero"] += float(provided_atoms > 0)
            for reason in self._UNAVAILABLE_REASONS:
                self._counts[f"unavailable_{reason}"] += float(
                    log_dict.get(f"hbplus_unavailable_{reason}", 0)
                )

    def on_train_epoch_end(self, trainer: Any) -> None:
        """Reduce counters across ranks and log interpretable fractions."""
        names = (
            "examples",
            "attempted",
            "available",
            "nonzero",
            "detected_atoms",
            "provided_atoms",
            *(f"unavailable_{reason}" for reason in self._UNAVAILABLE_REASONS),
        )
        local_counts = trainer.fabric.to_device(
            torch.tensor([self._counts[name] for name in names], dtype=torch.float64)
        )
        reduced_counts = trainer.fabric.all_reduce(local_counts, reduce_op="sum")
        counts = dict(zip(names, reduced_counts.tolist(), strict=True))

        if trainer.fabric.is_global_zero:
            examples = max(counts["examples"], 1)
            attempted = max(counts["attempted"], 1)
            available = max(counts["available"], 1)
            metrics = {
                "train/hbond_conditioning/attempt_fraction": (
                    counts["attempted"] / examples
                ),
                "train/hbond_conditioning/availability_fraction": (
                    counts["available"] / attempted
                ),
                "train/hbond_conditioning/nonzero_fraction": (
                    counts["nonzero"] / available
                ),
                "train/hbond_conditioning/mean_detected_atoms": (
                    counts["detected_atoms"] / available
                ),
                "train/hbond_conditioning/mean_provided_atoms": (
                    counts["provided_atoms"] / available
                ),
            }
            metrics.update(
                {
                    f"train/hbond_conditioning/unavailable/{reason}": (
                        counts[f"unavailable_{reason}"] / attempted
                    )
                    for reason in self._UNAVAILABLE_REASONS
                }
            )
            trainer.fabric.log_dict(metrics, step=trainer.state["global_step"])

        self._counts.clear()


class LogModelDiagnosticsCallback(BaseCallback):
    """Aggregate and log model-provided diagnostics through Lightning Fabric."""

    _CHECKPOINT_STATE_KEY = (
        "nucleus.callbacks.train_logging.LogModelDiagnosticsCallback"
    )

    def __init__(
        self,
        log_every_n: int = 25,
        prefix: str = "train/activation_scale",
    ) -> None:
        if log_every_n < 1:
            raise ValueError(f"log_every_n must be positive, got {log_every_n}")
        self.log_every_n = log_every_n
        self.prefix = prefix.rstrip("/")
        self._epoch_diagnostics: ModelDiagnostics = {"rms": {}, "max": {}}
        self._restored_epoch_diagnostics = False
        self._epoch_aggregation_complete = True

    def on_fit_start(self, trainer: Any) -> None:
        """Restore rank-local diagnostics after a step-checkpoint resume."""
        batches_completed = int(trainer.state.get("batches_completed_in_epoch", 0))
        callback_states = trainer.state.get("callback_states", {})
        rank_states = callback_states.get(self._CHECKPOINT_STATE_KEY)
        if rank_states is None:
            self._epoch_aggregation_complete = batches_completed == 0
            return

        rank = int(getattr(trainer.fabric, "global_rank", 0))
        if rank >= len(rank_states):
            raise ValueError(
                "Checkpoint diagnostics state does not contain this distributed "
                f"rank: rank={rank}, saved_ranks={len(rank_states)}."
            )
        self._epoch_diagnostics = trainer.fabric.to_device(rank_states[rank])
        self._restored_epoch_diagnostics = True
        self._epoch_aggregation_complete = True

    def on_save_checkpoint(self, trainer: Any, state: dict[str, Any]) -> None:
        """Save rank-local diagnostic sufficient statistics."""
        cpu_diagnostics: ModelDiagnostics = {
            "rms": {
                name: {key: value.detach().cpu() for key, value in accumulator.items()}
                for name, accumulator in self._epoch_diagnostics["rms"].items()
            },
            "max": {
                name: value.detach().cpu()
                for name, value in self._epoch_diagnostics["max"].items()
            },
        }
        state.setdefault("callback_states", {})[self._CHECKPOINT_STATE_KEY] = (
            trainer._gather_rank_states(cpu_diagnostics)
        )

    def on_train_epoch_start(self, trainer: Any) -> None:
        """Reset local sufficient statistics for the next epoch."""
        if self._restored_epoch_diagnostics:
            self._restored_epoch_diagnostics = False
            return
        self._epoch_diagnostics = {"rms": {}, "max": {}}

    @staticmethod
    def _accumulate(destination: ModelDiagnostics, source: ModelDiagnostics) -> None:
        for name, accumulator in source.get("rms", {}).items():
            if name not in destination["rms"]:
                destination["rms"][name] = {
                    "sum_squares": accumulator["sum_squares"].detach().clone(),
                    "count": accumulator["count"].detach().clone(),
                }
                continue
            destination["rms"][name]["sum_squares"].add_(accumulator["sum_squares"])
            destination["rms"][name]["count"].add_(accumulator["count"])

        for name, value in source.get("max", {}).items():
            value = value.detach()
            if name not in destination["max"]:
                destination["max"][name] = value.clone()
                continue
            destination["max"][name] = torch.maximum(destination["max"][name], value)

    def _reduce_and_format(
        self,
        trainer: Any,
        diagnostics: ModelDiagnostics,
        scope: str,
    ) -> dict[str, torch.Tensor]:
        metrics: dict[str, torch.Tensor] = {}
        for name in sorted(diagnostics.get("rms", {})):
            accumulator = diagnostics["rms"][name]
            sum_squares = trainer.fabric.all_reduce(
                accumulator["sum_squares"].clone(), reduce_op="sum"
            )
            count = trainer.fabric.all_reduce(
                accumulator["count"].clone(), reduce_op="sum"
            )
            metrics[f"{self.prefix}/{scope}/{name}"] = (
                sum_squares / count.clamp_min(1)
            ).sqrt()

        for name in sorted(diagnostics.get("max", {})):
            metrics[f"{self.prefix}/{scope}/{name}"] = trainer.fabric.all_reduce(
                diagnostics["max"][name].clone(), reduce_op="max"
            )
        return metrics

    def on_train_batch_end(
        self, trainer: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Accumulate every batch and periodically log global batch statistics."""
        diagnostics = outputs.get("model_diagnostics", {})
        if not diagnostics:
            return

        self._accumulate(self._epoch_diagnostics, diagnostics)
        if batch_idx % self.log_every_n != 0:
            return

        metrics = self._reduce_and_format(trainer, diagnostics, scope="batch")
        if trainer.fabric.is_global_zero and metrics:
            trainer.fabric.log_dict(metrics, step=trainer.state["global_step"])

    def on_train_epoch_end(self, trainer: Any) -> None:
        """Log an exact RMS and maximum over this complete epoch."""
        metrics = self._reduce_and_format(
            trainer, self._epoch_diagnostics, scope="epoch"
        )
        if (
            self._epoch_aggregation_complete
            and trainer.fabric.is_global_zero
            and metrics
        ):
            trainer.fabric.log_dict(metrics, step=trainer.state["global_step"])
        self._epoch_diagnostics = {"rms": {}, "max": {}}
        trainer.state.get("callback_states", {}).pop(self._CHECKPOINT_STATE_KEY, None)
        self._epoch_aggregation_complete = True


class LogAF3TrainingLossesCallback(FoundryLogAF3TrainingLossesCallback):
    """Log the primary model losses for AF3 across multiple examples."""

    _CHECKPOINT_STATE_KEY = (
        "nucleus.callbacks.train_logging.LogAF3TrainingLossesCallback"
    )

    @staticmethod
    def _materialize_scalar_tensors(
        scalar_tensors: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Convert scalar tensors to floats with one transfer per source device."""
        if not scalar_tensors:
            return {}

        tensors_by_device: dict[torch.device, list[tuple[str, torch.Tensor]]] = {}
        for name, scalar_tensor in scalar_tensors.items():
            tensors_by_device.setdefault(scalar_tensor.device, []).append(
                (name, scalar_tensor)
            )

        materialized_by_name = {}
        for device_tensors in tensors_by_device.values():
            names, tensors = zip(*device_tensors, strict=True)
            values = torch.stack(tensors).cpu().tolist()
            materialized_by_name.update(zip(names, values, strict=True))

        return {name: materialized_by_name[name] for name in scalar_tensors}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._epoch_aggregation_complete = True
        self._elapsed_time_before_resume = 0.0

    def on_fit_start(self, trainer: Any) -> None:
        """Restore rank-local epoch loss accumulators after a step resume."""
        batches_completed = int(trainer.state.get("batches_completed_in_epoch", 0))
        callback_states = trainer.state.get("callback_states", {})
        rank_states = callback_states.get(self._CHECKPOINT_STATE_KEY)
        if rank_states is None:
            self._epoch_aggregation_complete = batches_completed == 0
            if not self._epoch_aggregation_complete:
                self.logger.warning(
                    "The loaded step checkpoint predates callback-state saving; "
                    "epoch loss means will be omitted for this partial epoch."
                )
            return

        rank = int(getattr(trainer.fabric, "global_rank", 0))
        if rank >= len(rank_states):
            raise ValueError(
                "Checkpoint callback state does not contain this distributed rank: "
                f"rank={rank}, saved_ranks={len(rank_states)}."
            )

        saved_state = rank_states[rank]
        self.loss_trackers = {}
        for name, tracker_state in saved_state["trackers"].items():
            tracker = trainer.fabric.to_device(MeanMetric())
            weight = tracker_state["weight"].to(tracker.device)
            if weight.item() > 0:
                mean = tracker_state["mean_value"].to(tracker.device) / weight
                tracker.update(mean, weight=weight)
                tracker._update_count = int(tracker_state["update_count"])
            self.loss_trackers[name] = tracker
        self._elapsed_time_before_resume = float(saved_state["elapsed_time"])
        self._epoch_aggregation_complete = True

    def on_save_checkpoint(self, trainer: Any, state: dict[str, Any]) -> None:
        """Save rank-local sufficient statistics for exact resumed epoch means."""
        local_tracker_state = {
            "trackers": {
                name: {
                    "mean_value": tracker.mean_value.detach().cpu(),
                    "weight": tracker.weight.detach().cpu(),
                    "update_count": int(tracker.update_count),
                }
                for name, tracker in self.loss_trackers.items()
            },
            "elapsed_time": self._elapsed_time_before_resume
            + (time.time() - self.start_time),
        }
        state.setdefault("callback_states", {})[self._CHECKPOINT_STATE_KEY] = (
            trainer._gather_rank_states(local_tracker_state)
        )

    def on_train_batch_end(self, trainer, outputs: Any, batch: Any, batch_idx: int):
        """Logs the training losses for this batch."""
        mean_loss_dict: dict[str, torch.Tensor] = {}
        if "loss_dict" in outputs:
            # Keep aggregation on-device.  rf3.utils.loss.mean_losses calls
            # ``.item()`` once per metric, forcing every DDP rank to synchronize
            # its CUDA stream after every batch.  Apart from being expensive,
            # that rank-local synchronization can overlap the token loader's next
            # distributed packing collective.  Only rank zero materializes values,
            # and only at the configured logging interval below.
            mean_loss_dict.update(
                {
                    key: batched_loss.detach().mean()
                    for key, batched_loss in outputs["loss_dict"].items()
                }
            )

        for key, val in mean_loss_dict.items():
            if key not in self.loss_trackers:
                self.loss_trackers[key] = trainer.fabric.to_device(MeanMetric())
            self.loss_trackers[key].update(val)

        if trainer.fabric.is_global_zero and batch_idx % self.log_every_n == 0:
            mean_loss_values = self._materialize_scalar_tensors(mean_loss_dict)

            # ... log losses for each structure in the batch
            if self.log_full_batch_losses:
                # Move each complete tensor to CPU before the legacy converter
                # iterates over its elements.  This avoids thousands of scalar
                # device synchronizations in per-structure logging.
                cpu_loss_dict = {
                    key: value.detach().cpu()
                    for key, value in outputs["loss_dict"].items()
                }
                full_batch_loss_dicts = convert_batched_losses_to_list_of_dicts(
                    cpu_loss_dict
                )
                for loss_dict in full_batch_loss_dicts:
                    loss_dict = {
                        f"train/per_structure/{k}": v for k, v in loss_dict.items()
                    }
                    trainer.fabric.log_dict(
                        loss_dict, step=trainer.state["global_step"]
                    )

            # ... log losses meaned across the batch
            # (Prepend "train/batch_mean" to the keys in the loss dictionary)
            mean_loss_dict_for_logging = {
                f"train/batch_mean/{k}": v for k, v in mean_loss_values.items()
            }
            trainer.fabric.log_dict(
                mean_loss_dict_for_logging, step=trainer.state["global_step"]
            )

            # ... print the mean losses in a table
            df_losses = pd.DataFrame(
                {
                    "Train Loss Name": [
                        k.replace("_", " ").title() for k in mean_loss_values.keys()
                    ],
                    "Value": [v for v in mean_loss_values.values()],
                }
            )
            table = table_from_df(df_losses, title="Training Losses")

            percentage_complete = _epoch_progress_percentage(trainer, batch)

            # Simple progress bar using Unicode blocks
            progress_bar_length = 10  # Length of the progress bar
            if percentage_complete is None:
                progress_bar = "?" * progress_bar_length
                percentage_str = "[yellow]unavailable after legacy resume[/yellow]"
            else:
                filled_length = int(progress_bar_length * percentage_complete // 100)
                progress_bar = "█" * filled_length + "░" * (
                    progress_bar_length - filled_length
                )
                percentage_str = (
                    f"[bold magenta]{percentage_complete:.2f}%[/bold magenta]"
                )

            # Create a panel for the epoch and batch info with a progress bar
            epoch_batch_info = (
                f"[grey]<Rank {trainer.fabric.global_rank}>[/grey] "
                f"Epoch {trainer.state['current_epoch']} Batch {batch_idx} "
                f"[{progress_bar}] {percentage_str}"
            )

            epoch_batch_panel = Panel(
                epoch_batch_info,
                border_style="bold blue",
            )

            # Create a panel for the example ID
            example_ids = [example["example_id"] for example in batch]
            example_ids_str = f"[bold yellow]{', '.join(example_ids)}[/bold yellow]"
            example_ids_panel = Panel(
                example_ids_str,
                border_style="bold green",
            )

            # Combine all components vertically
            combined_content = Group(epoch_batch_panel, example_ids_panel, table)

            safe_print(combined_content)

    def on_train_epoch_end(self, trainer: Any) -> None:
        """Log exact epoch means on the optimizer-step axis."""
        final_mean_tensors = {
            name: tracker.compute() for name, tracker in self.loss_trackers.items()
        }
        elapsed_time = self._elapsed_time_before_resume + (
            time.time() - self.start_time
        )
        num_batches = (
            self.loss_trackers["total_loss"].update_count
            if "total_loss" in self.loss_trackers
            else trainer.state.get("batches_completed_in_epoch", 0)
        )

        if trainer.fabric.is_global_zero:
            final_means = self._materialize_scalar_tensors(final_mean_tensors)
            table = Table(
                title=f"Epoch {trainer.state['current_epoch']} Summary",
                show_header=False,
                header_style="bold magenta",
            )
            table.add_column("Loss Name", style="bold cyan", justify="left")
            table.add_column("Value", style="green", justify="right")
            for name, value in final_means.items():
                table.add_row(f"<Train> Mean {name}", f"{value:.4f}")
            table.add_section()
            table.add_row("Total Optimizer Steps", str(trainer.state["global_step"]))
            table.add_row("Number of Batches", str(num_batches))
            table.add_row("Elapsed Time (s)", f"{elapsed_time:.2f}")
            if num_batches:
                table.add_row(
                    "Mean Time per Batch (s)", f"{elapsed_time / num_batches:.2f}"
                )
            safe_print(table)

            if self._epoch_aggregation_complete:
                trainer.fabric.log_dict(
                    {
                        f"train/per_epoch_{name}": value
                        for name, value in final_means.items()
                    },
                    step=trainer.state["global_step"],
                )

        for tracker in self.loss_trackers.values():
            tracker.reset()
        trainer.state.get("callback_states", {}).pop(self._CHECKPOINT_STATE_KEY, None)
        self._epoch_aggregation_complete = True
        self._elapsed_time_before_resume = 0.0
