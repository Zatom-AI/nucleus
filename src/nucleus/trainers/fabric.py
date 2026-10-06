"""Generic training harness built atop PyTorch Lightning Fabric, adapted from Foundry.

In addition to standard harness features (gradient accumulation, mixed precision, etc.), includes native support for EMA and model checkpointing.

References:
    - Pytorch Lightning Trainer Example (https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/fabric/build_your_own_trainer/trainer.py)
    - Lightning Hydra Template (https://github.com/ashleve/lightning-hydra-template)
"""

import math
import random
from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import cast

import hydra
import lightning as L
import numpy as np
import torch
from beartype.typing import Any, Literal, Mapping
from foundry.callbacks.callback import BaseCallback
from foundry.training.EMA import EMA
from foundry.training.schedulers import SchedulerConfig
from foundry.utils.ddp import RankedLogger
from foundry.utils.weights import (
    CheckpointConfig,
    WeightLoadingConfig,
    freeze_parameters_with_config,
    load_weights_with_policies,
)
from foundry.utils.xpu import (
    SingleXPUStrategy,
    XPUAccelerator,
    XPUMixedPrecision,
)
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import DDPStrategy, Strategy
from lightning.fabric.utilities.apply_func import move_data_to_device
from lightning.fabric.utilities.data import _set_sampler_epoch
from lightning.fabric.wrappers import (
    _FabricDataLoader,
    _FabricModule,
    _FabricOptimizer,
)
from torch.utils.data.distributed import DistributedSampler

from nucleus.trainers.checkpointing import (
    CheckpointKind,
    checkpoint_train_config,
    get_checkpoint_kind,
    get_checkpoint_path,
    prune_checkpoints,
)
from nucleus.trainers.checkpointing import (
    get_latest_checkpoint as find_latest_checkpoint,
)

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def _capture_rng_state() -> dict[str, Any]:
    """
    Capture process-local Python, NumPy, CPU, and accelerator RNG state.

    Returns:
        A dictionary containing the RNG state for Python, NumPy, CPU, and
        available accelerators (CUDA, MPS, XPU). The keys are "python",
        "numpy", "torch", and the names of available accelerators. The
        values are the corresponding RNG states.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    for backend_name in ("cuda", "mps", "xpu"):
        backend = getattr(torch, backend_name, None)
        if backend is None or not backend.is_available():
            continue
        get_rng_state_all = getattr(backend, "get_rng_state_all", None)
        state[backend_name] = (
            get_rng_state_all()
            if get_rng_state_all is not None
            else backend.get_rng_state()
        )
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    """
    Restore process-local RNG state captured by :func:`_capture_rng_state`.

    Args:
        state: A dictionary containing the RNG state for Python, NumPy, CPU, and
            available accelerators (CUDA, MPS, XPU). The keys are "python",
            "numpy", "torch", and the names of available accelerators. The
            values are the corresponding RNG states.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    for backend_name in ("cuda", "mps", "xpu"):
        if backend_name not in state:
            continue
        backend = getattr(torch, backend_name, None)
        if backend is None or not backend.is_available():
            ranked_logger.warning(
                f"Checkpoint contains {backend_name.upper()} RNG state, but that "
                "backend is unavailable; accelerator RNG state was not restored."
            )
            continue
        set_rng_state_all = getattr(backend, "set_rng_state_all", None)
        if set_rng_state_all is not None:
            set_rng_state_all(state[backend_name])
        else:
            backend.set_rng_state(state[backend_name])


def is_interactive_environment() -> bool:
    """
    Utility function to check if we're in an interactive environment (e.g., Jupyter notebook or IPython).

    Returns:
        True if in an interactive environment, False otherwise.
    """
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


class FabricTrainer(ABC):
    """Base Trainer class built around Lightning Fabric.

    Args:
        accelerator: The hardware to run on. See (1) for details. Possible choices are:
            ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
        strategy: Strategy for how to run across multiple devices. See (1) for details. Possible choices are:
            ``"dp"``, ``"ddp"``, ``"ddp_spawn"``, ``"deepspeed"``, ``"fsdp"``.
        devices_per_node: Number of devices to train on per machine (``int``), which GPUs to train on (``list`` or ``str``), or ``"auto"``.
            See (1) for details.
            EXAMPLE: If you run on 2 nodes with 8 GPUs each, you would set ``devices_per_node=8``, not ``16``.
        num_nodes: Number of machines (nodes) for distributed training (default: 1). See (1) for details.
        precision: Double precision (``"64"``), full precision (``"32"``), half precision AMP (``"16-mixed"``),
            or bfloat16 precision AMP (``"bf16-mixed"``). See (2) for details.
        callbacks: A single callback or a list of callbacks, each inheriting the BaseCallback Abstract Base Class.
        loggers:  A single logger or a list of loggers. See (3) for details.
        max_epochs: Maximum number of epochs to train for (default: 1000).
        grad_accum_steps: Number of batches whose gradients are averaged before
            calling ``optimizer.step()`` (default: 1). See (4) for details on
            gradient accumulation in Fabric.
        validate_every_n_epochs: Number of epochs between validation runs (default: 1).
        n_examples_per_epoch: Number of examples to sample per epoch, across all GPUs. E.g., number of distinct examples that will
            be "seen" by the model in a single epoch. If smaller than the the number implied by the dataloader, we will
            alert a warning and use the smaller number.
        output_dir: Directory to save checkpoints, metrics, intermediate validation strructures, etc. (default: None).
        checkpoint_every_n_epochs: Number of epochs between saving checkpoints (default: 1).
            If None, epoch-based checkpointing is disabled.
        checkpoint_every_n_steps: Number of optimizer steps between saving checkpoints (default: None).
            If set, checkpoints will be saved every N optimizer steps.
        clip_grad_max_norm: Maximum gradient norm to clip to (default: None). If None, no gradient clipping is performed.
        skip_nan_grad: Whether to skip optimizer updates when gradients contain NaN or Inf values (default: False).
            Useful for training stability, especially with mixed precision or challenging datasets.
        error_if_grad_nonfinite: Whether to raise when gradient clipping detects NaN or Inf gradients (default: False).
        limit_train_batches: Limit on the number of training batches per epoch (default: float("inf")).
            Helpful for debugging; should NOT be used when training production models.
        limit_val_batches: Limit on the number of validation batches per epoch (default: float("inf")).
            Helpful for debugging; should NOT be used when training production models.
        prevalidate: Whether to run validation before training starts (default: False).
        nccl_timeout: Timeout for NCCL operations (default: 1800). Only used with DDP strategy.
        find_unused_parameters: Whether to let DDP find and skip gradient synchronization for unused parameters in the model (default: False). NOTE: Setting to True will incur a performance penalty,
            but allow for training for bespoke use cases where parts of the model are frozen.
        skip_optimizer_loading: Whether to skip loading the optimizer/scheduler state when restoring from checkpoints (default: False).
        allow_world_size_change_on_resume: Whether to allow an epoch checkpoint
            to resume with a different physical world size (default: False). When
            enabled, incompatible rank-local RNG state is discarded. Step
            checkpoints remain restricted to their original world size.
        validate_first_epoch: Whether to run validation on the first epoch (default: True).

    References:
        (1) Fabric Arguments (https://lightning.ai/docs/fabric/stable/api/fabric_args.html)
        (2) Fabric Precision Documentation (https://lightning.ai/docs/fabric/stable/fundamentals/precision.html)
        (3) Fabric Loggers (https://lightning.ai/docs/fabric/2.4.0/api/loggers.html)
        (4) Efficient Gradient Accumulation (https://lightning.ai/docs/fabric/2.4.0/advanced/gradient_accumulation.html)
    """

    def __init__(
        self,
        *,
        accelerator: str | Accelerator = "auto",
        strategy: str | Strategy = "ddp",
        devices_per_node: list[int] | int | str = "auto",
        num_nodes: int = 1,
        precision: str | int = "bf16-mixed",
        callbacks: BaseCallback | list[BaseCallback] | None = None,
        loggers: Logger | list[Logger] | None = None,
        max_epochs: int = 1000,
        grad_accum_steps: int = 1,
        validate_every_n_epochs: int = 1,
        validate_loader_every_n_epochs: Mapping[str, int | None] | None = None,
        n_examples_per_epoch: int = 24_000,
        output_dir: Path | str | None = None,
        checkpoint_every_n_epochs: int | None = 1,
        checkpoint_every_n_steps: int | None = None,
        checkpoint_before_validation: bool = False,
        checkpoint_keep_last_n: int | None = None,
        checkpoint_retention_root: Path | str | None = None,
        clip_grad_max_norm: float | None = None,
        skip_nan_grad: bool = False,
        error_if_grad_nonfinite: bool = False,
        limit_train_batches: int | float = float("inf"),
        limit_val_batches: int | float = float("inf"),
        prevalidate: bool = False,
        nccl_timeout: int = 1_800,
        find_unused_parameters: bool = False,
        skip_optimizer_loading: bool = False,
        allow_world_size_change_on_resume: bool = False,
        validate_first_epoch: bool = True,
    ) -> None:
        # Handle XPU accelerator
        is_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
        if accelerator == "xpu" or (
            accelerator == "auto" and is_xpu and not torch.cuda.is_available()
        ):
            accelerator = XPUAccelerator()
            precision_plugin = None
            if precision in ("16-mixed", "bf16-mixed"):
                precision_plugin = XPUMixedPrecision(precision=precision)
                precision = None  # Handled by plugin
            strategy = SingleXPUStrategy(precision_plugin=precision_plugin)
            ranked_logger.info("Using Intel XPU with SingleXPUStrategy")
        elif (
            strategy == "ddp"
            and not is_interactive_environment()
            and not (num_nodes == 1 and devices_per_node == 1)
        ):
            strategy = DDPStrategy(
                timeout=timedelta(seconds=nccl_timeout),
                find_unused_parameters=find_unused_parameters,
            )
        else:
            strategy = "auto"  # type: ignore

        # See (1) for initialization arguments for Fabric()
        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices_per_node,
            num_nodes=num_nodes,
            precision=precision,
            callbacks=callbacks,
            loggers=loggers,
        )

        # Training
        self.clip_grad_max_norm = clip_grad_max_norm
        self.skip_nan_grad = skip_nan_grad
        self.error_if_grad_nonfinite = error_if_grad_nonfinite
        if (
            not isinstance(grad_accum_steps, int)
            or isinstance(grad_accum_steps, bool)
            or grad_accum_steps <= 0
        ):
            raise ValueError("grad_accum_steps must be a positive integer.")
        self.grad_accum_steps = grad_accum_steps

        # Stopping
        self.max_epochs = max_epochs
        self.should_stop = False
        self.n_examples_per_epoch = n_examples_per_epoch
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches

        # Validation
        self.validate_every_n_epochs = validate_every_n_epochs
        self.validate_loader_every_n_epochs = {
            name: interval
            for name, interval in (validate_loader_every_n_epochs or {}).items()
            if interval is not None
        }
        invalid_validation_intervals = {
            name: interval
            for name, interval in self.validate_loader_every_n_epochs.items()
            if interval <= 0
        }
        if invalid_validation_intervals:
            raise ValueError(
                "Per-loader validation intervals must be positive: "
                f"{invalid_validation_intervals}."
            )
        self.validate_first_epoch = validate_first_epoch
        self.prevalidate = prevalidate

        # Checkpoints
        if checkpoint_every_n_epochs is not None and checkpoint_every_n_epochs <= 0:
            raise ValueError("checkpoint_every_n_epochs must be positive or None.")
        if checkpoint_every_n_steps is not None and checkpoint_every_n_steps <= 0:
            raise ValueError("checkpoint_every_n_steps must be positive or None.")
        if checkpoint_keep_last_n is not None and checkpoint_keep_last_n <= 0:
            raise ValueError("checkpoint_keep_last_n must be positive or None.")

        self.output_dir = Path(output_dir) if output_dir else None
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self.checkpoint_before_validation = checkpoint_before_validation
        self.checkpoint_keep_last_n = checkpoint_keep_last_n
        self.checkpoint_retention_root = (
            Path(checkpoint_retention_root)
            if checkpoint_retention_root is not None
            else self.output_dir
        )
        self.skip_optimizer_loading = skip_optimizer_loading
        self.allow_world_size_change_on_resume = allow_world_size_change_on_resume

    def initialize_or_update_trainer_state(
        self,
        updates: dict,
    ):
        """Initialize or update the state dictionary for the trainer.

        Args:
            updates: A dictionary containing the following keys and values to update the trainer state:
                model: The model to train.
                optimizer: The optimizer to use with the model. May be None for validation/inference.
                scheduler_cfg: Learning rate SchedulerConfig (e.g., a LRScheduler with intervals/frequency). May be None for validation/inference or if no scheduler is used.
                global_step: Global optimizer step; used by W&B logger, learning rate schedulers, etc. Default is 0.
                current_epoch: Global epoch counter; used for validation, learning rate schedulers, checkpointing, etc. Default is 0.
                batches_completed_in_epoch: Number of training batches whose optimizer
                    work is complete in the current epoch. Default is 0.
                examples_completed_in_epoch: Number of rank-local training examples
                    completed in the current epoch. Default is 0.
                epoch_progress_is_exact: Whether the example progress counter was
                    restored exactly. Default is True.
                callback_states: Checkpointable callback-owned state. Default is an
                    empty dictionary.
                train_cfg: The training configuration dictionary. Used for reinitializing the trainer with the same configuration
                    (for training or for inference). Default is an empty dictionary.
        """
        # Default values for the state
        default_state = {
            "model": None,
            "optimizer": None,
            "scheduler_cfg": None,
            "global_step": 0,
            "current_epoch": 0,
            "batches_completed_in_epoch": 0,
            "examples_completed_in_epoch": 0,
            "epoch_progress_is_exact": True,
            "callback_states": {},
            "train_cfg": {},
        }

        # Initialize self.state with default values if it doesn't exist
        if not hasattr(self, "state"):
            self.state = default_state.copy()
        else:
            # Ensure existing state has all default keys
            for key, value in default_state.items():
                self.state.setdefault(key, value)

        # Merge the updates into the existing state
        self.state.update(updates)

    def construct_optimizer(self) -> None:
        """Instantiate the optimizer(s)

        We provide a default implementation that instantiates the optimizer(s) from the Hydra configuration.
        More complex models (e.g., GANs) may require custom implementations.
        """
        assert (
            "model" in self.state and hasattr(self.state["model"], "parameters")
        ), "Model not found in state dictionary! You must call `construct_model()` before constructing the optimizer."

        if self.state["train_cfg"].model.optimizer:
            # ... instantiate the optimizer
            optimizer = hydra.utils.instantiate(
                self.state["train_cfg"].model.optimizer,
                params=self.state["model"].parameters(),
            )
            self.initialize_or_update_trainer_state({"optimizer": optimizer})

    def construct_scheduler(self) -> None:
        """Instantiate the configured learning rate scheduler.

        ``model.lr_scheduler_interval`` selects optimizer-step or epoch updates
        and defaults to ``"step"`` for existing configurations.
        """
        assert (
            "optimizer" in self.state and self.state["optimizer"]
        ), "Optimizer not found in state dictionary! You must call `construct_optimizer()` before constructing the scheduler."

        model_config = self.state["train_cfg"].model
        interval = model_config.get("lr_scheduler_interval", "step")
        if interval not in ("step", "epoch"):
            raise ValueError(
                "model.lr_scheduler_interval must be 'step' or 'epoch', "
                f"got {interval!r}."
            )

        # ...  instantiate the LR scheduler(s)
        lr_scheduler = (
            hydra.utils.instantiate(
                model_config.lr_scheduler,
                optimizer=self.state["optimizer"],
            )
            if model_config.lr_scheduler
            else None
        )

        if lr_scheduler:
            scheduler_cfg = SchedulerConfig(
                scheduler=lr_scheduler,
                interval=interval,
                frequency=1,
            )
            self.initialize_or_update_trainer_state({"scheduler_cfg": scheduler_cfg})

    def construct_model(self):
        """Instantiate the model, updating the trainer state in-place.

        This method must set the "model" key in the state dictionary using `self.initialize_or_update_trainer_state()`.
        For an example, see the `construct_model` method in the `AF3Trainer`
        Construct the model and optionally wrap with EMA.
        """
        # ... instantiate model with Hydra and Fabric
        with self.fabric.init_module():
            ranked_logger.info("Instantiating model...")

            model = hydra.utils.instantiate(
                self.state["train_cfg"].model.net,
                _recursive_=False,
            )

            # Optional whole-model compilation. Keep it before EMA wrapping so the
            # compiled object remains the model core.
            if self.state["train_cfg"].model.get("compile_model", False):
                compile_fn = getattr(torch, "compile", None)
                if compile_fn is None:
                    ranked_logger.warning(
                        "model.compile_model=True, but torch.compile is unavailable; continuing without compilation."
                    )
                else:
                    ranked_logger.info("Compiling model with torch.compile()...")
                    try:
                        model = compile_fn(model)
                    except Exception as exc:
                        ranked_logger.warning(
                            f"torch.compile failed; continuing without compilation ({exc!r})."
                        )

            # Optionally, wrap the model with EMA
            if self.state["train_cfg"].model.ema is not None:
                ranked_logger.info("Wrapping model with EMA...")
                model = EMA(model, **self.state["train_cfg"].model.ema)

        self.initialize_or_update_trainer_state({"model": model})

    def setup_model_optimizers_and_schedulers(self) -> None:
        """Setup the model, optimizer(s), and scheduler(s) with Fabric.

        Note that we must call this method after constructing (instantiating) the model, optimizer(s), and scheduler(s).
        For details on multi-model and multi-optimizer setups, see: https://lightning.ai/docs/fabric/2.2.3/advanced/multiple_setup.html
        """
        assert self.state[
            "model"
        ], "You must construct the model before setting up the model, optimizer, and scheduler."
        model = self.state["model"]
        optimizer = self.state["optimizer"]

        # ... setup the model and optimizer
        if optimizer:
            model, optimizer = self.fabric.setup(model, optimizer)
        else:
            model = self.fabric.setup(model)

        # ... update the state dictionary (we avoid updating the state dictionary in-place, which is an anti-pattern)
        self.initialize_or_update_trainer_state(
            {
                "model": model,
                "optimizer": optimizer,
            }
        )

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loaders: dict[str, torch.utils.data.DataLoader] | None = None,
        ckpt_config: CheckpointConfig | None = None,
    ) -> None:
        """Main entry point for training a model.

        Args:
            train_loader: Dataloader for training. Must have an iterable returning batches.
            val_loaders: Dictionary of dataloaders for validation. The keys are the names of the loaders, and the values are the loaders themselves.
            ckpt_config: Configuration for loading a checkpoint. May contain:
                - ckpt_path: Path to either:
                    (a) A previous checkpoint directory from which to resume training from. In this case, we will automatically load
                        the latest checkpoint using `self.get_latest_checkpoint()`.
                    (b) A specific checkpoint file to load. In this case, we will load the checkpoint from the specified file.
                    If None, no checkpoint is loaded, and the model will be trained from scratch.
                - weight_loading_config: Weight loading policies to apply to the checkpoint weights. If None, default policies are used (copy weights with re-initialization as a fallback
                    if shapes do not match)
                - reset_optimizer: Whether to reset the optimizer state when loading a checkpoint. If True, the optimizer will not be loaded from the checkpoint.
        """
        assert (
            hasattr(self, "state") and "model" in self.state
        ), "Model not found in state dictionary! You must call `instantiate_model()` before running fit()."

        # (If we don't have enough examples to sample, we will log a warning and use the smaller number)
        if len(train_loader) * self.fabric.world_size < self.n_examples_per_epoch:
            ranked_logger.warning(
                f"Number of examples per epoch ({self.n_examples_per_epoch}) exceeds the number of examples in the loader: "
                f"({len(train_loader) * self.fabric.world_size}). Using the latter."
            )
            self.n_examples_per_epoch = len(train_loader) * self.fabric.world_size
        self.n_examples_per_rank = math.ceil(
            self.n_examples_per_epoch / self.fabric.world_size
        )
        # Retained for callbacks that expect this attribute. For dynamically
        # packed loaders this is an example-count upper bound, not a batch count.
        self.n_batches_per_epoch = self.n_examples_per_rank

        is_token_budgeted_loader = hasattr(train_loader, "max_num_tokens")
        effective_batches_per_epoch = min(len(train_loader), self.limit_train_batches)
        expected_optimizer_steps = (
            None
            if is_token_budgeted_loader
            else math.ceil(effective_batches_per_epoch / self.grad_accum_steps)
        )
        if expected_optimizer_steps == 0:
            raise ValueError(
                "Training would run zero optimizer steps per epoch: "
                f"loader_batches={len(train_loader)}, "
                f"grad_accum_steps={self.grad_accum_steps}. "
                "Reduce trainer.grad_accum_steps, increase trainer.n_examples_per_epoch, "
                "or use more training examples."
            )

        # ... setup training and validation dataloaders with Fabric
        train_loader = self.fabric.setup_dataloaders(
            # Our sampler is already distributed, so we don't need to wrap with a DistributedSampler
            train_loader,
            use_distributed_sampler=False,
            # Keep the list of per-example dictionaries on CPU. Zatom's
            # SparseCollate packs it into a compact batch before one H2D copy.
            move_to_device=not getattr(
                self, "collate_train_batch_before_device_transfer", False
            ),
        )
        self._train_loader = train_loader
        raw_train_loader = self._unwrap_dataloader(train_loader)
        if (
            self.checkpoint_every_n_steps is not None
            and getattr(raw_train_loader, "num_workers", 0) > 0
            and getattr(raw_train_loader, "in_order", True) is False
        ):
            raise ValueError(
                "Mid-epoch checkpointing requires dataloader in_order=True when "
                "num_workers > 0; torchdata does not guarantee state restoration "
                "for out-of-order multiprocessing."
            )
        if (
            hasattr(raw_train_loader, "state_dict")
            and getattr(raw_train_loader, "generator", None) is None
        ):
            # Keep DataLoader iterator/base-seed draws off the model's process RNG.
            # The generator state is saved alongside the loader cursor below.
            loader_generator = torch.Generator()
            loader_generator.manual_seed(
                torch.initial_seed() + int(getattr(self.fabric, "global_rank", 0))
            )
            raw_train_loader.generator = loader_generator
            self._set_missing_sampler_generators(
                getattr(raw_train_loader, "sampler", None), loader_generator
            )

        if val_loaders is not None:
            for key, loader in val_loaders.items():
                val_loaders[key] = self.fabric.setup_dataloaders(
                    loader,
                    use_distributed_sampler=False,
                    move_to_device=not getattr(
                        self,
                        "collate_validation_batch_before_device_transfer",
                        False,
                    ),
                )

        self.setup_model_optimizers_and_schedulers()

        if ckpt_config is not None:
            assert hasattr(
                ckpt_config, "path"
            ), "Checkpoint path not found in checkpoint configuration!"
            ckpt_path = Path(ckpt_config.path)
            loaded_checkpoint_path = ckpt_path

            reset_optimizer = bool(
                getattr(ckpt_config, "reset_optimizer", False)
                or self.skip_optimizer_loading
            )

            if ckpt_path.is_dir():
                # If given a directory, load the latest checkpoint from the directory
                ranked_logger.info(
                    f"Loading latest checkpoint within the directory {ckpt_path}..."
                )
                latest_checkpoint = self.get_latest_checkpoint(ckpt_path)
                if latest_checkpoint is None:
                    raise FileNotFoundError(
                        f"No checkpoint files found under {ckpt_path}."
                    )
                loaded_checkpoint_path = latest_checkpoint
                self.load_checkpoint(
                    latest_checkpoint,
                    weight_loading_config=ckpt_config.weight_loading_config,
                    reset_optimizer=reset_optimizer,
                )
            else:
                # If given a specific checkpoint file, load that checkpoint
                loaded_checkpoint_path = ckpt_path
                self.load_checkpoint(
                    ckpt_path,
                    weight_loading_config=ckpt_config.weight_loading_config,
                    reset_optimizer=reset_optimizer,
                )

            # Apply parameter freezing if a freezing config is provided
            if getattr(ckpt_config, "parameter_freezing_config", None) is not None:
                ranked_logger.info(
                    "Applying parameter freezing according to CheckpointConfig..."
                )
                freeze_parameters_with_config(
                    # We must access the model through "module", since the model may be wrapped
                    self.state["model"].module,
                    ckpt_config.parameter_freezing_config,
                )

            # Epoch checkpoints are saved after the epoch completes. Step
            # checkpoints are saved inside the train loop, so keep the epoch.
            checkpoint_kind = getattr(
                self, "_loaded_checkpoint_kind", None
            ) or get_checkpoint_kind(loaded_checkpoint_path)
            self._handle_resume_world_size_change(
                checkpoint_kind=checkpoint_kind,
                checkpoint_path=loaded_checkpoint_path,
            )
            if checkpoint_kind == "epoch":
                self.state["current_epoch"] += 1
                self.state["batches_completed_in_epoch"] = 0
                self.state["examples_completed_in_epoch"] = 0
                self.state["epoch_progress_is_exact"] = True
            elif checkpoint_kind == "step":
                self._restore_train_dataloader_state(
                    train_loader,
                    checkpoint_path=loaded_checkpoint_path,
                )
            # Stopping conditions
            if (
                self.max_epochs is not None
                and self.state["current_epoch"] >= self.max_epochs
            ):
                self.should_stop = True
        else:
            ranked_logger.info("No checkpoint provided; training from scratch.")

        # Set the _num_iter_calls internal attribute of the wrapped loader to the current epoch
        # (NOTE: This addresses a bug in Lightning Fabric, where there the iter() method calls the `_set_sampler_epoch()` method,
        # relying on the _num_iter_calls attribute to determine the current epoch)
        train_loader._num_iter_calls = self.state["current_epoch"]

        self.fabric.call("on_fit_start", trainer=self)

        # Prevalidate
        if self.prevalidate and val_loaders:
            # Temporarily decrement the current epoch, since we haven't done any training this epoch
            self.state["current_epoch"] -= 1  # (Will be -1 if training from scratch)
            ranked_logger.info(
                f"Prevalidating with epoch {self.state['current_epoch']} before training; to avoid this behavior, set `prevalidate=False` in the Trainer config."
            )
            self.validation_loop(
                val_loaders=val_loaders,
                limit_batches=self.limit_val_batches,
            )
            self.state["current_epoch"] += 1  # (Restore the current epoch)

        while not self.should_stop:
            # ... train for one epoch
            optimizer_step_description = (
                "dynamic (token-budget packed)"
                if expected_optimizer_steps is None
                else str(expected_optimizer_steps)
            )
            ranked_logger.info(
                f"\n+ Starting epoch {self.state['current_epoch']}/{self.max_epochs - 1}\n"
                f"+ Total examples per epoch (across all GPU): {self.n_examples_per_epoch}\n"
                f"+ Examples per rank: {self.n_examples_per_rank}\n"
                f"+ Gradient accumulation steps: {self.grad_accum_steps}\n"
                "+ Expected optimizer steps per epoch: "
                f"{optimizer_step_description}\n"
            )

            epoch_completed = self.train_loop(
                train_loader=train_loader,
                limit_batches=self.limit_train_batches,
            )
            if not epoch_completed:
                ranked_logger.info(
                    f"Stopped partway through epoch {self.state['current_epoch']}; "
                    "skipping epoch-end bookkeeping."
                )
                break

            ranked_logger.info(f"Finished epoch {self.state['current_epoch']}!")

            self._log_epoch_progress()

            # Epoch checkpoints resume at the next epoch, so persist the learning
            # rate for that epoch even if validation is interrupted.
            self.step_scheduler(
                level="epoch", current_value=self.state["current_epoch"]
            )

            scheduled_val_loaders = self._scheduled_validation_loaders(val_loaders)

            # Persist completed training before a potentially long validation.
            # An epoch checkpoint advances to the next epoch when resumed, so an
            # allocation ending during validation cannot replay training.
            saved_pre_validation_checkpoint = False
            if self.checkpoint_before_validation and scheduled_val_loaders:
                self.save_checkpoint("epoch")
                saved_pre_validation_checkpoint = True

            # ... validate loaders that are due this epoch
            if scheduled_val_loaders:
                ranked_logger.info(
                    f"Starting validation for epoch {self.state['current_epoch']} "
                    f"on {sorted(scheduled_val_loaders)}!"
                )
                self.validation_loop(
                    val_loaders=scheduled_val_loaders,
                    limit_batches=self.limit_val_batches,
                )

            # ... save checkpoint, if we've reached the checkpoint interval
            if (
                self.checkpoint_every_n_epochs is not None
                and self.state["current_epoch"] % self.checkpoint_every_n_epochs == 0
                and not saved_pre_validation_checkpoint
            ):
                self.save_checkpoint("epoch")

            # ... increment the epoch
            self.state["current_epoch"] += 1

            # Stopping conditions
            if (
                self.max_epochs is not None
                and self.state["current_epoch"] >= self.max_epochs
            ):
                self.should_stop = True

        # Reset for next `fit()` call
        self.should_stop = False

        self.fabric.call("on_fit_end", trainer=self)

    def train_loop(
        self,
        *,
        train_loader: _FabricDataLoader,
        limit_batches: int | float = float("inf"),
    ) -> bool:
        """Train model for a single epoch.

        Args:
            train_loader: Dataloader for training.
            limit_batches: Limit on the batches during this training epoch. If greater than the number of batches in the
                `train_loader`, this argument has no effect. Helpful for debugging; should NOT be used when training production models.

        Returns:
            Whether the dataloader epoch (or configured batch limit) completed.
        """
        assert self.state["model"].training

        loaded_checkpoint_kind = getattr(self, "_loaded_checkpoint_kind", None)
        if getattr(self, "_loaded_rng_state", None) is not None:
            self._restore_checkpoint_loader_generator_state()
            if loaded_checkpoint_kind == "epoch":
                # This is the start of the *next* epoch. Restore before its
                # callback so all legitimate callback/iterator RNG consumption
                # advances the checkpoint stream exactly once.
                self._restore_checkpoint_rng_state()
                self._loaded_rng_state = None

        # Step resumes intentionally start a new callback segment for the
        # unfinished epoch. Model-side RNG is restored below so this setup does
        # not perturb the training stream.
        self.fabric.call("on_train_epoch_start", trainer=self)

        # NOTE: When we call iter(), Fabric calls the `set_sampler_epoch()` method on the sampler behind the scenes, so we don't need to call it explicitly
        train_iter = self._create_train_iterator(train_loader)
        self.fabric.call("on_after_train_loader_iter", trainer=self)
        if (
            loaded_checkpoint_kind == "step"
            and getattr(self, "_loaded_dataloader_generator_states", None) is not None
        ):
            generators = self._train_dataloader_generators()
            generator_states = self._loaded_dataloader_generator_states
            if len(generators) != len(generator_states):
                raise ValueError(
                    "Train dataloader generator structure changed during iterator "
                    "reconstruction."
                )
            for generator, generator_state in zip(
                generators, generator_states, strict=True
            ):
                generator.set_state(generator_state)
            self._loaded_dataloader_generator_states = None
        elif loaded_checkpoint_kind == "epoch":
            # Iterator construction above must advance the restored generator.
            self._loaded_dataloader_generator_states = None
        if (
            loaded_checkpoint_kind != "epoch"
            and getattr(self, "_loaded_rng_state", None) is not None
        ):
            # Iterator construction consumes main-process torch RNG even when its
            # base seed is then replaced by restored loader state. Restore here,
            # after resume-only hooks and before fetching/training the next batch.
            self._restore_checkpoint_rng_state()
            self._loaded_rng_state = None

        is_token_budgeted_loader = hasattr(train_loader, "max_num_tokens")
        batch_idx = int(self.state.get("batches_completed_in_epoch", 0))
        self.state["batches_completed_in_epoch"] = batch_idx
        accumulated_num_examples = 0
        last_processed_batch_completed_epoch = False
        epoch_completed = False
        while True:
            if self.should_stop:
                epoch_completed = last_processed_batch_completed_epoch
                break
            if batch_idx >= limit_batches:
                epoch_completed = True
                break

            self.fabric.call("on_before_train_loader_next", trainer=self)
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch_completed = True
                break

            self.fabric.call(
                "on_train_batch_start", trainer=self, batch=batch, batch_idx=batch_idx
            )

            # Step on the accumulation boundary and on the final processed batch.
            # Token-budgeted loaders mark their true final batch because their
            # ``len`` is necessarily a conservative upper bound.
            has_token_budget_marker = (
                is_token_budgeted_loader
                and isinstance(batch, list)
                and bool(batch)
                and "_token_budget_is_last_batch" in batch[0]
            )
            is_last_token_budget_batch = has_token_budget_marker and bool(
                batch[0].get("_token_budget_is_last_batch", False)
            )
            is_last_batch = (
                is_last_token_budget_batch
                or (not has_token_budget_marker and batch_idx + 1 == len(train_loader))
                or batch_idx + 1 >= limit_batches
            )
            should_optimizer_step = (
                batch_idx + 1
            ) % self.grad_accum_steps == 0 or is_last_batch

            batch_num_examples = self._num_examples_in_training_batch(batch)
            accumulated_num_examples += batch_num_examples
            self._current_training_batch_num_examples = batch_num_examples
            try:
                self.training_step(
                    batch=batch,
                    batch_idx=batch_idx,
                    is_accumulating=not should_optimizer_step,  # triggers gradient syncing
                )
            finally:
                self._current_training_batch_num_examples = None

            self.fabric.call(
                "on_train_batch_end",
                trainer=self,
                outputs=self._current_train_return,
                batch=batch,
                batch_idx=batch_idx,
            )

            if should_optimizer_step:
                # Each training loss is weighted by its local example count before
                # backward. DDP averages those weighted gradients across ranks;
                # normalize by the global example count so variable-size packed
                # microbatches are equivalent to one larger mean batch.
                self._normalize_accumulated_gradients(accumulated_num_examples)
                accumulated_num_examples = 0

                self.fabric.call(
                    "on_before_optimizer_step",
                    trainer=self,
                    optimizer=self.state["optimizer"],
                )

                # ... step the optimizer, clipping gradients and updating EMA parameters if applicable
                # Note: step_optimizer() calls optimizer.step(), which internally triggers
                # on_after_optimizer_step callbacks via _FabricOptimizer
                self.step_optimizer()

                # ... call optimizer_step hook (distinct from on_after_optimizer_step which is called by Fabric)
                self.fabric.call(
                    "optimizer_step",
                    trainer=self,
                    optimizer=self.state["optimizer"],
                )

                # ... step the scheduler, if we're adjusting the learning rate at the optimizer step-level
                self.step_scheduler(
                    level="step", current_value=self.state["global_step"]
                )

            # ... increment the global step, if optimizer stepped
            # NOTE: Each node maintains its own global step
            self.state["global_step"] += int(should_optimizer_step)
            self.state["batches_completed_in_epoch"] = batch_idx + 1
            self.state["examples_completed_in_epoch"] = (
                int(self.state.get("examples_completed_in_epoch", 0))
                + batch_num_examples
            )

            # ... save checkpoint if we've reached the step-based checkpoint interval
            if (
                should_optimizer_step
                and self.checkpoint_every_n_steps is not None
                and self.state["global_step"] % self.checkpoint_every_n_steps == 0
            ):
                self.save_checkpoint("step")

            last_processed_batch_completed_epoch = is_last_batch
            batch_idx += 1

        if epoch_completed:
            self.fabric.call("on_train_epoch_end", trainer=self)
            self.state["batches_completed_in_epoch"] = 0
            self.state["examples_completed_in_epoch"] = 0
            self.state["epoch_progress_is_exact"] = True
        return epoch_completed

    def validation_loop(
        self,
        *,
        val_loaders: dict[str, _FabricDataLoader],
        limit_batches: int | float = float("inf"),
    ):
        """Run validation loop for a single validation epoch.

        Args:
            val_loader: Dictionary of Dataloaders (more precisely, _FabricDataLoader) for validation.
            limit_batches: Limit on the batches during this validation epoch. If greater than the number of batches in the
                `val_loader`, this argument has no effect. Helpful for debugging; should NOT be used for production.
        """
        # ... set model to evaluation mode
        self.state["model"].eval()

        with torch.no_grad():
            # ... assert we're in evaluation mode
            assert not self.state["model"].training

            self.fabric.call("on_validation_epoch_start", trainer=self)

            # ... iterate over all validation loaders
            for val_loader_name, val_loader in val_loaders.items():
                ranked_logger.info(
                    f"Running validation on dataset: {val_loader_name}, with {len(val_loader)} batches, with world_size={self.fabric.world_size}."
                )

                for batch_idx, batch in enumerate(val_loader):
                    # ... end validation epoch if stopping training completely or maximum desired batches for this epoch reached
                    if self.should_stop or batch_idx >= limit_batches:
                        break

                    is_distributed_padding = self._is_distributed_validation_padding(
                        val_loader,
                        batch_idx=batch_idx,
                    )
                    self.fabric.call(
                        "on_validation_batch_start",
                        trainer=self,
                        batch=batch,
                        batch_idx=batch_idx,
                        num_batches=len(val_loader),
                        dataset_name=val_loader_name,
                        is_distributed_padding=is_distributed_padding,
                    )

                    validation_result = self.validation_step(
                        batch=batch,
                        batch_idx=batch_idx,
                    )

                    self.fabric.call(
                        "on_validation_batch_end",
                        trainer=self,
                        outputs=validation_result,
                        batch=batch,
                        batch_idx=batch_idx,
                        num_batches=len(val_loader),
                        dataset_name=val_loader_name,
                        is_distributed_padding=is_distributed_padding,
                    )

            self.fabric.call("on_validation_epoch_end", trainer=self)

            # ... reset the model to training mode
            self.state["model"].train()

    @abstractmethod
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

        Subclasses should call :meth:`_backward_training_loss` for scalar losses
        averaged over examples. It weights each microbatch before backward so
        accumulation remains correct for variable batch sizes.

        Returns:
            torch.Tensor | Mapping[str, Any]: The loss tensor or a dictionary containing the loss tensor.
        """
        pass

    @abstractmethod
    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
        val_loader_name: str | None = None,
    ) -> dict:
        """Validation step, running forward pass.

        Args:
            batch: The current batch; can be of any form.
            batch_idx: The index of the current batch (within that validation loader).
            val_loader_name: The name of the validation loader, if applicable.

        Returns:
            dict: A dictionary containing the output of the designated validation metrics.
        """
        pass

    def validate(
        self,
        val_loaders: dict,
        ckpt_path: Path | str,
    ) -> None:
        """Validate a model using the given dataloaders and checkpoint.

        Args:
            val_loaders: A dictionary of dataloaders for validation, where keys are names and values are dataloaders.
            ckpt_path: Path to a specific checkpoint file to load. If None, the model will be validated as is.
        """
        assert (
            hasattr(self, "state") and "model" in self.state
        ), "Model not found in state dictionary! You must call `instantiate_model()` before running validate()."

        self.setup_model_optimizers_and_schedulers()

        self.load_checkpoint(ckpt_path)

        # Setup validation dataloaders with Fabric
        for key, loader in val_loaders.items():
            val_loaders[key] = self.fabric.setup_dataloaders(
                loader,
                use_distributed_sampler=False,
                move_to_device=not getattr(
                    self,
                    "collate_validation_batch_before_device_transfer",
                    False,
                ),
            )

        # Run the validation loop
        self.validation_loop(
            val_loaders=val_loaders, limit_batches=self.limit_val_batches
        )

    def step_optimizer(self):
        """Step the optimizer.

        This method must be called only when the optimizer is stepped (i.e., after accumulating the desired number of gradients).

        We then perform following steps:
            1. Check for NaN/Inf gradients (skip update if skip_nan_grad=True and found).
            2. Clip gradients, if applicable.
            3. Step the optimizer.
            4. Zero the gradients.
            5. Update the EMA parameters, if applicable.
        """
        assert "optimizer" in self.state and isinstance(
            self.state["optimizer"], _FabricOptimizer
        )
        assert "model" in self.state and isinstance(
            self.state["model"], _FabricModule | EMA
        )

        optimizer = self.state["optimizer"]
        model = self.state["model"]

        # ... check for NaN/Inf gradients, if applicable
        if self.skip_nan_grad:
            has_nan_or_inf = False
            for param in model.parameters():
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        has_nan_or_inf = True
                        break

            if has_nan_or_inf:
                ranked_logger.warning(
                    f"Skipping optimizer step at global_step={self.state['global_step']} due to NaN/Inf gradients"
                )
                optimizer.zero_grad()
                return  # Skip this update

        # ... clip gradients, if applicable
        if self.clip_grad_max_norm is not None:
            self.fabric.clip_gradients(
                module=model,
                optimizer=optimizer,
                max_norm=self.clip_grad_max_norm,
                error_if_nonfinite=self.error_if_grad_nonfinite,
            )

        # ... step the optimizer
        optimizer.step()

        # ... zero gradients
        optimizer.zero_grad()

        # ... update EMA parameters, if applicable
        if hasattr(model, "update"):
            model.update()

    @staticmethod
    def _num_examples_in_training_batch(batch: Any) -> int:
        """Infer the number of examples represented by a training batch.

        Nucleus training loaders return lists of per-example dictionaries, while
        ordinary PyTorch loaders return batched tensor trees. Subclasses with an
        ambiguous custom collate format can override this method.

        Args:
            batch: Batch passed to :meth:`training_step`.

        Returns:
            Positive number of examples represented by ``batch``.
        """

        def infer(value: Any) -> int | None:
            if torch.is_tensor(value):
                return int(value.shape[0]) if value.ndim > 0 else None
            if isinstance(value, np.ndarray):
                return int(value.shape[0]) if value.ndim > 0 else None
            if isinstance(value, Mapping):
                for child in value.values():
                    inferred = infer(child)
                    if inferred is not None:
                        return inferred
                return None
            if isinstance(value, tuple):
                for child in value:
                    inferred = infer(child)
                    if inferred is not None:
                        return inferred
                return None
            if isinstance(value, list):
                if value and all(isinstance(child, Mapping) for child in value):
                    return len(value)
                inferred_sizes = [
                    inferred
                    for child in value
                    if (inferred := infer(child)) is not None
                ]
                if inferred_sizes:
                    first_size = inferred_sizes[0]
                    if any(size != first_size for size in inferred_sizes[1:]):
                        raise ValueError(
                            "Could not infer a consistent training batch size from "
                            f"list fields with leading sizes {inferred_sizes}."
                        )
                    return first_size
                return len(value) if value else None
            return None

        num_examples = infer(batch)
        if num_examples is None or num_examples <= 0:
            raise ValueError(
                "Could not infer a positive number of examples from training batch "
                f"of type {type(batch).__name__}. Override "
                "_num_examples_in_training_batch() for this collate format."
            )
        return num_examples

    def _backward_training_loss(self, loss: torch.Tensor, **kwargs: Any) -> None:
        """Backpropagate a mean training loss with example-count weighting.

        The matching normalization is applied immediately before the optimizer
        step by :meth:`_normalize_accumulated_gradients`. Keeping the reported
        loss unscaled preserves logging semantics.

        Args:
            loss: Scalar loss averaged over the current batch's examples.
            **kwargs: Additional keyword arguments forwarded to Fabric backward.
        """
        num_examples = getattr(self, "_current_training_batch_num_examples", None)
        if num_examples is None:
            raise RuntimeError(
                "_backward_training_loss() must be called from training_step()."
            )
        self.fabric.backward(loss * num_examples, **kwargs)

    def _normalize_accumulated_gradients(self, local_num_examples: int) -> None:
        """Normalize example-weighted gradients over all distributed ranks.

        Args:
            local_num_examples: Number of rank-local examples contributing to
                the current optimizer-step window. This includes a possibly
                partial final accumulation window.
        """
        if local_num_examples <= 0:
            raise ValueError("local_num_examples must be positive.")

        model = self.state["model"]
        parameters = getattr(model, "parameters", None)
        if parameters is None:
            return

        parameters_with_grad = [
            parameter for parameter in parameters() if parameter.grad is not None
        ]
        world_size = int(getattr(self.fabric, "world_size", 1))
        if world_size <= 0:
            raise ValueError("Fabric world_size must be positive.")

        count_device = (
            parameters_with_grad[0].grad.device
            if parameters_with_grad
            else getattr(self.fabric, "device", torch.device("cpu"))
        )
        local_count = torch.tensor(
            local_num_examples,
            dtype=torch.long,
            device=count_device,
        )
        global_count = (
            self.fabric.all_reduce(local_count, reduce_op="sum")
            if world_size > 1
            else local_count
        )
        # All ranks must participate in the count collective even if a dynamic
        # task leaves this rank without parameter gradients.
        if not parameters_with_grad:
            return

        # DDP has already divided the synchronized gradient by ``world_size``.
        # Undo that division while averaging by the actual global example count.
        scale = (
            torch.as_tensor(
                world_size,
                dtype=torch.float32,
                device=global_count.device,
            )
            / global_count
        )
        with torch.no_grad():
            for parameter in parameters_with_grad:
                parameter.grad.mul_(
                    scale.to(device=parameter.grad.device, dtype=parameter.grad.dtype)
                )

    def step_scheduler(
        self,
        level: Literal["epoch", "step"],
        current_value: int,
    ):
        """Step the learning rate scheduler.

        Args:
            level: The level at which to step the scheduler. Either "epoch" or "step".
            current_value: The current epoch or step value.
        """
        # (No scheduler)
        if "scheduler_cfg" not in self.state or self.state["scheduler_cfg"] is None:
            return
        else:
            scheduler_cfg = self.state["scheduler_cfg"]

        # (Wrong interval; e.g., we adjust learning rate every epoch, but we are stepping at the step level)
        if scheduler_cfg.interval != level:
            return

        # (Right interval, but wrong frequency)
        if current_value % cast(int, scheduler_cfg.frequency) != 0:
            return

        # ... step the scheduler
        scheduler_cfg.scheduler.step()

    def _gather_rank_states(self, local_state: Any) -> list[Any]:
        """
        Gather a small checkpoint object from every distributed rank.

        Args:
            local_state: The local checkpoint state to gather from this rank.

        Returns:
            A list of checkpoint states from all ranks, where the index corresponds to the rank.
        """
        world_size = int(getattr(self.fabric, "world_size", 1))
        if world_size == 1:
            return [local_state]
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            raise RuntimeError(
                "Cannot gather rank-local checkpoint state before torch.distributed "
                "is initialized."
            )

        rank_states = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(rank_states, local_state)
        return rank_states

    @staticmethod
    def _unwrap_dataloader(train_loader: _FabricDataLoader) -> Any:
        """
        Return the stateful loader wrapped by Lightning Fabric.

        Args:
            train_loader: The Lightning Fabric dataloader to unwrap.

        Returns:
            The underlying stateful dataloader, or the original dataloader if it is not wrapped by Lightning Fabric.
        """
        return getattr(train_loader, "_dataloader", train_loader)

    @classmethod
    def _is_distributed_validation_padding(
        cls,
        val_loader: _FabricDataLoader,
        *,
        batch_idx: int,
    ) -> bool:
        """Return whether the first evaluated example is sampler padding.

        PyTorch's distributed validation sampler appends examples until every
        rank has the same number of forwards. AtomWorks' load-balanced variant
        follows the same append-then-rank-stride scheme. The model forward must
        still run for these slots to keep distributed ranks aligned, but metric
        and dump callbacks should not count their outputs.

        Zatom validation evaluates only the first example when a loader batch
        contains multiple examples, so this method classifies that same local
        sample position.
        """
        loader = cls._unwrap_dataloader(val_loader)
        sampler = getattr(loader, "sampler", None)
        visited_sampler_ids: set[int] = set()
        while sampler is not None and not isinstance(sampler, DistributedSampler):
            sampler_id = id(sampler)
            if sampler_id in visited_sampler_ids:
                return False
            visited_sampler_ids.add(sampler_id)
            sampler = getattr(sampler, "sampler", None)

        if sampler is None or bool(sampler.drop_last):
            return False

        dataset_size = len(sampler.dataset)
        batch_size = getattr(loader, "batch_size", None)
        if not isinstance(batch_size, int) or batch_size <= 0:
            return False

        local_sample_ordinal = int(batch_idx) * batch_size
        global_sample_position = int(sampler.rank) + local_sample_ordinal * int(
            sampler.num_replicas
        )
        return global_sample_position >= dataset_size

    @staticmethod
    def _set_missing_sampler_generators(
        sampler: Any, generator: torch.Generator
    ) -> None:
        """
        Attach a loader generator to sampler nodes that use global RNG.

        Args:
            sampler: The sampler to attach the generator to.
            generator: The global RNG generator to attach.
        """
        visited: set[int] = set()

        def visit(current_sampler: Any) -> None:
            """Visit a sampler node and attach the generator if it is missing."""
            if current_sampler is None or id(current_sampler) in visited:
                return
            visited.add(id(current_sampler))
            if (
                hasattr(current_sampler, "generator")
                and current_sampler.generator is None
            ):
                current_sampler.generator = generator
            for attribute in ("sampler", "fallback_sampler"):
                visit(getattr(current_sampler, attribute, None))
            for child_sampler in getattr(current_sampler, "samplers", ()):
                visit(child_sampler)

        visit(sampler)

    def _train_dataloader_generators(self) -> list[torch.Generator]:
        """
        Return unique generators used by the train loader and sampler tree.

        Returns:
            A list of unique torch.Generator instances used by the train loader and its sampler tree.
        """
        train_loader = self._unwrap_dataloader(self._train_loader)
        generators: list[torch.Generator] = []
        visited_objects: set[int] = set()
        visited_generators: set[int] = set()

        def visit(current: Any) -> None:
            """Visit a sampler node and collect its generator if it is a torch.Generator."""
            if current is None or id(current) in visited_objects:
                return
            visited_objects.add(id(current))
            generator = getattr(current, "generator", None)
            if (
                isinstance(generator, torch.Generator)
                and id(generator) not in visited_generators
            ):
                visited_generators.add(id(generator))
                generators.append(generator)
            for attribute in ("sampler", "fallback_sampler"):
                visit(getattr(current, attribute, None))
            for child in getattr(current, "samplers", ()):
                visit(child)

        visit(train_loader)
        return generators

    @staticmethod
    def _create_train_iterator(train_loader: _FabricDataLoader) -> Any:
        """Eagerly construct Fabric's underlying loader iterator.

        Lightning's wrapper implements ``__iter__`` as a lazy generator, which
        otherwise delays sampler setup and base-seed draws until the first
        ``next()`` call—too late to restore process RNG safely.

        Args:
            train_loader: The Lightning Fabric dataloader to create an iterator for.

        Returns:
            An iterator over the underlying dataloader, with sampler epoch set and data moved to the appropriate device if applicable.
        """
        raw_train_loader = getattr(train_loader, "_dataloader", None)
        if raw_train_loader is None:
            return iter(train_loader)

        _set_sampler_epoch(raw_train_loader, train_loader._num_iter_calls)
        train_loader._num_iter_calls += 1
        raw_iterator = iter(raw_train_loader)
        if train_loader.device is None:
            return raw_iterator
        return (
            move_data_to_device(batch, train_loader.device) for batch in raw_iterator
        )

    def _capture_train_dataloader_state(self) -> dict[str, Any] | None:
        """
        Capture rank-local train-loader cursors in one checkpoint payload.

        Returns:
            A dictionary containing the state of the train loader, including the dataloader state and generator state, or None if the train loader is not stateful.
        """
        train_loader = self._unwrap_dataloader(self._train_loader)
        if not hasattr(train_loader, "state_dict"):
            ranked_logger.warning(
                f"Train loader {type(train_loader).__name__} is not stateful. "
                "Saving model/optimizer state without a mid-epoch dataloader "
                "cursor; resuming this step checkpoint will restart the epoch."
            )
            return None

        generator = getattr(train_loader, "generator", None)
        local_state = {
            "dataloader": train_loader.state_dict(),
            "generator": (
                generator.get_state()
                if isinstance(generator, torch.Generator)
                else None
            ),
        }
        sampler = getattr(train_loader, "sampler", None)
        underlying_sampler = getattr(sampler, "sampler", sampler)
        dataset = getattr(train_loader, "dataset", None)
        underlying_dataset = getattr(dataset, "dataset", dataset)
        return {
            "version": 2,
            "world_size": int(getattr(self.fabric, "world_size", 1)),
            "loader_type": type(train_loader).__qualname__,
            "num_batches": len(train_loader),
            "batch_size": getattr(train_loader, "batch_size", None),
            "num_workers": getattr(train_loader, "num_workers", None),
            "persistent_workers": getattr(train_loader, "persistent_workers", None),
            "drop_last": getattr(train_loader, "drop_last", None),
            "in_order": getattr(train_loader, "in_order", None),
            "max_num_tokens": getattr(train_loader, "max_num_tokens", None),
            "candidate_batch_size": getattr(train_loader, "candidate_batch_size", None),
            "worker_chunk_size": getattr(train_loader, "worker_chunk_size", None),
            "sampler_type": type(underlying_sampler).__qualname__,
            "dataset_type": type(underlying_dataset).__qualname__,
            "dataset_length": len(dataset) if dataset is not None else None,
            "grad_accum_steps": self.grad_accum_steps,
            "limit_train_batches": self.limit_train_batches,
            "rank_states": self._gather_rank_states(local_state),
        }

    def _restore_train_dataloader_state(
        self,
        train_loader: _FabricDataLoader,
        *,
        checkpoint_path: Path,
    ) -> None:
        """
        Restore this rank's train-loader cursor from a step checkpoint.

        Args:
            train_loader: The Lightning Fabric dataloader to restore.
            checkpoint_path: The path to the checkpoint file.
        """
        checkpoint_state = getattr(self, "_loaded_train_dataloader_state", None)
        if checkpoint_state is None:
            ranked_logger.warning(
                f"Step checkpoint {checkpoint_path} predates stateful dataloader "
                "checkpointing; restarting its current epoch from batch 0."
            )
            self.state["batches_completed_in_epoch"] = 0
            return
        if checkpoint_state.get("version") != 2:
            raise ValueError("Unsupported train dataloader checkpoint state version.")

        world_size = int(getattr(self.fabric, "world_size", 1))
        if checkpoint_state["world_size"] != world_size:
            raise ValueError(
                "Cannot resume a step checkpoint with a different world size: "
                f"checkpoint={checkpoint_state['world_size']}, current={world_size}."
            )

        raw_train_loader = self._unwrap_dataloader(train_loader)
        if (
            getattr(raw_train_loader, "num_workers", 0) > 0
            and getattr(raw_train_loader, "in_order", True) is False
        ):
            raise ValueError(
                "Cannot restore a mid-epoch checkpoint with num_workers > 0 and "
                "in_order=False; torchdata does not guarantee state restoration "
                "for out-of-order multiprocessing."
            )
        if type(raw_train_loader).__qualname__ != checkpoint_state["loader_type"]:
            raise ValueError(
                "Train loader type changed while resuming a step checkpoint: "
                f"checkpoint={checkpoint_state['loader_type']}, "
                f"current={type(raw_train_loader).__qualname__}."
            )
        if len(raw_train_loader) != checkpoint_state["num_batches"]:
            raise ValueError(
                "Train loader length changed while resuming a step checkpoint: "
                f"checkpoint={checkpoint_state['num_batches']}, "
                f"current={len(raw_train_loader)}."
            )
        if (
            getattr(raw_train_loader, "batch_size", None)
            != checkpoint_state["batch_size"]
        ):
            raise ValueError(
                "Train loader batch size changed while resuming a step checkpoint: "
                f"checkpoint={checkpoint_state['batch_size']}, "
                f"current={getattr(raw_train_loader, 'batch_size', None)}."
            )

        compatibility_fields = {
            "num_workers": getattr(raw_train_loader, "num_workers", None),
            "persistent_workers": getattr(raw_train_loader, "persistent_workers", None),
            "drop_last": getattr(raw_train_loader, "drop_last", None),
            "in_order": getattr(raw_train_loader, "in_order", None),
            "max_num_tokens": getattr(raw_train_loader, "max_num_tokens", None),
            "candidate_batch_size": getattr(
                raw_train_loader, "candidate_batch_size", None
            ),
            "worker_chunk_size": getattr(raw_train_loader, "worker_chunk_size", None),
            "grad_accum_steps": self.grad_accum_steps,
            "limit_train_batches": self.limit_train_batches,
        }
        for field, current_value in compatibility_fields.items():
            if checkpoint_state[field] != current_value:
                raise ValueError(
                    f"{field} changed while resuming a step checkpoint: "
                    f"checkpoint={checkpoint_state[field]!r}, "
                    f"current={current_value!r}."
                )

        sampler = getattr(raw_train_loader, "sampler", None)
        underlying_sampler = getattr(sampler, "sampler", sampler)
        dataset = getattr(raw_train_loader, "dataset", None)
        underlying_dataset = getattr(dataset, "dataset", dataset)
        structural_fields = {
            "sampler_type": type(underlying_sampler).__qualname__,
            "dataset_type": type(underlying_dataset).__qualname__,
            "dataset_length": len(dataset) if dataset is not None else None,
        }
        for field, current_value in structural_fields.items():
            if checkpoint_state[field] != current_value:
                raise ValueError(
                    f"{field} changed while resuming a step checkpoint: "
                    f"checkpoint={checkpoint_state[field]!r}, "
                    f"current={current_value!r}."
                )
        if not hasattr(raw_train_loader, "load_state_dict"):
            raise TypeError(
                f"Train loader {type(raw_train_loader).__name__} cannot restore state."
            )

        rank = int(getattr(self.fabric, "global_rank", 0))
        rank_state = checkpoint_state["rank_states"][rank]
        generator = getattr(raw_train_loader, "generator", None)
        if rank_state["generator"] is not None:
            if not isinstance(generator, torch.Generator):
                raise ValueError(
                    "Checkpoint contains a train-loader generator, but the current "
                    "loader has none."
                )
            generator.set_state(rank_state["generator"])
        raw_train_loader.load_state_dict(rank_state["dataloader"])
        ranked_logger.info(
            "Restored train dataloader at batch "
            f"{self.state['batches_completed_in_epoch']} of epoch "
            f"{self.state['current_epoch']}."
        )

    def _capture_checkpoint_rng_state(self) -> dict[str, Any]:
        """
        Capture process RNG state independently on every distributed rank.

        Returns:
            A dictionary containing the RNG state for the current rank, including the state of the dataloader generators.
        """
        dataloader_generator_states = (
            [generator.get_state() for generator in self._train_dataloader_generators()]
            if hasattr(self, "_train_loader")
            else []
        )
        return {
            "version": 3,
            "world_size": int(getattr(self.fabric, "world_size", 1)),
            "rank_states": self._gather_rank_states(
                {
                    "process": _capture_rng_state(),
                    "dataloader_generators": dataloader_generator_states,
                }
            ),
        }

    def _handle_resume_world_size_change(
        self,
        *,
        checkpoint_kind: CheckpointKind | None,
        checkpoint_path: Path,
    ) -> None:
        """Discard incompatible epoch RNG state for an allowed world-size change.

        Step checkpoints also contain rank-local dataloader cursors and gradient
        accumulation metadata, so they cannot be made compatible by dropping RNG
        state alone.

        Args:
            checkpoint_kind: Checkpoint kind loaded by :meth:`fit`.
            checkpoint_path: Loaded checkpoint path, used in diagnostics.
        """
        checkpoint_state = getattr(self, "_loaded_rng_state", None)
        if checkpoint_state is None or checkpoint_state.get("version") != 3:
            return

        checkpoint_world_size = checkpoint_state["world_size"]
        current_world_size = int(getattr(self.fabric, "world_size", 1))
        if checkpoint_world_size == current_world_size:
            return
        if not self.allow_world_size_change_on_resume:
            return
        if checkpoint_kind != "epoch":
            raise ValueError(
                "allow_world_size_change_on_resume only supports epoch "
                "checkpoints; cannot resume "
                f"{checkpoint_path} with checkpoint world size "
                f"{checkpoint_world_size} and current world size "
                f"{current_world_size}."
            )

        ranked_logger.warning(
            f"Resuming epoch checkpoint {checkpoint_path} with a different world "
            f"size (checkpoint={checkpoint_world_size}, "
            f"current={current_world_size}). Discarding rank-local RNG state; "
            "the resumed trajectory will not be bitwise reproducible."
        )
        self._loaded_rng_state = None

    def _restore_checkpoint_loader_generator_state(self) -> None:
        """Restore loader-generator state before reconstructing an iterator."""
        checkpoint_state = self._loaded_rng_state
        if checkpoint_state.get("version") != 3:
            raise ValueError("Unsupported RNG checkpoint state version.")
        world_size = int(getattr(self.fabric, "world_size", 1))
        if checkpoint_state["world_size"] != world_size:
            raise ValueError(
                "Cannot restore rank-local RNG state with a different world size: "
                f"checkpoint={checkpoint_state['world_size']}, current={world_size}."
            )

        rank = int(getattr(self.fabric, "global_rank", 0))
        generator_states = checkpoint_state["rank_states"][rank][
            "dataloader_generators"
        ]
        generators = self._train_dataloader_generators()
        if len(generators) != len(generator_states):
            raise ValueError(
                "Train dataloader generator structure changed while restoring RNG "
                f"state: checkpoint={len(generator_states)}, current={len(generators)}."
            )
        self._loaded_dataloader_generator_states = generator_states
        for generator, generator_state in zip(
            generators, generator_states, strict=True
        ):
            generator.set_state(generator_state)

    def _restore_checkpoint_rng_state(self) -> None:
        """Restore this rank's process RNG state when the checkpoint contains it."""
        checkpoint_state = getattr(self, "_loaded_rng_state", None)
        if checkpoint_state is None:
            return
        if checkpoint_state.get("version") != 3:
            raise ValueError("Unsupported RNG checkpoint state version.")

        world_size = int(getattr(self.fabric, "world_size", 1))
        if checkpoint_state["world_size"] != world_size:
            raise ValueError(
                "Cannot restore rank-local RNG state with a different world size: "
                f"checkpoint={checkpoint_state['world_size']}, current={world_size}."
            )
        rank = int(getattr(self.fabric, "global_rank", 0))
        _restore_rng_state(checkpoint_state["rank_states"][rank]["process"])

    def save_checkpoint(self, checkpoint_kind: CheckpointKind = "epoch") -> None:
        """Saves a checkpoint with current state to `self.output_dir/ckpt`.

        If no output directory is specified, then no checkpoint is saved.

        Args:
            checkpoint_kind: The kind of checkpoint to save. Either "epoch" or "step". Defaults to "epoch".
        """
        # No checkpoint directory; skip saving
        if not self.output_dir:
            ranked_logger.warning(
                "No output directory specified; skipping model checkpointing of state dictionary."
            )
            return

        # (Provide a hook to modify the state before saving)
        self.fabric.call("on_save_checkpoint", trainer=self, state=self.state)

        # ... construct the checkpoint file path using Path
        checkpoint_file = get_checkpoint_path(
            self.output_dir,
            checkpoint_kind,
            current_epoch=self.state["current_epoch"],
            global_step=self.state["global_step"],
        )

        checkpoint_state = dict(self.state)
        if "train_cfg" in checkpoint_state:
            checkpoint_state["train_cfg"] = checkpoint_train_config(
                checkpoint_state["train_cfg"],
                self.state.get("model"),
                getattr(self, "loss", None),
            )
        checkpoint_state["checkpoint_kind"] = checkpoint_kind
        checkpoint_state["train_dataloader_state"] = (
            self._capture_train_dataloader_state()
            if checkpoint_kind == "step" and hasattr(self, "_train_loader")
            else None
        )
        checkpoint_state["rng_state"] = self._capture_checkpoint_rng_state()

        # NOTE: Fabric's `save()` will call the `state_dict()` method on the model, optimizer, and scheduler_cfg
        self.fabric.save(checkpoint_file, checkpoint_state)
        ranked_logger.info(f"Saved checkpoint to: {checkpoint_file}")
        checkpoint_keep_last_n = getattr(self, "checkpoint_keep_last_n", None)
        checkpoint_retention_root = getattr(self, "checkpoint_retention_root", None)
        if (
            checkpoint_keep_last_n is not None
            and checkpoint_retention_root is not None
            and self.fabric.is_global_zero
        ):
            deleted_paths = prune_checkpoints(
                checkpoint_retention_root,
                keep_last_n=checkpoint_keep_last_n,
            )
            if deleted_paths:
                ranked_logger.info(
                    f"Pruned {len(deleted_paths)} old checkpoints under "
                    f"{checkpoint_retention_root}; retaining the newest "
                    f"{checkpoint_keep_last_n}."
                )

    def _load_optimizer(self, ckpt: Mapping) -> None:
        """
        Loads the optimizer state from the checkpoint.

        Args:
            ckpt: The checkpoint mapping containing the optimizer state.
        """
        if "optimizer" in ckpt and self.state["optimizer"]:
            self.state["optimizer"].load_state_dict(ckpt["optimizer"])
        else:
            ranked_logger.warning("Skipping optimizer loading...")

    def _load_scheduler(self, ckpt: Mapping) -> None:
        """
        Loads the learning rate scheduler state from the checkpoint.

        Args:
            ckpt: The checkpoint mapping containing the scheduler state.
        """
        if "scheduler_cfg" in ckpt and self.state["scheduler_cfg"]:
            self.state["scheduler_cfg"].load_state_dict(ckpt["scheduler_cfg"])
        else:
            ranked_logger.warning("Skipping scheduler loading...")

    def _load_model(
        self, ckpt: Mapping, weight_loading_config: WeightLoadingConfig | None = None
    ) -> None:
        """
        Loads the model state from the checkpoint, handling EMA and size mismatches.

        Args:
            ckpt: The checkpoint mapping containing the model state.
            weight_loading_config: Weight loading policies to apply. Defaults to ``None``.
        """
        # ... load pre-trained weights from the CHECKPOINT into the MODEL (that at this point has random weights)
        model = self.state["model"]
        model.load_state_dict(
            load_weights_with_policies(
                model=self.state["model"],
                ckpt=ckpt["model"],
                config=weight_loading_config,
            ),
            strict=True,
        )

    def load_checkpoint(
        self,
        checkpoint: Path | str | dict,
        weight_loading_config: WeightLoadingConfig | None = None,
        reset_optimizer: bool = False,
    ) -> None:
        """Loads a checkpoint from the specified path or uses a pre-loaded checkpoint dict.

        Args:
          checkpoint: Either a path to a checkpoint file or a pre-loaded checkpoint dict.
          weight_loading_config: Weight loading policies to apply. Defaults to ``None``.
          reset_optimizer: Whether to reset the optimizer state. Defaults to ``False``.
        """
        # ... load the checkpoint or use the provided dict
        if isinstance(checkpoint, dict):
            ranked_logger.info("Using pre-loaded checkpoint...")
            ckpt = checkpoint
        else:
            ranked_logger.info(f"Loading checkpoint from: {checkpoint}...")
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

        self._loaded_train_dataloader_state = ckpt.get("train_dataloader_state")
        self._loaded_rng_state = ckpt.get("rng_state")
        self._loaded_checkpoint_kind = ckpt.get("checkpoint_kind")
        if (
            "examples_completed_in_epoch" not in ckpt
            and int(ckpt.get("batches_completed_in_epoch", 0)) > 0
        ):
            self.state["epoch_progress_is_exact"] = False
        if self._loaded_checkpoint_kind not in {None, "epoch", "step"}:
            raise ValueError(
                f"Unsupported checkpoint kind: {self._loaded_checkpoint_kind!r}."
            )

        try:
            # ... optimize, scheduler
            if not reset_optimizer:
                self._load_optimizer(ckpt)
                self._load_scheduler(ckpt)
            # ... model
            self._load_model(ckpt, weight_loading_config)

            # ... stateless keys
            # (We do not want to load the `train_cfg` in this instance, as it may contain different configurations)
            keys_to_ignore = {
                "model",
                "optimizer",
                "scheduler_cfg",
                "train_cfg",
                "train_dataloader_state",
                "rng_state",
                "checkpoint_kind",
            }
            self.state.update(
                {
                    key: value
                    for key, value in ckpt.items()
                    if key not in keys_to_ignore and key in self.state
                }
            )

            # Log warnings for missing and extra keys
            state_keys = set(self.state) - keys_to_ignore
            ckpt_keys = set(ckpt) - keys_to_ignore

            if missing := state_keys - ckpt_keys:
                ranked_logger.warning(
                    f"Keys found in STATE but not CKPT: {sorted(missing)}"
                )
            if extra := ckpt_keys - state_keys:
                ranked_logger.warning(
                    f"Keys found in CKPT but not STATE: {sorted(extra)}"
                )

            ranked_logger.info(
                f"Loaded checkpoint. Current epoch: {self.state['current_epoch']}, global step: {self.state['global_step']}"
            )
        except Exception as e:
            ranked_logger.exception(
                f"Error loading checkpoint: {e}. Please ensure that the model architecture matches the checkpoint."
            )
            raise

    @staticmethod
    def get_latest_checkpoint(ckpt_load_dir: Path) -> Path | None:
        """Returns the latest checkpoint file from the given directory.

        Args:
            ckpt_load_dir (Path): The directory to search for checkpoint files.

        Returns:
            Path: The path to the latest checkpoint file, or None if no checkpoints are found
            or if the directory does not exist.
        """
        return find_latest_checkpoint(ckpt_load_dir)

    @property
    def should_validate(self) -> bool:
        """
        Whether to currently run validation.

        Returns:
            bool: True if validation should be run, False otherwise.
        """
        if self.state["current_epoch"] == 0 and not self.validate_first_epoch:
            return False
        return self.state["current_epoch"] % self.validate_every_n_epochs == 0

    def _scheduled_validation_loaders(
        self,
        val_loaders: Mapping[str, _FabricDataLoader] | None,
    ) -> dict[str, _FabricDataLoader]:
        """Return validation loaders scheduled for the current epoch."""
        if not val_loaders or not self.should_validate:
            return {}

        current_epoch = int(self.state["current_epoch"])
        unknown_loader_names = set(self.validate_loader_every_n_epochs) - set(
            val_loaders
        )
        if unknown_loader_names:
            raise ValueError(
                "Validation intervals configured for unknown loaders: "
                f"{sorted(unknown_loader_names)}."
            )

        return {
            name: loader
            for name, loader in val_loaders.items()
            if current_epoch % self.validate_loader_every_n_epochs.get(name, 1) == 0
        }

    def _log_epoch_progress(self) -> None:
        """Log the zero-based epoch on the optimizer-step axis."""
        if self.fabric.is_global_zero:
            self.fabric.log_dict(
                {"trainer/epoch": self.state["current_epoch"]},
                step=self.state["global_step"],
            )
