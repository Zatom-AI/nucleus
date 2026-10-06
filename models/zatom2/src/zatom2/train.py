import logging
import os

import hydra
import rootutils
from dotenv import load_dotenv
from omegaconf import DictConfig

# Setup root dir and environment variables (more info: https://github.com/ashleve/rootutils)
# NOTE: Sets the `PROJECT_ROOT` environment variable to the root directory of the project (where `.project-root` is located)
try:
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
except FileNotFoundError:
    os.environ.setdefault("PROJECT_ROOT", os.getcwd())

load_dotenv(override=True)

from zatom2.cli import _get_config_path

_config_path = _get_config_path()

_spawning_process_logger = logging.getLogger(__name__)


@hydra.main(config_path=_config_path, config_name="train", version_base="1.3")
def train(cfg: DictConfig) -> None:
    # ==============================================================================
    # Import dependencies and resolve Hydra configuration
    # ==============================================================================

    from nucleus.utils.logging import install_zatom_logger_name_aliases

    install_zatom_logger_name_aliases("zatom2")

    _spawning_process_logger.info("Importing dependencies...")

    # Lazy imports to make config generation fast
    import torch
    from foundry.utils.logging import suppress_warnings
    from foundry.utils.weights import CheckpointConfig
    from lightning.fabric import seed_everything
    from lightning.fabric.loggers import Logger

    # If training on DIGS L40, set precision of matrix multiplication to balance speed and accuracy
    # Reference: https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("medium")

    from foundry.callbacks.callback import BaseCallback  # noqa
    from foundry.utils.instantiators import instantiate_loggers, instantiate_callbacks  # noqa
    from foundry.utils.logging import (
        print_config_tree,
        log_hyperparameters_with_all_loggers,
    )  # noqa
    from foundry.utils.ddp import RankedLogger  # noqa
    from foundry.utils.ddp import is_rank_zero, set_accelerator_based_on_availability  # noqa
    from foundry.utils.datasets import (
        assemble_val_loader_dict,
    )  # noqa

    from nucleus.trainers.checkpointing import get_latest_checkpoint  # noqa
    from nucleus.utils.datasets import (
        assemble_distributed_loader,
        patch_atomworks_pandas_dataset_query_engine,
    )  # noqa
    from nucleus.utils.training_data import (
        build_train_dataset_and_sampler,
    )  # noqa

    patch_atomworks_pandas_dataset_query_engine()

    # Respect an explicit runtime override like `trainer_runtime=cpu`.
    # Only auto-pick hardware when the config asks for `accelerator=auto`.
    if cfg.trainer.accelerator == "auto":
        set_accelerator_based_on_availability(cfg)
    else:
        _spawning_process_logger.info(
            f"Using explicit trainer accelerator={cfg.trainer.accelerator!r}; skipping hardware auto-detection."
        )

    ranked_logger = RankedLogger(__name__, rank_zero_only=True)
    _spawning_process_logger.info("Completed dependency imports ...")

    # ... print the configuration tree (NOTE: Only prints for rank 0)
    print_config_tree(cfg, resolve=True)

    # ==============================================================================
    # Logging and Callback instantiation
    # ==============================================================================

    # Reduce the logging level for all dataset and sampler loggers (unless rank 0)
    # We will still see messages from Rank 0; they are identical, since all ranks load and sample from the same datasets
    if not is_rank_zero():
        dataset_logger = logging.getLogger("datasets")
        sampler_logger = logging.getLogger("atomworks.ml.samplers")
        dataset_logger.setLevel(logging.WARNING)
        sampler_logger.setLevel(logging.ERROR)

    # ... seed everything (NOTE: By setting `workers=True`, we ensure that the dataloaders are seeded as well)
    # (`PL_GLOBAL_SEED` environment variable will be passed to the spawned subprocesses; e.g., through `ddp_spawn` backend)
    if cfg.get("seed") is not None:
        ranked_logger.info(f"Seeding everything with seed={cfg.seed}...")
        seed_everything(cfg.seed, workers=True, verbose=True)
    else:
        ranked_logger.warning("No seed provided - Not seeding anything!")

    ranked_logger.info("Instantiating loggers...")
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))

    ranked_logger.info("Instantiating callbacks...")
    callbacks: list[BaseCallback] = instantiate_callbacks(cfg.get("callbacks"))

    # ==============================================================================
    # Trainer and model instantiation
    # ==============================================================================

    # ... instantiate the trainer
    ranked_logger.info("Instantiating trainer...")
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        loggers=loggers or None,
        callbacks=callbacks or None,
        _convert_="partial",
        _recursive_=False,
    )
    # (Store the Hydra configuration in the trainer state)
    trainer.initialize_or_update_trainer_state({"train_cfg": cfg})

    # ... spawn processes for distributed training
    # (We spawn here, rather than within `fit`, so we can use Fabric's `init_module` to efficiently initialize the model on the appropriate device)
    ranked_logger.info(
        f"Spawning {trainer.fabric.world_size} processes from {trainer.fabric.global_rank}..."
    )
    trainer.fabric.launch()

    # ... construct the model
    trainer.construct_model()

    # ... construct the optimizer and schedule (which requires the model to be constructed)
    trainer.construct_optimizer()
    trainer.construct_scheduler()

    # ==============================================================================
    # Dataset instantiation
    # ==============================================================================

    # Number of examples per epoch (across all GPUs)
    # (We must sample this many indices from our sampler)
    n_examples_per_epoch = cfg.trainer.n_examples_per_epoch

    # Build the canonical AtomWorks training dataset.
    assert (
        "train" in cfg.datasets and cfg.datasets.train
    ), "No 'train' dataloader configuration provided! If only performing validation, use `validate.py` instead."
    train_dataset, train_sampler = build_train_dataset_and_sampler(
        cfg,
        n_examples_per_epoch=n_examples_per_epoch,
        fabric=trainer.fabric,
    )

    # ... compose the train loader
    #     (`data_id` / `example_id` subsetting is handled inside
    #     `build_train_dataset_and_sampler` for both data paths)

    train_loader = assemble_distributed_loader(
        dataset=train_dataset,
        sampler=train_sampler,
        rank=trainer.fabric.global_rank,
        world_size=trainer.fabric.world_size,
        n_examples_per_epoch=n_examples_per_epoch,
        loader_cfg=cfg.dataloader["train"],
        max_num_tokens=cfg.datasets.get("max_num_tokens"),
        stateful=(
            trainer.checkpoint_every_n_steps is not None
            or bool(cfg.get("ckpt_config"))
            or bool(cfg.get("ckpt_path"))
        ),
    )

    # ... compose the validation loader(s)
    if "val" in cfg.datasets and cfg.datasets.val:
        val_loaders = assemble_val_loader_dict(
            cfg=cfg.datasets.val,
            rank=trainer.fabric.global_rank,
            world_size=trainer.fabric.world_size,
            loader_cfg=cfg.dataloader["val"],
        )
    else:
        ranked_logger.warning("No validation datasets provided! Skipping validation...")
        val_loaders = None

    ranked_logger.info("Logging hyperparameters...")
    log_hyperparameters_with_all_loggers(
        trainer=trainer, cfg=cfg, model=trainer.state["model"]
    )

    # ... load the checkpoint configuration
    def should_reset_optimizer(checkpoint_path: str) -> bool:
        """Reset only when requested or when the checkpoint lacks model state."""
        if bool(cfg.reset_optimizer_on_resume):
            return True
        if not bool(cfg.reset_optimizer_on_model_change):
            return False

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        checkpoint_model = checkpoint.get("model", {})
        current_model_keys = set(trainer.state["model"].state_dict())
        missing_model_keys = current_model_keys - checkpoint_model.keys()
        if missing_model_keys:
            ranked_logger.warning(
                "Resetting optimizer because the checkpoint predates model "
                f"parameters: {sorted(missing_model_keys)}"
            )
            return True
        return False

    ckpt_config = None
    if "ckpt_config" in cfg and cfg.ckpt_config:
        ckpt_config = hydra.utils.instantiate(cfg.ckpt_config)
    elif "ckpt_path" in cfg and cfg.ckpt_path:
        ckpt_path = str(cfg.ckpt_path)
        if os.path.exists(ckpt_path):
            if os.path.isdir(ckpt_path):
                latest_checkpoint = get_latest_checkpoint(ckpt_path)
                if latest_checkpoint is not None:
                    ckpt_config = CheckpointConfig(
                        path=str(latest_checkpoint),
                        reset_optimizer=should_reset_optimizer(str(latest_checkpoint)),
                    )
                else:
                    ranked_logger.warning(
                        f"No checkpoint files found under {ckpt_path}. "
                        "Training with new model weights."
                    )
            else:
                ckpt_config = CheckpointConfig(
                    path=ckpt_path,
                    reset_optimizer=should_reset_optimizer(ckpt_path),
                )
        elif ckpt_path.endswith(".ckpt"):
            raise ValueError("`ckpt_path` was given, but the path does not exist.")
        else:
            os.makedirs(ckpt_path, exist_ok=True)
            ranked_logger.info(
                f"Created checkpoint search directory {ckpt_path}. "
                "Training with new model weights."
            )

    # ... train the model
    ranked_logger.info("Training model...")

    with suppress_warnings():
        trainer.fit(
            train_loader=train_loader, val_loaders=val_loaders, ckpt_config=ckpt_config
        )


if __name__ == "__main__":
    train()
