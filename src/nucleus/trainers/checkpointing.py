"""Utilities for naming and discovering trainer checkpoints."""

import re
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.nn import Module

CheckpointKind = Literal["epoch", "step"]

_CHECKPOINT_NAME_PATTERN = re.compile(r"^(?P<kind>epoch|step)-(?P<index>\d+)\.ckpt$")
_CHECKPOINT_NAME_WIDTH = {
    "epoch": 4,
    "step": 8,
}
_CHECKPOINT_KIND_ORDER = {
    "epoch": 0,
    "step": 1,
}


def checkpoint_train_config(
    cfg: "DictConfig | dict",
    model: "Module | None",
    loss: "Module | None" = None,
) -> "DictConfig | dict":
    """Copy external property-statistics artifacts into the serialized config.

    Use the artifacts retained by the instantiated modules, never reread the
    source files at save time. The original config and runtime modules remain
    untouched; state_dict still carries the model's current buffers and weights.

    Args:
        cfg: Live training config.
        model: Model, optionally wrapped by Fabric, compilation, or EMA.
        loss: Instantiated training loss container, when present.

    Returns:
        A copied config with inline statistics, or cfg if no artifacts are used.
    """
    from omegaconf import OmegaConf

    saved_cfg = cfg
    for component, key in (
        (model, "model.net.diffusion_module.atomistic_property_head"),
        (loss, "trainer.loss.atomistic_property_loss"),
    ):
        if component is None:
            continue
        for module in component.modules():
            # Inspect owned attributes so wrappers cannot forward the same
            # artifact twice. EMA visits the live model before its shadow.
            statistics = vars(module).get("_property_statistics")
            if statistics is None:
                continue
            if saved_cfg is cfg:
                saved_cfg = OmegaConf.create(cfg)
            OmegaConf.update(
                saved_cfg,
                f"{key}.statistics",
                statistics.to_dict(),
                merge=False,
                force_add=True,
            )
            OmegaConf.update(saved_cfg, f"{key}.statistics_path", None, force_add=True)
            break
    return saved_cfg


def format_checkpoint_name(
    checkpoint_kind: CheckpointKind,
    *,
    current_epoch: int,
    global_step: int,
) -> str:
    """
    Return the checkpoint filename for a trainer state.

    Args:
        checkpoint_kind: The kind of checkpoint to format ("epoch" or "step").
        current_epoch: The current epoch number.
        global_step: The current global step number.

    Returns:
        The formatted checkpoint filename as a string.
    """
    if checkpoint_kind == "epoch":
        checkpoint_index = current_epoch
    elif checkpoint_kind == "step":
        checkpoint_index = global_step
    else:
        raise ValueError(f"Unsupported checkpoint kind: {checkpoint_kind!r}")

    if checkpoint_index < 0:
        raise ValueError(
            f"Checkpoint index must be non-negative, got {checkpoint_index}."
        )

    width = _CHECKPOINT_NAME_WIDTH[checkpoint_kind]
    return f"{checkpoint_kind}-{checkpoint_index:0{width}d}.ckpt"


def get_checkpoint_kind(checkpoint_path: Path | str) -> CheckpointKind | None:
    """Return the checkpoint kind encoded in a checkpoint filename.

    Args:
        checkpoint_path: The path to the checkpoint file.

    Returns:
        The checkpoint kind ("epoch" or "step") if the filename matches the pattern, otherwise None.
    """
    match = _CHECKPOINT_NAME_PATTERN.match(Path(checkpoint_path).name)
    if match is None:
        return None

    checkpoint_kind = match.group("kind")
    if checkpoint_kind == "epoch" or checkpoint_kind == "step":
        return checkpoint_kind
    return None


def checkpoint_completes_epoch(checkpoint_path: Path | str) -> bool:
    """Return whether resuming from a checkpoint should advance the epoch.

    Args:
        checkpoint_path: The path to the checkpoint file.

    Returns:
        True if the checkpoint completes an epoch, False otherwise.
    """
    return get_checkpoint_kind(checkpoint_path) == "epoch"


def get_checkpoint_path(
    output_dir: Path | str,
    checkpoint_kind: CheckpointKind,
    *,
    current_epoch: int,
    global_step: int,
) -> Path:
    """
    Return the checkpoint path under an output directory.

    Args:
        output_dir: The base directory for checkpoints.
        checkpoint_kind: The kind of checkpoint to format ("epoch" or "step").
        current_epoch: The current epoch number.
        global_step: The current global step number.

    Returns:
        The full path to the checkpoint file as a Path object.
    """
    checkpoint_name = format_checkpoint_name(
        checkpoint_kind,
        current_epoch=current_epoch,
        global_step=global_step,
    )
    return Path(output_dir) / "ckpt" / checkpoint_name


def iter_checkpoint_paths(ckpt_load_dir: Path | str) -> Iterator[Path]:
    """
    Yield checkpoint files recursively from a checkpoint directory.

    Args:
        ckpt_load_dir: The directory to search for checkpoint files.

    Yields:
        Paths to checkpoint files found in the directory and its subdirectories.
    """
    ckpt_load_dir = Path(ckpt_load_dir)
    if not ckpt_load_dir.is_dir():
        return

    yield from (
        checkpoint_path
        for checkpoint_path in ckpt_load_dir.rglob("*.ckpt")
        if checkpoint_path.is_file()
    )


def _checkpoint_sort_key(checkpoint_path: Path) -> tuple[int, int, int, str]:
    """
    Sort checkpoints by file write time, with deterministic filename tie-breaks.

    Args:
        checkpoint_path: The path to the checkpoint file.

    Returns:
        A tuple used for sorting checkpoints.
    """
    match = _CHECKPOINT_NAME_PATTERN.match(checkpoint_path.name)
    if match is None:
        checkpoint_index = -1
        checkpoint_kind_order = -1
    else:
        checkpoint_index = int(match.group("index"))
        checkpoint_kind_order = _CHECKPOINT_KIND_ORDER[match.group("kind")]

    return (
        checkpoint_path.stat().st_mtime_ns,
        checkpoint_index,
        checkpoint_kind_order,
        str(checkpoint_path),
    )


def get_latest_checkpoint(ckpt_load_dir: Path | str) -> Path | None:
    """Return the latest checkpoint file under a directory, if one exists.

    Checkpoint discovery is recursive so callers may pass either a run directory
    or its nested ``ckpt`` directory.

    Args:
        ckpt_load_dir: The directory to search for checkpoint files.

    Returns:
        The path to the latest checkpoint file, or None if no checkpoints are found.
    """
    return max(
        iter_checkpoint_paths(ckpt_load_dir),
        key=_checkpoint_sort_key,
        default=None,
    )


def prune_checkpoints(
    ckpt_load_dir: Path | str,
    *,
    keep_last_n: int,
) -> list[Path]:
    """Delete old trainer checkpoints while preserving the newest checkpoints.

    Only files whose names follow the trainer's ``epoch-*.ckpt`` or
    ``step-*.ckpt`` convention are eligible. Unrecognized checkpoint files are
    deliberately left untouched.

    Args:
        ckpt_load_dir: Directory to search recursively for checkpoints.
        keep_last_n: Number of newest trainer checkpoints to retain.

    Returns:
        Paths that were deleted, oldest first.
    """
    if keep_last_n <= 0:
        raise ValueError(f"keep_last_n must be positive, got {keep_last_n}.")

    checkpoint_paths = sorted(
        (
            path
            for path in iter_checkpoint_paths(ckpt_load_dir)
            if get_checkpoint_kind(path) is not None
        ),
        key=_checkpoint_sort_key,
    )
    paths_to_delete = checkpoint_paths[:-keep_last_n]
    for checkpoint_path in paths_to_delete:
        checkpoint_path.unlink()
    return paths_to_delete
