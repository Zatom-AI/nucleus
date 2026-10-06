import logging
import warnings
from collections.abc import Callable

_LEGACY_ZATOM_LOGGER_ROOTS = ("rf3", "rfd3", "rfd3na")


class _ZatomLogRecordFactory:
    """Rewrite legacy package names on records without renaming their loggers."""

    def __init__(
        self,
        wrapped_factory: Callable[..., logging.LogRecord],
        model_package: str,
    ) -> None:
        self.wrapped_factory = wrapped_factory
        self.model_package = model_package

    def __call__(self, *args, **kwargs) -> logging.LogRecord:
        record = self.wrapped_factory(*args, **kwargs)
        for legacy_root in _LEGACY_ZATOM_LOGGER_ROOTS:
            if record.name == legacy_root:
                record.name = self.model_package
                break
            if record.name.startswith(f"{legacy_root}."):
                record.name = f"{self.model_package}{record.name[len(legacy_root) :]}"
                break
        return record


def install_zatom_logger_name_aliases(model_package: str) -> None:
    """Display legacy RF3-family logger names under the active Zatom package.

    This changes only ``LogRecord.name``. The underlying logger objects retain their
    original names, so logger hierarchy, level selection, and dependency behavior are
    unaffected.
    """
    if model_package not in {"zatom2"}:
        raise ValueError(f"Unsupported Zatom package: {model_package!r}")

    current_factory = logging.getLogRecordFactory()
    if isinstance(current_factory, _ZatomLogRecordFactory):
        current_factory.model_package = model_package
        return

    logging.setLogRecordFactory(
        _ZatomLogRecordFactory(current_factory, model_package=model_package)
    )


def configure_minimal_inference_logging() -> None:
    """Configure minimal logging for inference (quiet mode)."""
    # Suppress most logging by default
    logging.getLogger().setLevel(logging.WARNING)

    # Suppress specific noisy loggers
    for logger_name in [
        "atomworks",
        "transforms",
        "foundry.metrics",
        "foundry.trainers",
        "foundry.inference_engines.base",
        "nucleus.trainers",
        "nucleus.inference_engines.base",
        "rf3.trainers",
        "rf3.utils.inference",
        "lightning",
        "lightning_fabric",
        "root",
    ]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    # Keep model-specific inference engine loggers at INFO for progress messages
    logging.getLogger("rf3.inference_engines").setLevel(logging.INFO)
    logging.getLogger("rfd3.engine").setLevel(logging.INFO)
    logging.getLogger("zatom2.engine").setLevel(logging.INFO)

    # Suppress warnings
    warnings.filterwarnings("ignore")
