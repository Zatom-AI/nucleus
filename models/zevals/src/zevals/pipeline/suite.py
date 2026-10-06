"""`ZevalSuite` — the unified entrypoint for zevals.

Turns folders of generated samples (one per modality) into a single
WandB-loggable metrics dict, dispatching each modality through its own
Hydra-configured `EvalPipeline`. This is the intended integration point for
a training loop's validation epochs: construct one `ZevalSuite` at the start
of training and call `.evaluate(...)` on it repeatedly — see the class
docstring for why "construct once, reuse" matters (model-loading cost and
stable GPU device assignment, not just convenience).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from zevals.pipeline.base import EvalPipeline
from zevals.pipeline.loaders import LOADER_REGISTRY
from zevals.utils.reporting import print_merged_report, save_json_dict

logger = logging.getLogger(__name__)


def _resolve_config_dir() -> Path:
    """Locate the `configs/` directory in either a source checkout or an
    installed wheel.

    Installed wheels ship `configs/` as a sibling of the `zevals`
    package root (via `[tool.hatch.build.targets.wheel.force-include]`
    "configs" = "zevals/configs" in pyproject.toml), i.e.
    `<site-packages>/zevals/configs`. A source checkout keeps
    `configs/` at the repository root, i.e. `models/zevals/configs`, one
    level above `src/`. These are different relative offsets from this
    file (`pipeline/suite.py`), so both candidates are checked in order
    (installed-wheel layout first, since that's the common case for the
    "pip install and use in your own code" story this module exists for).

    Checking each candidate for `proteins.yaml` specifically (not just
    `.is_dir()`) matters: a source checkout also has an empty
    `src/zevals/configs/` placeholder (a `.gitkeep` used by the build
    system for the force-include target above) at the same relative offset
    as the installed-wheel layout, which would otherwise be picked over the
    real `configs/` one level further up.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "configs",  # installed: <pkg>/zevals/configs
        here.parents[3] / "configs",  # source checkout: zevals/configs
    ]
    for candidate in candidates:
        if (candidate / "proteins.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate the zevals configs/ directory. Checked: "
        + ", ".join(str(c) for c in candidates)
    )


class ZevalSuite:
    """Unified entrypoint for zevals: evaluates one or more modalities' worth
    of generated samples and returns a single WandB-loggable metrics dict.

    Construct once (e.g. at training start) and reuse across every
    validation epoch — pipelines (and the oracle models they own) are built
    lazily on first use per modality and cached, so checkpoints load once
    and GPU device assignment (`utils.devices.resolve_device`) stays stable
    across calls rather than drifting or reloading every epoch.

    zevals never gates on distributed rank itself — call `.evaluate()` only
    from whichever rank your training loop designates (e.g. rank 0):

        suite = ZevalSuite(modality_overrides={
            "proteins": ["oracle/inverse_folder=soluble_mpnn"],
        })
        ...
        for epoch in range(num_epochs):
            train_one_epoch(...)
            if is_main_process():
                metrics = suite.evaluate({
                    "proteins": f"outputs/epoch_{epoch}/backbones",
                    "molecules": f"outputs/epoch_{epoch}/ligands",
                })
                wandb.log(metrics, step=epoch)

    Metric keys in the returned dict are namespaced `"{modality}/{metric}"`
    (e.g. `"protein/designability"`, `"molecule/valid_rate"`), matching
    WandB's own `/`-based panel-grouping convention.
    """

    def __init__(
        self,
        modality_overrides: dict[str, list[str]] | None = None,
        results_dir: Path | str | None = None,
    ) -> None:
        self._modality_overrides = modality_overrides or {}
        self._pipelines: dict[str, EvalPipeline] = {}
        self.last_metadata: dict[str, dict[str, Any]] = {}
        self.results_dir = Path(results_dir) if results_dir else None

    def evaluate(
        self,
        samples_dirs: dict[str, Path | str],
        ground_truth_dirs: dict[str, Path | str] | None = None,
        print_results: bool = True,
    ) -> dict[str, float]:
        """Evaluate each `{modality: samples_dir}` pair and merge the results.

        A failure evaluating one modality (bad samples_dir, an oracle
        crashing, a missing checkpoint/binary) is caught, logs a warning,
        and does not block the other modalities in the same call — matches
        the graceful-degradation behaviour of each individual pipeline's
        `run()` (see `pipeline/base.py::safe_update`).

        Args:
            samples_dirs: Mapping of modality name (`"proteins"`,
                `"molecules"`, `"materials"`, `"rna"`, `"ligands"`) to the
                directory of generated samples for that modality.
            ground_truth_dirs: Optional mapping of modality name to a
                reference/ground-truth directory. Molecule conformer and
                material structure-prediction metrics use explicit matching
                ``<target_id>`` subdirectories in both directory trees.
            print_results: If True, print a rich summary table.

        Returns:
            Flat `dict[str, float]`, keys namespaced `"{modality}/{metric}"`.
            Directly JSON-serialisable and `wandb.log()`-able.
        """
        merged: dict[str, float] = {}
        self.last_metadata = {}
        for modality, samples_dir in samples_dirs.items():
            try:
                if modality not in LOADER_REGISTRY:
                    raise KeyError(
                        f"Unknown modality {modality!r}. Available: {sorted(LOADER_REGISTRY)}"
                    )
                pipeline = self._get_pipeline(modality)
                loader = LOADER_REGISTRY[modality]
                gt_dir = (ground_truth_dirs or {}).get(modality)
                samples = loader(Path(samples_dir), Path(gt_dir) if gt_dir else None)
                results = pipeline.run(samples)
                self.last_metadata[modality] = {
                    "pipeline_status": {"status": "complete"},
                    **results.metadata,
                }
                merged.update(
                    {f"{modality}/{k}": v for k, v in results.aggregate.items()}
                )
            except Exception as exc:
                self.last_metadata[modality] = {
                    "pipeline_status": {
                        "status": "failed",
                        "error": repr(exc),
                    }
                }
                logger.warning(
                    "Skipping modality %r this evaluate() call", modality, exc_info=True
                )

        if print_results:
            print_merged_report(merged, title="zevals Results")
        if self.results_dir is not None:
            save_json_dict(merged, self.results_dir / "results.json")
            save_json_dict(self.last_metadata, self.results_dir / "metric_status.json")

        return merged

    def _get_pipeline(self, modality: str) -> EvalPipeline:
        """Build (or return the cached) Hydra-instantiated pipeline for `modality`."""
        if modality not in self._pipelines:
            config_dir = _resolve_config_dir()
            overrides = self._modality_overrides.get(modality, [])
            with initialize_config_dir(config_dir=str(config_dir), version_base=None):
                cfg = compose(config_name=modality, overrides=overrides)
            self._pipelines[modality] = instantiate(cfg.pipeline)
        return self._pipelines[modality]
