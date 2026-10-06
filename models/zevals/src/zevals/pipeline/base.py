"""Abstract evaluation pipeline.

All domain-specific pipelines (molecules, ligands, proteins, materials) inherit
from EvalPipeline and implement the same three-method interface:

  run(samples)      → EvalResults   (per-sample metrics)
  aggregate(results) → dict          (dataset-level summary statistics)
  report(results)   → str            (human-readable summary)

Oracle backends are injected at construction time (not hard-coded), so the
pipeline can be reconfigured entirely through Hydra YAML without touching
Python code.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


def safe_update(
    agg: dict[str, float],
    label: str,
    fn: Callable[..., dict],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Merge `fn(*args, **kwargs)`'s output into `agg`, tolerating failure.

    Every domain pipeline's `run()` computes several independent metrics
    (designability, diversity, novelty, ...), each depending on its own
    oracle/external binary. A single missing binary or oracle failure
    shouldn't take down every other metric — especially for a call meant to
    run unattended inside a training loop. On failure, `label`'s metrics are
    simply absent from `agg` (not filled with NaN placeholders, since we
    don't know their keys without having run `fn`), and a warning is logged.
    """
    try:
        agg.update(fn(*args, **kwargs))
    except Exception:
        logger.warning("Skipping %s metrics due to an error", label, exc_info=True)


@dataclass
class EvalResults:
    """Container for per-run evaluation outputs."""

    per_sample: list[dict[str, Any]] = field(default_factory=list)
    aggregate: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class EvalPipeline(ABC):
    """Abstract base for domain evaluation pipelines.

    Subclasses declare which oracles they need in __init__ and implement
    run() / aggregate() for their domain. Oracles are passed as constructor
    arguments so Hydra can instantiate and inject them via _target_.
    """

    @abstractmethod
    def run(self, samples: Any) -> EvalResults:
        """Evaluate a batch of generated samples.

        Args:
            samples: Domain-specific input (list of SMILES, structure file
                paths — `.cif`/`.pdb`, etc.).

        Returns:
            EvalResults with per-sample metrics and an aggregate summary.
        """

    @abstractmethod
    def aggregate(self, per_sample: list[dict[str, Any]]) -> dict[str, float]:
        """Compute dataset-level statistics from per-sample metric dicts.

        Args:
            per_sample: Output of run().per_sample.

        Returns:
            Dict of metric_name → scalar value.
        """

    def report(self, results: EvalResults) -> str:
        """Format a human-readable summary of results."""
        lines = ["=== Eval Summary ==="]
        for k, v in results.aggregate.items():
            lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        return "\n".join(lines)
