"""AFD metrics for small-molecule generation.

AFD is a distributional metric: score all generated records for a model in one
call, and keep the generated and reference sample counts matched for comparison.
Generated records should be torch-geometric ``Data`` objects or dicts with at
least ``z`` and ``pos``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

AFD_KEYS: tuple[str, ...] = (
    "afd_score",
    "afd_num_samples",
    "afd_ref_sample_size",
)

AFD_MIN_RECOMMENDED_SAMPLES = 500


def _load_afd_score_class():
    try:
        from zevals.atomistic.afd import AFDScore
    except ImportError as exc:
        raise ImportError(
            "AFD molecule metrics require torch_geometric and huggingface_hub. "
            "Install the `zevals` atomistic or molecules extras."
        ) from exc
    return AFDScore


def compute_afd_metrics(
    generated_records: Sequence[Any],
    *,
    scorer: Any | None = None,
    reference: str | Sequence[Any] = "qm9",
    data_modality: str = "molecules",
    model_name: str | None = None,
    batch_size: int = 256,
    device: str | None = None,
    feature_cache_dir: str | Path | None = None,
    ref_sample_size: int | None = None,
    match_reference_size: bool = True,
    seed: int | None = 0,
    **featurizer_kwargs: Any,
) -> dict[str, float]:
    """Compute molecule AFD against a canonical or custom reference.

    Args:
        generated_records: Generated atomistic records. Each record must carry
            ``z`` and ``pos``.
        scorer: Optional reusable ``AFDScore`` instance.
        reference: Canonical reference name, usually ``"qm9"``, or custom
            reference records.
        data_modality: Modality used to select a CT checkpoint for custom
            references when ``model_name`` is omitted.
        model_name: Optional explicit CT checkpoint name.
        batch_size: CT featurization batch size.
        device: CT featurization device.
        feature_cache_dir: Optional cache for custom-reference embeddings.
        ref_sample_size: Explicit reference subsample size.
        match_reference_size: If True, default ``ref_sample_size`` to the
            generated record count, following the AFD guide.
        seed: Reference subsampling seed.
        **featurizer_kwargs: Forwarded to ``AFDScore.__call__``.

    Returns:
        Dict containing ``afd_score``, ``afd_num_samples``, and
        ``afd_ref_sample_size``.
    """
    records = list(generated_records)
    if not records:
        return {key: 0.0 for key in AFD_KEYS}

    if scorer is None:
        AFDScore = _load_afd_score_class()
        try:
            scorer = AFDScore(
                reference,
                data_modality=data_modality,
                model_name=model_name,
                batch_size=batch_size,
                device=device,
                feature_cache_dir=feature_cache_dir,
            )
        except ImportError as exc:
            raise ImportError(
                "AFD molecule metrics require torch_geometric and huggingface_hub. "
                "Install the `zevals` atomistic or molecules extras."
            ) from exc

    resolved_ref_sample_size = ref_sample_size
    if resolved_ref_sample_size is None and match_reference_size:
        resolved_ref_sample_size = len(records)

    score_kwargs = dict(featurizer_kwargs)
    if resolved_ref_sample_size is not None:
        score_kwargs["ref_sample_size"] = resolved_ref_sample_size
        score_kwargs["seed"] = seed

    score = float(scorer(records, **score_kwargs))
    return {
        "afd_score": score,
        "afd_num_samples": float(len(records)),
        "afd_ref_sample_size": float(resolved_ref_sample_size or 0),
    }


def __getattr__(name: str):
    """Lazily expose the core scorer without importing heavy deps at module load."""
    if name == "AFDScore":
        return _load_afd_score_class()
    raise AttributeError(name)


__all__ = [
    "AFD_KEYS",
    "AFD_MIN_RECOMMENDED_SAMPLES",
    "compute_afd_metrics",
]
