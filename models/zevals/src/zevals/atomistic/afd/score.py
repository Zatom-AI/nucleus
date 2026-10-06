"""
Fréchet distance on CT feature distributions: AFD (Atomistic Fréchet Distance).

AFD mirrors the closed-form Wasserstein-2-between-Gaussians used by image FID,
but uses ``mol_emb`` from a chemistry-pretrained Conditional Equivariant
Transformer instead of Inception-v3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

_HF_REPO = "Ty-Perez/AtomisticEval"
_REFERENCE_FEATURES = {
    "qm9": "qm9_features.pt",
    "mp20": "mp20_features.pt",
}


def torch_sqrtm(matrix: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Return the principal matrix square root using Torch operations."""
    if isinstance(matrix, torch.Tensor):
        if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError(f"matrix must be square, got shape {tuple(matrix.shape)}")

        if matrix.dtype in (
            torch.complex64,
            torch.float16,
            torch.float32,
            torch.bfloat16,
        ):
            work = matrix.to(dtype=torch.complex64)
        else:
            work = matrix.to(dtype=torch.complex128)

        eigvals, eigvecs = torch.linalg.eig(work)
        sqrt_diag = torch.diag_embed(torch.sqrt(eigvals))
        return eigvecs @ sqrt_diag @ torch.linalg.inv(eigvecs)

    matrix = np.asarray(matrix)
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"matrix must be square, got shape {matrix.shape}")

    work = torch.as_tensor(matrix, dtype=torch.complex128)
    eigvals, eigvecs = torch.linalg.eig(work)
    sqrt_diag = torch.diag_embed(torch.sqrt(eigvals))
    return (eigvecs @ sqrt_diag @ torch.linalg.inv(eigvecs)).cpu().numpy()


def gaussian_stats(
    features: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mean, covariance)`` of a ``[N, D]`` feature tensor or array."""
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")

    mu = features.mean(axis=0)
    cov = np.cov(features, rowvar=False, ddof=1)
    return mu, cov


def frechet_distance(
    mu_a: np.ndarray,
    cov_a: np.ndarray,
    mu_b: np.ndarray,
    cov_b: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Closed-form squared Fréchet distance between two Gaussians."""
    mu_a = np.asarray(mu_a, dtype=np.float64)
    mu_b = np.asarray(mu_b, dtype=np.float64)
    cov_a = np.asarray(cov_a, dtype=np.float64)
    cov_b = np.asarray(cov_b, dtype=np.float64)

    diff = mu_a - mu_b
    offset = eps * np.eye(cov_a.shape[0], dtype=np.float64)
    covmean = torch_sqrtm((cov_a + offset).dot(cov_b + offset))

    if np.iscomplexobj(covmean):
        max_imag = np.max(np.abs(covmean.imag))
        if max_imag > 1e-3:
            import warnings

            warnings.warn(
                f"sqrtm returned non-trivial imaginary component "
                f"(max |imag| = {max_imag:.3e}); using real part only.",
                RuntimeWarning,
            )
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(cov_a) + np.trace(cov_b)
    fid -= 2.0 * np.trace(covmean)
    return float(max(fid, 0.0))


def afd_score(
    features_ref: np.ndarray | torch.Tensor,
    features_gen: np.ndarray | torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """Compute AFD from reference and generated feature matrices."""
    mu_r, cov_r = gaussian_stats(features_ref)
    mu_g, cov_g = gaussian_stats(features_gen)
    return frechet_distance(mu_r, cov_r, mu_g, cov_g, eps=eps)


def _get_ref_features(name: str) -> tuple[torch.Tensor, Any]:
    """Download and load canonical reference embeddings from HuggingFace."""
    name = name.lower()
    if name not in _REFERENCE_FEATURES:
        raise ValueError(
            f"Unrecognized reference dataset name {name!r}. "
            f"Available options are: {set(_REFERENCE_FEATURES)}"
        )

    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=_HF_REPO,
        filename=_REFERENCE_FEATURES[name],
        repo_type="dataset",
    )
    features, ids = torch.load(local_path, map_location="cpu", weights_only=False)
    return features, ids


class AFDScore:
    """End-to-end AFD scorer with a reusable CT feature extractor.

    Args:
        ref_dataset: Canonical reference name (``"qm9"`` or ``"mp20"``), or a
            sequence of atomistic records to featurize as the reference.
        data_modality: Required for custom reference records when ``model_name``
            is omitted. Use ``"molecules"`` or ``"materials"``.
        model_name: Optional explicit CT checkpoint name.
        batch_size: Batch size for CT featurization.
        device: Device used for CT featurization.
        feature_cache_dir: Optional directory for custom-reference embeddings.
            Ignored for canonical references, which use precomputed embeddings.
    """

    def __init__(
        self,
        ref_dataset,
        data_modality: str | None = None,
        model_name: str | None = None,
        batch_size: int = 256,
        device: str | torch.device | None = "cuda:0",
        feature_cache_dir: str | Path | None = None,
    ) -> None:
        from zevals.atomistic.afd.features import FeatureExtractor, _get_model_name

        if isinstance(ref_dataset, str):
            ref_dataset = ref_dataset.lower()
            self.ref_features, self.ref_ids = _get_ref_features(ref_dataset)
            model_name = (
                _get_model_name(ref_dataset) if model_name is None else model_name
            )
            self.featurizer = FeatureExtractor(
                model_name=model_name,
                batch_size=batch_size,
                device=device,
            )
            return

        if data_modality is None and model_name is None:
            raise ValueError(
                "If ref_dataset is custom records, either data_modality or "
                "model_name must be provided to select the CT checkpoint."
            )
        if model_name is None:
            model_name = _get_model_name(data_modality)

        self.ref_ids = None
        self.featurizer = FeatureExtractor(
            model_name=model_name,
            batch_size=batch_size,
            device=device,
        )
        self.ref_features = self._load_or_compute_ref_features(
            ref_dataset,
            model_name=model_name,
            feature_cache_dir=feature_cache_dir,
        )

    def _load_or_compute_ref_features(
        self,
        ref_dataset,
        *,
        model_name: str,
        feature_cache_dir: str | Path | None,
    ) -> torch.Tensor:
        if feature_cache_dir is None:
            return self.featurizer(ref_dataset)

        cache_dir = Path(feature_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_model_name = model_name.replace("/", "__")
        cache_path = cache_dir / f"{safe_model_name}_reference_features.pt"
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu", weights_only=False)

        ref_features = self.featurizer(ref_dataset)
        torch.save(ref_features, cache_path)
        return ref_features

    def sample_ref(self, size: int, seed: int | None = None) -> torch.Tensor:
        """Sample reference features for matched-N AFD scoring."""
        if size > len(self.ref_features):
            raise ValueError(
                f"Requested sample size {size} exceeds total number of reference "
                f"features {len(self.ref_features)}."
            )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.ref_features.device).manual_seed(
                seed
            )
        indices = torch.randperm(
            len(self.ref_features),
            generator=generator,
            device=self.ref_features.device,
        )[:size]
        return self.ref_features[indices]

    def embeddings(self, dataset, **kwargs) -> torch.Tensor:
        """Return CT graph-level embeddings for atomistic records."""
        return self.featurizer(dataset, **kwargs)

    def __call__(self, dataset, **kwargs) -> float:
        """Compute AFD between the stored reference and generated records."""
        ref_sample_size = kwargs.pop("ref_sample_size", None)
        if ref_sample_size is not None:
            ref_feats = self.sample_ref(ref_sample_size, seed=kwargs.pop("seed", None))
        else:
            ref_feats = self.ref_features

        target_feats = self.embeddings(dataset, **kwargs)
        return afd_score(ref_feats, target_feats)


__all__ = [
    "AFDScore",
    "afd_score",
    "frechet_distance",
    "gaussian_stats",
    "torch_sqrtm",
]
