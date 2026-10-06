"""RFD3-inspired attention kernels for zatom."""

try:
    from nucleus.kernels.evo_attention import TritonEvoformer
except ImportError:  # pragma: no cover
    TritonEvoformer = None

__all__ = [
    "TritonEvoformer",
]
