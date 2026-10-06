"""Metrics and losses for nucleus."""

try:
    from nucleus.metrics.atomistic_properties import (
        AtomisticPropertyMetrics,
        compute_atomistic_property_metrics,
    )
except ImportError:
    AtomisticPropertyMetrics = None
    compute_atomistic_property_metrics = None

try:
    from nucleus.metrics.losses import (
        DiffusionLoss,
        MaterialGeometryLoss,
        SequenceLoss,
    )
except ImportError:
    DiffusionLoss = None
    MaterialGeometryLoss = None
    SequenceLoss = None

try:
    from nucleus.metrics.molecule_generation import (
        POSEBUSTERS_KEYS,
        MoleculeGenerationMetrics,
        compute_molecule_generation_metrics,
    )
except ImportError:
    POSEBUSTERS_KEYS = ()
    MoleculeGenerationMetrics = None
    compute_molecule_generation_metrics = None

try:
    from nucleus.metrics.material_generation import (
        MaterialGenerationMetrics,
        compute_material_generation_metrics,
        compute_material_structure_fingerprints,
        compute_material_uniqueness,
    )
except ImportError:
    MaterialGenerationMetrics = None
    compute_material_generation_metrics = None
    compute_material_uniqueness = None
    compute_material_structure_fingerprints = None

try:
    from nucleus.metrics.structure_prediction import ZevalsStructurePredictionMetrics
except ImportError:
    ZevalsStructurePredictionMetrics = None

__all__ = [
    "AtomisticPropertyMetrics",
    "compute_atomistic_property_metrics",
    "DiffusionLoss",
    "MaterialGeometryLoss",
    "SequenceLoss",
    "MoleculeGenerationMetrics",
    "compute_molecule_generation_metrics",
    "POSEBUSTERS_KEYS",
    "MaterialGenerationMetrics",
    "compute_material_generation_metrics",
    "compute_material_uniqueness",
    "compute_material_structure_fingerprints",
    "ZevalsStructurePredictionMetrics",
]
