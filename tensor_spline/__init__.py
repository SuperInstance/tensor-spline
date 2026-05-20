"""
tensor-spline — Compressed neural network layers.

Novel weight parameterization using Eisenstein lattice control points,
plus standard low-rank factorization. Language-agnostic, framework-agnostic.

Layers:
  SplineLinear: weights interpolated from hex lattice control points
  LowRankLinear: learned U@V factorization
  HierarchicalSplineLinear: multi-scale coarse+fine interpolation

Utilities:
  inject_spline: replace nn.Linear with SplineLinear
  inject_low_rank: replace nn.Linear with LowRankLinear
  recommend_variant: task-aware compression selection
"""

from .spline import SplineLinear, inject_spline, compression_ratio, EisensteinLattice
from .low_rank import LowRankLinear, LowRankClassifier, inject_low_rank, recommend_variant, VARIANT_GUIDE
from .hierarchical_spline import HierarchicalSplineLinear, HierarchicalSplineClassifier, inject_hierarchical_spline

__all__ = [
    "SplineLinear",
    "EisensteinLattice",
    "inject_spline",
    "compression_ratio",
    "LowRankLinear",
    "LowRankClassifier",
    "inject_low_rank",
    "recommend_variant",
    "VARIANT_GUIDE",
    "HierarchicalSplineLinear",
    "HierarchicalSplineClassifier",
    "inject_hierarchical_spline",
    "__version__",
]

__version__ = "1.0.0"

# Auto-register with mesh if plato-core is installed
try:
    from plato_core.registry import registry
    from tensor_spline.mesh import register_tensor_spline
    register_tensor_spline(registry)
except ImportError:
    pass  # Standalone mode
