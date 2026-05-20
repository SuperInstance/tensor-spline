"""SuperInstance mesh registration for tensor-spline."""

def register_tensor_spline(registry):
    """Register tensor-spline capabilities with the mesh."""
    from tensor_spline.spline import SplineLinear
    from tensor_spline.low_rank import LowRankLinear

    registry.register("compressors", "spline-linear", SplineLinear)
    registry.register("compressors", "low-rank", LowRankLinear)

    # Utility functions
    from tensor_spline.spline import inject_spline, compression_ratio
    from tensor_spline.low_rank import recommend_variant
    registry.register("compressors", "inject", lambda: inject_spline)
    registry.register("compressors", "measure", lambda: compression_ratio)
    registry.register("compressors", "recommend", lambda: recommend_variant)
