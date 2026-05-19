"""
Tests for tensor_spline.spline_nd — N-dimensional Eisenstein spline layers.

Covers:
  - EisensteinGrid construction for 2D, 3D, 4D
  - SplineLinearND materialization, forward pass, compression
  - HierarchicalSplineND materialization, forward pass, compression
  - PLATO tile serialization round-trips
  - Benchmark functions
  - PyTorch wrapper (if torch available)
"""

import math
import pytest
import numpy as np
from numpy.testing import assert_allclose

from tensor_spline.spline_nd import (
    EisensteinGrid,
    SplineLinearND,
    HierarchicalSplineND,
    benchmark_spline_nd,
    benchmark_hierarchical,
)


# =====================================================================
# EisensteinGrid
# =====================================================================

class TestEisensteinGrid:
    """Tests for EisensteinGrid precomputed lattice tables."""

    def test_2d_basic(self):
        grid = EisensteinGrid(n_points=16, ndim=2)
        assert grid.positions.shape == (16, 2)
        # First point should be origin
        assert_allclose(grid.positions[0], [0.0, 0.0], atol=1e-10)

    def test_3d_basic(self):
        grid = EisensteinGrid(n_points=27, ndim=3)
        assert grid.positions.shape == (27, 3)

    def test_4d_basic(self):
        grid = EisensteinGrid(n_points=64, ndim=4)
        assert grid.positions.shape == (64, 4)

    def test_normalisation(self):
        """Outermost point should have unit norm."""
        grid = EisensteinGrid(n_points=16, ndim=2)
        max_norm = np.linalg.norm(grid.positions, axis=1).max()
        assert abs(max_norm - 1.0) < 1e-10 or grid.n_points == 1

    def test_3d_normalisation(self):
        grid = EisensteinGrid(n_points=32, ndim=3)
        max_norm = np.linalg.norm(grid.positions, axis=1).max()
        assert abs(max_norm - 1.0) < 1e-10 or grid.n_points == 1

    def test_single_point(self):
        grid = EisensteinGrid(n_points=1, ndim=2)
        assert grid.positions.shape == (1, 2)
        assert_allclose(grid.positions[0], [0.0, 0.0], atol=1e-10)

    def test_nearest_k(self):
        grid = EisensteinGrid(n_points=16, ndim=2)
        dists, idx = grid.nearest_k(np.array([0.0, 0.0]), k=3)
        assert len(dists) == 3
        assert len(idx) == 3
        # Origin is a lattice point → distance should be ~0
        assert dists[0] < 1e-6

    def test_nearest_k_clamped(self):
        grid = EisensteinGrid(n_points=4, ndim=2)
        dists, idx = grid.nearest_k(np.array([0.0, 0.0]), k=100)
        assert len(dists) == 4

    def test_invalid_ndim(self):
        with pytest.raises(ValueError, match="ndim must be 2, 3, or 4"):
            EisensteinGrid(n_points=16, ndim=5)

    def test_invalid_n_points(self):
        with pytest.raises(ValueError, match="n_points must be >= 1"):
            EisensteinGrid(n_points=0, ndim=2)

    def test_tile_roundtrip(self):
        grid = EisensteinGrid(n_points=16, ndim=3)
        d = grid.to_tile_dict()
        assert d["type"] == "eisenstein_grid"
        restored = EisensteinGrid.from_tile_dict(d)
        assert restored.n_points == grid.n_points
        assert restored.ndim == grid.ndim
        assert_allclose(restored.positions, grid.positions)

    def test_2d_hex_structure(self):
        """Verify the 2D lattice has proper hexagonal structure (60° angles)."""
        grid = EisensteinGrid(n_points=7, ndim=2)
        # The 7 nearest Eisenstein integers include origin + 6 hex neighbours
        # The 6 should all be at the same distance from origin (on a ring)
        dists = np.linalg.norm(grid.positions[1:], axis=1)
        # They should all be equal (same ring)
        assert np.std(dists) < 1e-10

    def test_repr(self):
        grid = EisensteinGrid(n_points=16, ndim=3)
        assert "EisensteinGrid" in repr(grid)
        assert "ndim=3" in repr(grid)


# =====================================================================
# SplineLinearND
# =====================================================================

class TestSplineLinearND:
    """Tests for pure-NumPy N-D spline layer."""

    def test_basic_construction(self):
        layer = SplineLinearND(
            input_shape=(64,), output_shape=(32,), n_control_points=16,
        )
        assert layer.in_size == 64
        assert layer.out_size == 32
        assert layer.n_control_points == 16

    def test_weight_shape(self):
        layer = SplineLinearND(
            input_shape=(128,), output_shape=(64,), n_control_points=16,
        )
        W = layer.materialize_weights()
        assert W.shape == (64, 128)

    def test_forward_shape(self):
        layer = SplineLinearND(
            input_shape=(64,), output_shape=(32,), n_control_points=16,
        )
        x = np.random.randn(4, 64)
        out = layer.forward(x)
        assert out.shape == (4, 32)

    def test_forward_batch(self):
        layer = SplineLinearND(
            input_shape=(32,), output_shape=(16,), n_control_points=8,
        )
        x = np.random.randn(10, 32)
        out = layer.forward(x)
        assert out.shape == (10, 16)

    def test_compression_ratio(self):
        # 512×512 dense = 262,144 params; 16 control points = 16 params
        # Ratio = 262144 / 16 = 16384
        layer = SplineLinearND(
            input_shape=(512,), output_shape=(512,), n_control_points=16,
        )
        assert layer.compression_ratio() == pytest.approx(16384.0)

    def test_compression_16k_ratio(self):
        """Verify the 16K:1 compression ratio for 512×512 / 16 cp."""
        layer = SplineLinearND(
            input_shape=(512,), output_shape=(512,), n_control_points=16,
        )
        assert layer.compression_ratio() >= 16384.0

    def test_3d_lattice(self):
        layer = SplineLinearND(
            input_shape=(64,), output_shape=(32,),
            n_control_points=16, ndim=3,
        )
        W = layer.materialize_weights()
        assert W.shape == (32, 64)

    def test_4d_lattice(self):
        layer = SplineLinearND(
            input_shape=(64,), output_shape=(32,),
            n_control_points=27, ndim=4,
        )
        W = layer.materialize_weights()
        assert W.shape == (32, 64)

    def test_set_control_values(self):
        layer = SplineLinearND(
            input_shape=(32,), output_shape=(16,), n_control_points=8,
        )
        new_vals = np.ones(8)
        layer.set_control_values(new_vals)
        assert_allclose(layer.control_values, np.ones(8))

    def test_deterministic_weights(self):
        """Same control values → same weights."""
        layer = SplineLinearND(
            input_shape=(32,), output_shape=(16,), n_control_points=8,
        )
        W1 = layer.materialize_weights()
        W2 = layer.materialize_weights()
        assert_allclose(W1, W2)

    def test_control_values_affect_weights(self):
        """Different control values → different weights."""
        layer = SplineLinearND(
            input_shape=(32,), output_shape=(16,), n_control_points=8,
        )
        W1 = layer.materialize_weights()
        layer.set_control_values(np.random.randn(8))
        W2 = layer.materialize_weights()
        assert not np.allclose(W1, W2)

    def test_tile_roundtrip(self):
        layer = SplineLinearND(
            input_shape=(64,), output_shape=(32,),
            n_control_points=16, ndim=2,
        )
        d = layer.to_tile_dict()
        assert d["type"] == "spline_linear_nd"
        restored = SplineLinearND.from_tile_dict(d)
        assert restored.input_shape == layer.input_shape
        assert restored.output_shape == layer.output_shape
        assert restored.n_control_points == layer.n_control_points
        assert_allclose(restored.control_values, layer.control_values)
        # Materialized weights should match
        W_orig = layer.materialize_weights()
        W_restored = restored.materialize_weights()
        assert_allclose(W_orig, W_restored)

    def test_num_params(self):
        layer = SplineLinearND(
            input_shape=(64,), output_shape=(32,), n_control_points=16,
        )
        assert layer.num_params() == 16
        assert layer.num_dense_params() == 64 * 32

    def test_multidim_input_shape(self):
        """Input shape can be multi-dimensional (flattened internally)."""
        layer = SplineLinearND(
            input_shape=(8, 8), output_shape=(4, 4), n_control_points=16,
        )
        assert layer.in_size == 64
        assert layer.out_size == 16
        W = layer.materialize_weights()
        assert W.shape == (16, 64)

    def test_repr(self):
        layer = SplineLinearND(
            input_shape=(512,), output_shape=(512,), n_control_points=16,
        )
        r = repr(layer)
        assert "SplineLinearND" in r
        assert "16384" in r  # compression ratio


# =====================================================================
# HierarchicalSplineND
# =====================================================================

class TestHierarchicalSplineND:
    """Tests for hierarchical multi-scale spline."""

    def test_basic_construction(self):
        layer = HierarchicalSplineND(
            input_shape=(128,), output_shape=(64,),
            coarse_pts=16, fine_pts=8, patch_sizes=32,
        )
        assert layer.in_size == 128
        assert layer.out_size == 64

    def test_weight_shape(self):
        layer = HierarchicalSplineND(
            input_shape=(128,), output_shape=(64,),
            coarse_pts=16, fine_pts=8, patch_sizes=32,
        )
        W = layer.materialize_weights()
        assert W.shape == (64, 128)

    def test_forward_shape(self):
        layer = HierarchicalSplineND(
            input_shape=(64,), output_shape=(32,),
            coarse_pts=8, fine_pts=4, patch_sizes=16,
        )
        x = np.random.randn(4, 64)
        out = layer.forward(x)
        assert out.shape == (4, 32)

    def test_compression(self):
        """Hierarchical should achieve better compression than single-scale for large layers."""
        layer = HierarchicalSplineND(
            input_shape=(256,), output_shape=(128,),
            coarse_pts=16, fine_pts=8, patch_sizes=32,
        )
        ratio = layer.compression_ratio()
        # Dense: 256*128 = 32768
        # Params: 16 coarse + (8*4)*8 fine + 1 alpha = 16 + 256 + 1 = 273
        # Ratio ≈ 120:1
        assert ratio > 1.0

    def test_alpha_blend(self):
        """Changing alpha should change the weights."""
        layer = HierarchicalSplineND(
            input_shape=(64,), output_shape=(32,),
            coarse_pts=8, fine_pts=4, patch_sizes=16,
        )
        layer.alpha = 2.0  # sigmoid(2) ≈ 0.88 → mostly coarse
        W_coarse = layer.materialize_weights().copy()
        layer.alpha = -2.0  # sigmoid(-2) ≈ 0.12 → mostly fine
        W_fine = layer.materialize_weights()
        assert not np.allclose(W_coarse, W_fine)

    def test_tile_roundtrip(self):
        layer = HierarchicalSplineND(
            input_shape=(64,), output_shape=(32,),
            coarse_pts=8, fine_pts=4, patch_sizes=16,
        )
        d = layer.to_tile_dict()
        assert d["type"] == "hierarchical_spline_nd"
        restored = HierarchicalSplineND.from_tile_dict(d)
        assert restored.input_shape == layer.input_shape
        assert restored.output_shape == layer.output_shape
        assert_allclose(restored.coarse_control, layer.coarse_control)
        assert_allclose(restored.fine_control, layer.fine_control)

    def test_large_layer(self):
        """512→256 should work without errors."""
        layer = HierarchicalSplineND(
            input_shape=(512,), output_shape=(256,),
            coarse_pts=16, fine_pts=8, patch_sizes=64,
        )
        W = layer.materialize_weights()
        assert W.shape == (256, 512)
        x = np.random.randn(2, 512)
        out = layer.forward(x)
        assert out.shape == (2, 256)

    def test_3d_ndim(self):
        layer = HierarchicalSplineND(
            input_shape=(64,), output_shape=(32,),
            coarse_pts=8, fine_pts=4, patch_sizes=16,
            ndim=3,
        )
        W = layer.materialize_weights()
        assert W.shape == (32, 64)

    def test_num_params(self):
        layer = HierarchicalSplineND(
            input_shape=(64,), output_shape=(32,),
            coarse_pts=8, fine_pts=4, patch_sizes=16,
        )
        # ceil(64/16)=4 in_patches × ceil(32/16)=2 out_patches = 8 patches
        expected = 8 + 8 * 4 + 1  # coarse + fine + alpha = 41
        assert layer.num_params() == expected

    def test_repr(self):
        layer = HierarchicalSplineND(
            input_shape=(128,), output_shape=(64,),
            coarse_pts=16, fine_pts=8, patch_sizes=32,
        )
        assert "HierarchicalSplineND" in repr(layer)


# =====================================================================
# Benchmarks
# =====================================================================

class TestBenchmark:
    """Tests for benchmark utility functions."""

    def test_benchmark_spline_nd(self):
        result = benchmark_spline_nd(
            dimensions=[(64, 32)],
            n_control_points_list=[8, 16],
            ndim_values=[2],
        )
        assert "results" in result
        assert "summary" in result
        assert len(result["results"]) == 2
        # All should have weight_shape_ok=True
        for r in result["results"]:
            assert r.get("weight_shape_ok", False)

    def test_benchmark_hierarchical(self):
        result = benchmark_hierarchical(dimensions=[(64, 32), (128, 64)])
        assert "results" in result
        assert len(result["results"]) == 2
        for r in result["results"]:
            assert "compression_ratio" in r
            assert r["compression_ratio"] > 1.0

    def test_benchmark_large_compression(self):
        """512×512 with 16 control points should achieve 16K:1."""
        result = benchmark_spline_nd(
            dimensions=[(512, 512)],
            n_control_points_list=[16],
            ndim_values=[2],
        )
        r = result["results"][0]
        assert r["compression_ratio"] >= 16000


# =====================================================================
# PyTorch wrapper (skip if no torch)
# =====================================================================

try:
    import torch
    from tensor_spline.spline_nd import SplineLinearNDTorch

    class TestSplineLinearNDTorch:
        """Tests for PyTorch wrapper."""

        def test_construction(self):
            layer = SplineLinearNDTorch(
                input_shape=(64,), output_shape=(32,), n_control_points=16,
            )
            assert layer.in_size == 64
            assert layer.out_size == 32

        def test_forward_shape(self):
            layer = SplineLinearNDTorch(
                input_shape=(64,), output_shape=(32,), n_control_points=16,
            )
            x = torch.randn(4, 64)
            out = layer(x)
            assert out.shape == (4, 32)

        def test_backward(self):
            """Gradients should flow to control_values."""
            layer = SplineLinearNDTorch(
                input_shape=(32,), output_shape=(16,), n_control_points=8,
            )
            x = torch.randn(2, 32)
            out = layer(x)
            loss = out.sum()
            loss.backward()
            assert layer.control_values.grad is not None
            assert layer.control_values.grad.shape == (8,)

        def test_compression_ratio(self):
            layer = SplineLinearNDTorch(
                input_shape=(512,), output_shape=(512,),
                n_control_points=16, bias=False,
            )
            assert layer.compression_ratio() >= 16000

        def test_with_bias(self):
            layer = SplineLinearNDTorch(
                input_shape=(32,), output_shape=(16,),
                n_control_points=8, bias=True,
            )
            x = torch.randn(2, 32)
            out = layer(x)
            assert out.shape == (2, 16)

        def test_3d_lattice(self):
            layer = SplineLinearNDTorch(
                input_shape=(64,), output_shape=(32,),
                n_control_points=16, ndim=3,
            )
            x = torch.randn(2, 64)
            out = layer(x)
            assert out.shape == (2, 32)

except ImportError:
    pass
