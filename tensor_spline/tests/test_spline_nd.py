"""Tests for spline_nd — N-dimensional Eisenstein spline layers."""

import numpy as np
import pytest

from tensor_spline.spline_nd import (
    EisensteinGrid, SplineLinearND, HierarchicalSplineND,
    benchmark_spline_nd, benchmark_hierarchical,
)


class TestEisensteinGrid:
    def test_2d_shape(self):
        g = EisensteinGrid(n_points=16, ndim=2)
        assert g.positions.shape == (16, 2)

    def test_3d_shape(self):
        g = EisensteinGrid(n_points=16, ndim=3)
        assert g.positions.shape == (16, 3)

    def test_4d_shape(self):
        g = EisensteinGrid(n_points=16, ndim=4)
        assert g.positions.shape == (16, 4)

    def test_first_point_is_origin(self):
        g = EisensteinGrid(n_points=7, ndim=2)
        assert np.allclose(g.positions[0], [0.0, 0.0], atol=1e-10)

    def test_normalised_to_unit_disk(self):
        for n in [7, 16, 25]:
            g = EisensteinGrid(n_points=n, ndim=2)
            norms = np.linalg.norm(g.positions, axis=1)
            assert norms.max() <= 1.0 + 1e-10

    def test_no_duplicates(self):
        g = EisensteinGrid(n_points=19, ndim=2)
        assert np.unique(g.positions, axis=0).shape[0] == 19

    def test_nearest_k(self):
        g = EisensteinGrid(n_points=16, ndim=2)
        dists, idx = g.nearest_k(np.array([0.0, 0.0]), k=4)
        assert dists.shape == (4,)
        assert idx.shape == (4,)
        # Origin should be closest
        assert dists[0] < 0.01

    def test_nearest_k_larger_than_n(self):
        g = EisensteinGrid(n_points=5, ndim=2)
        dists, idx = g.nearest_k(np.array([0.0, 0.0]), k=100)
        assert len(dists) == 5

    def test_invalid_ndim(self):
        with pytest.raises(ValueError):
            EisensteinGrid(n_points=16, ndim=5)

    def test_invalid_npoints(self):
        with pytest.raises(ValueError):
            EisensteinGrid(n_points=0, ndim=2)

    def test_repr(self):
        g = EisensteinGrid(n_points=16, ndim=3)
        assert "16" in repr(g) and "3" in repr(g)

    def test_tile_dict_roundtrip(self):
        g = EisensteinGrid(n_points=16, ndim=3)
        d = g.to_tile_dict()
        assert d["type"] == "eisenstein_grid"
        g2 = EisensteinGrid.from_tile_dict(d)
        assert g2.n_points == g.n_points
        assert g2.ndim == g.ndim
        np.testing.assert_allclose(g2.positions, g.positions)


class TestSplineLinearND:
    def test_basic_shape(self):
        layer = SplineLinearND(input_shape=(64,), output_shape=(32,), n_control_points=16)
        W = layer.materialize_weights()
        assert W.shape == (32, 64)

    def test_forward(self):
        layer = SplineLinearND(input_shape=(32,), output_shape=(16,), n_control_points=8)
        x = np.random.randn(4, 32)
        y = layer.forward(x)
        assert y.shape == (4, 16)

    def test_compression_ratio(self):
        layer = SplineLinearND(input_shape=(128,), output_shape=(64,), n_control_points=8)
        assert layer.compression_ratio() > 10

    def test_set_control_values(self):
        layer = SplineLinearND(input_shape=(32,), output_shape=(16,), n_control_points=8)
        new_vals = np.ones(8) * 0.5
        layer.set_control_values(new_vals)
        np.testing.assert_allclose(layer.control_values, new_vals)

    def test_num_params(self):
        layer = SplineLinearND(input_shape=(64,), output_shape=(32,), n_control_points=16)
        assert layer.num_params() == 16

    def test_3d_ndim(self):
        layer = SplineLinearND(input_shape=(32,), output_shape=(16,), n_control_points=8, ndim=3)
        W = layer.materialize_weights()
        assert W.shape == (16, 32)

    def test_4d_ndim(self):
        layer = SplineLinearND(input_shape=(32,), output_shape=(16,), n_control_points=8, ndim=4)
        W = layer.materialize_weights()
        assert W.shape == (16, 32)

    def test_tile_dict_roundtrip(self):
        layer = SplineLinearND(input_shape=(32,), output_shape=(16,), n_control_points=8)
        d = layer.to_tile_dict()
        assert d["type"] == "spline_linear_nd"
        layer2 = SplineLinearND.from_tile_dict(d)
        assert layer2.in_size == layer.in_size
        assert layer2.out_size == layer.out_size
        np.testing.assert_allclose(layer2.control_values, layer.control_values)

    def test_repr(self):
        layer = SplineLinearND(input_shape=(64,), output_shape=(32,), n_control_points=16)
        r = repr(layer)
        assert "64" in r and "32" in r


class TestHierarchicalSplineND:
    def test_basic_shape(self):
        layer = HierarchicalSplineND(
            input_shape=(128,), output_shape=(64,),
            coarse_pts=16, fine_pts=8, patch_sizes=32,
        )
        W = layer.materialize_weights()
        assert W.shape == (64, 128)

    def test_forward(self):
        layer = HierarchicalSplineND(
            input_shape=(64,), output_shape=(32,),
            coarse_pts=8, fine_pts=4, patch_sizes=16,
        )
        x = np.random.randn(4, 64)
        y = layer.forward(x)
        assert y.shape == (4, 32)

    def test_compression_ratio(self):
        layer = HierarchicalSplineND(
            input_shape=(128,), output_shape=(64,),
            coarse_pts=16, fine_pts=8, patch_sizes=32,
        )
        assert layer.compression_ratio() > 1.0

    def test_num_params(self):
        layer = HierarchicalSplineND(
            input_shape=(128,), output_shape=(64,),
            coarse_pts=16, fine_pts=8, patch_sizes=32,
        )
        expected = 16 + layer.total_patches * 8 + 1
        assert layer.num_params() == expected

    def test_tile_dict_roundtrip(self):
        layer = HierarchicalSplineND(
            input_shape=(64,), output_shape=(32,),
            coarse_pts=8, fine_pts=4, patch_sizes=16,
        )
        d = layer.to_tile_dict()
        assert d["type"] == "hierarchical_spline_nd"
        layer2 = HierarchicalSplineND.from_tile_dict(d)
        assert layer2.in_size == layer.in_size
        assert layer2.out_size == layer.out_size
        np.testing.assert_allclose(layer2.coarse_control, layer.coarse_control)
        np.testing.assert_allclose(layer2.fine_control, layer.fine_control)

    def test_repr(self):
        layer = HierarchicalSplineND(
            input_shape=(64,), output_shape=(32,),
            coarse_pts=8, fine_pts=4, patch_sizes=16,
        )
        r = repr(layer)
        assert "HierarchicalSplineND" in r


class TestBenchmark:
    def test_benchmark_spline_nd(self):
        result = benchmark_spline_nd(
            dimensions=[(64, 32)],
            n_control_points_list=[4, 8],
            ndim_values=[2],
        )
        assert "results" in result
        assert "summary" in result
        assert len(result["results"]) == 2

    def test_benchmark_hierarchical(self):
        result = benchmark_hierarchical(
            dimensions=[(64, 32)],
            coarse_pts=8, fine_pts=4, patch_size=16,
        )
        assert "results" in result
        assert len(result["results"]) == 1


# Optional PyTorch wrapper tests
try:
    import torch
    from tensor_spline.spline_nd import SplineLinearNDTorch

    class TestSplineLinearNDTorch:
        def test_output_shape(self):
            layer = SplineLinearNDTorch(
                input_shape=(32,), output_shape=(16,),
                n_control_points=8, ndim=2,
            )
            x = torch.randn(4, 32)
            y = layer(x)
            assert y.shape == (4, 16)

        def test_gradients(self):
            layer = SplineLinearNDTorch(
                input_shape=(32,), output_shape=(16,),
                n_control_points=8,
            )
            layer(torch.randn(2, 32)).sum().backward()
            assert layer.control_values.grad is not None

        def test_compression_ratio(self):
            layer = SplineLinearNDTorch(
                input_shape=(64,), output_shape=(32,),
                n_control_points=8, bias=False,
            )
            assert layer.compression_ratio() > 10

        def test_no_bias(self):
            layer = SplineLinearNDTorch(
                input_shape=(32,), output_shape=(16,),
                n_control_points=8, bias=False,
            )
            assert layer.bias is None
            assert layer(torch.randn(2, 32)).shape == (2, 16)
except ImportError:
    pass
