"""Tests for tensor_spline.spline — EisensteinLattice, SplineLinear, inject_spline."""

import pytest
import torch
from tensor_spline.spline import EisensteinLattice, SplineLinear


class TestEisensteinLattice:
    def test_basic(self):
        lattice = EisensteinLattice(7)
        assert lattice.n_points == 7
        pos = lattice.positions()
        assert pos.shape == (7, 2)

    def test_origin_first(self):
        lattice = EisensteinLattice(7)
        pos = lattice.positions()
        assert torch.allclose(pos[0], torch.zeros(2), atol=1e-6)

    def test_single_point(self):
        lattice = EisensteinLattice(1)
        pos = lattice.positions()
        assert pos.shape == (1, 2)

    def test_invalid_n_points(self):
        with pytest.raises(ValueError):
            EisensteinLattice(0)

    def test_large_lattice(self):
        lattice = EisensteinLattice(64)
        assert lattice.positions().shape == (64, 2)


class TestSplineLinear:
    def test_basic(self):
        layer = SplineLinear(32, 16, n_control_points=8)
        x = torch.randn(4, 32)
        y = layer(x)
        assert y.shape == (4, 16)

    def test_compression(self):
        layer = SplineLinear(512, 512, n_control_points=16)
        ratio = layer.compression_ratio()
        assert ratio > 1.0

    def test_internal_materialization(self):
        layer = SplineLinear(32, 16, n_control_points=8)
        W = layer._materialize()
        assert W.shape == (16, 32)

    def test_num_params(self):
        layer = SplineLinear(32, 16, n_control_points=8)
        n = layer.num_trainable_params()
        assert n > 0

    def test_dense_params(self):
        layer = SplineLinear(32, 16, n_control_points=8)
        n = layer.num_equivalent_dense_params()
        assert n == 528  # 32*16 + bias


class TestInjectSpline:
    def test_inject(self):
        from tensor_spline.spline import inject_spline
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4),
        )
        inject_spline(model, n_control_points=4)
        has_spline = any(isinstance(m, SplineLinear) for m in model.modules())
        assert has_spline
