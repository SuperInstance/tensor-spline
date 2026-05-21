"""Tests for tensor_spline.low_rank — LowRankLinear, LowRankClassifier."""

import pytest
import torch
from tensor_spline.low_rank import LowRankLinear, LowRankClassifier


class TestLowRankLinear:
    def test_forward_shape(self):
        layer = LowRankLinear(32, 16, rank=4)
        x = torch.randn(4, 32)
        y = layer(x)
        assert y.shape == (4, 16)

    def test_low_rank(self):
        layer = LowRankLinear(64, 32, rank=4)
        params = sum(p.numel() for p in layer.parameters())
        assert params < 64 * 32  # fewer than dense


class TestLowRankClassifier:
    def test_forward_shape(self):
        model = LowRankClassifier(input_dim=8, hidden=16, num_classes=4, rank=8)
        x = torch.randn(4, 8)
        y = model(x)
        assert y.shape == (4, 4)


class TestHierarchicalSpline:
    def test_import(self):
        from tensor_spline.hierarchical_spline import HierarchicalSplineLinear
        assert HierarchicalSplineLinear is not None
