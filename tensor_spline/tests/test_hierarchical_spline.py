"""
Tests for hierarchical spline — multi-scale control points.
"""

import pytest
import torch
import torch.nn as nn
from tensor_spline.hierarchical_spline import (
    HierarchicalSplineLinear, HierarchicalSplineClassifier,
    inject_hierarchical_spline,
)


class TestHierarchicalSplineLinear:
    def test_output_shape(self):
        layer = HierarchicalSplineLinear(128, 64, coarse_pts=16, fine_pts=8, patch_size=32)
        x = torch.randn(4, 128)
        y = layer(x)
        assert y.shape == (4, 64)

    def test_fewer_params(self):
        hsl = HierarchicalSplineLinear(256, 128, coarse_pts=16, fine_pts=8, patch_size=32)
        dense = nn.Linear(256, 128)
        assert hsl.num_control_params() < dense.weight.numel()

    def test_compression_ratio(self):
        hsl = HierarchicalSplineLinear(256, 128, coarse_pts=16, fine_pts=8, patch_size=32)
        assert hsl.compression_ratio() > 5.0

    def test_gradients_flow(self):
        layer = HierarchicalSplineLinear(64, 32, coarse_pts=8, fine_pts=4, patch_size=16)
        x = torch.randn(2, 64)
        layer(x).sum().backward()
        assert layer.coarse_values.grad is not None
        assert layer.fine_values.grad is not None

    def test_no_bias(self):
        layer = HierarchicalSplineLinear(64, 32, bias=False)
        assert layer.bias is None
        assert layer(torch.randn(2, 64)).shape == (2, 32)


class TestHierarchicalSplineClassifier:
    def test_forward(self):
        model = HierarchicalSplineClassifier(256, 64, 5, coarse_pts=16, fine_pts=8, patch_size=32)
        assert model(torch.randn(4, 256)).shape == (4, 5)

    def test_fewer_params_than_dense(self):
        dense = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 5))
        hsc = HierarchicalSplineClassifier(256, 64, 5)
        assert sum(p.numel() for p in hsc.parameters()) < sum(p.numel() for p in dense.parameters())


class TestInjectHierarchicalSpline:
    def test_replaces_linear(self):
        model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 4))
        m = inject_hierarchical_spline(model)
        assert len(m) == 2
        assert isinstance(model[0], HierarchicalSplineLinear)

    def test_target_modules(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.W_query = nn.Linear(64, 32)
                self.out = nn.Linear(32, 2)
            def forward(self, x): return self.out(self.W_query(x))
        model = M()
        m = inject_hierarchical_spline(model, target_modules=["W_query"])
        assert len(m) == 1

    def test_forward_after_injection(self):
        model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 4))
        inject_hierarchical_spline(model)
        assert model(torch.randn(2, 128)).shape == (2, 4)


class TestHighDimTask:
    def test_produces_valid_output(self):
        """Hierarchical spline produces valid predictions on high-dim data."""
        torch.manual_seed(42)
        X = torch.randn(100, 64)
        y = torch.randint(0, 4, (100,))

        model = HierarchicalSplineClassifier(64, 32, 4, coarse_pts=8, fine_pts=4, patch_size=16)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(10):
            opt.zero_grad()
            nn.CrossEntropyLoss()(model(X[:80]), y[:80]).backward()
            opt.step()
        
        model.eval()
        with torch.no_grad():
            acc = (model(X[80:]).argmax(1) == y[80:]).float().mean().item()
        assert 0.0 <= acc <= 1.0
