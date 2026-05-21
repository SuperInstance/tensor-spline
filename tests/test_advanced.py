"""Advanced tests for tensor-spline — hierarchical spline, low-rank edge cases, injection, recommendations."""

import pytest
import torch
import torch.nn as nn
import math


# ─── LowRankLinear Advanced ───────────────────────────────────────────────────

class TestLowRankLinearAdvanced:
    def test_compression_ratio(self):
        """Compression ratio is correct."""
        from tensor_spline.low_rank import LowRankLinear
        layer = LowRankLinear(128, 64, rank=8, bias=True)
        expected_ratio = (128*64 + 64) / (128*8 + 8*64 + 64)
        assert abs(layer.compression_ratio() - expected_ratio) < 0.01

    def test_compression_ratio_no_bias(self):
        """Compression without bias."""
        from tensor_spline.low_rank import LowRankLinear
        layer = LowRankLinear(64, 32, rank=4, bias=False)
        assert layer.bias is None
        expected = (64*32) / (64*4 + 4*32)
        assert abs(layer.compression_ratio() - expected) < 0.01

    def test_param_count(self):
        """Parameter count matches formula."""
        from tensor_spline.low_rank import LowRankLinear
        layer = LowRankLinear(32, 16, rank=4)
        manual = 32*4 + 4*16 + 16  # U + V + bias
        assert layer.num_low_rank_params() == manual

    def test_dense_param_count(self):
        """Dense equivalent param count."""
        from tensor_spline.low_rank import LowRankLinear
        layer = LowRankLinear(32, 16, rank=4)
        assert layer.num_equivalent_dense_params() == 32*16 + 16

    def test_invalid_in_features(self):
        """Reject in_features < 1."""
        from tensor_spline.low_rank import LowRankLinear
        with pytest.raises(ValueError):
            LowRankLinear(0, 16)

    def test_invalid_out_features(self):
        """Reject out_features < 1."""
        from tensor_spline.low_rank import LowRankLinear
        with pytest.raises(ValueError):
            LowRankLinear(16, 0)

    def test_invalid_rank(self):
        """Reject rank < 1."""
        from tensor_spline.low_rank import LowRankLinear
        with pytest.raises(ValueError):
            LowRankLinear(16, 8, rank=0)

    def test_rank_one(self):
        """Rank 1 still works."""
        from tensor_spline.low_rank import LowRankLinear
        layer = LowRankLinear(8, 4, rank=1)
        x = torch.randn(2, 8)
        y = layer(x)
        assert y.shape == (2, 4)

    def test_high_rank(self):
        """High rank (> min(in, out)) still works, just not compressed."""
        from tensor_spline.low_rank import LowRankLinear
        layer = LowRankLinear(8, 4, rank=32)
        x = torch.randn(2, 8)
        y = layer(x)
        assert y.shape == (2, 4)

    def test_gradient_flows(self):
        """Gradients flow through the layer."""
        from tensor_spline.low_rank import LowRankLinear
        layer = LowRankLinear(8, 4, rank=2)
        x = torch.randn(2, 8, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert layer.U.grad is not None
        assert layer.V.grad is not None


# ─── Injection ─────────────────────────────────────────────────────────────────

class TestInjectLowRank:
    def test_inject_replaces_linear(self):
        """Inject replaces all nn.Linear layers."""
        from tensor_spline.low_rank import inject_low_rank, LowRankLinear
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
        result = inject_low_rank(model, rank=4)
        assert len(result) == 2
        assert isinstance(model[0], LowRankLinear)
        assert isinstance(model[2], LowRankLinear)

    def test_inject_target_modules(self):
        """Inject only targeted modules."""
        from tensor_spline.low_rank import inject_low_rank, LowRankLinear
        model = nn.Module()
        model.target = nn.Linear(8, 4)
        model.other = nn.Linear(8, 4)
        result = inject_low_rank(model, rank=2, target_modules=["target"])
        assert "target" in result
        assert "other" not in result
        assert isinstance(model.target, LowRankLinear)
        assert isinstance(model.other, nn.Linear)


# ─── Variant Recommendation ───────────────────────────────────────────────────

class TestRecommendVariant:
    def test_smooth_tasks(self):
        from tensor_spline.low_rank import recommend_variant
        assert recommend_variant("drift detection") == "spline"
        assert recommend_variant("sensor regression") == "spline"
        assert recommend_variant("continuous signal") == "spline"
        assert recommend_variant("smooth interpolation") == "spline"

    def test_sharp_tasks(self):
        from tensor_spline.low_rank import recommend_variant
        assert recommend_variant("classify images") == "lowrank"
        assert recommend_variant("intent detection") == "lowrank"
        assert recommend_variant("topic categorization") == "lowrank"

    def test_unknown_defaults_to_lowrank(self):
        from tensor_spline.low_rank import recommend_variant
        assert recommend_variant("random unknown task") == "lowrank"


# ─── HierarchicalSplineLinear ─────────────────────────────────────────────────

class TestHierarchicalSplineLinear:
    def test_forward_shape(self):
        from tensor_spline.hierarchical_spline import HierarchicalSplineLinear
        layer = HierarchicalSplineLinear(64, 32, coarse_pts=8, fine_pts=4, patch_size=16)
        x = torch.randn(2, 64)
        y = layer(x)
        assert y.shape == (2, 32)

    def test_compression_ratio(self):
        from tensor_spline.hierarchical_spline import HierarchicalSplineLinear
        layer = HierarchicalSplineLinear(64, 32, coarse_pts=8, fine_pts=4, patch_size=16)
        ratio = layer.compression_ratio()
        assert ratio > 1.0  # Must compress

    def test_control_params(self):
        from tensor_spline.hierarchical_spline import HierarchicalSplineLinear
        layer = HierarchicalSplineLinear(64, 32, coarse_pts=8, fine_pts=4, patch_size=16)
        assert layer.num_control_params() < layer.num_equivalent_dense_params()

    def test_no_bias(self):
        from tensor_spline.hierarchical_spline import HierarchicalSplineLinear
        layer = HierarchicalSplineLinear(64, 32, coarse_pts=8, fine_pts=4, bias=False)
        assert layer.bias is None

    def test_invalid_in_features(self):
        from tensor_spline.hierarchical_spline import HierarchicalSplineLinear
        with pytest.raises(ValueError):
            HierarchicalSplineLinear(0, 32)

    def test_invalid_out_features(self):
        from tensor_spline.hierarchical_spline import HierarchicalSplineLinear
        with pytest.raises(ValueError):
            HierarchicalSplineLinear(64, 0)

    def test_gradient_flows(self):
        from tensor_spline.hierarchical_spline import HierarchicalSplineLinear
        layer = HierarchicalSplineLinear(32, 16, coarse_pts=4, fine_pts=4, patch_size=8)
        x = torch.randn(2, 32, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert layer.coarse_values.grad is not None


class TestHierarchicalSplineClassifier:
    def test_forward_shape(self):
        from tensor_spline.hierarchical_spline import HierarchicalSplineClassifier
        model = HierarchicalSplineClassifier(input_dim=64, hidden=32, num_classes=4)
        x = torch.randn(4, 64)
        y = model(x)
        assert y.shape == (4, 4)


class TestInjectHierarchicalSpline:
    def test_inject_replaces_linear(self):
        from tensor_spline.hierarchical_spline import inject_hierarchical_spline, HierarchicalSplineLinear
        model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4))
        result = inject_hierarchical_spline(model, coarse_pts=4, fine_pts=4, patch_size=8)
        assert len(result) == 2
        assert isinstance(model[0], HierarchicalSplineLinear)

    def test_inject_with_targets(self):
        from tensor_spline.hierarchical_spline import inject_hierarchical_spline, HierarchicalSplineLinear
        model = nn.Module()
        model.target = nn.Linear(32, 16)
        model.skip = nn.Linear(32, 16)
        result = inject_hierarchical_spline(model, target_modules=["target"])
        assert "target" in result
        assert "skip" not in result


# ─── Mesh Registration ────────────────────────────────────────────────────────

class TestMeshRegistration:
    def test_register_tensor_spline(self):
        """Mesh registration works."""
        from tensor_spline.mesh import register_tensor_spline
        mock_registry = type("Registry", (), {
            "register": lambda self, cat, name, fn: None
        })()
        # Should not raise
        register_tensor_spline(mock_registry)
